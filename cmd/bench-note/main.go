// Command bench-note attaches benchmark results to git commits as notes.
//
// The note content is raw go test output, directly usable as benchstat input.
// By default, a new run overwrites the previous note. Use --append to
// concatenate multiple runs (e.g., different configs on the same commit).
//
// Usage:
//
//	bench-note run [flags]       Run benchmarks and attach as git note to HEAD (or --commit)
//	bench-note show [commit]     Display bench note (default: HEAD)
//	bench-note compare [commit...] Run benchstat across commits (default: HEAD~1 HEAD)
//	bench-note history           List commits that have bench notes
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"
)

const notesRef = "refs/notes/benchmarks"

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd := os.Args[1]
	var err error
	switch cmd {
	case "run":
		err = cmdRun(os.Args[2:])
	case "show":
		err = cmdShow(os.Args[2:])
	case "compare":
		err = cmdCompare(os.Args[2:])
	case "history":
		err = cmdHistory(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench-note %s: %v\n", cmd, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: bench-note <command> [flags]

Commands:
  run [--benchtime=T] [--count=6] [--from-file=FILE] [--append] [--commit=SHA]
      Run benchmarks and attach as git note to HEAD (or --commit).
      --append concatenates to existing note instead of overwriting.
      --commit attaches to a specific commit instead of HEAD.
  show [commit]
      Display bench note (default: HEAD).
  compare [commit...] [-- benchstat-flags...]
      Run benchstat across commits (default: HEAD~1 vs HEAD).
      0 args: HEAD~1 vs HEAD. 1 arg: <commit> vs HEAD. N args: all compared.
      Flags after -- are passed to benchstat (e.g. -- -filter "/mode:GPU").
  history
      List commits with bench notes.
`)
	os.Exit(2)
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	benchtime := fs.String("benchtime", "", "benchtime flag for go test (omit for go test default)")
	count := fs.Int("count", 6, "number of benchmark runs")
	fromFile := fs.String("from-file", "", "use existing benchmark output file instead of running")
	appendMode := fs.Bool("append", false, "append to existing note instead of overwriting")
	commitFlag := fs.String("commit", "", "attach note to this commit instead of HEAD")
	fs.Parse(args)

	ref := "HEAD"
	if *commitFlag != "" {
		ref = *commitFlag
	}
	commit, err := gitRevParse(ref)
	if err != nil {
		return err
	}

	var raw []byte
	if *fromFile != "" {
		raw, err = os.ReadFile(*fromFile)
		if err != nil {
			return fmt.Errorf("reading file: %w", err)
		}
	} else {
		raw, err = runBenchmarks(*benchtime, *count)
		if err != nil {
			return fmt.Errorf("running benchmarks: %w", err)
		}
	}

	header := fmt.Sprintf("# bench-note v1 %s\n", time.Now().UTC().Format(time.RFC3339))
	raw = append([]byte(header), raw...)

	if *appendMode {
		err = gitNoteAppend(commit, raw)
	} else {
		err = gitNoteAdd(commit, raw)
	}
	if err != nil {
		return fmt.Errorf("adding note: %w", err)
	}
	mode := "attached"
	if *appendMode {
		mode = "appended"
	}
	fmt.Printf("bench note %s to %s\n", mode, commit[:min(len(commit), 8)])
	return nil
}

func cmdShow(args []string) error {
	commit := "HEAD"
	if len(args) > 0 {
		commit = args[0]
	}
	note, err := readNote(commit)
	if err != nil {
		return err
	}
	os.Stdout.Write(note)
	return nil
}

func cmdCompare(args []string) error {
	// Split on "--": commits before, benchstat flags after.
	var commits, benchstatFlags []string
	for i, a := range args {
		if a == "--" {
			benchstatFlags = args[i+1:]
			break
		}
		commits = append(commits, a)
	}

	if len(commits) == 0 {
		commits = []string{"HEAD~1", "HEAD"}
	} else if len(commits) == 1 {
		commits = []string{commits[0], "HEAD"}
	}

	dir, err := os.MkdirTemp("", "bench-note-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	var filenames []string
	for _, c := range commits {
		raw, err := readNote(c)
		if err != nil {
			return fmt.Errorf("commit %s: %w", c, err)
		}
		label := shortRef(c)
		path := dir + "/" + label + ".txt"
		if err := os.WriteFile(path, raw, 0644); err != nil {
			return err
		}
		filenames = append(filenames, label+".txt")
	}

	cmdArgs := append(filenames, benchstatFlags...)
	cmd := exec.Command("benchstat", cmdArgs...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("benchstat: %w\n%s", err, out)
	}
	os.Stdout.Write(out)
	return nil
}

func cmdHistory(_ []string) error {
	out, err := cmdOutput("git", "notes", "--ref="+notesRef, "list")
	if err != nil {
		return fmt.Errorf("no bench notes found")
	}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		subject, _ := cmdOutput("git", "log", "-1", "--format=%h %s (%ar)", parts[1])
		fmt.Println(strings.TrimSpace(subject))
	}
	return nil
}

// runBenchmarks runs go test -bench and returns the output.
func runBenchmarks(benchtime string, count int) ([]byte, error) {
	args := []string{"test", "-bench=.", "-benchmem",
		fmt.Sprintf("-count=%d", count),
		"-run=^$", "-timeout=30m",
	}
	if benchtime != "" {
		args = append(args, fmt.Sprintf("-benchtime=%s", benchtime))
	}
	fmt.Fprintf(os.Stderr, "running: go %s\n", strings.Join(args, " "))
	cmd := exec.Command("go", args...)
	var buf bytes.Buffer
	cmd.Stdout = io.MultiWriter(&buf, os.Stdout)
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func shortRef(ref string) string {
	out, err := cmdOutput("git", "rev-parse", "--short", ref)
	if err != nil {
		return ref
	}
	return strings.TrimSpace(out)
}

func readNote(ref string) ([]byte, error) {
	hash, err := gitRevParse(ref)
	if err != nil {
		return nil, err
	}
	out, err := cmdOutput("git", "notes", "--ref="+notesRef, "show", hash)
	if err != nil {
		return nil, fmt.Errorf("no bench note for %s", hash[:min(len(hash), 8)])
	}
	return []byte(out), nil
}

func gitNoteAdd(commit string, content []byte) error {
	cmd := exec.Command("git", "notes", "--ref="+notesRef, "add", "-f", "-F", "-", commit)
	cmd.Stdin = bytes.NewReader(content)
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func gitNoteAppend(commit string, content []byte) error {
	cmd := exec.Command("git", "notes", "--ref="+notesRef, "append", "-F", "-", commit)
	cmd.Stdin = bytes.NewReader(content)
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func gitRevParse(ref string) (string, error) {
	out, err := cmdOutput("git", "rev-parse", ref)
	if err != nil {
		return "", fmt.Errorf("rev-parse %s: %w", ref, err)
	}
	return strings.TrimSpace(out), nil
}

func cmdOutput(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.Output()
	return string(out), err
}
