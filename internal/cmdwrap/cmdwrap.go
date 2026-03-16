package cmdwrap

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type FlagKind uint8

const (
	StringFlag FlagKind = iota
	IntFlag
	BoolFlag
)

type FlagSpec struct {
	Name  string
	Env   string
	Usage string
	Kind  FlagKind
}

type Prepared struct {
	Args []string
	Env  []string
	Help bool
}

func Run(target string, args []string) int {
	return RunWithFlags(target, args, nil)
}

func RunWithFlags(target string, args []string, specs []FlagSpec) int {
	root, err := findRepoRoot(target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}
	prepared, err := Prepare(args, specs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 2
	}
	if prepared.Help && len(specs) > 0 {
		PrintHelp(specs)
	}
	goArgs := []string{"run", "-tags", "ane_appleneuralengine", "./" + target}
	goArgs = append(goArgs, prepared.Args...)
	cmd := exec.Command("go", goArgs...)
	cmd.Dir = root
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(append(os.Environ(), prepared.Env...), "GOWORK="+filepath.Join(root, "go.work"))
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "Error: run %s: %v\n", target, err)
		return 1
	}
	return 0
}

func Prepare(args []string, specs []FlagSpec) (Prepared, error) {
	byName := make(map[string]FlagSpec, len(specs))
	for _, spec := range specs {
		byName[spec.Name] = spec
	}
	var passArgs []string
	var envArgs []string
	help := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "-h" || arg == "--help" {
			help = true
			passArgs = append(passArgs, arg)
			continue
		}
		if arg == "--" {
			passArgs = append(passArgs, args[i:]...)
			break
		}
		name, value, hasValue, ok := splitLongFlag(arg)
		if !ok {
			passArgs = append(passArgs, arg)
			continue
		}
		spec, ok := byName[name]
		if !ok {
			passArgs = append(passArgs, arg)
			continue
		}
		switch spec.Kind {
		case BoolFlag:
			if !hasValue {
				value = "true"
			}
		case StringFlag, IntFlag:
			if !hasValue {
				if i+1 >= len(args) {
					return Prepared{}, fmt.Errorf("missing value for --%s", name)
				}
				i++
				value = args[i]
			}
		}
		envArgs = append(envArgs, spec.Env+"="+value)
	}
	return Prepared{Args: passArgs, Env: envArgs, Help: help}, nil
}

func splitLongFlag(arg string) (name, value string, hasValue, ok bool) {
	if !strings.HasPrefix(arg, "--") || len(arg) == 2 {
		return "", "", false, false
	}
	raw := strings.TrimPrefix(arg, "--")
	if raw == "" {
		return "", "", false, false
	}
	name = raw
	if idx := strings.IndexByte(raw, '='); idx >= 0 {
		name = raw[:idx]
		value = raw[idx+1:]
		hasValue = true
	}
	return name, value, hasValue, true
}

func PrintHelp(specs []FlagSpec) {
	fmt.Fprintln(os.Stdout, "ANE wrapper flags:")
	for _, spec := range specs {
		fmt.Fprintf(os.Stdout, "  --%s\n    \t%s\n", spec.Name, spec.Usage)
	}
	fmt.Fprintln(os.Stdout)
}

func ApplyEnv(env []string) error {
	for _, kv := range env {
		key, val, ok := strings.Cut(kv, "=")
		if !ok || key == "" {
			return fmt.Errorf("invalid env assignment %q", kv)
		}
		if err := os.Setenv(key, val); err != nil {
			return fmt.Errorf("set env %s: %w", key, err)
		}
	}
	return nil
}

func findRepoRoot(target string) (string, error) {
	for _, start := range []string{cwd(), sourceRoot()} {
		if start == "" {
			continue
		}
		root, ok := walkUp(start, target)
		if ok {
			return root, nil
		}
	}
	return "", fmt.Errorf("find repo root for %s", target)
}

func cwd() string {
	dir, _ := os.Getwd()
	return dir
}

func sourceRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
}

func walkUp(start, target string) (string, bool) {
	dir := start
	for {
		if looksLikeRoot(dir, target) {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

func looksLikeRoot(dir, target string) bool {
	if _, err := os.Stat(filepath.Join(dir, "go.work")); err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(target))); err != nil {
		return false
	}
	work, err := os.ReadFile(filepath.Join(dir, "go.work"))
	if err != nil {
		return false
	}
	return strings.Contains(string(work), "./examples/mlx-go-lm")
}
