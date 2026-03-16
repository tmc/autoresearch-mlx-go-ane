package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func benchmarkModelPath(b *testing.B) string {
	if p := os.Getenv("MLX_TEST_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		b.Skipf("MLX_TEST_MODEL not found at %s", p)
	}
	home, _ := os.UserHomeDir()
	fallback := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots")
	entries, err := os.ReadDir(fallback)
	if err != nil || len(entries) == 0 {
		b.Skip("no cached model found; set MLX_TEST_MODEL")
	}
	return filepath.Join(fallback, entries[0].Name())
}

func benchmarkDraftModelPath(b *testing.B) string {
	p := os.Getenv("MLX_TEST_DRAFT_MODEL")
	if p == "" {
		return ""
	}
	if _, err := os.Stat(p); err != nil {
		b.Skipf("MLX_TEST_DRAFT_MODEL not found at %s", p)
	}
	return p
}

func benchFreePort(b *testing.B) int {
	b.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		b.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func benchWaitForServer(b *testing.B, addr string, timeout time.Duration) {
	b.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := http.Get("http://" + addr + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(300 * time.Millisecond)
	}
	b.Fatalf("server did not become ready at %s", addr)
}

func BenchmarkServeCompletionsANE(b *testing.B) {
	modelPath := benchmarkModelPath(b)
	draftPath := benchmarkDraftModelPath(b)

	binPath := filepath.Join(b.TempDir(), "mlx-lm-serve")
	buildArgs := []string{"build"}
	if tags := strings.TrimSpace(os.Getenv("MLX_TEST_GO_TAGS")); tags != "" {
		buildArgs = append(buildArgs, "-tags", tags)
	}
	buildArgs = append(buildArgs, "-o", binPath, ".")
	build := exec.Command("go", buildArgs...)
	if out, err := build.CombinedOutput(); err != nil {
		b.Fatalf("build failed: %v\n%s", err, out)
	}

	tests := []struct {
		name      string
		args      []string
		needDraft bool
	}{
		{name: "baseline"},
		{
			name: "ane_forward_prefill",
			args: []string{
				"--ane-forward", "prefill",
				"--ane-forward-min-seq", "1",
			},
		},
		{
			name: "ane_forward_all",
			args: []string{
				"--ane-forward", "all",
				"--ane-forward-min-seq", "1",
			},
		},
		{
			name:      "ane_both_prefill",
			needDraft: true,
			args: []string{
				"--ane-speculative", "both-prefill",
				"--ane-speculative-min-seq", "1",
				"--num-draft-tokens", "4",
			},
		},
		{
			name:      "ane_both_prefill_seq16",
			needDraft: true,
			args: []string{
				"--ane-speculative", "both-prefill",
				"--ane-speculative-min-seq", "16",
				"--num-draft-tokens", "4",
			},
		},
		{
			name:      "ane_both_all",
			needDraft: true,
			args: []string{
				"--ane-speculative", "both-all",
				"--ane-speculative-min-seq", "1",
				"--num-draft-tokens", "4",
			},
		},
		{
			name:      "ssd_ane_backend_prefill",
			needDraft: true,
			args: []string{
				"--speculative-path", "ssd",
				"--ssd-draft-backend", "ane",
				"--ane-speculative", "both-prefill",
				"--ane-speculative-min-seq", "1",
				"--num-draft-tokens", "4",
			},
		},
	}

	for _, tc := range tests {
		if tc.needDraft && draftPath == "" {
			continue
		}

		b.Run(tc.name, func(b *testing.B) {
			port := benchFreePort(b)
			addr := fmt.Sprintf("127.0.0.1:%d", port)
			args := []string{
				"--model", modelPath,
				"--host", "127.0.0.1",
				"--port", strconv.Itoa(port),
				"--max-tokens", "32",
				"--log-level", "warn",
			}
			if tc.needDraft {
				args = append(args, "--draft-model", draftPath)
			}
			args = append(args, tc.args...)

			cmd := exec.Command(binPath, args...)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			if err := cmd.Start(); err != nil {
				b.Fatalf("start server: %v", err)
			}
			defer func() {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}()

			benchWaitForServer(b, addr, 120*time.Second)

			payload := map[string]any{
				"model":       "bench",
				"prompt":      "Write one short sentence about benchmarking.",
				"max_tokens":  32,
				"temperature": 0.0,
				"top_k":       0,
				"top_p":       1.0,
				"min_p":       0.0,
			}
			body, _ := json.Marshal(payload)
			warmupResp, err := http.Post("http://"+addr+"/v1/completions", "application/json", bytes.NewReader(body))
			if err != nil {
				b.Fatalf("warmup request failed: %v", err)
			}
			warmupResp.Body.Close()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				start := time.Now()
				resp, err := http.Post("http://"+addr+"/v1/completions", "application/json", bytes.NewReader(body))
				if err != nil {
					b.Fatalf("request failed: %v", err)
				}
				var out struct {
					Usage struct {
						CompletionTokens int `json:"completion_tokens"`
					} `json:"usage"`
				}
				if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
					resp.Body.Close()
					b.Fatalf("decode response: %v", err)
				}
				resp.Body.Close()
				if out.Usage.CompletionTokens > 0 {
					secs := time.Since(start).Seconds()
					b.ReportMetric(float64(out.Usage.CompletionTokens)/secs, "tok/s")
				}
			}
		})
	}
}
