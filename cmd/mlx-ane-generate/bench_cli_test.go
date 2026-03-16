package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// BenchmarkCLI_ANELeverage compares generate CLI throughput with and without
// speculative ANE routing enabled.
func BenchmarkCLI_ANELeverage(b *testing.B) {
	modelPath := os.Getenv("MLX_TEST_MODEL")
	if modelPath == "" {
		home, _ := os.UserHomeDir()
		modelPath = filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/main")
	}
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model not found at %s", modelPath)
	}

	draftModelPath := os.Getenv("MLX_TEST_DRAFT_MODEL")
	if draftModelPath != "" {
		if _, err := os.Stat(draftModelPath); err != nil {
			b.Skipf("draft model not found at %s", draftModelPath)
		}
	}

	prompt := os.Getenv("MLX_TEST_PROMPT")
	if prompt == "" {
		prompt = "Summarize why deterministic benchmarks need warmup."
	}
	maxTokens := 100
	if v := os.Getenv("MLX_TEST_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	}

	binPath := filepath.Join(b.TempDir(), "mlx-lm-generate")
	buildArgs := []string{"build"}
	if tags := strings.TrimSpace(os.Getenv("MLX_TEST_GO_TAGS")); tags != "" {
		buildArgs = append(buildArgs, "-tags", tags)
	}
	buildArgs = append(buildArgs, "-o", binPath, ".")
	cmd := exec.Command("go", buildArgs...)
	if out, err := cmd.CombinedOutput(); err != nil {
		b.Fatalf("build failed: %v\n%s", err, out)
	}

	tests := []struct {
		name      string
		args      []string
		needDraft bool
	}{
		{
			name: "baseline",
			args: []string{},
		},
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
			name:      "ane_draft_prefill",
			needDraft: true,
			args: []string{
				"--ane-speculative", "draft-prefill",
				"--ane-speculative-min-seq", "1",
				"--num-draft-tokens", "4",
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
		if tc.needDraft && draftModelPath == "" {
			continue
		}
		b.Run(tc.name, func(b *testing.B) {
			warmupArgs := []string{
				"--model", modelPath,
				"--prompt", prompt,
				"--max-tokens", strconv.Itoa(maxTokens),
				"--temperature", "0.0",
				"--top-k", "0",
				"--top-p", "1.0",
				"--min-p", "0.0",
			}
			if tc.needDraft {
				warmupArgs = append(warmupArgs, "--draft-model", draftModelPath)
			}
			warmupArgs = append(warmupArgs, tc.args...)
			warmup := exec.Command(binPath, warmupArgs...)
			if out, err := warmup.CombinedOutput(); err != nil {
				b.Logf("warmup output:\n%s", out)
				b.Fatalf("warmup failed: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				args := []string{
					"--model", modelPath,
					"--prompt", prompt,
					"--max-tokens", strconv.Itoa(maxTokens),
					"--temperature", "0.0",
					"--top-k", "0",
					"--top-p", "1.0",
					"--min-p", "0.0",
				}
				if tc.needDraft {
					args = append(args, "--draft-model", draftModelPath)
				}
				args = append(args, tc.args...)
				cmd := exec.Command(binPath, args...)
				out, err := cmd.CombinedOutput()
				if err != nil {
					b.Logf("output:\n%s", out)
					b.Fatalf("run failed: %v", err)
				}
				tps := parseGoTPS(string(out))
				b.ReportMetric(tps, "tps")
			}
		})
	}
}

func BenchmarkCLI_ANEAcceptanceMatrix(b *testing.B) {
	modelPath := os.Getenv("MLX_TEST_MODEL")
	if modelPath == "" {
		home, _ := os.UserHomeDir()
		modelPath = filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/main")
	}
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model not found at %s", modelPath)
	}
	draftModelPath := os.Getenv("MLX_TEST_DRAFT_MODEL")
	if draftModelPath == "" {
		b.Skip("draft model not configured in MLX_TEST_DRAFT_MODEL")
	}
	if _, err := os.Stat(draftModelPath); err != nil {
		b.Skipf("draft model not found at %s", draftModelPath)
	}

	prompt := os.Getenv("MLX_TEST_PROMPT")
	if prompt == "" {
		prompt = "Explain why accepted tokens per second is the right speculative decoding metric."
	}
	maxTokens := 48
	if v := os.Getenv("MLX_TEST_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	}

	binPath := filepath.Join(b.TempDir(), "mlx-lm-generate")
	buildArgs := []string{"build"}
	if tags := strings.TrimSpace(os.Getenv("MLX_TEST_GO_TAGS")); tags != "" {
		buildArgs = append(buildArgs, "-tags", tags)
	}
	buildArgs = append(buildArgs, "-o", binPath, ".")
	cmd := exec.Command("go", buildArgs...)
	if out, err := cmd.CombinedOutput(); err != nil {
		b.Fatalf("build failed: %v\n%s", err, out)
	}

	type benchCase struct {
		name string
		env  map[string]string
	}
	cases := []benchCase{
		{name: "baseline_spec"},
		{name: "ane_auto", env: map[string]string{"MLXGO_ANE_DRAFT_STRATEGY": "auto"}},
		{name: "ane_partial_4", env: map[string]string{"MLXGO_ANE_DRAFT_STRATEGY": "partial4"}},
		{name: "ane_partial_2", env: map[string]string{"MLXGO_ANE_DRAFT_STRATEGY": "partial2"}},
		{name: "ane_partial_1", env: map[string]string{"MLXGO_ANE_DRAFT_STRATEGY": "partial1"}},
		{name: "ane_reference_only", env: map[string]string{"MLXGO_ANE_DRAFT_STRATEGY": "reference-only"}},
		{name: "ane_auto_guard_off", env: map[string]string{
			"MLXGO_ANE_DRAFT_STRATEGY":               "auto",
			"MLXGO_ANE_DRAFT_DISABLE_AUTO_REF_GUARD": "1",
		}},
	}

	baseArgs := []string{
		"--model", modelPath,
		"--draft-model", draftModelPath,
		"--prompt", prompt,
		"--max-tokens", strconv.Itoa(maxTokens),
		"--num-draft-tokens", "4",
		"--temperature", "0.0",
		"--top-k", "0",
		"--top-p", "1.0",
		"--min-p", "0.0",
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			warmStats, err := runCLIWithSpecStats(binPath, baseArgs, tc.env)
			if err != nil {
				if cliStatsRunTimedOut(err) {
					b.Skipf("warmup timed out: %v", err)
				}
				b.Fatalf("warmup failed: %v", err)
			}
			b.Logf("warmup mode=%s backend=%s kind=%s accept=%.3f accepted_tps=%.3f", warmStats.Mode, warmStats.DraftBackend, warmStats.DraftModelKind, warmStats.AcceptRate, warmStats.AcceptedTokensPerSec)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				stats, err := runCLIWithSpecStats(binPath, baseArgs, tc.env)
				if err != nil {
					if cliStatsRunTimedOut(err) {
						b.Skipf("run timed out: %v", err)
					}
					b.Fatalf("run failed: %v", err)
				}
				b.ReportMetric(stats.AcceptRate, "accept_rate")
				b.ReportMetric(stats.AcceptedTokensPerSec, "accepted_tps")
				b.ReportMetric(stats.GenerationTokensPerSec, "gen_tps")
				b.ReportMetric(stats.PrefillTokensPerSec, "prefill_tps")
				b.ReportMetric(stats.InitMS, "init_ms")
			}
		})
	}
}

func BenchmarkCLI_QwenFamilySpeculativeMatrix(b *testing.B) {
	targets := qwenFamilyDenseTargets()
	if len(targets) == 0 {
		b.Skip("no local Qwen family checkpoints found")
	}
	prompt := os.Getenv("MLX_TEST_PROMPT")
	if prompt == "" {
		prompt = "Explain why accepted tokens per second and TTFT need to be reported together."
	}
	maxTokens := 48
	if v := os.Getenv("MLX_TEST_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	}
	binPath := buildGenerateBenchmarkBinary(b)
	type benchCase struct {
		name string
		mode string
		args []string
		env  map[string]string
	}
	cases := []benchCase{
		{
			name: "reference_only_draft_prefill",
			mode: "draft-prefill",
			env:  map[string]string{"MLXGO_ANE_DRAFT_STRATEGY": "reference-only"},
		},
		{
			name: "direct_modelir_draft_prefill",
			mode: "draft-prefill",
			env: map[string]string{
				"MLXGO_ANE_DRAFT_STRATEGY":               "full",
				"MLXGO_ANE_DRAFT_OUTPUT_MODE":            "final_hidden",
				"MLXGO_ANE_DRAFT_DISABLE_AUTO_REF_GUARD": "1",
				"MLXGO_ANE_DRAFT_MIL_MAX_LAYERS":         "1",
			},
		},
		{
			name: "mil_transformer_both_prefill",
			mode: "both-prefill",
			env: map[string]string{
				"MLXGO_ANE_DRAFT_STRATEGY":    "full",
				"MLXGO_ANE_DRAFT_OUTPUT_MODE": "final_hidden",
			},
		},
		{
			name: "mil_attention_decode_both_prefill",
			mode: "both-prefill",
			env: map[string]string{
				"MLXGO_ANE_DRAFT_STRATEGY":    "full",
				"MLXGO_ANE_DRAFT_OUTPUT_MODE": "last_attn_residual",
				"MLXGO_ANE_DRAFT_MIL_MAX_LAYERS": "1",
			},
		},
		{
			name: "layer_stack_both_all",
			mode: "both-all",
			env: map[string]string{
				"MLXGO_ANE_DRAFT_STRATEGY":    "full",
				"MLXGO_ANE_DRAFT_OUTPUT_MODE": "lm_head",
			},
		},
		{
			name: "ffn_fallback_draft_all",
			mode: "draft-all",
			env: map[string]string{
				"MLXGO_ANE_DRAFT_STRATEGY":    "ffn",
				"MLXGO_ANE_DRAFT_OUTPUT_MODE": "last_ffn_residual",
			},
		},
		{
			name: "ssd_ane_backend_prefill",
			mode: "both-prefill",
			args: []string{"--speculative-path", "ssd", "--ssd-draft-backend", "ane"},
			env: map[string]string{
				"MLXGO_ANE_DRAFT_STRATEGY":    "full",
				"MLXGO_ANE_DRAFT_OUTPUT_MODE": "final_hidden",
			},
		},
	}
	for _, target := range targets {
		target := target
		b.Run(target.Name, func(b *testing.B) {
			draftModelPath := qwenFamilyDraftModelPath(target)
			if draftModelPath == "" {
				b.Skip("no tokenizer-compatible draft model found")
			}
			baseArgs := []string{
				"--model", target.ModelPath,
				"--draft-model", draftModelPath,
				"--prompt", prompt,
				"--max-tokens", strconv.Itoa(maxTokens),
				"--num-draft-tokens", "4",
				"--temperature", "0.0",
				"--top-k", "0",
				"--top-p", "1.0",
				"--min-p", "0.0",
			}
			baselineStats, err := runCLIWithSpecStats(binPath, baseArgs, nil)
			if err != nil {
				if cliStatsRunTimedOut(err) {
					b.Skipf("baseline timed out: %v", err)
				}
				b.Skipf("baseline failed: %v", err)
			}
			for _, tc := range cases {
				tc := tc
				b.Run(tc.name, func(b *testing.B) {
					args := append([]string{}, baseArgs...)
					args = append(args, "--ane-speculative", tc.mode, "--ane-speculative-min-seq", "1")
					args = append(args, tc.args...)
					warmStats, err := runCLIWithSpecStats(binPath, args, tc.env)
					if err != nil {
						if cliStatsRunTimedOut(err) {
							b.Skipf("warmup timed out: %v", err)
						}
						b.Skipf("warmup failed: %v", err)
					}
					b.Logf("warmup backend=%s kind=%s direct=%t accept=%.3f accepted_tps=%.3f", warmStats.DraftBackend, warmStats.DraftModelKind, warmStats.DirectModelIR, warmStats.AcceptRate, warmStats.AcceptedTokensPerSec)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						stats, err := runCLIWithSpecStats(binPath, args, tc.env)
						if err != nil {
							if cliStatsRunTimedOut(err) {
								b.Skipf("run timed out: %v", err)
							}
							b.Fatalf("run failed: %v", err)
						}
						b.ReportMetric(stats.AcceptRate, "accept_rate")
						b.ReportMetric(stats.AcceptedTokensPerSec, "accepted_tps")
						b.ReportMetric(stats.GenerationTokensPerSec, "gen_tps")
						b.ReportMetric(stats.PrefillTokensPerSec, "prefill_tps")
						b.ReportMetric(stats.InitMS, "init_ms")
						b.ReportMetric(baselineStats.AcceptedTokensPerSec, "baseline_accepted_tps")
						b.ReportMetric(baselineStats.GenerationTokensPerSec, "baseline_gen_tps")
						b.ReportMetric(baselineStats.PrefillTokensPerSec, "baseline_prefill_tps")
						b.ReportMetric(float64(stats.RebuildCount), "rebuilds")
						if baselineStats.AcceptedTokensPerSec > 0 {
							b.ReportMetric(stats.AcceptedTokensPerSec/baselineStats.AcceptedTokensPerSec, "vs_baseline_accepted_tps")
						}
						if baselineStats.GenerationTokensPerSec > 0 {
							b.ReportMetric(stats.GenerationTokensPerSec/baselineStats.GenerationTokensPerSec, "vs_baseline_gen_tps")
						}
						if baselineStats.PrefillTokensPerSec > 0 {
							b.ReportMetric(stats.PrefillTokensPerSec/baselineStats.PrefillTokensPerSec, "vs_baseline_prefill_tps")
						}
						if stats.DirectModelIR {
							b.ReportMetric(1, "direct_modelir")
						} else {
							b.ReportMetric(0, "direct_modelir")
						}
						if stats.FFNFallbackUsed {
							b.ReportMetric(1, "ffn_fallback")
						} else {
							b.ReportMetric(0, "ffn_fallback")
						}
						if stats.DraftBackend == "ane" {
							b.ReportMetric(1, "ane_backend")
						} else {
							b.ReportMetric(0, "ane_backend")
						}
					}
				})
			}
		})
	}
}

func BenchmarkCLI_QwenFamilyPrefillMatrix(b *testing.B) {
	targets := qwenFamilyAllTargets()
	if len(targets) == 0 {
		b.Skip("no local Qwen family checkpoints found")
	}
	maxTokens := 32
	if v := os.Getenv("MLX_TEST_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	}
	promptLens := []int{8, 64, 256, 1024}
	type benchCase struct {
		name      string
		args      []string
		needDraft bool
		specStats bool
	}
	cases := []benchCase{
		{name: "baseline"},
		{name: "ane_forward_prefill", args: []string{"--ane-forward", "prefill", "--ane-forward-min-seq", "1"}},
		{name: "target_prefill", args: []string{"--ane-speculative", "target-prefill", "--ane-speculative-min-seq", "1", "--num-draft-tokens", "4"}, needDraft: true, specStats: true},
		{name: "both_prefill", args: []string{"--ane-speculative", "both-prefill", "--ane-speculative-min-seq", "1", "--num-draft-tokens", "4"}, needDraft: true, specStats: true},
		{name: "both_all", args: []string{"--ane-speculative", "both-all", "--ane-speculative-min-seq", "1", "--num-draft-tokens", "4"}, needDraft: true, specStats: true},
	}
	binPath := buildGenerateBenchmarkBinary(b)
	for _, target := range targets {
		target := target
		b.Run(target.Name, func(b *testing.B) {
			draftModelPath := qwenFamilyDraftModelPath(target)
			for _, promptLen := range promptLens {
				promptLen := promptLen
				prompt := benchmarkPrompt(promptLen)
				b.Run(fmt.Sprintf("prompt_%d", promptLen), func(b *testing.B) {
					baselineArgs := []string{
						"--model", target.ModelPath,
						"--prompt", prompt,
						"--max-tokens", strconv.Itoa(maxTokens),
						"--temperature", "0.0",
						"--top-k", "0",
						"--top-p", "1.0",
						"--min-p", "0.0",
					}
					baselineOut, baselineErr := runCLIOutput(binPath, baselineArgs, nil)
					if baselineErr != nil {
						if cliStatsRunTimedOut(baselineErr) {
							b.Skipf("baseline timed out: %v", baselineErr)
						}
						b.Skipf("baseline failed: %v", baselineErr)
					}
					baselinePrefillTPS := parseGoPrefillTPS(baselineOut)
					baselineGenTPS := parseGoTPS(baselineOut)
					for _, tc := range cases {
						tc := tc
						if tc.needDraft && draftModelPath == "" {
							continue
						}
						b.Run(tc.name, func(b *testing.B) {
							args := []string{
								"--model", target.ModelPath,
								"--prompt", prompt,
								"--max-tokens", strconv.Itoa(maxTokens),
								"--temperature", "0.0",
								"--top-k", "0",
								"--top-p", "1.0",
								"--min-p", "0.0",
							}
							if tc.needDraft {
								if draftModelPath == "" {
									b.Skip("no tokenizer-compatible draft model found")
								}
								args = append(args, "--draft-model", draftModelPath)
							}
							args = append(args, tc.args...)
							if tc.specStats {
								warmStats, err := runCLIWithSpecStats(binPath, args, nil)
								if err != nil {
									if cliStatsRunTimedOut(err) {
										b.Skipf("warmup timed out: %v", err)
									}
									b.Skipf("warmup failed: %v", err)
								}
								b.Logf("warmup prefill=%.3f gen=%.3f accepted=%.3f", warmStats.PrefillTokensPerSec, warmStats.GenerationTokensPerSec, warmStats.AcceptedTokensPerSec)
								b.ResetTimer()
								for i := 0; i < b.N; i++ {
									stats, err := runCLIWithSpecStats(binPath, args, nil)
									if err != nil {
										if cliStatsRunTimedOut(err) {
											b.Skipf("run timed out: %v", err)
										}
										b.Fatalf("run failed: %v", err)
									}
									b.ReportMetric(stats.PrefillTokensPerSec, "prefill_tps")
									b.ReportMetric(stats.GenerationTokensPerSec, "gen_tps")
									b.ReportMetric(stats.AcceptedTokensPerSec, "accepted_tps")
									b.ReportMetric(stats.InitMS, "init_ms")
									b.ReportMetric(baselinePrefillTPS, "baseline_prefill_tps")
									b.ReportMetric(baselineGenTPS, "baseline_gen_tps")
									if baselinePrefillTPS > 0 {
										b.ReportMetric(stats.PrefillTokensPerSec/baselinePrefillTPS, "vs_baseline_prefill_tps")
									}
									if baselineGenTPS > 0 {
										b.ReportMetric(stats.GenerationTokensPerSec/baselineGenTPS, "vs_baseline_gen_tps")
									}
								}
								return
							}
							if _, err := runCLIOutput(binPath, args, nil); err != nil {
								if cliStatsRunTimedOut(err) {
									b.Skipf("warmup timed out: %v", err)
								}
								b.Skipf("warmup failed: %v", err)
							}
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								out, err := runCLIOutput(binPath, args, nil)
								if err != nil {
									b.Fatalf("run failed: %v", err)
								}
								prefillTPS := parseGoPrefillTPS(out)
								genTPS := parseGoTPS(out)
								b.ReportMetric(prefillTPS, "prefill_tps")
								b.ReportMetric(genTPS, "gen_tps")
								b.ReportMetric(baselinePrefillTPS, "baseline_prefill_tps")
								b.ReportMetric(baselineGenTPS, "baseline_gen_tps")
								if baselinePrefillTPS > 0 {
									b.ReportMetric(prefillTPS/baselinePrefillTPS, "vs_baseline_prefill_tps")
								}
								if baselineGenTPS > 0 {
									b.ReportMetric(genTPS/baselineGenTPS, "vs_baseline_gen_tps")
								}
							}
						})
					}
				})
			}
		})
	}
}

func benchmarkPrompt(tokens int) string {
	if tokens <= 0 {
		return "benchmark"
	}
	var b strings.Builder
	for i := 0; i < tokens; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "token%d", i%97)
	}
	return b.String()
}

func buildGenerateBenchmarkBinary(b *testing.B) string {
	b.Helper()
	binPath := filepath.Join(b.TempDir(), "mlx-lm-generate")
	buildArgs := []string{"build"}
	if tags := strings.TrimSpace(os.Getenv("MLX_TEST_GO_TAGS")); tags != "" {
		buildArgs = append(buildArgs, "-tags", tags)
	}
	buildArgs = append(buildArgs, "-o", binPath, ".")
	cmd := exec.Command("go", buildArgs...)
	if out, err := cmd.CombinedOutput(); err != nil {
		b.Fatalf("build failed: %v\n%s", err, out)
	}
	return binPath
}

func runCLIOutput(binPath string, args []string, extraEnv map[string]string) (string, error) {
	timeout := benchmarkCLITimeout()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, binPath, args...)
	env := benchmarkCLIEnv()
	for k, v := range extraEnv {
		env = append(env, k+"="+v)
	}
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return "", &cliStatsRunError{
			Timeout: timeout,
			Output:  string(out),
			Err:     ctx.Err(),
		}
	}
	if err != nil {
		return "", &cliStatsRunError{
			Output: string(out),
			Err:    err,
		}
	}
	return string(out), nil
}

func runCLIWithSpecStats(binPath string, args []string, extraEnv map[string]string) (speculativeStatsJSON, error) {
	timeout := benchmarkCLITimeout()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, binPath, args...)
	env := benchmarkCLIEnv()
	env = append(env, "MLXGO_ANE_SPEC_STATS_JSON=1")
	for k, v := range extraEnv {
		env = append(env, k+"="+v)
	}
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return speculativeStatsJSON{}, &cliStatsRunError{
			Timeout: timeout,
			Output:  string(out),
			Err:     ctx.Err(),
		}
	}
	if err != nil {
		return speculativeStatsJSON{}, &cliStatsRunError{
			Output: string(out),
			Err:    err,
		}
	}
	stats, parseErr := parseSpeculativeStatsLine(string(out))
	if parseErr != nil {
		return speculativeStatsJSON{}, fmt.Errorf("parse stats json: %w\n%s", parseErr, out)
	}
	return stats, nil
}

func benchmarkCLIEnv() []string {
	env := append([]string{}, os.Environ()...)
	for _, key := range []string{
		"MLXGO_ANE_SPEC_STATS_JSON",
		"MLXGO_ANE_FORWARD",
		"MLXGO_ANE_FORWARD_MIN_SEQ",
		"MLXGO_ANE_SPECULATIVE",
		"MLXGO_ANE_SPECULATIVE_MIN_SEQ",
		"MLXGO_ANE_DRAFT_STRATEGY",
		"MLXGO_ANE_DRAFT_OUTPUT_MODE",
		"MLXGO_ANE_DRAFT_DISABLE_AUTO_REF_GUARD",
		"MLXGO_ANE_DRAFT_MIL_MAX_LAYERS",
		"MLXGO_ANE_DRAFT_MODELC",
		"MLXGO_ANE_DECODE_PLANE",
		"MLXGO_ANE_QWEN35_OUTPUT_MODE",
		"MLXGO_ANE_QWEN35_CONSUMER_MODE",
		"MLXGO_ANE_QWEN35_OUTPUT_POOL_DEPTH",
		"MLXGO_ANE_QWEN35_WAIT_MODE",
		"MLXGO_ANE_QWEN35_COMPILED_PREPARE",
		"MLXGO_ANE_QWEN35_DIRECT_BLOCK",
		"MLXGO_ANE_QWEN35_DIRECT_BLOCK_OFFSETS",
	} {
		env = append(env, key+"=")
	}
	return env
}

type cliStatsRunError struct {
	Timeout time.Duration
	Output  string
	Err     error
}

func (e *cliStatsRunError) Error() string {
	if e == nil {
		return ""
	}
	if e.Timeout > 0 {
		return fmt.Sprintf("cli run timed out after %s\n%s", e.Timeout, e.Output)
	}
	return fmt.Sprintf("cli run: %v\n%s", e.Err, e.Output)
}

func (e *cliStatsRunError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func cliStatsRunTimedOut(err error) bool {
	var runErr *cliStatsRunError
	return errors.As(err, &runErr) && runErr.Timeout > 0
}

func benchmarkCLITimeout() time.Duration {
	if v := strings.TrimSpace(os.Getenv("MLX_TEST_CLI_TIMEOUT_SEC")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 30 * time.Second
}

func parseSpeculativeStatsLine(out string) (speculativeStatsJSON, error) {
	lines := strings.Split(out, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" || !strings.HasPrefix(line, "{") {
			continue
		}
		var stats speculativeStatsJSON
		if err := json.Unmarshal([]byte(line), &stats); err == nil && stats.Mode != "" {
			return stats, nil
		}
	}
	return speculativeStatsJSON{}, fmt.Errorf("speculative stats json not found")
}
