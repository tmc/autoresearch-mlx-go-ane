// Package specdec benchmarks stable and exploratory mlx-lm-generate vs
// mlx-ane-generate comparisons.
//
// Run release comparisons:
//
//	go test -bench=BenchmarkSpecDec -benchtime=1x -count=4 -run=^$ -timeout=30m ./benchmarks/specdec
//
// Run exploratory generation configs:
//
//	go test -bench=BenchmarkSpecDecExplore -benchtime=1x -count=4 -run=^$ -timeout=30m ./benchmarks/specdec
//
// Compare GPU vs ANE with benchstat:
//
//	benchstat -col /engine results.txt
//
// Filter to speculative only:
//
//	benchstat -col /engine -filter ".name:spec" results.txt
package specdec

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

var benchModel = flag.String("model", "", "single model ID override (default: MLX_BENCH_MODEL or stable model set)")
var benchModels = flag.String("models", "", "comma-separated model IDs (default: MLX_BENCH_MODELS or stable Qwen3/Qwen3.5 pair)")
var benchDraftModel = flag.String("draft-model", "", "draft model for specdec (default: MLX_BENCH_DRAFT_MODEL)")
var benchConfigs = flag.String("configs", "", "configs to benchmark: all, or comma-separated list")

var defaultBenchModels = []string{
	"mlx-community/Qwen3-0.6B-4bit",
	"mlx-community/Qwen3.5-0.8B-4bit",
}

var specdecExploreConfigLabels = []string{
	"decode-plane",
	"ane-forward-prefill",
	"ane-forward-all",
	"spec-standard",
	"spec-ssd",
	"spec-ssd+decode-plane",
	"spec-ssd+ane-speculative",
	"spec-ssd+ane-draft",
	"mtp",
	"mtp+decode-plane",
}

var cliStatsRe = regexp.MustCompile(`(?m)^(Prefill|Generation):\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec`)
var peakMemRe = regexp.MustCompile(`(?m)^Peak memory:\s+([\d.]+)\s+GB`)

type cliStats struct {
	prefillTokens int
	prefillTPS    float64
	genTokens     int
	genTPS        float64
	peakMemGB     float64
}

func parseCLIStats(output []byte) (cliStats, error) {
	var s cliStats
	for _, m := range cliStatsRe.FindAllSubmatch(output, -1) {
		tokens, e := strconv.Atoi(string(m[2]))
		if e != nil {
			return s, fmt.Errorf("parse %s tokens: %w", m[1], e)
		}
		tps, e := strconv.ParseFloat(string(m[3]), 64)
		if e != nil {
			return s, fmt.Errorf("parse %s tps: %w", m[1], e)
		}
		switch string(m[1]) {
		case "Prefill":
			s.prefillTokens = tokens
			s.prefillTPS = tps
		case "Generation":
			s.genTokens = tokens
			s.genTPS = tps
		}
	}
	if s.prefillTPS == 0 || s.genTPS == 0 {
		return s, fmt.Errorf("missing stats in output:\n%s", output)
	}
	if m := peakMemRe.FindSubmatch(output); m != nil {
		s.peakMemGB, _ = strconv.ParseFloat(string(m[1]), 64)
	}
	return s, nil
}

func benchmarkSkippableError(output []byte) string {
	text := string(output)
	switch {
	case strings.Contains(text, "does not have native mtp weights"):
		return "model does not expose native mtp weights"
	case strings.Contains(text, "does not expose hidden states"):
		return "model does not expose hidden states required for MTP"
	case strings.Contains(text, "flag provided but not defined: -ane-preset"):
		return "installed mlx-ane-generate does not support --ane-preset"
	default:
		return ""
	}
}

func runCLI(t testing.TB, cmd string, args []string) (cliStats, error) {
	t.Helper()
	if testing.Verbose() {
		t.Logf("running: %s %s", cmd, strings.Join(args, " "))
	}
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		if reason := benchmarkSkippableError(out); reason != "" {
			t.Skipf("%s: %s", cmd, reason)
		}
		return cliStats{}, fmt.Errorf("%s: %w\n%s", cmd, err, out)
	}
	if testing.Verbose() {
		t.Logf("%s output:\n%s", cmd, string(out))
	}
	return parseCLIStats(out)
}

func shortModelName(model string) string {
	if i := strings.LastIndex(model, "/"); i >= 0 {
		name := model[i+1:]
		if len(name) >= 32 && !strings.Contains(name, "-") {
			parent := filepath.Base(filepath.Dir(model))
			if parent != "." && parent != "/" {
				return parent
			}
		}
		return name
	}
	return model
}

func envOr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

func csvList(raw string) []string {
	var out []string
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		out = append(out, item)
	}
	return out
}

func resolveBenchModels() []string {
	switch {
	case *benchModels != "":
		return csvList(*benchModels)
	case *benchModel != "":
		return []string{*benchModel}
	}
	if raw := envOr("MLX_BENCH_MODELS", ""); raw != "" {
		return csvList(raw)
	}
	if model := envOr("MLX_BENCH_MODEL", ""); model != "" {
		return []string{model}
	}
	return append([]string(nil), defaultBenchModels...)
}

type runConfig struct {
	name  string
	cmd   string
	args  []string
	extra bool
}

// BenchmarkSpecDec runs the stable release comparison set.
func BenchmarkSpecDec(b *testing.B) {
	benchmarkSpecDec(b, false)
}

// BenchmarkSpecDecExplore runs the wider exploratory configuration set.
func BenchmarkSpecDecExplore(b *testing.B) {
	benchmarkSpecDec(b, true)
}

func benchmarkSpecDec(b *testing.B, includeExplore bool) {
	models := resolveBenchModels()
	draftModel := *benchDraftModel
	if draftModel == "" {
		draftModel = envOr("MLX_BENCH_DRAFT_MODEL", "")
	}
	prompt := envOr("MLX_BENCH_PROMPT", "Explain the theory of relativity in simple terms.")
	maxTokens := envOr("MLX_BENCH_MAX_TOKENS", "200")

	// Resolve which extra configs to include.
	raw := *benchConfigs
	if raw == "" {
		raw = envOr("MLX_BENCH_CONFIGS", "")
	}
	want := make(map[string]bool)
	if raw == "all" {
		for _, label := range specdecExploreConfigLabels {
			want[label] = true
		}
	} else if raw != "" {
		for _, s := range strings.Split(raw, ",") {
			want[strings.TrimSpace(s)] = true
		}
	}

	for _, model := range models {
		modelName := shortModelName(model)
		goBase := []string{"-model", model, "-prompt", prompt, "-max-tokens", maxTokens}
		aneBase := []string{"-model", model, "-prompt", prompt, "-max-tokens", maxTokens}

		configs := []runConfig{
			{
				name: "engine=Go/config=standard",
				cmd:  "mlx-lm-generate",
				args: append([]string{}, goBase...),
			},
			{
				name: "engine=ANE/config=standard",
				cmd:  "mlx-ane-generate",
				args: append([]string{}, aneBase...),
			},
			{
				name: "engine=ANE/config=auto-preset",
				cmd:  "mlx-ane-generate",
				args: append(append([]string{}, aneBase...), "--ane-preset", "auto"),
			},
			{
				name:  "engine=ANE/config=decode-plane",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, aneBase...), "--ane-decode-plane", "qwen35"),
				extra: true,
			},
			{
				name:  "engine=ANE/config=ane-forward-prefill",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, aneBase...), "--ane-forward", "prefill"),
				extra: true,
			},
			{
				name:  "engine=ANE/config=ane-forward-all",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, aneBase...), "--ane-forward", "all"),
				extra: true,
			},
		}

		if draftModel != "" {
			specGoBase := append(append([]string{}, goBase...), "-draft-model", draftModel)
			specANEBase := append(append([]string{}, aneBase...), "-draft-model", draftModel)

			configs = append(configs,
				runConfig{
					name:  "engine=Go/config=spec-standard",
					cmd:   "mlx-lm-generate",
					args:  append(append([]string{}, specGoBase...), "-speculative-path", "standard"),
					extra: true,
				},
				runConfig{
					name:  "engine=Go/config=spec-ssd",
					cmd:   "mlx-lm-generate",
					args:  append(append([]string{}, specGoBase...), "-speculative-path", "ssd"),
					extra: true,
				},
				runConfig{
					name:  "engine=ANE/config=spec-standard",
					cmd:   "mlx-ane-generate",
					args:  append(append([]string{}, specANEBase...), "-speculative-path", "standard"),
					extra: true,
				},
				runConfig{
					name:  "engine=ANE/config=spec-ssd",
					cmd:   "mlx-ane-generate",
					args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd"),
					extra: true,
				},
				runConfig{
					name:  "engine=ANE/config=spec-ssd+decode-plane",
					cmd:   "mlx-ane-generate",
					args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd", "--ane-decode-plane", "qwen35"),
					extra: true,
				},
				runConfig{
					name:  "engine=ANE/config=spec-ssd+ane-speculative",
					cmd:   "mlx-ane-generate",
					args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd", "--ane-speculative", "both-all"),
					extra: true,
				},
				runConfig{
					name:  "engine=ANE/config=spec-ssd+ane-draft",
					cmd:   "mlx-ane-generate",
					args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd", "--ane-draft-modelc", "auto"),
					extra: true,
				},
			)
		}

		configs = append(configs,
			runConfig{
				name:  "engine=Go/config=mtp",
				cmd:   "mlx-lm-generate",
				args:  append(append([]string{}, goBase...), "-mtp"),
				extra: true,
			},
			runConfig{
				name:  "engine=ANE/config=mtp",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, aneBase...), "-mtp"),
				extra: true,
			},
			runConfig{
				name:  "engine=ANE/config=mtp+decode-plane",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, aneBase...), "-mtp", "--ane-decode-plane", "qwen35"),
				extra: true,
			},
		)

		for _, cfg := range configs {
			if cfg.extra != includeExplore {
				continue
			}
			if includeExplore && len(want) > 0 {
				configName := specdecConfigLabel(cfg.name)
				if !want[configName] {
					continue
				}
			}

			prefix := "model=" + modelName + "/" + cfg.name
			if _, err := exec.LookPath(cfg.cmd); err != nil {
				b.Run(prefix, func(b *testing.B) { b.Skipf("%s not found in PATH", cfg.cmd) })
				continue
			}

			b.Run(prefix, func(b *testing.B) {
				for b.Loop() {
					time.Sleep(time.Second)
					b.ResetTimer()
					s, err := runCLI(b, cfg.cmd, cfg.args)
					if err != nil {
						b.Fatal(err)
					}
					b.ReportMetric(s.genTPS, "tok/s")
					b.ReportMetric(s.prefillTPS, "prefill-tok/s")
					b.ReportMetric(float64(s.genTokens), "gen-tokens")
					if s.peakMemGB > 0 {
						b.ReportMetric(s.peakMemGB, "peak-GB")
					}
				}
			})
		}
	}
}

func specdecConfigLabel(name string) string {
	if i := strings.LastIndex(name, "config="); i >= 0 {
		return name[i+len("config="):]
	}
	return name
}
