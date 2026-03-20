// Package specdec benchmarks mlx-lm-generate vs mlx-ane-generate across
// standard and speculative decoding configurations.
//
// Run default configs (standard decode, GPU vs ANE):
//
//	go test -bench=BenchmarkSpecDec -benchtime=1x -count=4 -run=^$ -timeout=30m ./benchmarks/specdec
//
// Full matrix (all specdec + ANE configs):
//
//	go test -bench=BenchmarkSpecDec -benchtime=1x -count=4 -run=^$ -timeout=30m ./benchmarks/specdec -args -configs=all
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

var benchModel = flag.String("model", "", "model ID (default: MLX_BENCH_MODEL or mlx-community/Qwen3.5-4B-4bit)")
var benchDraftModel = flag.String("draft-model", "", "draft model for specdec (default: MLX_BENCH_DRAFT_MODEL)")
var benchConfigs = flag.String("configs", "", "configs to benchmark: all, or comma-separated list")

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

func runCLI(t testing.TB, cmd string, args []string) (cliStats, error) {
	t.Helper()
	if testing.Verbose() {
		t.Logf("running: %s %s", cmd, strings.Join(args, " "))
	}
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
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

type runConfig struct {
	name  string
	cmd   string
	args  []string
	extra bool
}

// BenchmarkSpecDec compares mlx-lm-generate (GPU) vs mlx-ane-generate (ANE)
// across standard and speculative decoding configurations.
//
// Default configs run standard decode for both engines. Use -configs=all
// or MLX_BENCH_CONFIGS to expand the matrix to include speculative decoding,
// MTP, and ANE-specific decode plane modes.
func BenchmarkSpecDec(b *testing.B) {
	model := *benchModel
	if model == "" {
		model = envOr("MLX_BENCH_MODEL", "mlx-community/Qwen3.5-4B-4bit")
	}
	draftModel := *benchDraftModel
	if draftModel == "" {
		draftModel = envOr("MLX_BENCH_DRAFT_MODEL", "")
	}
	prompt := envOr("MLX_BENCH_PROMPT", "Explain the theory of relativity in simple terms.")
	maxTokens := envOr("MLX_BENCH_MAX_TOKENS", "200")
	modelName := shortModelName(model)

	goBase := []string{"-model", model, "-prompt", prompt, "-max-tokens", maxTokens}
	aneBase := []string{"-model", model, "-prompt", prompt, "-max-tokens", maxTokens}

	configs := []runConfig{
		// Default: standard decode, GPU vs ANE
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

		// ANE with decode plane enabled
		{
			name:  "engine=ANE/config=decode-plane",
			cmd:   "mlx-ane-generate",
			args:  append(append([]string{}, aneBase...), "--ane-decode-plane", "qwen35"),
			extra: true,
		},

		// ANE with forward routing
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

	// Speculative decoding configs (require draft model)
	if draftModel != "" {
		specGoBase := append(append([]string{}, goBase...), "-draft-model", draftModel)
		specANEBase := append(append([]string{}, aneBase...), "-draft-model", draftModel)

		configs = append(configs,
			// Go speculative (standard path)
			runConfig{
				name:  "engine=Go/config=spec-standard",
				cmd:   "mlx-lm-generate",
				args:  append(append([]string{}, specGoBase...), "-speculative-path", "standard"),
				extra: true,
			},
			// Go speculative (SSD path)
			runConfig{
				name:  "engine=Go/config=spec-ssd",
				cmd:   "mlx-lm-generate",
				args:  append(append([]string{}, specGoBase...), "-speculative-path", "ssd"),
				extra: true,
			},

			// ANE speculative (standard path)
			runConfig{
				name:  "engine=ANE/config=spec-standard",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, specANEBase...), "-speculative-path", "standard"),
				extra: true,
			},
			// ANE speculative (SSD path)
			runConfig{
				name:  "engine=ANE/config=spec-ssd",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd"),
				extra: true,
			},

			// ANE speculative + decode plane
			runConfig{
				name:  "engine=ANE/config=spec-ssd+decode-plane",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd", "--ane-decode-plane", "qwen35"),
				extra: true,
			},

			// ANE speculative + ANE draft routing
			runConfig{
				name:  "engine=ANE/config=spec-ssd+ane-speculative",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd", "--ane-speculative", "both-all"),
				extra: true,
			},

			// ANE speculative + draft model on ANE
			runConfig{
				name:  "engine=ANE/config=spec-ssd+ane-draft",
				cmd:   "mlx-ane-generate",
				args:  append(append([]string{}, specANEBase...), "-speculative-path", "ssd", "--ane-draft-modelc", "auto"),
				extra: true,
			},
		)
	}

	// MTP configs (no draft model needed)
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

	// Resolve which extra configs to include.
	raw := *benchConfigs
	if raw == "" {
		raw = envOr("MLX_BENCH_CONFIGS", "")
	}
	want := make(map[string]bool)
	if raw == "all" {
		for _, c := range configs {
			if c.extra {
				if i := strings.LastIndex(c.name, "config="); i >= 0 {
					want[c.name[i+len("config="):]] = true
				}
			}
		}
	} else if raw != "" {
		for _, s := range strings.Split(raw, ",") {
			want[strings.TrimSpace(s)] = true
		}
	}

	for _, cfg := range configs {
		if cfg.extra {
			configName := cfg.name
			if i := strings.LastIndex(configName, "config="); i >= 0 {
				configName = configName[i+len("config="):]
			}
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
				time.Sleep(time.Second) // settle between runs
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
