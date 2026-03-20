// Package training benchmarks mlx-lm-train vs mlx-ane-train across
// fine-tuning configurations and ANE routing profiles.
//
// Run default configs (LoRA, GPU vs ANE):
//
//	go test -bench=BenchmarkTraining -benchtime=1x -count=4 -run=^$ -timeout=30m ./benchmarks/training
//
// Full matrix (all ANE routing profiles):
//
//	go test -bench=BenchmarkTraining -benchtime=1x -count=4 -run=^$ -timeout=30m ./benchmarks/training -args -configs=all
//
// Compare GPU vs ANE with benchstat:
//
//	benchstat -col /engine results.txt
//
// Filter to a specific fine-tune type:
//
//	benchstat -col /engine -filter ".name:lora" results.txt
package training

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
var benchData = flag.String("data", "", "training data directory (default: MLX_BENCH_DATA or bundled data)")
var benchConfigs = flag.String("configs", "", "configs to benchmark: all, or comma-separated list")

// Training output format (verbose):
//
//	Train      1     2.3456 loss      1234.5 tokens/sec       3.45 it/sec  1.000e-05 lr  2.72 GB      1234 trained-tokens
//
// Training output format (short):
//
//	Iter 1: loss=2.3456, 3.5 it/s, 1234 tok/s
var trainStatsVerboseRe = regexp.MustCompile(
	`(?m)^Train\s+\d+\s+([\d.]+)\s+loss\s+([\d.]+)\s+tokens/sec\s+([\d.]+)\s+it/sec\s+([\d.e+-]+)\s+lr\s+([\d.]+)\s+GB`,
)
var trainStatsShortRe = regexp.MustCompile(
	`(?m)Iter\s+(\d+):\s+loss=([\d.]+),\s+([\d.]+)\s+it/s,\s+([\d.]+)\s+tok/s`,
)
var valLossRe = regexp.MustCompile(`(?m)val_loss=([\d.]+)`)

type trainStats struct {
	lastLoss   float64
	tokPerSec  float64
	itPerSec   float64
	peakMemGB  float64
	valLoss    float64
	iterations int
}

func parseTrainStats(output []byte) (trainStats, error) {
	var s trainStats

	// Try verbose format first
	matches := trainStatsVerboseRe.FindAllSubmatch(output, -1)
	if len(matches) > 0 {
		last := matches[len(matches)-1]
		s.lastLoss, _ = strconv.ParseFloat(string(last[1]), 64)
		s.tokPerSec, _ = strconv.ParseFloat(string(last[2]), 64)
		s.itPerSec, _ = strconv.ParseFloat(string(last[3]), 64)
		s.peakMemGB, _ = strconv.ParseFloat(string(last[5]), 64)
		s.iterations = len(matches)
	} else {
		// Try short format
		shortMatches := trainStatsShortRe.FindAllSubmatch(output, -1)
		if len(shortMatches) == 0 {
			return s, fmt.Errorf("no training stats found in output:\n%s", output)
		}
		last := shortMatches[len(shortMatches)-1]
		s.lastLoss, _ = strconv.ParseFloat(string(last[2]), 64)
		s.itPerSec, _ = strconv.ParseFloat(string(last[3]), 64)
		s.tokPerSec, _ = strconv.ParseFloat(string(last[4]), 64)
		s.iterations = len(shortMatches)
	}

	if m := valLossRe.FindAllSubmatch(output, -1); len(m) > 0 {
		s.valLoss, _ = strconv.ParseFloat(string(m[len(m)-1][1]), 64)
	}
	return s, nil
}

func runTrainCLI(t testing.TB, cmd string, args []string) (trainStats, error) {
	t.Helper()
	if testing.Verbose() {
		t.Logf("running: %s %s", cmd, strings.Join(args, " "))
	}
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		return trainStats{}, fmt.Errorf("%s: %w\n%s", cmd, err, out)
	}
	if testing.Verbose() {
		t.Logf("%s output:\n%s", cmd, string(out))
	}
	return parseTrainStats(out)
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

// findDefaultData locates the bundled training data in the mlx-go-lm repo.
// Prefers the 20-sample dataset over the 1-sample smoke test data.
func findDefaultData() string {
	candidates := []string{
		// 20-sample dataset (preferred)
		filepath.Join(os.Getenv("HOME"), "go/src/github.com/tmc/mlx-go/examples/mlx-go-lm/cmd/mlx-lm-train/data"),
		filepath.Join(os.Getenv("HOME"), "ml-explore/mlx-go/examples/mlx-go-lm/cmd/mlx-lm-train/data"),
		// 1-sample fallback
		filepath.Join(os.Getenv("HOME"), "go/src/github.com/tmc/mlx-go/examples/mlx-go-lm/data"),
		filepath.Join(os.Getenv("HOME"), "ml-explore/mlx-go/examples/mlx-go-lm/data"),
	}
	for _, d := range candidates {
		if _, err := os.Stat(filepath.Join(d, "train.jsonl")); err == nil {
			return d
		}
	}
	return ""
}

type runConfig struct {
	name  string
	cmd   string
	args  []string
	extra bool
}

// BenchmarkTraining compares mlx-lm-train (GPU) vs mlx-ane-train (ANE)
// across fine-tuning types and ANE routing configurations.
//
// Default configs run LoRA fine-tuning for both engines. Use -configs=all
// or MLX_BENCH_CONFIGS to expand the matrix to include DoRA, full
// fine-tuning, and ANE routing profiles.
func BenchmarkTraining(b *testing.B) {
	model := *benchModel
	if model == "" {
		model = envOr("MLX_BENCH_MODEL", "mlx-community/Qwen3.5-4B-4bit")
	}
	data := *benchData
	if data == "" {
		data = envOr("MLX_BENCH_DATA", "")
	}
	if data == "" {
		data = findDefaultData()
	}
	if data == "" {
		b.Skip("no training data found; set MLX_BENCH_DATA or -data flag")
	}

	iters := envOr("MLX_BENCH_TRAIN_ITERS", "20")
	batchSize := envOr("MLX_BENCH_BATCH_SIZE", "1")
	modelName := shortModelName(model)

	// Small, fast training runs for benchmarking throughput.
	goBase := []string{
		"-model", model,
		"-data", data,
		"-train",
		"-iters", iters,
		"-batch-size", batchSize,
		"-steps-per-report", iters, // report at end
		"-steps-per-eval", "0",     // skip validation during timed run
		"-save-every", "0",         // don't save adapters
		"-num-layers", "4",
		"-max-seq-length", "512",
		"-seed", "42",
	}
	aneBase := append([]string{}, goBase...)

	configs := []runConfig{
		// Default: LoRA, GPU vs ANE
		{
			name: "engine=Go/config=lora",
			cmd:  "mlx-lm-train",
			args: append(append([]string{}, goBase...), "-fine-tune-type", "lora"),
		},
		{
			name: "engine=ANE/config=lora",
			cmd:  "mlx-ane-train",
			args: append(append([]string{}, aneBase...), "-fine-tune-type", "lora"),
		},

		// DoRA
		{
			name:  "engine=Go/config=dora",
			cmd:   "mlx-lm-train",
			args:  append(append([]string{}, goBase...), "-fine-tune-type", "dora"),
			extra: true,
		},
		{
			name:  "engine=ANE/config=dora",
			cmd:   "mlx-ane-train",
			args:  append(append([]string{}, aneBase...), "-fine-tune-type", "dora"),
			extra: true,
		},

		// Full fine-tuning
		{
			name:  "engine=Go/config=full",
			cmd:   "mlx-lm-train",
			args:  append(append([]string{}, goBase...), "-fine-tune-type", "full"),
			extra: true,
		},
		{
			name:  "engine=ANE/config=full",
			cmd:   "mlx-ane-train",
			args:  append(append([]string{}, aneBase...), "-fine-tune-type", "full"),
			extra: true,
		},

		// ANE routing profiles (LoRA)
		{
			name:  "engine=ANE/config=lora+forward-all",
			cmd:   "mlx-ane-train",
			args:  append(append([]string{}, aneBase...), "-fine-tune-type", "lora", "--ane-forward", "all"),
			extra: true,
		},
		{
			name:  "engine=ANE/config=lora+route-balanced",
			cmd:   "mlx-ane-train",
			args:  append(append([]string{}, aneBase...), "-fine-tune-type", "lora", "--ane-forward", "all", "--ane-route-profile", "balanced"),
			extra: true,
		},
		{
			name:  "engine=ANE/config=lora+route-conservative",
			cmd:   "mlx-ane-train",
			args:  append(append([]string{}, aneBase...), "-fine-tune-type", "lora", "--ane-forward", "all", "--ane-route-profile", "conservative"),
			extra: true,
		},
		{
			name:  "engine=ANE/config=lora+route-aggressive",
			cmd:   "mlx-ane-train",
			args:  append(append([]string{}, aneBase...), "-fine-tune-type", "lora", "--ane-forward", "all", "--ane-route-profile", "aggressive"),
			extra: true,
		},

		// ANE with fallback
		{
			name:  "engine=ANE/config=lora+fallback",
			cmd:   "mlx-ane-train",
			args:  append(append([]string{}, aneBase...), "-fine-tune-type", "lora", "--ane-forward", "all", "--ane-allow-fallback"),
			extra: true,
		},
	}

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
				s, err := runTrainCLI(b, cfg.cmd, cfg.args)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(s.tokPerSec, "tok/s")
				b.ReportMetric(s.itPerSec, "it/s")
				b.ReportMetric(s.lastLoss, "loss")
				if s.peakMemGB > 0 {
					b.ReportMetric(s.peakMemGB, "peak-GB")
				}
				if s.valLoss > 0 {
					b.ReportMetric(s.valLoss, "val-loss")
				}
			}
		})
	}
}
