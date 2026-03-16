package mlxgoane

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

var (
	testEngine       *Engine
	testPromptTokens []int32
)

func TestMain(m *testing.M) {
	modelID := os.Getenv("MODEL")
	if modelID == "" {
		modelID = DefaultModel
	}

	fmt.Fprintf(os.Stderr, "loading model %s...\n", modelID)
	var err error
	testEngine, err = setupEngine(modelID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "setup engine: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "model loaded from %s\n", testEngine.ModelPath)

	prompt := os.Getenv("PROMPT")
	if prompt == "" {
		prompt = DefaultPrompt
	}
	testPromptTokens, err = testEngine.encodePrompt(prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode prompt: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "prompt tokens: %d\n", len(testPromptTokens))

	if WarmupEnabled {
		fmt.Fprintf(os.Stderr, "warming up...\n")
		testEngine.warmup()
	}
	fmt.Fprintf(os.Stderr, "ready\n")

	code := m.Run()
	os.Exit(code)
}

// BenchmarkGenerate measures end-to-end generation (prefill + decode).
// The Go benchmark timer captures wall time per iteration (ns/op).
// Custom metrics report the last iteration's detailed breakdown.
func BenchmarkGenerate(b *testing.B) {
	var lastRes GenerateResult
	for b.Loop() {
		res, err := testEngine.generateN(testPromptTokens, GenerateTokens)
		if err != nil {
			b.Fatal(err)
		}
		lastRes = res
	}

	b.ReportMetric(lastRes.TokPerSec(), "tok/s")
	b.ReportMetric(lastRes.DecodeTokPerSec(), "decode_tok/s")
	b.ReportMetric(float64(lastRes.PrefillDuration.Microseconds())/1000, "prefill_ms")
	b.ReportMetric(float64(lastRes.GenerateDuration.Microseconds())/1000, "gen_ms")

	var peakMemGB float64
	if peakBytes, err := mlx.GetPeakMemory(); err == nil {
		peakMemGB = float64(peakBytes) / (1 << 30)
	} else {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		peakMemGB = float64(m.Sys) / (1 << 30)
	}
	b.ReportMetric(peakMemGB, "peak_mem_gb")
}

// BenchmarkPrefill measures prefill-only throughput (prompt processing + first token).
func BenchmarkPrefill(b *testing.B) {
	var lastRes GenerateResult
	for b.Loop() {
		res, err := testEngine.generateN(testPromptTokens, 1)
		if err != nil {
			b.Fatal(err)
		}
		lastRes = res
	}

	b.ReportMetric(lastRes.PrefillTokPerSec(), "prompt_tok/s")
}
