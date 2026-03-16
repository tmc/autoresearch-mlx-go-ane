package mlxgoane

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/mlx-go-lm/mlxlm"
	"github.com/tmc/mlx-go-lm/mlxlm/decode"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/runtime/anedecode"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/random"
)

// BenchmarkInference benchmarks prefill, decode, and end-to-end generation
// with GPU-only and ANE modes. Sub-benchmarks use key=value format for
// benchstat projections.
//
// Run:
//
//	go test -bench=BenchmarkInference -benchtime=3x -count=10 -run=^$ -timeout=30m | tee results.txt
//
// Compare GPU vs ANE:
//
//	benchstat -col /mode results.txt
//
// Filter to decode only:
//
//	benchstat -col /mode -filter ".name:Decode" results.txt
func BenchmarkInference(b *testing.B) {
	modelID := os.Getenv("MODEL")
	if modelID == "" {
		modelID = DefaultModel
	}
	prompt := os.Getenv("PROMPT")
	if prompt == "" {
		prompt = DefaultPrompt
	}

	maxTokens := GenerateTokens

	modes := []struct {
		label   string
		aneMode string
	}{
		{"GPU", "off"},
		{"ANE", ANEDecodePlaneMode},
	}

	for _, mode := range modes {
		b.Run("mode="+mode.label, func(b *testing.B) {
			engine, err := newBenchEngine(modelID, mode.aneMode)
			if err != nil {
				b.Fatalf("setup %s: %v", mode.label, err)
			}
			if mode.aneMode != "off" && !engine.aneActive {
				b.Skipf("ANE runtime unavailable")
				return
			}

			engine.warmup()

			promptTokens, err := engine.encodePrompt(prompt)
			if err != nil {
				b.Fatalf("encode prompt: %v", err)
			}

			b.Run("Prefill", func(b *testing.B) {
				for b.Loop() {
					res, err := engine.generate(promptTokens, 1)
					if err != nil {
						b.Fatal(err)
					}
					b.ReportMetric(res.PrefillTokPerSec(), "prompt_tok/s")
					b.ReportMetric(float64(res.PrefillDuration.Milliseconds()), "prefill_ms")
				}
			})

			b.Run("Decode", func(b *testing.B) {
				var lastRes GenerateResult
				for b.Loop() {
					res, err := engine.generate(promptTokens, maxTokens)
					if err != nil {
						b.Fatal(err)
					}
					lastRes = res
				}
				b.StopTimer()
				b.ReportMetric(lastRes.DecodeTokPerSec(), "decode_tok/s")
			})

			b.Run("Generate", func(b *testing.B) {
				var lastRes GenerateResult
				for b.Loop() {
					res, err := engine.generate(promptTokens, maxTokens)
					if err != nil {
						b.Fatal(err)
					}
					lastRes = res
				}
				b.StopTimer()
				b.ReportMetric(lastRes.TokPerSec(), "tok/s")
				b.ReportMetric(float64(lastRes.PrefillDuration.Milliseconds()), "prefill_ms")
				b.ReportMetric(float64(lastRes.GenerateDuration.Milliseconds()), "gen_ms")
				peakMemGB := 0.0
				if peakBytes, err := mlx.GetPeakMemory(); err == nil {
					peakMemGB = float64(peakBytes) / (1024 * 1024 * 1024)
				} else {
					var m runtime.MemStats
					runtime.ReadMemStats(&m)
					peakMemGB = float64(m.Sys) / (1024 * 1024 * 1024)
				}
				b.ReportMetric(peakMemGB, "peak_mem_gb")
			})
		})
	}
}

// benchEngine holds a loaded model for benchmarking with a specific mode.
type benchEngine struct {
	model     models.LanguageModel
	tokenizer mlxlm.Tokenizer
	modelPath string
	aneActive bool
}

func newBenchEngine(modelID, aneMode string) (*benchEngine, error) {
	resolvedPath, err := resolveModelPath(modelID)
	if err != nil {
		return nil, fmt.Errorf("resolve model: %w", err)
	}

	model, _, err := models.LoadModel(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	weightFiles, err := models.DiscoverWeightFiles(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("discover weights: %w", err)
	}
	if err := model.LoadWeights(weightFiles...); err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	aneActive := false
	if aneMode != "off" {
		wrapped, err := anedecode.Wrap(model, anedecode.ANEDecodePlaneOptions{
			Mode:      aneMode,
			ModelPath: resolvedPath,
			Warn: func(format string, args ...any) {
				slog.Warn(fmt.Sprintf(format, args...))
			},
		})
		if err != nil {
			slog.Warn("ANE decode plane unavailable", "error", err)
		} else {
			model = wrapped
			aneActive = true
		}
	}

	tok, err := mlxlm.LoadTokenizer(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	return &benchEngine{
		model:     model,
		tokenizer: tok,
		modelPath: resolvedPath,
		aneActive: aneActive,
	}, nil
}

func (e *benchEngine) warmup() {
	input, _ := mlx.Zeros([]int{1, 1}, mlx.Int32, nil)
	cache := models.NewMultiLayerCache(e.model.Config().NumLayers)
	out, _, err := e.model.Forward(input, cache)
	if err == nil {
		mlx.Eval(out)
		mlx.Synchronize(nil)
	}
	input.Free()

	if warmer, ok := e.model.(anedecode.ANEDecodePlaneWarmer); ok {
		warmer.PrewarmANEDecodePlane()
	}
	mlx.ClearCache()
}

func (e *benchEngine) encodePrompt(prompt string) ([]int32, error) {
	messages := []mlxlm.Message{
		{Role: "user", Content: prompt},
	}
	return e.tokenizer.ApplyChatTemplate(messages, true)
}

func (e *benchEngine) generate(promptTokens []int32, maxTokens int) (GenerateResult, error) {
	input, err := mlx.FromSlice(promptTokens, []int{1, len(promptTokens)}, mlx.Int32)
	if err != nil {
		return GenerateResult{}, fmt.Errorf("create input: %w", err)
	}
	defer input.Free()

	cache := models.NewMultiLayerCache(e.model.Config().NumLayers)
	key := random.MustKey(0)
	keyReshaped := mlx.MustReshape(key, []int{2}, nil)
	keyFresh := mlx.MustCopy(keyReshaped, nil)
	randState := decode.NewRandomState(keyFresh)

	forwardPass := func(inp *mlx.Array, c models.Cache) (*mlx.Array, models.Cache, error) {
		return e.model.Forward(inp, c)
	}

	opts := decode.Options{
		Temperature:      0.0,
		TopP:             1.0,
		MaxTokens:        maxTokens,
		EOSTokens:        e.tokenizer.EOSTokenIDs(),
		SamplingStrategy: "lazy",
	}

	start := time.Now()
	var prefillDuration time.Duration
	var firstToken bool
	var generated int

	iterator, err := decode.NewTokenIterator(context.Background(), forwardPass, input, cache, opts)
	if err != nil {
		return GenerateResult{}, fmt.Errorf("create iterator: %w", err)
	}
	iterator.SetRandomState(randState)

	for _, err := range iterator.Tokens() {
		if err != nil {
			return GenerateResult{}, fmt.Errorf("generate: %w", err)
		}
		if !firstToken {
			prefillDuration = time.Since(start)
			firstToken = true
		}
		generated++
	}

	total := time.Since(start)
	return GenerateResult{
		PromptTokens:    len(promptTokens),
		GeneratedTokens: generated,
		PrefillDuration: prefillDuration,
		GenerateDuration: total,
		TotalDuration:   total,
	}, nil
}
