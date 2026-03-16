// harness.go provides model loading, generation, and timing functions
// used by bench_test.go and bench_ane_test.go.

package mlxgoane

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/mlx-go-lm/mlxlm"
	"github.com/tmc/mlx-go-lm/mlxlm/decode"
	"github.com/tmc/mlx-go-lm/mlxlm/hfcache"
	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/runtime/anedecode"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/random"
)

// GenerateResult holds timing and throughput data from a generation run.
type GenerateResult struct {
	PromptTokens    int
	GeneratedTokens int
	PrefillDuration time.Duration
	GenerateDuration time.Duration
	TotalDuration   time.Duration
}

// TokPerSec returns the end-to-end generation throughput.
func (r GenerateResult) TokPerSec() float64 {
	if r.TotalDuration <= 0 {
		return 0
	}
	return float64(r.GeneratedTokens) / r.TotalDuration.Seconds()
}

// DecodeTokPerSec returns the decode-only throughput (excluding prefill).
func (r GenerateResult) DecodeTokPerSec() float64 {
	d := r.GenerateDuration - r.PrefillDuration
	if d <= 0 {
		return 0
	}
	return float64(r.GeneratedTokens) / d.Seconds()
}

// PrefillTokPerSec returns the prefill throughput.
func (r GenerateResult) PrefillTokPerSec() float64 {
	if r.PrefillDuration <= 0 {
		return 0
	}
	return float64(r.PromptTokens) / r.PrefillDuration.Seconds()
}

// Engine holds a loaded model, tokenizer, and configuration for benchmarking.
type Engine struct {
	Model     models.LanguageModel
	Tokenizer mlxlm.Tokenizer
	ModelPath string
}

// setupEngine loads the model, weights, tokenizer, and wraps with ANE decode plane.
func setupEngine(modelID string) (*Engine, error) {
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

	// ANE decode plane wrapping.
	if ANEDecodePlaneMode != "off" {
		wrapped, err := anedecode.Wrap(model, anedecode.ANEDecodePlaneOptions{
			Mode:      ANEDecodePlaneMode,
			ModelPath: resolvedPath,
			Warn: func(format string, args ...any) {
				slog.Warn(fmt.Sprintf(format, args...))
			},
		})
		if err != nil {
			slog.Warn("ANE decode plane unavailable", "error", err)
		} else {
			model = wrapped
			mlx.SetCompileMode(mlx.CompileModeDisabled)
		}
	}

	// Sanitize weights to Float16.
	if sanitizable, ok := model.(interface{ SanitizeWeights(mlx.Dtype) error }); ok {
		sanitizable.SanitizeWeights(mlx.Float16)
	}

	tok, err := mlxlm.LoadTokenizer(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	return &Engine{
		Model:     model,
		Tokenizer: tok,
		ModelPath: resolvedPath,
	}, nil
}

// resolveModelPath resolves a model path to a local directory,
// downloading from HuggingFace if it looks like a model ID.
func resolveModelPath(path string) (string, error) {
	if _, err := os.Stat(path); err == nil {
		return filepath.Abs(path)
	}
	if strings.Contains(path, "/") {
		cache, err := hfcache.NewHFCacheWithOptions(true)
		if err != nil {
			return "", fmt.Errorf("initialize cache: %w", err)
		}
		return cache.GetModelPath(path)
	}
	return "", fmt.Errorf("model path not found: %s", path)
}

// newCache creates a fresh KV cache for the engine's model.
func (e *Engine) newCache() *models.MultiLayerCache {
	numLayers := e.Model.Config().NumLayers
	config := kvcache.DefaultConfig()
	switch CacheType {
	case "inplace":
		if mlx.HasKVCacheInplace() {
			config.Type = kvcache.TypeInplace
		}
	case "rotating":
		config.Type = kvcache.TypeRotating
		config.RotatingMaxSize = 2048
	case "prealloc":
		config.Type = kvcache.TypePrealloc
	}
	caches := make([]kvcache.Cache, numLayers)
	for i := range caches {
		c, err := kvcache.New(config)
		if err != nil {
			c, _ = kvcache.NewConcat()
		}
		caches[i] = c
	}
	return models.NewMultiLayerCacheFromList(caches)
}

// encodePrompt tokenizes the prompt using the configured template.
func (e *Engine) encodePrompt(prompt string) ([]int32, error) {
	if UseChatTemplate {
		messages := []mlxlm.Message{
			{Role: "user", Content: prompt},
		}
		return e.Tokenizer.ApplyChatTemplate(messages, true)
	}
	return e.Tokenizer.Encode(prompt)
}

// warmup runs a single-token forward pass to trigger any lazy compilation.
func (e *Engine) warmup() {
	input, _ := mlx.Zeros([]int{1, 1}, mlx.Int32, nil)
	cache := e.newCache()
	out, _, err := e.Model.Forward(input, cache)
	if err == nil {
		mlx.Eval(out)
		mlx.Synchronize(nil)
	}
	input.Free()

	if warmer, ok := e.Model.(anedecode.ANEDecodePlaneWarmer); ok {
		warmer.PrewarmANEDecodePlane()
	}
	mlx.ClearCache()
}

// generateN runs generation for maxTokens tokens and returns timing results.
func (e *Engine) generateN(promptTokens []int32, maxTokens int) (GenerateResult, error) {
	input, err := mlx.FromSlice(promptTokens, []int{1, len(promptTokens)}, mlx.Int32)
	if err != nil {
		return GenerateResult{}, fmt.Errorf("create input array: %w", err)
	}
	defer input.Free()

	cache := e.newCache()
	key := random.MustKey(0)
	keyReshaped := mlx.MustReshape(key, []int{2}, nil)
	keyFresh := mlx.MustCopy(keyReshaped, nil)
	randState := decode.NewRandomState(keyFresh)

	forwardPass := func(inp *mlx.Array, c models.Cache) (*mlx.Array, models.Cache, error) {
		return e.Model.Forward(inp, c)
	}

	var eosTokenIDs []int32
	if e.Tokenizer != nil {
		eosTokenIDs = e.Tokenizer.EOSTokenIDs()
	}

	opts := decode.Options{
		Temperature:      Temperature,
		TopP:             TopP,
		MinP:             MinP,
		TopK:             TopK,
		MaxTokens:        maxTokens,
		EOSTokens:        eosTokenIDs,
		SamplingStrategy: "lazy",
		UseStridedCache:  false,
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
			return GenerateResult{}, fmt.Errorf("generate token: %w", err)
		}
		if !firstToken {
			prefillDuration = time.Since(start)
			firstToken = true
		}
		generated++
	}

	total := time.Since(start)
	return GenerateResult{
		PromptTokens:     len(promptTokens),
		GeneratedTokens:  generated,
		PrefillDuration:  prefillDuration,
		GenerateDuration: total,
		TotalDuration:    total,
	}, nil
}
