// Command mlx-lm-generate-ane is a simplified, ANE-focused text generator.
//
// It defaults to ANE decode plane mode "qwen35" (enabled) and removes all
// complexity unrelated to ANE generation: speculative decoding, VLM/images,
// GPU tracing, compilation, XTC sampling, logit bias, etc.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/tmc/mlx-go-lm/mlxlm"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/runtime/anedecode"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/random"
	"github.com/tmc/mlx-go/mlxc"

	"github.com/tmc/mlx-go-lm/mlxlm/decode"

	_ "github.com/tmc/mlx-go-ane/register"
)

func main() {
	loadANEEnv()

	mlxc.SetPanicOnError(false)
	models.UseFastRoPE = true
	models.UseFastSDPA = true

	if os.Getenv("MLXGO_FREE_QUEUE_THRESHOLD") == "" {
		mlx.SetFreeQueueThreshold(0)
	}

	// ANE path: always disable compilation.
	if err := mlx.SetCompileMode(mlx.CompileModeDisabled); err != nil {
		slog.Warn("failed to disable compile mode", "error", err)
	}

	flag.Parse()

	// Short flag handling
	if *prompt == "" && *promptShort != "" {
		*prompt = *promptShort
	}
	if *systemPrompt == "" && *systemPromptShort != "" {
		*systemPrompt = *systemPromptShort
	}
	if *maxTokens == 100 && *maxTokensShort != 100 {
		*maxTokens = *maxTokensShort
	}
	*temperature = getFloat64Flag(temperature, temperatureShort, 0.0)
	if *topP == 1.0 && *topPShort != 1.0 {
		*topP = *topPShort
	}
	if *minP == 0.0 && *minPShort != 0.0 {
		*minP = *minPShort
	}
	if *topK == 0 && *topKShort != 0 {
		*topK = *topKShort
	}
	if !*quiet && *quietShort {
		*quiet = true
	}

	if err := validateANEDecodePlaneFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(2)
	}

	// CPU profiling
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			slog.Error("could not create CPU profile", "error", err)
			os.Exit(1)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			slog.Error("could not start CPU profile", "error", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}

	// Setup logging
	opts := &slog.HandlerOptions{Level: slog.LevelInfo}
	if *debug {
		opts.Level = slog.LevelDebug
		mlx.SetEvalCounterEnabled(true)
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, opts)))

	if *modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	if *seed != 0 {
		random.Seed(uint64(*seed))
	}

	if err := run(*modelPath, *prompt, *maxTokens, *temperature, *topP, *minP, *topK, *extraEOSToken, *seed); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Write memory profile
	if *memProfile != "" {
		f, err := os.Create(*memProfile)
		if err != nil {
			slog.Error("could not create memory profile", "error", err)
			return
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			slog.Error("could not write memory profile", "error", err)
		}
	}
}

func run(modelDir, promptText string, maxTokensToGenerate int, temp, topPVal, minPVal float64, topKVal int, extraEOS string, seedVal int) error {
	slog.Info("mlx-lm-generate-ane starting...")

	resolvedPath, err := resolveModelPath(modelDir)
	if err != nil {
		return fmt.Errorf("failed to resolve model path: %w", err)
	}
	modelDir = resolvedPath

	ctx := context.Background()

	// Load model
	loadingStart := time.Now()
	pprof.Do(ctx, pprof.Labels("phase", "load_model"), func(ctx context.Context) {})
	model, _, err := models.LoadModel(modelDir)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	weightFiles, err := models.DiscoverWeightFiles(modelDir)
	if err != nil {
		return fmt.Errorf("failed to discover weights: %w", err)
	}
	slog.Info("Loading weights...", "files", len(weightFiles))
	if err := model.LoadWeights(weightFiles...); err != nil {
		return fmt.Errorf("failed to load weights: %w", err)
	}

	// ANE decode plane wrapping — the key enablement.
	model = wrapANEDecodePlane(model, modelDir)
	slog.Info("Model loaded", "duration", time.Since(loadingStart))

	// Set wired limit for optimal Metal GPU performance.
	if mlx.Metal.IsAvailable() {
		deviceInfo, err := mlx.Metal.DeviceInfo()
		if err == nil {
			maxRec := uint(deviceInfo.MaxRecommendedWorkingSetSize)
			oldLimit, _ := mlx.SetWiredLimit(maxRec)
			defer mlx.SetWiredLimit(oldLimit)
		}
	}

	// Sanitize weights to Float16 if needed.
	if sanitizable, ok := model.(interface{ SanitizeWeights(mlx.Dtype) error }); ok {
		slog.Info("Sanitizing model weights to Float16")
		if err := sanitizable.SanitizeWeights(mlx.Float16); err != nil {
			slog.Warn("Failed to sanitize weights", "error", err)
		}
	}

	if *memoryStats {
		runtime.GC()
		slog.Info("After model loading")
	}

	// Load tokenizer
	tok, err := mlxlm.LoadTokenizer(modelDir)
	if err != nil {
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Load prompt from file if @-prefixed.
	if strings.HasPrefix(promptText, "@") {
		bytes, err := os.ReadFile(promptText[1:])
		if err != nil {
			return fmt.Errorf("failed to read prompt file: %w", err)
		}
		promptText = string(bytes)
	}

	// Encode prompt
	var promptTokens []int32
	useChat := *useChatTemplate && !*noUseChatTemplate
	if !useChat {
		promptTokens, err = tok.Encode(promptText)
	} else {
		messages := []mlxlm.Message{
			{Role: "user", Content: promptText},
		}
		if *systemPrompt != "" {
			messages = append([]mlxlm.Message{{Role: "system", Content: *systemPrompt}}, messages...)
		}
		promptTokens, err = tok.ApplyChatTemplate(messages, true)
	}
	if err != nil {
		return fmt.Errorf("failed to encode prompt: %w", err)
	}
	slog.Info("Prompt Tokens", "tokens", promptTokens)

	// Setup display
	quietMode := *quiet
	var outw *bufio.Writer
	if !quietMode {
		outw = bufio.NewWriterSize(os.Stdout, 64*1024)
		_, _ = outw.WriteString(promptText)
		_ = outw.Flush()
	}

	// Create random state
	key := random.MustKey(uint64(seedVal))
	keyReshaped := mlx.MustReshape(key, []int{2}, nil)
	keyFresh := mlx.MustCopy(keyReshaped, nil)
	randState := decode.NewRandomState(keyFresh)

	modelConfig := model.Config()
	cacheConfig := buildCacheConfig()
	cache := createMultiLayerCache(modelConfig.NumLayers, cacheConfig)

	// Warmup
	if *warmup {
		if !quietMode {
			slog.Info("Warming up model...")
		}
		warmupInput, _ := mlx.Zeros([]int{1, 1}, mlx.Int32, nil)
		warmupCache := createMultiLayerCache(modelConfig.NumLayers, cacheConfig)
		warmupOut, _, err := model.Forward(warmupInput, warmupCache)
		if err != nil {
			slog.Warn("Warmup failed, continuing anyway", "error", err)
		} else {
			mlx.Eval(warmupOut)
			mlx.Synchronize(nil)
		}
		warmupInput.Free()

		if warmer, ok := model.(anedecode.ANEDecodePlaneWarmer); ok {
			if !quietMode {
				slog.Info("Warming up ANE decode plane...")
			}
			if err := warmer.PrewarmANEDecodePlane(); err != nil {
				slog.Warn("ANE decode plane warmup failed", "error", err)
			}
		}

		mlx.ClearCache()
	}

	// EOS tokens
	var eosTokenIDs []int32
	if tok != nil {
		eosTokenIDs = tok.EOSTokenIDs()
	}
	if extraEOS != "" && tok != nil {
		tokens, err := tok.Encode(extraEOS)
		if err == nil && len(tokens) > 0 {
			eosTokenIDs = append(eosTokenIDs, tokens[0])
		}
	}

	// Convert prompt to mlx.Array
	currentTokensArr, err := mlx.FromSlice(promptTokens, []int{1, len(promptTokens)}, mlx.Int32)
	if err != nil {
		return fmt.Errorf("failed to convert prompt tokens: %w", err)
	}
	defer currentTokensArr.Free()

	// Forward pass
	forwardPass := func(input *mlx.Array, c models.Cache) (*mlx.Array, models.Cache, error) {
		return model.Forward(input, c)
	}

	genStart := time.Now()
	var prefillDuration time.Duration
	var genDuration time.Duration
	var firstTokenReceived bool
	var generatedTokens []int32

	genIter := Generate(forwardPass, currentTokensArr, cache, temp, topPVal, minPVal, topKVal, eosTokenIDs, maxTokensToGenerate, tok, randState)

	for item, err := range genIter {
		if err != nil {
			return fmt.Errorf("generation failed: %w", err)
		}
		switch {
		case item.Chunk != nil:
			if !firstTokenReceived {
				prefillDuration = time.Since(genStart)
				firstTokenReceived = true
			}
			if outw != nil {
				_, _ = outw.WriteString(*item.Chunk)
				_ = outw.Flush()
			}
		case item.Info != nil:
			genDuration = time.Since(genStart)
			generatedTokens = make([]int32, item.Info.GenerationTokenCount)
		}
	}
	if outw != nil {
		_ = outw.Flush()
	}

	printStatistics(quietMode, len(promptTokens), generatedTokens, prefillDuration, genDuration)
	printANEDecodePlaneStatistics(quietMode, model)

	return nil
}
