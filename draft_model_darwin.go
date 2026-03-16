//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
	"github.com/tmc/mlx-go/modelir"
	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
)

const defaultANEDraftModelKey = "s"

// ANEDraftModel runs a small draft model on ANE for speculative decoding.
//
// The daemon-backed client model is opened once and reused per token.
//
// ANEDraftModel satisfies SurfaceSyncRuntime when the active path owns a
// persistent surface plan.
type ANEDraftModel struct {
	mu sync.Mutex

	client       appleneuralengine.ANEClient
	clientModel  objectivec.IObject
	clientOpts   foundation.NSMutableDictionary
	modelPath    string
	modelKey     string
	inMemoryMIL  appleneuralengine.ANEInMemoryModel
	milLayerEval []draftMILLayerEval
	milMultiPlan *MultiSurfaceEvalPlan
	milInput     *IOSurfaceFloat32
	milPosCos    *IOSurfaceFloat32
	milPosSin    *IOSurfaceFloat32
	milAttnMask  *IOSurfaceFloat32
	milOutput    *IOSurfaceFloat32
	milState     []*IOSurfaceFloat32
	milKCurr     *IOSurfaceFloat32
	milVCurr     *IOSurfaceFloat32
	inputBytes   uintptr
	outputBytes  uintptr
	outputDim    int
	hiddenDim    int
	vocabSize    int
	embeddings   []float32 // vocabSize x hiddenDim

	inBuf  []float32
	outBuf []float32
	closed bool

	kvState     *DraftKVSurfaceState
	decodePos   int
	ropeCos     []float32
	ropeSin     []float32
	ropeHeadDim int
	ropeAttnDim int
	ropeMaxSeq  int
	ropeTmpCos  []float32
	ropeTmpSin  []float32
	attnMaskTmp []float32

	statefulMIL *statefulMILRuntime
}

type draftMILLayerEval struct {
	model       appleneuralengine.ANEInMemoryModel
	clientModel *ANEClientMILModel
	plan        *milTransformerPlan
}

type statefulMILRuntime struct {
	label       string
	milText     string
	files       []modelWeightFile
	dynamicRoPE bool
	planConfig  MultiSurfaceEvalPlanConfig
}

// DecodeStateSnapshot captures the externally visible decode state for a
// stateful MIL draft model.
//
// The snapshot is intended for short-lived rewind/restore flows in speculative
// decoding. It is not a stable serialization format.
type DecodeStateSnapshot struct {
	decodePos int
	milState  [][]float32
}

type milTransformerPlan struct {
	input  *IOSurfaceFloat32
	posCos *IOSurfaceFloat32
	posSin *IOSurfaceFloat32
	output *IOSurfaceFloat32
	states []*IOSurfaceFloat32
	plan   *MultiSurfaceEvalPlan
}

// DraftGenerateTiming reports where time is spent in one draft generation call.
type DraftGenerateTiming struct {
	ANE   time.Duration
	CPU   time.Duration
	Total time.Duration
}

type draftConfigRoot struct {
	ModelType  string          `json:"model_type"`
	HiddenSize *int            `json:"hidden_size"`
	VocabSize  *int            `json:"vocab_size"`
	TextConfig json.RawMessage `json:"text_config"`
}

type draftTextConfig struct {
	ModelType  string `json:"model_type"`
	HiddenSize *int   `json:"hidden_size"`
	VocabSize  *int   `json:"vocab_size"`
}

// NewANEDraftModel opens a draft model using a bridgeless
// _ANEClient + _ANEModel execution path.
func NewANEDraftModel(modelcPath string, hiddenDim, vocabSize int, embeddings []float32) (*ANEDraftModel, error) {
	return NewANEDraftModelWithOutputDim(modelcPath, hiddenDim, vocabSize, vocabSize, embeddings)
}

// NewANEDraftModelWithOutputDim opens a draft model with an explicit
// output dimension. outputDim usually equals vocabSize for logits models, but can
// be set to hiddenDim when the ANE model returns hidden states.
func NewANEDraftModelWithOutputDim(modelcPath string, hiddenDim, vocabSize, outputDim int, embeddings []float32) (*ANEDraftModel, error) {
	if modelcPath == "" {
		return nil, fmt.Errorf("new ane draft model: model path is empty")
	}
	if hiddenDim <= 0 || vocabSize <= 0 || outputDim <= 0 {
		return nil, fmt.Errorf(
			"new ane draft model: invalid dims hiddenDim=%d vocabSize=%d outputDim=%d",
			hiddenDim,
			vocabSize,
			outputDim,
		)
	}
	wantEmb := hiddenDim * vocabSize
	if len(embeddings) > 0 && len(embeddings) != wantEmb {
		return nil, fmt.Errorf("new ane draft model: embeddings len=%d want=%d (or 0)", len(embeddings), wantEmb)
	}

	key := os.Getenv("MLXGO_ANE_DRAFT_MODEL_KEY")
	if key == "" {
		key = defaultANEDraftModelKey
	}
	client, modelObj, opts, err := openDraftClientModel(modelcPath, key, defaultANEQoS)
	if err != nil {
		return nil, fmt.Errorf("new ane draft model: open client model: %w", err)
	}

	return &ANEDraftModel{
		client:      client,
		clientModel: modelObj,
		clientOpts:  opts,
		modelPath:   modelcPath,
		modelKey:    key,
		inputBytes:  uintptr(hiddenDim * 4),
		outputBytes: uintptr(outputDim * 4),
		outputDim:   outputDim,
		hiddenDim:   hiddenDim,
		vocabSize:   vocabSize,
		embeddings:  append([]float32(nil), embeddings...),
		inBuf:       make([]float32, hiddenDim),
		outBuf:      make([]float32, outputDim),
	}, nil
}

// NewANEDraftModelFromMILTransformer builds and loads a strict MIL in-memory
// model (isMILModel initializer required) and wraps it in ANEDraftModel.
func NewANEDraftModelFromMILTransformer(
	cfg MILTransformerConfig,
	weights MILTransformerWeights,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
) (*ANEDraftModel, error) {
	if hiddenDim <= 0 || vocabSize <= 0 || outputDim <= 0 {
		return nil, fmt.Errorf(
			"new ane draft model from mil: invalid dims hiddenDim=%d vocabSize=%d outputDim=%d",
			hiddenDim,
			vocabSize,
			outputDim,
		)
	}
	wantEmb := hiddenDim * vocabSize
	if len(embeddings) > 0 && len(embeddings) != wantEmb {
		return nil, fmt.Errorf("new ane draft model from mil: embeddings len=%d want=%d (or 0)", len(embeddings), wantEmb)
	}
	milText, files, err := anereify.BuildMILTransformerArtifacts(
		cfg,
		weights,
	)
	if err != nil {
		return nil, fmt.Errorf("new ane draft model from mil: build artifacts: %w", err)
	}
	return newANEDraftModelFromMILArtifacts(
		cfg,
		milText,
		cloneModelWeightFiles(files),
		hiddenDim,
		vocabSize,
		outputDim,
		embeddings,
		nil,
	)
}

// NewANEDraftModelFromModelIRProgram lowers prog through the canonical ANE
// reifier and opens the resulting draft runtime.
func NewANEDraftModelFromModelIRProgram(
	prog *modelir.Program,
	opts ReifyOptions,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
) (*ANEDraftModel, ReifiedMIL, error) {
	return NewANEDraftModelFromModelIRProgramWithPlanConfig(
		prog,
		opts,
		hiddenDim,
		vocabSize,
		outputDim,
		embeddings,
		nil,
	)
}

// NewANEDraftModelFromModelIRProgramWithPlanConfig lowers prog through the
// canonical ANE reifier and opens the resulting draft runtime using the
// provided multi-surface plan configuration when non-nil.
func NewANEDraftModelFromModelIRProgramWithPlanConfig(
	prog *modelir.Program,
	opts ReifyOptions,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
	planCfg *MultiSurfaceEvalPlanConfig,
) (*ANEDraftModel, ReifiedMIL, error) {
	trace := func(phase string, args ...any) {
		if !envTruthy("MLXGO_ANE_DEBUG_MODELIR_COMPILE") {
			return
		}
		slog.Info("ANE draft modelir compile phase", append([]any{"phase", phase}, args...)...)
	}
	if prog == nil {
		return nil, ReifiedMIL{}, fmt.Errorf("new ane draft model from modelir: nil program")
	}
	trace("reify_begin")
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		return nil, ReifiedMIL{}, fmt.Errorf("new ane draft model from modelir: reify: %w", err)
	}
	trace("reify_done", "weights", len(reified.WeightFiles), "selected_layers", reified.SelectedLayers)
	trace("compile_begin")
	model, err := newANEDraftModelFromMILArtifacts(
		reified.TransformerConfig,
		reified.MILText,
		cloneModelWeightFiles(reified.WeightFiles),
		hiddenDim,
		vocabSize,
		outputDim,
		embeddings,
		planCfg,
	)
	if err == nil {
		trace("compile_done", "compile_fallback", false)
		return model, reified, nil
	}
	trace("compile_failed", "error", err)
	if disableModelIRCompileFallback() {
		return nil, ReifiedMIL{}, fmt.Errorf(
			"new ane draft model from modelir: %w (fallback disabled by MLXGO_ANE_MODELIR_DISABLE_COMPILE_FALLBACK)",
			err,
		)
	}
	if !isANECCompileFailure(err) {
		return nil, ReifiedMIL{}, fmt.Errorf("new ane draft model from modelir: %w", err)
	}

	baseErr := err
	attemptErrs := []string{fmt.Sprintf("base: %v", err)}
	for _, fallback := range modelIRCompileFallbackProfiles(opts, reified.TransformerConfig) {
		trace("fallback_reify_begin", "fallback", fallback.Key)
		tryReified, reifyErr := ReifyToANEMIL(prog, fallback.Opts)
		if reifyErr != nil {
			trace("fallback_reify_failed", "fallback", fallback.Key, "error", reifyErr)
			attemptErrs = append(attemptErrs, fmt.Sprintf("%s reify: %v", fallback.Label, reifyErr))
			continue
		}
		trace("fallback_reify_done", "fallback", fallback.Key, "weights", len(tryReified.WeightFiles))
		trace("fallback_compile_begin", "fallback", fallback.Key)
		tryModel, compileErr := newANEDraftModelFromMILArtifacts(
			tryReified.TransformerConfig,
			tryReified.MILText,
			cloneModelWeightFiles(tryReified.WeightFiles),
			hiddenDim,
			vocabSize,
			outputDim,
			embeddings,
			planCfg,
		)
		if compileErr == nil {
			tryReified.Diagnostics = append(tryReified.Diagnostics, ReifyDiagnostic{
				Code: "compile_fallback",
				Message: fmt.Sprintf(
					"in-memory compile fallback applied: %s",
					fallback.Key,
				),
			})
			trace("fallback_compile_done", "fallback", fallback.Key)
			return tryModel, tryReified, nil
		}
		trace("fallback_compile_failed", "fallback", fallback.Key, "error", compileErr)
		attemptErrs = append(attemptErrs, fmt.Sprintf("%s compile: %v", fallback.Label, compileErr))
	}

	return nil, ReifiedMIL{}, fmt.Errorf(
		"new ane draft model from modelir: %w (fallback attempts: %s)",
		baseErr,
		strings.Join(attemptErrs, "; "),
	)
}

// NewANEDraftModelFromModelIRProgramLayerStack derives canonical transformer
// weights from prog and compiles one ANE model per selected layer. This is a
// fallback path for multi-layer experiments when the monolithic direct ModelIR
// compile path is too fragile on the current host.
func NewANEDraftModelFromModelIRProgramLayerStack(
	prog *modelir.Program,
	opts ReifyOptions,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
) (*ANEDraftModel, ReifiedMIL, error) {
	return NewANEDraftModelFromModelIRProgramLayerStackWithPlanConfig(
		prog,
		opts,
		hiddenDim,
		vocabSize,
		outputDim,
		embeddings,
		nil,
	)
}

// NewANEDraftModelFromModelIRProgramLayerStack derives canonical transformer
// weights from prog and compiles one ANE model per selected layer using the
// provided multi-surface plan configuration when non-nil.
func NewANEDraftModelFromModelIRProgramLayerStackWithPlanConfig(
	prog *modelir.Program,
	opts ReifyOptions,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
	planCfg *MultiSurfaceEvalPlanConfig,
) (*ANEDraftModel, ReifiedMIL, error) {
	if prog == nil {
		return nil, ReifiedMIL{}, fmt.Errorf("new ane draft model layer stack from modelir: nil program")
	}
	cfg := opts.TransformerConfig
	weights := opts.TransformerWeights
	if len(weights.Layers) == 0 {
		derivedCfg, derivedWeights, err := DeriveTransformerFromProgram(prog)
		if err != nil {
			return nil, ReifiedMIL{}, fmt.Errorf(
				"new ane draft model layer stack from modelir: derive transformer from modelir weights: %w",
				err,
			)
		}
		cfg = mergeDerivedTransformerConfig(derivedCfg, cfg)
		weights = derivedWeights
	}
	requested := opts.RequestedLayers
	if requested <= 0 {
		requested = cfg.NumLayers
	}
	if requested <= 0 {
		requested = len(weights.Layers)
	}
	if requested <= 0 {
		return nil, ReifiedMIL{}, fmt.Errorf("new ane draft model layer stack from modelir: requested layers must be > 0")
	}
	selected := opts.SelectedLayers
	if selected <= 0 {
		selected = requested
	}
	if selected <= 0 {
		return nil, ReifiedMIL{}, fmt.Errorf("new ane draft model layer stack from modelir: selected layers must be > 0")
	}
	if selected > requested {
		return nil, ReifiedMIL{}, fmt.Errorf(
			"new ane draft model layer stack from modelir: selected %d/%d layers (selected must be <= requested)",
			selected,
			requested,
		)
	}
	if len(weights.Layers) < selected {
		return nil, ReifiedMIL{}, fmt.Errorf(
			"new ane draft model layer stack from modelir: transformer weights contain %d layers, need %d",
			len(weights.Layers),
			selected,
		)
	}
	reified := ReifiedMIL{
		TransformerConfig: cfg,
		RequestedLayers:   requested,
		SelectedLayers:    selected,
	}
	if fn, err := entryFunction(prog); err == nil {
		reified.Inputs = append([]modelir.Value(nil), fn.Inputs...)
		reified.Outputs = resolveReturnValues(fn)
		if pattern, detectErr := DetectWrapperPattern(prog); detectErr == nil {
			reified.Wrapper = pattern
		} else if strings.EqualFold(strings.TrimSpace(prog.Target), "coreml-ane-v1") ||
			strings.EqualFold(strings.TrimSpace(prog.Target), "mil-ane-v1") {
			reified.Wrapper = WrapperPatternDirectProgram
		}
	}
	if selected < requested {
		if !allowPartialReify(opts.AllowPartial) {
			return nil, ReifiedMIL{}, fmt.Errorf(
				"new ane draft model layer stack from modelir: selected %d/%d layers; partial coverage disabled (set MLXGO_ANE_DRAFT_ALLOW_PARTIAL=1 for experiments)",
				selected,
				requested,
			)
		}
		reified.Diagnostics = append(reified.Diagnostics, ReifyDiagnostic{
			Code: "partial_coverage",
			Message: fmt.Sprintf(
				"selected %d/%d layers; continuing in partial experimental mode",
				selected,
				requested,
			),
		})
	}
	cfg.NumLayers = selected
	weights.Layers = append([]MILTransformerLayerWeights(nil), weights.Layers[:selected]...)
	reified.TransformerConfig = cfg

	model, err := NewANEDraftModelFromMILTransformerLayerStackWithPlanConfig(
		cfg,
		weights,
		hiddenDim,
		vocabSize,
		outputDim,
		embeddings,
		planCfg,
	)
	if err == nil {
		return model, reified, nil
	}
	if disableModelIRCompileFallback() {
		return nil, ReifiedMIL{}, fmt.Errorf(
			"new ane draft model layer stack from modelir: %w (fallback disabled by MLXGO_ANE_MODELIR_DISABLE_COMPILE_FALLBACK)",
			err,
		)
	}
	if !isANECCompileFailure(err) {
		return nil, ReifiedMIL{}, fmt.Errorf("new ane draft model layer stack from modelir: %w", err)
	}

	baseErr := err
	attemptErrs := []string{fmt.Sprintf("base: %v", err)}
	baseOpts := opts
	baseOpts.TransformerConfig = cfg
	baseOpts.TransformerWeights = weights
	baseOpts.RequestedLayers = requested
	baseOpts.SelectedLayers = selected
	for _, fallback := range modelIRCompileFallbackProfiles(baseOpts, cfg) {
		tryCfg := fallback.Opts.TransformerConfig
		tryCfg.NumLayers = selected
		tryWeights := weights
		tryModel, compileErr := NewANEDraftModelFromMILTransformerLayerStackWithPlanConfig(
			tryCfg,
			tryWeights,
			hiddenDim,
			vocabSize,
			outputDim,
			embeddings,
			planCfg,
		)
		if compileErr == nil {
			reified.TransformerConfig = tryCfg
			reified.Diagnostics = append(reified.Diagnostics, ReifyDiagnostic{
				Code:    "compile_fallback",
				Message: fmt.Sprintf("in-memory compile fallback applied: %s", fallback.Key),
			})
			return tryModel, reified, nil
		}
		attemptErrs = append(attemptErrs, fmt.Sprintf("%s compile: %v", fallback.Label, compileErr))
	}
	return nil, ReifiedMIL{}, fmt.Errorf(
		"new ane draft model layer stack from modelir: %w (fallback attempts: %s)",
		baseErr,
		strings.Join(attemptErrs, "; "),
	)
}

func newANEDraftModelFromMILArtifacts(
	cfg anereify.MILTransformerConfig,
	milText string,
	files []modelWeightFile,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
	planCfg *MultiSurfaceEvalPlanConfig,
) (*ANEDraftModel, error) {
	if cfg.KVCacheState && cfg.AttentionMaskInput {
		return nil, fmt.Errorf("new ane draft model from mil: stateful attention mask input is not supported by draft runtime")
	}
	model, err := buildModelFromMILTextWithDescriptorFallback(
		"ane draft mil transformer",
		milText,
		files,
	)
	if err != nil {
		return nil, fmt.Errorf("new ane draft model from mil: compile model: %w", err)
	}
	out := &ANEDraftModel{
		inMemoryMIL: model,
		outputDim:   outputDim,
		hiddenDim:   hiddenDim,
		vocabSize:   vocabSize,
		embeddings:  append([]float32(nil), embeddings...),
		inBuf:       make([]float32, hiddenDim),
		outBuf:      make([]float32, outputDim),
		ropeAttnDim: transformerAttentionDim(cfg),
	}
	if cfg.DynamicRoPEInputs || cfg.KVCacheState {
		plan, err := newMILTransformerPlanWithConfig(model, hiddenDim, outputDim, out.ropeAttnDim, cfg.DynamicRoPEInputs, planCfg)
		if err != nil {
			out.Close()
			return nil, fmt.Errorf("new ane draft model from mil: init eval plan: %w", err)
		}
		out.applyMILTransformerPlan(plan)
		if cfg.KVCacheState {
			planConfig := normalizeMultiSurfaceEvalPlanConfig(planCfg)
			out.statefulMIL = &statefulMILRuntime{
				label:       "ane draft mil transformer",
				milText:     milText,
				files:       cloneModelWeightFiles(files),
				dynamicRoPE: cfg.DynamicRoPEInputs,
				planConfig:  planConfig,
			}
		}
	}
	return out, nil
}

func cloneModelWeightFiles(in []modelWeightFile) []modelWeightFile {
	out := make([]modelWeightFile, 0, len(in))
	for _, file := range in {
		out = append(out, modelWeightFile{
			Path: file.Path,
			Blob: append([]byte(nil), file.Blob...),
		})
	}
	return out
}

// NewANEDraftModelFromMILAttentionDecode builds a one-layer attention-only
// decode model with external KV cache surfaces and dynamic RoPE inputs.
func NewANEDraftModelFromMILAttentionDecode(
	cfg MILAttentionDecodeConfig,
	layer MILTransformerLayerWeights,
	hiddenDim,
	vocabSize int,
	embeddings []float32,
) (*ANEDraftModel, error) {
	cfg = anereify.NormalizeMILAttentionDecodeConfig(cfg)
	if err := anereify.ValidateMILAttentionDecodeConfig(cfg); err != nil {
		return nil, err
	}
	if hiddenDim <= 0 || vocabSize <= 0 {
		return nil, fmt.Errorf(
			"new ane draft model from mil attention decode: invalid dims hiddenDim=%d vocabSize=%d",
			hiddenDim,
			vocabSize,
		)
	}
	if cfg.Dim != hiddenDim {
		return nil, fmt.Errorf(
			"new ane draft model from mil attention decode: cfg dim=%d want hiddenDim=%d",
			cfg.Dim,
			hiddenDim,
		)
	}
	wantEmb := hiddenDim * vocabSize
	if len(embeddings) > 0 && len(embeddings) != wantEmb {
		return nil, fmt.Errorf(
			"new ane draft model from mil attention decode: embeddings len=%d want=%d (or 0)",
			len(embeddings),
			wantEmb,
		)
	}
	milText, err := anereify.AttentionDecodeMILText(cfg)
	if err != nil {
		return nil, fmt.Errorf("new ane draft model from mil attention decode: mil text: %w", err)
	}
	files, err := anereify.BuildMILAttentionDecodeWeightFiles(cfg, layer)
	if err != nil {
		return nil, fmt.Errorf("new ane draft model from mil attention decode: weight files: %w", err)
	}
	model, err := buildModelFromMILTextWithDescriptorFallback("ane draft mil attention decode", milText, files)
	if err != nil {
		return nil, fmt.Errorf("new ane draft model from mil attention decode: compile model: %w", err)
	}
	out := &ANEDraftModel{
		inMemoryMIL: model,
		outputDim:   hiddenDim,
		hiddenDim:   hiddenDim,
		vocabSize:   vocabSize,
		embeddings:  append([]float32(nil), embeddings...),
		inBuf:       make([]float32, hiddenDim),
		outBuf:      make([]float32, hiddenDim),
		ropeAttnDim: cfg.AttentionDim,
	}
	if err := out.initMILAttentionDecodePlan(cfg.NumHeads, cfg.HeadDim, cfg.MaxSeqLen); err != nil {
		out.Close()
		return nil, fmt.Errorf("new ane draft model from mil attention decode: init eval plan: %w", err)
	}
	return out, nil
}

// NewANEDraftModelFromMILTransformerLayerStack builds one strict MIL model per
// transformer layer and evaluates them sequentially. This avoids monolithic MIL
// compile failures on larger decode trunks while preserving full-layer behavior.
func NewANEDraftModelFromMILTransformerLayerStack(
	cfg MILTransformerConfig,
	weights MILTransformerWeights,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
) (*ANEDraftModel, error) {
	return NewANEDraftModelFromMILTransformerLayerStackWithPlanConfig(
		cfg,
		weights,
		hiddenDim,
		vocabSize,
		outputDim,
		embeddings,
		nil,
	)
}

// NewANEDraftModelFromMILTransformerLayerStack builds one strict MIL model per
// transformer layer and evaluates them sequentially. It uses the provided
// multi-surface plan configuration when non-nil.
func NewANEDraftModelFromMILTransformerLayerStackWithPlanConfig(
	cfg MILTransformerConfig,
	weights MILTransformerWeights,
	hiddenDim,
	vocabSize,
	outputDim int,
	embeddings []float32,
	planCfg *MultiSurfaceEvalPlanConfig,
) (*ANEDraftModel, error) {
	if cfg.NumLayers <= 0 {
		return nil, fmt.Errorf("new ane draft model from mil layer stack: invalid NumLayers=%d", cfg.NumLayers)
	}
	if outputDim != hiddenDim {
		return nil, fmt.Errorf(
			"new ane draft model from mil layer stack: outputDim=%d want hiddenDim=%d",
			outputDim,
			hiddenDim,
		)
	}
	if len(weights.Layers) != cfg.NumLayers {
		return nil, fmt.Errorf(
			"new ane draft model from mil layer stack: layer weights=%d want=%d",
			len(weights.Layers),
			cfg.NumLayers,
		)
	}
	wantEmb := hiddenDim * vocabSize
	if len(embeddings) > 0 && len(embeddings) != wantEmb {
		return nil, fmt.Errorf(
			"new ane draft model from mil layer stack: embeddings len=%d want=%d (or 0)",
			len(embeddings),
			wantEmb,
		)
	}

	baseLayerCfg := cfg
	baseLayerCfg.NumLayers = 1
	baseLayerCfg.IncludeLMHead = false

	client, err := openPreferredANEClient()
	if err != nil {
		return nil, fmt.Errorf("new ane draft model from mil layer stack: open client: %w", err)
	}

	layers := make([]draftMILLayerEval, 0, cfg.NumLayers)
	cleanupLayers := func() {
		for i := len(layers) - 1; i >= 0; i-- {
			closeDraftMILLayerEval(layers[i])
		}
	}
	for i := 0; i < cfg.NumLayers; i++ {
		layerCfg := baseLayerCfg
		if i < cfg.NumLayers-1 {
			layerCfg.DisableFinalNormOp = true
		}
		layerWeights := MILTransformerWeights{
			Layers: []MILTransformerLayerWeights{weights.Layers[i]},
		}
		milText, files, err := anereify.BuildMILTransformerArtifacts(
			layerCfg,
			layerWeights,
		)
		if err != nil {
			cleanupLayers()
			return nil, fmt.Errorf("new ane draft model from mil layer stack: layer %d artifacts: %w", i, err)
		}
		label := fmt.Sprintf("ane draft mil transformer layer %d", i)
		model, err := buildModelFromMILTextWithDescriptorFallback(label, milText, cloneModelWeightFiles(files))
		if err != nil {
			cleanupLayers()
			return nil, fmt.Errorf("new ane draft model from mil layer stack: layer %d compile: %w", i, err)
		}
		clientModel, clientErr := loadClientModelFromInMemory(client, model, defaultMILFastPathKey, defaultANEQoS)
		if clientErr == nil {
			plan, err := newMILTransformerPlanWithClientModelConfig(
				clientModel,
				hiddenDim,
				outputDim,
				transformerAttentionDim(layerCfg),
				layerCfg.DynamicRoPEInputs,
				planCfg,
			)
			if err == nil {
				_ = callObjCBoolWithNSError("ane draft mil layer unload", model.ID, "unloadWithQoS:error:", defaultANEQoS)
				layers = append(layers, draftMILLayerEval{
					clientModel: clientModel,
					plan:        plan,
				})
				continue
			}
			clientErr = fmt.Errorf("client map plan: %w", err)
			clientModel.Close()
		}
		plan, err := newMILTransformerPlanWithConfig(
			model,
			hiddenDim,
			outputDim,
			transformerAttentionDim(layerCfg),
			layerCfg.DynamicRoPEInputs,
			planCfg,
		)
		if err != nil {
			_ = callObjCBoolWithNSError("ane draft mil layer unload", model.ID, "unloadWithQoS:error:", defaultANEQoS)
			cleanupLayers()
			if clientErr != nil {
				return nil, fmt.Errorf(
					"new ane draft model from mil layer stack: layer %d map plan: client path: %v; in-memory path: %w",
					i,
					clientErr,
					err,
				)
			}
			return nil, fmt.Errorf("new ane draft model from mil layer stack: layer %d map plan: %w", i, err)
		}
		layers = append(layers, draftMILLayerEval{
			model: model,
			plan:  plan,
		})
	}

	return &ANEDraftModel{
		milLayerEval: layers,
		outputDim:    outputDim,
		hiddenDim:    hiddenDim,
		vocabSize:    vocabSize,
		embeddings:   append([]float32(nil), embeddings...),
		inBuf:        make([]float32, hiddenDim),
		outBuf:       make([]float32, outputDim),
		inputBytes:   uintptr(hiddenDim * 4),
		outputBytes:  uintptr(outputDim * 4),
		ropeAttnDim:  transformerAttentionDim(cfg),
	}, nil
}

// NewANEDraftModelFromConfig opens a draft model using dimensions parsed from
// configPath.
func NewANEDraftModelFromConfig(modelcPath, configPath string, embeddings []float32) (*ANEDraftModel, error) {
	hiddenDim, vocabSize, _, err := LoadDraftModelDimsFromConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("new ane draft model from config: %w", err)
	}
	return NewANEDraftModel(modelcPath, hiddenDim, vocabSize, embeddings)
}

// LoadDraftModelDimsFromConfig extracts hidden and vocab dimensions from a model
// config JSON.
func LoadDraftModelDimsFromConfig(configPath string) (hiddenDim int, vocabSize int, modelType string, err error) {
	b, err := os.ReadFile(configPath)
	if err != nil {
		return 0, 0, "", err
	}
	var root draftConfigRoot
	if err := json.Unmarshal(b, &root); err != nil {
		return 0, 0, "", err
	}
	modelType = root.ModelType
	if root.HiddenSize != nil && root.VocabSize != nil {
		return *root.HiddenSize, *root.VocabSize, modelType, nil
	}
	if len(root.TextConfig) > 0 {
		var txt draftTextConfig
		if err := json.Unmarshal(root.TextConfig, &txt); err != nil {
			return 0, 0, "", err
		}
		if txt.ModelType != "" {
			modelType = txt.ModelType
		}
		if txt.HiddenSize != nil && txt.VocabSize != nil {
			return *txt.HiddenSize, *txt.VocabSize, modelType, nil
		}
	}
	return 0, 0, modelType, fmt.Errorf("draft dims parse failed for %s", configPath)
}

// EvalToken runs one single-token draft evaluation.
func (m *ANEDraftModel) EvalToken(input []float32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model: model is nil")
	}
	if len(input) != m.hiddenDim {
		return nil, fmt.Errorf("ane draft model: input len=%d want=%d", len(input), m.hiddenDim)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, fmt.Errorf("ane draft model: model is closed")
	}
	return m.evalTokenLocked(input)
}

func (m *ANEDraftModel) evalTokenLocked(input []float32) ([]float32, error) {
	if len(m.milLayerEval) > 0 {
		curr := append([]float32(nil), input...)
		for i, layer := range m.milLayerEval {
			if layer.plan == nil || layer.plan.input == nil || layer.plan.output == nil || layer.plan.plan == nil {
				return nil, fmt.Errorf("ane draft model: layer %d eval resources are nil", i)
			}
			if err := layer.plan.input.Write(curr); err != nil {
				return nil, fmt.Errorf("ane draft model: layer %d write input: %w", i, err)
			}
			if layer.plan.posCos != nil || layer.plan.posSin != nil {
				if layer.plan.posCos == nil || layer.plan.posSin == nil {
					return nil, fmt.Errorf("ane draft model: layer %d dynamic rope surfaces are inconsistent", i)
				}
				if err := m.writeRoPEInputsLocked(layer.plan.posCos, layer.plan.posSin); err != nil {
					return nil, fmt.Errorf("ane draft model: layer %d write rope inputs: %w", i, err)
				}
			}
			if err := layer.plan.plan.Eval(context.Background()); err != nil {
				return nil, fmt.Errorf("ane draft model: layer %d eval: %w", i, err)
			}
			out, err := layer.plan.output.Read()
			if err != nil {
				return nil, fmt.Errorf("ane draft model: layer %d read output: %w", i, err)
			}
			curr = out
		}
		return curr, nil
	}
	if m.milMultiPlan != nil {
		if m.milInput == nil || m.milOutput == nil {
			return nil, fmt.Errorf("ane draft model: multi-input eval surfaces are nil")
		}
		if err := m.milInput.Write(input); err != nil {
			return nil, fmt.Errorf("ane draft model: write mil input: %w", err)
		}
		if m.milPosCos != nil || m.milPosSin != nil {
			if m.milPosCos == nil || m.milPosSin == nil {
				return nil, fmt.Errorf("ane draft model: inconsistent dynamic rope surfaces")
			}
			if err := m.writeCurrentRoPEInputsLocked(); err != nil {
				return nil, fmt.Errorf("ane draft model: write rope inputs: %w", err)
			}
		}
		if m.milAttnMask != nil {
			if err := m.writeCurrentAttentionMaskLocked(); err != nil {
				return nil, fmt.Errorf("ane draft model: write attention mask: %w", err)
			}
		}
		if err := m.milMultiPlan.Eval(context.Background()); err != nil {
			return nil, fmt.Errorf("ane draft model: mil multi-input eval: %w", err)
		}
		out, err := m.milOutput.Read()
		if err != nil {
			return nil, fmt.Errorf("ane draft model: read mil output: %w", err)
		}
		if m.milKCurr != nil || m.milVCurr != nil {
			if m.milKCurr == nil || m.milVCurr == nil {
				return nil, fmt.Errorf("ane draft model: inconsistent kv output surfaces")
			}
			if m.kvState == nil {
				return nil, fmt.Errorf("ane draft model: kv output available without kv state")
			}
			kRow, err := m.milKCurr.Read()
			if err != nil {
				return nil, fmt.Errorf("ane draft model: read current K: %w", err)
			}
			vRow, err := m.milVCurr.Read()
			if err != nil {
				return nil, fmt.Errorf("ane draft model: read current V: %w", err)
			}
			if err := m.kvState.WriteLayerKV(0, kRow, vRow); err != nil {
				return nil, fmt.Errorf("ane draft model: write current KV row: %w", err)
			}
		}
		return out, nil
	}
	if m.inMemoryMIL.ID != 0 {
		return evalModelSingleIO(context.Background(), m.inMemoryMIL, input, m.outputDim, "ane draft mil eval")
	}
	if m.client.ID != 0 && iobjID(m.clientModel) != 0 {
		return evalClientModelSingleIO(
			context.Background(),
			m.client,
			m.clientModel,
			m.clientOpts,
			input,
			m.outputDim,
			"ane draft client eval",
		)
	}
	return nil, fmt.Errorf("ane draft model: no executable model path is initialized")
}

// GenerateDraft generates k autoregressive draft tokens.
//
// inputEmbed is the starting token embedding for step 0.
func (m *ANEDraftModel) GenerateDraft(inputEmbed []float32, k int) ([]int, [][]float32, error) {
	tokenIDs, allLogits, _, err := m.GenerateDraftWithTiming(inputEmbed, k)
	return tokenIDs, allLogits, err
}

// EmbeddingForToken returns a copied embedding row for tokenID.
func (m *ANEDraftModel) EmbeddingForToken(tokenID int) ([]float32, error) {
	return m.embeddingForToken(tokenID)
}

// GenerateDraftFromToken looks up tokenID embedding, then runs k draft steps.
func (m *ANEDraftModel) GenerateDraftFromToken(tokenID int, k int) ([]int, [][]float32, error) {
	inputEmbed, err := m.embeddingForToken(tokenID)
	if err != nil {
		return nil, nil, err
	}
	return m.GenerateDraft(inputEmbed, k)
}

// GenerateDraftWithTiming generates k autoregressive draft tokens and records
// ANE versus CPU-side time split.
func (m *ANEDraftModel) GenerateDraftWithTiming(inputEmbed []float32, k int) ([]int, [][]float32, DraftGenerateTiming, error) {
	startTotal := time.Now()
	if m == nil {
		return nil, nil, DraftGenerateTiming{}, fmt.Errorf("ane draft model: model is nil")
	}
	if len(inputEmbed) != m.hiddenDim {
		return nil, nil, DraftGenerateTiming{}, fmt.Errorf("ane draft model: input embedding len=%d want=%d", len(inputEmbed), m.hiddenDim)
	}
	if k <= 0 {
		return nil, nil, DraftGenerateTiming{}, fmt.Errorf("ane draft model: k must be > 0 (got %d)", k)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, nil, DraftGenerateTiming{}, fmt.Errorf("ane draft model: model is closed")
	}

	curr := make([]float32, m.hiddenDim)
	copy(curr, inputEmbed)
	tokenIDs := make([]int, 0, k)
	allLogits := make([][]float32, 0, k)
	var timing DraftGenerateTiming

	for step := 0; step < k; step++ {
		startANE := time.Now()
		logits, err := m.evalTokenLocked(curr)
		if err != nil {
			return nil, nil, DraftGenerateTiming{}, fmt.Errorf("ane draft model: eval step=%d: %w", step, err)
		}
		timing.ANE += time.Since(startANE)

		startCPU := time.Now()
		nextID := argmaxFloat32(logits)
		tokenIDs = append(tokenIDs, nextID)
		allLogits = append(allLogits, logits)

		rowStart := nextID * m.hiddenDim
		rowEnd := rowStart + m.hiddenDim
		if rowEnd > len(m.embeddings) {
			return nil, nil, DraftGenerateTiming{}, fmt.Errorf("ane draft model: embedding bounds out of range for token=%d", nextID)
		}
		copy(curr, m.embeddings[rowStart:rowEnd])
		if err := m.advanceDecodePositionLocked(); err != nil {
			return nil, nil, DraftGenerateTiming{}, fmt.Errorf(
				"ane draft model: advance decode position step=%d: %w",
				step,
				err,
			)
		}
		timing.CPU += time.Since(startCPU)
	}
	timing.Total = time.Since(startTotal)
	return tokenIDs, allLogits, timing, nil
}

func argmaxFloat32(xs []float32) int {
	if len(xs) == 0 {
		return 0
	}
	bestIdx := 0
	best := xs[0]
	for i := 1; i < len(xs); i++ {
		if xs[i] > best {
			best = xs[i]
			bestIdx = i
		}
	}
	return bestIdx
}

// Close releases draft-model ANE resources.
func (m *ANEDraftModel) Close() {
	if m == nil {
		return
	}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	m.closed = true
	client := m.client
	clientModel := m.clientModel
	clientOpts := m.clientOpts
	milModel := m.inMemoryMIL
	layerEval := m.milLayerEval
	multiPlan := m.milMultiPlan
	milInput := m.milInput
	milPosCos := m.milPosCos
	milPosSin := m.milPosSin
	milAttnMask := m.milAttnMask
	milOutput := m.milOutput
	milState := m.milState
	milKCurr := m.milKCurr
	milVCurr := m.milVCurr
	m.client = appleneuralengine.ANEClient{}
	m.clientModel = nil
	m.clientOpts = foundation.NSMutableDictionary{}
	m.inMemoryMIL = appleneuralengine.ANEInMemoryModel{}
	m.milLayerEval = nil
	m.milMultiPlan = nil
	m.milInput = nil
	m.milPosCos = nil
	m.milPosSin = nil
	m.milAttnMask = nil
	m.milOutput = nil
	m.milState = nil
	m.milKCurr = nil
	m.milVCurr = nil
	m.embeddings = nil
	m.inBuf = nil
	m.outBuf = nil
	kvState := m.kvState
	m.kvState = nil
	m.decodePos = 0
	m.ropeCos = nil
	m.ropeSin = nil
	m.ropeHeadDim = 0
	m.ropeMaxSeq = 0
	m.mu.Unlock()

	if client.ID != 0 && iobjID(clientModel) != 0 {
		_, _ = client.UnloadModelOptionsQosError(clientModel, clientOpts, defaultANEQoS)
	}
	if milModel.ID != 0 {
		_ = callObjCBoolWithNSError(
			"ane draft model unload",
			milModel.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
	}
	if kvState != nil {
		kvState.Close()
	}
	if multiPlan != nil {
		multiPlan.Close()
	}
	if milOutput != nil {
		milOutput.Close()
	}
	for _, state := range milState {
		if state != nil {
			state.Close()
		}
	}
	if milVCurr != nil {
		milVCurr.Close()
	}
	if milKCurr != nil {
		milKCurr.Close()
	}
	if milPosSin != nil {
		milPosSin.Close()
	}
	if milPosCos != nil {
		milPosCos.Close()
	}
	if milAttnMask != nil {
		milAttnMask.Close()
	}
	if milInput != nil {
		milInput.Close()
	}
	for _, layer := range layerEval {
		closeDraftMILLayerEval(layer)
	}
}

// WaitEventPort returns the Metal->ANE wait-event port for the active
// multi-surface MIL plan. Zero means no wait-event graph is attached.
func (m *ANEDraftModel) WaitEventPort() uint32 {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return 0
	}
	return m.milMultiPlan.WaitEventPort()
}

// WaitEvent returns the Metal->ANE wait event for the active multi-surface
// MIL plan.
func (m *ANEDraftModel) WaitEvent() *SharedEvent {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return nil
	}
	return m.milMultiPlan.WaitEvent()
}

// NewWaitMetalSharedEvent imports the active wait event into Metal.
func (m *ANEDraftModel) NewWaitMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model wait metal event: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return nil, fmt.Errorf("ane draft model wait metal event: no multi-surface plan is active")
	}
	return m.milMultiPlan.NewWaitMetalSharedEvent(device)
}

// NewDefaultWaitMetalSharedEvent imports the active wait event into Metal
// using the default Metal device.
func (m *ANEDraftModel) NewDefaultWaitMetalSharedEvent() (*MetalSharedEvent, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model wait metal event: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return nil, fmt.Errorf("ane draft model wait metal event: no multi-surface plan is active")
	}
	return m.milMultiPlan.NewDefaultWaitMetalSharedEvent()
}

// WaitPort is a shorthand alias for WaitEventPort.
func (m *ANEDraftModel) WaitPort() uint32 {
	return m.WaitEventPort()
}

// SignalEventPort returns the ANE->Metal signal-event port for the active
// multi-surface MIL plan. Zero means no signal-event graph is attached.
func (m *ANEDraftModel) SignalEventPort() uint32 {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return 0
	}
	return m.milMultiPlan.SignalEventPort()
}

// SignalEvent returns the ANE->Metal signal event for the active
// multi-surface MIL plan.
func (m *ANEDraftModel) SignalEvent() *SharedEvent {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return nil
	}
	return m.milMultiPlan.SignalEvent()
}

// NewSignalMetalSharedEvent imports the active signal event into Metal.
func (m *ANEDraftModel) NewSignalMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model signal metal event: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return nil, fmt.Errorf("ane draft model signal metal event: no multi-surface plan is active")
	}
	return m.milMultiPlan.NewSignalMetalSharedEvent(device)
}

// NewDefaultSignalMetalSharedEvent imports the active signal event into Metal
// using the default Metal device.
func (m *ANEDraftModel) NewDefaultSignalMetalSharedEvent() (*MetalSharedEvent, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model signal metal event: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return nil, fmt.Errorf("ane draft model signal metal event: no multi-surface plan is active")
	}
	return m.milMultiPlan.NewDefaultSignalMetalSharedEvent()
}

// SignalPort is a shorthand alias for SignalEventPort.
func (m *ANEDraftModel) SignalPort() uint32 {
	return m.SignalEventPort()
}

// InputSurface returns the current MIL input IOSurface when the multi-surface
// decode path is active.
func (m *ANEDraftModel) InputSurface() *IOSurfaceFloat32 {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.milInput
}

// PosCosSurface returns the active dynamic-RoPE cos input IOSurface.
func (m *ANEDraftModel) PosCosSurface() *IOSurfaceFloat32 {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.milPosCos
}

// PosSinSurface returns the active dynamic-RoPE sin input IOSurface.
func (m *ANEDraftModel) PosSinSurface() *IOSurfaceFloat32 {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.milPosSin
}

// WaitValue returns the wait-event value attached to the active multi-surface
// plan. Zero means no wait-event graph is attached.
func (m *ANEDraftModel) WaitValue() uint64 {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return 0
	}
	return m.milMultiPlan.WaitValue()
}

// SignalValue returns the signal-event value attached to the active
// multi-surface plan. Zero means no signal-event graph is attached.
func (m *ANEDraftModel) SignalValue() uint64 {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return 0
	}
	return m.milMultiPlan.SignalValue()
}

// OutputSurface returns the current MIL output IOSurface when the multi-surface
// decode path is active.
func (m *ANEDraftModel) OutputSurface() *IOSurfaceFloat32 {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.milOutput
}

// NewOutputMetalBufferBinding creates a reusable no-copy Metal buffer binding
// over the active MIL output IOSurface.
func (m *ANEDraftModel) NewOutputMetalBufferBinding(device metal.MTLDevice) (*MetalBufferBinding, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model metal buffer binding: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milOutput == nil {
		return nil, fmt.Errorf("ane draft model metal buffer binding: output IOSurface unavailable")
	}
	return m.milOutput.NewMetalBufferBinding(device)
}

// NewDefaultOutputMetalBufferBinding creates a reusable no-copy Metal buffer
// binding over the active MIL output IOSurface using the default Metal device.
func (m *ANEDraftModel) NewDefaultOutputMetalBufferBinding() (*MetalBufferBinding, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model metal buffer binding: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milOutput == nil {
		return nil, fmt.Errorf("ane draft model metal buffer binding: output IOSurface unavailable")
	}
	return m.milOutput.NewDefaultMetalBufferBinding()
}

// SetWaitEventSignaledValue updates the active wait-event shared value.
func (m *ANEDraftModel) SetWaitEventSignaledValue(value uint64) error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return fmt.Errorf("ane draft model: no multi-surface plan is active")
	}
	return m.milMultiPlan.SetWaitEventSignaledValue(value)
}

// SetSignalEventSignaledValue updates the active signal-event shared value.
func (m *ANEDraftModel) SetSignalEventSignaledValue(value uint64) error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return fmt.Errorf("ane draft model: no multi-surface plan is active")
	}
	return m.milMultiPlan.SetSignalEventSignaledValue(value)
}

// EvalPreparedSurface evaluates the active multi-surface plan after the caller
// has already prepared the input and dynamic-RoPE surfaces.
func (m *ANEDraftModel) EvalPreparedSurface(ctx context.Context) error {
	if m == nil {
		return fmt.Errorf("ane draft model prepared eval: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.milMultiPlan == nil {
		return fmt.Errorf("ane draft model prepared eval: no multi-surface plan is active")
	}
	return m.milMultiPlan.Eval(ctx)
}

// Reset clears decode state.
func (m *ANEDraftModel) Reset() error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("ane draft model: model is closed")
	}
	if m.inMemoryMIL.ID != 0 || len(m.milLayerEval) > 0 || m.milMultiPlan != nil || (m.client.ID != 0 && iobjID(m.clientModel) != 0) {
		if m.statefulMIL != nil {
			if err := m.resetStatefulMILLocked(); err != nil {
				return fmt.Errorf("ane draft model: reset stateful mil runtime: %w", err)
			}
		}
		m.decodePos = 0
		if m.kvState != nil {
			if err := m.kvState.Reset(); err != nil {
				return fmt.Errorf("ane draft model: reset kv state: %w", err)
			}
		}
		return nil
	}
	return fmt.Errorf("ane draft model: no executable model path is initialized")
}

// SnapshotDecodeState captures the current decode position and stateful MIL
// state surfaces for later RestoreDecodeState.
func (m *ANEDraftModel) SnapshotDecodeState() (*DecodeStateSnapshot, error) {
	if m == nil {
		return nil, fmt.Errorf("ane draft model: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, fmt.Errorf("ane draft model: model is closed")
	}
	if m.statefulMIL == nil || len(m.milState) == 0 {
		return nil, fmt.Errorf("ane draft model: decode snapshot unavailable for this runtime")
	}
	snap := &DecodeStateSnapshot{
		decodePos: m.decodePos,
		milState:  make([][]float32, len(m.milState)),
	}
	for i, surf := range m.milState {
		if surf == nil {
			return nil, fmt.Errorf("ane draft model: state surface %d is nil", i)
		}
		vals, err := surf.Read()
		if err != nil {
			return nil, fmt.Errorf("ane draft model: read state surface %d: %w", i, err)
		}
		snap.milState[i] = vals
	}
	return snap, nil
}

// RestoreDecodeState restores a snapshot captured by SnapshotDecodeState.
func (m *ANEDraftModel) RestoreDecodeState(snap *DecodeStateSnapshot) error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	if snap == nil {
		return fmt.Errorf("ane draft model: decode snapshot is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("ane draft model: model is closed")
	}
	if m.statefulMIL == nil || len(m.milState) == 0 {
		return fmt.Errorf("ane draft model: decode restore unavailable for this runtime")
	}
	if len(snap.milState) != len(m.milState) {
		return fmt.Errorf(
			"ane draft model: decode snapshot state count=%d want=%d",
			len(snap.milState),
			len(m.milState),
		)
	}
	for i, surf := range m.milState {
		if surf == nil {
			return fmt.Errorf("ane draft model: state surface %d is nil", i)
		}
		if surf.Count() != len(snap.milState[i]) {
			return fmt.Errorf(
				"ane draft model: decode snapshot surface %d len=%d want=%d",
				i,
				len(snap.milState[i]),
				surf.Count(),
			)
		}
		if err := surf.Write(snap.milState[i]); err != nil {
			return fmt.Errorf("ane draft model: restore state surface %d: %w", i, err)
		}
	}
	m.decodePos = snap.decodePos
	return nil
}

// RestoreStatefulMILState restores stateful MIL surfaces from raw state slices
// and rewinds decode position to decodePos.
func (m *ANEDraftModel) RestoreStatefulMILState(decodePos int, milState [][]float32) error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("ane draft model: model is closed")
	}
	if m.statefulMIL == nil || len(m.milState) == 0 {
		return fmt.Errorf("ane draft model: stateful mil restore unavailable for this runtime")
	}
	if len(milState) != len(m.milState) {
		return fmt.Errorf(
			"ane draft model: raw state count=%d want=%d",
			len(milState),
			len(m.milState),
		)
	}
	for i, surf := range m.milState {
		if surf == nil {
			return fmt.Errorf("ane draft model: state surface %d is nil", i)
		}
		if surf.Count() != len(milState[i]) {
			return fmt.Errorf(
				"ane draft model: raw state surface %d len=%d want=%d",
				i,
				len(milState[i]),
				surf.Count(),
			)
		}
		if err := surf.Write(milState[i]); err != nil {
			return fmt.Errorf("ane draft model: restore raw state surface %d: %w", i, err)
		}
	}
	if decodePos < 0 {
		decodePos = 0
	}
	m.decodePos = decodePos
	return nil
}

func (m *ANEDraftModel) resetStatefulMILLocked() error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	spec := m.statefulMIL
	if spec == nil {
		return nil
	}
	model, err := buildModelFromMILTextWithDescriptorFallback(spec.label, spec.milText, cloneModelWeightFiles(spec.files))
	if err != nil {
		return err
	}
	plan, err := newMILTransformerPlanWithConfig(model, m.hiddenDim, m.outputDim, m.ropeAttnDim, spec.dynamicRoPE, &spec.planConfig)
	if err != nil {
		_ = callObjCBoolWithNSError(spec.label+" reset unload", model.ID, "unloadWithQoS:error:", defaultANEQoS)
		return err
	}
	prevModel := m.inMemoryMIL
	m.inMemoryMIL = model
	m.applyMILTransformerPlan(plan)
	if prevModel.ID != 0 {
		_ = callObjCBoolWithNSError(spec.label+" reset unload", prevModel.ID, "unloadWithQoS:error:", defaultANEQoS)
	}
	return nil
}

func openDraftClientModel(
	modelcPath string,
	modelKey string,
	qos uint32,
) (appleneuralengine.ANEClient, objectivec.IObject, foundation.NSMutableDictionary, error) {
	client, err := openPreferredANEClient()
	if err != nil {
		return appleneuralengine.ANEClient{}, nil, foundation.NSMutableDictionary{}, err
	}
	modelURL := foundation.NewURLFileURLWithPath(modelcPath)
	modelObj := appleneuralengine.GetANEModelClass().ModelAtURLKey(
		modelURL,
		foundation.NewStringWithString(modelKey),
	)
	if modelObj.GetID() == 0 {
		return appleneuralengine.ANEClient{}, nil, foundation.NSMutableDictionary{}, fmt.Errorf("_ANEModel modelAtURL:key: returned nil")
	}
	opts := foundation.NewMutableDictionaryWithCapacity(0)
	if _, err := client.CompileModelOptionsQosError(modelObj, opts, qos); err != nil {
		return appleneuralengine.ANEClient{}, nil, foundation.NSMutableDictionary{}, fmt.Errorf("compile model: %w", err)
	}
	if _, err := client.LoadModelOptionsQosError(modelObj, opts, qos); err != nil {
		time.Sleep(100 * time.Millisecond)
		if _, retryErr := client.LoadModelOptionsQosError(modelObj, opts, qos); retryErr != nil {
			return appleneuralengine.ANEClient{}, nil, foundation.NSMutableDictionary{}, fmt.Errorf("load model: %w", retryErr)
		}
	}
	return client, modelObj, opts, nil
}

func loadClientModelFromInMemory(
	client appleneuralengine.ANEClient,
	model appleneuralengine.ANEInMemoryModel,
	key string,
	qos uint32,
) (*ANEClientMILModel, error) {
	if client.ID == 0 {
		return nil, fmt.Errorf("load client model from in-memory: client is nil")
	}
	if model.ID == 0 {
		return nil, fmt.Errorf("load client model from in-memory: model is nil")
	}
	modelPath := inMemoryModelPath(model)
	if strings.TrimSpace(modelPath) == "" {
		return nil, fmt.Errorf("load client model from in-memory: local model path is empty")
	}
	if loaded, err := loadModelDirectoryWithoutCompile(client, modelPath, key, qos); err == nil {
		return loaded, nil
	}
	loaded, err := compileAndLoadModelDirectory(client, modelPath, key, qos)
	if err != nil {
		return nil, fmt.Errorf("load client model from in-memory: %w", err)
	}
	return loaded, nil
}

func inMemoryModelPath(model appleneuralengine.ANEInMemoryModel) string {
	if model.ID == 0 {
		return ""
	}
	if path := objectiveCPath(model.LocalModelPath()); path != "" {
		return path
	}
	return objectiveCPath(model.SaveModelFiles())
}

func objectiveCPath(obj objectivec.IObject) string {
	if obj == nil || obj.GetID() == 0 {
		return ""
	}
	if objc.Send[bool](obj.GetID(), objc.Sel("respondsToSelector:"), objc.Sel("path")) {
		pathID := objc.Send[objc.ID](obj.GetID(), objc.Sel("path"))
		if pathID != 0 {
			return foundation.NSStringFromID(pathID).String()
		}
	}
	if objc.Send[bool](obj.GetID(), objc.Sel("respondsToSelector:"), objc.Sel("string")) {
		strID := objc.Send[objc.ID](obj.GetID(), objc.Sel("string"))
		if strID != 0 {
			return foundation.NSStringFromID(strID).String()
		}
	}
	return ""
}

func openPreferredANEClient() (appleneuralengine.ANEClient, error) {
	cls := appleneuralengine.GetANEClientClass()
	if shared := cls.SharedConnection(); shared.GetID() != 0 {
		return *shared, nil
	}
	if private := cls.SharedPrivateConnection(); private.GetID() != 0 {
		return *private, nil
	}
	client := cls.Alloc()
	if client.GetID() != 0 {
		client = client.InitWithRestrictedAccessAllowed(true)
		if client.GetID() != 0 {
			return client, nil
		}
	}
	client = appleneuralengine.NewANEClientWithRestrictedAccessAllowed(true)
	if client.GetID() != 0 {
		return client, nil
	}
	client = appleneuralengine.NewANEClient()
	if client.GetID() != 0 {
		return client, nil
	}
	return appleneuralengine.ANEClient{}, fmt.Errorf("shared _ANEClient unavailable")
}

func evalClientModelSingleIO(
	ctx context.Context,
	client appleneuralengine.ANEClient,
	model objectivec.IObject,
	options objectivec.IObject,
	input []float32,
	outputCount int,
	label string,
) ([]float32, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if outputCount <= 0 {
		return nil, fmt.Errorf("%s: invalid outputCount=%d", label, outputCount)
	}

	inSurf, err := newFloatSurface(len(input))
	if err != nil {
		return nil, err
	}
	defer releaseIOSurface(inSurf)
	if err := writeFloat32IOSurface(inSurf, input); err != nil {
		return nil, err
	}

	outSurf, err := newFloatSurface(outputCount)
	if err != nil {
		return nil, err
	}
	defer releaseIOSurface(outSurf)

	inObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(inSurf)
	outObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(outSurf)
	if inObj.GetID() == 0 || outObj.GetID() == 0 {
		return nil, fmt.Errorf("%s: create IOSurface object failed", label)
	}
	inputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{inObj}))
	inputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	outputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{outObj}))
	outputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	reqObj := appleneuralengine.GetANERequestClass().RequestWithInputsInputIndicesOutputsOutputIndicesProcedureIndex(
		inputs,
		inputIndices,
		outputs,
		outputIndices,
		foundation.NewNumberWithInt(0),
	)
	if reqObj.GetID() == 0 {
		return nil, fmt.Errorf("%s: create request failed", label)
	}
	req := appleneuralengine.ANERequestFromID(reqObj.GetID())
	if !req.Validate() {
		return nil, fmt.Errorf("%s: request validation failed", label)
	}

	if _, err := client.MapIOSurfacesWithModelRequestCacheInferenceError(model, reqObj, true); err != nil {
		return nil, fmt.Errorf("%s: map IOSurfaces: %w", label, err)
	}
	defer client.UnmapIOSurfacesWithModelRequest(model, reqObj)

	if objc.Send[bool](client.ID, objc.Sel("respondsToSelector:"), objc.Sel("doEvaluateDirectWithModel:options:request:qos:error:")) {
		if _, err := client.DoEvaluateDirectWithModelOptionsRequestQosError(model, options, reqObj, defaultANEQoS); err != nil {
			return nil, fmt.Errorf("%s: direct evaluate: %w", label, err)
		}
	} else {
		if _, err := client.EvaluateWithModelOptionsRequestQosError(model, options, reqObj, defaultANEQoS); err != nil {
			return nil, fmt.Errorf("%s: evaluate: %w", label, err)
		}
	}
	return readFloat32IOSurface(outSurf, outputCount)
}

// ConfigureKVState allocates persistent KV IOSurfaces for decode loopback.
func (m *ANEDraftModel) ConfigureKVState(numLayers, numHeads, headDim, maxSeqLen int) error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	// Attention-decode plans bind specific KV IOSurfaces into a mapped request.
	// Replacing kvState later can desynchronize writes from mapped inputs.
	if m.milMultiPlan != nil && (m.milKCurr != nil || m.milVCurr != nil) {
		return fmt.Errorf("ane draft model: kv state is bound by attention decode eval plan and cannot be replaced")
	}
	state, err := NewDraftKVSurfaceState(numLayers, numHeads, headDim, maxSeqLen)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		state.Close()
		return fmt.Errorf("ane draft model: model is closed")
	}
	if m.kvState != nil {
		m.kvState.Close()
	}
	m.kvState = state
	return nil
}

// KVState returns the configured KV surface state, if any.
func (m *ANEDraftModel) KVState() *DraftKVSurfaceState {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.kvState
}

// SetRoPETables stores flattened [maxSeqLen*headDim] cos/sin tables used for
// host-side per-step position slicing.
func (m *ANEDraftModel) SetRoPETables(cosTable, sinTable []float32, headDim, maxSeqLen int) error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	_, _, err := SliceRoPERow(cosTable, sinTable, headDim, maxSeqLen, 0)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("ane draft model: model is closed")
	}
	m.ropeCos = append([]float32(nil), cosTable...)
	m.ropeSin = append([]float32(nil), sinTable...)
	m.ropeHeadDim = headDim
	m.ropeMaxSeq = maxSeqLen
	return nil
}

// CurrentRoPESlice returns the cos/sin row for the current KV decode position.
func (m *ANEDraftModel) CurrentRoPESlice() ([]float32, []float32, error) {
	if m == nil {
		return nil, nil, fmt.Errorf("ane draft model: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.ropeCos) == 0 || len(m.ropeSin) == 0 || m.ropeHeadDim <= 0 || m.ropeMaxSeq <= 0 {
		return nil, nil, fmt.Errorf("ane draft model: rope tables are not configured")
	}
	pos := m.decodePos
	if pos >= m.ropeMaxSeq {
		pos = m.ropeMaxSeq - 1
	}
	return SliceRoPERow(m.ropeCos, m.ropeSin, m.ropeHeadDim, m.ropeMaxSeq, pos)
}

func (m *ANEDraftModel) writeCurrentRoPEInputsLocked() error {
	return m.writeRoPEInputsLocked(m.milPosCos, m.milPosSin)
}

func (m *ANEDraftModel) writeRoPEInputsLocked(cosSurf, sinSurf *IOSurfaceFloat32) error {
	if cosSurf == nil || sinSurf == nil {
		return fmt.Errorf("ane draft model: dynamic rope surfaces are nil")
	}
	cosRow, sinRow, err := m.currentRoPESliceLocked()
	if err != nil {
		return err
	}
	cosExp, sinExp, err := m.expandRoPERowForAttentionLocked(cosRow, sinRow)
	if err != nil {
		return err
	}
	if err := cosSurf.Write(cosExp); err != nil {
		return fmt.Errorf("write pos_cos: %w", err)
	}
	if err := sinSurf.Write(sinExp); err != nil {
		return fmt.Errorf("write pos_sin: %w", err)
	}
	return nil
}

func (m *ANEDraftModel) writeCurrentAttentionMaskLocked() error {
	if m.milAttnMask == nil {
		return fmt.Errorf("ane draft model: attention mask surface is nil")
	}
	if m.kvState == nil {
		return fmt.Errorf("ane draft model: kv state is nil for attention mask")
	}
	if m.ropeAttnDim <= 0 || m.ropeHeadDim <= 0 || m.ropeAttnDim%m.ropeHeadDim != 0 {
		return fmt.Errorf(
			"ane draft model: invalid attention dims for mask attention=%d head=%d",
			m.ropeAttnDim,
			m.ropeHeadDim,
		)
	}
	numHeads := m.ropeAttnDim / m.ropeHeadDim
	maxSeq := m.kvState.MaxSeqLen()
	if maxSeq <= 0 {
		return fmt.Errorf("ane draft model: invalid kv max seq=%d", maxSeq)
	}
	maskLenPerHead := maxSeq + 1 // +1 appended current-token slot in MIL graph
	total := numHeads * maskLenPerHead
	if len(m.attnMaskTmp) != total {
		m.attnMaskTmp = make([]float32, total)
	}
	const negInf = float32(-1e4)
	for i := range m.attnMaskTmp {
		m.attnMaskTmp[i] = negInf
	}
	pos := m.decodePos
	if pos < 0 {
		pos = 0
	}
	if pos > maxSeq {
		pos = maxSeq
	}
	for h := 0; h < numHeads; h++ {
		base := h * maskLenPerHead
		for t := 0; t < pos && t < maxSeq; t++ {
			m.attnMaskTmp[base+t] = 0
		}
		// The appended current-token K/V is always enabled.
		m.attnMaskTmp[base+maxSeq] = 0
	}
	if err := m.milAttnMask.Write(m.attnMaskTmp); err != nil {
		return fmt.Errorf("write attention mask surface: %w", err)
	}
	return nil
}

func (m *ANEDraftModel) currentRoPESliceLocked() ([]float32, []float32, error) {
	if len(m.ropeCos) == 0 || len(m.ropeSin) == 0 || m.ropeHeadDim <= 0 || m.ropeMaxSeq <= 0 {
		return nil, nil, fmt.Errorf("ane draft model: rope tables are not configured")
	}
	pos := m.decodePos
	if pos >= m.ropeMaxSeq {
		pos = m.ropeMaxSeq - 1
	}
	return SliceRoPERow(m.ropeCos, m.ropeSin, m.ropeHeadDim, m.ropeMaxSeq, pos)
}

func (m *ANEDraftModel) expandRoPERowForAttentionLocked(cosRow, sinRow []float32) ([]float32, []float32, error) {
	if m.ropeAttnDim <= 0 || m.ropeHeadDim <= 0 {
		return nil, nil, fmt.Errorf(
			"ane draft model: invalid rope dims attention=%d head=%d",
			m.ropeAttnDim,
			m.ropeHeadDim,
		)
	}
	if len(cosRow) != m.ropeHeadDim || len(sinRow) != m.ropeHeadDim {
		return nil, nil, fmt.Errorf(
			"ane draft model: rope row len mismatch cos=%d sin=%d want=%d",
			len(cosRow),
			len(sinRow),
			m.ropeHeadDim,
		)
	}
	if m.ropeAttnDim%m.ropeHeadDim != 0 {
		return nil, nil, fmt.Errorf(
			"ane draft model: attention dim=%d not divisible by head dim=%d",
			m.ropeAttnDim,
			m.ropeHeadDim,
		)
	}
	if len(m.ropeTmpCos) != m.ropeAttnDim {
		m.ropeTmpCos = make([]float32, m.ropeAttnDim)
	}
	if len(m.ropeTmpSin) != m.ropeAttnDim {
		m.ropeTmpSin = make([]float32, m.ropeAttnDim)
	}
	for off := 0; off < m.ropeAttnDim; off += m.ropeHeadDim {
		copy(m.ropeTmpCos[off:off+m.ropeHeadDim], cosRow)
		copy(m.ropeTmpSin[off:off+m.ropeHeadDim], sinRow)
	}
	return m.ropeTmpCos, m.ropeTmpSin, nil
}

func normalizeMultiSurfaceEvalPlanConfig(planCfg *MultiSurfaceEvalPlanConfig) MultiSurfaceEvalPlanConfig {
	cfg := DefaultMultiSurfaceEvalPlanConfig()
	if planCfg != nil {
		cfg = *planCfg
	} else {
		applySharedEventModeFromEnv(&cfg)
	}
	if cfg.EnableMetalWait && cfg.WaitValue == 0 {
		cfg.WaitValue = 1
	}
	if cfg.EnableMetalSignal && cfg.SignalValue == 0 {
		cfg.SignalValue = 1
	}
	return cfg
}

func newMILTransformerPlanWithClientModel(
	model *ANEClientMILModel,
	hiddenDim int,
	outputDim int,
	ropeAttnDim int,
	dynamicRoPE bool,
) (*milTransformerPlan, error) {
	return newMILTransformerPlanWithClientModelConfig(
		model,
		hiddenDim,
		outputDim,
		ropeAttnDim,
		dynamicRoPE,
		nil,
	)
}

func newMILTransformerPlanWithClientModelConfig(
	model *ANEClientMILModel,
	hiddenDim int,
	outputDim int,
	ropeAttnDim int,
	dynamicRoPE bool,
	planCfg *MultiSurfaceEvalPlanConfig,
) (*milTransformerPlan, error) {
	if model == nil || model.model.GetID() == 0 {
		return nil, fmt.Errorf("ane draft model: client MIL model is nil")
	}
	return newMILTransformerPlanForModel(
		model.model,
		func(inputs []SurfaceBinding, outputs []SurfaceBinding, cfg MultiSurfaceEvalPlanConfig) (*MultiSurfaceEvalPlan, error) {
			return NewMultiSurfaceEvalPlanWithClientModel(model, inputs, outputs, cfg)
		},
		hiddenDim,
		outputDim,
		ropeAttnDim,
		dynamicRoPE,
		planCfg,
	)
}

func newMILTransformerPlan(
	model appleneuralengine.ANEInMemoryModel,
	hiddenDim int,
	outputDim int,
	ropeAttnDim int,
	dynamicRoPE bool,
) (*milTransformerPlan, error) {
	return newMILTransformerPlanWithConfig(model, hiddenDim, outputDim, ropeAttnDim, dynamicRoPE, nil)
}

func newMILTransformerPlanWithConfig(
	model appleneuralengine.ANEInMemoryModel,
	hiddenDim int,
	outputDim int,
	ropeAttnDim int,
	dynamicRoPE bool,
	planCfg *MultiSurfaceEvalPlanConfig,
) (*milTransformerPlan, error) {
	if model.ID == 0 {
		return nil, fmt.Errorf("ane draft model: in-memory MIL model is nil")
	}
	return newMILTransformerPlanForModel(
		model.Model(),
		func(inputs []SurfaceBinding, outputs []SurfaceBinding, cfg MultiSurfaceEvalPlanConfig) (*MultiSurfaceEvalPlan, error) {
			return NewMultiSurfaceEvalPlan(model, inputs, outputs, cfg)
		},
		hiddenDim,
		outputDim,
		ropeAttnDim,
		dynamicRoPE,
		planCfg,
	)
}

func newMILTransformerPlanForModel(
	base objectivec.IObject,
	buildPlan func([]SurfaceBinding, []SurfaceBinding, MultiSurfaceEvalPlanConfig) (*MultiSurfaceEvalPlan, error),
	hiddenDim int,
	outputDim int,
	ropeAttnDim int,
	dynamicRoPE bool,
	planCfg *MultiSurfaceEvalPlanConfig,
) (*milTransformerPlan, error) {
	trace := func(phase string, args ...any) {
		if !envTruthy("MLXGO_ANE_DEBUG_DRAFT_PLAN") {
			return
		}
		slog.Info("ANE draft mil plan phase", append([]any{"phase", phase}, args...)...)
	}
	if base == nil || base.GetID() == 0 {
		return nil, fmt.Errorf("ane draft model: compiled MIL model is nil")
	}
	if hiddenDim <= 0 || outputDim <= 0 {
		return nil, fmt.Errorf(
			"ane draft model: invalid mil plan dims hidden=%d output=%d",
			hiddenDim,
			outputDim,
		)
	}
	if dynamicRoPE && ropeAttnDim <= 0 {
		return nil, fmt.Errorf("ane draft model: invalid dynamic rope attention dim=%d", ropeAttnDim)
	}
	var inputLayouts []compiledTensorLayout
	var outputLayouts []compiledTensorLayout
	var stateLayouts []compiledTensorLayout
	if schema, layoutErr := parseCompiledModelSchema(base); layoutErr == nil {
		inputLayouts = schema.Inputs
		outputLayouts = schema.Outputs
		stateLayouts = schema.States
	}
	newInputSurface := func(logicalCount int, names ...string) (*IOSurfaceFloat32, error) {
		if len(inputLayouts) > 0 {
			index, err := compiledLayoutIndexForNames(inputLayouts, names...)
			if err == nil {
				return newIOSurfaceFloat32WithLayout(inputLayouts[index])
			}
		}
		return NewIOSurfaceFloat32(logicalCount)
	}
	newOutputSurface := func(logicalCount int, names ...string) (*IOSurfaceFloat32, error) {
		if len(outputLayouts) > 0 {
			index, err := compiledLayoutIndexForNames(outputLayouts, names...)
			if err == nil {
				return newIOSurfaceFloat32WithLayout(outputLayouts[index])
			}
		}
		return NewIOSurfaceFloat32(logicalCount)
	}

	inSurf, err := newInputSurface(hiddenDim, "x_in", "x")
	if err != nil {
		return nil, fmt.Errorf("allocate mil input surface: %w", err)
	}
	var cosSurf *IOSurfaceFloat32
	var sinSurf *IOSurfaceFloat32
	var stateSurfs []*IOSurfaceFloat32
	cleanup := func(outSurf *IOSurfaceFloat32) {
		if outSurf != nil {
			outSurf.Close()
		}
		for _, stateSurf := range stateSurfs {
			if stateSurf != nil {
				stateSurf.Close()
			}
		}
		if sinSurf != nil {
			sinSurf.Close()
		}
		if cosSurf != nil {
			cosSurf.Close()
		}
		inSurf.Close()
	}
	if dynamicRoPE {
		cosSurf, err = newInputSurface(ropeAttnDim, "pos_cos_in", "pos_cos")
		if err != nil {
			inSurf.Close()
			return nil, fmt.Errorf("allocate mil pos_cos surface: %w", err)
		}
		sinSurf, err = newInputSurface(ropeAttnDim, "pos_sin_in", "pos_sin")
		if err != nil {
			cosSurf.Close()
			inSurf.Close()
			return nil, fmt.Errorf("allocate mil pos_sin surface: %w", err)
		}
	}
	outSurf, err := newOutputSurface(outputDim, "y", "y@output")
	if err != nil {
		cleanup(nil)
		return nil, fmt.Errorf("allocate mil output surface: %w", err)
	}
	if err := inSurf.Write(make([]float32, hiddenDim)); err != nil {
		cleanup(outSurf)
		return nil, fmt.Errorf("init mil input surface: %w", err)
	}
	if dynamicRoPE {
		ones := make([]float32, ropeAttnDim)
		for i := range ones {
			ones[i] = 1
		}
		if err := cosSurf.Write(ones); err != nil {
			cleanup(outSurf)
			return nil, fmt.Errorf("init mil pos_cos surface: %w", err)
		}
		if err := sinSurf.Write(make([]float32, ropeAttnDim)); err != nil {
			cleanup(outSurf)
			return nil, fmt.Errorf("init mil pos_sin surface: %w", err)
		}
	}
	if err := outSurf.Write(make([]float32, outputDim)); err != nil {
		cleanup(outSurf)
		return nil, fmt.Errorf("init mil output surface: %w", err)
	}
	if len(stateLayouts) > 0 {
		stateSurfs = make([]*IOSurfaceFloat32, 0, len(stateLayouts))
		for _, layout := range stateLayouts {
			stateSurf, stateErr := newIOSurfaceFloat32WithLayout(layout)
			if stateErr != nil {
				cleanup(outSurf)
				return nil, fmt.Errorf("allocate mil state surface %q: %w", layout.Name, stateErr)
			}
			if stateErr := stateSurf.Write(make([]float32, stateSurf.Count())); stateErr != nil {
				stateSurf.Close()
				cleanup(outSurf)
				return nil, fmt.Errorf("init mil state surface %q: %w", layout.Name, stateErr)
			}
			stateSurfs = append(stateSurfs, stateSurf)
		}
	}

	trace("resolve_symbol_indices_begin")
	inSyms, outSyms, usedCompiledLayout := milTransformerProcedureSymbolIndices(base, dynamicRoPE)
	trace("resolve_symbol_indices_done", "inputs", inSyms, "outputs", outSyms)
	wantInputs := 1
	if dynamicRoPE {
		wantInputs = 3
	}
	if len(inSyms) < wantInputs || len(outSyms) < 1 {
		cleanup(outSurf)
		return nil, fmt.Errorf(
			"resolve mil transformer symbol indices: inputs=%v outputs=%v (want at least %d inputs, 1 output)",
			inSyms,
			outSyms,
			wantInputs,
		)
	}

	inputs := []SurfaceBinding{{Surface: inSurf, SymbolIndex: inSyms[0]}}
	if dynamicRoPE {
		inputs = append(
			inputs,
			SurfaceBinding{Surface: cosSurf, SymbolIndex: inSyms[1]},
			SurfaceBinding{Surface: sinSurf, SymbolIndex: inSyms[2]},
		)
	}
	for i, layout := range stateLayouts {
		if i >= len(stateSurfs) {
			break
		}
		symbolIndex := layout.SymbolIndex
		if symbolIndex < 0 {
			cleanup(outSurf)
			return nil, fmt.Errorf("resolve mil transformer state symbol index for %q", layout.Name)
		}
		inputs = append(inputs, SurfaceBinding{Surface: stateSurfs[i], SymbolIndex: symbolIndex})
	}
	cfg := normalizeMultiSurfaceEvalPlanConfig(planCfg)
	trace(
		"new_multi_surface_plan_begin",
		"dynamic_rope",
		dynamicRoPE,
		"disable_cache_mapping",
		cfg.DisableCacheMapping,
		"compiled_layout",
		usedCompiledLayout,
	)
	plan, err := buildPlan(inputs, []SurfaceBinding{{Surface: outSurf, SymbolIndex: outSyms[0]}}, cfg)
	if err != nil && !cfg.DisableCacheMapping {
		retry := cfg
		retry.DisableCacheMapping = true
		trace("new_multi_surface_plan_retry", "dynamic_rope", dynamicRoPE, "disable_cache_mapping", retry.DisableCacheMapping)
		plan, err = buildPlan(inputs, []SurfaceBinding{{Surface: outSurf, SymbolIndex: outSyms[0]}}, retry)
	}
	if err != nil {
		cleanup(outSurf)
		return nil, err
	}
	trace("new_multi_surface_plan_done", "dynamic_rope", dynamicRoPE)
	return &milTransformerPlan{
		input:  inSurf,
		posCos: cosSurf,
		posSin: sinSurf,
		output: outSurf,
		states: stateSurfs,
		plan:   plan,
	}, nil
}

func closeDraftMILLayerEval(layer draftMILLayerEval) {
	if layer.plan != nil {
		if layer.plan.plan != nil {
			layer.plan.plan.Close()
		}
		if layer.plan.output != nil {
			layer.plan.output.Close()
		}
		for _, state := range layer.plan.states {
			if state != nil {
				state.Close()
			}
		}
		if layer.plan.posSin != nil {
			layer.plan.posSin.Close()
		}
		if layer.plan.posCos != nil {
			layer.plan.posCos.Close()
		}
		if layer.plan.input != nil {
			layer.plan.input.Close()
		}
	}
	if layer.clientModel != nil {
		layer.clientModel.Close()
		return
	}
	if layer.model.ID != 0 {
		_ = callObjCBoolWithNSError(
			"ane draft model layer unload",
			layer.model.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
	}
}

func (m *ANEDraftModel) applyMILTransformerPlan(plan *milTransformerPlan) {
	prevInput := m.milInput
	prevPosCos := m.milPosCos
	prevPosSin := m.milPosSin
	prevOutput := m.milOutput
	prevState := m.milState
	prevPlan := m.milMultiPlan
	m.milInput = nil
	m.milPosCos = nil
	m.milPosSin = nil
	m.milAttnMask = nil
	m.milOutput = nil
	m.milState = nil
	m.milKCurr = nil
	m.milVCurr = nil
	m.milMultiPlan = nil
	if plan != nil {
		m.milInput = plan.input
		m.milPosCos = plan.posCos
		m.milPosSin = plan.posSin
		m.milOutput = plan.output
		m.milState = append([]*IOSurfaceFloat32(nil), plan.states...)
		m.milMultiPlan = plan.plan
	}
	if prevPlan != nil {
		prevPlan.Close()
	}
	if prevOutput != nil {
		prevOutput.Close()
	}
	for _, state := range prevState {
		if state != nil {
			state.Close()
		}
	}
	if prevPosSin != nil {
		prevPosSin.Close()
	}
	if prevPosCos != nil {
		prevPosCos.Close()
	}
	if prevInput != nil {
		prevInput.Close()
	}
}

func (m *ANEDraftModel) initMILRoPEPlan() error {
	plan, err := newMILTransformerPlan(m.inMemoryMIL, m.hiddenDim, m.outputDim, m.ropeAttnDim, true)
	if err != nil {
		return err
	}
	m.applyMILTransformerPlan(plan)
	return nil
}

func (m *ANEDraftModel) initMILAttentionDecodePlan(numHeads, headDim, maxSeqLen int) error {
	if m.inMemoryMIL.ID == 0 {
		return fmt.Errorf("ane draft model: in-memory MIL model is nil")
	}
	if m.hiddenDim <= 0 || m.outputDim <= 0 || numHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return fmt.Errorf(
			"ane draft model: invalid attention decode plan dims hidden=%d output=%d heads=%d headDim=%d maxSeq=%d",
			m.hiddenDim,
			m.outputDim,
			numHeads,
			headDim,
			maxSeqLen,
		)
	}
	kvState, err := NewDraftKVSurfaceState(1, numHeads, headDim, maxSeqLen)
	if err != nil {
		return fmt.Errorf("allocate kv state: %w", err)
	}
	cleanupLocal := func(
		inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf *IOSurfaceFloat32,
	) {
		if currVSurf != nil {
			currVSurf.Close()
		}
		if currKSurf != nil {
			currKSurf.Close()
		}
		if outSurf != nil {
			outSurf.Close()
		}
		if maskSurf != nil {
			maskSurf.Close()
		}
		if posSinSurf != nil {
			posSinSurf.Close()
		}
		if posCosSurf != nil {
			posCosSurf.Close()
		}
		if inSurf != nil {
			inSurf.Close()
		}
		kvState.Close()
	}
	kCacheSurf, vCacheSurf, err := kvState.LayerSurfaces(0)
	if err != nil {
		kvState.Close()
		return fmt.Errorf("resolve kv cache surfaces: %w", err)
	}
	inSurf, err := NewIOSurfaceFloat32(m.hiddenDim)
	if err != nil {
		kvState.Close()
		return fmt.Errorf("allocate input surface: %w", err)
	}
	posCosSurf, err := NewIOSurfaceFloat32(m.ropeAttnDim)
	if err != nil {
		inSurf.Close()
		kvState.Close()
		return fmt.Errorf("allocate pos_cos surface: %w", err)
	}
	posSinSurf, err := NewIOSurfaceFloat32(m.ropeAttnDim)
	if err != nil {
		posCosSurf.Close()
		inSurf.Close()
		kvState.Close()
		return fmt.Errorf("allocate pos_sin surface: %w", err)
	}
	outSurf, err := NewIOSurfaceFloat32(m.outputDim)
	if err != nil {
		posSinSurf.Close()
		posCosSurf.Close()
		inSurf.Close()
		kvState.Close()
		return fmt.Errorf("allocate output surface: %w", err)
	}
	currKSurf, err := NewIOSurfaceFloat32(m.ropeAttnDim)
	if err != nil {
		outSurf.Close()
		posSinSurf.Close()
		posCosSurf.Close()
		inSurf.Close()
		kvState.Close()
		return fmt.Errorf("allocate current K surface: %w", err)
	}
	currVSurf, err := NewIOSurfaceFloat32(m.ropeAttnDim)
	if err != nil {
		currKSurf.Close()
		outSurf.Close()
		posSinSurf.Close()
		posCosSurf.Close()
		inSurf.Close()
		kvState.Close()
		return fmt.Errorf("allocate current V surface: %w", err)
	}
	maskSurf, err := NewIOSurfaceFloat32(numHeads * (maxSeqLen + 1))
	if err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, nil, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("allocate attention mask surface: %w", err)
	}
	if err := inSurf.Write(make([]float32, m.hiddenDim)); err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("init input surface: %w", err)
	}
	ones := make([]float32, m.ropeAttnDim)
	for i := range ones {
		ones[i] = 1
	}
	if err := posCosSurf.Write(ones); err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("init pos_cos surface: %w", err)
	}
	if err := posSinSurf.Write(make([]float32, m.ropeAttnDim)); err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("init pos_sin surface: %w", err)
	}
	if err := maskSurf.Write(make([]float32, numHeads*(maxSeqLen+1))); err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("init attention mask surface: %w", err)
	}
	if err := outSurf.Write(make([]float32, m.outputDim)); err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("init output surface: %w", err)
	}
	if err := currKSurf.Write(make([]float32, m.ropeAttnDim)); err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("init current K surface: %w", err)
	}
	if err := currVSurf.Write(make([]float32, m.ropeAttnDim)); err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf("init current V surface: %w", err)
	}

	base := m.inMemoryMIL.Model()
	inSyms, outSyms := milProcedureSymbolIndices(base)
	if len(inSyms) < 6 || len(outSyms) < 3 {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return fmt.Errorf(
			"resolve attention decode symbol indices: inputs=%v outputs=%v (want at least 6 inputs, 3 outputs)",
			inSyms,
			outSyms,
		)
	}
	cfg := DefaultMultiSurfaceEvalPlanConfig()
	applySharedEventModeFromEnv(&cfg)
	plan, err := NewMultiSurfaceEvalPlan(
		m.inMemoryMIL,
		[]SurfaceBinding{
			{Surface: inSurf, SymbolIndex: inSyms[0]},
			{Surface: kCacheSurf, SymbolIndex: inSyms[1]},
			{Surface: vCacheSurf, SymbolIndex: inSyms[2]},
			{Surface: posCosSurf, SymbolIndex: inSyms[3]},
			{Surface: posSinSurf, SymbolIndex: inSyms[4]},
			{Surface: maskSurf, SymbolIndex: inSyms[5]},
		},
		[]SurfaceBinding{
			{Surface: outSurf, SymbolIndex: outSyms[0]},
			{Surface: currKSurf, SymbolIndex: outSyms[1]},
			{Surface: currVSurf, SymbolIndex: outSyms[2]},
		},
		cfg,
	)
	if err != nil && !cfg.DisableCacheMapping {
		retry := cfg
		retry.DisableCacheMapping = true
		plan, err = NewMultiSurfaceEvalPlan(
			m.inMemoryMIL,
			[]SurfaceBinding{
				{Surface: inSurf, SymbolIndex: inSyms[0]},
				{Surface: kCacheSurf, SymbolIndex: inSyms[1]},
				{Surface: vCacheSurf, SymbolIndex: inSyms[2]},
				{Surface: posCosSurf, SymbolIndex: inSyms[3]},
				{Surface: posSinSurf, SymbolIndex: inSyms[4]},
				{Surface: maskSurf, SymbolIndex: inSyms[5]},
			},
			[]SurfaceBinding{
				{Surface: outSurf, SymbolIndex: outSyms[0]},
				{Surface: currKSurf, SymbolIndex: outSyms[1]},
				{Surface: currVSurf, SymbolIndex: outSyms[2]},
			},
			retry,
		)
	}
	if err != nil {
		cleanupLocal(inSurf, posCosSurf, posSinSurf, maskSurf, outSurf, currKSurf, currVSurf)
		return err
	}
	prevKVState := m.kvState
	prevInput := m.milInput
	prevPosCos := m.milPosCos
	prevPosSin := m.milPosSin
	prevMask := m.milAttnMask
	prevOutput := m.milOutput
	prevKCurr := m.milKCurr
	prevVCurr := m.milVCurr
	prevPlan := m.milMultiPlan
	m.kvState = kvState
	m.milInput = inSurf
	m.milPosCos = posCosSurf
	m.milPosSin = posSinSurf
	m.milAttnMask = maskSurf
	m.milOutput = outSurf
	m.milKCurr = currKSurf
	m.milVCurr = currVSurf
	m.milMultiPlan = plan
	if prevPlan != nil {
		prevPlan.Close()
	}
	if prevVCurr != nil {
		prevVCurr.Close()
	}
	if prevKCurr != nil {
		prevKCurr.Close()
	}
	if prevOutput != nil {
		prevOutput.Close()
	}
	if prevMask != nil {
		prevMask.Close()
	}
	if prevPosSin != nil {
		prevPosSin.Close()
	}
	if prevPosCos != nil {
		prevPosCos.Close()
	}
	if prevInput != nil {
		prevInput.Close()
	}
	if prevKVState != nil {
		prevKVState.Close()
	}
	if err := m.writeCurrentAttentionMaskLocked(); err != nil {
		m.milMultiPlan.Close()
		m.milMultiPlan = nil
		m.milVCurr.Close()
		m.milVCurr = nil
		m.milKCurr.Close()
		m.milKCurr = nil
		m.milOutput.Close()
		m.milOutput = nil
		m.milAttnMask.Close()
		m.milAttnMask = nil
		m.milPosSin.Close()
		m.milPosSin = nil
		m.milPosCos.Close()
		m.milPosCos = nil
		m.milInput.Close()
		m.milInput = nil
		m.kvState.Close()
		m.kvState = nil
		return fmt.Errorf("init attention mask: %w", err)
	}
	return nil
}

func symbolIndicesForProcedure(container objectivec.IObject) []int {
	if container == nil || container.GetID() == 0 {
		return nil
	}
	if objc.Send[bool](container.GetID(), objc.Sel("respondsToSelector:"), objc.Sel("firstIndex")) &&
		objc.Send[bool](container.GetID(), objc.Sel("respondsToSelector:"), objc.Sel("indexGreaterThanIndex:")) {
		const nsNotFound = ^uint(0)
		out := make([]int, 0, 8)
		for idx := objc.Send[uint](container.GetID(), objc.Sel("firstIndex")); idx != nsNotFound; {
			out = append(out, int(idx))
			next := objc.Send[uint](container.GetID(), objc.Sel("indexGreaterThanIndex:"), idx)
			if next == idx {
				break
			}
			idx = next
		}
		return out
	}
	if !objc.Send[bool](container.GetID(), objc.Sel("respondsToSelector:"), objc.Sel("count")) ||
		!objc.Send[bool](container.GetID(), objc.Sel("respondsToSelector:"), objc.Sel("objectAtIndex:")) {
		return nil
	}
	count := int(objc.Send[uint](container.GetID(), objc.Sel("count")))
	out := make([]int, 0, count)
	for i := 0; i < count; i++ {
		elem := objc.Send[objc.ID](container.GetID(), objc.Sel("objectAtIndex:"), uint(i))
		if elem == 0 {
			continue
		}
		val := objc.Send[uint32](elem, objc.Sel("unsignedIntValue"))
		out = append(out, int(val))
	}
	return out
}

func milProcedureSymbolIndices(base objectivec.IObject) ([]int, []int) {
	if base == nil || base.GetID() == 0 {
		return nil, nil
	}
	if inputLayouts, outputLayouts, err := parseCompiledModelLayouts(base); err == nil {
		if envTruthy("MLXGO_ANE_DEBUG_DRAFT_PLAN") {
			inputDesc := make([]string, 0, len(inputLayouts))
			for i, layout := range inputLayouts {
				inputDesc = append(inputDesc, fmt.Sprintf("%d:%s/%s", i, layout.Name, layout.Symbol))
			}
			outputDesc := make([]string, 0, len(outputLayouts))
			for i, layout := range outputLayouts {
				outputDesc = append(outputDesc, fmt.Sprintf("%d:%s/%s", i, layout.Name, layout.Symbol))
			}
			slog.Info(
				"ANE mil compiled layout symbols",
				"inputs",
				inputDesc,
				"outputs",
				outputDesc,
			)
		}
		return contiguousIndices(len(inputLayouts)), contiguousIndices(len(outputLayouts))
	}
	model := appleneuralengine.ANEModelFromID(base.GetID())
	return symbolIndicesForProcedure(model.InputSymbolIndicesForProcedureIndex(0)),
		symbolIndicesForProcedure(model.OutputSymbolIndicesForProcedureIndex(0))
}

func milTransformerProcedureSymbolIndices(base objectivec.IObject, dynamicRoPE bool) ([]int, []int, bool) {
	if base != nil && base.GetID() != 0 {
		if inputLayouts, outputLayouts, err := parseCompiledModelLayouts(base); err == nil {
			inSyms, err := milTransformerInputSymbolIndices(inputLayouts, dynamicRoPE)
			if err == nil {
				outSyms, outErr := milTransformerOutputSymbolIndices(outputLayouts)
				if outErr == nil {
					return inSyms, outSyms, true
				}
			}
		}
	}
	inSyms, outSyms := milProcedureSymbolIndices(base)
	return inSyms, outSyms, false
}

func milTransformerInputSymbolIndices(layouts []compiledTensorLayout, dynamicRoPE bool) ([]int, error) {
	xIndex, err := compiledLayoutIndexForNames(layouts, "x_in", "x")
	if err != nil {
		return nil, err
	}
	if !dynamicRoPE {
		return []int{compiledLayoutSymbolIndex(layouts, xIndex)}, nil
	}
	posCosIndex, err := compiledLayoutIndexForNames(layouts, "pos_cos_in", "pos_cos")
	if err != nil {
		return nil, err
	}
	posSinIndex, err := compiledLayoutIndexForNames(layouts, "pos_sin_in", "pos_sin")
	if err != nil {
		return nil, err
	}
	return []int{
		compiledLayoutSymbolIndex(layouts, xIndex),
		compiledLayoutSymbolIndex(layouts, posCosIndex),
		compiledLayoutSymbolIndex(layouts, posSinIndex),
	}, nil
}

func milTransformerOutputSymbolIndices(layouts []compiledTensorLayout) ([]int, error) {
	yIndex, err := compiledLayoutIndexForNames(layouts, "y", "y@output")
	if err != nil {
		return nil, err
	}
	return []int{compiledLayoutSymbolIndex(layouts, yIndex)}, nil
}

func compiledLayoutIndexForNames(layouts []compiledTensorLayout, names ...string) (int, error) {
	for _, name := range names {
		for i, layout := range layouts {
			if layout.Symbol == name || layout.Name == name {
				return i, nil
			}
		}
	}
	return 0, fmt.Errorf("missing compiled layout for %v", names)
}

func contiguousIndices(n int) []int {
	if n <= 0 {
		return nil
	}
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func compiledLayoutSymbolIndex(layouts []compiledTensorLayout, idx int) int {
	if idx >= 0 && idx < len(layouts) && layouts[idx].SymbolIndex >= 0 {
		return layouts[idx].SymbolIndex
	}
	return idx
}

// DecodePosition returns the current decode position.
func (m *ANEDraftModel) DecodePosition() int {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.decodePos
}

// AdvanceDecodePosition increments decode position and KV state (if configured).
func (m *ANEDraftModel) AdvanceDecodePosition() error {
	if m == nil {
		return fmt.Errorf("ane draft model: model is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("ane draft model: model is closed")
	}
	return m.advanceDecodePositionLocked()
}

func (m *ANEDraftModel) advanceDecodePositionLocked() error {
	if m.kvState != nil {
		if err := m.kvState.Advance(); err != nil {
			return err
		}
	}
	m.decodePos++
	return nil
}

// RewindDecodePosition rewinds decode position and KV state.
func (m *ANEDraftModel) RewindDecodePosition(n int) error {
	if m == nil || n <= 0 {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("ane draft model: model is closed")
	}
	if m.statefulMIL != nil {
		return fmt.Errorf("ane draft model: stateful mil runtime requires rebuild on rewind")
	}
	if m.kvState != nil {
		if err := m.kvState.Rewind(n); err != nil {
			return err
		}
	}
	if n >= m.decodePos {
		m.decodePos = 0
		return nil
	}
	m.decodePos -= n
	return nil
}

// TargetModel verifies draft token batches using the large target model.
type TargetModel interface {
	// VerifyBatch runs the target model on k+1 tokens and returns logits.
	VerifyBatch(tokenIDs []int) ([][]float32, error)
}

// SpeculativeDecoder coordinates ANE draft generation with target verification.
type SpeculativeDecoder struct {
	draft  *ANEDraftModel
	target TargetModel
	k      int
}

// Step generates k draft tokens then verifies on the target model.
func (s *SpeculativeDecoder) Step(inputTokenID int) ([]int, error) {
	if s == nil {
		return nil, fmt.Errorf("speculative decoder: decoder is nil")
	}
	if s.draft == nil {
		return nil, fmt.Errorf("speculative decoder: draft model is nil")
	}
	if s.target == nil {
		return nil, fmt.Errorf("speculative decoder: target model is nil")
	}
	if s.k <= 0 {
		return nil, fmt.Errorf("speculative decoder: k must be > 0 (got %d)", s.k)
	}

	inputEmbed, err := s.draft.embeddingForToken(inputTokenID)
	if err != nil {
		return nil, fmt.Errorf("speculative decoder: input embedding lookup: %w", err)
	}
	draftTokenIDs, _, err := s.draft.GenerateDraft(inputEmbed, s.k)
	if err != nil {
		return nil, fmt.Errorf("speculative decoder: draft generation: %w", err)
	}
	if len(draftTokenIDs) != s.k {
		return nil, fmt.Errorf(
			"speculative decoder: draft generation returned %d tokens want %d",
			len(draftTokenIDs),
			s.k,
		)
	}

	verifyIDs := make([]int, 1+len(draftTokenIDs))
	verifyIDs[0] = inputTokenID
	copy(verifyIDs[1:], draftTokenIDs)

	targetLogits, err := s.target.VerifyBatch(verifyIDs)
	if err != nil {
		return nil, fmt.Errorf("speculative decoder: target verification: %w", err)
	}
	accepted, err := mergeDraftWithTargetGreedy(draftTokenIDs, targetLogits)
	if err != nil {
		return nil, fmt.Errorf("speculative decoder: merge draft with target: %w", err)
	}
	return accepted, nil
}

func (m *ANEDraftModel) embeddingForToken(tokenID int) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("draft embedding lookup: model is nil")
	}
	if len(m.embeddings) == 0 {
		return nil, fmt.Errorf("draft embedding lookup: embeddings are unavailable")
	}
	if tokenID < 0 || tokenID >= m.vocabSize {
		return nil, fmt.Errorf(
			"draft embedding lookup: token id %d out of range [0,%d)",
			tokenID,
			m.vocabSize,
		)
	}
	if m.hiddenDim <= 0 {
		return nil, fmt.Errorf("draft embedding lookup: hidden dim is invalid")
	}
	start := tokenID * m.hiddenDim
	end := start + m.hiddenDim
	if start < 0 || end > len(m.embeddings) {
		return nil, fmt.Errorf(
			"draft embedding lookup: token row out of bounds start=%d end=%d embeddings=%d",
			start,
			end,
			len(m.embeddings),
		)
	}
	row := make([]float32, m.hiddenDim)
	copy(row, m.embeddings[start:end])
	return row, nil
}

// mergeDraftWithTargetGreedy applies classic speculative decoding acceptance:
// accept matching draft tokens until first mismatch; on mismatch accept the
// target token for that position; if all draft tokens match, accept one extra
// token from target at position k.
func mergeDraftWithTargetGreedy(draftTokenIDs []int, targetLogits [][]float32) ([]int, error) {
	k := len(draftTokenIDs)
	if k == 0 {
		return nil, fmt.Errorf("empty draft token sequence")
	}
	if len(targetLogits) < k+1 {
		return nil, fmt.Errorf(
			"target logits length=%d want>=%d for k=%d",
			len(targetLogits),
			k+1,
			k,
		)
	}
	accepted := make([]int, 0, k+1)
	for i, draftTokenID := range draftTokenIDs {
		if len(targetLogits[i]) == 0 {
			return nil, fmt.Errorf("target logits[%d] is empty", i)
		}
		targetTokenID := argmaxFloat32(targetLogits[i])
		if targetTokenID != draftTokenID {
			accepted = append(accepted, targetTokenID)
			return accepted, nil
		}
		accepted = append(accepted, draftTokenID)
	}
	if len(targetLogits[k]) == 0 {
		return nil, fmt.Errorf("target logits[%d] is empty", k)
	}
	accepted = append(accepted, argmaxFloat32(targetLogits[k]))
	return accepted, nil
}
