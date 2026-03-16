//go:build darwin && ane_appleneuralengine

package anedraftimpl

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/tmc/mlx-go-lm/mlxlm/decode"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/sample"
	mlxgoane "github.com/tmc/mlx-go-ane"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/modelir"
)

const autoANEDraftModelCValue = "auto"

type preparedANEDraftModel struct {
	path      string
	outputDim int
	cleanup   func()
	model     *mlxgoane.ANEDraftModel
	tapLayout *mlxgoane.TransformerForwardTapLayout
	mode      string
	strategy  draftStrategy
	modelKind draftModelKind
	maxLayers int
	// needsFFNFallback defers FFN modelc generation until ANE execution is
	// confirmed necessary by caller-side policy.
	needsFFNFallback bool
	ffnFallbackUsed  bool
	widenedAttention bool
	directModelIR    bool
	compileFallback  bool
	compileProfile   string
	// lowConfidence marks generated draft graphs that are expected to have
	// poor acceptance and should be guarded with reference logits fallback.
	lowConfidence bool
}

func NewANEDrafter(modelcPath string, draftModel models.LanguageModel, options ANEDraftOptions) (decode.Drafter, func(), error) {
	cfg := draftModel.Config()
	if cfg == nil {
		return nil, nil, fmt.Errorf("draft model config is nil")
	}
	hiddenDim := cfg.HiddenSize
	vocabSize := cfg.VocabSize
	if hiddenDim <= 0 || vocabSize <= 0 {
		return nil, nil, fmt.Errorf("invalid draft model dims hidden=%d vocab=%d", hiddenDim, vocabSize)
	}

	prepared, err := prepareANEDraftModelC(modelcPath, draftModel, options)
	if err != nil {
		return nil, nil, err
	}
	if prepared == nil {
		return nil, nil, fmt.Errorf("prepare ane draft model returned nil")
	}
	if prepared.directModelIR && prepared.compileFallback {
		compileTier := mlxgoane.ModelIRCompileProfileTier(prepared.compileProfile)
		compileNote := mlxgoane.ModelIRCompileProfileNote(prepared.compileProfile)
		qwenOnANE, qwenOnMLX := mlxgoane.ModelIRCompileProfileQwenSupport(prepared.compileProfile)
		slog.Warn(
			"ANE draft direct ModelIR path degraded to defended compile tier",
			"compile_tier", compileTier,
			"compile_profile", prepared.compileProfile,
			"compile_note", compileNote,
			"qwen_on_ane", qwenOnANE,
			"qwen_on_mlx", qwenOnMLX,
		)
	}
	cleanupModelC := prepared.cleanup
	if cleanupModelC == nil {
		cleanupModelC = func() {}
	}
	useRefLogits := envTruthy("MLXGO_ANE_DRAFT_USE_REFERENCE_LOGITS")
	// Guard is enabled by default on low-confidence draft paths. The
	// reference model is only stepped when needed (reference-only/useRef/taps),
	// so the guard can stay cheap on healthy paths.
	autoRefGuard := prepared.lowConfidence && !envTruthy("MLXGO_ANE_DRAFT_DISABLE_AUTO_REF_GUARD")
	needsReference := options.EnableTaps || useRefLogits || autoRefGuard || prepared.strategy == draftStrategyReferenceOnly
	var refRunner *aneDraftReferenceRunner
	if needsReference {
		refRunner, err = newANEDraftReferenceRunner(draftModel)
		if err != nil {
			slog.Info("ANE draft reference runner unavailable", "error", err)
		}
	}
	if prepared.lowConfidence && !autoRefGuard && !useRefLogits {
		slog.Info(
			"ANE draft low-confidence path: reference guard disabled",
			"env_disable", "MLXGO_ANE_DRAFT_DISABLE_AUTO_REF_GUARD",
		)
	}
	if useRefLogits {
		slog.Info("ANE draft using reference logits override", "env", "MLXGO_ANE_DRAFT_USE_REFERENCE_LOGITS")
	}
	if autoRefGuard {
		if refRunner != nil {
			slog.Info("ANE draft guard enabled for low-confidence model path")
		} else {
			slog.Warn("ANE draft guard requested but reference runner is unavailable")
		}
	}
	preferReferenceOnly := prepared.strategy == draftStrategyReferenceOnly
	if preferReferenceOnly && refRunner == nil {
		cleanupModelC()
		return nil, nil, fmt.Errorf("configure reference-only draft model: reference runner unavailable")
	}
	referenceOnlyEnabled := envTruthy("MLXGO_ANE_DRAFT_ENABLE_LOW_CONFIDENCE_REFERENCE_ONLY")
	if !preferReferenceOnly {
		preferReferenceOnly = prepared.lowConfidence &&
			refRunner != nil &&
			!useRefLogits &&
			!options.EnableTaps &&
			referenceOnlyEnabled &&
			!envTruthy("MLXGO_ANE_DRAFT_DISABLE_LOW_CONFIDENCE_REFERENCE_ONLY")
	}
	if preferReferenceOnly {
		slog.Info(
			"ANE draft path: starting in reference-only mode",
			"strategy",
			prepared.strategy.String(),
			"model_kind",
			prepared.modelKind,
			"env_enable", "MLXGO_ANE_DRAFT_ENABLE_LOW_CONFIDENCE_REFERENCE_ONLY",
			"env_disable", "MLXGO_ANE_DRAFT_DISABLE_LOW_CONFIDENCE_REFERENCE_ONLY",
		)
		if !envTruthy("MLXGO_ANE_DRAFT_KEEP_MODEL_FOR_REFERENCE_ONLY") {
			if prepared.model != nil {
				prepared.model.Close()
				prepared.model = nil
			}
			if prepared.path != "" && strings.EqualFold(strings.TrimSpace(modelcPath), autoANEDraftModelCValue) {
				prevCleanup := cleanupModelC
				path := prepared.path
				cleanupModelC = func() {
					if prevCleanup != nil {
						prevCleanup()
					}
					_ = os.RemoveAll(path)
				}
			}
		}
	} else if prepared.lowConfidence && refRunner != nil && !useRefLogits && !options.EnableTaps {
		slog.Info(
			"ANE draft low-confidence path: keeping ANE runtime active; reference-only mode disabled by default",
			"env_enable", "MLXGO_ANE_DRAFT_ENABLE_LOW_CONFIDENCE_REFERENCE_ONLY",
		)
	}
	if prepared.needsFFNFallback {
		dir, outputDim, genErr := generateANEDraftFFNModelCFromModel(draftModel)
		if genErr != nil {
			cleanupModelC()
			return nil, nil, fmt.Errorf("generate ane draft modelc: %w", genErr)
		}
		prevCleanup := cleanupModelC
		cleanupModelC = func() {
			if prevCleanup != nil {
				prevCleanup()
			}
			_ = os.RemoveAll(dir)
		}
		prepared.path = dir
		prepared.outputDim = outputDim
		prepared.needsFFNFallback = false
		prepared.ffnFallbackUsed = true
		prepared.modelKind = draftModelKindFFNFallback
		slog.Info("Generated ANE draft modelc from draft model weights", "path", dir, "output_dim", outputDim)
	}

	_, _, embeddings, embErr := extractDraftEmbeddings(draftModel)
	lmProjection, lmBias, lmProjErr := extractLMHeadWeightsOutIn(draftModel, hiddenDim, vocabSize)
	if lmProjErr != nil {
		slog.Info("ANE draft LM head projection unavailable; falling back to embeddings projection", "error", lmProjErr)
		lmProjection = nil
		lmBias = nil
	}
	outputDim := prepared.outputDim
	if outputDim <= 0 {
		defaultOutputDim := vocabSize
		switch prepared.modelKind {
		case draftModelKindLMHead:
			defaultOutputDim = vocabSize
		case draftModelKindReferenceOnly:
			defaultOutputDim = vocabSize
		default:
			defaultOutputDim = hiddenDim
		}
		outputDim, err = resolveANEDraftOutputDim(defaultOutputDim)
		if err != nil {
			cleanupModelC()
			return nil, nil, err
		}
	}
	if outputDim == hiddenDim && len(embeddings) == 0 {
		lookupModel, ok := draftModel.(draftEmbeddingLookupModel)
		if !ok {
			cleanupModelC()
			return nil, nil, fmt.Errorf(
				"draft output dim=%d needs embedding matrix for projection, but extraction failed: %v",
				outputDim,
				embErr,
			)
		}
		embeddings, err = materializeEmbeddingsViaLookup(lookupModel, hiddenDim, vocabSize)
		if err != nil {
			cleanupModelC()
			return nil, nil, fmt.Errorf("materialize draft embeddings via lookup: %w", err)
		}
	}
	embedLookup, err := makeDraftTokenEmbeddingLookup(draftModel, embeddings, hiddenDim, vocabSize)
	if err != nil {
		cleanupModelC()
		return nil, nil, err
	}
	aneDraft := prepared.model
	if aneDraft == nil && prepared.strategy != draftStrategyReferenceOnly {
		aneDraft, err = mlxgoane.NewANEDraftModelWithOutputDim(prepared.path, hiddenDim, vocabSize, outputDim, embeddings)
		if err != nil {
			cleanupModelC()
			return nil, nil, fmt.Errorf("open ane draft model: %w", err)
		}
	}
	closeFn := func() {
		if aneDraft != nil {
			aneDraft.Close()
		}
		cleanupModelC()
	}
	return &aneDraftDrafter{
		model:               aneDraft,
		hiddenDim:           hiddenDim,
		vocabSize:           vocabSize,
		outputDim:           outputDim,
		embeddings:          embeddings,
		lmProj:              lmProjection,
		lmBias:              lmBias,
		embedToken:          embedLookup,
		tapLayout:           prepared.tapLayout,
		tapRecorder:         newANEDraftTapRecorder(options, outputDim),
		reference:           refRunner,
		useRefLogits:        useRefLogits,
		autoRefGuard:        autoRefGuard && refRunner != nil,
		lowConfidence:       prepared.lowConfidence,
		referenceOnly:       preferReferenceOnly,
		preferReferenceOnly: preferReferenceOnly,
		mode:                prepared.mode,
		strategy:            prepared.strategy,
		modelKind:           prepared.modelKind,
		maxLayers:           prepared.maxLayers,
		directModelIR:       prepared.directModelIR,
		compileFallback:     prepared.compileFallback,
		compileProfile:      prepared.compileProfile,
		ffnFallbackUsed:     prepared.ffnFallbackUsed,
		widenedAttention:    prepared.widenedAttention,
	}, closeFn, nil
}

func newANEDraftDrafter(modelcPath string, draftModel models.LanguageModel, options aneDraftOptions) (decode.Drafter, func(), error) {
	return NewANEDrafter(modelcPath, draftModel, options)
}

func prepareANEDraftModelC(
	modelcPath string,
	draftModel models.LanguageModel,
	options aneDraftOptions,
) (*preparedANEDraftModel, error) {
	if !strings.EqualFold(strings.TrimSpace(modelcPath), autoANEDraftModelCValue) {
		return &preparedANEDraftModel{
			path:      modelcPath,
			cleanup:   func() {},
			mode:      "external",
			strategy:  draftStrategyFullMIL,
			modelKind: draftModelKindExternalModelC,
		}, nil
	}

	cfg := draftModel.Config()
	policy, err := resolveANEDraftPolicy(cfg)
	if err != nil {
		return nil, err
	}
	if policy.ExplicitReference {
		return &preparedANEDraftModel{
			cleanup:   func() {},
			mode:      policy.Mode,
			strategy:  draftStrategyReferenceOnly,
			modelKind: draftModelKindReferenceOnly,
			outputDim: func() int {
				if cfg != nil {
					return cfg.VocabSize
				}
				return 0
			}(),
			maxLayers:        0,
			widenedAttention: policy.WidenedAttention,
		}, nil
	}

	tryMIL := func(opts aneDraftOptions) (*preparedANEDraftModel, error) {
		var lastErr error
		for _, strategy := range policy.Strategies {
			switch strategy {
			case draftStrategyReferenceOnly:
				return &preparedANEDraftModel{
					cleanup:   func() {},
					mode:      policy.Mode,
					strategy:  strategy,
					modelKind: draftModelKindReferenceOnly,
					outputDim: func() int {
						if cfg != nil {
							return cfg.VocabSize
						}
						return 0
					}(),
					maxLayers:        0,
					widenedAttention: policy.WidenedAttention,
				}, nil
			case draftStrategyFFNFallback:
				return &preparedANEDraftModel{
					cleanup:          func() {},
					mode:             policy.Mode,
					strategy:         strategy,
					modelKind:        draftModelKindFFNFallback,
					needsFFNFallback: true,
					ffnFallbackUsed:  true,
					lowConfidence:    true,
					widenedAttention: policy.WidenedAttention,
				}, nil
			}
			maxLayers := draftStrategyMaxLayers(strategy, policy.TotalLayers)
			if maxLayers == 0 {
				maxLayers = policy.TotalLayers
			}
			for _, outputMode := range draftOutputModeCandidates(strategy, policy.OutputMode, policy.WidenedAttention) {
				model, outputDim, tapLayout, kind, directModelIR, compileFallback, compileProfile, err := generateANEDraftMILTransformerFromModel(
					draftModel,
					opts,
					maxLayers,
					outputMode,
				)
				if err == nil {
					slog.Info(
						"Generated ANE draft MIL transformer model",
						"strategy",
						strategy.String(),
						"output_mode",
						kind,
						"output_dim",
						outputDim,
						"max_layers",
						maxLayers,
					)
					return &preparedANEDraftModel{
						outputDim:        outputDim,
						cleanup:          func() {},
						model:            model,
						tapLayout:        tapLayout,
						mode:             policy.Mode,
						strategy:         strategy,
						modelKind:        kind,
						maxLayers:        maxLayers,
						lowConfidence:    draftSelectionLowConfidence(strategy, kind, policy.WidenedAttention),
						widenedAttention: policy.WidenedAttention,
						directModelIR:    directModelIR,
						compileFallback:  compileFallback,
						compileProfile:   compileProfile,
					}, nil
				}
				lastErr = fmt.Errorf(
					"strategy %s output %s: %w",
					strategy.String(),
					outputMode.String(),
					err,
				)
			}
		}
		if lastErr == nil {
			lastErr = fmt.Errorf("mil transformer compile failed without error detail")
		}
		return nil, lastErr
	}

	prepared, milErr := tryMIL(options)
	if milErr == nil {
		return prepared, nil
	}
	if options.EnableTaps {
		slog.Warn("ANE draft taps requested but MIL transformer compile failed; retrying without taps", "error", milErr)
		noTapOptions := options
		noTapOptions.EnableTaps = false
		prepared, milErr = tryMIL(noTapOptions)
		if milErr == nil {
			slog.Info("Generated ANE draft MIL transformer model without taps fallback", "output_dim", prepared.outputDim)
			return prepared, nil
		}
	}

	slog.Info("ANE draft MIL transformer generation failed, deferring FFN-only modelc generation", "error", milErr)
	return &preparedANEDraftModel{
		cleanup:          func() {},
		mode:             policy.Mode,
		strategy:         draftStrategyFFNFallback,
		modelKind:        draftModelKindFFNFallback,
		needsFFNFallback: true,
		ffnFallbackUsed:  true,
		lowConfidence:    true,
		widenedAttention: policy.WidenedAttention,
	}, nil
}

type draftLayerMLPProvider interface {
	NumLayers() int
	LayerMLP(layer int) interface{}
}

type draftLayerAttentionProvider interface {
	NumLayers() int
	LayerAttention(layer int) interface{}
}

type attentionProjectionDimsProvider interface {
	ProjectionDims(key string) (inputDim, outputDim int, ok bool)
}

type reflectDraftLayerProvider struct {
	layers []reflect.Value
}

func debugReflectProviderEnabled() bool {
	return envTruthy("MLXGO_ANE_DRAFT_DEBUG_REFLECT")
}

func newReflectDraftLayerProvider(model models.LanguageModel) (*reflectDraftLayerProvider, error) {
	mv := reflect.ValueOf(model)
	if !mv.IsValid() {
		return nil, fmt.Errorf("draft model is nil")
	}
	if debugReflectProviderEnabled() {
		slog.Info("ANE draft reflect provider: model type", "type", mv.Type().String())
	}
	layersMethod := mv.MethodByName("Layers")
	var out reflect.Value
	if layersMethod.IsValid() {
		mt := layersMethod.Type()
		if mt.NumIn() != 0 || mt.NumOut() != 1 {
			return nil, fmt.Errorf("draft model Layers() has unsupported signature %s", mt)
		}
		out = forceReadableValue(layersMethod.Call(nil)[0])
	} else {
		var ok bool
		out, ok = findLayerSliceRecursive(mv, make(map[uintptr]bool), 0)
		if !ok {
			return nil, fmt.Errorf("draft model does not expose Layer* providers or discoverable layer slice")
		}
	}
	if out.Kind() != reflect.Slice {
		return nil, fmt.Errorf("draft model layers value is %s, want slice", out.Kind())
	}
	if debugReflectProviderEnabled() {
		slog.Info(
			"ANE draft reflect provider: layers slice",
			"type",
			out.Type().String(),
			"len",
			out.Len(),
		)
		linearCount := 0
		selfCount := 0
		for i := 0; i < out.Len(); i++ {
			layer := forceReadableValue(out.Index(i))
			for layer.IsValid() && (layer.Kind() == reflect.Interface || layer.Kind() == reflect.Ptr) {
				if layer.IsNil() {
					break
				}
				layer = forceReadableValue(layer.Elem())
			}
			if layer.IsValid() {
				if f := fieldByAnyName(layer, "linearAttn", "linearAttention"); f.IsValid() {
					ff := forceReadableValue(f)
					if !(ff.Kind() == reflect.Ptr && ff.IsNil()) {
						linearCount++
					}
				}
				if f := fieldByAnyName(layer, "selfAttn", "selfAttention"); f.IsValid() {
					ff := forceReadableValue(f)
					if !(ff.Kind() == reflect.Ptr && ff.IsNil()) {
						selfCount++
					}
				}
			}
		}
		slog.Info("ANE draft reflect provider: attention mode counts", "linear_layers", linearCount, "self_layers", selfCount)
		if out.Len() > 0 {
			first := forceReadableValue(out.Index(0))
			for first.IsValid() && (first.Kind() == reflect.Interface || first.Kind() == reflect.Ptr) {
				if first.IsNil() {
					break
				}
				first = forceReadableValue(first.Elem())
			}
			slog.Info(
				"ANE draft reflect provider: first layer",
				"type",
				first.Type().String(),
				"fields",
				structFieldNames(first),
			)
		}
	}
	layers := make([]reflect.Value, out.Len())
	for i := 0; i < out.Len(); i++ {
		layers[i] = forceReadableValue(out.Index(i))
	}
	return &reflectDraftLayerProvider{layers: layers}, nil
}

func findLayerSliceRecursive(v reflect.Value, seen map[uintptr]bool, depth int) (reflect.Value, bool) {
	if !v.IsValid() || depth > 16 {
		return reflect.Value{}, false
	}
	v = forceReadableValue(v)
	for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr) {
		if v.IsNil() {
			return reflect.Value{}, false
		}
		if v.Kind() == reflect.Ptr {
			ptr := v.Pointer()
			if ptr != 0 {
				if seen[ptr] {
					return reflect.Value{}, false
				}
				seen[ptr] = true
			}
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() {
		return reflect.Value{}, false
	}

	switch v.Kind() {
	case reflect.Slice:
		if looksLikeLayerSlice(v) {
			return v, true
		}
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			fieldName := strings.ToLower(t.Field(i).Name)
			field := forceReadableValue(v.Field(i))
			if field.Kind() == reflect.Slice && strings.Contains(fieldName, "layer") && looksLikeLayerSlice(field) {
				return field, true
			}
		}
		for i := 0; i < v.NumField(); i++ {
			if out, ok := findLayerSliceRecursive(forceReadableValue(v.Field(i)), seen, depth+1); ok {
				return out, true
			}
		}
	}
	return reflect.Value{}, false
}

func looksLikeLayerSlice(v reflect.Value) bool {
	if !v.IsValid() || v.Kind() != reflect.Slice || v.Len() == 0 {
		return false
	}
	elem := forceReadableValue(v.Index(0))
	for elem.IsValid() && (elem.Kind() == reflect.Interface || elem.Kind() == reflect.Ptr) {
		if elem.IsNil() {
			return false
		}
		elem = forceReadableValue(elem.Elem())
	}
	if !elem.IsValid() || elem.Kind() != reflect.Struct {
		return false
	}
	hasAttn := fieldByAnyName(elem, "selfAttn", "selfAttention", "linearAttn", "linearAttention", "attention", "attn").IsValid()
	hasMLP := fieldByAnyName(elem, "mlp", "MLP", "feedForward", "ffn").IsValid()
	return hasAttn && hasMLP
}

func (p *reflectDraftLayerProvider) NumLayers() int {
	if p == nil {
		return 0
	}
	return len(p.layers)
}

func (p *reflectDraftLayerProvider) LayerAttention(layer int) interface{} {
	v, ok := p.layerField(layer, "selfAttn", "selfAttention", "linearAttn", "linearAttention", "attention", "attn")
	if !ok {
		return nil
	}
	return v.Interface()
}

func (p *reflectDraftLayerProvider) LayerMLP(layer int) interface{} {
	v, ok := p.layerField(layer, "mlp", "MLP", "feedForward", "ffn")
	if !ok {
		return nil
	}
	return v.Interface()
}

func (p *reflectDraftLayerProvider) layerField(layer int, names ...string) (reflect.Value, bool) {
	if p == nil || layer < 0 || layer >= len(p.layers) {
		return reflect.Value{}, false
	}
	v := forceReadableValue(p.layers[layer])
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return reflect.Value{}, false
		}
		v = forceReadableValue(v.Elem())
	}
	for _, name := range names {
		if name == "" {
			continue
		}
		nameSet := map[string]struct{}{name: {}}
		field, ok := findFieldByNamesRecursive(v, nameSet, make(map[uintptr]bool), 0)
		if !ok || !field.IsValid() {
			continue
		}
		field = forceReadableValue(field)
		if !field.IsValid() || !field.CanInterface() {
			continue
		}
		return field, true
	}
	if debugReflectProviderEnabled() && layer == 0 {
		slog.Info(
			"ANE draft reflect provider: field not found",
			"layer",
			layer,
			"candidates",
			strings.Join(names, ","),
			"layer_type",
			v.Type().String(),
			"fields",
			structFieldNames(v),
		)
	}
	return reflect.Value{}, false
}

func structFieldNames(v reflect.Value) string {
	v = forceReadableValue(v)
	for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr) {
		if v.IsNil() {
			return ""
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() || v.Kind() != reflect.Struct {
		return ""
	}
	t := v.Type()
	names := make([]string, 0, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		names = append(names, t.Field(i).Name)
	}
	return strings.Join(names, ",")
}

func generateANEDraftFFNModelCFromModel(
	draftModel models.LanguageModel,
) (modelcPath string, outputDim int, err error) {
	dim, hiddenDim, w1, w3, w2, err := extractLayer0FFNWeights(draftModel)
	if err != nil {
		return "", 0, err
	}
	dir, err := os.MkdirTemp("", "mlxgo_ane_draft_ffn_*.mlmodelc")
	if err != nil {
		return "", 0, fmt.Errorf("create temp modelc dir: %w", err)
	}
	if err := mlxgoane.GenerateFFNEspressoDir(dir, dim, hiddenDim, w1, w3, w2); err != nil {
		_ = os.RemoveAll(dir)
		return "", 0, fmt.Errorf("write espresso ffn dir: %w", err)
	}
	return dir, dim, nil
}

func generateANEDraftMILTransformerFromModel(
	draftModel models.LanguageModel,
	options aneDraftOptions,
	maxLayersOverride int,
	outputMode draftOutputMode,
) (*mlxgoane.ANEDraftModel, int, *mlxgoane.TransformerForwardTapLayout, draftModelKind, bool, bool, string, error) {
	cfg := draftModel.Config()
	if cfg == nil {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("draft model config is nil")
	}
	dim := cfg.HiddenSize
	hiddenDim := cfg.IntermediateSize
	vocabSize := cfg.VocabSize
	if dim <= 0 || hiddenDim <= 0 || vocabSize <= 0 {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf(
			"invalid draft transformer dims hidden=%d intermediate=%d vocab=%d",
			dim,
			hiddenDim,
			vocabSize,
		)
	}
	if cfg.NumLayers <= 0 || cfg.NumHeads <= 0 {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("invalid draft transformer config layers=%d heads=%d", cfg.NumLayers, cfg.NumHeads)
	}
	numLayers, err := resolveDraftMILNumLayers(cfg.NumLayers, maxLayersOverride)
	if err != nil {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", err
	}
	if model, outDim, tapLayout, kind, ok, compileFallback, compileProfile, err := tryGenerateANEDraftModelIRFromModel(
		draftModel,
		options,
		numLayers,
		outputMode,
	); ok {
		if err != nil {
			slog.Info(
				"ANE draft direct ModelIR path unavailable; falling back to reflection extractor",
				"output_mode", outputMode.String(),
				"layers", numLayers,
				"error", err,
			)
		} else {
			return model, outDim, tapLayout, kind, true, compileFallback, compileProfile, nil
		}
	}

	numHeads := cfg.NumHeads
	numKVHeads := numHeads
	if cfg.NumKeyValueHeads != nil && *cfg.NumKeyValueHeads > 0 {
		numKVHeads = *cfg.NumKeyValueHeads
	}

	attnProvider, attnOK := draftModel.(draftLayerAttentionProvider)
	mlpProvider, mlpOK := draftModel.(draftLayerMLPProvider)
	reflectProvider, reflectErr := newReflectDraftLayerProvider(draftModel)
	if !attnOK || !mlpOK {
		if reflectErr != nil {
			if !attnOK {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("draft model does not expose LayerAttention provider: %w", reflectErr)
			}
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("draft model does not expose LayerMLP provider: %w", reflectErr)
		}
		if !attnOK {
			attnProvider = reflectProvider
		}
		if !mlpOK {
			mlpProvider = reflectProvider
		}
	}
	// Some model implementations expose LayerAttention but return nil for linear-attention variants.
	// Prefer reflection-backed provider when the typed provider has nil at layer 0.
	if reflectErr == nil {
		if attnProvider != nil && attnProvider.NumLayers() > 0 && attnProvider.LayerAttention(0) == nil {
			if probe := reflectProvider.LayerAttention(0); probe != nil {
				attnProvider = reflectProvider
				slog.Info("ANE draft: switched attention provider to reflection fallback", "reason", "layer0 typed provider returned nil")
			}
		}
		if mlpProvider != nil && mlpProvider.NumLayers() > 0 && mlpProvider.LayerMLP(0) == nil {
			if probe := reflectProvider.LayerMLP(0); probe != nil {
				mlpProvider = reflectProvider
				slog.Info("ANE draft: switched MLP provider to reflection fallback", "reason", "layer0 typed provider returned nil")
			}
		}
	}
	if attnProvider.NumLayers() < numLayers {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("attention layers=%d want>=%d", attnProvider.NumLayers(), numLayers)
	}
	if mlpProvider.NumLayers() < numLayers {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("mlp layers=%d want>=%d", mlpProvider.NumLayers(), numLayers)
	}
	layerIndices, err := selectDraftMILLayerIndices(attnProvider, mlpProvider, numLayers)
	if err != nil {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", err
	}
	if len(layerIndices) < numLayers {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf(
			"mil transformer extraction selected %d/%d layers",
			len(layerIndices),
			numLayers,
		)
	}
	if numLayers == 0 {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("no attention-capable layers available for MIL transformer generation")
	}

	layer0Attn := attnProvider.LayerAttention(0)
	if layer0Attn == nil {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer 0: attention is nil")
	}
	layer0AttnValue := forceReadableValue(reflect.ValueOf(layer0Attn))
	projInfo, err := inferAttentionProjectionInfo(layer0Attn, layer0AttnValue, dim, numHeads, numKVHeads, cfg.HeadDim)
	if err != nil {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", err
	}
	headDim := projInfo.HeadDim
	attnDim := projInfo.AttnDim
	kvDim := projInfo.KVDim
	attentionOutputGate := projInfo.WidenedQProj
	if attnDim <= 0 || kvDim <= 0 || kvDim > attnDim {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf(
			"invalid attention dims attnDim=%d kvDim=%d heads=%d kvHeads=%d headDim=%d",
			attnDim,
			kvDim,
			numHeads,
			numKVHeads,
			headDim,
		)
	}
	rmsNormEps := cfg.RMSNormEps
	if rmsNormEps <= 0 {
		rmsNormEps = 1e-6
	}
	maxSeqLen := options.MaxSeqLen
	if maxSeqLen <= 0 {
		maxSeqLen = cfg.MaxPositionEmbeddings
	}
	if maxSeqLen <= 0 {
		maxSeqLen = 2048
	}
	ropeTheta := cfg.RopeTheta
	if ropeTheta <= 0 {
		ropeTheta = 10000
	}

	weights := mlxgoane.MILTransformerWeights{
		Layers: make([]mlxgoane.MILTransformerLayerWeights, numLayers),
	}
	inputNorms := make([][]float32, numLayers)
	postNorms := make([][]float32, numLayers)
	for layer := 0; layer < numLayers; layer++ {
		modelLayer := layerIndices[layer]
		inNorm, postNorm, err := extractTransformerLayerNormVectors(draftModel, modelLayer, dim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) norm weights: %w", layer, modelLayer, err)
		}
		inputNorms[layer] = inNorm
		postNorms[layer] = postNorm
	}
	finalNorm, err := extractFinalNormVector(draftModel, dim)
	if err != nil {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("extract final norm: %w", err)
	}
	ropeCos, ropeSin := buildRoPETables(maxSeqLen, headDim, ropeTheta)
	weights.FinalNorm = finalNorm
	weights.RopeCos = ropeCos
	weights.RopeSin = ropeSin

	for layer := 0; layer < numLayers; layer++ {
		modelLayer := layerIndices[layer]
		attn := attnProvider.LayerAttention(modelLayer)
		if attn == nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d): attention is nil", layer, modelLayer)
		}
		attnValue := forceReadableValue(reflect.ValueOf(attn))

		var (
			qw     []float32
			qb     []float32
			qGateW []float32
			qGateB []float32
		)
		if attentionOutputGate {
			qw, qGateW, qb, qGateB, err = extractSplitProjectionWeightBias(
				attnValue,
				[]string{"qProj", "QProj"},
				[]string{"qBias", "QBias"},
				dim,
				attnDim,
			)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) q_proj split: %w", layer, modelLayer, err)
			}
		} else {
			qw, err = extractProjectionWeightOutIn(attnValue, []string{"qProj", "QProj"}, dim, attnDim)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) q_proj: %w", layer, modelLayer, err)
			}
			qb, err = extractProjectionBiasOut(attnValue, []string{"qProj", "QProj"}, []string{"qBias", "QBias"}, attnDim)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) q_proj bias: %w", layer, modelLayer, err)
			}
		}
		qNorm, err := extractVectorFieldOrDefault(attnValue, []string{"qNorm", "QNorm"}, headDim, fmt.Sprintf("layer %d q_norm", layer))
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", err
		}

		kw, err := extractProjectionWeightOutIn(attnValue, []string{"kProj", "KProj"}, dim, kvDim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) k_proj: %w", layer, modelLayer, err)
		}
		kb, err := extractProjectionBiasOut(attnValue, []string{"kProj", "KProj"}, []string{"kBias", "KBias"}, kvDim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) k_proj bias: %w", layer, modelLayer, err)
		}
		kNorm, err := extractVectorFieldOrDefault(attnValue, []string{"kNorm", "KNorm"}, headDim, fmt.Sprintf("layer %d k_norm", layer))
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", err
		}
		vw, err := extractProjectionWeightOutIn(attnValue, []string{"vProj", "VProj"}, dim, kvDim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) v_proj: %w", layer, modelLayer, err)
		}
		vb, err := extractProjectionBiasOut(attnValue, []string{"vProj", "VProj"}, []string{"vBias", "VBias"}, kvDim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) v_proj bias: %w", layer, modelLayer, err)
		}
		ow, err := extractProjectionWeightOutIn(attnValue, []string{"oProj", "OProj"}, attnDim, dim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) o_proj: %w", layer, modelLayer, err)
		}
		ob, err := extractProjectionBiasOut(attnValue, []string{"oProj", "OProj"}, []string{"oBias", "OBias"}, dim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) o_proj bias: %w", layer, modelLayer, err)
		}

		if numKVHeads != numHeads {
			kw, err = expandGQAProjectionRows(kw, dim, numHeads, numKVHeads, headDim)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d k_proj gqa expand: %w", layer, err)
			}
			kb, err = expandGQABias(kb, numHeads, numKVHeads, headDim)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d k_proj bias gqa expand: %w", layer, err)
			}
			vw, err = expandGQAProjectionRows(vw, dim, numHeads, numKVHeads, headDim)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d v_proj gqa expand: %w", layer, err)
			}
			vb, err = expandGQABias(vb, numHeads, numKVHeads, headDim)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d v_proj bias gqa expand: %w", layer, err)
			}
		}

		mlp := mlpProvider.LayerMLP(modelLayer)
		if mlp == nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d): mlp is nil", layer, modelLayer)
		}
		_, _, w1, w3, w2, err := extractSwiGLUWeightsFromMLPValue(forceReadableValue(reflect.ValueOf(mlp)), dim, hiddenDim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d (model layer %d) mlp weights: %w", layer, modelLayer, err)
		}
		b1, err := extractProjectionBiasOut(forceReadableValue(reflect.ValueOf(mlp)), []string{"gateProj", "GateProj"}, nil, hiddenDim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d gate_proj bias: %w", layer, err)
		}
		b3, err := extractProjectionBiasOut(forceReadableValue(reflect.ValueOf(mlp)), []string{"upProj", "UpProj"}, nil, hiddenDim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d up_proj bias: %w", layer, err)
		}
		b2, err := extractProjectionBiasOut(forceReadableValue(reflect.ValueOf(mlp)), []string{"downProj", "DownProj"}, nil, dim)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("layer %d down_proj bias: %w", layer, err)
		}

		weights.Layers[layer] = mlxgoane.MILTransformerLayerWeights{
			QW:                qw,
			QB:                qb,
			QGateW:            qGateW,
			QGateB:            qGateB,
			KW:                kw,
			KB:                kb,
			VW:                vw,
			VB:                vb,
			OW:                ow,
			OB:                ob,
			W1:                w1,
			B1:                b1,
			W3:                w3,
			B3:                b3,
			W2:                w2,
			B2:                b2,
			InputNorm:         inputNorms[layer],
			PostAttentionNorm: postNorms[layer],
			QNorm:             qNorm,
			KNorm:             kNorm,
		}
	}

	lmHeadW, lmHeadB, lmHeadErr := extractLMHeadWeightsOutIn(draftModel, dim, vocabSize)
	haveLMHead := lmHeadErr == nil
	if haveLMHead {
		weights.LMHeadW = lmHeadW
		weights.LMHeadB = lmHeadB
	} else {
		slog.Info("ANE draft MIL transformer: lm_head extraction unavailable; using hidden-state output fallback", "error", lmHeadErr)
	}

	baseMILCfg := mlxgoane.MILTransformerConfig{
		NumLayers:           numLayers,
		Dim:                 dim,
		AttentionDim:        attnDim,
		NumHeads:            numHeads,
		HeadDim:             headDim,
		HiddenDim:           hiddenDim,
		VocabSize:           vocabSize,
		RMSNormEps:          rmsNormEps,
		MaxSeqLen:           maxSeqLen,
		EnableTaps:          options.EnableTaps,
		AttentionOutputGate: attentionOutputGate,
		// Keep dynamic RoPE opt-in for now while compiler stability is evaluated.
		DynamicRoPEInputs: false,
	}
	if raw := strings.TrimSpace(os.Getenv("MLXGO_ANE_DRAFT_DYNAMIC_ROPE")); raw != "" {
		switch strings.ToLower(raw) {
		case "0", "false", "no", "off":
			baseMILCfg.DynamicRoPEInputs = false
		case "1", "true", "yes", "on":
			baseMILCfg.DynamicRoPEInputs = true
		default:
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("parse MLXGO_ANE_DRAFT_DYNAMIC_ROPE=%q: want bool-like value", raw)
		}
	}
	forceAttnDecode := envTruthy("MLXGO_ANE_DRAFT_FORCE_ATTN_DECODE")
	tryAttnDecode := forceAttnDecode || envTruthy("MLXGO_ANE_DRAFT_TRY_ATTN_DECODE")
	widenedAttention := attentionOutputGate
	var model *mlxgoane.ANEDraftModel
	compileWithCfg := func(
		cfg mlxgoane.MILTransformerConfig,
		outDim int,
		kind draftModelKind,
	) (*mlxgoane.ANEDraftModel, int, *mlxgoane.TransformerForwardTapLayout, draftModelKind, error) {
		actualOutDim := outDim
		var tapLayout *mlxgoane.TransformerForwardTapLayout
		if cfg.EnableTaps {
			layout, err := mlxgoane.TransformerTapLayoutForConfig(cfg)
			if err != nil {
				return nil, 0, nil, draftModelKindUnknown, err
			}
			tapLayout = &layout
			actualOutDim = layout.TotalDim
		}
		model, err := mlxgoane.NewANEDraftModelFromMILTransformer(cfg, weights, dim, vocabSize, actualOutDim, nil)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, err
		}
		configureGeneratedANEDraftState(model, weights, headDim, maxSeqLen, numLayers, numHeads)
		return model, actualOutDim, tapLayout, kind, nil
	}
	if forceAttnDecode {
		slog.Info(
			"ANE draft: forcing single-layer attention decode path",
			"heads",
			numHeads,
			"head_dim",
			headDim,
			"max_seq",
			maxSeqLen,
		)
		attentionDecodeCfg := mlxgoane.MILAttentionDecodeConfig{
			Dim:          dim,
			AttentionDim: attnDim,
			NumHeads:     numHeads,
			HeadDim:      headDim,
			MaxSeqLen:    maxSeqLen,
		}
		model, err = mlxgoane.NewANEDraftModelFromMILAttentionDecode(
			attentionDecodeCfg,
			weights.Layers[0],
			dim,
			vocabSize,
			nil,
		)
		if err == nil {
			configureGeneratedANEDraftState(model, weights, headDim, maxSeqLen, 1, numHeads)
			return model, dim, nil, draftModelKindLastAttnResidual, false, false, "", nil
		}
		slog.Warn("ANE draft: forced attention decode compile failed; falling back to standard chain", "error", err)
	}
	if outputMode == draftOutputModeLastAttnResidual && maxLayersOverride == 1 && widenedAttention && !options.EnableTaps {
		quickCfg := baseMILCfg
		quickCfg.IncludeLMHead = false
		quickCfg.SkipFFN = true
		quickCfg.DisableNormOps = true
		model, err = mlxgoane.NewANEDraftModelFromMILTransformer(quickCfg, weights, dim, vocabSize, dim, nil)
		if err == nil {
			configureGeneratedANEDraftState(model, weights, headDim, maxSeqLen, 1, numHeads)
			slog.Info("Generated ANE draft MIL transformer model (single-layer widened-attention fast path)", "output_dim", dim)
			return model, dim, nil, draftModelKindLastAttnResidual, false, false, "", nil
		}
		slog.Info(
			"ANE draft widened-attention fast path failed; retrying full fallback ladder",
			"error", err,
			"layers", numLayers,
			"dim", dim,
			"attn_dim", attnDim,
		)
	}

	withoutLMHead := baseMILCfg
	withoutLMHead.IncludeLMHead = false
	withoutLMHeadOutputDim := dim
	switch outputMode {
	case draftOutputModeLMHead:
		if !haveLMHead {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("lm head weights unavailable")
		}
		withLMHead := baseMILCfg
		withLMHead.IncludeLMHead = true
		model, outDim, tapLayout, kind, err := compileWithCfg(withLMHead, vocabSize, draftModelKindLMHead)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		if options.EnableTaps {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("mil transformer compile with taps enabled failed: %w", err)
		}
		withLMHeadNoNorm := withLMHead
		withLMHeadNoNorm.DisableNormOps = true
		model, outDim, tapLayout, kind, err = compileWithCfg(withLMHeadNoNorm, vocabSize, draftModelKindLMHead)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("compile lm_head output: %w", err)

	case draftOutputModeFinalHidden:
		model, outDim, tapLayout, kind, err := compileWithCfg(withoutLMHead, withoutLMHeadOutputDim, draftModelKindFinalHidden)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		withoutLMHeadNoNorm := withoutLMHead
		withoutLMHeadNoNorm.DisableNormOps = true
		model, outDim, tapLayout, kind, err = compileWithCfg(withoutLMHeadNoNorm, withoutLMHeadOutputDim, draftModelKindFinalHidden)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		withoutLMHeadNoNormLinearFFN := withoutLMHeadNoNorm
		withoutLMHeadNoNormLinearFFN.UseConvFFN = false
		withoutLMHeadNoNormLinearFFN.LinearFFN = true
		model, outDim, tapLayout, kind, err = compileWithCfg(withoutLMHeadNoNormLinearFFN, withoutLMHeadOutputDim, draftModelKindFinalHidden)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		model, err = mlxgoane.NewANEDraftModelFromMILTransformerLayerStack(withoutLMHead, weights, dim, vocabSize, dim, nil)
		if err != nil {
			return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("compile final_hidden output: %w", err)
		}
		configureGeneratedANEDraftState(model, weights, headDim, maxSeqLen, numLayers, numHeads)
		return model, dim, nil, draftModelKindFinalHidden, false, false, "", nil

	case draftOutputModeLastFFNResidual:
		withoutLMHeadNoNorm := withoutLMHead
		withoutLMHeadNoNorm.DisableNormOps = true
		model, outDim, tapLayout, kind, err := compileWithCfg(withoutLMHeadNoNorm, withoutLMHeadOutputDim, draftModelKindLastFFNResidual)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		withoutLMHeadNoNormLinearFFN := withoutLMHeadNoNorm
		withoutLMHeadNoNormLinearFFN.UseConvFFN = false
		withoutLMHeadNoNormLinearFFN.LinearFFN = true
		model, outDim, tapLayout, kind, err = compileWithCfg(withoutLMHeadNoNormLinearFFN, withoutLMHeadOutputDim, draftModelKindLastFFNResidual)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("compile last_ffn_residual output: %w", err)

	case draftOutputModeLastAttnResidual:
		attentionOnly := withoutLMHead
		attentionOnly.SkipFFN = true
		attentionOnly.DisableNormOps = true
		model, outDim, tapLayout, kind, err := compileWithCfg(attentionOnly, withoutLMHeadOutputDim, draftModelKindLastAttnResidual)
		if err == nil {
			return model, outDim, tapLayout, kind, false, false, "", nil
		}
		if tryAttnDecode {
			attentionDecodeCfg := mlxgoane.MILAttentionDecodeConfig{
				Dim:          dim,
				AttentionDim: attnDim,
				NumHeads:     numHeads,
				HeadDim:      headDim,
				MaxSeqLen:    maxSeqLen,
			}
			model, err = mlxgoane.NewANEDraftModelFromMILAttentionDecode(
				attentionDecodeCfg,
				weights.Layers[0],
				dim,
				vocabSize,
				nil,
			)
			if err == nil {
				configureGeneratedANEDraftState(model, weights, headDim, maxSeqLen, 1, numHeads)
				return model, dim, nil, draftModelKindLastAttnResidual, false, false, "", nil
			}
		}
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("compile last_attn_residual output: %w", err)

	default:
		return nil, 0, nil, draftModelKindUnknown, false, false, "", fmt.Errorf("unsupported draft output mode %s", outputMode.String())
	}
}

func resolveDraftMILNumLayers(totalLayers, maxLayersOverride int) (int, error) {
	if totalLayers <= 0 {
		return 0, fmt.Errorf("invalid draft transformer layer count=%d", totalLayers)
	}
	numLayers := totalLayers
	switch {
	case maxLayersOverride > 0:
		if maxLayersOverride < numLayers {
			numLayers = maxLayersOverride
		}
	default:
		if raw := strings.TrimSpace(os.Getenv("MLXGO_ANE_DRAFT_MIL_MAX_LAYERS")); raw != "" {
			v, err := strconv.Atoi(raw)
			if err != nil {
				return 0, fmt.Errorf("parse MLXGO_ANE_DRAFT_MIL_MAX_LAYERS=%q: %w", raw, err)
			}
			if v > 0 && v < numLayers {
				numLayers = v
			}
		}
	}
	return numLayers, nil
}

func tryGenerateANEDraftModelIRFromModel(
	draftModel models.LanguageModel,
	options aneDraftOptions,
	numLayers int,
	outputMode draftOutputMode,
) (*mlxgoane.ANEDraftModel, int, *mlxgoane.TransformerForwardTapLayout, draftModelKind, bool, bool, string, error) {
	trace := envTruthy("MLXGO_ANE_DRAFT_DEBUG_DIRECT_MODELIR")
	logPhase := func(phase string, args ...any) {
		if !trace {
			return
		}
		slog.Info("ANE draft direct ModelIR phase", append([]any{"phase", phase}, args...)...)
	}
	lowerable, ok := draftModel.(models.DecodeModelIRLowerable)
	if !ok {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", nil
	}
	if options.EnableTaps {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", nil
	}
	cfg := draftModel.Config()
	if cfg == nil {
		return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf("draft model config is nil")
	}
	includeLMHead, outputDim, kind, ok := draftModelIROutputMode(cfg, outputMode)
	if !ok {
		return nil, 0, nil, draftModelKindUnknown, false, false, "", nil
	}
	if numLayers <= 0 {
		return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf("invalid direct modelir layer count=%d", numLayers)
	}
	maxSeqLen := options.MaxSeqLen
	if maxSeqLen <= 0 {
		maxSeqLen = cfg.MaxPositionEmbeddings
	}
	if maxSeqLen <= 0 {
		maxSeqLen = 2048
	}
	logPhase(
		"config_ready",
		"layers", numLayers,
		"max_seq_len", maxSeqLen,
		"output_mode", outputMode.String(),
		"include_lm_head", includeLMHead,
	)
	headDim := cfg.HeadDim
	if headDim <= 0 {
		if cfg.NumHeads <= 0 || cfg.HiddenSize <= 0 || cfg.HiddenSize%cfg.NumHeads != 0 {
			return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf(
				"derive direct modelir head dim: hidden=%d heads=%d headDim=%d",
				cfg.HiddenSize,
				cfg.NumHeads,
				cfg.HeadDim,
			)
		}
		headDim = cfg.HiddenSize / cfg.NumHeads
	}
	logPhase("lower_begin")
	lowered, err := lowerable.LowerDecodeModelIR(models.DecodeModelIROptions{
		MaxLayers:     numLayers,
		MaxSeqLen:     maxSeqLen,
		IncludeLMHead: includeLMHead,
		StatefulKV:    true,
		AttentionMask: false,
	})
	if err != nil {
		return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf("lower decode modelir: %w", err)
	}
	logPhase("lower_done", "functions", len(lowered.Functions), "weights", len(lowered.Weights))
	logPhase("check_begin")
	if err := modelir.Check(lowered); err != nil {
		return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf("check lowered modelir: %w", err)
	}
	logPhase("check_done")
	logPhase("validate_begin")
	if ds := modelir.ValidateForTarget(lowered, "coreml-ane-v1"); len(ds) > 0 {
		return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf("validate lowered modelir for ANE: %s", ds[0])
	}
	logPhase("validate_done")
	reifyOpts := mlxgoane.ReifyOptions{
		TransformerConfig: mlxgoane.MILTransformerConfig{
			NumLayers:          numLayers,
			MaxSeqLen:          maxSeqLen,
			KVCacheState:       true,
			KVCacheMaxLen:      maxSeqLen,
			AttentionMaskInput: false,
		},
		RequestedLayers: numLayers,
		SelectedLayers:  numLayers,
	}
	logPhase("compile_begin")
	model, reified, err := mlxgoane.NewANEDraftModelFromModelIRProgram(
		lowered,
		reifyOpts,
		cfg.HiddenSize,
		cfg.VocabSize,
		outputDim,
		nil,
	)
	if err != nil {
		return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf("open direct modelir draft model: %w", err)
	}
	logPhase("compile_done", "selected_layers", reified.SelectedLayers)
	ropeTheta := cfg.RopeTheta
	if ropeTheta <= 0 {
		ropeTheta = 10000
	}
	ropeCos, ropeSin := buildRoPETables(maxSeqLen, headDim, ropeTheta)
	logPhase("rope_tables_begin", "head_dim", headDim)
	if err := model.SetRoPETables(ropeCos, ropeSin, headDim, maxSeqLen); err != nil {
		model.Close()
		return nil, 0, nil, draftModelKindUnknown, true, false, "", fmt.Errorf("configure direct modelir rope tables: %w", err)
	}
	logPhase("rope_tables_done")
	compileFallback := reifiedHasDiagnostic(reified, "compile_fallback")
	compileProfile := reifiedDiagnosticDetail(reified, "compile_fallback", "in-memory compile fallback applied: ")
	compileTier := mlxgoane.ModelIRCompileProfileTier(compileProfile)
	compileNote := mlxgoane.ModelIRCompileProfileNote(compileProfile)
	qwenOnANE, qwenOnMLX := mlxgoane.ModelIRCompileProfileQwenSupport(compileProfile)
	slog.Info(
		"Generated ANE draft model from direct Go->ModelIR lowering",
		"output_mode", outputMode.String(),
		"layers", reified.SelectedLayers,
		"stateful_kv", reified.TransformerConfig.KVCacheState,
		"compile_fallback", compileFallback,
		"compile_tier", compileTier,
		"compile_profile", compileProfile,
		"compile_note", compileNote,
		"qwen_on_ane", qwenOnANE,
		"qwen_on_mlx", qwenOnMLX,
	)
	return model, outputDim, nil, kind, true, compileFallback, compileProfile, nil
}

func reifiedHasDiagnostic(reified mlxgoane.ReifiedMIL, code string) bool {
	for _, d := range reified.Diagnostics {
		if d.Code == code {
			return true
		}
	}
	return false
}

func reifiedDiagnosticDetail(reified mlxgoane.ReifiedMIL, code, prefix string) string {
	for _, d := range reified.Diagnostics {
		if d.Code != code {
			continue
		}
		msg := strings.TrimSpace(d.Message)
		if prefix != "" {
			msg = strings.TrimPrefix(msg, prefix)
		}
		return strings.TrimSpace(msg)
	}
	return ""
}

func draftModelIROutputMode(
	cfg *models.ModelConfig,
	outputMode draftOutputMode,
) (includeLMHead bool, outputDim int, kind draftModelKind, ok bool) {
	if cfg == nil {
		return false, 0, draftModelKindUnknown, false
	}
	switch outputMode {
	case draftOutputModeLMHead:
		return true, cfg.VocabSize, draftModelKindLMHead, true
	case draftOutputModeFinalHidden:
		return false, cfg.HiddenSize, draftModelKindFinalHidden, true
	default:
		return false, 0, draftModelKindUnknown, false
	}
}

func configureGeneratedANEDraftState(
	model *mlxgoane.ANEDraftModel,
	weights mlxgoane.MILTransformerWeights,
	headDim int,
	maxSeqLen int,
	numLayers int,
	numHeads int,
) {
	if model == nil {
		return
	}
	if err := model.SetRoPETables(weights.RopeCos, weights.RopeSin, headDim, maxSeqLen); err != nil {
		slog.Warn("ANE draft: set rope tables failed", "error", err)
	}
	enableKVState := strings.TrimSpace(strings.ToLower(os.Getenv("MLXGO_ANE_DRAFT_ENABLE_KV_STATE")))
	if enableKVState != "1" && enableKVState != "true" && enableKVState != "yes" {
		return
	}
	if model.KVState() != nil {
		return
	}
	if err := model.ConfigureKVState(numLayers, numHeads, headDim, maxSeqLen); err != nil {
		slog.Warn("ANE draft: configure kv state failed", "error", err)
		return
	}
	slog.Info(
		"ANE draft: configured persistent KV state",
		"layers",
		numLayers,
		"heads",
		numHeads,
		"head_dim",
		headDim,
		"max_seq",
		maxSeqLen,
	)
}

func extractVectorFieldOrDefault(parent reflect.Value, fieldNames []string, outDim int, label string) ([]float32, error) {
	if outDim <= 0 {
		return nil, fmt.Errorf("%s: invalid vector dim %d", label, outDim)
	}
	parent = forceReadableValue(parent)
	field := fieldByAnyName(parent, fieldNames...)
	if !field.IsValid() {
		return onesFloat32(outDim), nil
	}
	field = forceReadableValue(field)
	if arr, ok := mlxArrayFromReflectPtr(field); ok && arr != nil && !arr.IsNil() {
		vals, err := copyArrayToFloat32(arr, label)
		if err != nil {
			return nil, err
		}
		switch {
		case len(vals) == outDim:
			return vals, nil
		case len(vals) > outDim && len(vals)%outDim == 0:
			// Some models store per-head vectors repeated across heads.
			return append([]float32(nil), vals[:outDim]...), nil
		case len(vals) < outDim && outDim%len(vals) == 0:
			out := make([]float32, outDim)
			for off := 0; off < outDim; off += len(vals) {
				copy(out[off:off+len(vals)], vals)
			}
			return out, nil
		default:
			return nil, fmt.Errorf("%s: vector len=%d want=%d", label, len(vals), outDim)
		}
	}
	return onesFloat32(outDim), nil
}

func inferVectorFieldDim(parent reflect.Value, fieldNames []string) (int, bool) {
	field := fieldByAnyName(parent, fieldNames...)
	if !field.IsValid() {
		return 0, false
	}
	field = forceReadableValue(field)
	arr, ok := mlxArrayFromReflectPtr(field)
	if !ok || arr == nil || arr.IsNil() {
		return 0, false
	}
	shape := arr.Shape()
	if len(shape) == 0 {
		return 0, false
	}
	total := 1
	for _, d := range shape {
		total *= d
	}
	if total <= 0 {
		return 0, false
	}
	return total, true
}

func extractTransformerLayerNormVectors(
	draftModel models.LanguageModel,
	layer int,
	dim int,
) ([]float32, []float32, error) {
	if dim <= 0 {
		return nil, nil, fmt.Errorf("invalid transformer norm dim %d", dim)
	}
	layerValue, ok := findTransformerLayerByIndex(reflect.ValueOf(draftModel), layer, make(map[uintptr]bool), 0)
	if !ok {
		return onesFloat32(dim), onesFloat32(dim), nil
	}
	layerValue = forceReadableValue(layerValue)
	inputNorm, err := extractVectorFieldOrDefault(
		layerValue,
		[]string{"inputLayernorm", "inputLayerNorm", "inputNorm"},
		dim,
		fmt.Sprintf("layer %d input_layernorm", layer),
	)
	if err != nil {
		return nil, nil, err
	}
	postNorm, err := extractVectorFieldOrDefault(
		layerValue,
		[]string{"postAttentionLayernorm", "postAttentionLayerNorm", "postAttentionNorm", "postAttnNorm"},
		dim,
		fmt.Sprintf("layer %d post_attention_layernorm", layer),
	)
	if err != nil {
		return nil, nil, err
	}
	return inputNorm, postNorm, nil
}

func findTransformerLayerByIndex(
	v reflect.Value,
	targetIndex int,
	seen map[uintptr]bool,
	depth int,
) (reflect.Value, bool) {
	if targetIndex < 0 || !v.IsValid() || depth > 24 {
		return reflect.Value{}, false
	}
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return reflect.Value{}, false
		}
		if v.Kind() == reflect.Ptr {
			ptr := v.Pointer()
			if ptr != 0 {
				if seen[ptr] {
					return reflect.Value{}, false
				}
				seen[ptr] = true
			}
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() {
		return reflect.Value{}, false
	}
	switch v.Kind() {
	case reflect.Slice, reflect.Array:
		if v.Len() > targetIndex {
			candidate := forceReadableValue(v.Index(targetIndex))
			if hasTransformerLayerNormFields(candidate) {
				return candidate, true
			}
		}
		n := v.Len()
		if n > 4 {
			n = 4
		}
		for i := 0; i < n; i++ {
			if out, ok := findTransformerLayerByIndex(forceReadableValue(v.Index(i)), targetIndex, seen, depth+1); ok {
				return out, true
			}
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if out, ok := findTransformerLayerByIndex(forceReadableValue(v.Field(i)), targetIndex, seen, depth+1); ok {
				return out, true
			}
		}
	}
	return reflect.Value{}, false
}

func hasTransformerLayerNormFields(v reflect.Value) bool {
	v = forceReadableValue(v)
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return false
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() || v.Kind() != reflect.Struct {
		return false
	}
	input := fieldByAnyName(v, "inputLayernorm", "inputLayerNorm", "inputNorm")
	post := fieldByAnyName(v, "postAttentionLayernorm", "postAttentionLayerNorm", "postAttentionNorm", "postAttnNorm")
	return input.IsValid() && post.IsValid()
}

func extractFinalNormVector(draftModel models.LanguageModel, dim int) ([]float32, error) {
	arr := findNamedArrayWithLength(
		reflect.ValueOf(draftModel),
		[]string{"norm", "Norm"},
		dim,
		make(map[uintptr]bool),
		0,
	)
	if arr == nil || arr.IsNil() {
		return onesFloat32(dim), nil
	}
	return copyArrayAsFloat32(arr, dim, "final norm")
}

func findNamedArrayWithLength(
	v reflect.Value,
	names []string,
	want int,
	seen map[uintptr]bool,
	depth int,
) *mlx.Array {
	if want <= 0 || !v.IsValid() || depth > 32 {
		return nil
	}
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return nil
		}
		if v.Kind() == reflect.Ptr {
			ptr := v.Pointer()
			if ptr != 0 {
				if seen[ptr] {
					return nil
				}
				seen[ptr] = true
			}
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() {
		return nil
	}
	switch v.Kind() {
	case reflect.Struct:
		for _, name := range names {
			field := forceReadableValue(v.FieldByName(name))
			if arr, ok := mlxArrayFromReflectPtr(field); ok && arr != nil && !arr.IsNil() {
				if arr.Size() == want {
					return arr
				}
			}
		}
		for i := 0; i < v.NumField(); i++ {
			if arr := findNamedArrayWithLength(forceReadableValue(v.Field(i)), names, want, seen, depth+1); arr != nil {
				return arr
			}
		}
	case reflect.Slice, reflect.Array:
		n := v.Len()
		if n > 4 {
			n = 4
		}
		for i := 0; i < n; i++ {
			if arr := findNamedArrayWithLength(forceReadableValue(v.Index(i)), names, want, seen, depth+1); arr != nil {
				return arr
			}
		}
	}
	return nil
}

func buildRoPETables(maxSeqLen, headDim int, theta float64) ([]float32, []float32) {
	if maxSeqLen <= 0 || headDim <= 0 {
		return nil, nil
	}
	if theta <= 0 {
		theta = 10000
	}
	cosTable := make([]float32, maxSeqLen*headDim)
	sinTable := make([]float32, maxSeqLen*headDim)
	for pos := 0; pos < maxSeqLen; pos++ {
		for d := 0; d < headDim; d += 2 {
			pow := float64(d) / float64(headDim)
			angle := float64(pos) / math.Pow(theta, pow)
			c := float32(math.Cos(angle))
			s := float32(math.Sin(angle))
			cosTable[pos*headDim+d] = c
			sinTable[pos*headDim+d] = s
			if d+1 < headDim {
				cosTable[pos*headDim+d+1] = c
				sinTable[pos*headDim+d+1] = s
			}
		}
	}
	return cosTable, sinTable
}

func onesFloat32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func extractLayer0FFNWeights(
	draftModel models.LanguageModel,
) (dim int, hiddenDim int, w1 []float32, w3 []float32, w2 []float32, err error) {
	cfg := draftModel.Config()
	if cfg == nil {
		return 0, 0, nil, nil, nil, fmt.Errorf("draft model config is nil")
	}
	dim = cfg.HiddenSize
	hiddenDim = cfg.IntermediateSize
	if dim <= 0 || hiddenDim <= 0 {
		return 0, 0, nil, nil, nil, fmt.Errorf(
			"invalid draft ffn dims hidden=%d intermediate=%d",
			dim,
			hiddenDim,
		)
	}

	if provider, ok := draftModel.(draftLayerMLPProvider); ok && provider.NumLayers() > 0 {
		if mlp := provider.LayerMLP(0); mlp != nil {
			return extractSwiGLUWeightsFromMLPValue(forceReadableValue(reflect.ValueOf(mlp)), dim, hiddenDim)
		}
	}

	mlpValue, ok := findFirstSwiGLUMLPValue(reflect.ValueOf(draftModel), make(map[uintptr]bool), 0)
	if !ok {
		return 0, 0, nil, nil, nil, fmt.Errorf("could not locate layer-0 MLP with gate/up/down projections")
	}
	return extractSwiGLUWeightsFromMLPValue(mlpValue, dim, hiddenDim)
}

func findFirstSwiGLUMLPValue(v reflect.Value, seen map[uintptr]bool, depth int) (reflect.Value, bool) {
	if !v.IsValid() || depth > 24 {
		return reflect.Value{}, false
	}
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return reflect.Value{}, false
		}
		if v.Kind() == reflect.Ptr {
			ptr := v.Pointer()
			if ptr != 0 {
				if seen[ptr] {
					return reflect.Value{}, false
				}
				seen[ptr] = true
			}
		}
		v = v.Elem()
	}
	if !v.IsValid() {
		return reflect.Value{}, false
	}

	switch v.Kind() {
	case reflect.Struct:
		if hasSwiGLUFields(v) {
			return v, true
		}
		for i := 0; i < v.NumField(); i++ {
			field := forceReadableValue(v.Field(i))
			if out, ok := findFirstSwiGLUMLPValue(field, seen, depth+1); ok {
				return out, true
			}
		}
	case reflect.Slice, reflect.Array:
		n := v.Len()
		if n > 2 {
			n = 2
		}
		for i := 0; i < n; i++ {
			if out, ok := findFirstSwiGLUMLPValue(forceReadableValue(v.Index(i)), seen, depth+1); ok {
				return out, true
			}
		}
	}
	return reflect.Value{}, false
}

func hasSwiGLUFields(v reflect.Value) bool {
	return fieldByAnyName(v, "gateProj", "GateProj").IsValid() &&
		fieldByAnyName(v, "upProj", "UpProj").IsValid() &&
		fieldByAnyName(v, "downProj", "DownProj").IsValid()
}

func extractSwiGLUWeightsFromMLPValue(
	mlpValue reflect.Value,
	dim int,
	hiddenDim int,
) (int, int, []float32, []float32, []float32, error) {
	w1, err := extractProjectionWeightOutIn(mlpValue, []string{"gateProj", "GateProj"}, dim, hiddenDim)
	if err != nil {
		return 0, 0, nil, nil, nil, fmt.Errorf("extract gate_proj: %w", err)
	}
	w3, err := extractProjectionWeightOutIn(mlpValue, []string{"upProj", "UpProj"}, dim, hiddenDim)
	if err != nil {
		return 0, 0, nil, nil, nil, fmt.Errorf("extract up_proj: %w", err)
	}
	w2, err := extractProjectionWeightOutIn(mlpValue, []string{"downProj", "DownProj"}, hiddenDim, dim)
	if err != nil {
		return 0, 0, nil, nil, nil, fmt.Errorf("extract down_proj: %w", err)
	}
	return dim, hiddenDim, w1, w3, w2, nil
}

func extractProjectionWeightOutIn(
	mlpValue reflect.Value,
	fieldNames []string,
	inDim int,
	outDim int,
) ([]float32, error) {
	field, fieldName := fieldByAnyNameWithName(mlpValue, fieldNames...)
	if !field.IsValid() {
		return nil, fmt.Errorf("missing field %v", fieldNames)
	}
	field = forceReadableValue(field)
	if !field.IsValid() {
		return nil, fmt.Errorf("field %v is invalid", fieldNames)
	}

	for field.Kind() == reflect.Interface {
		if field.IsNil() {
			return nil, fmt.Errorf("field %v is nil", fieldNames)
		}
		field = forceReadableValue(field.Elem())
	}

	if array, ok := mlxArrayFromReflectPtr(field); ok {
		outIn, err := projectionArrayToOutIn(array, inDim, outDim, fmt.Sprintf("%v", fieldNames))
		if err == nil {
			return outIn, nil
		}
		dequantized, dqErr := dequantizeProjectionWeightFromStruct(mlpValue, fieldName, array)
		if dqErr != nil {
			return nil, fmt.Errorf("%w (dequantize fallback failed: %v)", err, dqErr)
		}
		defer dequantized.Free()
		return projectionArrayToOutIn(dequantized, inDim, outDim, fmt.Sprintf("%v", fieldNames))
	}

	if layer := reflectValueAsLinearLayer(field); layer != nil {
		return linearLayerWeightOutIn(layer, inDim, outDim, fmt.Sprintf("%v", fieldNames))
	}
	return nil, fmt.Errorf("field %v has unsupported type %s", fieldNames, field.Type())
}

func selectDraftMILLayerIndices(
	attnProvider draftLayerAttentionProvider,
	mlpProvider draftLayerMLPProvider,
	limit int,
) ([]int, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("invalid layer limit %d", limit)
	}
	maxLayers := attnProvider.NumLayers()
	if mlp := mlpProvider.NumLayers(); mlp < maxLayers {
		maxLayers = mlp
	}
	selected := make([]int, 0, limit)
	for i := 0; i < maxLayers; i++ {
		attn := attnProvider.LayerAttention(i)
		mlp := mlpProvider.LayerMLP(i)
		if attn == nil || mlp == nil {
			continue
		}
		attnValue := forceReadableValue(reflect.ValueOf(attn))
		if !fieldByAnyName(attnValue, "qProj", "QProj").IsValid() {
			continue
		}
		if !fieldByAnyName(attnValue, "kProj", "KProj").IsValid() {
			continue
		}
		if !fieldByAnyName(attnValue, "vProj", "VProj").IsValid() {
			continue
		}
		if !fieldByAnyName(attnValue, "oProj", "OProj").IsValid() {
			continue
		}
		selected = append(selected, i)
		if len(selected) == limit {
			break
		}
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("no transformer layers expose q/k/v/o projections for MIL extraction")
	}
	return selected, nil
}

type attentionProjectionInfo struct {
	HeadDim      int
	AttnDim      int
	KVDim        int
	WidenedQProj bool
}

func inferAttentionProjectionInfo(
	attn interface{},
	attnValue reflect.Value,
	dim int,
	numHeads int,
	numKVHeads int,
	configuredHeadDim int,
) (attentionProjectionInfo, error) {
	info := attentionProjectionInfo{}
	if dim <= 0 || numHeads <= 0 || numKVHeads <= 0 {
		return info, fmt.Errorf("invalid attention dims hidden=%d heads=%d kvHeads=%d", dim, numHeads, numKVHeads)
	}

	headDim := configuredHeadDim
	var qOutDim int
	var kOutDim int
	var oInDim int
	if dimsProvider, ok := attn.(attentionProjectionDimsProvider); ok {
		if _, out, ok := dimsProvider.ProjectionDims("q_proj"); ok && out > 0 {
			qOutDim = out
		}
		if _, out, ok := dimsProvider.ProjectionDims("k_proj"); ok && out > 0 {
			kOutDim = out
		}
		if in, _, ok := dimsProvider.ProjectionDims("o_proj"); ok && in > 0 {
			oInDim = in
		}
	}
	if qOutDim <= 0 {
		if out, err := inferProjectionOutDim(attnValue, []string{"qProj", "QProj"}, dim); err == nil && out > 0 {
			qOutDim = out
		}
	}
	if kOutDim <= 0 {
		if out, err := inferProjectionOutDim(attnValue, []string{"kProj", "KProj"}, dim); err == nil && out > 0 {
			kOutDim = out
		}
	}
	if headDim <= 0 {
		if normDim, ok := inferVectorFieldDim(attnValue, []string{"qNorm", "QNorm"}); ok {
			headDim = normDim
		}
	}
	if headDim <= 0 {
		if normDim, ok := inferVectorFieldDim(attnValue, []string{"kNorm", "KNorm"}); ok {
			headDim = normDim
		}
	}
	if headDim <= 0 {
		switch {
		case oInDim > 0 && oInDim%numHeads == 0:
			headDim = oInDim / numHeads
		case qOutDim > 0 && qOutDim%(2*numHeads) == 0:
			headDim = qOutDim / (2 * numHeads)
		case qOutDim > 0 && qOutDim%numHeads == 0:
			headDim = qOutDim / numHeads
		case dim%numHeads == 0:
			headDim = dim / numHeads
		default:
			return info, fmt.Errorf("derive head dim from q_proj: hidden=%d heads=%d qOut=%d oIn=%d", dim, numHeads, qOutDim, oInDim)
		}
	}

	attnDim := oInDim
	if attnDim <= 0 {
		attnDim = numHeads * headDim
	}
	widenedQProj := qOutDim > 0 && qOutDim == 2*attnDim
	if attnDim <= 0 && qOutDim > 0 {
		if qOutDim%(2*numHeads) == 0 {
			attnDim = qOutDim / 2
			widenedQProj = true
		} else {
			attnDim = qOutDim
		}
	}

	kvDim := numKVHeads * headDim
	if kOutDim > 0 {
		kvDim = kOutDim
	}
	info.HeadDim = headDim
	info.AttnDim = attnDim
	info.KVDim = kvDim
	info.WidenedQProj = widenedQProj
	return info, nil
}

func inferProjectionOutDim(parent reflect.Value, fieldNames []string, inDim int) (int, error) {
	if inDim <= 0 {
		return 0, fmt.Errorf("invalid in dim %d for fields %v", inDim, fieldNames)
	}
	field := fieldByAnyName(parent, fieldNames...)
	if !field.IsValid() {
		return 0, fmt.Errorf("missing field %v", fieldNames)
	}
	field = forceReadableValue(field)
	if arr, ok := mlxArrayFromReflectPtr(field); ok && arr != nil && !arr.IsNil() {
		return inferOutDimFromArray(arr, inDim, fmt.Sprintf("%v", fieldNames))
	}
	if layer := reflectValueAsLinearLayer(field); layer != nil {
		weight := layer.Weight()
		if weight == nil || weight.IsNil() {
			return 0, fmt.Errorf("%v: weight is nil", fieldNames)
		}
		inOut := weight
		var dequantized *mlx.Array
		var err error
		if layer.IsQuantized() {
			dequantized, err = dequantizeLinearLayerWeightToInOut(layer, weight)
			if err != nil {
				return 0, fmt.Errorf("%v: dequantize weight: %w", fieldNames, err)
			}
			inOut = dequantized
			defer dequantized.Free()
		}
		return inferOutDimFromArray(inOut, inDim, fmt.Sprintf("%v", fieldNames))
	}
	return 0, fmt.Errorf("unsupported field type %s", field.Type())
}

func inferOutDimFromArray(arr *mlx.Array, inDim int, label string) (int, error) {
	if arr == nil || arr.IsNil() {
		return 0, fmt.Errorf("%s: array is nil", label)
	}
	shape := arr.Shape()
	if len(shape) == 2 {
		if shape[0] == inDim && shape[1] > 0 {
			return shape[1], nil
		}
		if shape[1] == inDim && shape[0] > 0 {
			return shape[0], nil
		}
	}
	total := 1
	for _, d := range shape {
		total *= d
	}
	if total <= 0 || total%inDim != 0 {
		return 0, fmt.Errorf("%s: shape=%v cannot infer out dim from in=%d", label, shape, inDim)
	}
	return total / inDim, nil
}

func extractProjectionBiasOut(
	parent reflect.Value,
	projFieldNames []string,
	biasFieldNames []string,
	outDim int,
) ([]float32, error) {
	if outDim <= 0 {
		return nil, fmt.Errorf("invalid bias dim %d for fields %v", outDim, projFieldNames)
	}
	parent = forceReadableValue(parent)
	if parent.IsValid() {
		if bias := fieldByAnyName(parent, biasFieldNames...); bias.IsValid() {
			if arr, ok := mlxArrayFromReflectPtr(forceReadableValue(bias)); ok && arr != nil && !arr.IsNil() {
				vals, err := copyArrayToFloat32(arr, fmt.Sprintf("bias %v", biasFieldNames))
				if err != nil {
					return nil, err
				}
				return fitVectorToDim(vals, outDim, fmt.Sprintf("bias %v", biasFieldNames))
			}
		}
	}

	projField := fieldByAnyName(parent, projFieldNames...)
	if projField.IsValid() {
		if layer := reflectValueAsLinearLayer(forceReadableValue(projField)); layer != nil {
			if arr := linearLayerBiasArray(layer); arr != nil && !arr.IsNil() {
				vals, err := copyArrayToFloat32(arr, fmt.Sprintf("bias %v", projFieldNames))
				if err != nil {
					return nil, err
				}
				return fitVectorToDim(vals, outDim, fmt.Sprintf("bias %v", projFieldNames))
			}
		}
	}

	return make([]float32, outDim), nil
}

func extractSplitProjectionWeightBias(
	parent reflect.Value,
	projFieldNames []string,
	biasFieldNames []string,
	inDim int,
	outDim int,
) (mainW []float32, extraW []float32, mainB []float32, extraB []float32, err error) {
	if outDim <= 0 {
		return nil, nil, nil, nil, fmt.Errorf("invalid split projection out dim %d for %v", outDim, projFieldNames)
	}
	fullW, err := extractProjectionWeightOutIn(parent, projFieldNames, inDim, 2*outDim)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	fullB, err := extractProjectionBiasOut(parent, projFieldNames, biasFieldNames, 2*outDim)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	split := outDim * inDim
	if len(fullW) != 2*split {
		return nil, nil, nil, nil, fmt.Errorf("%v split weight len=%d want=%d", projFieldNames, len(fullW), 2*split)
	}
	if len(fullB) != 2*outDim {
		return nil, nil, nil, nil, fmt.Errorf("%v split bias len=%d want=%d", projFieldNames, len(fullB), 2*outDim)
	}
	return append([]float32(nil), fullW[:split]...),
		append([]float32(nil), fullW[split:2*split]...),
		append([]float32(nil), fullB[:outDim]...),
		append([]float32(nil), fullB[outDim:2*outDim]...),
		nil
}

func linearLayerBiasArray(layer models.LinearLayer) *mlx.Array {
	v := forceReadableValue(reflect.ValueOf(layer))
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return nil
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() || v.Kind() != reflect.Struct {
		return nil
	}
	for _, name := range []string{"bias", "Bias", "addBias", "AddBias"} {
		if arr := arrayFieldByName(v, name); arr != nil && !arr.IsNil() {
			return arr
		}
	}
	return nil
}

func expandGQAProjectionRows(
	rowsByIn []float32,
	inDim int,
	numHeads int,
	numKVHeads int,
	headDim int,
) ([]float32, error) {
	if numKVHeads <= 0 || numHeads <= 0 || headDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf(
			"invalid gqa dims heads=%d kvHeads=%d headDim=%d inDim=%d",
			numHeads,
			numKVHeads,
			headDim,
			inDim,
		)
	}
	if numKVHeads == numHeads {
		return append([]float32(nil), rowsByIn...), nil
	}
	kvRows := numKVHeads * headDim
	if got, want := len(rowsByIn), kvRows*inDim; got != want {
		return nil, fmt.Errorf("gqa projection len=%d want=%d", got, want)
	}
	out := make([]float32, numHeads*headDim*inDim)
	for h := 0; h < numHeads; h++ {
		kvHead := h * numKVHeads / numHeads
		for d := 0; d < headDim; d++ {
			srcRow := kvHead*headDim + d
			dstRow := h*headDim + d
			copy(
				out[dstRow*inDim:(dstRow+1)*inDim],
				rowsByIn[srcRow*inDim:(srcRow+1)*inDim],
			)
		}
	}
	return out, nil
}

func expandGQABias(
	bias []float32,
	numHeads int,
	numKVHeads int,
	headDim int,
) ([]float32, error) {
	if numKVHeads <= 0 || numHeads <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("invalid gqa bias dims heads=%d kvHeads=%d headDim=%d", numHeads, numKVHeads, headDim)
	}
	if numKVHeads == numHeads {
		return append([]float32(nil), bias...), nil
	}
	kvDim := numKVHeads * headDim
	if got, want := len(bias), kvDim; got != want {
		return nil, fmt.Errorf("gqa bias len=%d want=%d", got, want)
	}
	out := make([]float32, numHeads*headDim)
	for h := 0; h < numHeads; h++ {
		kvHead := h * numKVHeads / numHeads
		copy(out[h*headDim:(h+1)*headDim], bias[kvHead*headDim:(kvHead+1)*headDim])
	}
	return out, nil
}

func extractLMHeadWeightsOutIn(
	draftModel models.LanguageModel,
	dim int,
	vocabSize int,
) ([]float32, []float32, error) {
	field, ok := findFieldByNamesRecursive(
		reflect.ValueOf(draftModel),
		map[string]struct{}{"lmHead": {}, "LMHead": {}},
		make(map[uintptr]bool),
		0,
	)
	if ok {
		field = forceReadableValue(field)
		for field.Kind() == reflect.Interface {
			if field.IsNil() {
				break
			}
			field = forceReadableValue(field.Elem())
		}
		if field.Kind() == reflect.Ptr && field.IsNil() {
			ok = false
		}
		if ok {
			if arr, ok := mlxArrayFromReflectPtr(field); ok && arr != nil && !arr.IsNil() {
				w, err := lmHeadArrayToOutIn(arr, dim, vocabSize)
				if err != nil {
					return nil, nil, err
				}
				return w, make([]float32, vocabSize), nil
			}
			if layer := reflectValueAsLinearLayer(field); layer != nil {
				w, err := linearLayerWeightOutIn(layer, dim, vocabSize, "lm_head")
				if err != nil {
					return nil, nil, err
				}
				b, err := extractProjectionBiasOut(reflect.ValueOf(struct {
					LMHead models.LinearLayer
				}{LMHead: layer}), []string{"LMHead"}, nil, vocabSize)
				if err != nil {
					return nil, nil, err
				}
				return w, b, nil
			}
		}
	}

	hiddenDim, embVocab, embeddings, err := extractDraftEmbeddings(draftModel)
	if err == nil && hiddenDim == dim && embVocab == vocabSize && len(embeddings) == dim*vocabSize {
		return append([]float32(nil), embeddings...), make([]float32, vocabSize), nil
	}
	if lookupModel, ok := draftModel.(draftEmbeddingLookupModel); ok {
		materialized, matErr := materializeEmbeddingsViaLookup(lookupModel, dim, vocabSize)
		if matErr == nil && len(materialized) == dim*vocabSize {
			return materialized, make([]float32, vocabSize), nil
		}
	}

	if !ok {
		return nil, nil, fmt.Errorf("extract lm_head: could not locate lmHead field and embeddings fallback unavailable")
	}
	elemPkg := ""
	if field.IsValid() && field.Type().Kind() == reflect.Ptr && field.Type().Elem().PkgPath() != "" {
		elemPkg = field.Type().Elem().PkgPath()
	}
	return nil, nil, fmt.Errorf("extract lm_head: unsupported field type %s (elem_pkg=%s)", field.Type(), elemPkg)
}

func lmHeadArrayToOutIn(arr *mlx.Array, dim int, vocabSize int) ([]float32, error) {
	if arr == nil || arr.IsNil() {
		return nil, fmt.Errorf("lm_head array is nil")
	}
	shape := arr.Shape()
	want := dim * vocabSize
	vals, err := copyArrayAsFloat32(arr, want, "lm_head")
	if err != nil {
		return nil, err
	}
	if len(shape) == 2 {
		if shape[0] == vocabSize && shape[1] == dim {
			return vals, nil
		}
		if shape[0] == dim && shape[1] == vocabSize {
			return transposeRowMajor(vals, dim, vocabSize), nil
		}
	}
	return transposeRowMajor(vals, dim, vocabSize), nil
}

func findFieldByNamesRecursive(
	v reflect.Value,
	names map[string]struct{},
	seen map[uintptr]bool,
	depth int,
) (reflect.Value, bool) {
	if !v.IsValid() || depth > 32 {
		return reflect.Value{}, false
	}
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return reflect.Value{}, false
		}
		if v.Kind() == reflect.Ptr {
			ptr := v.Pointer()
			if ptr != 0 {
				if seen[ptr] {
					return reflect.Value{}, false
				}
				seen[ptr] = true
			}
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() {
		return reflect.Value{}, false
	}
	switch v.Kind() {
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			if _, ok := names[t.Field(i).Name]; ok {
				field := forceReadableValue(v.Field(i))
				if field.IsValid() && (field.Kind() == reflect.Interface || field.Kind() == reflect.Ptr) && field.IsNil() {
					continue
				}
				return field, true
			}
		}
		for i := 0; i < v.NumField(); i++ {
			if field, ok := findFieldByNamesRecursive(forceReadableValue(v.Field(i)), names, seen, depth+1); ok {
				return field, true
			}
		}
	case reflect.Slice, reflect.Array:
		n := v.Len()
		if n > 4 {
			n = 4
		}
		for i := 0; i < n; i++ {
			if field, ok := findFieldByNamesRecursive(forceReadableValue(v.Index(i)), names, seen, depth+1); ok {
				return field, true
			}
		}
	}
	return reflect.Value{}, false
}

func reflectValueAsLinearLayer(v reflect.Value) models.LinearLayer {
	if !v.IsValid() {
		return nil
	}
	if v.CanInterface() {
		if layer, ok := v.Interface().(models.LinearLayer); ok {
			return layer
		}
	}
	if v.Kind() != reflect.Ptr && v.CanAddr() {
		addr := v.Addr()
		if addr.CanInterface() {
			if layer, ok := addr.Interface().(models.LinearLayer); ok {
				return layer
			}
		}
	}
	return nil
}

func linearLayerWeightOutIn(
	layer models.LinearLayer,
	inDim int,
	outDim int,
	label string,
) ([]float32, error) {
	if layer == nil {
		return nil, fmt.Errorf("%s: linear layer is nil", label)
	}
	weight := layer.Weight()
	if weight == nil || weight.IsNil() {
		return nil, fmt.Errorf("%s: weight is nil", label)
	}

	inOut := weight
	var dequantized *mlx.Array
	var err error
	if layer.IsQuantized() {
		dequantized, err = dequantizeLinearLayerWeightToInOut(layer, weight)
		if err != nil {
			return nil, fmt.Errorf("%s: dequantize: %w", label, err)
		}
		inOut = dequantized
		defer dequantized.Free()
	}
	return arrayInOutToOutIn(inOut, inDim, outDim, label)
}

func dequantizeLinearLayerWeightToInOut(layer models.LinearLayer, weight *mlx.Array) (*mlx.Array, error) {
	v := forceReadableValue(reflect.ValueOf(layer))
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return nil, fmt.Errorf("linear layer is nil")
		}
		v = forceReadableValue(v.Elem())
	}
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("unexpected quantized linear kind %s", v.Kind())
	}

	scales := arrayFieldByName(v, "scales")
	if scales == nil || scales.IsNil() {
		return nil, fmt.Errorf("quantized linear scales are nil")
	}
	biases := arrayFieldByName(v, "biases")
	groupSize := int(intFieldByName(v, "groupSize", 64))
	bits := int(intFieldByName(v, "bits", 4))
	mode := quantModeFieldByName(v, "mode", mlx.QuantizationModeAffine)

	dequantized, err := mlx.Dequantize(
		weight,
		scales,
		biases,
		mlx.OptInt(groupSize),
		mlx.OptInt(bits),
		mode,
		mlx.OptionalDtype{Value: mlx.Float32, Has_value: true},
		nil,
	)
	if err != nil {
		return nil, err
	}
	inOut, err := mlx.Transpose(dequantized, nil)
	dequantized.Free()
	if err != nil {
		return nil, fmt.Errorf("transpose dequantized weight: %w", err)
	}
	return inOut, nil
}

func arrayFieldByName(v reflect.Value, name string) *mlx.Array {
	field := forceReadableValue(v.FieldByName(name))
	array, ok := mlxArrayFromReflectPtr(field)
	if !ok {
		return nil
	}
	return array
}

func intFieldByName(v reflect.Value, name string, fallback int64) int64 {
	field := forceReadableValue(v.FieldByName(name))
	if !field.IsValid() {
		return fallback
	}
	switch field.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return field.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return int64(field.Uint())
	default:
		return fallback
	}
}

func quantModeFieldByName(v reflect.Value, name string, fallback mlx.QuantizationMode) mlx.QuantizationMode {
	field := forceReadableValue(v.FieldByName(name))
	if !field.IsValid() {
		return fallback
	}
	if field.CanInterface() {
		if mode, ok := field.Interface().(mlx.QuantizationMode); ok {
			return mode
		}
	}
	switch field.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return mlx.QuantizationMode(field.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return mlx.QuantizationMode(field.Uint())
	default:
		return fallback
	}
}

func arrayInOutToOutIn(arr *mlx.Array, inDim int, outDim int, label string) ([]float32, error) {
	return projectionArrayToOutIn(arr, inDim, outDim, label)
}

func copyArrayAsFloat32(arr *mlx.Array, want int, label string) ([]float32, error) {
	vals, err := copyArrayToFloat32(arr, label)
	if err != nil {
		return nil, err
	}
	if len(vals) != want {
		return nil, fmt.Errorf("%s: weight len=%d want=%d", label, len(vals), want)
	}
	return vals, nil
}

func copyArrayToFloat32(arr *mlx.Array, label string) ([]float32, error) {
	if arr == nil || arr.IsNil() {
		return nil, fmt.Errorf("%s: array is nil", label)
	}
	source := arr
	var converted *mlx.Array
	if arr.Dtype() != mlx.Float32 {
		var err error
		converted, err = mlx.Astype(arr, mlx.Float32, nil)
		if err != nil {
			return nil, fmt.Errorf("%s: cast to float32: %w", label, err)
		}
		source = converted
		defer converted.Free()
	}
	vals, err := mlx.ToSlice[float32](source)
	if err != nil {
		return nil, fmt.Errorf("%s: copy tensor values: %w", label, err)
	}
	return vals, nil
}

func projectionArrayToOutIn(arr *mlx.Array, inDim int, outDim int, label string) ([]float32, error) {
	if arr == nil || arr.IsNil() {
		return nil, fmt.Errorf("%s: weight array is nil", label)
	}
	want := inDim * outDim
	vals, err := copyArrayToFloat32(arr, label)
	if err != nil {
		return nil, err
	}
	if len(vals) != want && len(vals) != 2*want {
		return nil, fmt.Errorf("%s: weight len=%d want=%d", label, len(vals), want)
	}
	shape := arr.Shape()
	if len(shape) == 2 {
		switch {
		case shape[0] == outDim && shape[1] == inDim:
			return vals, nil
		case shape[0] == inDim && shape[1] == outDim:
			return transposeRowMajor(vals, inDim, outDim), nil
		case shape[0] == 2*outDim && shape[1] == inDim:
			return append([]float32(nil), vals[:outDim*inDim]...), nil
		case shape[0] == inDim && shape[1] == 2*outDim:
			full := transposeRowMajor(vals, inDim, 2*outDim)
			return append([]float32(nil), full[:outDim*inDim]...), nil
		}
	}
	if len(vals) == 2*want {
		return append([]float32(nil), vals[:want]...), nil
	}
	return transposeRowMajor(vals, inDim, outDim), nil
}

func fitVectorToDim(vals []float32, outDim int, label string) ([]float32, error) {
	switch {
	case len(vals) == outDim:
		return vals, nil
	case len(vals) == 2*outDim:
		return append([]float32(nil), vals[:outDim]...), nil
	case len(vals) > outDim && len(vals)%outDim == 0:
		return append([]float32(nil), vals[:outDim]...), nil
	default:
		return nil, fmt.Errorf("%s len=%d want=%d", label, len(vals), outDim)
	}
}

func transposeRowMajor(values []float32, rows int, cols int) []float32 {
	out := make([]float32, len(values))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = values[r*cols+c]
		}
	}
	return out
}

func fieldByAnyName(v reflect.Value, names ...string) reflect.Value {
	field, _ := fieldByAnyNameWithName(v, names...)
	return field
}

func fieldByAnyNameWithName(v reflect.Value, names ...string) (reflect.Value, string) {
	v = forceReadableValue(v)
	for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr) {
		if v.IsNil() {
			return reflect.Value{}, ""
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() || v.Kind() != reflect.Struct {
		return reflect.Value{}, ""
	}
	for _, name := range names {
		if name == "" {
			continue
		}
		field := v.FieldByName(name)
		if field.IsValid() {
			return field, name
		}
	}
	return reflect.Value{}, ""
}

func dequantizeProjectionWeightFromStruct(parent reflect.Value, fieldName string, weight *mlx.Array) (*mlx.Array, error) {
	if fieldName == "" {
		return nil, fmt.Errorf("missing projection field name")
	}
	parent = forceReadableValue(parent)
	for parent.IsValid() && (parent.Kind() == reflect.Interface || parent.Kind() == reflect.Ptr) {
		if parent.IsNil() {
			return nil, fmt.Errorf("projection parent is nil")
		}
		parent = forceReadableValue(parent.Elem())
	}
	if !parent.IsValid() || parent.Kind() != reflect.Struct {
		return nil, fmt.Errorf("projection parent kind=%s is not a struct", parent.Kind())
	}

	scales := arrayFieldByName(parent, fieldName+"Scales")
	if scales == nil || scales.IsNil() {
		return nil, fmt.Errorf("projection %s scales are nil", fieldName)
	}
	biases := arrayFieldByName(parent, fieldName+"Biases")
	groupSize := int(intFieldByName(parent, "groupSize", 64))
	bits := int(intFieldByName(parent, "bits", 4))
	mode := quantModeFieldByName(parent, "quantMode", mlx.QuantizationModeAffine)

	dequantized, err := mlx.Dequantize(
		weight,
		scales,
		biases,
		mlx.OptInt(groupSize),
		mlx.OptInt(bits),
		mode,
		mlx.OptionalDtype{Value: mlx.Float32, Has_value: true},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("dequantize projection %s: %w", fieldName, err)
	}
	return dequantized, nil
}

func mlxArrayFromReflectPtr(v reflect.Value) (*mlx.Array, bool) {
	if !v.IsValid() || v.Kind() != reflect.Ptr || v.IsNil() || v.Type() != reflect.TypeOf(&mlx.Array{}) {
		return nil, false
	}
	if v.CanInterface() {
		arr, ok := v.Interface().(*mlx.Array)
		if ok {
			return arr, true
		}
	}
	// Unexported pointer fields may not allow Interface/Addr, but Pointer is still available.
	raw := (*mlx.Array)(unsafe.Pointer(v.Pointer()))
	if raw != nil {
		return raw, true
	}
	if v.CanAddr() {
		raw := *(**mlx.Array)(unsafe.Pointer(v.UnsafeAddr()))
		if raw != nil {
			return raw, true
		}
	}
	return nil, false
}

func newReferenceOnlyDraftModelDrafter(draftModel models.LanguageModel) (decode.Drafter, error) {
	newCache, err := cacheFactoryForLanguageModel(draftModel)
	if err != nil {
		return nil, err
	}
	cache := newCache()
	if cache == nil {
		return nil, fmt.Errorf("reference-only draft cache is nil")
	}

	var multiCache *models.MultiLayerCache
	if mc, ok := cache.(*models.MultiLayerCache); ok {
		multiCache = mc
	} else {
		multiCache = models.NewMultiLayerCacheFromList([]models.Cache{cache})
	}

	forward := func(input *mlx.Array, c models.Cache) (*mlx.Array, models.Cache, error) {
		return draftModel.Forward(input, c)
	}
	sampleGreedy := func(logits *mlx.Array) (int32, error) {
		tokArr, err := sample.Token(context.Background(), logits, 0, 1.0, 0, 0)
		if err != nil {
			return 0, err
		}
		defer tokArr.Free()
		if err := mlx.Eval(tokArr); err != nil {
			return 0, fmt.Errorf("eval sampled token: %w", err)
		}
		v, err := tokArr.Item()
		if err != nil {
			return 0, fmt.Errorf("sampled token item: %w", err)
		}
		switch tv := v.(type) {
		case int32:
			return tv, nil
		case int64:
			return int32(tv), nil
		case uint32:
			return int32(tv), nil
		default:
			return 0, fmt.Errorf("unexpected sampled token type %T", v)
		}
	}
	lazyGreedy := func(logits *mlx.Array) (*mlx.Array, error) {
		return sample.Token(context.Background(), logits, 0, 1.0, 0, 0)
	}

	d := decode.NewDraftModelDrafter(forward, multiCache, sampleGreedy)
	d.SetLazySampler(lazyGreedy)
	return d, nil
}

type aneDraftDrafter struct {
	model               *mlxgoane.ANEDraftModel
	hiddenDim           int
	vocabSize           int
	outputDim           int
	embeddings          []float32
	lmProj              []float32
	lmBias              []float32
	tapLayout           *mlxgoane.TransformerForwardTapLayout
	tapRecorder         *aneDraftTapRecorder
	reference           *aneDraftReferenceRunner
	useRefLogits        bool
	autoRefGuard        bool
	lowConfidence       bool
	loggedDraftCap      bool
	preferReferenceOnly bool
	referenceOnly       bool
	guardTriggered      bool
	mode                string
	strategy            draftStrategy
	modelKind           draftModelKind
	maxLayers           int
	directModelIR       bool
	compileFallback     bool
	compileProfile      string
	widenedAttention    bool
	ffnFallbackUsed     bool
	draftedCount        int
	rewoundCount        int
	stateCorrections    int
	prefillDuration     time.Duration
	draftCalls          int
	draftDuration       time.Duration
	ensureTokenCalls    int
	ensureTokenDuration time.Duration
	feedTokenCalls      int
	feedTokenDuration   time.Duration
	embedLookups        int
	embedDuration       time.Duration
	modelEvalCalls      int
	modelEvalDuration   time.Duration
	nextTokenCalls      int
	nextTokenDuration   time.Duration
	advanceDecodeCalls  int
	advanceDecodeDur    time.Duration
	rewindCalls         int
	rewindDuration      time.Duration
	rebuildCount        int
	rebuildDuration     time.Duration
	modelSnapshots      []*mlxgoane.DecodeStateSnapshot
	embedToken          func(tokenID int) ([]float32, error)
	history             []int
	lastLogits          []float32
}

func (d *aneDraftDrafter) Draft(lastToken *mlx.Array, n int) ([]int32, error) {
	if d == nil || (d.model == nil && d.reference == nil) {
		return nil, fmt.Errorf("ane draft drafter is not initialized")
	}
	start := time.Now()
	d.draftCalls++
	defer func() {
		d.draftDuration += time.Since(start)
	}()
	if n > 1 &&
		!d.referenceOnly &&
		d.lowConfidence &&
		envTruthy("MLXGO_ANE_DRAFT_ENABLE_LOW_CONFIDENCE_DRAFT_CAP") {
		if !d.loggedDraftCap {
			slog.Info(
				"ANE draft low-confidence path: capping draft width to 1 (opt-in)",
				"env_enable",
				"MLXGO_ANE_DRAFT_ENABLE_LOW_CONFIDENCE_DRAFT_CAP",
			)
			d.loggedDraftCap = true
		}
		n = 1
	}
	d.maybeEnableReferenceOnly()
	tokenID, err := tokenIDFromArray(lastToken)
	if err != nil {
		return nil, fmt.Errorf("read last token id: %w", err)
	}

	if err := d.ensureTokenState(tokenID); err != nil {
		return nil, fmt.Errorf("advance drafter state: %w", err)
	}
	if n <= 0 {
		return nil, nil
	}

	out := make([]int32, 0, n)
	for i := 0; i < n; i++ {
		if len(d.lastLogits) == 0 {
			return nil, fmt.Errorf("draft logits missing after state advance")
		}
		nextStart := time.Now()
		nextID, err := d.nextTokenFromOutput(d.lastLogits)
		d.nextTokenCalls++
		d.nextTokenDuration += time.Since(nextStart)
		if err != nil {
			return nil, fmt.Errorf("select next token from output: %w", err)
		}
		out = append(out, int32(nextID))
		if err := d.captureDraftSnapshot(); err != nil {
			slog.Warn("ANE draft snapshot capture failed", "error", err, "step", i)
		}
		if err := d.feedToken(nextID); err != nil {
			return nil, fmt.Errorf("feed drafted token %d at step %d: %w", nextID, i, err)
		}
	}
	d.draftedCount += len(out)
	return out, nil
}

func (d *aneDraftDrafter) Rewind(n int) {
	if d == nil || n <= 0 {
		return
	}
	start := time.Now()
	d.rewindCalls++
	defer func() {
		d.rewindDuration += time.Since(start)
	}()
	d.rewoundCount += n
	rewindCount := n
	if n >= len(d.history) {
		rewindCount = len(d.history)
		d.history = d.history[:0]
	} else {
		d.history = d.history[:len(d.history)-n]
	}
	d.lastLogits = nil
	if rewindCount <= 0 {
		return
	}
	if d.model == nil {
		if d.needsReferenceShadow() {
			if err := d.reference.Rewind(rewindCount); err != nil {
				if syncErr := d.syncReferenceState(); syncErr != nil {
					slog.Warn("ANE draft rewind reference sync failed", "error", syncErr, "rewind", n)
				}
			}
		}
		return
	}
	if err := d.restoreDraftSnapshot(rewindCount); err == nil {
		if d.needsReferenceShadow() {
			if err := d.reference.Rewind(rewindCount); err != nil {
				slog.Warn("ANE draft rewind reference fast path failed after snapshot restore; rebuilding state", "error", err, "rewind", rewindCount)
				if rebuildErr := d.rebuildState(); rebuildErr != nil {
					slog.Warn("ANE draft rewind rebuild failed", "error", rebuildErr, "rewind", n)
				}
			}
		}
		return
	}
	if err := d.model.RewindDecodePosition(rewindCount); err != nil {
		slog.Warn("ANE draft rewind model fast path failed; rebuilding state", "error", err, "rewind", rewindCount)
		if rebuildErr := d.rebuildState(); rebuildErr != nil {
			slog.Warn("ANE draft rewind rebuild failed", "error", rebuildErr, "rewind", n)
		}
		return
	}
	if d.needsReferenceShadow() {
		if err := d.reference.Rewind(rewindCount); err != nil {
			slog.Warn("ANE draft rewind reference fast path failed; rebuilding state", "error", err, "rewind", rewindCount)
			if rebuildErr := d.rebuildState(); rebuildErr != nil {
				slog.Warn("ANE draft rewind rebuild failed", "error", rebuildErr, "rewind", n)
			}
		}
	}
}

func (d *aneDraftDrafter) Prefill(prompt *mlx.Array) error {
	if d == nil || (d.model == nil && d.reference == nil) {
		return fmt.Errorf("ane draft drafter is not initialized")
	}
	start := time.Now()
	defer func() {
		d.prefillDuration += time.Since(start)
	}()
	if d.model != nil {
		if err := d.model.Reset(); err != nil {
			return fmt.Errorf("reset model before prefill: %w", err)
		}
	}
	ids, err := tokenIDsFromArray(prompt)
	if err != nil {
		return fmt.Errorf("extract prompt token ids: %w", err)
	}
	d.history = d.history[:0]
	d.lastLogits = nil
	d.modelSnapshots = nil
	d.referenceOnly = d.preferReferenceOnly
	d.guardTriggered = false
	d.draftedCount = 0
	d.rewoundCount = 0
	d.stateCorrections = 0
	if d.tapRecorder != nil {
		d.tapRecorder.Reset()
	}
	if d.reference != nil {
		d.reference.Reset()
	}
	for _, id := range ids {
		if err := d.feedToken(id); err != nil {
			return fmt.Errorf("prefill feed token %d: %w", id, err)
		}
	}
	return nil
}

func (d *aneDraftDrafter) Reset() {
	if d == nil {
		return
	}
	if d.model != nil {
		if err := d.model.Reset(); err != nil {
			slog.Warn("ANE draft model reset failed", "error", err)
		}
	}
	d.history = d.history[:0]
	d.lastLogits = nil
	d.modelSnapshots = nil
	d.referenceOnly = d.preferReferenceOnly
	d.guardTriggered = false
	d.draftedCount = 0
	d.rewoundCount = 0
	d.stateCorrections = 0
	if d.tapRecorder != nil {
		d.tapRecorder.Reset()
	}
	if d.reference != nil {
		d.reference.Reset()
	}
}

func (d *aneDraftDrafter) captureDraftSnapshot() error {
	if d == nil || d.model == nil || !d.directModelIR {
		return nil
	}
	snap, err := d.model.SnapshotDecodeState()
	if err != nil {
		return err
	}
	d.modelSnapshots = append(d.modelSnapshots, snap)
	return nil
}

func (d *aneDraftDrafter) restoreDraftSnapshot(n int) error {
	if d == nil || d.model == nil || n <= 0 {
		return fmt.Errorf("restore draft snapshot: unavailable")
	}
	if len(d.modelSnapshots) < n {
		return fmt.Errorf("restore draft snapshot: have %d want %d", len(d.modelSnapshots), n)
	}
	idx := len(d.modelSnapshots) - n
	snap := d.modelSnapshots[idx]
	if err := d.model.RestoreDecodeState(snap); err != nil {
		return err
	}
	d.modelSnapshots = d.modelSnapshots[:idx]
	return nil
}

var _ decode.Drafter = (*aneDraftDrafter)(nil)

func (d *aneDraftDrafter) ANEDraftRuntimeStats() aneDraftRuntimeStats {
	if d == nil {
		return aneDraftRuntimeStats{}
	}
	backend := "ane"
	switch {
	case d.model == nil && d.reference != nil:
		backend = "reference-only"
	case d.referenceOnly && d.guardTriggered:
		backend = "reference-only"
	case d.referenceOnly && d.model == nil:
		backend = "reference-only"
	}
	qwenOnANE, qwenOnMLX := mlxgoane.ModelIRCompileProfileQwenSupport(d.compileProfile)
	return aneDraftRuntimeStats{
		Mode:                  d.mode,
		DraftBackend:          backend,
		DraftModelKind:        string(d.modelKind),
		MaxLayers:             d.maxLayers,
		DirectModelIR:         d.directModelIR,
		DirectCompileFallback: d.compileFallback,
		DirectCompileProfile:  d.compileProfile,
		DirectCompileTier:     mlxgoane.ModelIRCompileProfileTier(d.compileProfile),
		DirectCompileNote:     mlxgoane.ModelIRCompileProfileNote(d.compileProfile),
		DirectQwenOnANE:       qwenOnANE,
		DirectQwenOnMLX:       qwenOnMLX,
		ReferenceOnly:         d.referenceOnly,
		GuardTriggered:        d.guardTriggered,
		FFNFallbackUsed:       d.ffnFallbackUsed,
		PrefillDuration:       d.prefillDuration,
		DraftCalls:            d.draftCalls,
		DraftDuration:         d.draftDuration,
		EnsureTokenCalls:      d.ensureTokenCalls,
		EnsureTokenDuration:   d.ensureTokenDuration,
		FeedTokenCalls:        d.feedTokenCalls,
		FeedTokenDuration:     d.feedTokenDuration,
		EmbeddingLookups:      d.embedLookups,
		EmbeddingDuration:     d.embedDuration,
		ModelEvalCalls:        d.modelEvalCalls,
		ModelEvalDuration:     d.modelEvalDuration,
		NextTokenCalls:        d.nextTokenCalls,
		NextTokenDuration:     d.nextTokenDuration,
		AdvanceDecodeCalls:    d.advanceDecodeCalls,
		AdvanceDecodeDuration: d.advanceDecodeDur,
		RewindCalls:           d.rewindCalls,
		RewindDuration:        d.rewindDuration,
		RebuildCount:          d.rebuildCount,
		RebuildDuration:       d.rebuildDuration,
	}
}

func extractDraftEmbeddings(draftModel models.LanguageModel) (hiddenDim int, vocabSize int, embeddings []float32, err error) {
	wa, ok := draftModel.(models.WeightAccessor)
	if !ok {
		emb, embErr := extractEmbeddingsFromKnownModels(draftModel)
		if embErr != nil {
			return 0, 0, nil, fmt.Errorf("draft model does not implement models.WeightAccessor: %w", embErr)
		}
		return copyEmbeddingsFromArray(emb, draftModel.Config())
	}
	return copyEmbeddingsFromArray(wa.Embeddings(), draftModel.Config())
}

type draftEmbeddingLookupModel interface {
	EmbedLookup(inputIDs *mlx.Array) (*mlx.Array, error)
}

func makeDraftTokenEmbeddingLookup(
	draftModel models.LanguageModel,
	embeddings []float32,
	hiddenDim int,
	vocabSize int,
) (func(tokenID int) ([]float32, error), error) {
	if m, ok := draftModel.(draftEmbeddingLookupModel); ok {
		return func(tokenID int) ([]float32, error) {
			if tokenID < 0 || tokenID >= vocabSize {
				return nil, fmt.Errorf("embedding lookup token=%d out of range [0,%d)", tokenID, vocabSize)
			}
			ids, err := mlx.FromSlice([]int32{int32(tokenID)}, []int{1}, mlx.Int32)
			if err != nil {
				return nil, fmt.Errorf("build token id array: %w", err)
			}
			defer ids.Free()
			embed, err := m.EmbedLookup(ids)
			if err != nil {
				return nil, fmt.Errorf("embed lookup: %w", err)
			}
			defer embed.Free()
			embF32 := embed
			if embed.Dtype() != mlx.Float32 {
				embF32, err = mlx.Astype(embed, mlx.Float32, nil)
				if err != nil {
					return nil, fmt.Errorf("cast embedding to float32: %w", err)
				}
				defer embF32.Free()
			}
			row, err := mlx.ToSlice[float32](embF32)
			if err != nil {
				return nil, fmt.Errorf("copy embedding row: %w", err)
			}
			if len(row) != hiddenDim {
				return nil, fmt.Errorf("embedding row len=%d want=%d", len(row), hiddenDim)
			}
			return row, nil
		}, nil
	}

	if len(embeddings) == hiddenDim*vocabSize {
		return func(tokenID int) ([]float32, error) {
			if tokenID < 0 || tokenID >= vocabSize {
				return nil, fmt.Errorf("embedding lookup token=%d out of range [0,%d)", tokenID, vocabSize)
			}
			start := tokenID * hiddenDim
			end := start + hiddenDim
			row := make([]float32, hiddenDim)
			copy(row, embeddings[start:end])
			return row, nil
		}, nil
	}
	return nil, fmt.Errorf("draft model cannot provide token embeddings")
}

func materializeEmbeddingsViaLookup(
	m draftEmbeddingLookupModel,
	hiddenDim int,
	vocabSize int,
) ([]float32, error) {
	const chunk = 2048
	embeddings := make([]float32, hiddenDim*vocabSize)

	for start := 0; start < vocabSize; start += chunk {
		end := start + chunk
		if end > vocabSize {
			end = vocabSize
		}
		n := end - start
		idsRaw := make([]int32, n)
		for i := 0; i < n; i++ {
			idsRaw[i] = int32(start + i)
		}
		ids, err := mlx.FromSlice(idsRaw, []int{n}, mlx.Int32)
		if err != nil {
			return nil, fmt.Errorf("build id chunk [%d,%d): %w", start, end, err)
		}
		embed, err := m.EmbedLookup(ids)
		ids.Free()
		if err != nil {
			return nil, fmt.Errorf("embed lookup chunk [%d,%d): %w", start, end, err)
		}
		embF32 := embed
		if embed.Dtype() != mlx.Float32 {
			embF32, err = mlx.Astype(embed, mlx.Float32, nil)
			if err != nil {
				embed.Free()
				return nil, fmt.Errorf("cast chunk [%d,%d) to float32: %w", start, end, err)
			}
		}
		chunkVals, err := mlx.ToSlice[float32](embF32)
		if embF32 != embed {
			embF32.Free()
		}
		embed.Free()
		if err != nil {
			return nil, fmt.Errorf("copy chunk [%d,%d): %w", start, end, err)
		}
		if len(chunkVals) != n*hiddenDim {
			return nil, fmt.Errorf(
				"chunk [%d,%d) embedding len=%d want=%d",
				start,
				end,
				len(chunkVals),
				n*hiddenDim,
			)
		}
		copy(embeddings[start*hiddenDim:end*hiddenDim], chunkVals)
	}
	return embeddings, nil
}

func copyEmbeddingsFromArray(emb *mlx.Array, cfg *models.ModelConfig) (hiddenDim int, vocabSize int, embeddings []float32, err error) {
	if emb == nil || emb.IsNil() {
		return 0, 0, nil, fmt.Errorf("draft embeddings are nil")
	}
	shape := emb.Shape()
	if len(shape) != 2 {
		return 0, 0, nil, fmt.Errorf("draft embeddings shape=%v want rank-2", shape)
	}
	vocabSize = shape[0]
	hiddenDim = shape[1]
	if vocabSize <= 0 || hiddenDim <= 0 {
		return 0, 0, nil, fmt.Errorf("draft embeddings shape=%v has invalid dims", shape)
	}

	if cfg != nil {
		if cfg.HiddenSize > 0 && cfg.HiddenSize != hiddenDim {
			return 0, 0, nil, fmt.Errorf(
				"draft hidden mismatch config=%d embeddings=%d",
				cfg.HiddenSize,
				hiddenDim,
			)
		}
		if cfg.VocabSize > 0 && cfg.VocabSize != vocabSize {
			return 0, 0, nil, fmt.Errorf(
				"draft vocab mismatch config=%d embeddings=%d",
				cfg.VocabSize,
				vocabSize,
			)
		}
	}

	embF32 := emb
	if emb.Dtype() != mlx.Float32 {
		embF32, err = mlx.Astype(emb, mlx.Float32, nil)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("cast embeddings to float32: %w", err)
		}
		defer embF32.Free()
	}
	embeddings, err = mlx.ToSlice[float32](embF32)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("copy embeddings to slice: %w", err)
	}
	if len(embeddings) != vocabSize*hiddenDim {
		return 0, 0, nil, fmt.Errorf(
			"embedding slice len=%d want=%d",
			len(embeddings),
			vocabSize*hiddenDim,
		)
	}
	return hiddenDim, vocabSize, embeddings, nil
}

func extractEmbeddingsFromKnownModels(draftModel models.LanguageModel) (*mlx.Array, error) {
	// Fallback for generated/wrapper models (for example qwen3 wrappers) that keep
	// embeddings in unexported fields and don't implement WeightAccessor.
	seen := make(map[uintptr]bool)
	emb := findEmbeddingsArray(reflect.ValueOf(draftModel), seen, 0)
	if emb == nil || emb.IsNil() {
		return nil, fmt.Errorf("could not locate embedTokens in model graph (%T)", draftModel)
	}
	return emb, nil
}

func findEmbeddingsArray(v reflect.Value, seen map[uintptr]bool, depth int) *mlx.Array {
	if !v.IsValid() || depth > 16 {
		return nil
	}
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return nil
		}
		if v.Kind() == reflect.Ptr {
			ptr := v.Pointer()
			if ptr != 0 {
				if seen[ptr] {
					return nil
				}
				seen[ptr] = true
			}
		}
		v = v.Elem()
		if !v.IsValid() {
			return nil
		}
	}
	if v.Kind() != reflect.Struct {
		return nil
	}

	// Prefer direct embedTokens field if present.
	if f := v.FieldByName("embedTokens"); f.IsValid() {
		f = forceReadableValue(f)
		if f.IsValid() && f.Kind() == reflect.Ptr && !f.IsNil() {
			if emb, ok := f.Interface().(*mlx.Array); ok && emb != nil && !emb.IsNil() {
				return emb
			}
		}
	}

	// Recurse through all fields for wrapper stacks like Qwen3Wrapper.inner.model.
	for i := 0; i < v.NumField(); i++ {
		f := forceReadableValue(v.Field(i))
		if emb := findEmbeddingsArray(f, seen, depth+1); emb != nil {
			return emb
		}
	}
	return nil
}

func forceReadableValue(v reflect.Value) reflect.Value {
	if !v.IsValid() {
		return v
	}
	if v.CanInterface() {
		return v
	}
	if v.CanAddr() {
		return reflect.NewAt(v.Type(), unsafe.Pointer(v.UnsafeAddr())).Elem()
	}
	return v
}

func resolveANEDraftOutputDim(defaultDim int) (int, error) {
	raw := os.Getenv("MLXGO_ANE_DRAFT_OUTPUT_DIM")
	if raw == "" {
		return defaultDim, nil
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("parse MLXGO_ANE_DRAFT_OUTPUT_DIM=%q: %w", raw, err)
	}
	if v <= 0 {
		return 0, fmt.Errorf("MLXGO_ANE_DRAFT_OUTPUT_DIM must be > 0 (got %d)", v)
	}
	return v, nil
}

func tokenIDFromArray(token *mlx.Array) (int, error) {
	if token == nil || token.IsNil() {
		return 0, fmt.Errorf("token array is nil")
	}
	v, err := token.Item()
	if err != nil {
		return 0, fmt.Errorf("array item: %w", err)
	}
	switch t := v.(type) {
	case int:
		return t, nil
	case int32:
		return int(t), nil
	case int64:
		return int(t), nil
	case uint32:
		return int(t), nil
	case uint64:
		return int(t), nil
	default:
		return 0, fmt.Errorf("unexpected token item type %T", v)
	}
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

func tokenIDsFromArray(tokens *mlx.Array) ([]int, error) {
	if tokens == nil || tokens.IsNil() {
		return nil, fmt.Errorf("token array is nil")
	}
	if err := mlx.Eval(tokens); err != nil {
		return nil, fmt.Errorf("eval token array: %w", err)
	}
	switch tokens.Dtype() {
	case mlx.Int32:
		ids, err := mlx.ToSlice[int32](tokens)
		if err != nil {
			return nil, fmt.Errorf("copy int32 tokens: %w", err)
		}
		out := make([]int, len(ids))
		for i := range ids {
			out[i] = int(ids[i])
		}
		return out, nil
	case mlx.Int64:
		ids, err := mlx.ToSlice[int64](tokens)
		if err != nil {
			return nil, fmt.Errorf("copy int64 tokens: %w", err)
		}
		out := make([]int, len(ids))
		for i := range ids {
			out[i] = int(ids[i])
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported token dtype %v (want int32/int64)", tokens.Dtype())
	}
}

func (d *aneDraftDrafter) ensureTokenState(tokenID int) error {
	start := time.Now()
	d.ensureTokenCalls++
	defer func() {
		d.ensureTokenDuration += time.Since(start)
	}()
	if len(d.history) > 0 && d.history[len(d.history)-1] == tokenID && len(d.lastLogits) > 0 {
		return nil
	}
	if len(d.history) > 0 {
		d.stateCorrections++
	}
	return d.feedToken(tokenID)
}

func (d *aneDraftDrafter) feedToken(tokenID int) error {
	start := time.Now()
	d.feedTokenCalls++
	defer func() {
		d.feedTokenDuration += time.Since(start)
	}()
	if d.referenceOnly {
		if d.reference == nil {
			return fmt.Errorf("reference-only mode requested but reference runner is nil")
		}
		refLogits, err := d.reference.Step(tokenID)
		if err != nil {
			return fmt.Errorf("reference step token %d: %w", tokenID, err)
		}
		d.history = append(d.history, tokenID)
		d.lastLogits = refLogits
		return nil
	}
	if d.embedToken == nil {
		return fmt.Errorf("token embedding callback is nil")
	}
	embedStart := time.Now()
	inputEmbed, err := d.embedToken(tokenID)
	d.embedLookups++
	d.embedDuration += time.Since(embedStart)
	if err != nil {
		return fmt.Errorf("embedding lookup %d: %w", tokenID, err)
	}
	modelStart := time.Now()
	modelOut, err := d.model.EvalToken(inputEmbed)
	d.modelEvalCalls++
	d.modelEvalDuration += time.Since(modelStart)
	if err != nil {
		return fmt.Errorf("eval token %d: %w", tokenID, err)
	}
	logits, taps, err := d.logitsFromModelOutput(modelOut)
	if err != nil {
		return err
	}
	aneLogits := logits
	var refLogits []float32
	if d.needsReferenceStep() {
		refLogits, err = d.reference.Step(tokenID)
		if err != nil {
			slog.Warn("ANE draft taps: reference step failed", "token_id", tokenID, "error", err)
			refLogits = nil
		}
	}
	if d.useRefLogits && len(refLogits) > 0 {
		logits = refLogits
	}
	if d.tapRecorder != nil {
		d.tapRecorder.Record(tokenID, modelOut, aneLogits, taps, refLogits)
	}
	advanceStart := time.Now()
	if err := d.model.AdvanceDecodePosition(); err != nil {
		return fmt.Errorf("advance decode position: %w", err)
	}
	d.advanceDecodeCalls++
	d.advanceDecodeDur += time.Since(advanceStart)
	d.history = append(d.history, tokenID)
	d.lastLogits = logits
	return nil
}

func (d *aneDraftDrafter) maybeEnableReferenceOnly() {
	if d == nil || !d.autoRefGuard || d.referenceOnly || d.reference == nil || d.useRefLogits {
		return
	}
	const minDrafted = 8
	const minAcceptRate = 0.05
	const minCorrections = 2
	if d.stateCorrections >= minCorrections {
		if err := d.syncReferenceState(); err != nil {
			slog.Warn(
				"ANE draft guard: failed to switch to reference-only mode after state corrections",
				"state_corrections", d.stateCorrections,
				"error", err,
			)
			return
		}
		d.referenceOnly = true
		d.guardTriggered = true
		slog.Info(
			"ANE draft guard: switching to reference-only mode after repeated state corrections",
			"state_corrections", d.stateCorrections,
			"drafted", d.draftedCount,
			"rewound", d.rewoundCount,
		)
		return
	}
	if d.draftedCount < minDrafted {
		return
	}
	accepted := d.draftedCount - d.rewoundCount
	if accepted < 0 {
		accepted = 0
	}
	acceptRate := float64(accepted) / float64(d.draftedCount)
	if acceptRate >= minAcceptRate {
		return
	}
	if err := d.syncReferenceState(); err != nil {
		slog.Warn(
			"ANE draft guard: failed to switch to reference-only mode after low acceptance",
			"accept_rate", acceptRate,
			"error", err,
		)
		return
	}
	d.referenceOnly = true
	d.guardTriggered = true
	slog.Info(
		"ANE draft guard: switching to reference-only mode after low acceptance",
		"drafted", d.draftedCount,
		"rewound", d.rewoundCount,
		"accept_rate", acceptRate,
	)
}

func (d *aneDraftDrafter) needsReferenceStep() bool {
	if d == nil || d.reference == nil {
		return false
	}
	if d.referenceOnly || d.useRefLogits {
		return true
	}
	return d.tapRecorder != nil && d.tapRecorder.enabled
}

func (d *aneDraftDrafter) needsReferenceShadow() bool {
	if d == nil || d.reference == nil {
		return false
	}
	return d.referenceOnly || d.useRefLogits || (d.tapRecorder != nil && d.tapRecorder.enabled)
}

func (d *aneDraftDrafter) syncReferenceState() error {
	if d == nil || d.reference == nil {
		return fmt.Errorf("reference runner unavailable")
	}
	d.reference.Reset()
	var last []float32
	for _, tokenID := range d.history {
		refLogits, err := d.reference.Step(tokenID)
		if err != nil {
			return fmt.Errorf("sync reference token %d: %w", tokenID, err)
		}
		last = refLogits
	}
	if len(last) > 0 {
		d.lastLogits = last
	} else {
		d.lastLogits = nil
	}
	return nil
}

func envTruthy(name string) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	switch raw {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func allowPartialANEDraftAuto() bool {
	return envTruthy("MLXGO_ANE_DRAFT_ALLOW_PARTIAL")
}

func milLayerRetryCandidates(allowPartial bool) []int {
	if !allowPartial {
		return []int{0}
	}
	// Prefer full-depth attempts first. Keep bounded fallbacks for hosts where
	// full monolithic compile remains unstable.
	return []int{0, 4, 3, 2, 1}
}

func (d *aneDraftDrafter) rebuildState() error {
	start := time.Now()
	d.rebuildCount++
	defer func() {
		d.rebuildDuration += time.Since(start)
	}()
	if err := d.model.Reset(); err != nil {
		return fmt.Errorf("reset model: %w", err)
	}
	if d.reference != nil {
		d.reference.Reset()
	}
	d.lastLogits = nil
	d.modelSnapshots = nil
	orig := append([]int(nil), d.history...)
	d.history = d.history[:0]
	for _, tokenID := range orig {
		if err := d.feedToken(tokenID); err != nil {
			return fmt.Errorf("replay token %d: %w", tokenID, err)
		}
	}
	return nil
}

func (d *aneDraftDrafter) logitsFromModelOutput(
	out []float32,
) ([]float32, *mlxgoane.TransformerForwardTaps, error) {
	if d.tapLayout == nil {
		return out, nil, nil
	}
	taps, err := mlxgoane.SplitTransformerForwardTaps(out, *d.tapLayout)
	if err != nil {
		return nil, nil, fmt.Errorf("split transformer taps: %w", err)
	}
	return taps.Logits, &taps, nil
}

func (d *aneDraftDrafter) nextTokenFromOutput(logits []float32) (int, error) {
	switch len(logits) {
	case d.vocabSize:
		return argmaxFloat32(logits), nil
	case d.hiddenDim:
		if len(d.lmProj) == d.vocabSize*d.hiddenDim {
			return argmaxProjectionWithBias(logits, d.lmProj, d.lmBias, d.vocabSize, d.hiddenDim), nil
		}
		return argmaxProjectionWithBias(logits, d.embeddings, nil, d.vocabSize, d.hiddenDim), nil
	default:
		return 0, fmt.Errorf(
			"unsupported ANE draft output logits len=%d (expected hidden=%d or vocab=%d; raw_output_dim=%d)",
			len(logits),
			d.hiddenDim,
			d.vocabSize,
			d.outputDim,
		)
	}
}

type draftCacheFactoryWithSlice interface {
	MakeCache() []models.Cache
}

type draftCacheFactoryWithSingle interface {
	MakeCache() models.Cache
}

type draftCacheFactoryWithMulti interface {
	MakeCache() *models.MultiLayerCache
}

type aneDraftReferenceRunner struct {
	model     models.LanguageModel
	cache     models.Cache
	newCache  func() models.Cache
	stepCount int
	history   []int
}

func newANEDraftReferenceRunner(model models.LanguageModel) (*aneDraftReferenceRunner, error) {
	if model == nil {
		return nil, fmt.Errorf("draft reference runner: model is nil")
	}
	newCache, err := cacheFactoryForLanguageModel(model)
	if err != nil {
		return nil, err
	}
	r := &aneDraftReferenceRunner{
		model:    model,
		newCache: newCache,
	}
	r.Reset()
	return r, nil
}

func cacheFactoryForLanguageModel(model models.LanguageModel) (func() models.Cache, error) {
	switch m := model.(type) {
	case draftCacheFactoryWithSingle:
		return func() models.Cache { return m.MakeCache() }, nil
	case draftCacheFactoryWithMulti:
		return func() models.Cache { return m.MakeCache() }, nil
	case draftCacheFactoryWithSlice:
		return func() models.Cache { return models.NewMultiLayerCacheFromList(m.MakeCache()) }, nil
	}
	if f := cacheFactoryFromReflectValue(reflect.ValueOf(model), make(map[uintptr]bool), 0); f != nil {
		return f, nil
	}
	if cfg := model.Config(); cfg != nil && cfg.NumLayers > 0 {
		return func() models.Cache {
			return models.NewMultiLayerCache(cfg.NumLayers)
		}, nil
	}
	return nil, fmt.Errorf("draft reference runner: model does not expose MakeCache (type=%T)", model)
}

func cacheFactoryFromReflectValue(v reflect.Value, seen map[uintptr]bool, depth int) func() models.Cache {
	if !v.IsValid() || depth > 24 {
		return nil
	}
	for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return nil
		}
		if v.Kind() == reflect.Ptr {
			ptr := v.Pointer()
			if ptr != 0 {
				if seen[ptr] {
					return nil
				}
				seen[ptr] = true
			}
		}
		v = forceReadableValue(v.Elem())
	}
	if !v.IsValid() {
		return nil
	}
	if v.CanInterface() {
		switch m := v.Interface().(type) {
		case draftCacheFactoryWithSingle:
			return func() models.Cache { return m.MakeCache() }
		case draftCacheFactoryWithMulti:
			return func() models.Cache { return m.MakeCache() }
		case draftCacheFactoryWithSlice:
			return func() models.Cache { return models.NewMultiLayerCacheFromList(m.MakeCache()) }
		}
	}
	if v.Kind() != reflect.Ptr && v.CanAddr() && v.Addr().CanInterface() {
		switch m := v.Addr().Interface().(type) {
		case draftCacheFactoryWithSingle:
			return func() models.Cache { return m.MakeCache() }
		case draftCacheFactoryWithMulti:
			return func() models.Cache { return m.MakeCache() }
		case draftCacheFactoryWithSlice:
			return func() models.Cache { return models.NewMultiLayerCacheFromList(m.MakeCache()) }
		}
	}
	switch v.Kind() {
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if f := cacheFactoryFromReflectValue(forceReadableValue(v.Field(i)), seen, depth+1); f != nil {
				return f
			}
		}
	case reflect.Slice, reflect.Array:
		n := v.Len()
		if n > 4 {
			n = 4
		}
		for i := 0; i < n; i++ {
			if f := cacheFactoryFromReflectValue(forceReadableValue(v.Index(i)), seen, depth+1); f != nil {
				return f
			}
		}
	}
	return nil
}

func (r *aneDraftReferenceRunner) Reset() {
	if r == nil || r.newCache == nil {
		return
	}
	r.cache = r.newCache()
	r.stepCount = 0
	r.history = r.history[:0]
}

type cacheTrimmer interface {
	Offset() int
	TrimToOffset(int) int
}

func (r *aneDraftReferenceRunner) Rewind(n int) error {
	if r == nil || n <= 0 {
		return nil
	}
	if len(r.history) == 0 {
		return nil
	}
	if n >= len(r.history) {
		r.Reset()
		return nil
	}
	targetLen := len(r.history) - n
	if trim, ok := r.cache.(cacheTrimmer); ok {
		trim.TrimToOffset(targetLen)
		r.history = r.history[:targetLen]
		r.stepCount = targetLen
		return nil
	}
	remaining := append([]int(nil), r.history[:targetLen]...)
	r.Reset()
	for _, tokenID := range remaining {
		if _, err := r.Step(tokenID); err != nil {
			return fmt.Errorf("rewind replay token %d: %w", tokenID, err)
		}
	}
	return nil
}

func (r *aneDraftReferenceRunner) Step(tokenID int) ([]float32, error) {
	if r == nil || r.model == nil {
		return nil, fmt.Errorf("draft reference runner: runner is nil")
	}
	tokenArray, err := mlx.FromSlice([]int32{int32(tokenID)}, []int{1, 1}, mlx.Int32)
	if err != nil {
		return nil, fmt.Errorf("draft reference runner: build token array: %w", err)
	}
	defer tokenArray.Free()
	logits, nextCache, err := r.model.Forward(tokenArray, r.cache)
	if err != nil {
		return nil, fmt.Errorf("draft reference runner: forward: %w", err)
	}
	r.cache = nextCache
	r.stepCount++
	r.history = append(r.history, tokenID)
	defer logits.Free()
	return lastTokenLogitsFromOutput(logits)
}

func lastTokenLogitsFromOutput(logits *mlx.Array) ([]float32, error) {
	if logits == nil || logits.IsNil() {
		return nil, fmt.Errorf("draft reference logits: output is nil")
	}
	shape := logits.Shape()
	if len(shape) == 0 {
		return nil, fmt.Errorf("draft reference logits: invalid output shape=%v", shape)
	}
	logitsDim := shape[len(shape)-1]
	if logitsDim <= 0 {
		return nil, fmt.Errorf("draft reference logits: invalid logits dim=%d shape=%v", logitsDim, shape)
	}
	source := logits
	if logits.Dtype() != mlx.Float32 {
		cast, err := mlx.Astype(logits, mlx.Float32, nil)
		if err != nil {
			return nil, fmt.Errorf("draft reference logits: cast to float32: %w", err)
		}
		defer cast.Free()
		source = cast
	}
	vals, err := mlx.ToSlice[float32](source)
	if err != nil {
		return nil, fmt.Errorf("draft reference logits: copy logits: %w", err)
	}
	if len(vals)%logitsDim != 0 {
		return nil, fmt.Errorf("draft reference logits: output len=%d incompatible with logits dim=%d", len(vals), logitsDim)
	}
	start := len(vals) - logitsDim
	out := make([]float32, logitsDim)
	copy(out, vals[start:])
	return out, nil
}

type aneDraftTapRecorder struct {
	enabled   bool
	dir       string
	maxSteps  int
	step      int
	outputDim int
}

type aneDraftTapSample struct {
	Index int     `json:"index"`
	Value float32 `json:"value"`
}

type aneDraftTapRecord struct {
	Step             int                 `json:"step"`
	TokenID          int                 `json:"token_id"`
	OutputDim        int                 `json:"output_dim"`
	RawOutputLen     int                 `json:"raw_output_len"`
	LogitsLen        int                 `json:"logits_len"`
	CapturedAt       time.Time           `json:"captured_at"`
	LogitsTop        []aneDraftTapSample `json:"ane_logits_top"`
	ReferenceTop     []aneDraftTapSample `json:"reference_logits_top,omitempty"`
	LogitsMaxAbsDiff *float32            `json:"logits_max_abs_diff,omitempty"`
	TapLayout        *string             `json:"tap_layout,omitempty"`
	QRopeTop         []aneDraftTapSample `json:"q_rope_top,omitempty"`
	KRopeTop         []aneDraftTapSample `json:"k_rope_top,omitempty"`
	AttnScoresTop    []aneDraftTapSample `json:"attn_scores_top,omitempty"`
}

func newANEDraftTapRecorder(options aneDraftOptions, outputDim int) *aneDraftTapRecorder {
	if !options.EnableTaps {
		return nil
	}
	maxSteps := options.TapsMaxSteps
	if maxSteps <= 0 {
		maxSteps = 8
	}
	dir := strings.TrimSpace(options.TapsDir)
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "mlxgo-ane-draft-taps")
	}
	return &aneDraftTapRecorder{
		enabled:   true,
		dir:       dir,
		maxSteps:  maxSteps,
		outputDim: outputDim,
	}
}

func (r *aneDraftTapRecorder) Record(
	tokenID int,
	rawOutput []float32,
	logits []float32,
	taps *mlxgoane.TransformerForwardTaps,
	refLogits []float32,
) {
	if r == nil || !r.enabled || r.step >= r.maxSteps {
		return
	}
	step := r.step
	r.step++
	if err := os.MkdirAll(r.dir, 0o755); err != nil {
		slog.Warn("ANE draft taps: create directory failed", "dir", r.dir, "error", err)
		return
	}
	rec := aneDraftTapRecord{
		Step:         step,
		TokenID:      tokenID,
		OutputDim:    r.outputDim,
		RawOutputLen: len(rawOutput),
		LogitsLen:    len(logits),
		CapturedAt:   time.Now().UTC(),
		LogitsTop:    topKSamples(logits, 16),
	}
	if len(refLogits) > 0 {
		rec.ReferenceTop = topKSamples(refLogits, 16)
		if len(refLogits) == len(logits) {
			diff := maxAbsDiff(logits, refLogits)
			rec.LogitsMaxAbsDiff = &diff
		}
	}
	if taps != nil {
		layout := fmt.Sprintf(
			"q_rope=%d k_rope=%d attn_scores=%d logits=%d",
			len(taps.QRope),
			len(taps.KRope),
			len(taps.AttnScores),
			len(taps.Logits),
		)
		rec.TapLayout = &layout
		rec.QRopeTop = topKSamples(taps.QRope, 8)
		rec.KRopeTop = topKSamples(taps.KRope, 8)
		rec.AttnScoresTop = topKSamples(taps.AttnScores, 8)
	}
	path := filepath.Join(r.dir, fmt.Sprintf("step-%04d.json", step))
	buf, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		slog.Warn("ANE draft taps: marshal failed", "path", path, "error", err)
		return
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		slog.Warn("ANE draft taps: write failed", "path", path, "error", err)
	}
}

func (r *aneDraftTapRecorder) Reset() {
	if r == nil {
		return
	}
	r.step = 0
}

func topKSamples(xs []float32, k int) []aneDraftTapSample {
	if len(xs) == 0 || k <= 0 {
		return nil
	}
	if k > len(xs) {
		k = len(xs)
	}
	best := make([]aneDraftTapSample, 0, k)
	for i, v := range xs {
		inserted := false
		for j := range best {
			if v > best[j].Value {
				best = append(best, aneDraftTapSample{})
				copy(best[j+1:], best[j:])
				best[j] = aneDraftTapSample{Index: i, Value: v}
				inserted = true
				break
			}
		}
		if inserted {
			if len(best) > k {
				best = best[:k]
			}
			continue
		}
		if len(best) < k {
			best = append(best, aneDraftTapSample{Index: i, Value: v})
		}
	}
	return best
}

func maxAbsDiff(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	maxDiff := float32(0)
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

func argmaxProjectionWithBias(hidden, proj, bias []float32, vocabSize, hiddenDim int) int {
	bestTok := 0
	bestScore := float32(-3.4028235e+38) // -MaxFloat32
	for tok := 0; tok < vocabSize; tok++ {
		row := tok * hiddenDim
		score := float32(0)
		for i := 0; i < hiddenDim; i++ {
			score += hidden[i] * proj[row+i]
		}
		if len(bias) == vocabSize {
			score += bias[tok]
		}
		if score > bestScore {
			bestScore = score
			bestTok = tok
		}
	}
	return bestTok
}
