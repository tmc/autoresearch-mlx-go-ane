//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"
	"os"
	"strings"

	"github.com/tmc/apple/private/appleneuralengine"
	"github.com/tmc/mlx-go/modelir"
	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
)

// WrapperPattern identifies the recognized wrapper entry shape.
type WrapperPattern = anereify.WrapperPattern

const (
	WrapperPatternUnknown       = anereify.WrapperPatternUnknown
	WrapperPatternDirectProgram = anereify.WrapperPatternDirectProgram
	WrapperPatternPythonCall    = anereify.WrapperPatternPythonCall
	WrapperPatternGoForward     = anereify.WrapperPatternGoForward
	WrapperPatternSwiftCallAsFn = anereify.WrapperPatternSwiftCallAsFn
)

// ReifyDiagnostic reports non-fatal canonical reifier decisions.
type ReifyDiagnostic = anereify.ReifyDiagnostic

// ReifiedMIL is the canonical compile-consumable MIL artifact bundle.
type ReifiedMIL = anereify.ReifiedMIL

// ReifyOptions controls ModelIR -> ANE MIL reification.
type ReifyOptions = anereify.ReifyOptions

// DetectWrapperPattern returns the recognized wrapper shape of prog entry.
func DetectWrapperPattern(prog *modelir.Program) (WrapperPattern, error) {
	return anereify.DetectWrapperPattern(prog)
}

// DeriveTransformerFromProgram extracts the canonical transformer config and
// weights expected by the ANE transformer builder from a direct ModelIR
// program's weights.
func DeriveTransformerFromProgram(prog *modelir.Program) (MILTransformerConfig, MILTransformerWeights, error) {
	cfg, weights, err := anereify.DeriveTransformerFromProgram(prog)
	if err != nil {
		return MILTransformerConfig{}, MILTransformerWeights{}, err
	}
	return cfg, weights, nil
}

// ReifyToANEMIL reifies a wrapper-level ModelIR program to runtime MIL artifacts.
func ReifyToANEMIL(prog *modelir.Program, opts ReifyOptions) (ReifiedMIL, error) {
	return anereify.ReifyToANEMIL(prog, opts)
}

// ReifyModelIRTextToANEMIL parses modelir/v1 txtar bytes and reifies them to
// compile-consumable ANE MIL artifacts.
func ReifyModelIRTextToANEMIL(textData []byte, opts ReifyOptions) (ReifiedMIL, error) {
	return anereify.ReifyModelIRTextToANEMIL(textData, opts)
}

// CompileAndLoadReifiedMIL compiles and loads a reified MIL artifact bundle.
func CompileAndLoadReifiedMIL(
	client appleneuralengine.ANEClient,
	reified ReifiedMIL,
	key string,
	qos uint32,
) (*ANEClientMILModel, error) {
	if strings.TrimSpace(reified.MILText) == "" {
		return nil, fmt.Errorf("compile/load reified mil: mil text is empty")
	}
	if len(reified.WeightFiles) == 0 {
		return nil, fmt.Errorf("compile/load reified mil: weight files are empty")
	}
	model, err := CompileAndLoadMILFiles(client, reified.MILText, cloneModelWeightFiles(reified.WeightFiles), key, qos)
	if err != nil {
		return nil, fmt.Errorf("compile/load reified mil: %w", err)
	}
	return model, nil
}

// CompileAndLoadModelIRText parses modelir/v1 txtar bytes, reifies to MIL
// artifacts, then compiles+loads them through _ANEClient.
func CompileAndLoadModelIRText(
	client appleneuralengine.ANEClient,
	textData []byte,
	opts ReifyOptions,
	key string,
	qos uint32,
) (*ANEClientMILModel, ReifiedMIL, error) {
	reified, err := ReifyModelIRTextToANEMIL(textData, opts)
	if err != nil {
		return nil, ReifiedMIL{}, fmt.Errorf("compile/load modelir text: %w", err)
	}
	model, err := CompileAndLoadReifiedMIL(client, reified, key, qos)
	if err != nil {
		return nil, ReifiedMIL{}, fmt.Errorf("compile/load modelir text: %w", err)
	}
	return model, reified, nil
}

// CompileAndLoadModelIRProgram reifies prog to MIL artifacts, then compiles
// and loads them through _ANEClient.
func CompileAndLoadModelIRProgram(
	client appleneuralengine.ANEClient,
	prog *modelir.Program,
	opts ReifyOptions,
	key string,
	qos uint32,
) (*ANEClientMILModel, ReifiedMIL, error) {
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		return nil, ReifiedMIL{}, fmt.Errorf("compile/load modelir program: %w", err)
	}
	model, err := CompileAndLoadReifiedMIL(client, reified, key, qos)
	if err != nil {
		return nil, ReifiedMIL{}, fmt.Errorf("compile/load modelir program: %w", err)
	}
	return model, reified, nil
}

// CompileModelIRTextInMemory parses modelir/v1 txtar bytes, reifies to MIL
// artifacts, then compiles+loads them through ANE in-memory MIL descriptor
// initialization (isMILModel path).
func CompileModelIRTextInMemory(
	textData []byte,
	opts ReifyOptions,
) (appleneuralengine.ANEInMemoryModel, ReifiedMIL, error) {
	compileInMemory := func(label string, reified ReifiedMIL) (appleneuralengine.ANEInMemoryModel, error) {
		if strings.TrimSpace(reified.MILText) == "" {
			return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: mil text is empty", label)
		}
		if len(reified.WeightFiles) == 0 {
			return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: weight files are empty", label)
		}
		model, err := buildModelFromMILTextWithDescriptorFallback(label, reified.MILText, cloneModelWeightFiles(reified.WeightFiles))
		if err != nil {
			return appleneuralengine.ANEInMemoryModel{}, err
		}
		return model, nil
	}

	reified, err := ReifyModelIRTextToANEMIL(textData, opts)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, ReifiedMIL{}, fmt.Errorf("compile modelir text in-memory: %w", err)
	}
	model, err := compileInMemory("modelir text in-memory", reified)
	if err == nil {
		return model, reified, nil
	}
	if disableModelIRCompileFallback() {
		return appleneuralengine.ANEInMemoryModel{}, ReifiedMIL{}, fmt.Errorf(
			"compile modelir text in-memory: %w (fallback disabled by MLXGO_ANE_MODELIR_DISABLE_COMPILE_FALLBACK)",
			err,
		)
	}
	if !isANECCompileFailure(err) {
		return appleneuralengine.ANEInMemoryModel{}, ReifiedMIL{}, fmt.Errorf("compile modelir text in-memory: %w", err)
	}

	baseErr := err
	attemptErrs := []string{fmt.Sprintf("base: %v", err)}
	for _, fallback := range modelIRCompileFallbackProfiles(opts, reified.TransformerConfig) {
		tryReified, reifyErr := ReifyModelIRTextToANEMIL(textData, fallback.Opts)
		if reifyErr != nil {
			attemptErrs = append(
				attemptErrs,
				fmt.Sprintf("%s reify: %v", fallback.Label, reifyErr),
			)
			continue
		}
		tryModel, compileErr := compileInMemory("modelir text in-memory "+fallback.Label, tryReified)
		if compileErr == nil {
			tryReified.Diagnostics = append(tryReified.Diagnostics, ReifyDiagnostic{
				Code: "compile_fallback",
				Message: fmt.Sprintf(
					"in-memory compile fallback applied: %s",
					fallback.Key,
				),
			})
			return tryModel, tryReified, nil
		}
		attemptErrs = append(attemptErrs, fmt.Sprintf("%s compile: %v", fallback.Label, compileErr))
	}
	return appleneuralengine.ANEInMemoryModel{}, ReifiedMIL{}, fmt.Errorf(
		"compile modelir text in-memory: %w (fallback attempts: %s)",
		baseErr,
		strings.Join(attemptErrs, "; "),
	)
}

func isANECCompileFailure(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "aneccompile() failed") || strings.Contains(msg, "invalidmilprogram")
}

func disableModelIRCompileFallback() bool {
	return envTruthy("MLXGO_ANE_MODELIR_DISABLE_COMPILE_FALLBACK")
}

func mergeDerivedTransformerConfig(derived, requested MILTransformerConfig) MILTransformerConfig {
	if requested.NumLayers > 0 {
		derived.NumLayers = requested.NumLayers
	}
	if requested.Dim > 0 {
		derived.Dim = requested.Dim
	}
	if requested.AttentionDim > 0 {
		derived.AttentionDim = requested.AttentionDim
	}
	if requested.NumHeads > 0 {
		derived.NumHeads = requested.NumHeads
	}
	if requested.HeadDim > 0 {
		derived.HeadDim = requested.HeadDim
	}
	if requested.HiddenDim > 0 {
		derived.HiddenDim = requested.HiddenDim
	}
	if requested.VocabSize > 0 {
		derived.VocabSize = requested.VocabSize
	}
	if requested.RMSNormEps > 0 {
		derived.RMSNormEps = requested.RMSNormEps
	}
	if requested.MaxSeqLen > 0 {
		derived.MaxSeqLen = requested.MaxSeqLen
	}
	if requested.EnableTaps {
		derived.EnableTaps = true
	}
	if requested.SkipFFN {
		derived.SkipFFN = true
	}
	if requested.UseConvFFN {
		derived.UseConvFFN = true
	}
	if requested.LinearFFN {
		derived.LinearFFN = true
		derived.UseConvFFN = false
	}
	if requested.FusedLinearFFN {
		derived.FusedLinearFFN = true
		derived.LinearFFN = true
		derived.UseConvFFN = false
	}
	if requested.DisableNormOps {
		derived.DisableNormOps = true
	}
	if requested.DisableInputNormOps {
		derived.DisableInputNormOps = true
	}
	if requested.DisableQKNormOps {
		derived.DisableQKNormOps = true
	}
	if requested.DisableFinalNormOp {
		derived.DisableFinalNormOp = true
	}
	if requested.AttentionMaskInput {
		derived.AttentionMaskInput = true
	}
	if requested.KVCacheState {
		derived.KVCacheState = true
	}
	if requested.KVCacheMaxLen > 0 {
		derived.KVCacheMaxLen = requested.KVCacheMaxLen
	}
	if requested.IncludeLMHead {
		derived.IncludeLMHead = true
	}
	if requested.DynamicRoPEInputs {
		derived.DynamicRoPEInputs = true
	}
	if requested.AttentionOutputGate {
		derived.AttentionOutputGate = true
	}
	if requested.DisableAttentionOutputGate {
		derived.AttentionOutputGate = false
	}
	return derived
}

func entryFunction(prog *modelir.Program) (*modelir.Function, error) {
	entry := strings.TrimSpace(prog.Entry)
	if entry == "" && len(prog.Functions) > 0 {
		entry = prog.Functions[0].Name
	}
	if entry == "" {
		return nil, fmt.Errorf("program has no entry function")
	}
	fn, ok := prog.FunctionByName(entry)
	if !ok {
		return nil, fmt.Errorf("entry function %q not found", entry)
	}
	return fn, nil
}

func resolveReturnValues(fn *modelir.Function) []modelir.Value {
	values := make(map[string]modelir.Value, len(fn.Inputs)+len(fn.Consts)+len(fn.Ops))
	for _, in := range fn.Inputs {
		values[in.Name] = in
	}
	for _, c := range fn.Consts {
		values[c.Name] = c.Value
	}
	for _, op := range fn.Ops {
		for _, out := range op.Outputs {
			values[out.Name] = out
		}
	}
	out := make([]modelir.Value, 0, len(fn.Returns))
	for _, name := range fn.Returns {
		if v, ok := values[name]; ok {
			out = append(out, v)
			continue
		}
		out = append(out, modelir.Value{
			Name: name,
			Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{-1}},
		})
	}
	return out
}

func allowPartialReify(explicit bool) bool {
	if explicit {
		return true
	}
	return envTruthy("MLXGO_ANE_DRAFT_ALLOW_PARTIAL")
}

func envTruthy(key string) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func enableDisableNormCompileFallback() bool {
	return envTruthy("MLXGO_ANE_MODELIR_ENABLE_DISABLE_NORM_FALLBACK")
}

func enableSelectiveNormCompileFallback() bool {
	return envTruthy("MLXGO_ANE_MODELIR_ENABLE_SELECTIVE_NORM_FALLBACK")
}
