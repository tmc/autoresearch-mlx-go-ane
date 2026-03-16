//go:build !darwin || !ane_appleneuralengine

package mlxgoane

import (
	"github.com/tmc/mlx-go/modelir"
	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
)

// ModelWeightFile is one MIL weight payload.
type ModelWeightFile = anereify.ModelWeightFile

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

// ReifiedMIL is a compile-consumable MIL artifact bundle.
type ReifiedMIL = anereify.ReifiedMIL

// MILTransformerConfig is available cross-platform for artifact generation.
type MILTransformerConfig = anereify.MILTransformerConfig

// MILTransformerLayerWeights is available cross-platform for artifact generation.
type MILTransformerLayerWeights = anereify.MILTransformerLayerWeights

// MILTransformerWeights is available cross-platform for artifact generation.
type MILTransformerWeights = anereify.MILTransformerWeights

// ReifyOptions controls ModelIR -> ANE MIL reification.
type ReifyOptions = anereify.ReifyOptions

// DetectWrapperPattern remains available without runtime execution support.
func DetectWrapperPattern(prog *modelir.Program) (WrapperPattern, error) {
	return anereify.DetectWrapperPattern(prog)
}

// ReifyToANEMIL remains available without runtime execution support.
func ReifyToANEMIL(prog *modelir.Program, opts ReifyOptions) (ReifiedMIL, error) {
	return anereify.ReifyToANEMIL(prog, opts)
}

// ReifyModelIRTextToANEMIL remains available without runtime execution support.
func ReifyModelIRTextToANEMIL(textData []byte, opts ReifyOptions) (ReifiedMIL, error) {
	return anereify.ReifyModelIRTextToANEMIL(textData, opts)
}
