//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"

	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
)

// MILTransformerConfig controls monolithic MIL decode-trunk generation.
type MILTransformerConfig = anereify.MILTransformerConfig

// MILTransformerLayerWeights contains one decode layer's projection tensors.
type MILTransformerLayerWeights = anereify.MILTransformerLayerWeights

// MILTransformerWeights contains all per-layer tensors.
type MILTransformerWeights = anereify.MILTransformerWeights

func normalizeMILTransformerConfig(cfg MILTransformerConfig) MILTransformerConfig {
	return anereify.NormalizeMILTransformerConfig(cfg)
}

func transformerAttentionDim(cfg MILTransformerConfig) int {
	return anereify.TransformerAttentionDim(cfg)
}

func validateMILTransformerConfig(cfg MILTransformerConfig) error {
	return anereify.ValidateMILTransformerConfig(cfg)
}

func validateMILTransformerWeights(cfg MILTransformerConfig, w MILTransformerWeights) error {
	return anereify.ValidateMILTransformerWeights(cfg, w)
}

func ensureMILTransformerWeights(cfg MILTransformerConfig, w MILTransformerWeights) MILTransformerWeights {
	return anereify.EnsureMILTransformerWeights(cfg, w)
}

func buildRoPERotateMatrix(attnDim, headDim int) []float32 {
	return anereify.BuildRoPERotateMatrix(attnDim, headDim)
}

func buildMILTransformerWeightFiles(cfg MILTransformerConfig, w MILTransformerWeights) ([]modelWeightFile, error) {
	return anereify.BuildMILTransformerWeightFiles(cfg, w)
}

// BuildMILTransformerArtifacts builds compile-ready MIL text and weight files.
func BuildMILTransformerArtifacts(
	cfg MILTransformerConfig,
	w MILTransformerWeights,
) (milText string, files []ModelWeightFile, err error) {
	return anereify.BuildMILTransformerArtifacts(cfg, w)
}

func transformerMILText(cfg MILTransformerConfig) (string, error) {
	return anereify.TransformerMILText(cfg)
}

func milTransformerGlobalWeightPath(name string) string {
	return fmt.Sprintf("%sweights/%s.bin", modelPathPrefix, name)
}
