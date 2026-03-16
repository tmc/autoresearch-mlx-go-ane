//go:build darwin && ane_appleneuralengine

package mlxgoane

import anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"

func toANEMILTransformerConfig(cfg MILTransformerConfig) anereify.MILTransformerConfig {
	return cfg
}

func toANEMILTransformerWeights(in MILTransformerWeights) anereify.MILTransformerWeights {
	return in
}

func fromANEMILTransformerConfig(cfg anereify.MILTransformerConfig) MILTransformerConfig {
	return cfg
}

func fromANEMILTransformerWeights(in anereify.MILTransformerWeights) MILTransformerWeights {
	return in
}

func fromANEModelWeightFiles(in []anereify.ModelWeightFile) []modelWeightFile {
	return cloneModelWeightFiles(in)
}

func fromModelWeightFiles(in []anereify.ModelWeightFile) []modelWeightFile {
	return cloneModelWeightFiles(in)
}
