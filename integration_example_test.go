package mlxgoane

import "fmt"

func ExampleAssessDecodeModel() {
	assessment := AssessDecodeModel(DecodeModelSpec{
		ModelDim:            768,
		NumHeads:            6,
		NumKVHeads:          6,
		HeadDim:             128,
		FFNDim:              3072,
		MLPKind:             MLPKindReluSquared,
		UsesResidualMix:     true,
		UsesQKNorm:          true,
		UsesValueEmbeddings: true,
		UsesSlidingWindow:   true,
	})
	fmt.Println(assessment.RecommendedPath)
	fmt.Println(assessment.NeedsModelSpecificDecode)
	// Output:
	// model_specific_decode
	// true
}
