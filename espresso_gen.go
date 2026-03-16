//go:build darwin && ane_appleneuralengine

package mlxgoane

import "fmt"

// EspressoNet represents model.espresso.net.
type EspressoNet struct {
	FormatVersion     int                    `json:"format_version"`
	Storage           string                 `json:"storage"`
	Layers            []EspressoLayer        `json:"layers"`
	Analyses          map[string]interface{} `json:"analyses"`
	Properties        map[string]interface{} `json:"properties"`
	MetadataInWeights []interface{}          `json:"metadata_in_weights"`
}

// EspressoLayer represents one layer entry in model.espresso.net.
type EspressoLayer struct {
	Name   string `json:"name"`
	Type   string `json:"type"`
	Top    string `json:"top"`
	Bottom string `json:"bottom,omitempty"`

	NC          int `json:"nC,omitempty"`
	NB          int `json:"nB,omitempty"`
	HasBiases   int `json:"has_biases,omitempty"`
	HasRelu     int `json:"has_relu,omitempty"`
	HasTanh     int `json:"has_tanh,omitempty"`
	HasPrelu    int `json:"has_prelu,omitempty"`
	BlobWeights int `json:"blob_weights,omitempty"`
	BlobBiases  int `json:"blob_biases,omitempty"`

	Mode      int     `json:"mode,omitempty"`
	Operation int     `json:"operation,omitempty"`
	Alpha     float64 `json:"alpha,omitempty"`
	Beta      float64 `json:"beta,omitempty"`
	FusedRelu int     `json:"fused_relu,omitempty"`
	Axis      int     `json:"axis,omitempty"`

	TransposeA int `json:"transpose_a,omitempty"`
	TransposeB int `json:"transpose_b,omitempty"`
	AdjX       int `json:"adj_x,omitempty"`
	AdjY       int `json:"adj_y,omitempty"`

	Shape      []int `json:"shape,omitempty"`
	Axes       []int `json:"axes,omitempty"`
	Interleave int   `json:"interleave,omitempty"`
	KeepDims   int   `json:"keep_dims,omitempty"`

	Attributes map[string]int         `json:"attributes,omitempty"`
	Weights    map[string]interface{} `json:"weights"`
	DebugInfo  string                 `json:"debug_info,omitempty"`
}

func newEspressoLayerWeights() map[string]interface{} {
	return map[string]interface{}{}
}

// EspressoEmbeddingLayer returns an embedding lookup layer represented as an
// Espresso inner_product with is_lookup=1.
func EspressoEmbeddingLayer(name, bottom, top string, vocabSize, dim int, blobWeights, blobBiases int) EspressoLayer {
	return EspressoLayer{
		Name:        name,
		Type:        "inner_product",
		Top:         top,
		Bottom:      bottom,
		NB:          vocabSize,
		NC:          dim,
		HasBiases:   boolToInt(blobBiases != 0),
		HasRelu:     0,
		HasTanh:     0,
		HasPrelu:    0,
		BlobWeights: blobWeights,
		BlobBiases:  blobBiases,
		Attributes: map[string]int{
			"is_lookup": 1,
		},
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

// EspressoLinearLayer returns an Espresso inner_product layer.
func EspressoLinearLayer(name, bottom, top string, inDim, outDim int, blobWeights, blobBiases int) EspressoLayer {
	return EspressoLayer{
		Name:        name,
		Type:        "inner_product",
		Top:         top,
		Bottom:      bottom,
		NB:          inDim,
		NC:          outDim,
		HasBiases:   1,
		HasRelu:     0,
		HasTanh:     0,
		HasPrelu:    0,
		BlobWeights: blobWeights,
		BlobBiases:  blobBiases,
		Weights:     newEspressoLayerWeights(),
		DebugInfo:   name,
	}
}

// EspressoSoftmaxLayer returns an Espresso softmax layer.
func EspressoSoftmaxLayer(name, bottom, top string, axis int) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "softmax",
		Top:       top,
		Bottom:    bottom,
		Axis:      axis,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoActivationLayer returns an Espresso activation layer.
func EspressoActivationLayer(name, bottom, top string, mode int) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "activation",
		Top:       top,
		Bottom:    bottom,
		Mode:      mode,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoMatMulLayer returns an Espresso matmul layer.
func EspressoMatMulLayer(name, bottom1, bottom2, top string) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "matmul",
		Top:       top,
		Bottom:    fmt.Sprintf("%s,%s", bottom1, bottom2),
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoBatchMatMulLayer returns an Espresso batch_matmul layer.
func EspressoBatchMatMulLayer(name, bottom1, bottom2, top string, transposeA, transposeB bool) EspressoLayer {
	return EspressoLayer{
		Name:       name,
		Type:       "batch_matmul",
		Top:        top,
		Bottom:     fmt.Sprintf("%s,%s", bottom1, bottom2),
		TransposeA: boolToInt(transposeA),
		TransposeB: boolToInt(transposeB),
		AdjX:       boolToInt(transposeA),
		AdjY:       boolToInt(transposeB),
		Weights:    newEspressoLayerWeights(),
		DebugInfo:  name,
	}
}

// EspressoReshapeLayer returns an Espresso reshape layer.
func EspressoReshapeLayer(name, bottom, top string, shape []int) EspressoLayer {
	shapeCopy := append([]int(nil), shape...)
	return EspressoLayer{
		Name:      name,
		Type:      "reshape",
		Top:       top,
		Bottom:    bottom,
		Shape:     shapeCopy,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoTransposeLayer returns an Espresso transpose layer.
func EspressoTransposeLayer(name, bottom, top string, axes []int) EspressoLayer {
	axesCopy := append([]int(nil), axes...)
	return EspressoLayer{
		Name:      name,
		Type:      "transpose",
		Top:       top,
		Bottom:    bottom,
		Axes:      axesCopy,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoLoadConstantLayer returns an Espresso load_constant layer.
func EspressoLoadConstantLayer(name, top string, blobWeights int, shape []int) EspressoLayer {
	shapeCopy := append([]int(nil), shape...)
	return EspressoLayer{
		Name:        name,
		Type:        "load_constant",
		Top:         top,
		BlobWeights: blobWeights,
		Shape:       shapeCopy,
		Weights:     newEspressoLayerWeights(),
		DebugInfo:   name,
	}
}

// EspressoReduceSumLayer returns an Espresso reduce_sum layer.
func EspressoReduceSumLayer(name, bottom, top string, axis int) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "reduce_sum",
		Top:       top,
		Bottom:    bottom,
		Axis:      axis,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoReduceMeanLayer returns an Espresso reduce_mean layer.
func EspressoReduceMeanLayer(name, bottom, top string, axis int, keepDims bool) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "reduce_mean",
		Top:       top,
		Bottom:    bottom,
		Axis:      axis,
		KeepDims:  boolToInt(keepDims),
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoReduceMaxLayer returns an Espresso reduce_max layer.
func EspressoReduceMaxLayer(name, bottom, top string, axis int, keepDims bool) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "reduce_max",
		Top:       top,
		Bottom:    bottom,
		Axis:      axis,
		KeepDims:  boolToInt(keepDims),
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoUnaryLayer returns a unary Espresso op layer.
func EspressoUnaryLayer(name, typ, bottom, top string) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      typ,
		Top:       top,
		Bottom:    bottom,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoBinaryLayer returns a binary Espresso op layer.
func EspressoBinaryLayer(name, typ, bottom1, bottom2, top string) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      typ,
		Top:       top,
		Bottom:    fmt.Sprintf("%s,%s", bottom1, bottom2),
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoConcatLayer returns an Espresso concat layer.
func EspressoConcatLayer(name string, bottoms []string, top string, axis int, interleave bool) EspressoLayer {
	return EspressoLayer{
		Name:       name,
		Type:       "concat",
		Top:        top,
		Bottom:     joinBottoms(bottoms),
		Axis:       axis,
		Interleave: boolToInt(interleave),
		Weights:    newEspressoLayerWeights(),
		DebugInfo:  name,
	}
}

// EspressoElementwiseAddLayer returns an Espresso add layer.
func EspressoElementwiseAddLayer(name, bottom1, bottom2, top string) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "elementwise",
		Top:       top,
		Bottom:    fmt.Sprintf("%s,%s", bottom1, bottom2),
		Operation: 0,
		Alpha:     1,
		Beta:      0,
		FusedRelu: 0,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoElementwiseMulLayer returns an Espresso multiply layer.
func EspressoElementwiseMulLayer(name, bottom1, bottom2, top string) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "elementwise",
		Top:       top,
		Bottom:    fmt.Sprintf("%s,%s", bottom1, bottom2),
		Operation: 1,
		Alpha:     1,
		Beta:      0,
		FusedRelu: 0,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

// EspressoElementwiseScaleLayer returns an Espresso scalar scale layer.
func EspressoElementwiseScaleLayer(name, bottom, top string, alpha float64) EspressoLayer {
	return EspressoLayer{
		Name:      name,
		Type:      "elementwise",
		Top:       top,
		Bottom:    bottom,
		Operation: 2,
		Alpha:     alpha,
		Beta:      0,
		FusedRelu: 0,
		Weights:   newEspressoLayerWeights(),
		DebugInfo: name,
	}
}

func joinBottoms(bottoms []string) string {
	if len(bottoms) == 0 {
		return ""
	}
	out := bottoms[0]
	for _, s := range bottoms[1:] {
		out += "," + s
	}
	return out
}

// EspressoRoPELayers returns elementwise layers that apply RoPE-like
// multiply/multiply/add using precomputed cos/sin constants.
func EspressoRoPELayers(namePrefix, bottom, top string, dim, maxSeq int, blobCos, blobSin int) []EspressoLayer {
	cosName := fmt.Sprintf("%s_cos_blob_%d", namePrefix, blobCos)
	sinName := fmt.Sprintf("%s_sin_blob_%d", namePrefix, blobSin)
	mulCosTop := namePrefix + "_mul_cos"
	mulSinTop := namePrefix + "_mul_sin"
	return []EspressoLayer{
		{
			Name:      namePrefix + "_mul_cos",
			Type:      "elementwise",
			Top:       mulCosTop,
			Bottom:    fmt.Sprintf("%s,%s", bottom, cosName),
			Operation: 1, // multiply
			Alpha:     1,
			Beta:      0,
			FusedRelu: 0,
			Weights: map[string]interface{}{
				"blob_cos": blobCos,
				"dim":      dim,
				"max_seq":  maxSeq,
			},
			DebugInfo: namePrefix + "_mul_cos",
		},
		{
			Name:      namePrefix + "_mul_sin",
			Type:      "elementwise",
			Top:       mulSinTop,
			Bottom:    fmt.Sprintf("%s,%s", bottom, sinName),
			Operation: 1, // multiply
			Alpha:     1,
			Beta:      0,
			FusedRelu: 0,
			Weights: map[string]interface{}{
				"blob_sin": blobSin,
				"dim":      dim,
				"max_seq":  maxSeq,
			},
			DebugInfo: namePrefix + "_mul_sin",
		},
		{
			Name:      namePrefix + "_add",
			Type:      "elementwise",
			Top:       top,
			Bottom:    fmt.Sprintf("%s,%s", mulCosTop, mulSinTop),
			Operation: 0, // add
			Alpha:     1,
			Beta:      0,
			FusedRelu: 0,
			Weights:   newEspressoLayerWeights(),
			DebugInfo: namePrefix + "_add",
		},
	}
}
