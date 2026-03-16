//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

type espressoShapeFile struct {
	LayerShapes map[string]espressoTensorShape `json:"layer_shapes"`
}

type espressoTensorShape struct {
	K    int `json:"k"`
	W    int `json:"w"`
	N    int `json:"n"`
	Rank int `json:"_rank"`
	H    int `json:"h"`
}

const espressoWeightsPayloadOffset = 0x100

type espressoFFNWeightLayout struct {
	Bias0Bytes   uint64
	Weight0Bytes uint64
	Bias1Bytes   uint64
	Weight1Bytes uint64
	Bias2Bytes   uint64
	Weight2Bytes uint64
	Bias0Off     uint64
	Weight0Off   uint64
	Bias1Off     uint64
	Weight1Off   uint64
	Bias2Off     uint64
	Weight2Off   uint64
	TotalBytes   uint64
}

func computeEspressoFFNWeightLayout(dim, hiddenDim int) espressoFFNWeightLayout {
	layout := espressoFFNWeightLayout{
		Bias0Bytes:   uint64(hiddenDim * 16),
		Weight0Bytes: uint64(dim * hiddenDim * 4),
		Bias1Bytes:   uint64(hiddenDim * 16),
		Weight1Bytes: uint64(dim * hiddenDim * 4),
		Bias2Bytes:   uint64(dim * 16),
		Weight2Bytes: uint64(hiddenDim * dim * 4),
	}
	layout.Bias0Off = 56
	layout.Weight0Off = uint64(alignUp(int(layout.Bias0Off+layout.Bias0Bytes), 256))
	layout.Bias1Off = layout.Weight0Off + layout.Weight0Bytes
	layout.Weight1Off = uint64(alignUp(int(layout.Bias1Off+layout.Bias1Bytes), 256))
	layout.Bias2Off = layout.Weight1Off + layout.Weight1Bytes
	layout.Weight2Off = uint64(alignUp(int(layout.Bias2Off+layout.Bias2Bytes), 256))
	layout.TotalBytes = layout.Weight2Off + layout.Weight2Bytes
	return layout
}

func espressoFFNHeaderWords(layout espressoFFNWeightLayout) [26]uint64 {
	return [26]uint64{
		12, 0,
		layout.Bias0Off, 1,
		layout.Bias0Bytes, 2,
		0, 3,
		layout.Weight0Bytes, 4,
		0, 5,
		layout.Bias1Bytes, 6,
		0, 7,
		layout.Weight1Bytes, 8,
		0, 9,
		layout.Bias2Bytes, 10,
		0, 11,
		layout.Weight2Bytes, 0,
	}
}

func alignUp(v, alignment int) int {
	if alignment <= 0 {
		return v
	}
	r := v % alignment
	if r == 0 {
		return v
	}
	return v + alignment - r
}

func espressoWeightsPayloadOffsetForSlots(slotCount int) int {
	headerLen := 16 + slotCount*16
	if headerLen < espressoWeightsPayloadOffset {
		headerLen = espressoWeightsPayloadOffset
	}
	return alignUp(headerLen, 64)
}

// GenerateFFNEspressoDir writes a fused SwiGLU FFN Espresso model directory.
//
// The generated directory contains:
//   - model.espresso.net
//   - model.espresso.shape
//   - model.espresso.weights
func GenerateFFNEspressoDir(dir string, dim, hiddenDim int, w1, w3, w2 []float32) error {
	if dim <= 0 || hiddenDim <= 0 {
		return fmt.Errorf("generate espresso dir: invalid dims dim=%d hiddenDim=%d", dim, hiddenDim)
	}
	if len(w1) != hiddenDim*dim {
		return fmt.Errorf("generate espresso dir: w1 len=%d want=%d", len(w1), hiddenDim*dim)
	}
	if len(w3) != hiddenDim*dim {
		return fmt.Errorf("generate espresso dir: w3 len=%d want=%d", len(w3), hiddenDim*dim)
	}
	if len(w2) != dim*hiddenDim {
		return fmt.Errorf("generate espresso dir: w2 len=%d want=%d", len(w2), dim*hiddenDim)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("generate espresso dir: mkdir %q: %w", dir, err)
	}

	net := generateFFNEspressoNet(dim, hiddenDim)
	netBytes, err := json.MarshalIndent(net, "", "  ")
	if err != nil {
		return fmt.Errorf("generate espresso dir: marshal net: %w", err)
	}
	netBytes = append(netBytes, '\n')
	if err := os.WriteFile(filepath.Join(dir, "model.espresso.net"), netBytes, 0o644); err != nil {
		return fmt.Errorf("generate espresso dir: write model.espresso.net: %w", err)
	}

	shape := generateFFNEspressoShape(dim, hiddenDim)
	shapeBytes, err := json.MarshalIndent(shape, "", "  ")
	if err != nil {
		return fmt.Errorf("generate espresso dir: marshal shape: %w", err)
	}
	shapeBytes = append(shapeBytes, '\n')
	if err := os.WriteFile(filepath.Join(dir, "model.espresso.shape"), shapeBytes, 0o644); err != nil {
		return fmt.Errorf("generate espresso dir: write model.espresso.shape: %w", err)
	}

	weights, err := buildFFNEspressoWeights(dim, hiddenDim, w1, w3, w2)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "model.espresso.weights"), weights, 0o644); err != nil {
		return fmt.Errorf("generate espresso dir: write model.espresso.weights: %w", err)
	}
	return nil
}

func generateFFNEspressoNet(dim, hiddenDim int) EspressoNet {
	emptyWeights := map[string]interface{}{}
	return EspressoNet{
		FormatVersion:     200,
		Storage:           "model.espresso.weights",
		Analyses:          map[string]interface{}{},
		Properties:        map[string]interface{}{},
		MetadataInWeights: []interface{}{},
		Layers: []EspressoLayer{
			{
				Name:        "linear_0",
				Type:        "inner_product",
				Top:         "linear_0",
				Bottom:      "x",
				NB:          dim,
				NC:          hiddenDim,
				HasBiases:   1,
				HasRelu:     0,
				HasTanh:     0,
				HasPrelu:    0,
				BlobWeights: 3,
				BlobBiases:  1,
				Weights:     emptyWeights,
				DebugInfo:   "linear_0",
			},
			{
				Name:      "8__silu_sigmoid__",
				Type:      "activation",
				Top:       "8__silu_sigmoid__",
				Bottom:    "linear_0",
				Mode:      3,
				Weights:   emptyWeights,
				DebugInfo: "8__silu_sigmoid__",
			},
			{
				Name:      "8",
				Type:      "elementwise",
				Top:       "8",
				Bottom:    "linear_0,8__silu_sigmoid__",
				Operation: 1,
				Alpha:     1,
				Beta:      0,
				FusedRelu: 0,
				Weights:   emptyWeights,
				DebugInfo: "8",
			},
			{
				Name:        "linear_1",
				Type:        "inner_product",
				Top:         "linear_1",
				Bottom:      "x",
				NB:          dim,
				NC:          hiddenDim,
				HasBiases:   1,
				HasRelu:     0,
				HasTanh:     0,
				HasPrelu:    0,
				BlobWeights: 7,
				BlobBiases:  5,
				Weights:     emptyWeights,
				DebugInfo:   "linear_1",
			},
			{
				Name:      "input",
				Type:      "elementwise",
				Top:       "input",
				Bottom:    "8,linear_1",
				Operation: 1,
				Alpha:     1,
				Beta:      0,
				FusedRelu: 0,
				Weights:   emptyWeights,
				DebugInfo: "input",
			},
			{
				Name:        "linear_2",
				Type:        "inner_product",
				Top:         "linear_2",
				Bottom:      "input",
				NB:          hiddenDim,
				NC:          dim,
				HasBiases:   1,
				HasRelu:     0,
				HasTanh:     0,
				HasPrelu:    0,
				BlobWeights: 11,
				BlobBiases:  9,
				Attributes: map[string]int{
					"is_output": 1,
				},
				Weights:   emptyWeights,
				DebugInfo: "linear_2",
			},
		},
	}
}

func generateFFNEspressoShape(dim, hiddenDim int) espressoShapeFile {
	r3 := func(w int) espressoTensorShape {
		return espressoTensorShape{
			K:    1,
			W:    w,
			N:    1,
			Rank: 3,
			H:    1,
		}
	}
	return espressoShapeFile{
		LayerShapes: map[string]espressoTensorShape{
			"x":                 r3(dim),
			"linear_0":          r3(hiddenDim),
			"8__silu_sigmoid__": r3(hiddenDim),
			"8":                 r3(hiddenDim),
			"linear_1":          r3(hiddenDim),
			"input":             r3(hiddenDim),
			"linear_2":          r3(dim),
		},
	}
}

func buildFFNEspressoWeights(dim, hiddenDim int, w1, w3, w2 []float32) ([]byte, error) {
	w1Bytes := float32SliceLEBytes(w1)
	w3Bytes := float32SliceLEBytes(w3)
	w2Bytes := float32SliceLEBytes(w2)
	layout := computeEspressoFFNWeightLayout(dim, hiddenDim)
	if layout.Weight0Bytes != uint64(len(w1Bytes)) {
		return nil, fmt.Errorf("build espresso weights: w1 bytes=%d want=%d", len(w1Bytes), layout.Weight0Bytes)
	}
	if layout.Weight1Bytes != uint64(len(w3Bytes)) {
		return nil, fmt.Errorf("build espresso weights: w3 bytes=%d want=%d", len(w3Bytes), layout.Weight1Bytes)
	}
	if layout.Weight2Bytes != uint64(len(w2Bytes)) {
		return nil, fmt.Errorf("build espresso weights: w2 bytes=%d want=%d", len(w2Bytes), layout.Weight2Bytes)
	}
	if layout.TotalBytes > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("build espresso weights: total bytes too large: %d", layout.TotalBytes)
	}

	data := make([]byte, int(layout.TotalBytes))
	words := espressoFFNHeaderWords(layout)
	header := bytes.NewBuffer(make([]byte, 0, len(words)*8))
	for i, word := range words {
		if err := binary.Write(header, binary.LittleEndian, word); err != nil {
			return nil, fmt.Errorf("build espresso weights: write header word[%d]: %w", i, err)
		}
	}
	copy(data, header.Bytes())
	copy(data[int(layout.Weight0Off):], w1Bytes)
	copy(data[int(layout.Weight1Off):], w3Bytes)
	copy(data[int(layout.Weight2Off):], w2Bytes)
	return data, nil
}

func float32SliceLEBytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}
