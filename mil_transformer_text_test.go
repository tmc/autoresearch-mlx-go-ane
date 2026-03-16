//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"strings"
	"testing"
)

func TestTransformerMILTextIncludesNormsAndTaps(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:     1,
		Dim:           8,
		NumHeads:      2,
		HeadDim:       4,
		HiddenDim:     16,
		VocabSize:     32,
		IncludeLMHead: true,
		EnableTaps:    true,
		MaxSeqLen:     32,
	}
	text, err := transformerMILText(cfg)
	if err != nil {
		t.Fatalf("transformerMILText: %v", err)
	}
	for _, want := range []string{
		"input_norm",
		"post_norm",
		"q_norm",
		"k_norm",
		"final_norm",
		"tap_out = concat(",
		"cast_out",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("transformer MIL text missing %q", want)
		}
	}
}

func TestBuildMILTransformerWeightFilesAddsNormAndRope(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers: 1,
		Dim:       4,
		NumHeads:  1,
		HeadDim:   4,
		HiddenDim: 8,
		MaxSeqLen: 16,
	}
	weights := MILTransformerWeights{
		Layers: []MILTransformerLayerWeights{
			{
				QW: make([]float32, 4*4),
				QB: make([]float32, 4),
				KW: make([]float32, 4*4),
				KB: make([]float32, 4),
				VW: make([]float32, 4*4),
				VB: make([]float32, 4),
				OW: make([]float32, 4*4),
				OB: make([]float32, 4),
				W1: make([]float32, 8*4),
				B1: make([]float32, 8),
				W3: make([]float32, 8*4),
				B3: make([]float32, 8),
				W2: make([]float32, 4*8),
				B2: make([]float32, 4),
			},
		},
	}
	files, err := buildMILTransformerWeightFiles(cfg, weights)
	if err != nil {
		t.Fatalf("buildMILTransformerWeightFiles: %v", err)
	}
	paths := make([]string, 0, len(files))
	for _, f := range files {
		paths = append(paths, f.Path)
	}
	joined := strings.Join(paths, "\n")
	for _, want := range []string{
		"input_norm.bin",
		"post_norm.bin",
		"q_norm.bin",
		"k_norm.bin",
		"final_norm.bin",
		"rope_cos.bin",
		"rope_sin.bin",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("weights missing %q", want)
		}
	}
}

func TestTransformerMILTextDynamicRoPEInputs(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:         1,
		Dim:               8,
		NumHeads:          2,
		HeadDim:           4,
		HiddenDim:         16,
		VocabSize:         32,
		IncludeLMHead:     false,
		DynamicRoPEInputs: true,
		MaxSeqLen:         32,
	}
	text, err := transformerMILText(cfg)
	if err != nil {
		t.Fatalf("transformerMILText: %v", err)
	}
	for _, want := range []string{
		"pos_cos_in",
		"pos_sin_in",
		"rope_rotate_w",
		"cast_pos_cos",
		"cast_pos_sin",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("transformer MIL text missing %q", want)
		}
	}
}

func TestBuildMILTransformerWeightFilesAddsDynamicRoPEWeights(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:         1,
		Dim:               8,
		NumHeads:          2,
		HeadDim:           4,
		HiddenDim:         16,
		MaxSeqLen:         32,
		DynamicRoPEInputs: true,
	}
	weights := MILTransformerWeights{
		Layers: []MILTransformerLayerWeights{
			{
				QW: make([]float32, 8*8),
				QB: make([]float32, 8),
				KW: make([]float32, 8*8),
				KB: make([]float32, 8),
				VW: make([]float32, 8*8),
				VB: make([]float32, 8),
				OW: make([]float32, 8*8),
				OB: make([]float32, 8),
				W1: make([]float32, 16*8),
				B1: make([]float32, 16),
				W3: make([]float32, 16*8),
				B3: make([]float32, 16),
				W2: make([]float32, 8*16),
				B2: make([]float32, 8),
			},
		},
	}
	files, err := buildMILTransformerWeightFiles(cfg, weights)
	if err != nil {
		t.Fatalf("buildMILTransformerWeightFiles: %v", err)
	}
	paths := make([]string, 0, len(files))
	for _, f := range files {
		paths = append(paths, f.Path)
	}
	joined := strings.Join(paths, "\n")
	for _, want := range []string{
		"rope_rotate_w.bin",
		"rope_rotate_b.bin",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("weights missing %q", want)
		}
	}
}

func TestBuildRoPERotateMatrix(t *testing.T) {
	got := buildRoPERotateMatrix(4, 4)
	want := []float32{
		0, -1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, -1,
		0, 0, 1, 0,
	}
	if len(got) != len(want) {
		t.Fatalf("buildRoPERotateMatrix len=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("buildRoPERotateMatrix[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}
