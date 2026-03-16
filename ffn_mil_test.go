//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/tmc/apple/private/appleneuralengine"
)

func TestFFNMILTextRejectsInvalidDims(t *testing.T) {
	tests := []struct {
		name        string
		dim, hidden int
		wantPart    string
	}{
		{"dim", 0, 256, "dim"},
		{"hidden", 64, 0, "hidden"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ffnMILText(tc.dim, tc.hidden)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantPart) {
				t.Fatalf("error=%q want substring %q", err, tc.wantPart)
			}
		})
	}
}

func TestFFNMILTextContainsExpectedOps(t *testing.T) {
	got, err := ffnMILText(64, 256)
	if err != nil {
		t.Fatal(err)
	}
	parts := []string{
		"program(1.3)",
		"func main<ios18>(tensor<fp32, [1, 64, 1, 1]> x)",
		"tensor<fp16, [256,64,1,1]> Wgate",
		"tensor<fp16, [256,64,1,1]> Wup",
		"tensor<fp16, [64,256,1,1]> Wdown",
		"sigmoid(x=gate)",
		"mul(x=gate,y=gate_sig)",
		"mul(x=gate_act,y=up)",
		"conv(",
		"BLOBFILE",
		"gate.bin",
		"up.bin",
		"down.bin",
	}
	for _, part := range parts {
		if !strings.Contains(got, part) {
			t.Errorf("MIL missing %q:\n%s", part, got)
		}
	}
}

func TestFFNMILSmall(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}

	dim := 64
	hidden := 256

	gate := makeDeterministicTensor(hidden*dim, 0.01, 29)
	up := makeDeterministicTensor(hidden*dim, 0.01, 31)
	down := makeDeterministicTensor(dim*hidden, 0.01, 37)

	milText, files, err := BuildFFNMILArtifacts(dim, hidden, gate, up, down)
	if err != nil {
		t.Fatal(err)
	}

	model, err := buildModelFromMILTextWithDescriptorFallback("ffn mil small", milText, files)
	if err != nil {
		t.Fatal(err)
	}

	input := makeDeterministicTensor(dim, 0.02, 23)

	// Use evalModelIRRuntimeSteps which handles layout-aware surfaces.
	target := compiledModelIRRuntimeTarget{inMemoryModel: model}
	defer target.Close()

	outs, err := evalModelIRRuntimeSteps(context.Background(), target, [][]float32{input}, dim)
	if err != nil {
		t.Fatal(err)
	}
	out := outs[0]
	t.Logf("first 5 output: %v", out[:min(5, len(out))])

	allZero := true
	for _, v := range out {
		if v != 0 {
			allZero = false
		}
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) || math.Abs(float64(v)) > 1e10 {
			t.Errorf("unreasonable output: %v", v)
			break
		}
	}
	if allZero {
		t.Error("output is all zeros")
	}
}

// TestFFNMILModelScale tests the MIL FFN at actual model dimensions and
// weight magnitudes that overflow the Espresso inner_product path.
func TestFFNMILModelScale(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}

	dim := 2048
	hidden := 11008

	// scale=0.02 is similar to real model weights (mean_abs=0.02).
	// This scale causes total overflow in the Espresso path.
	gate := makeDeterministicTensor(hidden*dim, 0.02, 29)
	up := makeDeterministicTensor(hidden*dim, 0.02, 31)
	down := makeDeterministicTensor(dim*hidden, 0.02, 37)

	milText, files, err := BuildFFNMILArtifacts(dim, hidden, gate, up, down)
	if err != nil {
		t.Fatal(err)
	}

	model, err := buildModelFromMILTextWithDescriptorFallback("ffn mil model scale", milText, files)
	if err != nil {
		t.Fatal(err)
	}

	input := makeDeterministicTensor(dim, 0.02, 23)

	target := compiledModelIRRuntimeTarget{inMemoryModel: model}
	defer target.Close()

	outs, err := evalModelIRRuntimeSteps(context.Background(), target, [][]float32{input}, dim)
	if err != nil {
		t.Fatal(err)
	}
	out := outs[0]
	t.Logf("first 5 output: %v", out[:min(5, len(out))])

	var maxAbs float64
	hasInf := false
	hasNaN := false
	for _, v := range out {
		a := math.Abs(float64(v))
		if a > maxAbs {
			maxAbs = a
		}
		if math.IsInf(float64(v), 0) {
			hasInf = true
		}
		if math.IsNaN(float64(v)) {
			hasNaN = true
		}
	}
	t.Logf("max abs: %.6g inf=%v nan=%v", maxAbs, hasInf, hasNaN)

	if maxAbs > 1e4 {
		t.Errorf("output overflow: max_abs=%.6g (MIL FFN should handle model-scale weights)", maxAbs)
	}
	if hasInf {
		t.Error("output contains inf")
	}
	if hasNaN {
		t.Error("output contains NaN")
	}
}
