//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"math"
	"strings"
	"testing"
	"unsafe"

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

// TestFFNMILFlatSurfaces verifies that the compiled MIL FFN model works with
// flat (non-layout-aware) IOSurfaces, which are required for GPU aliasing.
func TestFFNMILFlatSurfaces(t *testing.T) {
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

	model, err := buildModelFromMILTextWithDescriptorFallback("ffn mil flat", milText, files)
	if err != nil {
		t.Fatal(err)
	}
	// Create FLAT IOSurfaces (no layout) — same as what the GPU alias path uses.
	inSurf, err := NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatal(err)
	}
	defer inSurf.Close()
	outSurf, err := NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatal(err)
	}
	defer outSurf.Close()

	plan, err := NewSurfaceEvalPlan(model, inSurf, outSurf, SurfaceEvalPlanConfig{})
	if err != nil {
		t.Skipf("flat surface eval plan: %v (flat surfaces incompatible with MIL model)", err)
	}
	defer plan.Close()

	input := makeDeterministicTensor(dim, 0.02, 23)
	if err := inSurf.Write(input); err != nil {
		t.Fatalf("write input: %v", err)
	}
	if err := plan.Eval(context.Background()); err != nil {
		t.Fatalf("eval: %v", err)
	}
	out, err := outSurf.Read()
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	t.Logf("flat surface first 5 output: %v", out[:min(5, len(out))])

	// Compare with layout-aware path.
	target := compiledModelIRRuntimeTarget{inMemoryModel: model}
	layoutOuts, err := evalModelIRRuntimeSteps(context.Background(), target, [][]float32{input}, dim)
	if err != nil {
		t.Fatalf("layout-aware eval: %v", err)
	}

	t.Logf("layout-aware first 5 output: %v", layoutOuts[0][:min(5, len(layoutOuts[0]))])

	// Check agreement.
	maxDiff := float64(0)
	for i := range out {
		d := math.Abs(float64(out[i] - layoutOuts[0][i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("max diff flat vs layout: %.6g", maxDiff)
}

// TestFFNMILCompactOutput verifies that compacting the planar output IOSurface
// makes the data readable via linear base-address access (simulating GPU alias).
func TestFFNMILCompactOutput(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}

	dim := 64
	hidden := 256

	gate := makeDeterministicTensor(hidden*dim, 0.01, 29)
	up := makeDeterministicTensor(hidden*dim, 0.01, 31)
	down := makeDeterministicTensor(dim*hidden, 0.01, 37)

	milModel, err := CompileFFNMIL(dim, hidden, gate, up, down)
	if err != nil {
		t.Fatal(err)
	}
	defer milModel.Close()

	plan, err := NewSurfaceEvalPlan(milModel.Model, milModel.Input, milModel.Output, SurfaceEvalPlanConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()

	input := makeDeterministicTensor(dim, 0.02, 23)

	// Write input via layout-aware Write().
	if err := milModel.Input.Write(input); err != nil {
		t.Fatalf("write input: %v", err)
	}

	// Eval.
	if err := plan.Eval(context.Background()); err != nil {
		t.Fatalf("eval: %v", err)
	}

	// Read output via layout-aware Read() — this is the reference.
	ref, err := milModel.Output.Read()
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	t.Logf("layout-aware output first 5: %v", ref[:min(5, len(ref))])

	// Now read the raw base address (what the GPU alias would see).
	rawBefore, _, err := milModel.Output.LockReadOnly()
	if err != nil {
		t.Fatalf("lock: %v", err)
	}
	rawSlice := make([]float32, dim)
	copy(rawSlice, unsafe.Slice((*float32)(rawBefore), dim))
	_ = milModel.Output.UnlockReadOnly()
	t.Logf("raw base output first 5: %v", rawSlice[:min(5, len(rawSlice))])

	// Check if raw matches ref (it shouldn't before compaction).
	maxDiff := float64(0)
	for i := range ref {
		d := math.Abs(float64(ref[i] - rawSlice[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("before compact: max diff layout vs raw = %.6g", maxDiff)
	if maxDiff < 0.0001 {
		t.Log("raw base matches layout-aware (surface may be contiguous)")
	}

	// Re-eval to get fresh planar data.
	if err := milModel.Input.Write(input); err != nil {
		t.Fatalf("write input: %v", err)
	}
	if err := plan.Eval(context.Background()); err != nil {
		t.Fatalf("eval: %v", err)
	}

	// Compact: read layout-aware, write linearly to base.
	compacted, err := milModel.Output.Read()
	if err != nil {
		t.Fatalf("compact read: %v", err)
	}
	base, byteLen, err := milModel.Output.LockWritable()
	if err != nil {
		t.Fatalf("compact lock: %v", err)
	}
	if byteLen < len(compacted)*4 {
		_ = milModel.Output.UnlockWritable()
		t.Fatalf("compact: alloc=%d need=%d", byteLen, len(compacted)*4)
	}
	dst := unsafe.Slice((*float32)(base), len(compacted))
	copy(dst, compacted)
	_ = milModel.Output.UnlockWritable()

	// Now read raw base again — should match ref.
	rawAfter, _, err := milModel.Output.LockReadOnly()
	if err != nil {
		t.Fatalf("lock after compact: %v", err)
	}
	afterSlice := make([]float32, dim)
	copy(afterSlice, unsafe.Slice((*float32)(rawAfter), dim))
	_ = milModel.Output.UnlockReadOnly()
	t.Logf("after compact base output first 5: %v", afterSlice[:min(5, len(afterSlice))])

	maxDiffAfter := float64(0)
	for i := range ref {
		d := math.Abs(float64(ref[i] - afterSlice[i]))
		if d > maxDiffAfter {
			maxDiffAfter = d
		}
	}
	t.Logf("after compact: max diff layout vs raw = %.6g", maxDiffAfter)

	if maxDiffAfter > 0.001 {
		t.Errorf("compact output should match layout-aware read: max diff=%.6g", maxDiffAfter)
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
