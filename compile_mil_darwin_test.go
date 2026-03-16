//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/apple/private/appleneuralengine"
	"github.com/tmc/mlx-go/modelir"
)

// TestCompileMILTextFFN verifies that CompileMILText works with known-good
// MIL text from the hand-rolled BuildFFNMILArtifacts path.
func TestCompileMILTextFFN(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}

	const dim = 64
	const hidden = 256

	gate := makeDeterministicTensor(hidden*dim, 0.01, 29)
	up := makeDeterministicTensor(hidden*dim, 0.01, 31)
	down := makeDeterministicTensor(dim*hidden, 0.01, 37)

	milText, files, err := BuildFFNMILArtifacts(dim, hidden, gate, up, down)
	if err != nil {
		t.Fatal(err)
	}

	compiled, err := CompileMILText(milText, files)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer compiled.Close()

	if len(compiled.Inputs) == 0 {
		t.Fatal("compiled model has no inputs")
	}
	if len(compiled.Outputs) == 0 {
		t.Fatal("compiled model has no outputs")
	}
	t.Logf("inputs: %d, outputs: %d, states: %d", len(compiled.Inputs), len(compiled.Outputs), len(compiled.States))

	// Eval via the compiled model.
	input := makeDeterministicTensor(dim, 0.02, 23)
	target := compiledModelIRRuntimeTarget{inMemoryModel: compiled.Model}

	outs, err := evalModelIRRuntimeSteps(context.Background(), target, [][]float32{input}, dim)
	if err != nil {
		t.Fatalf("eval: %v", err)
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

// TestCompileFromReifiedFFN tests the full ReifyFunction → CompileFromReified
// round-trip. Uses standard 3D inputs — AdaptLayoutForANE rewrites the
// function signature to 4D NCHW.
func TestCompileFromReifiedFFN(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}

	const dim = 2048
	const hidden = 1024

	gate := makeDeterministicTensor(hidden*dim, 0.01, 29)
	up := makeDeterministicTensor(hidden*dim, 0.01, 31)
	down := makeDeterministicTensor(dim*hidden, 0.01, 37)

	fn, weights := buildFFNFunction(dim, hidden, gate, up, down)

	reified, err := ReifyFunction(fn, weights)
	if err != nil {
		t.Fatalf("reify: %v", err)
	}
	t.Logf("MIL text length: %d bytes, weight files: %d", len(reified.MILText), len(reified.WeightFiles))

	compiled, err := CompileFromReified(reified)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer compiled.Close()

	input := makeDeterministicTensor(dim, 0.02, 23)
	target := compiledModelIRRuntimeTarget{inMemoryModel: compiled.Model}

	outs, err := evalModelIRRuntimeSteps(context.Background(), target, [][]float32{input}, dim)
	if err != nil {
		t.Fatalf("eval: %v", err)
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

// TestCompileFromReifiedMatchesHandrolledFFN verifies that the new
// ReifyFunction pipeline produces output matching the hand-rolled
// BuildFFNMILArtifacts path for the same weights and input.
func TestCompileFromReifiedMatchesHandrolledFFN(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}

	const dim = 2048
	const hidden = 1024

	gate := makeDeterministicTensor(hidden*dim, 0.01, 29)
	up := makeDeterministicTensor(hidden*dim, 0.01, 31)
	down := makeDeterministicTensor(dim*hidden, 0.01, 37)
	input := makeDeterministicTensor(dim, 0.02, 23)

	// Hand-rolled path (reference).
	milText, files, err := BuildFFNMILArtifacts(dim, hidden, gate, up, down)
	if err != nil {
		t.Fatal(err)
	}
	handModel, err := buildModelFromMILTextWithDescriptorFallback("handrolled ffn", milText, files)
	if err != nil {
		t.Fatal(err)
	}
	handTarget := compiledModelIRRuntimeTarget{inMemoryModel: handModel}
	defer handTarget.Close()
	handOuts, err := evalModelIRRuntimeSteps(context.Background(), handTarget, [][]float32{input}, dim)
	if err != nil {
		t.Fatalf("hand-rolled eval: %v", err)
	}
	t.Logf("hand-rolled first 5: %v", handOuts[0][:min(5, len(handOuts[0]))])

	// New pipeline path.
	fn, weights := buildFFNFunction(dim, hidden, gate, up, down)
	reified, err := ReifyFunction(fn, weights)
	if err != nil {
		t.Fatalf("reify: %v", err)
	}
	compiled, err := CompileFromReified(reified)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer compiled.Close()
	newTarget := compiledModelIRRuntimeTarget{inMemoryModel: compiled.Model}
	newOuts, err := evalModelIRRuntimeSteps(context.Background(), newTarget, [][]float32{input}, dim)
	if err != nil {
		t.Fatalf("new pipeline eval: %v", err)
	}

	// Compare outputs.
	maxDiff := float64(0)
	for i := range handOuts[0] {
		d := math.Abs(float64(handOuts[0][i] - newOuts[0][i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("max diff hand-rolled vs new pipeline: %.6g", maxDiff)
	t.Logf("new pipeline first 5: %v", newOuts[0][:min(5, len(newOuts[0]))])

	if maxDiff > 0.1 {
		t.Errorf("outputs differ too much: max diff=%.6g", maxDiff)
	}
}

// buildFFNFunction constructs a modelir.Function for a SwiGLU FFN:
// y = down @ (silu(gate @ x) * (up @ x))
//
// Uses standard 3D [1,1,D] inputs. AdaptLayoutForANE rewrites the function
// signature to 4D NCHW [1,D,1,1] for ANE compilation.
func buildFFNFunction(dim, hidden int, gate, up, down []float32) (*modelir.Function, map[string]*modelir.Weight) {
	weights := map[string]*modelir.Weight{
		"gate_w": {DType: modelir.DTypeFP32, Shape: modelir.Shape{int64(hidden), int64(dim)}, Data: float32SliceToBytes(gate)},
		"up_w":   {DType: modelir.DTypeFP32, Shape: modelir.Shape{int64(hidden), int64(dim)}, Data: float32SliceToBytes(up)},
		"down_w": {DType: modelir.DTypeFP32, Shape: modelir.Shape{int64(dim), int64(hidden)}, Data: float32SliceToBytes(down)},
	}

	fn := &modelir.Function{
		Name: "main",
		Inputs: []modelir.Value{
			{Name: "x", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{1, 1, int64(dim)}}},
		},
		Consts: []modelir.Const{
			{Value: modelir.Value{Name: "Wgate", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{int64(hidden), int64(dim)}}}, WeightRef: "gate_w"},
			{Value: modelir.Value{Name: "Wup", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{int64(hidden), int64(dim)}}}, WeightRef: "up_w"},
			{Value: modelir.Value{Name: "Wdown", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{int64(dim), int64(hidden)}}}, WeightRef: "down_w"},
		},
		Ops: []modelir.Op{
			// gate_out = x @ Wgate^T  -> [1,1,hidden]
			{Outputs: []modelir.Value{{Name: "gate_out", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{1, 1, int64(hidden)}}}}, Name: "matmul", Inputs: []string{"x", "Wgate"}, Attrs: "transpose_y=true"},
			// up_out = x @ Wup^T  -> [1,1,hidden]
			{Outputs: []modelir.Value{{Name: "up_out", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{1, 1, int64(hidden)}}}}, Name: "matmul", Inputs: []string{"x", "Wup"}, Attrs: "transpose_y=true"},
			// gate_act = silu(gate_out) -> [1,1,hidden]
			{Outputs: []modelir.Value{{Name: "gate_act", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{1, 1, int64(hidden)}}}}, Name: "silu", Inputs: []string{"gate_out"}},
			// ffn_mid = gate_act * up_out -> [1,1,hidden]
			{Outputs: []modelir.Value{{Name: "ffn_mid", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{1, 1, int64(hidden)}}}}, Name: "mul", Inputs: []string{"gate_act", "up_out"}},
			// y = ffn_mid @ Wdown^T -> [1,1,dim]
			{Outputs: []modelir.Value{{Name: "y", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{1, 1, int64(dim)}}}}, Name: "matmul", Inputs: []string{"ffn_mid", "Wdown"}, Attrs: "transpose_y=true"},
		},
		Returns: []string{"y"},
	}

	return fn, weights
}

func float32SliceToBytes(s []float32) []byte {
	b := make([]byte, len(s)*4)
	for i, v := range s {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}
