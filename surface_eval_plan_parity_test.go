//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/apple/private/appleneuralengine"
)

// TestSurfaceEvalPlanVsEvalSingleIO compares IOSurface-based evaluation via
// SurfaceEvalPlan against the known-working EvalSingleIO path for the same
// compiled Espresso FFN model.
func TestSurfaceEvalPlanVsEvalSingleIO(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}
	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("ANE client unavailable")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	dim := 768
	hidden := 3072
	if s := os.Getenv("TEST_DIM"); s != "" {
		fmt.Sscanf(s, "%d", &dim)
	}
	if s := os.Getenv("TEST_HIDDEN"); s != "" {
		fmt.Sscanf(s, "%d", &hidden)
	}

	w1 := makeDeterministicTensor(hidden*dim, 0.0008, 29)
	w3 := makeDeterministicTensor(hidden*dim, 0.0007, 31)
	w2 := makeDeterministicTensor(dim*hidden, 0.0006, 37)

	dir := filepath.Join(t.TempDir(), fmt.Sprintf("ffn_%d_%d", dim, hidden))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := GenerateFFNEspressoDir(dir, dim, hidden, w1, w3, w2); err != nil {
		t.Fatalf("generate espresso: %v", err)
	}

	model, err := CompileAndLoadEspresso(client, dir, "", 0)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer model.Close()

	input := makeDeterministicTensor(dim, 0.02, 23)

	// Path 1: EvalSingleIO (known working).
	singleOut, _, err := model.EvalSingleIO(context.Background(), input, dim, true)
	if err != nil {
		t.Fatalf("EvalSingleIO: %v", err)
	}
	t.Logf("EvalSingleIO first 5: %v", singleOut[:min(5, len(singleOut))])

	// Path 2: SurfaceEvalPlan (under test).
	inputSurf, err := NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatalf("create input surface: %v", err)
	}
	defer inputSurf.Close()
	outputSurf, err := NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatalf("create output surface: %v", err)
	}
	defer outputSurf.Close()

	plan, err := NewSurfaceEvalPlanWithClientModel(model, inputSurf, outputSurf, SurfaceEvalPlanConfig{})
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}
	defer plan.Close()

	if err := inputSurf.Write(input); err != nil {
		t.Fatalf("write input: %v", err)
	}
	if err := plan.Eval(context.Background()); err != nil {
		t.Fatalf("plan eval: %v", err)
	}
	planOut, err := outputSurf.Read()
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	t.Logf("SurfaceEvalPlan first 5: %v", planOut[:min(5, len(planOut))])

	// Compare.
	if len(singleOut) != len(planOut) {
		t.Fatalf("length mismatch: EvalSingleIO=%d SurfaceEvalPlan=%d", len(singleOut), len(planOut))
	}

	var maxDiff float64
	var maxIdx int
	for i := range singleOut {
		d := math.Abs(float64(singleOut[i] - planOut[i]))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	t.Logf("max diff: %.6g at [%d] (single=%.6g plan=%.6g)", maxDiff, maxIdx, singleOut[maxIdx], planOut[maxIdx])

	// Also check if plan output is all zeros.
	allZero := true
	for _, v := range planOut {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("SurfaceEvalPlan output is all zeros")
	}

	// Check for garbage values.
	for i, v := range planOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)) > 1e10 {
			t.Errorf("SurfaceEvalPlan output[%d] = %g (garbage)", i, v)
			break
		}
	}

	// They should be very close (same model, same input, same hardware).
	if maxDiff > 0.01 {
		t.Errorf("outputs diverge: max diff = %.6g (expected < 0.01)", maxDiff)
	}
}

// TestSurfaceEvalPlanVsEvalSingleIO_ModelSize tests at the actual model
// dimension used by Qwen2.5-3B.
func TestSurfaceEvalPlanVsEvalSingleIO_ModelSize(t *testing.T) {
	if os.Getenv("MLXGO_ANE_TEST_ESPRESSO_PARITY") == "" {
		t.Skip("set MLXGO_ANE_TEST_ESPRESSO_PARITY=1 to run")
	}
	t.Setenv("TEST_DIM", "2048")
	t.Setenv("TEST_HIDDEN", "11008")
	TestSurfaceEvalPlanVsEvalSingleIO(t)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
