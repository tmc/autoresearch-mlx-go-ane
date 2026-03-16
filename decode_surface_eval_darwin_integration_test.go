//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

func TestSurfaceDecodeFFNSeq1Integration(t *testing.T) {
	if os.Getenv("MLXGO_ANE_INTEGRATION_DECODE_SEQ1") == "" {
		t.Skip("set MLXGO_ANE_INTEGRATION_DECODE_SEQ1=1 to run decode seq=1 ANE integration test")
	}
	if testing.Short() {
		t.Skip("skipping decode seq=1 ANE integration test in short mode")
	}

	const (
		dim       = 64
		hiddenDim = 64
		seq       = 1
	)
	rms2 := makeDeterministicTensor(dim, 0.005, 17)
	w1 := makeDeterministicTensor(hiddenDim*dim, 0.003, 23)
	w3 := makeDeterministicTensor(hiddenDim*dim, 0.004, 19)
	w2 := makeDeterministicTensor(dim*hiddenDim, 0.0025, 29)

	cfg := DefaultSurfaceEvalPlanConfig()
	stage, err := NewSurfaceDecodeFFN(dim, hiddenDim, seq, rms2, w1, w3, w2, cfg)
	if err != nil {
		t.Fatalf("NewSurfaceDecodeFFN: %v", err)
	}
	defer stage.Close()
	if !stage.UsingSurfacePlan() {
		t.Fatalf("SurfaceDecodeFFN plan unavailable at seq=1: %v", stage.PlanError())
	}

	input := makeDeterministicTensor(dim*seq, 0.02, 31)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	out, err := stage.Eval(ctx, input)
	if err != nil {
		t.Fatalf("stage.Eval: %v", err)
	}
	if got, want := len(out.Y), dim*seq; got != want {
		t.Fatalf("output len=%d want=%d", got, want)
	}
	for i, v := range out.Y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] is non-finite: %g", i, v)
		}
	}
}
