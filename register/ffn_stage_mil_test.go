//go:build darwin && ane_appleneuralengine

package register

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	mlxgoane "github.com/tmc/mlx-go-ane"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
)

// TestFFNStageMILAccuracy compares GPU LayerMLPForward output against
// ANE CompileFFNMIL output using real Qwen2.5-3B weights.
func TestFFNStageMILAccuracy(t *testing.T) {
	modelID := os.Getenv("MODEL")
	if modelID == "" {
		modelID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	cacheBase := filepath.Join(home, ".cache", "huggingface", "hub")
	modelDir := "models--" + filepath.Base(filepath.Dir(modelID)) + "--" + filepath.Base(modelID)
	snapshotsDir := filepath.Join(cacheBase, modelDir, "snapshots")
	entries, err := os.ReadDir(snapshotsDir)
	if err != nil || len(entries) == 0 {
		t.Skipf("model %q not found in HuggingFace cache", modelID)
	}
	resolvedPath := filepath.Join(snapshotsDir, entries[len(entries)-1].Name())
	model, _, err := models.LoadModel(resolvedPath)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	weightFiles, err := models.DiscoverWeightFiles(resolvedPath)
	if err != nil {
		t.Fatalf("discover weights: %v", err)
	}
	if err := model.LoadWeights(weightFiles...); err != nil {
		t.Fatalf("load weights: %v", err)
	}

	type ffnWeighter interface {
		LayerFFNWeights(i int) (gate, up, down []float32, err error)
	}
	type mlpForwarder interface {
		LayerMLPForward(layerIdx int, normalized *mlx.Array) (*mlx.Array, error)
	}
	fw, ok := model.(ffnWeighter)
	if !ok {
		t.Fatalf("model %T does not implement LayerFFNWeights", model)
	}
	mf, ok := model.(mlpForwarder)
	if !ok {
		t.Fatalf("model %T does not implement LayerMLPForward", model)
	}

	cfg := model.Config()
	dim := cfg.HiddenSize
	hidden := cfg.IntermediateSize

	// Get layer 0 FFN weights.
	gate, up, down, err := fw.LayerFFNWeights(0)
	if err != nil {
		t.Fatalf("LayerFFNWeights(0): %v", err)
	}
	t.Logf("dim=%d hidden=%d gate=%d up=%d down=%d", dim, hidden, len(gate), len(up), len(down))

	// Weights arrive in [in, out] from linearWeightRows (same as NewQwen35Stage).
	// CompileFFNMIL expects [out, in], so transpose.
	gate = transposeMatrix(gate, dim, hidden)
	up = transposeMatrix(up, dim, hidden)
	down = transposeMatrix(down, hidden, dim)

	// Compile MIL-based FFN.
	milModel, err := mlxgoane.CompileFFNMIL(dim, hidden, gate, up, down)
	if err != nil {
		t.Fatalf("CompileFFNMIL: %v", err)
	}
	defer milModel.Close()

	// Build eval plan using in-memory model path (no shared events for testing).
	plan, err := mlxgoane.NewSurfaceEvalPlan(milModel.Model, milModel.Input, milModel.Output, mlxgoane.SurfaceEvalPlanConfig{})
	if err != nil {
		t.Fatalf("NewSurfaceEvalPlan: %v", err)
	}
	defer plan.Close()

	// Create test input [1, 1, dim] with small values.
	inputData := make([]float32, dim)
	for i := range inputData {
		inputData[i] = float32(i%7-3) * 0.01
	}

	// GPU path: LayerMLPForward.
	inputArr, err := mlx.FromSlice(inputData, []int{1, 1, dim}, mlx.Float32)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	defer inputArr.Free()

	gpuOut, err := mf.LayerMLPForward(0, inputArr)
	if err != nil {
		t.Fatalf("GPU MLP forward: %v", err)
	}
	if err := mlx.Eval(gpuOut); err != nil {
		t.Fatalf("eval GPU output: %v", err)
	}
	gpuSlice, err := mlx.ToSlice[float32](gpuOut)
	if err != nil {
		t.Fatalf("GPU output to slice: %v", err)
	}

	// ANE path: write input, eval, read output.
	if err := milModel.Input.Write(inputData); err != nil {
		t.Fatalf("write ANE input: %v", err)
	}
	if err := plan.Eval(context.Background()); err != nil {
		t.Fatalf("eval ANE: %v", err)
	}
	aneSlice, err := milModel.Output.Read()
	if err != nil {
		t.Fatalf("read ANE output: %v", err)
	}

	// Compare.
	if len(gpuSlice) != len(aneSlice) {
		t.Fatalf("output length mismatch: GPU=%d ANE=%d", len(gpuSlice), len(aneSlice))
	}

	var maxDiff, sumDiff float64
	var maxDiffIdx int
	for i := range gpuSlice {
		diff := math.Abs(float64(gpuSlice[i] - aneSlice[i]))
		sumDiff += diff
		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
	}
	avgDiff := sumDiff / float64(len(gpuSlice))

	t.Logf("output comparison (n=%d):", len(gpuSlice))
	t.Logf("  max diff: %.6f at index %d (GPU=%.6f ANE=%.6f)", maxDiff, maxDiffIdx, gpuSlice[maxDiffIdx], aneSlice[maxDiffIdx])
	t.Logf("  avg diff: %.6f", avgDiff)
	t.Logf("  first 5 GPU: %v", gpuSlice[:min(5, len(gpuSlice))])
	t.Logf("  first 5 ANE: %v", aneSlice[:min(5, len(aneSlice))])

	// Check if ANE output is all zeros.
	allZero := true
	for _, v := range aneSlice {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("ANE output is all zeros — computation may not have occurred")
	}

	// fp16 tolerance threshold.
	if maxDiff > 1.0 {
		t.Errorf("max diff too large: %.6f (expected < 1.0 for float16 ANE)", maxDiff)
	}
}
