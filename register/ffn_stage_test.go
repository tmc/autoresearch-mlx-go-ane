//go:build darwin && ane_appleneuralengine

package register

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/apple/private/appleneuralengine"
	mlxgoane "github.com/tmc/mlx-go-ane"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
)

func TestFFNStageAccuracy(t *testing.T) {
	modelID := os.Getenv("MODEL")
	if modelID == "" {
		modelID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
	}
	// Find the snapshot directory in the HuggingFace cache.
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	cacheBase := filepath.Join(home, ".cache", "huggingface", "hub")
	// Convert "mlx-community/Qwen2.5-3B-Instruct-4bit" → "models--mlx-community--Qwen2.5-3B-Instruct-4bit"
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

	// Log weight statistics.
	weightStats := func(name string, w []float32) {
		var wmin, wmax, wsum float64
		wmin = float64(w[0])
		wmax = float64(w[0])
		for _, v := range w {
			f := float64(v)
			if f < wmin {
				wmin = f
			}
			if f > wmax {
				wmax = f
			}
			wsum += math.Abs(f)
		}
		t.Logf("  %s: min=%.6g max=%.6g mean_abs=%.6g", name, wmin, wmax, wsum/float64(len(w)))
	}
	weightStats("gate", gate)
	weightStats("up", up)
	weightStats("down", down)

	// linearWeightRows for quantized models transposes from [out, in] to [in, out].
	// But Espresso inner_product expects [NC, NB] = [out, in].
	// So we need to transpose back for quantized models.
	//
	// For gate_proj: out=hidden, in=dim → linearWeightRows returns [dim, hidden],
	//   but Espresso wants [hidden, dim]. Transpose with rows=dim, cols=hidden.
	// For down_proj: out=dim, in=hidden → linearWeightRows returns [hidden, dim],
	//   but Espresso wants [dim, hidden]. Transpose with rows=hidden, cols=dim.
	skipTranspose := os.Getenv("FFN_NO_TRANSPOSE") == "1"
	if !skipTranspose {
		t.Log("TRANSPOSING WEIGHTS back to [out, in] for Espresso")
		gate = transposeMatrix(gate, dim, hidden)
		up = transposeMatrix(up, dim, hidden)
		down = transposeMatrix(down, hidden, dim)
	}

	// Build ANE stage manually (without shared events for direct testing).
	cacheDir := t.TempDir()
	stageDir := filepath.Join(cacheDir, fmt.Sprintf("ffn_%d_%d", dim, hidden))
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		t.Fatalf("create stage dir: %v", err)
	}
	if err := mlxgoane.GenerateFFNEspressoDir(stageDir, dim, hidden, gate, up, down); err != nil {
		t.Fatalf("generate espresso: %v", err)
	}
	// Log the generated files for debugging.
	weightsFile := filepath.Join(stageDir, "model.espresso.weights")
	info, _ := os.Stat(weightsFile)
	if info != nil {
		t.Logf("weight file size: %d bytes (expected: %d)", info.Size(), 3*dim*hidden*4+3*16*max(dim, hidden)+1024)
	}

	// Get ANE client.
	clientCls := appleneuralengine.GetANEClientClass()
	clientObj := clientCls.SharedConnection()
	if clientObj == nil || clientObj.GetID() == 0 {
		t.Fatal("ANE client unavailable")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())
	model2, err := mlxgoane.CompileAndLoadEspresso(client, stageDir, "", 0)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	// Create a known test input [1, 1, dim] with small values.
	inputData := make([]float32, dim)
	for i := range inputData {
		inputData[i] = float32(i%7-3) * 0.01 // small values [-0.03, 0.03]
	}

	// First: verify with EvalSingleIO (known working path).
	evalSingleOut, _, err := model2.EvalSingleIO(context.Background(), inputData, dim, true)
	if err != nil {
		t.Fatalf("EvalSingleIO: %v", err)
	}
	t.Logf("EvalSingleIO first 5: %v", evalSingleOut[:min(5, len(evalSingleOut))])

	inputSurf, err := mlxgoane.NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatalf("create input surface: %v", err)
	}
	outputSurf, err := mlxgoane.NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatalf("create output surface: %v", err)
	}
	// No shared events — direct eval for testing.
	plan, err := mlxgoane.NewSurfaceEvalPlanWithClientModel(model2, inputSurf, outputSurf, mlxgoane.SurfaceEvalPlanConfig{})
	if err != nil {
		t.Fatalf("create eval plan: %v", err)
	}
	stage := &ffnStage{
		plan:   plan,
		input:  inputSurf,
		output: outputSurf,
		dim:    dim,
		mapSeq: 1,
	}
	defer stage.Close()

	inputArr, err := mlx.FromSlice(inputData, []int{1, 1, dim}, mlx.Float32)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	defer inputArr.Free()

	// GPU path: LayerMLPForward
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
	// First, try with zeros input to see what the baseline is.
	zeros := make([]float32, dim)
	if err := stage.input.Write(zeros); err != nil {
		t.Fatalf("write ANE zeros: %v", err)
	}
	if err := stage.plan.Eval(context.Background()); err != nil {
		t.Fatalf("eval ANE zeros: %v", err)
	}
	zerosOut, err := stage.output.Read()
	if err != nil {
		t.Fatalf("read ANE zeros output: %v", err)
	}
	t.Logf("zeros input → first 5 ANE output: %v", zerosOut[:min(5, len(zerosOut))])

	// Now with actual input.
	if err := stage.input.Write(inputData); err != nil {
		t.Fatalf("write ANE input: %v", err)
	}
	if err := stage.plan.Eval(context.Background()); err != nil {
		t.Fatalf("eval ANE: %v", err)
	}
	aneSlice, err := stage.output.Read()
	if err != nil {
		t.Fatalf("read ANE output: %v", err)
	}
	t.Logf("test input → first 5 ANE output: %v", aneSlice[:min(5, len(aneSlice))])

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

	// Check if ANE output is all zeros (would indicate computation didn't happen).
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

	// Permissive threshold since ANE uses float16 internally.
	if maxDiff > 1.0 {
		t.Errorf("max diff too large: %.6f (expected < 1.0 for float16 ANE)", maxDiff)
	}
}

// TestFFNStageSmall tests with a small identity-like FFN to verify weight layout.
func TestFFNStageSmall(t *testing.T) {
	dim := 2048
	hidden := 5504

	// Create simple weights: identity-like for debugging.
	// gate_proj: [hidden, dim] = eye-like (first dim columns are identity)
	gate := make([]float32, hidden*dim)
	for i := 0; i < dim && i < hidden; i++ {
		gate[i*dim+i] = 1.0 // row i, col i = 1
	}
	// up_proj: all ones (so gate * up = SiLU(gate(x)) * 1 = SiLU(x))
	up := make([]float32, hidden*dim)
	for i := range up {
		up[i] = 1.0 / float32(dim) // small values to avoid overflow
	}
	// down_proj: [dim, hidden] = sum reduction
	down := make([]float32, dim*hidden)
	for i := 0; i < dim; i++ {
		down[i*hidden+i] = 1.0 // identity-like for first dim rows
	}

	cacheDir := t.TempDir()
	stageDir := filepath.Join(cacheDir, fmt.Sprintf("ffn_%d_%d", dim, hidden))
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		t.Fatalf("create stage dir: %v", err)
	}
	if err := mlxgoane.GenerateFFNEspressoDir(stageDir, dim, hidden, gate, up, down); err != nil {
		t.Fatalf("generate espresso: %v", err)
	}

	clientCls := appleneuralengine.GetANEClientClass()
	clientObj := clientCls.SharedConnection()
	if clientObj == nil || clientObj.GetID() == 0 {
		t.Fatal("ANE client unavailable")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())
	model2, err := mlxgoane.CompileAndLoadEspresso(client, stageDir, "", 0)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer model2.Close()

	inputSurf, err := mlxgoane.NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatalf("create input surface: %v", err)
	}
	defer inputSurf.Close()
	outputSurf, err := mlxgoane.NewIOSurfaceFloat32(dim)
	if err != nil {
		t.Fatalf("create output surface: %v", err)
	}
	defer outputSurf.Close()

	plan, err := mlxgoane.NewSurfaceEvalPlanWithClientModel(model2, inputSurf, outputSurf, mlxgoane.SurfaceEvalPlanConfig{})
	if err != nil {
		t.Fatalf("create eval plan: %v", err)
	}
	defer plan.Close()

	// Input: [1, 2, 3, ..., dim]
	input := make([]float32, dim)
	for i := range input {
		input[i] = float32(i+1) * 0.1
	}
	if err := inputSurf.Write(input); err != nil {
		t.Fatalf("write input: %v", err)
	}
	if err := plan.Eval(context.Background()); err != nil {
		t.Fatalf("eval: %v", err)
	}
	output, err := outputSurf.Read()
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	t.Logf("input:  %v", input)
	t.Logf("output: %v", output)

	// Check if output is non-zero and reasonable.
	allZero := true
	for _, v := range output {
		if v != 0 {
			allZero = false
		}
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) || math.Abs(float64(v)) > 1e10 {
			t.Errorf("unreasonable output value: %v", v)
			break
		}
	}
	if allZero {
		t.Error("output is all zeros")
	}
}

// transposeMatrix transposes a [rows, cols] row-major matrix to [cols, rows].
func transposeMatrix(m []float32, rows, cols int) []float32 {
	out := make([]float32, len(m))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = m[r*cols+c]
		}
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Verify IOSurfaceFloat32 has a Read method.
var _ interface {
	Read() ([]float32, error)
} = (*mlxgoane.IOSurfaceFloat32)(nil)

func init() {
	_ = fmt.Sprintf
}
