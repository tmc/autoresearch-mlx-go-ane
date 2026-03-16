package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/runtime/anedecode"
	"github.com/tmc/mlx-go/mlx"
	_ "github.com/tmc/mlx-go-ane/register"
)

func toFloat32(a *mlx.Array) []float32 {
	if a.Dtype() != mlx.Float32 {
		cast, _ := mlx.Astype(a, mlx.Float32, nil)
		mlx.Eval(cast)
		vals, _ := mlx.ToSlice[float32](cast)
		return vals
	}
	mlx.Eval(a)
	vals, _ := mlx.ToSlice[float32](a)
	return vals
}

func makeCache(numLayers int) *models.MultiLayerCache {
	caches := make([]kvcache.Cache, numLayers)
	for i := range caches {
		caches[i] = kvcache.NewInplace()
	}
	return models.NewMultiLayerCacheFromList(caches)
}

func maxAbsDiff(a, b []float32) (float64, int) {
	maxD := 0.0
	maxIdx := 0
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxD {
			maxD = d
			maxIdx = i
		}
	}
	return maxD, maxIdx
}

func argmax(s []float32) int {
	best := 0
	for i, v := range s {
		if v > s[best] {
			best = i
		}
	}
	return best
}

type hookable interface {
	ForwardDecodeWithHook(inputs *mlx.Array, cache kvcache.Cache, hook func(layer int, h *mlx.Array)) (*mlx.Array, kvcache.Cache, error)
}

func main() {
	home, _ := os.UserHomeDir()
	cacheBase := filepath.Join(home, ".cache", "huggingface", "hub")
	snapshotsDir := filepath.Join(cacheBase, "models--mlx-community--Qwen2.5-3B-Instruct-4bit", "snapshots")
	entries, _ := os.ReadDir(snapshotsDir)
	resolvedPath := filepath.Join(snapshotsDir, entries[len(entries)-1].Name())

	model, _, err := models.LoadModel(resolvedPath)
	if err != nil {
		panic(err)
	}
	weightFiles, _ := models.DiscoverWeightFiles(resolvedPath)
	model.LoadWeights(weightFiles...)

	cfg := model.Config()
	numLayers := cfg.NumLayers
	fmt.Printf("Model: %d layers, dim=%d\n", numLayers, cfg.HiddenSize)

	prefillTokens := []int32{151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198}
	prefillInput, _ := mlx.FromSlice(prefillTokens, []int{1, len(prefillTokens)}, mlx.Int32)

	// === GPU fallback (correct) with per-layer hook ===
	fmt.Println("\n=== GPU fallback with per-layer hook ===")
	gpuWrapped, err := anedecode.Wrap(model, anedecode.ANEDecodePlaneOptions{
		Mode: "gpu_fallback", ModelPath: resolvedPath,
	})
	if err != nil {
		panic(err)
	}
	mlx.SetCompileMode(mlx.CompileModeDisabled)

	gpuCache := makeCache(numLayers)
	_, _, err = gpuWrapped.Forward(prefillInput, gpuCache)
	if err != nil {
		panic(err)
	}

	gpuLayerOutputs := make(map[int][]float32)
	decodeToken := int32(9707) // Use known token
	gpuDecodeIn, _ := mlx.FromSlice([]int32{decodeToken}, []int{1, 1}, mlx.Int32)

	gpuHookable, ok := gpuWrapped.(hookable)
	if !ok {
		panic("gpu wrapped does not implement ForwardDecodeWithHook")
	}
	gpuLogits, _, err := gpuHookable.ForwardDecodeWithHook(gpuDecodeIn, gpuCache, func(layer int, h *mlx.Array) {
		gpuLayerOutputs[layer] = toFloat32(h)
	})
	if err != nil {
		panic(fmt.Sprintf("gpu decode with hook: %v", err))
	}
	mlx.Eval(gpuLogits)
	gLogits := toFloat32(gpuLogits)
	fmt.Printf("GPU decode top: %d\n", argmax(gLogits))

	// === ANE path with per-layer hook ===
	fmt.Println("\n=== ANE with per-layer hook ===")
	aneWrapped, err := anedecode.Wrap(model, anedecode.ANEDecodePlaneOptions{
		Mode: "qwen35", ModelPath: resolvedPath,
	})
	if err != nil {
		panic(err)
	}

	aneCache := makeCache(numLayers)
	_, _, err = aneWrapped.Forward(prefillInput, aneCache)
	if err != nil {
		panic(err)
	}

	aneLayerOutputs := make(map[int][]float32)
	aneDecodeIn, _ := mlx.FromSlice([]int32{decodeToken}, []int{1, 1}, mlx.Int32)

	aneHookable, ok := aneWrapped.(hookable)
	if !ok {
		panic("ane wrapped does not implement ForwardDecodeWithHook")
	}
	aneLogits, _, err := aneHookable.ForwardDecodeWithHook(aneDecodeIn, aneCache, func(layer int, h *mlx.Array) {
		aneLayerOutputs[layer] = toFloat32(h)
	})
	if err != nil {
		panic(fmt.Sprintf("ane decode with hook: %v", err))
	}
	mlx.Eval(aneLogits)
	aLogits := toFloat32(aneLogits)
	fmt.Printf("ANE decode top: %d\n", argmax(aLogits))

	// === Compare per-layer outputs ===
	fmt.Println("\n=== Per-layer comparison (GPU fallback vs ANE) ===")
	for i := 0; i < numLayers; i++ {
		gpuOut, gpuOK := gpuLayerOutputs[i]
		aneOut, aneOK := aneLayerOutputs[i]
		if !gpuOK || !aneOK {
			fmt.Printf("Layer %2d: MISSING (gpu=%v ane=%v)\n", i, gpuOK, aneOK)
			continue
		}
		maxDiff, maxIdx := maxAbsDiff(gpuOut, aneOut)
		marker := ""
		if maxDiff > 0.1 {
			marker = " *** DIVERGED"
		}
		fmt.Printf("Layer %2d: max_diff=%.6g at [%d] (gpu=%.4f ane=%.4f)%s\n",
			i, maxDiff, maxIdx, gpuOut[maxIdx], aneOut[maxIdx], marker)
	}

	// Final logit comparison
	logitDiff, logitIdx := maxAbsDiff(gLogits, aLogits)
	fmt.Printf("\nFinal logits max_diff=%.6g at [%d]\n", logitDiff, logitIdx)
}
