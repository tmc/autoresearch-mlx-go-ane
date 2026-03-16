//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/apple/private/appleneuralengine"
	"github.com/tmc/apple/x/coremlcompiler"
	"github.com/tmc/mlx-go/modelir"
	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
)

const modelIRReifyCompiledPackageTestEnv = "MLXGO_ANE_TEST_MODELIR_REIFY_COMPILED_PACKAGE"

func TestReifyToANEMILStatefulCompiledPackageSmoke(t *testing.T) {
	if os.Getenv(modelIRReifyCompiledPackageTestEnv) == "" {
		t.Skipf("set %s=1 to run compiled-package modelir reify smoke test", modelIRReifyCompiledPackageTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := testTransformerConfig(1)
	cfg.Dim = 8
	cfg.NumHeads = 2
	cfg.HeadDim = 4
	cfg.HiddenDim = 16
	cfg.MaxSeqLen = 8
	cfg.KVCacheState = true
	cfg.KVCacheMaxLen = 4
	cfg.SkipFFN = true
	weights := testTransformerWeightsIdentity(cfg)
	prog := directStatefulDecodeProgram(cfg)
	prog.Weights = testModelIRTransformerWeights(cfg, weights)

	reified, err := ReifyToANEMIL(prog, ReifyOptions{
		TransformerConfig:  cfg,
		TransformerWeights: weights,
		RequestedLayers:    1,
		SelectedLayers:     1,
	})
	if err != nil {
		t.Fatalf("ReifyToANEMIL: %v", err)
	}

	tmpDir := t.TempDir()
	weightRoot := filepath.Join(tmpDir, "modelroot")
	if err := os.MkdirAll(weightRoot, 0o755); err != nil {
		t.Fatalf("MkdirAll(weightRoot): %v", err)
	}
	if err := writeCompiledPackageWeightRoot(weightRoot, reified.WeightFiles); err != nil {
		t.Fatalf("writeCompiledPackageWeightRoot: %v", err)
	}

	desc := coremlcompiler.ModelDescription{
		Inputs:  featureDescriptionsFromValues(reified.Inputs),
		Outputs: featureDescriptionsFromValues(reified.Outputs),
		States:  featureDescriptionsFromValues(prog.Functions[0].States),
	}
	outputDir := filepath.Join(tmpDir, "compiled.mlmodelc")
	if err := coremlcompiler.CompileMILText(reified.MILText, 9, desc, weightRoot, outputDir); err != nil {
		t.Fatalf("coremlcompiler.CompileMILText: %v", err)
	}

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	model, err := compileAndLoadModelDirectory(client, outputDir, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		t.Skipf("compiled package compile/load unavailable on this host: %v", err)
	}
	defer model.Close()

	outs, err := evalModelIRRuntimeSteps(context.Background(), compiledModelIRRuntimeTarget{clientModel: model}, [][]float32{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
	}, cfg.Dim)
	if err != nil {
		if isMILKnownMapFailure(err) {
			t.Skipf("compiled package runtime unavailable on this host: %v", err)
		}
		t.Fatalf("evalModelIRRuntimeSteps(compiled package): %v", err)
	}
	if len(outs) != 2 {
		t.Fatalf("compiled package runtime outputs=%d want=2", len(outs))
	}
	if allZeroFloat32s(outs[0]) && allZeroFloat32s(outs[1]) {
		t.Fatalf("compiled package runtime returned all-zero outputs for both steps")
	}
}

func directStatefulDecodeProgram(cfg MILTransformerConfig) *modelir.Program {
	return &modelir.Program{
		Version: "1",
		Target:  "coreml-ane-v1",
		Entry:   "decode",
		Functions: []modelir.Function{{
			Name: "decode",
			Inputs: []modelir.Value{
				{Name: "x", Type: modelir.TensorType{DType: modelir.DTypeFP16, Shape: modelir.Shape{1, 1, int64(cfg.Dim)}}},
			},
			States: []modelir.Value{
				{Name: "l0_k_cache_state", Type: modelir.TensorType{DType: modelir.DTypeFP16, Shape: modelir.Shape{1, int64(cfg.NumHeads), int64(cfg.KVCacheMaxLen), int64(cfg.HeadDim)}}},
				{Name: "l0_v_cache_state", Type: modelir.TensorType{DType: modelir.DTypeFP16, Shape: modelir.Shape{1, int64(cfg.NumHeads), int64(cfg.KVCacheMaxLen), int64(cfg.HeadDim)}}},
			},
			Returns: []string{"logits"},
			Ops: []modelir.Op{
				{
					Name:   "rmsnorm",
					Inputs: []string{"x", "final_norm"},
					Outputs: []modelir.Value{{
						Name: "normed",
						Type: modelir.TensorType{DType: modelir.DTypeFP16, Shape: modelir.Shape{1, 1, int64(cfg.Dim)}},
					}},
					Attrs: "eps=1e-05",
				},
				{
					Name:   "linear",
					Inputs: []string{"normed", "lm_head_w"},
					Outputs: []modelir.Value{{
						Name: "logits",
						Type: modelir.TensorType{DType: modelir.DTypeFP16, Shape: modelir.Shape{1, 1, int64(cfg.VocabSize)}},
					}},
				},
			},
		}},
	}
}

func featureDescriptionsFromValues(values []modelir.Value) []coremlcompiler.FeatureDescription {
	out := make([]coremlcompiler.FeatureDescription, 0, len(values))
	for _, v := range values {
		out = append(out, coremlcompiler.FeatureDescription{
			Name: v.Name,
			Type: modelIRFeatureType(v),
		})
	}
	return out
}

func modelIRFeatureType(v modelir.Value) *coremlcompiler.FeatureType {
	tt := v.Type
	return &coremlcompiler.FeatureType{
		MultiArrayType: &coremlcompiler.ArrayFeatureType{
			Shape:    shapeToInt64s(tt.Shape),
			DataType: modelIRArrayDataType(tt.DType),
		},
	}
}

func shapeToInt64s(shape modelir.Shape) []int64 {
	out := make([]int64, len(shape))
	copy(out, shape)
	return out
}

func modelIRArrayDataType(dt modelir.DType) coremlcompiler.ArrayDataType {
	switch dt {
	case modelir.DTypeFP16:
		return coremlcompiler.ArrayDataTypeFloat16
	case modelir.DTypeFP32:
		return coremlcompiler.ArrayDataTypeFloat32
	case modelir.DTypeInt32:
		return coremlcompiler.ArrayDataTypeInt32
	default:
		return coremlcompiler.ArrayDataTypeFloat32
	}
}

func writeCompiledPackageWeightRoot(root string, files []anereify.ModelWeightFile) error {
	for _, f := range files {
		relPath, err := relativeModelPath(f.Path)
		if err != nil {
			return fmt.Errorf("relativeModelPath(%q): %w", f.Path, err)
		}
		dstPath := filepath.Join(root, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
			return fmt.Errorf("mkdir %q: %w", filepath.Dir(dstPath), err)
		}
		if err := os.WriteFile(dstPath, f.Blob, 0o644); err != nil {
			return fmt.Errorf("write %q: %w", dstPath, err)
		}
	}
	return nil
}

func allZeroFloat32s(xs []float32) bool {
	for _, x := range xs {
		if x != 0 {
			return false
		}
	}
	return true
}
