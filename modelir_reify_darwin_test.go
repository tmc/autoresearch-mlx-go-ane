//go:build ignore

package mlxgoane

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
	"github.com/tmc/mlx-go/internal/modelir"
	"github.com/tmc/mlx-go/internal/modelir/lang/golang"
	"github.com/tmc/mlx-go/internal/modelir/lang/python"
	"github.com/tmc/mlx-go/internal/modelir/lang/swift"
	modelirtext "github.com/tmc/mlx-go/internal/modelir/text"
	"golang.org/x/tools/txtar"
)

const modelIRReifyRuntimeTestEnv = "MLXGO_ANE_TEST_MODELIR_REIFY_RUNTIME"
const modelIRReifyStatefulRuntimeTestEnv = "MLXGO_ANE_TEST_MODELIR_REIFY_STATEFUL_RUNTIME"
const modelIRReifyStatefulEvalTestEnv = "MLXGO_ANE_TEST_MODELIR_REIFY_STATEFUL_EVAL"

func TestReifyToANEMILWrapperDetection(t *testing.T) {
	tests := []struct {
		name    string
		prog    *modelir.Program
		want    WrapperPattern
		wantErr string
	}{
		{
			name: "python __call__ wrapper",
			prog: wrapperProgram("__call__", []string{"language_model", "lm_head"}),
			want: WrapperPatternPythonCall,
		},
		{
			name: "go Forward wrapper",
			prog: wrapperProgram("Forward", []string{"normalize_token_inputs", "forward"}),
			want: WrapperPatternGoForward,
		},
		{
			name: "swift callAsFunction wrapper",
			prog: wrapperProgram("callAsFunction", []string{"language_model", "reshape"}),
			want: WrapperPatternSwiftCallAsFn,
		},
		{
			name:    "unsupported wrapper",
			prog:    wrapperProgram("main", []string{"matmul"}),
			wantErr: `unsupported wrapper function "main"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testTransformerConfig(1)
			weights := testTransformerWeights(cfg)
			got, err := ReifyToANEMIL(tc.prog, ReifyOptions{
				TransformerConfig:  cfg,
				TransformerWeights: weights,
			})
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("ReifyToANEMIL error=%v want contains %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ReifyToANEMIL: %v", err)
			}
			if got.Wrapper != tc.want {
				t.Fatalf("ReifyToANEMIL wrapper=%q want=%q", got.Wrapper, tc.want)
			}
			if strings.TrimSpace(got.MILText) == "" {
				t.Fatal("ReifyToANEMIL returned empty MIL text")
			}
			if len(got.WeightFiles) == 0 {
				t.Fatal("ReifyToANEMIL returned no weight files")
			}
		})
	}
}

func TestReifyToANEMILArgumentValidation(t *testing.T) {
	cfg := testTransformerConfig(1)
	weights := testTransformerWeights(cfg)

	if _, err := ReifyToANEMIL(nil, ReifyOptions{}); err == nil || !strings.Contains(err.Error(), "nil program") {
		t.Fatalf("ReifyToANEMIL(nil) error=%v want nil program", err)
	}

	_, err := ReifyToANEMIL(wrapperProgram("__call__", []string{"language_model"}), ReifyOptions{
		TransformerConfig: MILTransformerConfig{},
		TransformerWeights: MILTransformerWeights{
			Layers: nil,
		},
	})
	if err == nil || !strings.Contains(err.Error(), "program has no weights") {
		t.Fatalf("ReifyToANEMIL missing layers error=%v", err)
	}

	_, err = ReifyToANEMIL(wrapperProgram("__call__", []string{"language_model"}), ReifyOptions{
		TransformerConfig:  cfg,
		TransformerWeights: weights,
		RequestedLayers:    1,
		SelectedLayers:     2,
	})
	if err == nil || !strings.Contains(err.Error(), "selected 2/1 layers") {
		t.Fatalf("ReifyToANEMIL selected>requested error=%v", err)
	}
}

func TestReifyToANEMILPartialCoveragePolicy(t *testing.T) {
	t.Setenv("MLXGO_ANE_DRAFT_ALLOW_PARTIAL", "")
	cfg := testTransformerConfig(4)
	weights := testTransformerWeights(cfg)
	prog := wrapperProgram("__call__", []string{"language_model", "lm_head"})

	_, err := ReifyToANEMIL(prog, ReifyOptions{
		TransformerConfig:  cfg,
		TransformerWeights: weights,
		RequestedLayers:    4,
		SelectedLayers:     2,
	})
	if err == nil || !strings.Contains(err.Error(), "selected 2/4 layers; partial coverage disabled") {
		t.Fatalf("ReifyToANEMIL partial disabled error=%v", err)
	}

	t.Setenv("MLXGO_ANE_DRAFT_ALLOW_PARTIAL", "1")
	got, err := ReifyToANEMIL(prog, ReifyOptions{
		TransformerConfig:  cfg,
		TransformerWeights: weights,
		RequestedLayers:    4,
		SelectedLayers:     2,
	})
	if err != nil {
		t.Fatalf("ReifyToANEMIL partial via env: %v", err)
	}
	if got.SelectedLayers != 2 || got.RequestedLayers != 4 {
		t.Fatalf("coverage mismatch selected=%d requested=%d", got.SelectedLayers, got.RequestedLayers)
	}
	if len(got.Diagnostics) == 0 || got.Diagnostics[0].Code != "partial_coverage" {
		t.Fatalf("expected partial_coverage diagnostic, got=%v", got.Diagnostics)
	}
}

func TestReifyToANEMILQwenWrapperFixtures(t *testing.T) {
	files := wrapperFixtureSources(t)

	cfg := testTransformerConfig(1)
	weights := testTransformerWeights(cfg)
	tests := []struct {
		name    string
		srcName string
		parse   func(string) (*modelir.Program, error)
		want    WrapperPattern
	}{
		{
			name:    "python wrapper",
			srcName: "py_wrapper.py",
			parse:   python.Parse,
			want:    WrapperPatternPythonCall,
		},
		{
			name:    "go wrapper",
			srcName: "go_wrapper.go",
			parse:   golang.Parse,
			want:    WrapperPatternGoForward,
		},
		{
			name:    "swift wrapper",
			srcName: "swift_wrapper.swift",
			parse:   swift.Parse,
			want:    WrapperPatternSwiftCallAsFn,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src, ok := files[tc.srcName]
			if !ok {
				t.Fatalf("fixture missing %q", tc.srcName)
			}
			prog, err := tc.parse(src)
			if err != nil {
				t.Fatalf("parse %s: %v", tc.srcName, err)
			}
			pattern, err := DetectWrapperPattern(prog)
			if err != nil {
				t.Fatalf("DetectWrapperPattern(%s): %v", tc.srcName, err)
			}
			if pattern != tc.want {
				t.Fatalf("DetectWrapperPattern(%s)=%q want=%q", tc.srcName, pattern, tc.want)
			}
			reified, err := ReifyToANEMIL(prog, ReifyOptions{
				TransformerConfig:  cfg,
				TransformerWeights: weights,
				RequestedLayers:    1,
				SelectedLayers:     1,
			})
			if err != nil {
				t.Fatalf("ReifyToANEMIL(%s): %v", tc.srcName, err)
			}
			if strings.TrimSpace(reified.MILText) == "" {
				t.Fatalf("ReifyToANEMIL(%s) returned empty MIL text", tc.srcName)
			}
			if len(reified.WeightFiles) == 0 {
				t.Fatalf("ReifyToANEMIL(%s) returned no weight files", tc.srcName)
			}
			if reified.Wrapper != tc.want {
				t.Fatalf("ReifyToANEMIL(%s) wrapper=%q want=%q", tc.srcName, reified.Wrapper, tc.want)
			}
		})
	}
}

func TestReifyToANEMILDerivesFromProgramWeights(t *testing.T) {
	cfg := testTransformerConfig(1)
	weights := testTransformerWeights(cfg)
	prog := wrapperProgram("__call__", []string{"language_model", "lm_head"})
	prog.Weights = testModelIRTransformerWeights(cfg, weights)

	reified, err := ReifyToANEMIL(prog, ReifyOptions{
		RequestedLayers: 1,
		SelectedLayers:  1,
	})
	if err != nil {
		t.Fatalf("ReifyToANEMIL derive weights: %v", err)
	}
	if strings.TrimSpace(reified.MILText) == "" {
		t.Fatal("ReifyToANEMIL derive weights returned empty MIL text")
	}
	if len(reified.WeightFiles) == 0 {
		t.Fatal("ReifyToANEMIL derive weights returned no weight files")
	}
}

func TestDeriveTransformerFromProgram(t *testing.T) {
	cfg := testTransformerConfig(2)
	weights := testTransformerWeights(cfg)
	prog := wrapperProgram("__call__", []string{"language_model", "lm_head"})
	prog.Weights = testModelIRTransformerWeights(cfg, weights)

	gotCfg, gotWeights, err := DeriveTransformerFromProgram(prog)
	if err != nil {
		t.Fatalf("DeriveTransformerFromProgram: %v", err)
	}
	if gotCfg.NumLayers != cfg.NumLayers {
		t.Fatalf("NumLayers=%d want=%d", gotCfg.NumLayers, cfg.NumLayers)
	}
	if gotCfg.Dim != cfg.Dim || gotCfg.HiddenDim != cfg.HiddenDim {
		t.Fatalf("dims=(%d,%d) want=(%d,%d)", gotCfg.Dim, gotCfg.HiddenDim, cfg.Dim, cfg.HiddenDim)
	}
	if len(gotWeights.Layers) != len(weights.Layers) {
		t.Fatalf("derived layers=%d want=%d", len(gotWeights.Layers), len(weights.Layers))
	}
	if len(gotWeights.FinalNorm) != len(weights.FinalNorm) {
		t.Fatalf("final norm len=%d want=%d", len(gotWeights.FinalNorm), len(weights.FinalNorm))
	}
}

func TestReifyToANEMILDirectProgram(t *testing.T) {
	cfg := testTransformerConfig(1)
	cfg.KVCacheState = true
	cfg.KVCacheMaxLen = cfg.MaxSeqLen
	weights := testTransformerWeights(cfg)
	prog := &modelir.Program{
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
			Ops: []modelir.Op{
				{
					Name:   "read_state",
					Inputs: []string{"l0_k_cache_state"},
					Outputs: []modelir.Value{
						{Name: "k_cache", Type: modelir.TensorType{DType: modelir.DTypeFP16, Shape: modelir.Shape{1, int64(cfg.NumHeads), int64(cfg.KVCacheMaxLen), int64(cfg.HeadDim)}}},
					},
				},
				{
					Name:   "matmul",
					Inputs: []string{"x", "l0_q_w"},
					Outputs: []modelir.Value{
						{Name: "q_proj", Type: modelir.TensorType{DType: modelir.DTypeFP16, Shape: modelir.Shape{1, 1, int64(cfg.AttentionDim)}}},
					},
				},
			},
			Returns: []string{"q_proj"},
		}},
		Weights: testModelIRTransformerWeights(cfg, weights),
	}

	reified, err := ReifyToANEMIL(prog, ReifyOptions{
		TransformerConfig: cfg,
		RequestedLayers:   1,
		SelectedLayers:    1,
	})
	if err != nil {
		t.Fatalf("ReifyToANEMIL direct program: %v", err)
	}
	if reified.Wrapper != WrapperPatternDirectProgram {
		t.Fatalf("Wrapper=%q want=%q", reified.Wrapper, WrapperPatternDirectProgram)
	}
	if !reified.TransformerConfig.KVCacheState {
		t.Fatal("TransformerConfig.KVCacheState=false want true")
	}
	if strings.TrimSpace(reified.MILText) == "" {
		t.Fatal("ReifyToANEMIL direct program returned empty MIL text")
	}
}

func TestReifyToANEMILDeriveRespectsCompatibilityFlags(t *testing.T) {
	cfg := testTransformerConfig(1)
	weights := testTransformerWeights(cfg)
	prog := wrapperProgram("__call__", []string{"language_model", "lm_head"})
	prog.Weights = testModelIRTransformerWeights(cfg, weights)

	reified, err := ReifyToANEMIL(prog, ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			SkipFFN:        true,
			DisableNormOps: true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
	})
	if err != nil {
		t.Fatalf("ReifyToANEMIL compatibility derive: %v", err)
	}
	if strings.Contains(reified.MILText, "l0_gate") {
		t.Fatal("expected SkipFFN=true to omit FFN ops from generated MIL text")
	}
	if strings.Contains(reified.MILText, "l0_in_norm_") {
		t.Fatal("expected DisableNormOps=true to omit RMSNorm ops from generated MIL text")
	}
}

func TestReifyToANEMILDeriveFailsMissingLayerTensor(t *testing.T) {
	cfg := testTransformerConfig(1)
	weights := testTransformerWeights(cfg)
	prog := wrapperProgram("__call__", []string{"language_model", "lm_head"})
	prog.Weights = testModelIRTransformerWeights(cfg, weights)
	delete(prog.Weights, "@model_path/weights/l0_q_w.bin")

	_, err := ReifyToANEMIL(prog, ReifyOptions{
		RequestedLayers: 1,
		SelectedLayers:  1,
	})
	if err == nil || !strings.Contains(err.Error(), `missing layer 0 tensor "q_w"`) {
		t.Fatalf("ReifyToANEMIL missing layer tensor error=%v", err)
	}
}

func TestReifyToANEMILDeriveFailsMissingGlobalTensor(t *testing.T) {
	cfg := testTransformerConfig(1)
	weights := testTransformerWeights(cfg)
	prog := wrapperProgram("__call__", []string{"language_model", "lm_head"})
	prog.Weights = testModelIRTransformerWeights(cfg, weights)
	delete(prog.Weights, "@model_path/weights/rope_cos.bin")

	_, err := ReifyToANEMIL(prog, ReifyOptions{
		RequestedLayers: 1,
		SelectedLayers:  1,
	})
	if err == nil || !strings.Contains(err.Error(), `missing global tensor "rope_cos"`) {
		t.Fatalf("ReifyToANEMIL missing global tensor error=%v", err)
	}
}

func TestReifyModelIRTextToANEMILDerivesFromArchiveWeights(t *testing.T) {
	cfg := testTransformerConfig(1)
	weights := testTransformerWeights(cfg)
	prog := wrapperProgram("__call__", []string{"language_model", "lm_head"})
	prog.Weights = testModelIRTransformerWeights(cfg, weights)
	textData, err := modelirtext.Format(prog)
	if err != nil {
		t.Fatalf("modelirtext.Format: %v", err)
	}

	reified, err := ReifyModelIRTextToANEMIL(textData, ReifyOptions{
		RequestedLayers: 1,
		SelectedLayers:  1,
	})
	if err != nil {
		t.Fatalf("ReifyModelIRTextToANEMIL derive weights: %v", err)
	}
	if strings.TrimSpace(reified.MILText) == "" {
		t.Fatal("ReifyModelIRTextToANEMIL returned empty MIL text")
	}
	if len(reified.WeightFiles) == 0 {
		t.Fatal("ReifyModelIRTextToANEMIL returned no weight files")
	}
}

func TestReifyToANEMILRuntimeSmoke(t *testing.T) {
	if os.Getenv(modelIRReifyRuntimeTestEnv) == "" {
		t.Skipf("set %s=1 to run modelir reify runtime smoke test", modelIRReifyRuntimeTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const pyWrapper = `
class TextModel:
    def __call__(self, inputs, cache=None):
        out = self.language_model(inputs, cache=cache)
        logits = self.lm_head(out)
        return logits
`
	prog, err := python.Parse(pyWrapper)
	if err != nil {
		t.Fatalf("python.Parse: %v", err)
	}

	cfg := testTransformerConfig(1)
	cfg.Dim = 64
	cfg.NumHeads = 4
	cfg.HeadDim = 16
	cfg.HiddenDim = 256
	cfg.MaxSeqLen = 16
	weights := testTransformerWeights(cfg)
	prog.Weights = testModelIRTransformerWeights(cfg, weights)
	textData, err := modelirtext.Format(prog)
	if err != nil {
		t.Fatalf("modelirtext.Format: %v", err)
	}

	reified := ReifiedMIL{}
	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())
	model, gotReified, err := CompileAndLoadModelIRText(client, textData, ReifyOptions{
		TransformerConfig:  cfg,
		TransformerWeights: weights,
		RequestedLayers:    1,
		SelectedLayers:     1,
	}, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		if errors.Is(err, ErrMILDirectoryUnsupported) {
			inMemoryModel, inMemoryReified, inMemoryErr := CompileModelIRTextInMemory(textData, ReifyOptions{
				TransformerConfig:  cfg,
				TransformerWeights: weights,
				RequestedLayers:    1,
				SelectedLayers:     1,
			})
			if inMemoryErr != nil {
				if strings.Contains(inMemoryErr.Error(), "ANECCompile() FAILED") {
					t.Skipf("in-memory MIL compile unavailable on this host: %v", inMemoryErr)
				}
				t.Fatalf("CompileModelIRTextInMemory fallback: %v", inMemoryErr)
			}
			defer func() {
				if unloadErr := callObjCBoolWithNSError(
					"modelir runtime smoke fallback unload",
					inMemoryModel.ID,
					"unloadWithQoS:error:",
					defaultANEQoS,
				); unloadErr != nil {
					t.Logf("warning: fallback unload failed: %v", unloadErr)
				}
			}()
			if strings.TrimSpace(inMemoryReified.MILText) == "" || len(inMemoryReified.WeightFiles) == 0 {
				t.Fatalf("CompileModelIRTextInMemory returned invalid reified artifacts: text=%d files=%d", len(inMemoryReified.MILText), len(inMemoryReified.WeightFiles))
			}
			input := make([]float32, cfg.Dim)
			out, evalErr := evalModelSingleIO(context.Background(), inMemoryModel, input, cfg.Dim, "modelir runtime smoke fallback eval")
			if evalErr != nil && isMILKnownMapFailure(evalErr) {
				out, evalErr = evalModelSingleIOStrictClient(
					context.Background(),
					inMemoryModel,
					input,
					cfg.Dim,
					"modelir runtime smoke fallback eval (daemon-backed)",
				)
				if evalErr != nil && isMILKnownMapFailure(evalErr) {
					t.Skipf("in-memory MIL eval unavailable on this host (mapper failure): %v", evalErr)
				}
			}
			if evalErr != nil {
				t.Fatalf("evalModelSingleIO fallback: %v", evalErr)
			}
			if len(out) != cfg.Dim {
				t.Fatalf("evalModelSingleIO fallback output len=%d want=%d", len(out), cfg.Dim)
			}
			return
		}
		t.Fatalf("CompileAndLoadModelIRText: %v", err)
	}
	reified = gotReified
	defer model.Close()
	if strings.TrimSpace(reified.MILText) == "" || len(reified.WeightFiles) == 0 {
		t.Fatalf("CompileAndLoadModelIRText returned invalid reified artifacts: text=%d files=%d", len(reified.MILText), len(reified.WeightFiles))
	}

	input := make([]float32, cfg.Dim)
	out, _, err := model.EvalSingleIO(context.Background(), input, cfg.Dim, true)
	if err != nil {
		t.Fatalf("EvalSingleIO: %v", err)
	}
	if len(out) != cfg.Dim {
		t.Fatalf("EvalSingleIO output len=%d want=%d", len(out), cfg.Dim)
	}
}

func TestReifyToANEMILStatefulCompileSmoke(t *testing.T) {
	if os.Getenv(modelIRReifyStatefulRuntimeTestEnv) == "" {
		t.Skipf("set %s=1 to run modelir stateful reify compile smoke test", modelIRReifyStatefulRuntimeTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const pyWrapper = `
class TextModel:
    def __call__(self, inputs, cache=None, mask=None):
        out = self.language_model(inputs, cache=cache, mask=mask)
        logits = self.lm_head(out)
        return logits
`
	prog, err := python.Parse(pyWrapper)
	if err != nil {
		t.Fatalf("python.Parse: %v", err)
	}

	cfg := testTransformerConfig(1)
	cfg.Dim = 64
	cfg.NumHeads = 4
	cfg.HeadDim = 16
	cfg.HiddenDim = 256
	cfg.VocabSize = 128
	cfg.MaxSeqLen = 16
	cfg.KVCacheState = true
	cfg.KVCacheMaxLen = 8
	cfg.AttentionMaskInput = true
	cfg.IncludeLMHead = true
	cfg.SkipFFN = true
	weights := testTransformerWeights(cfg)
	weights.LMHeadW = make([]float32, cfg.VocabSize*cfg.Dim)
	weights.LMHeadB = make([]float32, cfg.VocabSize)
	prog.Weights = testModelIRTransformerWeights(cfg, weights)
	textData, err := modelirtext.Format(prog)
	if err != nil {
		t.Fatalf("modelirtext.Format: %v", err)
	}

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	model, reified, err := CompileAndLoadModelIRText(client, textData, ReifyOptions{
		TransformerConfig:  cfg,
		TransformerWeights: weights,
		RequestedLayers:    1,
		SelectedLayers:     1,
	}, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		if errors.Is(err, ErrMILDirectoryUnsupported) {
			inMemoryModel, inMemoryReified, inMemoryErr := CompileModelIRTextInMemory(textData, ReifyOptions{
				TransformerConfig:  cfg,
				TransformerWeights: weights,
				RequestedLayers:    1,
				SelectedLayers:     1,
			})
			if inMemoryErr != nil {
				if strings.Contains(inMemoryErr.Error(), "ANECCompile() FAILED") {
					t.Skipf("stateful in-memory MIL compile unavailable on this host: %v", inMemoryErr)
				}
				t.Fatalf("CompileModelIRTextInMemory stateful fallback: %v", inMemoryErr)
			}
			defer func() {
				if unloadErr := callObjCBoolWithNSError(
					"modelir stateful runtime smoke fallback unload",
					inMemoryModel.ID,
					"unloadWithQoS:error:",
					defaultANEQoS,
				); unloadErr != nil {
					t.Logf("warning: stateful fallback unload failed: %v", unloadErr)
				}
			}()
			for _, want := range []string{
				"read_state(input=l0_k_cache_state)",
				"write_state(data = l0_k_cache_write, input = l0_k_cache_state)",
				"l0_k_cache_next = read_state(input = l0_k_cache_state)",
				"scaled_dot_product_attention(attn_mask=attn_mask_in",
			} {
				if !strings.Contains(inMemoryReified.MILText, want) {
					t.Fatalf("stateful fallback MIL text missing %q", want)
				}
			}
			if len(inMemoryReified.WeightFiles) == 0 {
				t.Fatal("CompileModelIRTextInMemory stateful fallback returned no weight files")
			}
			return
		}
		t.Fatalf("CompileAndLoadModelIRText stateful: %v", err)
	}
	defer model.Close()
	for _, want := range []string{
		"read_state(input=l0_k_cache_state)",
		"write_state(data = l0_k_cache_write, input = l0_k_cache_state)",
		"l0_k_cache_next = read_state(input = l0_k_cache_state)",
		"scaled_dot_product_attention(attn_mask=attn_mask_in",
		"lm_logits",
	} {
		if !strings.Contains(reified.MILText, want) {
			t.Fatalf("stateful MIL text missing %q", want)
		}
	}
	if len(reified.WeightFiles) == 0 {
		t.Fatal("CompileAndLoadModelIRText stateful returned no weight files")
	}
}

func TestReifyToANEMILStatefulRuntimeSmoke(t *testing.T) {
	if os.Getenv(modelIRReifyStatefulEvalTestEnv) == "" {
		t.Skipf("set %s=1 to run modelir stateful reify runtime smoke test", modelIRReifyStatefulEvalTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const pyWrapper = `
class TextModel:
    def __call__(self, inputs, cache=None):
        return self.language_model(inputs, cache=cache)
`
	prog, err := python.Parse(pyWrapper)
	if err != nil {
		t.Fatalf("python.Parse: %v", err)
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
	prog.Weights = testModelIRTransformerWeights(cfg, weights)
	textData, err := modelirtext.Format(prog)
	if err != nil {
		t.Fatalf("modelirtext.Format: %v", err)
	}

	statefulModel, statefulReified := compileModelIRRuntimeTarget(t, textData, ReifyOptions{
		TransformerConfig:  cfg,
		TransformerWeights: weights,
		RequestedLayers:    1,
		SelectedLayers:     1,
	})
	defer statefulModel.Close()
	for _, want := range []string{
		"read_state(input=l0_k_cache_state)",
		"write_state(data = l0_k_cache_write, input = l0_k_cache_state)",
		"l0_k_cache_next = read_state(input = l0_k_cache_state)",
	} {
		if !strings.Contains(statefulReified.MILText, want) {
			t.Fatalf("stateful runtime MIL text missing %q", want)
		}
	}

	statelessCfg := cfg
	statelessCfg.KVCacheState = false
	statelessCfg.KVCacheMaxLen = 0
	statelessModel, _ := compileModelIRRuntimeTarget(t, textData, ReifyOptions{
		TransformerConfig:  statelessCfg,
		TransformerWeights: weights,
		RequestedLayers:    1,
		SelectedLayers:     1,
	})
	defer statelessModel.Close()

	step1 := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	step2 := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	statefulOuts, err := evalModelIRRuntimeSteps(context.Background(), statefulModel, [][]float32{step1, step2}, cfg.Dim)
	if err != nil {
		if isMILKnownMapFailure(err) {
			t.Skipf("stateful runtime eval unavailable on this host (mapper failure): %v", err)
		}
		t.Fatalf("evalModelIRRuntimeSteps(stateful): %v", err)
	}
	if len(statefulOuts) != 2 {
		t.Fatalf("stateful runtime outputs=%d want=2", len(statefulOuts))
	}
	if len(statefulOuts[1]) != cfg.Dim {
		t.Fatalf("stateful runtime second output len=%d want=%d", len(statefulOuts[1]), cfg.Dim)
	}

	statelessOuts, err := evalModelIRRuntimeSteps(context.Background(), statelessModel, [][]float32{step2}, cfg.Dim)
	if err != nil {
		if isMILKnownMapFailure(err) {
			t.Skipf("stateless runtime eval unavailable on this host (mapper failure): %v", err)
		}
		t.Fatalf("evalModelIRRuntimeSteps(stateless): %v", err)
	}
	if len(statelessOuts) != 1 {
		t.Fatalf("stateless runtime outputs=%d want=1", len(statelessOuts))
	}
	if nearlyEqualFloat32s(statefulOuts[1], statelessOuts[0], 1e-3) {
		t.Fatalf("stateful runtime second-step output unexpectedly matched stateless output; state update may be inactive")
	}
}

func TestReifyToANEMILStatefulDraftRuntimeSmoke(t *testing.T) {
	if os.Getenv(modelIRReifyStatefulEvalTestEnv) == "" {
		t.Skipf("set %s=1 to run modelir stateful draft runtime smoke test", modelIRReifyStatefulEvalTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	prog := wrapperProgram("__call__", []string{"language_model"})
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
	prog.Weights = testModelIRTransformerWeights(cfg, weights)

	statefulModel, statefulReified, err := NewANEDraftModelFromModelIRProgram(
		prog,
		ReifyOptions{
			TransformerConfig:  cfg,
			TransformerWeights: weights,
			RequestedLayers:    1,
			SelectedLayers:     1,
		},
		cfg.Dim,
		32,
		cfg.Dim,
		nil,
	)
	if err != nil {
		if strings.Contains(err.Error(), "ANECCompile() FAILED") {
			t.Skipf("stateful draft runtime unavailable on this host: %v", err)
		}
		t.Fatalf("NewANEDraftModelFromModelIRProgram(stateful): %v", err)
	}
	defer statefulModel.Close()
	if profile := compileFallbackProfile(statefulReified); profile != "" {
		t.Logf("stateful draft compile fallback profile=%q", profile)
	}

	step1 := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	step2 := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	if err := statefulModel.Reset(); err != nil {
		t.Fatalf("statefulModel.Reset: %v", err)
	}
	statefulOut1, err := statefulModel.EvalToken(step1)
	if err != nil {
		t.Fatalf("statefulModel.EvalToken(step1): %v", err)
	}
	if len(statefulOut1) != cfg.Dim {
		t.Fatalf("len(statefulOut1)=%d want=%d", len(statefulOut1), cfg.Dim)
	}
	statefulOut2, err := statefulModel.EvalToken(step2)
	if err != nil {
		t.Fatalf("statefulModel.EvalToken(step2): %v", err)
	}
	if len(statefulOut2) != cfg.Dim {
		t.Fatalf("len(statefulOut2)=%d want=%d", len(statefulOut2), cfg.Dim)
	}

	if err := statefulModel.Reset(); err != nil {
		t.Fatalf("statefulModel.Reset(second pass): %v", err)
	}
	resetOut2, err := statefulModel.EvalToken(step2)
	if err != nil {
		t.Fatalf("statefulModel.EvalToken(step2 after reset): %v", err)
	}
	if len(resetOut2) != cfg.Dim {
		t.Fatalf("len(resetOut2)=%d want=%d", len(resetOut2), cfg.Dim)
	}
	if nearlyEqualFloat32s(statefulOut2, resetOut2, 1e-3) {
		t.Fatalf("stateful draft second-step output unexpectedly matched reset step2 output; state update may be inactive")
	}
}

func compileFallbackProfile(reified ReifiedMIL) string {
	for _, d := range reified.Diagnostics {
		if d.Code != "compile_fallback" {
			continue
		}
		return strings.TrimPrefix(d.Message, "in-memory compile fallback applied: ")
	}
	return ""
}

type compiledModelIRRuntimeTarget struct {
	clientModel   *ANEClientMILModel
	inMemoryModel appleneuralengine.ANEInMemoryModel
}

func (m compiledModelIRRuntimeTarget) Close() {
	switch {
	case m.clientModel != nil:
		m.clientModel.Close()
	case m.inMemoryModel.ID != 0:
		_ = callObjCBoolWithNSError(
			"modelir runtime target unload",
			m.inMemoryModel.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
	}
}

func compileModelIRRuntimeTarget(
	t *testing.T,
	textData []byte,
	opts ReifyOptions,
) (compiledModelIRRuntimeTarget, ReifiedMIL) {
	t.Helper()

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	model, reified, err := CompileAndLoadModelIRText(client, textData, opts, defaultMILFastPathKey, defaultANEQoS)
	if err == nil {
		return compiledModelIRRuntimeTarget{clientModel: model}, reified
	}
	if !errors.Is(err, ErrMILDirectoryUnsupported) {
		t.Fatalf("CompileAndLoadModelIRText: %v", err)
	}

	inMemoryModel, inMemoryReified, inMemoryErr := CompileModelIRTextInMemory(textData, opts)
	if inMemoryErr != nil {
		if strings.Contains(inMemoryErr.Error(), "ANECCompile() FAILED") {
			t.Skipf("in-memory MIL compile unavailable on this host: %v", inMemoryErr)
		}
		t.Fatalf("CompileModelIRTextInMemory fallback: %v", inMemoryErr)
	}
	return compiledModelIRRuntimeTarget{inMemoryModel: inMemoryModel}, inMemoryReified
}

func evalModelIRRuntimeSteps(
	ctx context.Context,
	target compiledModelIRRuntimeTarget,
	inputs [][]float32,
	outputCount int,
) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("eval modelir runtime steps: inputs are empty")
	}
	if outputCount <= 0 {
		return nil, fmt.Errorf("eval modelir runtime steps: invalid outputCount=%d", outputCount)
	}

	var (
		base objectivec.IObject
		err  error
	)
	switch {
	case target.clientModel != nil:
		base = target.clientModel.model
	case target.inMemoryModel.ID != 0:
		base = target.inMemoryModel.Model()
	default:
		return nil, fmt.Errorf("eval modelir runtime steps: target model is nil")
	}
	surfs, err := newRuntimeSurfaces(base, len(inputs[0]), outputCount)
	if err != nil {
		return nil, fmt.Errorf("eval modelir runtime steps: allocate IO surfaces: %w", err)
	}
	defer surfs.Close()

	inputBindings := []SurfaceBinding{{Surface: surfs.Input, SymbolIndex: surfs.InputSymbolIndex}}
	inputBindings = append(inputBindings, surfs.StateBindings...)
	outputBindings := []SurfaceBinding{{Surface: surfs.Output, SymbolIndex: surfs.OutputSymbolIndex}}
	cfg := DefaultMultiSurfaceEvalPlanConfig()
	var plan *MultiSurfaceEvalPlan
	switch {
	case target.clientModel != nil:
		plan, err = NewMultiSurfaceEvalPlanWithClientModel(target.clientModel, inputBindings, outputBindings, cfg)
	case target.inMemoryModel.ID != 0:
		plan, err = NewMultiSurfaceEvalPlan(target.inMemoryModel, inputBindings, outputBindings, cfg)
	}
	if err != nil {
		return nil, fmt.Errorf("eval modelir runtime steps: new multi-surface plan: %w", err)
	}
	defer plan.Close()

	outs := make([][]float32, 0, len(inputs))
	inputCount := len(inputs[0])
	for i, input := range inputs {
		if len(input) != inputCount {
			return nil, fmt.Errorf("eval modelir runtime steps: input[%d] len=%d want=%d", i, len(input), inputCount)
		}
		if err := surfs.Input.Write(input); err != nil {
			return nil, fmt.Errorf("eval modelir runtime steps: write input[%d]: %w", i, err)
		}
		if err := plan.Eval(ctx); err != nil {
			return nil, fmt.Errorf("eval modelir runtime steps: eval[%d]: %w", i, err)
		}
		out, err := surfs.Output.Read()
		if err != nil {
			return nil, fmt.Errorf("eval modelir runtime steps: read output[%d]: %w", i, err)
		}
		outs = append(outs, out)
	}
	return outs, nil
}

type runtimeSurfaces struct {
	Input             *IOSurfaceFloat32
	Output            *IOSurfaceFloat32
	InputSymbolIndex  int
	OutputSymbolIndex int
	StateSurfaces     []*IOSurfaceFloat32
	StateBindings     []SurfaceBinding
}

func (r *runtimeSurfaces) Close() {
	if r == nil {
		return
	}
	if r.Output != nil {
		r.Output.Close()
	}
	for _, state := range r.StateSurfaces {
		if state != nil {
			state.Close()
		}
	}
	if r.Input != nil {
		r.Input.Close()
	}
}

func newRuntimeSurfaces(
	base objectivec.IObject,
	inputCount int,
	outputCount int,
) (*runtimeSurfaces, error) {
	if base == nil || base.GetID() == 0 {
		return nil, fmt.Errorf("compiled model is nil")
	}
	schema, err := parseCompiledModelSchema(base)
	if err == nil && len(schema.Inputs) > 0 && len(schema.Outputs) > 0 {
		inSurf, inErr := newIOSurfaceFloat32WithLayout(schema.Inputs[0])
		if inErr != nil {
			return nil, fmt.Errorf("input layout surface: %w", inErr)
		}
		outSurf, outErr := newIOSurfaceFloat32WithLayout(schema.Outputs[0])
		if outErr != nil {
			inSurf.Close()
			return nil, fmt.Errorf("output layout surface: %w", outErr)
		}
		stateSurfs := make([]*IOSurfaceFloat32, 0, len(schema.States))
		stateBindings := make([]SurfaceBinding, 0, len(schema.States))
		for _, layout := range schema.States {
			stateSurf, stateErr := newIOSurfaceFloat32WithLayout(layout)
			if stateErr != nil {
				outSurf.Close()
				inSurf.Close()
				for _, state := range stateSurfs {
					state.Close()
				}
				return nil, fmt.Errorf("state layout surface %q: %w", layout.Name, stateErr)
			}
			if stateErr := stateSurf.Write(make([]float32, stateSurf.Count())); stateErr != nil {
				stateSurf.Close()
				outSurf.Close()
				inSurf.Close()
				for _, state := range stateSurfs {
					state.Close()
				}
				return nil, fmt.Errorf("init state layout surface %q: %w", layout.Name, stateErr)
			}
			stateSurfs = append(stateSurfs, stateSurf)
			stateBindings = append(stateBindings, SurfaceBinding{Surface: stateSurf, SymbolIndex: layout.SymbolIndex})
		}
		return &runtimeSurfaces{
			Input:             inSurf,
			Output:            outSurf,
			InputSymbolIndex:  compiledLayoutSymbolIndex(schema.Inputs, 0),
			OutputSymbolIndex: compiledLayoutSymbolIndex(schema.Outputs, 0),
			StateSurfaces:     stateSurfs,
			StateBindings:     stateBindings,
		}, nil
	}
	inSurf, err := NewIOSurfaceFloat32(inputCount)
	if err != nil {
		return nil, err
	}
	outSurf, err := NewIOSurfaceFloat32(outputCount)
	if err != nil {
		inSurf.Close()
		return nil, err
	}
	return &runtimeSurfaces{
		Input:             inSurf,
		Output:            outSurf,
		InputSymbolIndex:  0,
		OutputSymbolIndex: 0,
	}, nil
}

func nearlyEqualFloat32s(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			return false
		}
	}
	return true
}

func wrapperProgram(name string, ops []string) *modelir.Program {
	irOps := make([]modelir.Op, 0, len(ops))
	prev := "inputs"
	for i, op := range ops {
		outName := "t" + string(rune('0'+i))
		if i == len(ops)-1 {
			outName = "logits"
		}
		irOps = append(irOps, modelir.Op{
			Outputs: []modelir.Value{{
				Name: outName,
				Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{-1}},
			}},
			Name:   op,
			Inputs: []string{prev},
		})
		prev = outName
	}
	return &modelir.Program{
		Entry: name,
		Functions: []modelir.Function{{
			Name: name,
			Inputs: []modelir.Value{
				{Name: "inputs", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{-1}}},
				{Name: "cache", Type: modelir.TensorType{DType: modelir.DTypeFP32, Shape: modelir.Shape{-1}}},
			},
			Ops:     irOps,
			Returns: []string{prev},
		}},
	}
}

func testTransformerConfig(layers int) MILTransformerConfig {
	return MILTransformerConfig{
		NumLayers: layers,
		Dim:       8,
		NumHeads:  2,
		HeadDim:   4,
		HiddenDim: 16,
		MaxSeqLen: 8,
	}
}

func testTransformerWeights(cfg MILTransformerConfig) MILTransformerWeights {
	attnDim := cfg.NumHeads * cfg.HeadDim
	layers := make([]MILTransformerLayerWeights, cfg.NumLayers)
	for i := range layers {
		layers[i] = MILTransformerLayerWeights{
			QW: make([]float32, attnDim*cfg.Dim),
			QB: make([]float32, attnDim),
			KW: make([]float32, attnDim*cfg.Dim),
			KB: make([]float32, attnDim),
			VW: make([]float32, attnDim*cfg.Dim),
			VB: make([]float32, attnDim),
			OW: make([]float32, cfg.Dim*attnDim),
			OB: make([]float32, cfg.Dim),
			W1: make([]float32, cfg.HiddenDim*cfg.Dim),
			B1: make([]float32, cfg.HiddenDim),
			W3: make([]float32, cfg.HiddenDim*cfg.Dim),
			B3: make([]float32, cfg.HiddenDim),
			W2: make([]float32, cfg.Dim*cfg.HiddenDim),
			B2: make([]float32, cfg.Dim),

			InputNorm:         fillOnes(cfg.Dim),
			PostAttentionNorm: fillOnes(cfg.Dim),
			QNorm:             fillOnes(cfg.HeadDim),
			KNorm:             fillOnes(cfg.HeadDim),
		}
	}
	return MILTransformerWeights{
		Layers:    layers,
		FinalNorm: fillOnes(cfg.Dim),
		RopeCos:   fillOnes(cfg.MaxSeqLen * cfg.HeadDim),
		RopeSin:   make([]float32, cfg.MaxSeqLen*cfg.HeadDim),
	}
}

func testTransformerWeightsIdentity(cfg MILTransformerConfig) MILTransformerWeights {
	w := testTransformerWeights(cfg)
	attnDim := cfg.NumHeads * cfg.HeadDim
	for i := range w.Layers {
		layer := &w.Layers[i]
		fillLinearIdentity(layer.QW, attnDim, cfg.Dim, 1)
		fillLinearIdentity(layer.KW, attnDim, cfg.Dim, 1)
		fillLinearIdentity(layer.VW, attnDim, cfg.Dim, 1)
		fillLinearIdentity(layer.OW, cfg.Dim, attnDim, 1)
	}
	return w
}

func fillLinearIdentity(dst []float32, rows, cols int, scale float32) {
	clear(dst)
	n := rows
	if cols < n {
		n = cols
	}
	for i := 0; i < n; i++ {
		dst[i*cols+i] = scale
	}
}

func fillOnes(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func wrapperFixtureSources(t *testing.T) map[string]string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "internal", "modelir", "testdata", "qwen35_real_wrappers.txt"))
	if err != nil {
		return map[string]string{
			"py_wrapper.py": `# adapted from mlx-lm/mlx_lm/models/qwen3_5.py
class TextModel:
    def __call__(self, inputs, cache=None):
        out = self.language_model(inputs, cache=cache)
        logits = self.lm_head(out)
        return logits
`,
			"go_wrapper.go": `// adapted from mlx-go/examples/mlx-go-lm/mlxlm/models/qwen3_next.go
type qwen3NextLanguageModel struct{}

func (m *qwen3NextLanguageModel) Forward(inputs *mlx.Array, cache Cache) (*mlx.Array, Cache) {
	inputs2D := normalizeTokenInputs(inputs)
	out, cache2 := m.model.Forward(inputs2D, m.cacheSlice(cache))
	logits := m.lmHead.Forward(out)
	return logits, cache2
}
`,
			"swift_wrapper.swift": `// adapted from mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
struct Qwen35Model {
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let b = inputs.dim(0)
        let out = languageModel(inputs, cache: cache)
        return reshape(out, b, -1)
    }
}
`,
		}
	}
	ar := txtar.Parse(data)
	files := make(map[string]string, len(ar.Files))
	for _, f := range ar.Files {
		files[f.Name] = string(f.Data)
	}
	return files
}

func testModelIRTransformerWeights(cfg MILTransformerConfig, weights MILTransformerWeights) map[string]*modelir.Weight {
	out := map[string]*modelir.Weight{}
	add := func(path string, shape modelir.Shape, vals []float32) {
		out[path] = &modelir.Weight{
			DType: modelir.DTypeFP32,
			Shape: shape,
			Data:  float32sToBytes(vals),
		}
	}
	attnDim := cfg.NumHeads * cfg.HeadDim
	for i, layer := range weights.Layers {
		prefix := fmt.Sprintf("@model_path/weights/l%d_", i)
		add(prefix+"q_w.bin", modelir.Shape{int64(attnDim), int64(cfg.Dim)}, layer.QW)
		add(prefix+"q_b.bin", modelir.Shape{int64(attnDim)}, layer.QB)
		add(prefix+"k_w.bin", modelir.Shape{int64(attnDim), int64(cfg.Dim)}, layer.KW)
		add(prefix+"k_b.bin", modelir.Shape{int64(attnDim)}, layer.KB)
		add(prefix+"v_w.bin", modelir.Shape{int64(attnDim), int64(cfg.Dim)}, layer.VW)
		add(prefix+"v_b.bin", modelir.Shape{int64(attnDim)}, layer.VB)
		add(prefix+"o_w.bin", modelir.Shape{int64(cfg.Dim), int64(attnDim)}, layer.OW)
		add(prefix+"o_b.bin", modelir.Shape{int64(cfg.Dim)}, layer.OB)
		add(prefix+"w1.bin", modelir.Shape{int64(cfg.HiddenDim), int64(cfg.Dim)}, layer.W1)
		add(prefix+"b1.bin", modelir.Shape{int64(cfg.HiddenDim)}, layer.B1)
		add(prefix+"w3.bin", modelir.Shape{int64(cfg.HiddenDim), int64(cfg.Dim)}, layer.W3)
		add(prefix+"b3.bin", modelir.Shape{int64(cfg.HiddenDim)}, layer.B3)
		add(prefix+"w2.bin", modelir.Shape{int64(cfg.Dim), int64(cfg.HiddenDim)}, layer.W2)
		add(prefix+"b2.bin", modelir.Shape{int64(cfg.Dim)}, layer.B2)
		add(prefix+"input_norm.bin", modelir.Shape{int64(cfg.Dim)}, layer.InputNorm)
		add(prefix+"post_norm.bin", modelir.Shape{int64(cfg.Dim)}, layer.PostAttentionNorm)
		add(prefix+"q_norm.bin", modelir.Shape{int64(cfg.HeadDim)}, layer.QNorm)
		add(prefix+"k_norm.bin", modelir.Shape{int64(cfg.HeadDim)}, layer.KNorm)
	}
	add("@model_path/weights/final_norm.bin", modelir.Shape{int64(cfg.Dim)}, weights.FinalNorm)
	add("@model_path/weights/rope_cos.bin", modelir.Shape{int64(cfg.MaxSeqLen), int64(cfg.HeadDim)}, weights.RopeCos)
	add("@model_path/weights/rope_sin.bin", modelir.Shape{int64(cfg.MaxSeqLen), int64(cfg.HeadDim)}, weights.RopeSin)
	return out
}

func float32sToBytes(vals []float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(v))
	}
	return out
}
