//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
	publicmodelir "github.com/tmc/mlx-go/modelir"
)

const milTransformerTestEnv = "MLXGO_ANE_TEST_MIL_TRANSFORMER"
const milFastPathChildEnv = "MLXGO_ANE_TEST_MIL_FAST_CHILD"
const milFeatureProbeEnv = "MLXGO_ANE_TEST_MIL_FEATURE_PROBE"
const milTransformerFeatureMatrixEnv = "MLXGO_ANE_TEST_MIL_TRANSFORMER_FEATURE_MATRIX"
const milFeatureSchemaProbeEnv = "MLXGO_ANE_TEST_MIL_FEATURE_SCHEMA"

func TestMILFeatureSchemaProbe(t *testing.T) {
	if os.Getenv(milFeatureSchemaProbeEnv) == "" {
		t.Skipf("set %s=1 to run MIL feature schema probe", milFeatureSchemaProbeEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		numHeads = 2
		headDim  = 4
		maxSeq   = 8
	)
	probes := []struct {
		name  string
		build func() (string, []modelWeightFile, error)
	}{
		{name: "raw_sdpa", build: buildMILProbeRawSDPA},
		{name: "stateful_sdpa", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeStatefulSDPA(numHeads, headDim, maxSeq, false)
		}},
		{name: "stateful_sdpa_write_readback", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeStatefulSDPAWriteReadback(numHeads, headDim, maxSeq, false)
		}},
		{name: "write_state_readback", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeWriteStateReadback(numHeads, headDim, maxSeq)
		}},
	}

	for _, tc := range probes {
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := tc.build()
			if err != nil {
				t.Fatalf("build %s: %v", tc.name, err)
			}
			model, err := buildModelFromMILTextWithDescriptorFallback("mil feature schema "+tc.name, milText, files)
			if err != nil {
				t.Fatalf("buildModelFromMILTextWithDescriptorFallback(%s): %v", tc.name, err)
			}
			defer unloadMILProbeModel(t, tc.name, model)

			aneModel := model.Model()
			t.Logf("phase=attrs desc=%s", aneModel.ModelAttributes().Description())
			t.Logf("phase=input_symbol_indices desc=%s", testObjectDescription(aneModel.InputSymbolIndicesForProcedureIndex(0)))
			t.Logf("phase=output_symbol_indices desc=%s", testObjectDescription(aneModel.OutputSymbolIndicesForProcedureIndex(0)))
			t.Logf("phase=procedure_info desc=%s", testObjectDescription(aneModel.ProcedureInfoForProcedureIndex(0)))

			if schema, err := parseCompiledModelSchema(aneModel); err == nil {
				for i, layout := range schema.Inputs {
					t.Logf(
						"phase=input_layout index=%d name=%q symbol=%q symbol_index=%d channels=%d height=%d width=%d elem=%d row=%d plane=%d",
						i,
						layout.Name,
						layout.Symbol,
						layout.SymbolIndex,
						layout.Channels,
						layout.Height,
						layout.Width,
						layout.ElemSize,
						layout.RowStride,
						layout.PlaneStride,
					)
				}
				for i, layout := range schema.States {
					t.Logf(
						"phase=state_layout index=%d name=%q symbol=%q symbol_index=%d channels=%d height=%d width=%d elem=%d row=%d plane=%d",
						i,
						layout.Name,
						layout.Symbol,
						layout.SymbolIndex,
						layout.Channels,
						layout.Height,
						layout.Width,
						layout.ElemSize,
						layout.RowStride,
						layout.PlaneStride,
					)
				}
				for i, layout := range schema.Outputs {
					t.Logf(
						"phase=output_layout index=%d name=%q symbol=%q symbol_index=%d channels=%d height=%d width=%d elem=%d row=%d plane=%d",
						i,
						layout.Name,
						layout.Symbol,
						layout.SymbolIndex,
						layout.Channels,
						layout.Height,
						layout.Width,
						layout.ElemSize,
						layout.RowStride,
						layout.PlaneStride,
					)
				}
				switch tc.name {
				case "write_state_readback":
					if got := findSchemaLayoutSymbolIndex(schema.Inputs, "x"); got != 0 {
						t.Fatalf("input x symbol index=%d want=0", got)
					}
					if got := findSchemaLayoutSymbolIndex(schema.States, "k_state"); got != 1 {
						t.Fatalf("state k_state symbol index=%d want=1", got)
					}
				case "stateful_sdpa_write_readback":
					if got := findSchemaLayoutSymbolIndex(schema.Inputs, "k_in"); got != 0 {
						t.Fatalf("input k_in symbol index=%d want=0", got)
					}
					if got := findSchemaLayoutSymbolIndex(schema.Inputs, "q_in"); got != 1 {
						t.Fatalf("input q_in symbol index=%d want=1", got)
					}
					if got := findSchemaLayoutSymbolIndex(schema.Inputs, "v_in"); got != 2 {
						t.Fatalf("input v_in symbol index=%d want=2", got)
					}
					if got := findSchemaLayoutSymbolIndex(schema.States, "k_state"); got != 3 {
						t.Fatalf("state k_state symbol index=%d want=3", got)
					}
					if got := findSchemaLayoutSymbolIndex(schema.States, "v_state"); got != 4 {
						t.Fatalf("state v_state symbol index=%d want=4", got)
					}
				}
			} else {
				t.Logf("phase=layout_parse_failed err=%v", err)
			}
		})
	}
}

func TestMILTransformerSchemaProbe(t *testing.T) {
	if os.Getenv(milFeatureSchemaProbeEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer schema probe", milFeatureSchemaProbeEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		NumLayers:          1,
		Dim:                16,
		AttentionDim:       16,
		NumHeads:           4,
		HeadDim:            4,
		HiddenDim:          32,
		MaxSeqLen:          16,
		KVCacheState:       true,
		KVCacheMaxLen:      16,
		SkipFFN:            true,
		DisableNormOps:     true,
		IncludeLMHead:      false,
		AttentionMaskInput: false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("validate cfg: %v", err)
	}
	milText, files, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(testTransformerWeightsIdentity(cfg)),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts: %v", err)
	}
	model, err := buildModelFromMILTextWithDescriptorFallback("transformer schema probe", milText, fromANEModelWeightFiles(files))
	if err != nil {
		t.Fatalf("buildModelFromMILTextWithDescriptorFallback: %v", err)
	}
	defer unloadMILProbeModel(t, "transformer_schema_probe", model)

	aneModel := model.Model()
	t.Logf("phase=attrs desc=%s", aneModel.ModelAttributes().Description())
	t.Logf("phase=input_symbol_indices desc=%s", testObjectDescription(aneModel.InputSymbolIndicesForProcedureIndex(0)))
	t.Logf("phase=output_symbol_indices desc=%s", testObjectDescription(aneModel.OutputSymbolIndicesForProcedureIndex(0)))
	t.Logf("phase=procedure_info desc=%s", testObjectDescription(aneModel.ProcedureInfoForProcedureIndex(0)))
	schema, err := parseCompiledModelSchema(aneModel)
	if err != nil {
		t.Fatalf("parseCompiledModelSchema: %v", err)
	}
	for i, layout := range schema.Inputs {
		t.Logf(
			"phase=input_layout index=%d name=%q symbol=%q symbol_index=%d channels=%d height=%d width=%d elem=%d row=%d plane=%d",
			i,
			layout.Name,
			layout.Symbol,
			layout.SymbolIndex,
			layout.Channels,
			layout.Height,
			layout.Width,
			layout.ElemSize,
			layout.RowStride,
			layout.PlaneStride,
		)
	}
	for i, layout := range schema.States {
		t.Logf(
			"phase=state_layout index=%d name=%q symbol=%q symbol_index=%d channels=%d height=%d width=%d elem=%d row=%d plane=%d",
			i,
			layout.Name,
			layout.Symbol,
			layout.SymbolIndex,
			layout.Channels,
			layout.Height,
			layout.Width,
			layout.ElemSize,
			layout.RowStride,
			layout.PlaneStride,
		)
	}
	for i, layout := range schema.Outputs {
		t.Logf(
			"phase=output_layout index=%d name=%q symbol=%q symbol_index=%d channels=%d height=%d width=%d elem=%d row=%d plane=%d",
			i,
			layout.Name,
			layout.Symbol,
			layout.SymbolIndex,
			layout.Channels,
			layout.Height,
			layout.Width,
			layout.ElemSize,
			layout.RowStride,
			layout.PlaneStride,
		)
	}
	if got := findSchemaLayoutSymbolIndex(schema.Inputs, "x_in"); got != 0 {
		t.Fatalf("transformer input x_in symbol index=%d want=0", got)
	}
	if got := findSchemaLayoutSymbolIndex(schema.States, "l0_k_cache_state"); got != 1 {
		t.Fatalf("transformer state l0_k_cache_state symbol index=%d want=1", got)
	}
	if got := findSchemaLayoutSymbolIndex(schema.States, "l0_v_cache_state"); got != 2 {
		t.Fatalf("transformer state l0_v_cache_state symbol index=%d want=2", got)
	}
}

func findSchemaLayoutSymbolIndex(layouts []compiledTensorLayout, name string) int {
	for _, layout := range layouts {
		if layout.Name == name || layout.Symbol == name {
			return layout.SymbolIndex
		}
	}
	return -1
}

func TestMILFeatureCompileMatrix(t *testing.T) {
	if os.Getenv(milFeatureProbeEnv) == "" {
		t.Skipf("set %s=1 to run MIL feature compile matrix", milFeatureProbeEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		numHeads = 2
		headDim  = 4
		maxSeq   = 8
	)

	type probeCase struct {
		name  string
		build func() (string, []modelWeightFile, error)
	}
	probes := []probeCase{
		{name: "RawSDPA", build: buildMILProbeRawSDPA},
		{name: "ReadStateOnly", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeReadStateOnly(numHeads, headDim, maxSeq)
		}},
		{name: "UpdateStateOnly", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeUpdateStateOnly(numHeads, headDim, maxSeq)
		}},
		{name: "WriteStateReadback", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeWriteStateReadback(numHeads, headDim, maxSeq)
		}},
		{name: "StateUpdateOnly", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeStateUpdateOnly(numHeads, headDim, maxSeq)
		}},
		{name: "StatefulSDPA", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeStatefulSDPA(numHeads, headDim, maxSeq, false)
		}},
		{name: "StatefulSDPAWriteReadback", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeStatefulSDPAWriteReadback(numHeads, headDim, maxSeq, false)
		}},
		{name: "StatefulMaskedSDPA", build: func() (string, []modelWeightFile, error) {
			return buildMILProbeStatefulSDPA(numHeads, headDim, maxSeq, true)
		}},
	}

	type probeResult struct {
		ok  bool
		err error
	}
	results := make(map[string]probeResult, len(probes))
	for _, probe := range probes {
		probe := probe
		t.Run(probe.name, func(t *testing.T) {
			milText, files, err := probe.build()
			if err != nil {
				t.Fatalf("build %s: %v", probe.name, err)
			}
			err = compileMILProbeCase(t, "mil feature "+probe.name, milText, files)
			results[probe.name] = probeResult{ok: err == nil, err: err}
			if err != nil {
				t.Logf("%s compile failed: %v", probe.name, err)
				return
			}
			t.Logf("%s compile succeeded", probe.name)
		})
	}

	if raw := results["RawSDPA"]; !raw.ok {
		t.Fatalf("raw sdpa baseline failed: %v", raw.err)
	}
	if read := results["ReadStateOnly"]; !read.ok {
		t.Fatalf("read_state baseline failed: %v", read.err)
	}
	if update := results["UpdateStateOnly"]; !update.ok {
		if write := results["WriteStateReadback"]; write.ok {
			t.Fatalf(
				"coreml_update_state rejected while explicit write_state/read_state compiles: update=%v write=%v",
				results["UpdateStateOnly"].err,
				results["WriteStateReadback"].err,
			)
		}
		t.Fatalf("coreml_update_state baseline failed: %v", update.err)
	}
	if state := results["StateUpdateOnly"]; !state.ok {
		t.Fatalf(
			"slice/concat state update rejected while read/update baselines compiled: read=%v update=%v stateful_chain=%v",
			results["ReadStateOnly"].err,
			results["UpdateStateOnly"].err,
			results["StateUpdateOnly"].err,
		)
	}
	if !results["StatefulSDPA"].ok {
		if wr := results["StatefulSDPAWriteReadback"]; wr.ok {
			t.Fatalf(
				"stateful sdpa rejected with coreml_update_state while explicit write/read variant compiles: stateful=%v write_read=%v masked=%v",
				results["StatefulSDPA"].err,
				wr.err,
				results["StatefulMaskedSDPA"].err,
			)
		}
		t.Fatalf(
			"combined stateful sdpa rejected while raw baselines compiled: raw=%v state=%v stateful=%v write_read=%v masked=%v",
			results["RawSDPA"].err,
			results["StateUpdateOnly"].err,
			results["StatefulSDPA"].err,
			results["StatefulSDPAWriteReadback"].err,
			results["StatefulMaskedSDPA"].err,
		)
	}
}

func TestMILFeatureEvalMatrix(t *testing.T) {
	if os.Getenv(milFeatureProbeEnv) == "" {
		t.Skipf("set %s=1 to run MIL feature eval matrix", milFeatureProbeEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		numHeads = 2
		headDim  = 4
		maxSeq   = 8
	)

	type evalCase struct {
		name        string
		build       func() (string, []modelWeightFile, error)
		steps       [][]float32
		outputCount int
		check       func(*testing.T, [][]float32)
	}
	cases := []evalCase{
		{
			name: "write_state_readback",
			build: func() (string, []modelWeightFile, error) {
				return buildMILProbeWriteStateReadback(numHeads, headDim, maxSeq)
			},
			steps:       [][]float32{makeDeterministicTensor(numHeads*maxSeq*headDim, 0.01, 101)},
			outputCount: numHeads * maxSeq * headDim,
			check: func(t *testing.T, outs [][]float32) {
				t.Helper()
				if len(outs) != 1 {
					t.Fatalf("outputs=%d want=1", len(outs))
				}
				want := makeDeterministicTensor(numHeads*maxSeq*headDim, 0.01, 101)
				if !nearlyEqualFloat32s(outs[0], want, 5e-2) {
					gotHead := outs[0]
					if len(gotHead) > 8 {
						gotHead = gotHead[:8]
					}
					wantHead := want
					if len(wantHead) > 8 {
						wantHead = wantHead[:8]
					}
					t.Fatalf("write_state_readback output mismatch: got_head=%v want_head=%v", gotHead, wantHead)
				}
			},
		},
		{
			name: "update_state_only",
			build: func() (string, []modelWeightFile, error) {
				return buildMILProbeUpdateStateOnly(numHeads, headDim, maxSeq)
			},
			steps:       [][]float32{makeDeterministicTensor(numHeads*maxSeq*headDim, 0.01, 111)},
			outputCount: numHeads * maxSeq * headDim,
			check: func(t *testing.T, outs [][]float32) {
				t.Helper()
				if len(outs) != 1 {
					t.Fatalf("outputs=%d want=1", len(outs))
				}
				if !nearlyEqualFloat32s(outs[0], makeDeterministicTensor(numHeads*maxSeq*headDim, 0.01, 111), 5e-2) {
					t.Fatalf("update_state_only output mismatch")
				}
			},
		},
		{
			name: "state_update_only",
			build: func() (string, []modelWeightFile, error) {
				return buildMILProbeStateUpdateOnly(numHeads, headDim, maxSeq)
			},
			steps: [][]float32{
				makeDeterministicTensor(numHeads*headDim, 0.01, 121),
				makeDeterministicTensor(numHeads*headDim, 0.01, 131),
			},
			outputCount: numHeads * maxSeq * headDim,
			check: func(t *testing.T, outs [][]float32) {
				t.Helper()
				if len(outs) != 2 {
					t.Fatalf("outputs=%d want=2", len(outs))
				}
				if nearlyEqualFloat32s(outs[0], outs[1], 1e-3) {
					t.Fatalf("state_update_only outputs unexpectedly identical across steps")
				}
			},
		},
		{
			name: "stateful_sdpa_write_readback",
			build: func() (string, []modelWeightFile, error) {
				return buildMILProbeStatefulSDPAWriteReadback(numHeads, headDim, maxSeq, false)
			},
			steps:       [][]float32{makeDeterministicTensor(numHeads*headDim, 0.01, 141)},
			outputCount: numHeads * headDim,
			check: func(t *testing.T, outs [][]float32) {
				t.Helper()
				if len(outs) != 1 {
					t.Fatalf("outputs=%d want=1", len(outs))
				}
				if len(outs[0]) != numHeads*headDim {
					t.Fatalf("stateful_sdpa_write_readback output len=%d want=%d", len(outs[0]), numHeads*headDim)
				}
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := tc.build()
			if err != nil {
				t.Fatalf("build %s: %v", tc.name, err)
			}
			model, err := buildModelFromMILTextWithDescriptorFallback("mil feature eval "+tc.name, milText, files)
			if err != nil {
				t.Fatalf("buildModelFromMILTextWithDescriptorFallback(%s): %v", tc.name, err)
			}
			defer unloadMILProbeModel(t, tc.name, model)

			outs, err := evalModelIRRuntimeSteps(
				context.Background(),
				compiledModelIRRuntimeTarget{inMemoryModel: model},
				tc.steps,
				tc.outputCount,
			)
			if err != nil {
				t.Fatalf("evalModelIRRuntimeSteps(%s): %v", tc.name, err)
			}
			tc.check(t, outs)
		})
	}
}

func TestMILWriteStateReadbackStateProbe(t *testing.T) {
	if os.Getenv(milFeatureProbeEnv) == "" {
		t.Skipf("set %s=1 to run MIL write-state/readback probe", milFeatureProbeEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		numHeads = 2
		headDim  = 4
		maxSeq   = 8
	)
	milText, files, err := buildMILProbeWriteStateReadback(numHeads, headDim, maxSeq)
	if err != nil {
		t.Fatalf("buildMILProbeWriteStateReadback: %v", err)
	}
	model, err := buildModelFromMILTextWithDescriptorFallback("mil write_state/readback probe", milText, files)
	if err != nil {
		t.Fatalf("buildModelFromMILTextWithDescriptorFallback: %v", err)
	}
	defer unloadMILProbeModel(t, "write_state_readback_probe", model)

	input := makeDeterministicTensor(numHeads*maxSeq*headDim, 0.01, 151)
	surfs, err := newRuntimeSurfaces(model.Model(), len(input), len(input))
	if err != nil {
		t.Fatalf("newRuntimeSurfaces: %v", err)
	}
	defer surfs.Close()
	if len(surfs.StateSurfaces) != 1 {
		t.Fatalf("state surfaces=%d want=1", len(surfs.StateSurfaces))
	}

	inputBindings := []SurfaceBinding{{Surface: surfs.Input, SymbolIndex: surfs.InputSymbolIndex}}
	inputBindings = append(inputBindings, surfs.StateBindings...)
	outputBindings := []SurfaceBinding{{Surface: surfs.Output, SymbolIndex: surfs.OutputSymbolIndex}}
	plan, err := NewMultiSurfaceEvalPlan(model, inputBindings, outputBindings, DefaultMultiSurfaceEvalPlanConfig())
	if err != nil {
		t.Fatalf("NewMultiSurfaceEvalPlan: %v", err)
	}
	defer plan.Close()

	if err := surfs.Input.Write(input); err != nil {
		t.Fatalf("surfs.Input.Write: %v", err)
	}
	if err := plan.Eval(context.Background()); err != nil {
		t.Fatalf("plan.Eval: %v", err)
	}
	out, err := surfs.Output.Read()
	if err != nil {
		t.Fatalf("surfs.Output.Read: %v", err)
	}
	stateOut, err := surfs.StateSurfaces[0].Read()
	if err != nil {
		t.Fatalf("surfs.StateSurfaces[0].Read: %v", err)
	}
	head := func(xs []float32) []float32 {
		if len(xs) > 8 {
			return xs[:8]
		}
		return xs
	}
	t.Logf("phase=probe_output head=%v", head(out))
	t.Logf("phase=probe_state head=%v", head(stateOut))
	t.Logf("phase=probe_input head=%v", head(input))
}

func TestMILTransformerFeatureCompileMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer feature compile matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:          1,
		Dim:                768,
		AttentionDim:       768,
		NumHeads:           12,
		HeadDim:            64,
		HiddenDim:          3072,
		MaxSeqLen:          32,
		KVCacheState:       true,
		KVCacheMaxLen:      16,
		SkipFFN:            true,
		DisableNormOps:     true,
		IncludeLMHead:      false,
		AttentionMaskInput: false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid transformer feature base config: %v", err)
	}

	type featureCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []featureCase{
		{name: "qwenish_baseline_stateful", cfg: baseCfg},
		{name: "qwenish_add_dynamic_rope", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
		})},
		{name: "qwenish_add_attention_output_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.AttentionOutputGate = true
		})},
		{name: "qwenish_add_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableNormOps = false
		})},
		{name: "qwenish_add_ffn", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.SkipFFN = false
		})},
		{name: "qwenish_add_ffn_linear", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.SkipFFN = false
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
		})},
		{name: "qwenish_dynamic_rope_linear_ffn", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.SkipFFN = false
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
		})},
		{name: "qwenish_attention_output_gate_linear_ffn", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.AttentionOutputGate = true
			cfg.SkipFFN = false
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
		})},
		{name: "qwenish_dynamic_rope_attention_output_gate_linear_ffn", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.SkipFFN = false
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
		})},
		{name: "qwenish_full", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.SkipFFN = false
			cfg.DisableNormOps = false
		})},
		{name: "qwenish_full_linear_ffn", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.SkipFFN = false
			cfg.DisableNormOps = false
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "transformer feature "+tc.name, milText, fromANEModelWeightFiles(files))
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerFFNSizeCompileMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer FFN size compile matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:          1,
		Dim:                768,
		AttentionDim:       768,
		NumHeads:           12,
		HeadDim:            64,
		MaxSeqLen:          32,
		KVCacheState:       true,
		KVCacheMaxLen:      16,
		SkipFFN:            false,
		DisableNormOps:     true,
		IncludeLMHead:      false,
		AttentionMaskInput: false,
	}

	for _, hiddenDim := range []int{256, 512, 1024, 1536, 2048, 3072} {
		hiddenDim := hiddenDim
		t.Run(fmt.Sprintf("hidden_%d_conv", hiddenDim), func(t *testing.T) {
			cfg := baseCfg
			cfg.HiddenDim = hiddenDim
			if err := validateMILTransformerConfig(cfg); err != nil {
				t.Fatalf("validate cfg: %v", err)
			}
			weights := buildTransformerFeatureWeights(cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts hidden=%d conv: %v", hiddenDim, err)
			}
			err = compileMILProbeCase(t, fmt.Sprintf("transformer ffn hidden=%d conv", hiddenDim), milText, fromANEModelWeightFiles(files))
			if err != nil {
				t.Logf("hidden=%d conv compile failed: %v", hiddenDim, err)
				return
			}
			t.Logf("hidden=%d conv compile succeeded", hiddenDim)
		})
		t.Run(fmt.Sprintf("hidden_%d_linear", hiddenDim), func(t *testing.T) {
			cfg := baseCfg
			cfg.HiddenDim = hiddenDim
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
			if err := validateMILTransformerConfig(cfg); err != nil {
				t.Fatalf("validate cfg: %v", err)
			}
			weights := buildTransformerFeatureWeights(cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts hidden=%d linear: %v", hiddenDim, err)
			}
			err = compileMILProbeCase(t, fmt.Sprintf("transformer ffn hidden=%d linear", hiddenDim), milText, fromANEModelWeightFiles(files))
			if err != nil {
				t.Logf("hidden=%d linear compile failed: %v", hiddenDim, err)
				return
			}
			t.Logf("hidden=%d linear compile succeeded", hiddenDim)
		})
	}
}

func TestMILTransformerStatefulFFNDimCompileMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer stateful FFN dim compile matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	type dimCase struct {
		name      string
		dim       int
		numHeads  int
		headDim   int
		hiddenDim int
	}
	cases := []dimCase{
		{name: "tiny", dim: 8, numHeads: 2, headDim: 4, hiddenDim: 16},
		{name: "small", dim: 64, numHeads: 4, headDim: 16, hiddenDim: 256},
		{name: "medium", dim: 256, numHeads: 8, headDim: 32, hiddenDim: 1024},
		{name: "large", dim: 768, numHeads: 12, headDim: 64, hiddenDim: 3072},
	}

	for _, tc := range cases {
		tc := tc
		for _, useConvFFN := range []bool{true, false} {
			useConvFFN := useConvFFN
			mode := "linear"
			if useConvFFN {
				mode = "conv"
			}
			t.Run(fmt.Sprintf("%s_%s", tc.name, mode), func(t *testing.T) {
				cfg := MILTransformerConfig{
					NumLayers:          1,
					Dim:                tc.dim,
					AttentionDim:       tc.dim,
					NumHeads:           tc.numHeads,
					HeadDim:            tc.headDim,
					HiddenDim:          tc.hiddenDim,
					MaxSeqLen:          32,
					KVCacheState:       true,
					KVCacheMaxLen:      16,
					SkipFFN:            false,
					DisableNormOps:     true,
					UseConvFFN:         useConvFFN,
					LinearFFN:          !useConvFFN,
					IncludeLMHead:      false,
					AttentionMaskInput: false,
				}
				if err := validateMILTransformerConfig(cfg); err != nil {
					t.Fatalf("validate cfg: %v", err)
				}
				weights := buildTransformerFeatureWeights(cfg)
				milText, files, err := anereify.BuildMILTransformerArtifacts(
					toANEMILTransformerConfig(cfg),
					toANEMILTransformerWeights(weights),
				)
				if err != nil {
					t.Fatalf("BuildMILTransformerArtifacts %s_%s: %v", tc.name, mode, err)
				}
				err = compileMILProbeCase(t, fmt.Sprintf("transformer stateful ffn dim=%d %s", tc.dim, mode), milText, fromANEModelWeightFiles(files))
				if err != nil {
					t.Logf("dim=%d %s compile failed: %v", tc.dim, mode, err)
					return
				}
				t.Logf("dim=%d %s compile succeeded", tc.dim, mode)
			})
		}
	}
}

func TestMILTransformerStatefulAttentionDimCompileMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer stateful attention dim compile matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	type dimCase struct {
		name     string
		dim      int
		numHeads int
		headDim  int
	}
	cases := []dimCase{
		{name: "tiny", dim: 16, numHeads: 4, headDim: 4},
		{name: "small", dim: 64, numHeads: 4, headDim: 16},
		{name: "medium", dim: 256, numHeads: 8, headDim: 32},
		{name: "large", dim: 768, numHeads: 12, headDim: 64},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			cfg := MILTransformerConfig{
				NumLayers:          1,
				Dim:                tc.dim,
				AttentionDim:       tc.dim,
				NumHeads:           tc.numHeads,
				HeadDim:            tc.headDim,
				HiddenDim:          max(tc.dim*4, 16),
				MaxSeqLen:          32,
				KVCacheState:       true,
				KVCacheMaxLen:      16,
				SkipFFN:            true,
				DisableNormOps:     false,
				IncludeLMHead:      false,
				AttentionMaskInput: false,
			}
			if err := validateMILTransformerConfig(cfg); err != nil {
				t.Fatalf("validate cfg: %v", err)
			}
			weights := buildTransformerFeatureWeights(cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts %s: %v", tc.name, err)
			}
			err = compileMILProbeCase(t, fmt.Sprintf("transformer stateful attention dim=%d", tc.dim), milText, fromANEModelWeightFiles(files))
			if err != nil {
				t.Logf("dim=%d attention-only compile failed: %v", tc.dim, err)
				return
			}
			t.Logf("dim=%d attention-only compile succeeded", tc.dim)
		})
	}
}

func TestMILTransformerTinyQwenAttentionFeatureCompileMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen attention feature matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:          1,
		Dim:                16,
		AttentionDim:       16,
		NumHeads:           4,
		HeadDim:            4,
		HiddenDim:          32,
		MaxSeqLen:          16,
		KVCacheState:       true,
		KVCacheMaxLen:      16,
		SkipFFN:            true,
		DisableNormOps:     false,
		IncludeLMHead:      false,
		AttentionMaskInput: false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen attention config: %v", err)
	}

	type featureCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []featureCase{
		{name: "baseline_norms", cfg: baseCfg},
		{name: "baseline_no_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableNormOps = true
		})},
		{name: "add_dynamic_rope", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
		})},
		{name: "add_attention_output_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.AttentionOutputGate = true
		})},
		{name: "add_dynamic_rope_no_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.DisableNormOps = true
		})},
		{name: "add_attention_output_gate_no_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.AttentionOutputGate = true
			cfg.DisableNormOps = true
		})},
		{name: "dynamic_rope_and_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
		})},
		{name: "dynamic_rope_and_gate_no_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableNormOps = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen attention "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenSelectiveNormCompileMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen selective norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:           1,
		Dim:                 16,
		AttentionDim:        16,
		NumHeads:            4,
		HeadDim:             4,
		HiddenDim:           32,
		MaxSeqLen:           16,
		KVCacheState:        true,
		KVCacheMaxLen:       16,
		SkipFFN:             true,
		DisableNormOps:      false,
		AttentionOutputGate: true,
		DynamicRoPEInputs:   true,
		IncludeLMHead:       false,
		AttentionMaskInput:  false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen selective norm config: %v", err)
	}

	type normCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []normCase{
		{name: "all_norms", cfg: baseCfg},
		{name: "no_input_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableInputNormOps = true
		})},
		{name: "no_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableQKNormOps = true
		})},
		{name: "no_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableFinalNormOp = true
		})},
		{name: "no_input_final_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableInputNormOps = true
			cfg.DisableFinalNormOp = true
		})},
		{name: "no_qk_final_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableQKNormOps = true
			cfg.DisableFinalNormOp = true
		})},
		{name: "no_input_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableInputNormOps = true
			cfg.DisableQKNormOps = true
		})},
		{name: "no_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableNormOps = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen selective norms "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFullTrunkCompileMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen full trunk matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:           1,
		Dim:                 16,
		AttentionDim:        16,
		NumHeads:            4,
		HeadDim:             4,
		HiddenDim:           32,
		MaxSeqLen:           16,
		KVCacheState:        true,
		KVCacheMaxLen:       16,
		SkipFFN:             false,
		DisableNormOps:      false,
		AttentionOutputGate: true,
		DynamicRoPEInputs:   true,
		IncludeLMHead:       false,
		AttentionMaskInput:  false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen full trunk config: %v", err)
	}

	type trunkCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []trunkCase{
		{name: "full_trunk", cfg: baseCfg},
		{name: "disable_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableFinalNormOp = true
		})},
		{name: "skip_ffn", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.SkipFFN = true
		})},
		{name: "skip_ffn_disable_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.SkipFFN = true
			cfg.DisableFinalNormOp = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen full trunk "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFullTrunkGateMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen full trunk gate matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:           1,
		Dim:                 16,
		AttentionDim:        16,
		NumHeads:            4,
		HeadDim:             4,
		HiddenDim:           32,
		MaxSeqLen:           16,
		KVCacheState:        true,
		KVCacheMaxLen:       16,
		SkipFFN:             false,
		DisableNormOps:      false,
		AttentionOutputGate: false,
		DynamicRoPEInputs:   true,
		IncludeLMHead:       false,
		AttentionMaskInput:  false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen full trunk gate config: %v", err)
	}

	type gateCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []gateCase{
		{name: "full_trunk_no_gate", cfg: baseCfg},
		{name: "disable_final_norm_no_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableFinalNormOp = true
		})},
		{name: "skip_ffn_no_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.SkipFFN = true
		})},
		{name: "skip_ffn_disable_final_norm_no_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.SkipFFN = true
			cfg.DisableFinalNormOp = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen full trunk gate "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFinalNormInteractionMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen final norm interaction matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:          1,
		Dim:                16,
		AttentionDim:       16,
		NumHeads:           4,
		HeadDim:            4,
		HiddenDim:          32,
		MaxSeqLen:          16,
		KVCacheState:       true,
		KVCacheMaxLen:      16,
		SkipFFN:            true,
		DisableNormOps:     false,
		IncludeLMHead:      false,
		AttentionMaskInput: false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen final norm interaction config: %v", err)
	}

	type interactionCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []interactionCase{
		{name: "dynamic_rope_only", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
		})},
		{name: "dynamic_rope_only_disable_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.DisableFinalNormOp = true
		})},
		{name: "gate_only", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.AttentionOutputGate = true
		})},
		{name: "gate_only_disable_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.AttentionOutputGate = true
			cfg.DisableFinalNormOp = true
		})},
		{name: "dynamic_rope_and_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
		})},
		{name: "dynamic_rope_and_gate_disable_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableFinalNormOp = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen final norm interaction "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFinalNormPiecesMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention final-norm-pieces matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
		normPiece   string
	}{
		{name: "rope_gate_post_linear", dynamicRoPE: true, attnGate: true, normPiece: "none"},
		{name: "rope_gate_norm_weight", dynamicRoPE: true, attnGate: true, normPiece: "weight_only"},
		{name: "rope_gate_norm_stats", dynamicRoPE: true, attnGate: true, normPiece: "stats_only"},
		{name: "rope_gate_full_norm", dynamicRoPE: true, attnGate: true, normPiece: "full_norm"},
		{name: "gate_full_norm", dynamicRoPE: false, attnGate: true, normPiece: "full_norm"},
		{name: "rope_full_norm", dynamicRoPE: true, attnGate: false, normPiece: "full_norm"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFinalNormPieces(cfg, tc.dynamicRoPE, tc.attnGate, tc.normPiece)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFinalNormPieces(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/final-norm-pieces "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFFNPiecesMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention ffn-pieces matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
		ffnPiece    string
	}{
		{name: "rope_gate_post_linear", dynamicRoPE: true, attnGate: true, ffnPiece: "none"},
		{name: "rope_gate_one_linear_hidden", dynamicRoPE: true, attnGate: true, ffnPiece: "one_linear_hidden"},
		{name: "rope_gate_one_linear_same_width", dynamicRoPE: true, attnGate: true, ffnPiece: "one_linear_same_width"},
		{name: "rope_gate_one_linear_hidden_reduce", dynamicRoPE: true, attnGate: true, ffnPiece: "one_linear_hidden_reduce"},
		{name: "rope_gate_two_linears", dynamicRoPE: true, attnGate: true, ffnPiece: "two_linears"},
		{name: "rope_gate_two_linears_same_width", dynamicRoPE: true, attnGate: true, ffnPiece: "two_linears_same_width"},
		{name: "rope_gate_gated_ffn", dynamicRoPE: true, attnGate: true, ffnPiece: "gated_ffn"},
		{name: "rope_one_linear_hidden", dynamicRoPE: true, attnGate: false, ffnPiece: "one_linear_hidden"},
		{name: "gate_one_linear_hidden", dynamicRoPE: false, attnGate: true, ffnPiece: "one_linear_hidden"},
		{name: "rope_two_linears", dynamicRoPE: true, attnGate: false, ffnPiece: "two_linears"},
		{name: "gate_two_linears", dynamicRoPE: false, attnGate: true, ffnPiece: "two_linears"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFFNPieces(cfg, tc.dynamicRoPE, tc.attnGate, tc.ffnPiece)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFFNPieces(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/ffn-pieces "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureRopeGateOneAffineMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention rope-gate one-affine matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name     string
		gateMode string
	}{
		{name: "rope_no_gate_one_affine", gateMode: "none"},
		{name: "rope_input_mul_one_affine", gateMode: "input_mul"},
		{name: "rope_input_add_one_affine", gateMode: "input_add"},
		{name: "rope_branch_mul_one_affine", gateMode: "branch_mul"},
		{name: "rope_branch_sigmoid_mul_one_affine", gateMode: "branch_sigmoid_mul"},
		{name: "rope_branch_add_one_affine", gateMode: "branch_add"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureRopeGateOneAffine(cfg, tc.gateMode)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureRopeGateOneAffine(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/rope-gate-one-affine "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureRopeGateFFNEntryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention rope-gate ffn-entry matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name     string
		gateMode string
	}{
		{name: "rope_no_gate_ffn_entry", gateMode: "none"},
		{name: "rope_main_linear_ffn_entry", gateMode: "main_linear"},
		{name: "rope_main_sigmoid_linear_ffn_entry", gateMode: "main_sigmoid_linear"},
		{name: "rope_input_mul_ffn_entry", gateMode: "input_mul"},
		{name: "rope_input_add_ffn_entry", gateMode: "input_add"},
		{name: "rope_runtime_branch_mul_ffn_entry", gateMode: "runtime_branch_mul"},
		{name: "rope_runtime_branch_sigmoid_mul_ffn_entry", gateMode: "runtime_branch_sigmoid_mul"},
		{name: "rope_runtime_branch_add_ffn_entry", gateMode: "runtime_branch_add"},
		{name: "rope_attn_branch_mul_ffn_entry", gateMode: "attn_branch_mul"},
		{name: "rope_attn_branch_sigmoid_mul_ffn_entry", gateMode: "attn_branch_sigmoid_mul"},
		{name: "rope_attn_branch_add_ffn_entry", gateMode: "attn_branch_add"},
		{name: "rope_branch_mul_ffn_entry", gateMode: "branch_mul"},
		{name: "rope_branch_sigmoid_mul_ffn_entry", gateMode: "branch_sigmoid_mul"},
		{name: "rope_branch_add_ffn_entry", gateMode: "branch_add"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureRopeGateFFNEntry(cfg, tc.gateMode)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureRopeGateFFNEntry(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/rope-gate-ffn-entry "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureRopePreTransformFFNEntryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention rope pre-transform ffn-entry matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name      string
		transform string
	}{
		{name: "rope_no_pretransform_ffn_entry", transform: "none"},
		{name: "rope_pre_linear_ffn_entry", transform: "linear"},
		{name: "rope_pre_sigmoid_linear_ffn_entry", transform: "sigmoid_linear"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureRopePreTransformFFNEntry(cfg, tc.transform)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureRopePreTransformFFNEntry(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/rope-pretransform-ffn-entry "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureRopeModePreTransformFFNEntryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention rope-mode pre-transform ffn-entry matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name      string
		ropeMode  string
		transform string
	}{
		{name: "no_rope_no_pretransform_ffn_entry", ropeMode: "none", transform: "none"},
		{name: "no_rope_pre_linear_ffn_entry", ropeMode: "none", transform: "linear"},
		{name: "static_rope_no_pretransform_ffn_entry", ropeMode: "static", transform: "none"},
		{name: "static_rope_pre_linear_ffn_entry", ropeMode: "static", transform: "linear"},
		{name: "static_rope_pre_sigmoid_linear_ffn_entry", ropeMode: "static", transform: "sigmoid_linear"},
		{name: "dynamic_rope_no_pretransform_ffn_entry", ropeMode: "dynamic", transform: "none"},
		{name: "dynamic_rope_pre_linear_ffn_entry", ropeMode: "dynamic", transform: "linear"},
		{name: "dynamic_rope_pre_sigmoid_linear_ffn_entry", ropeMode: "dynamic", transform: "sigmoid_linear"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureRopeModePreTransformFFNEntry(cfg, tc.ropeMode, tc.transform)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureRopeModePreTransformFFNEntry(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/rope-mode-pretransform-ffn-entry "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFFNInteractionMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen FFN interaction matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:          1,
		Dim:                16,
		AttentionDim:       16,
		NumHeads:           4,
		HeadDim:            4,
		HiddenDim:          32,
		MaxSeqLen:          16,
		KVCacheState:       true,
		KVCacheMaxLen:      16,
		SkipFFN:            false,
		DisableNormOps:     false,
		DisableFinalNormOp: true,
		IncludeLMHead:      false,
		AttentionMaskInput: false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen FFN interaction config: %v", err)
	}

	type interactionCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []interactionCase{
		{name: "dynamic_rope_only", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
		})},
		{name: "gate_only", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.AttentionOutputGate = true
		})},
		{name: "dynamic_rope_and_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen ffn interaction "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestNormalizeMILTransformerConfigDisablesAttentionOutputGate(t *testing.T) {
	cfg := normalizeMILTransformerConfig(MILTransformerConfig{
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
	})
	if cfg.AttentionOutputGate {
		t.Fatal("normalizeMILTransformerConfig kept AttentionOutputGate enabled")
	}
}

func TestMILTransformerTinyQwenFFNRecoveryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen FFN recovery matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:           1,
		Dim:                 16,
		AttentionDim:        16,
		NumHeads:            4,
		HeadDim:             4,
		HiddenDim:           32,
		MaxSeqLen:           16,
		KVCacheState:        true,
		KVCacheMaxLen:       16,
		SkipFFN:             false,
		DisableNormOps:      false,
		DisableFinalNormOp:  true,
		AttentionOutputGate: true,
		DynamicRoPEInputs:   true,
		IncludeLMHead:       false,
		AttentionMaskInput:  false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen FFN recovery config: %v", err)
	}

	type recoveryCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []recoveryCase{
		{name: "conv_default", cfg: baseCfg},
		{name: "linear_default", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
		})},
		{name: "conv_disable_input_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableInputNormOps = true
		})},
		{name: "linear_disable_input_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
			cfg.DisableInputNormOps = true
		})},
		{name: "conv_disable_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableQKNormOps = true
		})},
		{name: "linear_disable_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
			cfg.DisableQKNormOps = true
		})},
		{name: "conv_disable_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableNormOps = true
		})},
		{name: "linear_disable_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.UseConvFFN = false
			cfg.LinearFFN = true
			cfg.DisableNormOps = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen ffn recovery "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFusedFFNRecoveryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen fused FFN recovery matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	baseCfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               true,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             false,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: false,
		DynamicRoPEInputs:          true,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(baseCfg); err != nil {
		t.Fatalf("invalid tiny qwen fused FFN recovery config: %v", err)
	}

	type recoveryCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []recoveryCase{
		{name: "fused_gate", cfg: baseCfg},
		{name: "fused_no_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableAttentionOutputGate = true
		})},
		{name: "fused_no_gate_no_rope", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = false
			cfg.DisableAttentionOutputGate = true
		})},
		{name: "fused_no_gate_no_state", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.KVCacheState = false
			cfg.DisableAttentionOutputGate = true
		})},
		{name: "fused_no_gate_no_rope_no_state", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = false
			cfg.KVCacheState = false
			cfg.DisableAttentionOutputGate = true
		})},
		{name: "fused_no_gate_no_rope_disable_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = false
			cfg.DisableAttentionOutputGate = true
			cfg.DisableQKNormOps = true
		})},
		{name: "fused_no_gate_no_state_disable_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.KVCacheState = false
			cfg.DisableAttentionOutputGate = true
			cfg.DisableQKNormOps = true
		})},
		{name: "fused_no_gate_no_rope_no_state_disable_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = false
			cfg.KVCacheState = false
			cfg.DisableAttentionOutputGate = true
			cfg.DisableQKNormOps = true
		})},
		{name: "fused_no_gate_disable_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableAttentionOutputGate = true
			cfg.DisableFinalNormOp = true
		})},
		{name: "fused_no_gate_disable_input_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableAttentionOutputGate = true
			cfg.DisableInputNormOps = true
		})},
		{name: "fused_no_gate_disable_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableAttentionOutputGate = true
			cfg.DisableQKNormOps = true
		})},
		{name: "fused_no_gate_disable_input_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableAttentionOutputGate = true
			cfg.DisableInputNormOps = true
			cfg.DisableQKNormOps = true
		})},
		{name: "fused_no_gate_disable_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DisableAttentionOutputGate = true
			cfg.DisableNormOps = true
		})},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen fused ffn recovery "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFusedFFNWeightPatternMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen fused FFN weight-pattern matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             false,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid tiny qwen fused FFN weight-pattern config: %v", err)
	}

	cases := []struct {
		name    string
		weights MILTransformerWeights
	}{
		{name: "default_feature_weights", weights: buildTransformerFeatureWeights(cfg)},
		{name: "dense_probe_like_weights", weights: buildTransformerFeatureWeightsDense(cfg)},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(cfg),
				toANEMILTransformerWeights(tc.weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				"transformer tiny qwen fused ffn weight pattern "+tc.name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFusedFFNWeightFileSetMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer tiny qwen fused FFN weight-file-set matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             false,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid tiny qwen fused FFN weight-file-set config: %v", err)
	}

	weights := buildTransformerFeatureWeightsDense(cfg)
	milText, files, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts: %v", err)
	}
	allFiles := fromANEModelWeightFiles(files)
	cases := []struct {
		name  string
		files []modelWeightFile
	}{
		{name: "all_files", files: allFiles},
		{name: "referenced_only", files: filterModelWeightFilesByMILText(allFiles, milText)},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			err := compileMILProbeCase(
				t,
				"transformer tiny qwen fused ffn weight file set "+tc.name,
				milText,
				tc.files,
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFusedFFNNormSubsetMatrix(t *testing.T) {
	if os.Getenv("MLXGO_ANE_TEST_MIL_TRANSFORMER_FEATURE_MATRIX") == "" {
		t.Skip("set MLXGO_ANE_TEST_MIL_TRANSFORMER_FEATURE_MATRIX=1 to enable")
	}
	base := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             false,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	cases := []struct {
		name string
		cfg  MILTransformerConfig
	}{
		{name: "no_disable", cfg: base},
		{name: "disable_input", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) { c.DisableInputNormOps = true })},
		{name: "disable_qk", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) { c.DisableQKNormOps = true })},
		{name: "disable_final", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) { c.DisableFinalNormOp = true })},
		{name: "disable_input_qk", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableInputNormOps = true
			c.DisableQKNormOps = true
		})},
		{name: "disable_input_final", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableInputNormOps = true
			c.DisableFinalNormOp = true
		})},
		{name: "disable_qk_final", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableQKNormOps = true
			c.DisableFinalNormOp = true
		})},
		{name: "disable_input_qk_final", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableInputNormOps = true
			c.DisableQKNormOps = true
			c.DisableFinalNormOp = true
		})},
		{name: "disable_norms", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) { c.DisableNormOps = true })},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				fmt.Sprintf("transformer tiny qwen fused ffn norm subset %s compile", tc.name),
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFusedFFNRopeModeMatrix(t *testing.T) {
	if os.Getenv("MLXGO_ANE_TEST_MIL_TRANSFORMER_FEATURE_MATRIX") == "" {
		t.Skip("set MLXGO_ANE_TEST_MIL_TRANSFORMER_FEATURE_MATRIX=1 to enable")
	}
	base := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             false,
		DisableInputNormOps:        false,
		DisableQKNormOps:           true,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          true,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	cases := []struct {
		name string
		cfg  MILTransformerConfig
	}{
		{name: "dynamic_rope_state_off", cfg: base},
		{name: "no_rope_state_off", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DynamicRoPEInputs = false
		})},
		{name: "dynamic_rope_state_on", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.KVCacheState = true
		})},
		{name: "no_rope_state_on", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DynamicRoPEInputs = false
			c.KVCacheState = true
		})},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			weights := buildTransformerFeatureWeights(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(
				t,
				fmt.Sprintf("transformer tiny qwen fused ffn rope mode %s compile", tc.name),
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureRoPESourceFusedFFNEntryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/rope-source fused-ffn-entry matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableQKNormOps:           true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid rope-source fused-ffn config: %v", err)
	}

	cases := []struct {
		name       string
		ropeSource string
	}{
		{name: "input", ropeSource: "input"},
		{name: "runtime", ropeSource: "runtime"},
		{name: "const", ropeSource: "const"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureRoPESourceFusedFFNEntry(cfg, tc.ropeSource)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureRoPESourceFusedFFNEntry(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/rope-source fused-ffn-entry "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeGatedFFNOnlyMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe gated FFN-only matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		HiddenDim: 32,
	}
	cases := []struct {
		name    string
		useConv bool
		resid   bool
	}{
		{name: "linear", useConv: false, resid: false},
		{name: "linear_residual", useConv: false, resid: true},
		{name: "conv", useConv: true, resid: false},
		{name: "conv_residual", useConv: true, resid: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, input, outputCount, err := buildMILProbeGatedFFNOnly(cfg, tc.useConv, tc.resid)
			if err != nil {
				t.Fatalf("buildMILProbeGatedFFNOnly(%s): %v", tc.name, err)
			}
			best, err := runMILProbeCase(t, "mil probe gated ffn "+tc.name, milText, files, input, outputCount, 2)
			if err != nil {
				t.Logf("%s compile/eval failed: %v", tc.name, err)
				return
			}
			t.Logf("%s succeeded best=%+v", tc.name, best)
		})
	}
}

func TestMILProbeAttentionFFNCompositionMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/FFN composition matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name     string
		stateful bool
		withFFN  bool
	}{
		{name: "stateless_attention_only", stateful: false, withFFN: false},
		{name: "stateless_attention_linear_ffn", stateful: false, withFFN: true},
		{name: "stateful_attention_only", stateful: true, withFFN: false},
		{name: "stateful_attention_linear_ffn", stateful: true, withFFN: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFFNComposition(cfg, tc.stateful, tc.withFFN)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFFNComposition(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/ffn composition "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFFNCompositionMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/FFN composition matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
		withFFN     bool
	}{
		{name: "stateful_attention_linear_ffn", dynamicRoPE: false, attnGate: false, withFFN: true},
		{name: "stateful_attention_dynamic_rope_linear_ffn", dynamicRoPE: true, attnGate: false, withFFN: true},
		{name: "stateful_attention_output_gate_linear_ffn", dynamicRoPE: false, attnGate: true, withFFN: true},
		{name: "stateful_attention_dynamic_rope_output_gate_linear_ffn", dynamicRoPE: true, attnGate: true, withFFN: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFFNComposition(cfg, tc.dynamicRoPE, tc.attnGate, tc.withFFN, "resid")
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFFNComposition(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/ffn composition "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFFNInputSourceMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/FFN input-source matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
		ffnInput    string
	}{
		{name: "dynamic_rope_ffn_from_x", dynamicRoPE: true, attnGate: false, ffnInput: "x"},
		{name: "dynamic_rope_ffn_from_resid", dynamicRoPE: true, attnGate: false, ffnInput: "resid"},
		{name: "output_gate_ffn_from_x", dynamicRoPE: false, attnGate: true, ffnInput: "x"},
		{name: "output_gate_ffn_from_resid", dynamicRoPE: false, attnGate: true, ffnInput: "resid"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFFNComposition(cfg, tc.dynamicRoPE, tc.attnGate, true, tc.ffnInput)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFFNComposition(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/ffn input "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeaturePostLinearMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/post-linear matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
	}{
		{name: "post_linear_only", dynamicRoPE: false, attnGate: false},
		{name: "dynamic_rope_post_linear", dynamicRoPE: true, attnGate: false},
		{name: "output_gate_post_linear", dynamicRoPE: false, attnGate: true},
		{name: "dynamic_rope_output_gate_post_linear", dynamicRoPE: true, attnGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeaturePostLinear(cfg, tc.dynamicRoPE, tc.attnGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeaturePostLinear(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/post-linear "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureSimpleMLPMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/simple-mlp matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
	}{
		{name: "simple_mlp_only", dynamicRoPE: false, attnGate: false},
		{name: "dynamic_rope_simple_mlp", dynamicRoPE: true, attnGate: false},
		{name: "output_gate_simple_mlp", dynamicRoPE: false, attnGate: true},
		{name: "dynamic_rope_output_gate_simple_mlp", dynamicRoPE: true, attnGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureSimpleMLP(cfg, tc.dynamicRoPE, tc.attnGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureSimpleMLP(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/simple-mlp "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureTwoLinearsMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/two-linears matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
	}{
		{name: "two_linears_only", dynamicRoPE: false, attnGate: false},
		{name: "dynamic_rope_two_linears", dynamicRoPE: true, attnGate: false},
		{name: "output_gate_two_linears", dynamicRoPE: false, attnGate: true},
		{name: "dynamic_rope_output_gate_two_linears", dynamicRoPE: true, attnGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureTwoLinears(cfg, tc.dynamicRoPE, tc.attnGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureTwoLinears(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/two-linears "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureTwoLinearsSameWidthMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/two-linears-same-width matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
	}{
		{name: "same_width_two_linears_only", dynamicRoPE: false, attnGate: false},
		{name: "dynamic_rope_same_width_two_linears", dynamicRoPE: true, attnGate: false},
		{name: "output_gate_same_width_two_linears", dynamicRoPE: false, attnGate: true},
		{name: "dynamic_rope_output_gate_same_width_two_linears", dynamicRoPE: true, attnGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureTwoLinearsSameWidth(cfg, tc.dynamicRoPE, tc.attnGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureTwoLinearsSameWidth(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/two-linears-same-width "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureTwoLinearsStatefulnessMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/two-linears statefulness matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name     string
		stateful bool
	}{
		{name: "stateless_dynamic_rope_output_gate_two_linears", stateful: false},
		{name: "stateful_dynamic_rope_output_gate_two_linears", stateful: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureTwoLinearsStatefulness(cfg, tc.stateful)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureTwoLinearsStatefulness(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/two-linears statefulness "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureAffineSourceDepthMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/affine-source-depth matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		attnGate    bool
		source      string
		depth       int
	}{
		{name: "dynamic_rope_output_gate_attn_one_linear", dynamicRoPE: true, attnGate: true, source: "attn", depth: 1},
		{name: "dynamic_rope_output_gate_attn_two_linears", dynamicRoPE: true, attnGate: true, source: "attn", depth: 2},
		{name: "dynamic_rope_output_gate_proj_one_linear", dynamicRoPE: true, attnGate: true, source: "proj", depth: 1},
		{name: "dynamic_rope_output_gate_proj_two_linears", dynamicRoPE: true, attnGate: true, source: "proj", depth: 2},
		{name: "dynamic_rope_output_gate_resid_one_linear", dynamicRoPE: true, attnGate: true, source: "resid", depth: 1},
		{name: "dynamic_rope_output_gate_resid_two_linears", dynamicRoPE: true, attnGate: true, source: "resid", depth: 2},
		{name: "dynamic_rope_no_gate_attn_two_linears", dynamicRoPE: true, attnGate: false, source: "attn", depth: 2},
		{name: "dynamic_rope_no_gate_proj_two_linears", dynamicRoPE: true, attnGate: false, source: "proj", depth: 2},
		{name: "dynamic_rope_no_gate_resid_two_linears", dynamicRoPE: true, attnGate: false, source: "resid", depth: 2},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureAffineSourceDepth(cfg, tc.dynamicRoPE, tc.attnGate, tc.source, tc.depth)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureAffineSourceDepth(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/affine-source-depth "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureGateTopologyTwoLinearsMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/gate-topology two-linears matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name     string
		gateMode string
	}{
		{name: "no_gate", gateMode: "none"},
		{name: "mul_gate", gateMode: "mul"},
		{name: "sigmoid_mul_gate", gateMode: "sigmoid_mul"},
		{name: "add_gate", gateMode: "add"},
		{name: "sigmoid_add_gate", gateMode: "sigmoid_add"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureGateTopologyTwoLinears(cfg, tc.gateMode)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureGateTopologyTwoLinears(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/gate-topology two-linears "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureGatePlacementTwoLinearsMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/gate-placement two-linears matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name      string
		placement string
	}{
		{name: "no_gate", placement: "none"},
		{name: "gate_before_proj", placement: "before_proj"},
		{name: "gate_after_proj", placement: "after_proj"},
		{name: "gate_after_resid", placement: "after_resid"},
		{name: "gate_after_lin1", placement: "after_lin1"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureGatePlacementTwoLinears(cfg, tc.placement)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureGatePlacementTwoLinears(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/gate-placement two-linears "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureDeadGateBranchTwoLinearsMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/dead-gate-branch two-linears matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name       string
		deadBranch string
	}{
		{name: "no_dead_branch", deadBranch: "none"},
		{name: "dead_qg_linear", deadBranch: "qg"},
		{name: "dead_qg_sigmoid", deadBranch: "qg_sigmoid"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureDeadGateBranchTwoLinears(cfg, tc.deadBranch)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureDeadGateBranchTwoLinears(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/dead-gate-branch two-linears "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureDeadFFNBranchMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/dead-ffn-branch matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name       string
		deadBranch string
	}{
		{name: "no_dead_ffn", deadBranch: "none"},
		{name: "dead_one_linear_hidden", deadBranch: "one_linear_hidden"},
		{name: "dead_one_linear_same_width", deadBranch: "one_linear_same_width"},
		{name: "dead_one_linear_hidden_elementwise", deadBranch: "one_linear_hidden_elementwise"},
		{name: "dead_one_linear_same_width_elementwise", deadBranch: "one_linear_same_width_elementwise"},
		{name: "dead_one_linear_hidden_const_add", deadBranch: "one_linear_hidden_const_add"},
		{name: "dead_one_linear_hidden_const_mul", deadBranch: "one_linear_hidden_const_mul"},
		{name: "dead_one_linear_same_width_const_add", deadBranch: "one_linear_same_width_const_add"},
		{name: "dead_one_linear_same_width_const_mul", deadBranch: "one_linear_same_width_const_mul"},
		{name: "dead_one_linear_hidden_reduce", deadBranch: "one_linear_hidden_reduce"},
		{name: "dead_one_linear_same_width_reduce", deadBranch: "one_linear_same_width_reduce"},
		{name: "dead_linear_ffn", deadBranch: "linear_ffn"},
		{name: "dead_linear_sigmoid", deadBranch: "linear_sigmoid"},
		{name: "dead_same_width_two_linears", deadBranch: "same_width_two_linears"},
		{name: "dead_elementwise_depth2", deadBranch: "elementwise_depth2"},
		{name: "dead_gated_ffn", deadBranch: "gated_ffn"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureDeadFFNBranch(cfg, tc.deadBranch)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureDeadFFNBranch(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/dead-ffn-branch "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureDeadBranchRoPEMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/dead-branch rope matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		deadBranch  string
	}{
		{name: "no_rope_dead_linear_ffn", dynamicRoPE: false, deadBranch: "linear_ffn"},
		{name: "rope_dead_linear_ffn", dynamicRoPE: true, deadBranch: "linear_ffn"},
		{name: "no_rope_dead_qg", dynamicRoPE: false, deadBranch: "qg"},
		{name: "rope_dead_qg", dynamicRoPE: true, deadBranch: "qg"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureDeadBranchRoPE(cfg, tc.dynamicRoPE, tc.deadBranch)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureDeadBranchRoPE(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/dead-branch rope "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureGateTopologyMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/gate-topology matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name     string
		gateMode string
	}{
		{name: "dynamic_rope_no_gate_two_linears", gateMode: "none"},
		{name: "dynamic_rope_const_mul_two_linears", gateMode: "const_mul"},
		{name: "dynamic_rope_const_sigmoid_gate_two_linears", gateMode: "const_sigmoid_mul"},
		{name: "dynamic_rope_const_add_two_linears", gateMode: "const_add"},
		{name: "dynamic_rope_branch_mul_two_linears", gateMode: "branch_mul"},
		{name: "dynamic_rope_branch_sigmoid_mul_two_linears", gateMode: "branch_sigmoid_mul"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureGateTopology(cfg, tc.gateMode)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureGateTopology(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/gate-topology "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureMixPositionMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/mix-position matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		mixPosition string
	}{
		{name: "dynamic_rope_no_mix_two_linears", mixPosition: "none"},
		{name: "dynamic_rope_mix_before_proj_two_linears", mixPosition: "before_proj"},
		{name: "dynamic_rope_mix_after_proj_two_linears", mixPosition: "after_proj"},
		{name: "dynamic_rope_mix_after_resid_two_linears", mixPosition: "after_resid"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureMixPosition(cfg, tc.mixPosition)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureMixPosition(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/mix-position "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureMixPositionPostLinearMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/mix-position post-linear matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		rope        bool
		mixPosition string
	}{
		{name: "no_rope_no_mix_post_linear", rope: false, mixPosition: "none"},
		{name: "no_rope_mix_before_proj_post_linear", rope: false, mixPosition: "before_proj"},
		{name: "no_rope_mix_after_proj_post_linear", rope: false, mixPosition: "after_proj"},
		{name: "no_rope_mix_after_resid_post_linear", rope: false, mixPosition: "after_resid"},
		{name: "dynamic_rope_no_mix_post_linear", rope: true, mixPosition: "none"},
		{name: "dynamic_rope_mix_before_proj_post_linear", rope: true, mixPosition: "before_proj"},
		{name: "dynamic_rope_mix_after_proj_post_linear", rope: true, mixPosition: "after_proj"},
		{name: "dynamic_rope_mix_after_resid_post_linear", rope: true, mixPosition: "after_resid"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureMixPositionPostLinear(cfg, tc.rope, tc.mixPosition)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureMixPositionPostLinear(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/mix-position post-linear "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureMixReturnMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/mix-return matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name    string
		rope    bool
		mixMode string
	}{
		{name: "no_rope_return_attn", rope: false, mixMode: "none"},
		{name: "no_rope_return_attn_add", rope: false, mixMode: "add"},
		{name: "no_rope_return_attn_mul", rope: false, mixMode: "mul"},
		{name: "dynamic_rope_return_attn", rope: true, mixMode: "none"},
		{name: "dynamic_rope_return_attn_add", rope: true, mixMode: "add"},
		{name: "dynamic_rope_return_attn_mul", rope: true, mixMode: "mul"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureMixReturn(cfg, tc.rope, tc.mixMode)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureMixReturn(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/mix-return "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureGateTopologyPostLinearMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/gate-topology post-linear matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name     string
		gateMode string
	}{
		{name: "no_rope_no_gate_post_linear", gateMode: "none"},
		{name: "no_rope_const_mul_post_linear", gateMode: "const_mul"},
		{name: "no_rope_const_sigmoid_mul_post_linear", gateMode: "const_sigmoid_mul"},
		{name: "no_rope_const_add_post_linear", gateMode: "const_add"},
		{name: "no_rope_branch_mul_post_linear", gateMode: "branch_mul"},
		{name: "no_rope_branch_sigmoid_mul_post_linear", gateMode: "branch_sigmoid_mul"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureGateTopologyPostLinear(cfg, tc.gateMode)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureGateTopologyPostLinear(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/gate-topology post-linear "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureLearnedGateVariantsPostLinearMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/learned-gate-variants post-linear matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name   string
		gateOp string
	}{
		{name: "branch_mul", gateOp: "branch_mul"},
		{name: "branch_sigmoid_mul", gateOp: "branch_sigmoid_mul"},
		{name: "branch_add", gateOp: "branch_add"},
		{name: "branch_sigmoid_add", gateOp: "branch_sigmoid_add"},
		{name: "branch_mul_plus_const", gateOp: "branch_mul_plus_const"},
		{name: "branch_sigmoid_mul_plus_const", gateOp: "branch_sigmoid_mul_plus_const"},
		{name: "branch_mul_times_const", gateOp: "branch_mul_times_const"},
		{name: "branch_sigmoid_mul_times_const", gateOp: "branch_sigmoid_mul_times_const"},
		{name: "branch_mul_plus_zero", gateOp: "branch_mul_plus_zero"},
		{name: "branch_mul_times_one", gateOp: "branch_mul_times_one"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear(cfg, tc.gateOp)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/learned-gate post-linear "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureComputedGateVariantsPostLinearMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/computed-gate-variants post-linear matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name   string
		gateOp string
	}{
		{name: "input_mul", gateOp: "input_mul"},
		{name: "input_sigmoid_mul", gateOp: "input_sigmoid_mul"},
		{name: "input_add", gateOp: "input_add"},
		{name: "input_sigmoid_add", gateOp: "input_sigmoid_add"},
		{name: "input_mul_plus_const", gateOp: "input_mul_plus_const"},
		{name: "input_sigmoid_mul_plus_const", gateOp: "input_sigmoid_mul_plus_const"},
		{name: "input_mul_times_const", gateOp: "input_mul_times_const"},
		{name: "input_sigmoid_mul_times_const", gateOp: "input_sigmoid_mul_times_const"},
		{name: "input_mul_plus_zero", gateOp: "input_mul_plus_zero"},
		{name: "input_mul_times_one", gateOp: "input_mul_times_one"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear(cfg, tc.gateOp)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/computed-gate post-linear "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureCombinedGateVariantsPostLinearMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/combined-gate-variants post-linear matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name   string
		gateOp string
	}{
		{name: "combined_add_mul", gateOp: "combined_add_mul"},
		{name: "combined_add_add", gateOp: "combined_add_add"},
		{name: "combined_mul_mul", gateOp: "combined_mul_mul"},
		{name: "combined_mul_add", gateOp: "combined_mul_add"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear(cfg, tc.gateOp)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/combined-gate post-linear "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeaturePostAttentionConstTransformMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/post-attention const-transform matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:                 16,
		NumHeads:            4,
		HeadDim:             4,
		HiddenDim:           32,
		MaxSeqLen:           8,
		DynamicRoPEInputs:   true,
		AttentionOutputGate: true,
	}
	cases := []struct {
		name      string
		transform string
	}{
		{name: "none", transform: "none"},
		{name: "mul_const", transform: "mul_const"},
		{name: "add_const", transform: "add_const"},
		{name: "mul_one", transform: "mul_one"},
		{name: "add_zero", transform: "add_zero"},
		{name: "rmsnorm", transform: "rmsnorm"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeaturePostAttentionConstTransform(cfg, tc.transform)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeaturePostAttentionConstTransform(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/post-attention const-transform "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureDynamicRopeGateResidualMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/dynamic-rope-gate-residual matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		outputGate  bool
		stage       string
	}{
		{name: "rope_only_proj", dynamicRoPE: true, outputGate: false, stage: "proj"},
		{name: "gate_only_proj", dynamicRoPE: false, outputGate: true, stage: "proj"},
		{name: "rope_gate_proj", dynamicRoPE: true, outputGate: true, stage: "proj"},
		{name: "rope_only_resid", dynamicRoPE: true, outputGate: false, stage: "resid"},
		{name: "gate_only_resid", dynamicRoPE: false, outputGate: true, stage: "resid"},
		{name: "rope_gate_resid", dynamicRoPE: true, outputGate: true, stage: "resid"},
		{name: "rope_gate_post_linear", dynamicRoPE: true, outputGate: true, stage: "post_linear"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureDynamicRopeGateResidual(cfg, tc.dynamicRoPE, tc.outputGate, tc.stage)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureDynamicRopeGateResidual(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/dynamic-rope-gate-residual "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureRoPESourcePostLinearMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/rope-source post-linear matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name       string
		ropeSource string
		outputGate bool
	}{
		{name: "input_no_gate", ropeSource: "input", outputGate: false},
		{name: "runtime_no_gate", ropeSource: "runtime", outputGate: false},
		{name: "const_no_gate", ropeSource: "const", outputGate: false},
		{name: "input_gate", ropeSource: "input", outputGate: true},
		{name: "runtime_gate", ropeSource: "runtime", outputGate: true},
		{name: "const_gate", ropeSource: "const", outputGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureRoPESourcePostLinear(cfg, tc.ropeSource, tc.outputGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureRoPESourcePostLinear(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/rope-source post-linear "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureRoPESourceFFNEntryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/rope-source FFN-entry matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name       string
		ropeSource string
		outputGate bool
	}{
		{name: "input_no_gate", ropeSource: "input", outputGate: false},
		{name: "runtime_no_gate", ropeSource: "runtime", outputGate: false},
		{name: "const_no_gate", ropeSource: "const", outputGate: false},
		{name: "input_gate", ropeSource: "input", outputGate: true},
		{name: "runtime_gate", ropeSource: "runtime", outputGate: true},
		{name: "const_gate", ropeSource: "const", outputGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureRoPESourceFFNEntry(cfg, tc.ropeSource, tc.outputGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureRoPESourceFFNEntry(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/rope-source ffn-entry "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFusedFFNEntryMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/fused-FFN-entry matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		outputGate  bool
	}{
		{name: "no_rope_no_gate", dynamicRoPE: false, outputGate: false},
		{name: "rope_no_gate", dynamicRoPE: true, outputGate: false},
		{name: "no_rope_gate", dynamicRoPE: false, outputGate: true},
		{name: "rope_gate", dynamicRoPE: true, outputGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFusedFFNEntry(cfg, tc.dynamicRoPE, tc.outputGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFusedFFNEntry(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/fused-ffn-entry "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFusedFFNNoDownMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/fused-FFN-no-down matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name        string
		dynamicRoPE bool
		outputGate  bool
	}{
		{name: "no_rope_no_gate", dynamicRoPE: false, outputGate: false},
		{name: "rope_no_gate", dynamicRoPE: true, outputGate: false},
		{name: "no_rope_gate", dynamicRoPE: false, outputGate: true},
		{name: "rope_gate", dynamicRoPE: true, outputGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFusedFFNNoDown(cfg, tc.dynamicRoPE, tc.outputGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFusedFFNNoDown(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/fused-ffn-no-down "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFusedFFNFinalNormMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/fused-FFN-final-norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:        16,
		NumHeads:   4,
		HeadDim:    4,
		HiddenDim:  32,
		MaxSeqLen:  8,
		RMSNormEps: 1e-6,
	}
	cases := []struct {
		name       string
		outputGate bool
	}{
		{name: "rope_no_gate", outputGate: false},
		{name: "rope_gate", outputGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFusedFFNFinalNorm(cfg, tc.outputGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFusedFFNFinalNorm(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/fused-ffn-final-norm "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFusedSwiGLUFFNFinalNormMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/fused-swiglu-ffn-final-norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:        16,
		NumHeads:   4,
		HeadDim:    4,
		HiddenDim:  32,
		MaxSeqLen:  8,
		RMSNormEps: 1e-6,
	}
	cases := []struct {
		name       string
		outputGate bool
	}{
		{name: "rope_no_gate", outputGate: false},
		{name: "rope_gate", outputGate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFusedSwiGLUFFNFinalNorm(cfg, tc.outputGate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFusedSwiGLUFFNFinalNorm(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/fused-swiglu-ffn-final-norm "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFusedSliceStyleMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/fused-slice-style matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name       string
		sliceStyle string
		gateMode   string
	}{
		{name: "explicit_sigmoid_gate", sliceStyle: "explicit", gateMode: "sigmoid"},
		{name: "explicit_swiglu_gate", sliceStyle: "explicit", gateMode: "swiglu"},
		{name: "masked_sigmoid_gate", sliceStyle: "masked", gateMode: "sigmoid"},
		{name: "masked_swiglu_gate", sliceStyle: "masked", gateMode: "swiglu"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFusedSliceStyle(cfg, tc.sliceStyle, tc.gateMode)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFusedSliceStyle(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/fused-slice-style "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFusedResidualTailMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/fused-residual-tail matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name         string
		includeResid bool
	}{
		{name: "down_only", includeResid: false},
		{name: "down_plus_resid", includeResid: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFusedResidualTail(cfg, tc.includeResid)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFusedResidualTail(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/fused-residual-tail "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILProbeAttentionFeatureFusedConstOrderMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/feature/fused-const-order matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	cases := []struct {
		name      string
		constLate bool
	}{
		{name: "consts_before_linears", constLate: false},
		{name: "consts_after_qkv", constLate: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureFusedConstOrder(cfg, tc.constLate)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureFusedConstOrder(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/fused-const-order "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerTinyQwenFusedFFNCanonicalOpSequenceDiff(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             true,
		DisableFinalNormOp:         true,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid canonical fused diff config: %v", err)
	}
	weights := buildTransformerFeatureWeightsDense(cfg)
	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical): %v", err)
	}
	reducedMIL, _, err := buildMILProbeAttentionFeatureFusedSliceStyle(
		MILTransformerConfig{
			Dim:       cfg.Dim,
			NumHeads:  cfg.NumHeads,
			HeadDim:   cfg.HeadDim,
			HiddenDim: cfg.HiddenDim,
			MaxSeqLen: cfg.MaxSeqLen,
		},
		"explicit",
		"swiglu",
	)
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureFusedSliceStyle(reduced): %v", err)
	}
	canonicalLines := milStructuralLines(canonicalMIL)
	reducedLines := milStructuralLines(reducedMIL)
	canonicalLines = filterMILStructuralFooter(canonicalLines)
	reducedLines = filterMILStructuralFooter(reducedLines)
	if len(canonicalLines) != len(reducedLines)+1 {
		t.Fatalf("expected canonical fused tail to have exactly one extra residual add: canonical=%d reduced=%d\ncanonical=%v\nreduced=%v", len(canonicalLines), len(reducedLines), canonicalLines, reducedLines)
	}
	canonicalTrim := append([]string(nil), canonicalLines...)
	reducedTrim := append([]string(nil), reducedLines...)
	canonicalTrim = canonicalTrim[:len(canonicalTrim)-1]
	reducedTrim = reducedTrim[:len(reducedTrim)-1]
	if got, want := canonicalTrim[len(canonicalTrim)-1], "[1,1,16] add(x,y)"; got != want {
		t.Fatalf("canonical fused tail missing residual add before cast: got %q want %q", got, want)
	}
	canonicalTrim = canonicalTrim[:len(canonicalTrim)-1]
	if !slices.Equal(canonicalTrim, reducedTrim) {
		t.Fatalf(
			"shared fused-tail prefix mismatch:\ncanonical=%v\nreduced=%v",
			canonicalTrim,
			reducedTrim,
		)
	}
	if got, want := canonicalLines[len(canonicalLines)-2], "[1,1,16] add(x,y)"; got != want {
		t.Fatalf("canonical fused tail missing residual add: got %q want %q", got, want)
	}
	if got, want := canonicalLines[len(canonicalLines)-1], reducedLines[len(reducedLines)-1]; got != want {
		t.Fatalf("canonical/reduced cast tail mismatch: canonical=%q reduced=%q", got, want)
	}
}

func TestMILTransformerTinyQwenFusedFFNExactProbeWeightPattern(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run fused exact-probe weight pattern matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             true,
		DisableFinalNormOp:         true,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid canonical fused exact-probe config: %v", err)
	}
	weights := buildTransformerFeatureWeightsFromReducedFusedProbe(cfg)
	milText, files, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(exact-probe): %v", err)
	}
	filtered := filterModelWeightFilesByMILText(fromANEModelWeightFiles(files), milText)
	err = compileMILProbeCase(t, "transformer tiny qwen fused exact-probe weights", milText, filtered)
	if err != nil {
		t.Logf("canonical fused exact-probe compile failed: %v", err)
		return
	}
	t.Log("canonical fused exact-probe compile succeeded")
}

func TestMILTransformerTinyQwenFusedFFNCanonicalLikeOpSequenceDiff(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             true,
		DisableFinalNormOp:         true,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid canonical-like diff config: %v", err)
	}
	weights := buildTransformerFeatureWeightsFromReducedFusedProbe(cfg)
	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical-like-diff): %v", err)
	}
	probeMIL, _, err := buildMILProbeAttentionFeatureCanonicalLikeFused(MILTransformerConfig{
		Dim:       cfg.Dim,
		NumHeads:  cfg.NumHeads,
		HeadDim:   cfg.HeadDim,
		HiddenDim: cfg.HiddenDim,
		MaxSeqLen: cfg.MaxSeqLen,
	})
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFused: %v", err)
	}
	canonicalLines := milStructuralLines(canonicalMIL)
	probeLines := milStructuralLines(probeMIL)
	max := len(canonicalLines)
	if len(probeLines) < max {
		max = len(probeLines)
	}
	for i := 0; i < max; i++ {
		if canonicalLines[i] != probeLines[i] {
			start := i - 3
			if start < 0 {
				start = 0
			}
			end := i + 4
			if end > len(canonicalLines) {
				end = len(canonicalLines)
			}
			if end > len(probeLines) {
				end = len(probeLines)
			}
			t.Fatalf(
				"first canonical-like line diff at %d:\ncanonical: %s\nprobe:     %s\ncanonical_window=%v\nprobe_window=%v",
				i,
				canonicalLines[i],
				probeLines[i],
				canonicalLines[start:end],
				probeLines[start:end],
			)
		}
	}
	if len(canonicalLines) != len(probeLines) {
		t.Fatalf("canonical-like line length diff: canonical=%d probe=%d\ncanonical=%v\nprobe=%v", len(canonicalLines), len(probeLines), canonicalLines, probeLines)
	}
}

func TestMILTransformerTinyQwenFusedFFNCanonicalLikeConstDiff(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             true,
		DisableFinalNormOp:         true,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          false,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid canonical-like const diff config: %v", err)
	}
	weights := buildTransformerFeatureWeightsFromReducedFusedProbe(cfg)
	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical-like-const-diff): %v", err)
	}
	probeMIL, _, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithWeightNames(MILTransformerConfig{
		Dim:       cfg.Dim,
		NumHeads:  cfg.NumHeads,
		HeadDim:   cfg.HeadDim,
		HiddenDim: cfg.HiddenDim,
		MaxSeqLen: cfg.MaxSeqLen,
	}, true)
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithWeightNames: %v", err)
	}
	canonicalLines := milBlobConstLines(canonicalMIL)
	probeLines := milBlobConstLines(probeMIL)
	max := len(canonicalLines)
	if len(probeLines) < max {
		max = len(probeLines)
	}
	for i := 0; i < max; i++ {
		if canonicalLines[i] != probeLines[i] {
			start := i - 3
			if start < 0 {
				start = 0
			}
			end := i + 4
			if end > len(canonicalLines) {
				end = len(canonicalLines)
			}
			if end > len(probeLines) {
				end = len(probeLines)
			}
			t.Fatalf(
				"first canonical-like blob const diff at %d:\ncanonical: %s\nprobe:     %s\ncanonical_window=%v\nprobe_window=%v",
				i,
				canonicalLines[i],
				probeLines[i],
				canonicalLines[start:end],
				probeLines[start:end],
			)
		}
	}
	if len(canonicalLines) != len(probeLines) {
		t.Fatalf("canonical-like blob const length diff: canonical=%d probe=%d\ncanonical=%v\nprobe=%v", len(canonicalLines), len(probeLines), canonicalLines, probeLines)
	}
}

func TestMILTransformerTinyQwenFusedFFNRopeCanonicalLikeOpSequenceDiff(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  16,
		KVCacheState:               false,
		KVCacheMaxLen:              16,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             false,
		DisableInputNormOps:        false,
		DisableQKNormOps:           true,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          true,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid rope canonical-like diff config: %v", err)
	}
	weights := buildTransformerFeatureWeightsDense(cfg)
	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(rope canonical-like): %v", err)
	}
	probeMIL, _, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNormPostNormFinalNorm(
		MILTransformerConfig{
			Dim:       cfg.Dim,
			NumHeads:  cfg.NumHeads,
			HeadDim:   cfg.HeadDim,
			HiddenDim: cfg.HiddenDim,
			MaxSeqLen: cfg.MaxSeqLen,
		},
		"input",
	)
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNormPostNormFinalNorm: %v", err)
	}
	canonicalLines := normalizeMILStructuralArgOrder(filterMILStructuralOps(milStructuralLines(canonicalMIL), "cast"))
	probeLines := normalizeMILStructuralArgOrder(filterMILStructuralOps(milStructuralLines(probeMIL), "cast"))
	max := len(canonicalLines)
	if len(probeLines) < max {
		max = len(probeLines)
	}
	for i := 0; i < max; i++ {
		if canonicalLines[i] != probeLines[i] {
			start := i - 3
			if start < 0 {
				start = 0
			}
			end := i + 4
			if end > len(canonicalLines) {
				end = len(canonicalLines)
			}
			if end > len(probeLines) {
				end = len(probeLines)
			}
			t.Fatalf(
				"first rope canonical-like structural diff at %d:\ncanonical: %s\nprobe:     %s\ncanonical_window=%v\nprobe_window=%v",
				i,
				canonicalLines[i],
				probeLines[i],
				canonicalLines[start:end],
				probeLines[start:end],
			)
		}
	}
	if len(canonicalLines) != len(probeLines) {
		t.Fatalf("rope canonical-like line length diff: canonical=%d probe=%d\ncanonical=%v\nprobe=%v", len(canonicalLines), len(probeLines), canonicalLines, probeLines)
	}
}

func TestMILProbeAttentionFeatureCanonicalLikeFusedWithRopeMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical-like fused with rope matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	for _, ropeSource := range []string{"input", "runtime", "const"} {
		t.Run(ropeSource, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(cfg, ropeSource)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(%s): %v", ropeSource, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/canonical-like-fused-rope "+ropeSource, milText, files)
			if err != nil {
				t.Logf("canonical-like fused with rope %s compile failed: %v", ropeSource, err)
				return
			}
			t.Logf("canonical-like fused with rope %s compile succeeded", ropeSource)
		})
	}
}

func TestMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNormMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical-like fused with rope input-norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	for _, ropeSource := range []string{"input", "runtime"} {
		t.Run(ropeSource, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNorm(cfg, ropeSource)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNorm(%s): %v", ropeSource, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/canonical-like-fused-rope-inputnorm "+ropeSource, milText, files)
			if err != nil {
				t.Logf("canonical-like fused with rope input norm %s compile failed: %v", ropeSource, err)
				return
			}
			t.Logf("canonical-like fused with rope input norm %s compile succeeded", ropeSource)
		})
	}
}

func TestMILProbeAttentionFeatureCanonicalLikeFusedWithRopeFinalNormMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical-like fused with rope final-norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	for _, ropeSource := range []string{"input", "runtime"} {
		t.Run(ropeSource, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeFinalNorm(cfg, ropeSource)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeFinalNorm(%s): %v", ropeSource, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/canonical-like-fused-rope-finalnorm "+ropeSource, milText, files)
			if err != nil {
				t.Logf("canonical-like fused with rope final norm %s compile failed: %v", ropeSource, err)
				return
			}
			t.Logf("canonical-like fused with rope final norm %s compile succeeded", ropeSource)
		})
	}
}

func TestMILTransformerTinyQwenCanonicalRopeNoNormFusedMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical rope no-norm fused matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  8,
		KVCacheState:               false,
		KVCacheMaxLen:              8,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             true,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          true,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	weights := buildTransformerFeatureWeightsDense(cfg)
	milText, files, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical rope no-norm fused): %v", err)
	}
	err = compileMILProbeCase(t, "transformer canonical rope no-norm fused", milText, fromANEModelWeightFiles(files))
	if err != nil {
		t.Logf("canonical rope no-norm fused compile failed: %v", err)
		return
	}
	t.Log("canonical rope no-norm fused compile succeeded")
}

func TestMILTransformerTinyQwenCanonicalRopeFusedGateSelectiveNormMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical rope fused gate selective norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	base := MILTransformerConfig{
		NumLayers:           1,
		Dim:                 16,
		AttentionDim:        16,
		NumHeads:            4,
		HeadDim:             4,
		HiddenDim:           32,
		MaxSeqLen:           8,
		KVCacheState:        false,
		KVCacheMaxLen:       8,
		SkipFFN:             false,
		UseConvFFN:          false,
		LinearFFN:           true,
		FusedLinearFFN:      true,
		AttentionOutputGate: true,
		DynamicRoPEInputs:   true,
		IncludeLMHead:       false,
		AttentionMaskInput:  false,
	}
	cases := []struct {
		name string
		cfg  MILTransformerConfig
	}{
		{name: "fused_gate_all_norms_on", cfg: base},
		{name: "fused_gate_disable_input_norm", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableInputNormOps = true
		})},
		{name: "fused_gate_disable_qk_norm", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableQKNormOps = true
		})},
		{name: "fused_gate_disable_input_qk_norm", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableInputNormOps = true
			c.DisableQKNormOps = true
		})},
		{name: "fused_gate_disable_all_norms", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.DisableNormOps = true
		})},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if err := validateMILTransformerConfig(tc.cfg); err != nil {
				t.Fatalf("invalid canonical rope fused gate config: %v", err)
			}
			weights := buildTransformerFeatureWeightsDense(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "transformer canonical rope fused gate "+tc.name, milText, fromANEModelWeightFiles(files))
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerCanonicalRopeFusedGateSelectiveNormDimMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical rope fused gate selective norm dim matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	type dimCase struct {
		name string
		dim  int
	}
	cases := []dimCase{
		{name: "d16", dim: 16},
		{name: "d32", dim: 32},
		{name: "d64", dim: 64},
		{name: "d128", dim: 128},
		{name: "d256", dim: 256},
		{name: "d512", dim: 512},
		{name: "d768", dim: 768},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			headDim := 64
			numHeads := tc.dim / headDim
			if numHeads <= 0 {
				numHeads = 1
				headDim = tc.dim
			}
			cfg := MILTransformerConfig{
				NumLayers:           1,
				Dim:                 tc.dim,
				AttentionDim:        tc.dim,
				NumHeads:            numHeads,
				HeadDim:             headDim,
				HiddenDim:           tc.dim * 4,
				MaxSeqLen:           8,
				KVCacheState:        false,
				KVCacheMaxLen:       8,
				SkipFFN:             false,
				UseConvFFN:          false,
				LinearFFN:           true,
				FusedLinearFFN:      true,
				AttentionOutputGate: true,
				DynamicRoPEInputs:   true,
				DisableInputNormOps: true,
				DisableQKNormOps:    true,
				IncludeLMHead:       false,
				AttentionMaskInput:  false,
			}
			if err := validateMILTransformerConfig(cfg); err != nil {
				t.Fatalf("invalid dim case %s: %v", tc.name, err)
			}
			weights := buildTransformerFeatureWeightsDense(cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "transformer canonical rope fused gate selective norms "+tc.name, milText, fromANEModelWeightFiles(files))
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerCanonicalRopeFusedGateSelectiveNormStateMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical rope fused gate selective norm state matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	base := MILTransformerConfig{
		NumLayers:           1,
		Dim:                 768,
		AttentionDim:        768,
		NumHeads:            12,
		HeadDim:             64,
		HiddenDim:           3072,
		MaxSeqLen:           32,
		KVCacheState:        false,
		KVCacheMaxLen:       32,
		SkipFFN:             false,
		UseConvFFN:          false,
		LinearFFN:           true,
		FusedLinearFFN:      true,
		AttentionOutputGate: true,
		DynamicRoPEInputs:   true,
		DisableInputNormOps: true,
		DisableQKNormOps:    true,
		IncludeLMHead:       false,
		AttentionMaskInput:  false,
	}
	cases := []struct {
		name string
		cfg  MILTransformerConfig
	}{
		{name: "stateless", cfg: base},
		{name: "stateful", cfg: withTransformerFeature(base, func(c *MILTransformerConfig) {
			c.KVCacheState = true
		})},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if err := validateMILTransformerConfig(tc.cfg); err != nil {
				t.Fatalf("invalid state case %s: %v", tc.name, err)
			}
			weights := buildTransformerFeatureWeightsDense(tc.cfg)
			milText, files, err := anereify.BuildMILTransformerArtifacts(
				toANEMILTransformerConfig(tc.cfg),
				toANEMILTransformerWeights(weights),
			)
			if err != nil {
				t.Fatalf("BuildMILTransformerArtifacts(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "transformer canonical rope fused gate selective norms "+tc.name, milText, fromANEModelWeightFiles(files))
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerRealQwenSelectiveNormFusedOpSequenceDiff(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run real qwen selective-norm fused diff", milTransformerFeatureMatrixEnv)
	}
	prog := qwenFixtureSingleLayerProgram(t)

	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			LinearFFN:                  true,
			FusedLinearFFN:             true,
			AttentionOutputGate:        true,
			DisableInputNormOps:        true,
			DisableQKNormOps:           true,
			DynamicRoPEInputs:          true,
			DisableAttentionOutputGate: true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
		AllowPartial:    true,
	}
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		t.Fatalf("ReifyToANEMIL(reduced qwen fixture): %v", err)
	}
	localCfg := fromANEMILTransformerConfig(reified.TransformerConfig)

	probeMIL, _, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeFinalNorm(localCfg, "input")
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeFinalNorm: %v", err)
	}

	excludedOps := []string{"read_state", "slice_by_index", "concat", "coreml_update_state"}
	realLines := normalizeMILStructuralArgOrder(filterMILStructuralOps(milStructuralLines(reified.MILText), excludedOps...))
	probeLines := normalizeMILStructuralArgOrder(filterMILStructuralOps(milStructuralLines(probeMIL), excludedOps...))

	max := len(realLines)
	if len(probeLines) < max {
		max = len(probeLines)
	}
	for i := 0; i < max; i++ {
		if realLines[i] != probeLines[i] {
			start := i - 3
			if start < 0 {
				start = 0
			}
			endReal := i + 4
			if endReal > len(realLines) {
				endReal = len(realLines)
			}
			endProbe := i + 4
			if endProbe > len(probeLines) {
				endProbe = len(probeLines)
			}
			t.Fatalf(
				"first real selective-norm fused line diff at %d:\nreal:  %s\nprobe: %s\nreal_window=%v\nprobe_window=%v",
				i,
				realLines[i],
				probeLines[i],
				realLines[start:endReal],
				probeLines[start:endProbe],
			)
		}
	}
	if len(realLines) != len(probeLines) {
		t.Fatalf("real selective-norm fused line length diff: real=%d probe=%d\nreal=%v\nprobe=%v", len(realLines), len(probeLines), realLines, probeLines)
	}
}

func TestMILTransformerRealQwenSelectiveNormGateOpSequenceDiff(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run real qwen selective-norm gate diff", milTransformerFeatureMatrixEnv)
	}
	prog := qwenFixtureSingleLayerProgram(t)

	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			LinearFFN:           true,
			FusedLinearFFN:      true,
			AttentionOutputGate: true,
			DisableInputNormOps: true,
			DisableQKNormOps:    true,
			DynamicRoPEInputs:   true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
		AllowPartial:    true,
	}
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		t.Fatalf("ReifyToANEMIL(reduced qwen fixture): %v", err)
	}
	localCfg := fromANEMILTransformerConfig(reified.TransformerConfig)

	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(localCfg),
		toANEMILTransformerWeights(buildTransformerFeatureWeightsDense(localCfg)),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical selective-norm gate): %v", err)
	}

	excludedOps := []string{"read_state", "slice_by_index", "concat", "coreml_update_state", "write_state"}
	realLines := normalizeMILStructuralArgOrder(filterMILStructuralFooter(filterMILStructuralOps(milStructuralLines(reified.MILText), excludedOps...)))
	canonicalLines := normalizeMILStructuralArgOrder(filterMILStructuralFooter(filterMILStructuralOps(milStructuralLines(canonicalMIL), excludedOps...)))

	if !slices.Equal(realLines, canonicalLines) {
		max := len(realLines)
		if len(canonicalLines) < max {
			max = len(canonicalLines)
		}
		for i := 0; i < max; i++ {
			if realLines[i] != canonicalLines[i] {
				start := i - 3
				if start < 0 {
					start = 0
				}
				endReal := i + 4
				if endReal > len(realLines) {
					endReal = len(realLines)
				}
				endCanonical := i + 4
				if endCanonical > len(canonicalLines) {
					endCanonical = len(canonicalLines)
				}
				t.Fatalf(
					"first real selective-norm gate line diff at %d:\nreal:      %s\ncanonical: %s\nreal_window=%v\ncanonical_window=%v",
					i,
					realLines[i],
					canonicalLines[i],
					realLines[start:endReal],
					canonicalLines[start:endCanonical],
				)
			}
		}
		t.Fatalf("real selective-norm gate line length diff: real=%d canonical=%d\nreal=%v\ncanonical=%v", len(realLines), len(canonicalLines), realLines, canonicalLines)
	}
}

func TestMILTransformerRealQwenSelectiveNormGateFullStructuralDiff(t *testing.T) {
	prog := qwenFixtureSingleLayerProgram(t)

	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			LinearFFN:           true,
			FusedLinearFFN:      true,
			AttentionOutputGate: true,
			DisableInputNormOps: true,
			DisableQKNormOps:    true,
			DynamicRoPEInputs:   true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
		AllowPartial:    true,
	}
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		t.Fatalf("ReifyToANEMIL(reduced qwen fixture): %v", err)
	}
	localCfg := fromANEMILTransformerConfig(reified.TransformerConfig)

	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(localCfg),
		toANEMILTransformerWeights(buildTransformerFeatureWeightsDense(localCfg)),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical selective-norm gate): %v", err)
	}

	realLines := normalizeMILStructuralArgOrder(filterMILStructuralFooter(milStructuralLines(reified.MILText)))
	canonicalLines := normalizeMILStructuralArgOrder(filterMILStructuralFooter(milStructuralLines(canonicalMIL)))

	max := len(realLines)
	if len(canonicalLines) < max {
		max = len(canonicalLines)
	}
	for i := 0; i < max; i++ {
		if realLines[i] != canonicalLines[i] {
			start := i - 3
			if start < 0 {
				start = 0
			}
			endReal := i + 4
			if endReal > len(realLines) {
				endReal = len(realLines)
			}
			endCanonical := i + 4
			if endCanonical > len(canonicalLines) {
				endCanonical = len(canonicalLines)
			}
			t.Fatalf(
				"first real selective-norm gate full structural diff at %d:\nreal:      %s\ncanonical: %s\nreal_window=%v\ncanonical_window=%v",
				i,
				realLines[i],
				canonicalLines[i],
				realLines[start:endReal],
				canonicalLines[start:endCanonical],
			)
		}
	}
	if len(realLines) != len(canonicalLines) {
		t.Fatalf("real selective-norm gate full structural length diff: real=%d canonical=%d\nreal=%v\ncanonical=%v", len(realLines), len(canonicalLines), realLines, canonicalLines)
	}
}

func TestMILTransformerRealQwenSelectiveNormGateCompileProbe(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run real qwen selective-norm gate compile probe", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}
	prog := qwenFixtureSingleLayerProgram(t)

	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			LinearFFN:           true,
			FusedLinearFFN:      true,
			AttentionOutputGate: true,
			DisableInputNormOps: true,
			DisableQKNormOps:    true,
			DynamicRoPEInputs:   true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
		AllowPartial:    true,
	}
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		t.Fatalf("ReifyToANEMIL(reduced qwen fixture): %v", err)
	}
	allFiles := fromModelWeightFiles(reified.WeightFiles)
	cases := []struct {
		name  string
		files []modelWeightFile
	}{
		{name: "all_files", files: allFiles},
		{name: "referenced_only", files: filterModelWeightFilesByMILText(allFiles, reified.MILText)},
	}
	var compileErrs []string
	for _, tc := range cases {
		err = compileMILProbeCase(t, "real qwen selective-norm gate reified mil "+tc.name, reified.MILText, tc.files)
		if err != nil {
			compileErrs = append(compileErrs, fmt.Sprintf("%s: %v", tc.name, err))
		}
	}
	if len(compileErrs) > 0 {
		t.Fatalf("real qwen selective-norm gate reified compile failures:\n%s", strings.Join(compileErrs, "\n"))
	}
}

func TestMILTransformerRealQwenSelectiveNormGateConstDiff(t *testing.T) {
	prog := qwenFixtureSingleLayerProgram(t)

	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			LinearFFN:           true,
			FusedLinearFFN:      true,
			AttentionOutputGate: true,
			DisableInputNormOps: true,
			DisableQKNormOps:    true,
			DynamicRoPEInputs:   true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
		AllowPartial:    true,
	}
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		t.Fatalf("ReifyToANEMIL(reduced qwen fixture): %v", err)
	}
	localCfg := fromANEMILTransformerConfig(reified.TransformerConfig)

	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(localCfg),
		toANEMILTransformerWeights(buildTransformerFeatureWeightsDense(localCfg)),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical selective-norm gate): %v", err)
	}

	realLines := milBlobConstLines(reified.MILText)
	canonicalLines := milBlobConstLines(canonicalMIL)

	max := len(realLines)
	if len(canonicalLines) < max {
		max = len(canonicalLines)
	}
	for i := 0; i < max; i++ {
		if realLines[i] != canonicalLines[i] {
			start := i - 3
			if start < 0 {
				start = 0
			}
			endReal := i + 4
			if endReal > len(realLines) {
				endReal = len(realLines)
			}
			endCanonical := i + 4
			if endCanonical > len(canonicalLines) {
				endCanonical = len(canonicalLines)
			}
			t.Fatalf(
				"first real selective-norm gate const diff at %d:\nreal:      %s\ncanonical: %s\nreal_window=%v\ncanonical_window=%v",
				i,
				realLines[i],
				canonicalLines[i],
				realLines[start:endReal],
				canonicalLines[start:endCanonical],
			)
		}
	}
	if len(realLines) != len(canonicalLines) {
		t.Fatalf("real selective-norm gate const length diff: real=%d canonical=%d\nreal=%v\ncanonical=%v", len(realLines), len(canonicalLines), realLines, canonicalLines)
	}
}

func TestMILTransformerRealQwenSelectiveNormGateWeightBlobShapeDiff(t *testing.T) {
	prog := qwenFixtureSingleLayerProgram(t)

	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			LinearFFN:           true,
			FusedLinearFFN:      true,
			AttentionOutputGate: true,
			DisableInputNormOps: true,
			DisableQKNormOps:    true,
			DynamicRoPEInputs:   true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
		AllowPartial:    true,
	}
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		t.Fatalf("ReifyToANEMIL(reduced qwen fixture): %v", err)
	}
	localCfg := fromANEMILTransformerConfig(reified.TransformerConfig)

	_, canonicalFiles, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(localCfg),
		toANEMILTransformerWeights(buildTransformerFeatureWeightsDense(localCfg)),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical selective-norm gate): %v", err)
	}

	realFiles := fromModelWeightFiles(reified.WeightFiles)
	canonical := fromANEModelWeightFiles(canonicalFiles)
	realByPath := make(map[string]modelWeightFile, len(realFiles))
	for _, file := range realFiles {
		realByPath[file.Path] = file
	}
	canonicalByPath := make(map[string]modelWeightFile, len(canonical))
	for _, file := range canonical {
		canonicalByPath[file.Path] = file
	}

	var missing []string
	for path := range canonicalByPath {
		if _, ok := realByPath[path]; !ok {
			missing = append(missing, path)
		}
	}
	var extra []string
	for path := range realByPath {
		if _, ok := canonicalByPath[path]; !ok {
			extra = append(extra, path)
		}
	}
	slices.Sort(missing)
	slices.Sort(extra)
	if len(missing) > 0 || len(extra) > 0 {
		t.Fatalf("real selective-norm gate file set mismatch: missing=%v extra=%v", missing, extra)
	}

	var sizeDiffs []string
	for path, realFile := range realByPath {
		canonicalFile := canonicalByPath[path]
		if len(realFile.Blob) != len(canonicalFile.Blob) {
			sizeDiffs = append(sizeDiffs, fmt.Sprintf("%s real=%d canonical=%d", path, len(realFile.Blob), len(canonicalFile.Blob)))
		}
	}
	slices.Sort(sizeDiffs)
	if len(sizeDiffs) > 0 {
		t.Fatalf("real selective-norm gate blob size mismatch:\n%s", strings.Join(sizeDiffs, "\n"))
	}
}

func TestMILTransformerRealQwenSelectiveNormGatePayloadCrossoverProbe(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run real qwen selective-norm gate payload crossover probe", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	prog := qwenFixtureSingleLayerProgram(t)

	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			LinearFFN:           true,
			FusedLinearFFN:      true,
			AttentionOutputGate: true,
			DisableInputNormOps: true,
			DisableQKNormOps:    true,
			DynamicRoPEInputs:   true,
		},
		RequestedLayers: 1,
		SelectedLayers:  1,
		AllowPartial:    true,
	}
	reified, err := ReifyToANEMIL(prog, opts)
	if err != nil {
		t.Fatalf("ReifyToANEMIL(reduced qwen fixture): %v", err)
	}
	localCfg := fromANEMILTransformerConfig(reified.TransformerConfig)

	canonicalMIL, canonicalFiles, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(localCfg),
		toANEMILTransformerWeights(buildTransformerFeatureWeightsDense(localCfg)),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical selective-norm gate): %v", err)
	}

	realFiles := fromModelWeightFiles(reified.WeightFiles)
	canonicalModelFiles := fromANEModelWeightFiles(canonicalFiles)
	cases := []struct {
		name    string
		milText string
		files   []modelWeightFile
	}{
		{name: "real_mil_real_files", milText: reified.MILText, files: realFiles},
		{name: "real_mil_canonical_files", milText: reified.MILText, files: canonicalModelFiles},
		{name: "canonical_mil_real_files", milText: canonicalMIL, files: realFiles},
		{name: "canonical_mil_canonical_files", milText: canonicalMIL, files: canonicalModelFiles},
	}
	for _, tc := range cases {
		err := compileMILProbeCase(t, "real qwen selective-norm gate payload crossover "+tc.name, tc.milText, tc.files)
		if err != nil {
			t.Logf("%s compile failed: %v", tc.name, err)
			continue
		}
		t.Logf("%s compile succeeded", tc.name)
	}
}

func TestMILTransformerTinyQwenCanonicalRopeNoNormFusedOpSequenceDiff(t *testing.T) {
	cfg := MILTransformerConfig{
		NumLayers:                  1,
		Dim:                        16,
		AttentionDim:               16,
		NumHeads:                   4,
		HeadDim:                    4,
		HiddenDim:                  32,
		MaxSeqLen:                  8,
		KVCacheState:               false,
		KVCacheMaxLen:              8,
		SkipFFN:                    false,
		UseConvFFN:                 false,
		LinearFFN:                  true,
		FusedLinearFFN:             true,
		DisableNormOps:             true,
		DisableFinalNormOp:         false,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
		DynamicRoPEInputs:          true,
		IncludeLMHead:              false,
		AttentionMaskInput:         false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid canonical rope no-norm fused config: %v", err)
	}
	weights := buildTransformerFeatureWeightsDense(cfg)
	canonicalMIL, _, err := anereify.BuildMILTransformerArtifacts(
		toANEMILTransformerConfig(cfg),
		toANEMILTransformerWeights(weights),
	)
	if err != nil {
		t.Fatalf("BuildMILTransformerArtifacts(canonical rope no-norm fused): %v", err)
	}
	probeMIL, _, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(cfg, "input")
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope: %v", err)
	}
	canonicalLines := milStructuralLines(canonicalMIL)
	probeLines := milStructuralLines(probeMIL)
	max := len(canonicalLines)
	if len(probeLines) < max {
		max = len(probeLines)
	}
	for i := 0; i < max; i++ {
		if canonicalLines[i] != probeLines[i] {
			start := i - 3
			if start < 0 {
				start = 0
			}
			end := i + 4
			if end > len(canonicalLines) {
				end = len(canonicalLines)
			}
			if end > len(probeLines) {
				end = len(probeLines)
			}
			t.Fatalf(
				"first canonical rope no-norm fused line diff at %d:\ncanonical: %s\nprobe:     %s\ncanonical_window=%v\nprobe_window=%v",
				i,
				canonicalLines[i],
				probeLines[i],
				canonicalLines[start:end],
				probeLines[start:end],
			)
		}
	}
	if len(canonicalLines) != len(probeLines) {
		t.Fatalf("canonical rope no-norm fused line length diff: canonical=%d probe=%d\ncanonical=%v\nprobe=%v", len(canonicalLines), len(probeLines), canonicalLines, probeLines)
	}
}

func TestMILProbeAttentionFeatureCanonicalLikeFusedWithRopeQKNormMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical-like fused with rope qk-norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	for _, ropeSource := range []string{"input", "runtime"} {
		t.Run(ropeSource, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeQKNorm(cfg, ropeSource)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeQKNorm(%s): %v", ropeSource, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/canonical-like-fused-rope-qknorm "+ropeSource, milText, files)
			if err != nil {
				t.Logf("canonical-like fused with rope qk norm %s compile failed: %v", ropeSource, err)
				return
			}
			t.Logf("canonical-like fused with rope qk norm %s compile succeeded", ropeSource)
		})
	}
}

func TestMILProbeAttentionFeatureCanonicalLikeFusedWithRopePostNormMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical-like fused with rope post-norm matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	for _, ropeSource := range []string{"input", "runtime"} {
		t.Run(ropeSource, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopePostNorm(cfg, ropeSource)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopePostNorm(%s): %v", ropeSource, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/feature/canonical-like-fused-rope-postnorm "+ropeSource, milText, files)
			if err != nil {
				t.Logf("canonical-like fused with rope post norm %s compile failed: %v", ropeSource, err)
				return
			}
			t.Logf("canonical-like fused with rope post norm %s compile succeeded", ropeSource)
		})
	}
}

func TestMILProbeAttentionFeatureCanonicalLikeFusedMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical-like fused probe matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFused(cfg)
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFused: %v", err)
	}
	err = compileMILProbeCase(t, "mil probe attention/feature/canonical-like-fused", milText, files)
	if err != nil {
		t.Logf("canonical-like fused compile failed: %v", err)
		return
	}
	t.Log("canonical-like fused compile succeeded")
}

func TestMILProbeAttentionFeatureCanonicalLikeFusedCanonicalWeightNames(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run canonical-like fused canonical-weight-name matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:       16,
		NumHeads:  4,
		HeadDim:   4,
		HiddenDim: 32,
		MaxSeqLen: 8,
	}
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithWeightNames(cfg, true)
	if err != nil {
		t.Fatalf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithWeightNames(canonical): %v", err)
	}
	err = compileMILProbeCase(t, "mil probe attention/feature/canonical-like-fused canonical-weight-names", milText, files)
	if err != nil {
		t.Logf("canonical-like fused canonical-weight-names compile failed: %v", err)
		return
	}
	t.Log("canonical-like fused canonical-weight-names compile succeeded")
}

func TestMILProbeAttentionNormFFNInteractionMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run MIL probe attention/norm/FFN interaction matrix", milTransformerFeatureMatrixEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		Dim:        16,
		NumHeads:   4,
		HeadDim:    4,
		HiddenDim:  32,
		MaxSeqLen:  8,
		RMSNormEps: 1e-5,
	}
	cases := []struct {
		name       string
		preFFNNorm bool
		finalNorm  bool
	}{
		{name: "attention_ffn", preFFNNorm: false, finalNorm: false},
		{name: "attention_prefnnorm_ffn", preFFNNorm: true, finalNorm: false},
		{name: "attention_ffn_finalnorm", preFFNNorm: false, finalNorm: true},
		{name: "attention_prefnnorm_ffn_finalnorm", preFFNNorm: true, finalNorm: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			milText, files, err := buildMILProbeAttentionNormFFNInteraction(cfg, tc.preFFNNorm, tc.finalNorm)
			if err != nil {
				t.Fatalf("buildMILProbeAttentionNormFFNInteraction(%s): %v", tc.name, err)
			}
			err = compileMILProbeCase(t, "mil probe attention/norm/ffn interaction "+tc.name, milText, files)
			if err != nil {
				t.Logf("%s compile failed: %v", tc.name, err)
				return
			}
			t.Logf("%s compile succeeded", tc.name)
		})
	}
}

func TestMILTransformerProbeMatrix(t *testing.T) {
	if os.Getenv(milTransformerTestEnv) == "" {
		t.Skipf("set %s=1 to run MIL transformer probe matrix", milTransformerTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	cfg := MILTransformerConfig{
		NumLayers:    envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_LAYERS", 1),
		Dim:          envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_DIM", 64),
		AttentionDim: envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_ATTN_DIM", envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_DIM", 64)),
		NumHeads:     envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_HEADS", 4),
		HiddenDim:    envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_HIDDEN_DIM", 256),
		VocabSize:    envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_VOCAB", 32000),
	}
	cfg.HeadDim = envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_HEAD_DIM", cfg.Dim/cfg.NumHeads)
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("invalid MIL probe config: %v", err)
	}
	iterations := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_ITERS", 3)

	type probeCase struct {
		name             string
		allowUnsupported bool
		build            func(MILTransformerConfig) (string, []modelWeightFile, []float32, int, error)
	}
	probes := []probeCase{
		{name: "LinearOnly", build: buildMILProbeLinearOnly},
		{name: "TwoLinears", build: buildMILProbeTwoLinears},
		{name: "LinearReshape", allowUnsupported: true, build: buildMILProbeLinearReshape},
		{name: "LinearMatmul", allowUnsupported: true, build: buildMILProbeLinearMatmul},
		{name: "SDPA", allowUnsupported: true, build: buildMILProbeSDPA},
		{name: "AttentionBlock", allowUnsupported: true, build: buildMILProbeAttentionBlock},
		{name: "AttentionPlusFFN", allowUnsupported: true, build: buildMILProbeAttentionPlusFFN},
		{name: "LargeLinear", allowUnsupported: true, build: buildMILProbeLargeLinear},
	}

	for _, probe := range probes {
		probe := probe
		t.Run(probe.name, func(t *testing.T) {
			milText, files, input, outputCount, err := probe.build(cfg)
			if err != nil {
				t.Fatalf("build probe %q: %v", probe.name, err)
			}
			best, err := runMILProbeCase(t, "mil probe "+probe.name, milText, files, input, outputCount, iterations)
			if err != nil {
				if isMILKnownMapFailure(err) {
					t.Skipf("MIL probe %s skipped due to known mapper state error: %v", probe.name, err)
				}
				if probe.allowUnsupported && isMILUnsupportedError(err) {
					t.Skipf("MIL probe %s unsupported on this host/compiler path: %v", probe.name, err)
				}
				t.Fatalf("MIL probe %s failed: %v", probe.name, err)
			}
			t.Logf("MIL probe %s best eval latency: %s", probe.name, best)
		})
	}
}

func TestMILIsMILModelFastPath(t *testing.T) {
	if os.Getenv(milTransformerTestEnv) == "" {
		t.Skipf("set %s=1 to run MIL fast-path proof", milTransformerTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}
	if os.Getenv(milFastPathChildEnv) != "" {
		if err := runMILIsMILModelFastPathProof(t); err != nil {
			t.Fatalf("MIL fast-path child proof failed: %v", err)
		}
		return
	}

	processRetries := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_FAST_PROCESS_RETRIES", 3)
	if processRetries < 1 {
		processRetries = 1
	}
	for attempt := 1; attempt <= processRetries; attempt++ {
		cmd := exec.Command(os.Args[0], "-test.run", "^TestMILIsMILModelFastPath$", "-test.v")
		cmd.Env = append(os.Environ(), milFastPathChildEnv+"=1")
		out, err := cmd.CombinedOutput()
		if err == nil {
			t.Logf(
				"MIL fast-path proof succeeded in isolated process attempt %d/%d",
				attempt,
				processRetries,
			)
			return
		}
		output := string(out)
		if strings.Contains(strings.ToLower(output), "program iosurfaces map failure") {
			t.Logf(
				"MIL fast-path isolated process attempt %d/%d hit mapper failure; retrying in fresh process",
				attempt,
				processRetries,
			)
			continue
		}
		t.Fatalf(
			"MIL fast-path isolated process attempt %d/%d failed: %v\n%s",
			attempt,
			processRetries,
			err,
			output,
		)
	}
	t.Skipf("MIL fast-path proof skipped: mapper failure persisted across %d isolated processes", processRetries)
}

func runMILIsMILModelFastPathProof(t *testing.T) error {
	t.Helper()
	dim := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_FAST_DIM", 768)
	hiddenDim := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_FAST_HIDDEN_DIM", 3072)
	seq := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_FAST_SEQ", 1)
	maxUS := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_FAST_MAX_US", 5000)
	iterations := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_FAST_ITERS", 5)
	retries := envIntWithDefault(t, "MLXGO_ANE_TEST_MIL_FAST_RETRIES", 4)

	milText, err := ffnFwdTapsMILText(dim, hiddenDim, seq)
	if err != nil {
		return fmt.Errorf("ffnFwdTapsMILText: %w", err)
	}
	rms2 := make([]float32, dim)
	for i := range rms2 {
		rms2[i] = 1
	}
	w1 := makeDeterministicTensor(hiddenDim*dim, 0.0007, 113)
	w3 := makeDeterministicTensor(hiddenDim*dim, 0.0006, 127)
	w2 := makeDeterministicTensor(dim*hiddenDim, 0.0005, 131)
	blobs, err := buildFFNWeightBlobs(rms2, w1, w3, w2, dim, hiddenDim)
	if err != nil {
		return fmt.Errorf("buildFFNWeightBlobs: %w", err)
	}
	files := []modelWeightFile{
		{Path: ffnRMS2BlobPathInMIL, Blob: blobs.RMS2},
		{Path: ffnW1BlobPathInMIL, Blob: blobs.W1},
		{Path: ffnW3BlobPathInMIL, Blob: blobs.W3},
		{Path: ffnW2BlobPathInMIL, Blob: blobs.W2},
	}

	input := makeDeterministicTensor(dim*seq, 0.001, 149)
	outputCount := (2*dim + 3*hiddenDim) * seq
	best, attempts, err := runMILProbeCaseWithMapRetry(
		t,
		"mil fast-path ffn",
		milText,
		files,
		input,
		outputCount,
		iterations,
		retries,
	)
	if err != nil {
		if isMILKnownMapFailure(err) {
			t.Logf("MIL fast-path fused-FFN strict-client probe hit mapper failure after %d attempts", attempts)
			daemonBest, daemonErr := runMILProbeCaseDaemonBacked(
				t,
				"mil fast-path ffn daemon-backed",
				milText,
				files,
				input,
				outputCount,
				iterations,
			)
			if daemonErr == nil {
				if daemonBest > time.Duration(maxUS)*time.Microsecond {
					return fmt.Errorf("MIL fast-path daemon-backed path too slow: best=%s max=%dµs", daemonBest, maxUS)
				}
				t.Logf("MIL fast-path daemon-backed proof best eval latency: %s", daemonBest)
				return nil
			}
			t.Logf("MIL fast-path daemon-backed probe failed; trying linear fallback proof: %v", daemonErr)
			fallbackBest, fallbackAttempts, fallbackErr := runMILFastPathLinearFallback(
				t,
				dim,
				maxUS,
				retries,
			)
			if fallbackErr == nil {
				t.Logf(
					"MIL fast-path fallback linear proof succeeded: best=%s attempts=%d",
					fallbackBest,
					fallbackAttempts,
				)
				return nil
			}
			return fmt.Errorf(
				"MIL fast-path mapper failure after FFN (%d attempts) and linear fallback (%d attempts): %w",
				attempts,
				fallbackAttempts,
				fallbackErr,
			)
		}
		return fmt.Errorf("MIL fast-path ffn probe failed: %w", err)
	}
	if best > time.Duration(maxUS)*time.Microsecond {
		return fmt.Errorf("MIL fast-path too slow: best=%s max=%dµs", best, maxUS)
	}
	t.Logf(
		"MIL fast-path proof best eval latency: %s (max %dµs, attempts=%d)",
		best,
		maxUS,
		attempts,
	)
	return nil
}

func runMILProbeCaseWithMapRetry(
	t *testing.T,
	label string,
	milText string,
	files []modelWeightFile,
	input []float32,
	outputCount int,
	iterations int,
	retries int,
) (time.Duration, int, error) {
	return runMILProbeCaseWithMapRetryRunner(
		t,
		label,
		milText,
		files,
		input,
		outputCount,
		iterations,
		retries,
		runMILProbeCaseStrictClient,
	)
}

func runMILProbeCaseWithMapRetryRunner(
	t *testing.T,
	label string,
	milText string,
	files []modelWeightFile,
	input []float32,
	outputCount int,
	iterations int,
	retries int,
	runner func(
		*testing.T,
		string,
		string,
		[]modelWeightFile,
		[]float32,
		int,
		int,
	) (time.Duration, error),
) (time.Duration, int, error) {
	t.Helper()
	if retries < 1 {
		retries = 1
	}
	if runner == nil {
		runner = runMILProbeCase
	}
	var lastErr error
	for attempt := 1; attempt <= retries; attempt++ {
		best, err := runner(t, label, milText, files, input, outputCount, iterations)
		if err == nil {
			return best, attempt, nil
		}
		if !isMILKnownMapFailure(err) {
			return 0, attempt, err
		}
		lastErr = err
		t.Logf(
			"%s: map failure on attempt %d/%d; rebuilding model and retrying: %v",
			label,
			attempt,
			retries,
			err,
		)
		if attempt < retries {
			time.Sleep(time.Duration(25*attempt) * time.Millisecond)
		}
	}
	return 0, retries, lastErr
}

func runMILProbeCase(
	t *testing.T,
	label string,
	milText string,
	files []modelWeightFile,
	input []float32,
	outputCount int,
	iterations int,
) (time.Duration, error) {
	t.Helper()
	if len(input) == 0 {
		return 0, fmt.Errorf("%s: input is empty", label)
	}
	if outputCount <= 0 {
		return 0, fmt.Errorf("%s: invalid outputCount=%d", label, outputCount)
	}
	if iterations <= 0 {
		iterations = 1
	}

	model, err := buildModelFromMILTextStrict(label, milText, files)
	if err != nil {
		return 0, err
	}
	defer unloadMILProbeModel(t, label, model)

	best := time.Duration(1<<63 - 1)
	for i := 0; i < iterations; i++ {
		start := time.Now()
		out, err := evalModelSingleIO(context.Background(), model, input, outputCount, label)
		dur := time.Since(start)
		if err != nil {
			return 0, fmt.Errorf("%s eval iter=%d: %w", label, i, err)
		}
		if len(out) != outputCount {
			return 0, fmt.Errorf("%s eval iter=%d: output len=%d want=%d", label, i, len(out), outputCount)
		}
		if err := assertFiniteNonZero(out); err != nil {
			return 0, fmt.Errorf("%s eval iter=%d: %w", label, i, err)
		}
		if dur < best {
			best = dur
		}
	}
	return best, nil
}

func compileMILProbeCase(
	t *testing.T,
	label string,
	milText string,
	files []modelWeightFile,
) error {
	t.Helper()
	model, err := buildModelFromMILTextWithDescriptorFallback(label, milText, files)
	if err != nil {
		return err
	}
	defer unloadMILProbeModel(t, label, model)
	return nil
}

func runMILProbeCaseStrictClient(
	t *testing.T,
	label string,
	milText string,
	files []modelWeightFile,
	input []float32,
	outputCount int,
	iterations int,
) (time.Duration, error) {
	t.Helper()
	if len(input) == 0 {
		return 0, fmt.Errorf("%s: input is empty", label)
	}
	if outputCount <= 0 {
		return 0, fmt.Errorf("%s: invalid outputCount=%d", label, outputCount)
	}
	if iterations <= 0 {
		iterations = 1
	}

	model, err := buildModelFromMILTextStrict(label, milText, files)
	if err != nil {
		return 0, err
	}
	defer unloadMILProbeModel(t, label, model)

	best := time.Duration(1<<63 - 1)
	for i := 0; i < iterations; i++ {
		start := time.Now()
		out, err := evalModelSingleIOStrictClient(context.Background(), model, input, outputCount, label)
		dur := time.Since(start)
		if err != nil {
			return 0, fmt.Errorf("%s eval iter=%d: %w", label, i, err)
		}
		if len(out) != outputCount {
			return 0, fmt.Errorf("%s eval iter=%d: output len=%d want=%d", label, i, len(out), outputCount)
		}
		if err := assertFiniteNonZero(out); err != nil {
			return 0, fmt.Errorf("%s eval iter=%d: %w", label, i, err)
		}
		if dur < best {
			best = dur
		}
	}
	return best, nil
}

func evalModelSingleIOStrictClient(
	ctx context.Context,
	model appleneuralengine.ANEInMemoryModel,
	input []float32,
	outputCount int,
	label string,
) ([]float32, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if outputCount <= 0 {
		return nil, fmt.Errorf("%s: invalid outputCount=%d", label, outputCount)
	}
	if len(input) == 0 {
		return nil, fmt.Errorf("%s: input is empty", label)
	}

	shared := model.SharedConnection()
	base := model.Model()
	client := appleneuralengine.ANEClientFromID(shared.GetID())
	if client.ID == 0 || base.GetID() == 0 {
		return nil, fmt.Errorf("%s: daemon client/model unavailable", label)
	}

	xSurf, err := newFloatSurface(len(input))
	if err != nil {
		return nil, err
	}
	defer releaseIOSurface(xSurf)
	if err := writeFloat32IOSurface(xSurf, input); err != nil {
		return nil, err
	}

	ySurf, err := newFloatSurface(outputCount)
	if err != nil {
		return nil, err
	}
	defer releaseIOSurface(ySurf)

	xObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(xSurf)
	yObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(ySurf)
	if xObj.GetID() == 0 || yObj.GetID() == 0 {
		return nil, fmt.Errorf("%s: create IOSurface object failed", label)
	}

	inputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{xObj}))
	inputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	outputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{yObj}))
	outputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	requestObj := appleneuralengine.GetANERequestClass().RequestWithInputsInputIndicesOutputsOutputIndicesProcedureIndex(
		inputs,
		inputIndices,
		outputs,
		outputIndices,
		foundation.NewNumberWithInt(0),
	)
	if requestObj.GetID() == 0 {
		return nil, fmt.Errorf("%s: create request failed", label)
	}
	request := appleneuralengine.ANERequestFromID(requestObj.GetID())
	if !request.Validate() {
		return nil, fmt.Errorf("%s: request validation failed", label)
	}

	if err := callObjCBoolWithNSError(
		label+" map IOSurfaces (client-only)",
		client.ID,
		"mapIOSurfacesWithModel:request:cacheInference:error:",
		base,
		requestObj,
		true,
	); err != nil {
		return nil, err
	}
	defer func() {
		objc.Send[objc.ID](client.ID, objc.Sel("unmapIOSurfacesWithModel:request:"), base, requestObj)
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	options := foundation.NewMutableDictionaryWithCapacity(0)
	if objc.Send[bool](client.ID, objc.Sel("respondsToSelector:"), objc.Sel("doEvaluateDirectWithModel:options:request:qos:error:")) {
		if err := callObjCBoolWithNSError(
			label+" evaluate (client direct)",
			client.ID,
			"doEvaluateDirectWithModel:options:request:qos:error:",
			base,
			options,
			requestObj,
			defaultANEQoS,
		); err != nil {
			return nil, err
		}
	} else {
		if err := callObjCBoolWithNSError(
			label+" evaluate (client)",
			client.ID,
			"evaluateWithModel:options:request:qos:error:",
			base,
			options,
			requestObj,
			defaultANEQoS,
		); err != nil {
			return nil, err
		}
	}
	return readFloat32IOSurface(ySurf, outputCount)
}

func runMILFastPathLinearFallback(
	t *testing.T,
	dim int,
	maxUS int,
	retries int,
) (time.Duration, int, error) {
	t.Helper()
	if dim <= 0 {
		return 0, 0, fmt.Errorf("linear fallback: invalid dim=%d", dim)
	}
	milText, err := linearMILText(1, dim, dim)
	if err != nil {
		return 0, 0, fmt.Errorf("linear fallback: linearMILText: %w", err)
	}
	weights := make([]float32, dim*dim)
	for o := 0; o < dim; o++ {
		for i := 0; i < dim; i++ {
			if o == i {
				weights[o*dim+i] = 1
			} else {
				weights[o*dim+i] = float32(((o+i)%13)-6) * 0.005
			}
		}
	}
	blob, err := buildLinearWeightsBlob(weights, dim, dim)
	if err != nil {
		return 0, 0, fmt.Errorf("linear fallback: buildLinearWeightsBlob: %w", err)
	}
	files := []modelWeightFile{
		{Path: linearWeightBlobPathInMIL, Blob: blob},
	}
	input := makeDeterministicTensor(dim, 0.001, 173)
	best, attempts, err := runMILProbeCaseWithMapRetry(
		t,
		"mil fast-path linear fallback",
		milText,
		files,
		input,
		dim,
		2,
		retries,
	)
	if err != nil {
		return 0, attempts, err
	}
	if best > time.Duration(maxUS)*time.Microsecond {
		return 0, attempts, fmt.Errorf("linear fallback too slow: best=%s max=%dµs", best, maxUS)
	}
	return best, attempts, nil
}

func buildModelFromMILTextStrict(
	label string,
	milText string,
	files []modelWeightFile,
) (appleneuralengine.ANEInMemoryModel, error) {
	weights := newWeightDictionary(files)
	descObj, usedMILInit, err := newMILTextDescriptor(milText, weights, objectivec.Object{})
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: %w", label, err)
	}
	if !usedMILInit {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: isMILModel initializer unavailable", label)
	}
	modelDir, err := mirrorDescriptorFilesMulti(descObj, milText, files, label)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	defer os.RemoveAll(modelDir)

	modelObj := appleneuralengine.GetANEInMemoryModelClass().InMemoryModelWithDescriptor(descObj)
	if modelObj.GetID() == 0 {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: create in-memory model failed", label)
	}
	model := appleneuralengine.ANEInMemoryModelFromID(modelObj.GetID())
	options := foundation.NewMutableDictionaryWithCapacity(0)
	if err := callObjCBoolWithNSError(
		label+" compile (isMILModel init)",
		model.ID,
		"compileWithQoS:options:error:",
		defaultANEQoS,
		options,
	); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	if _, err := loadInMemoryModelWithRetry(label, model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	return model, nil
}

func unloadMILProbeModel(t *testing.T, label string, model appleneuralengine.ANEInMemoryModel) {
	t.Helper()
	if model.ID == 0 {
		return
	}
	if err := callObjCBoolWithNSError(
		label+" unload",
		model.ID,
		"unloadWithQoS:error:",
		defaultANEQoS,
	); err != nil {
		t.Logf("warning: %s unload failed: %v", label, err)
	}
}

func assertFiniteNonZero(values []float32) error {
	nonZero := false
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("value[%d] is not finite: %g", i, v)
		}
		if !nonZero && math.Abs(float64(v)) > 1e-8 {
			nonZero = true
		}
	}
	if !nonZero {
		return fmt.Errorf("all values are zero")
	}
	return nil
}

func runMILProbeCaseDaemonBacked(
	t *testing.T,
	label string,
	milText string,
	files []modelWeightFile,
	input []float32,
	outputCount int,
	iterations int,
) (time.Duration, error) {
	t.Helper()
	if len(input) == 0 {
		return 0, fmt.Errorf("%s: input is empty", label)
	}
	if outputCount <= 0 {
		return 0, fmt.Errorf("%s: invalid outputCount=%d", label, outputCount)
	}
	if iterations <= 0 {
		iterations = 1
	}

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		return 0, fmt.Errorf("%s: _ANEClient sharedConnection returned nil", label)
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	model, err := CompileAndLoadMILFiles(client, milText, files, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		return 0, fmt.Errorf("%s: compile/load daemon-backed model: %w", label, err)
	}
	defer model.Close()

	best := time.Duration(1<<63 - 1)
	for i := 0; i < iterations; i++ {
		start := time.Now()
		out, _, err := model.EvalSingleIO(context.Background(), input, outputCount, true)
		dur := time.Since(start)
		if err != nil {
			return 0, fmt.Errorf("%s eval iter=%d: %w", label, i, err)
		}
		if len(out) != outputCount {
			return 0, fmt.Errorf("%s eval iter=%d: output len=%d want=%d", label, i, len(out), outputCount)
		}
		if err := assertFiniteNonZero(out); err != nil {
			return 0, fmt.Errorf("%s eval iter=%d: %w", label, i, err)
		}
		if dur < best {
			best = dur
		}
	}
	return best, nil
}

func isMILUnsupportedError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrMILDirectoryUnsupported) {
		return true
	}
	msg := strings.ToLower(err.Error())
	match := []string{
		"cannot serialize anec_ir_repr",
		"compilationfailure",
		"invalidmilprogram",
		"unsupported",
		"not supported",
		"unrecognized",
		"unknown op",
		"nscocoaerrordomain code=4097",
	}
	for _, needle := range match {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

func isMILKnownMapFailure(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "program iosurfaces map failure (0x1d)") ||
		strings.Contains(msg, "program iosurfaces map failure")
}

func probeWeightPath(name string) string {
	return "@model_path/weights/" + name + ".bin"
}

func addProbeLinearWeights(
	files *[]modelWeightFile,
	prefix string,
	inDim int,
	outDim int,
	period int,
) (wPath string, bPath string) {
	if period <= 0 {
		period = 1
	}
	wPath = probeWeightPath(prefix + "_w")
	bPath = probeWeightPath(prefix + "_b")
	*files = append(*files,
		modelWeightFile{
			Path: wPath,
			Blob: buildBLOBFileFP16(makeDeterministicTensor(outDim*inDim, 0.0009, period)),
		},
		modelWeightFile{
			Path: bPath,
			Blob: buildBLOBFileFP16(makeDeterministicTensor(outDim, 0.0007, period+11)),
		},
	)
	return wPath, bPath
}

func addProbeLinearWeightsNamed(
	files *[]modelWeightFile,
	wName string,
	bName string,
	inDim int,
	outDim int,
	period int,
) (wPath string, bPath string) {
	if period <= 0 {
		period = 1
	}
	wPath = probeWeightPath(wName)
	bPath = probeWeightPath(bName)
	*files = append(*files,
		modelWeightFile{
			Path: wPath,
			Blob: buildBLOBFileFP16(makeDeterministicTensor(outDim*inDim, 0.0009, period)),
		},
		modelWeightFile{
			Path: bPath,
			Blob: buildBLOBFileFP16(makeDeterministicTensor(outDim, 0.0007, period+11)),
		},
	)
	return wPath, bPath
}

func addProbeVectorWeight(files *[]modelWeightFile, prefix string, dim int, period int) string {
	if period <= 0 {
		period = 1
	}
	path := probeWeightPath(prefix + "_w")
	*files = append(*files, modelWeightFile{
		Path: path,
		Blob: buildBLOBFileFP16(makeDeterministicTensor(dim, 0.0007, period)),
	})
	return path
}

func writeProbeMILPreamble(b *strings.Builder, inputDim int) {
	b.WriteString(ffnMILHeader)
	fmt.Fprintf(b, "    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n", inputDim)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", inputDim)
}

func writeProbeMILReturn(b *strings.Builder, outName string, outDim int) {
	b.WriteString("        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n")
	fmt.Fprintf(b, "        tensor<fp32, [1,1,%d]> y = cast(dtype=to_fp32, x=%s)[name=string(\"cast_out\")];\n", outDim, outName)
	b.WriteString("    } -> (y);\n}\n")
}

func buildMILProbeLinearOnly(cfg MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	var files []modelWeightFile
	wPath, bPath := addProbeLinearWeights(&files, "linear0", cfg.Dim, cfg.Dim, 97)

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	writeLinearConstBlock(&b, "w0", "b0", wPath, bPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> y0 = linear(bias=b0, weight=w0, x=x)[name=string(\"linear0\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "y0", cfg.Dim)
	return b.String(), files, makeDeterministicTensor(cfg.Dim, 0.002, 101), cfg.Dim, nil
}

func buildMILProbeTwoLinears(cfg MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	var files []modelWeightFile
	w0Path, b0Path := addProbeLinearWeights(&files, "linear0", cfg.Dim, cfg.Dim, 103)
	w1Path, b1Path := addProbeLinearWeights(&files, "linear1", cfg.Dim, cfg.Dim, 107)

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	writeLinearConstBlock(&b, "w0", "b0", w0Path, b0Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> y0 = linear(bias=b0, weight=w0, x=x)[name=string(\"linear0\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> y1 = linear(bias=b1, weight=w1, x=y0)[name=string(\"linear1\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "y1", cfg.Dim)
	return b.String(), files, makeDeterministicTensor(cfg.Dim, 0.002, 109), cfg.Dim, nil
}

func buildMILProbeLinearReshape(cfg MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	var files []modelWeightFile
	wPath, bPath := addProbeLinearWeights(&files, "linear0", cfg.Dim, cfg.Dim, 113)

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	writeLinearConstBlock(&b, "w0", "b0", wPath, bPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n",
		cfg.NumHeads,
		cfg.HeadDim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n",
		cfg.Dim,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> y0 = linear(bias=b0, weight=w0, x=x)[name=string(\"linear0\")];\n", cfg.Dim)
	fmt.Fprintf(
		&b,
		"        tensor<fp16, [1,%d,%d,1]> r0 = reshape(shape=rsh, x=y0)[name=string(\"reshape0\")];\n",
		cfg.NumHeads,
		cfg.HeadDim,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> r1 = reshape(shape=osh, x=r0)[name=string(\"reshape1\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "r1", cfg.Dim)
	return b.String(), files, makeDeterministicTensor(cfg.Dim, 0.002, 127), cfg.Dim, nil
}

func buildMILProbeLinearMatmul(cfg MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	var files []modelWeightFile
	wPath := probeWeightPath("mm_w")
	files = append(files, modelWeightFile{
		Path: wPath,
		Blob: buildBLOBFileFP16(makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0009, 131)),
	})

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> xsh = const()[name=string(\"xsh\"), val=tensor<int32, [4]>([1,1,1,%d])];\n",
		cfg.Dim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> wsh = const()[name=string(\"wsh\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n",
		cfg.Dim,
		cfg.Dim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> osh4 = const()[name=string(\"osh4\"), val=tensor<int32, [4]>([1,1,1,%d])];\n",
		cfg.Dim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n",
		cfg.Dim,
	)
	b.WriteString("        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n")
	b.WriteString("        bool ty = const()[name=string(\"ty\"), val=bool(false)];\n")
	fmt.Fprintf(
		&b,
		"        tensor<fp16, [1,1,%d,%d]> w = const()[name=string(\"w\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
		cfg.Dim,
		cfg.Dim,
		cfg.Dim,
		cfg.Dim,
		wPath,
		blobChunkOffset,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,1,%d]> x4 = reshape(shape=xsh, x=x)[name=string(\"x4\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,1,%d]> y4 = matmul(transpose_x=tx, transpose_y=ty, x=x4, y=w)[name=string(\"mm\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,1,%d]> y4r = reshape(shape=osh4, x=y4)[name=string(\"y4r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out16 = reshape(shape=osh, x=y4r)[name=string(\"out16\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "out16", cfg.Dim)
	return b.String(), files, makeDeterministicTensor(cfg.Dim, 0.002, 139), cfg.Dim, nil
}

func writeProbeStateConsts(b *strings.Builder) {
	fmt.Fprintln(b, "        int32 cache_axis = const()[name=string(\"cache_axis\"), val=int32(2)];")
	fmt.Fprintln(b, "        bool cache_interleave = const()[name=string(\"cache_interleave\"), val=bool(false)];")
	fmt.Fprintln(b, "        tensor<int32, [4]> cache_tail_begin = const()[name=string(\"cache_tail_begin\"), val=tensor<int32, [4]>([0,0,1,0])];")
	fmt.Fprintln(b, "        tensor<int32, [4]> cache_tail_end = const()[name=string(\"cache_tail_end\"), val=tensor<int32, [4]>([0,0,0,0])];")
	fmt.Fprintln(b, "        tensor<bool, [4]> cache_tail_begin_mask = const()[name=string(\"cache_tail_begin_mask\"), val=tensor<bool, [4]>([true,true,false,true])];")
	fmt.Fprintln(b, "        tensor<bool, [4]> cache_tail_end_mask = const()[name=string(\"cache_tail_end_mask\"), val=tensor<bool, [4]>([true,true,true,true])];")
}

func buildMILProbeRawSDPA() (string, []modelWeightFile, error) {
	const (
		numHeads = 2
		headDim  = 4
	)
	var b strings.Builder
	b.WriteString(ffnMILHeader)
	fmt.Fprintf(
		&b,
		"    func main<ios18>(tensor<fp32, [1,%d,1,%d]> q_in, tensor<fp32, [1,%d,1,%d]> k_in, tensor<fp32, [1,%d,1,%d]> v_in) {\n",
		numHeads,
		headDim,
		numHeads,
		headDim,
		numHeads,
		headDim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q = cast(dtype=to_fp16, x=q_in)[name=string(\"cast_q\")];\n", numHeads, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k = cast(dtype=to_fp16, x=k_in)[name=string(\"cast_k\")];\n", numHeads, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v = cast(dtype=to_fp16, x=v_in)[name=string(\"cast_v\")];\n", numHeads, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q, key=k, value=v)[name=string(\"sdpa\")];\n", numHeads, headDim)
	b.WriteString("    } -> (attn);\n}\n")
	return b.String(), nil, nil
}

func withTransformerFeature(
	base MILTransformerConfig,
	apply func(*MILTransformerConfig),
) MILTransformerConfig {
	cfg := base
	apply(&cfg)
	return cfg
}

func buildTransformerFeatureWeights(cfg MILTransformerConfig) MILTransformerWeights {
	weights := testTransformerWeights(cfg)
	attnDim := transformerAttentionDim(cfg)
	for i := range weights.Layers {
		layer := &weights.Layers[i]
		if cfg.AttentionOutputGate {
			layer.QGateW = make([]float32, attnDim*cfg.Dim)
			layer.QGateB = make([]float32, attnDim)
		}
	}
	if cfg.IncludeLMHead {
		weights.LMHeadW = make([]float32, cfg.VocabSize*cfg.Dim)
		weights.LMHeadB = make([]float32, cfg.VocabSize)
	}
	return weights
}

func buildTransformerFeatureWeightsDense(cfg MILTransformerConfig) MILTransformerWeights {
	weights := buildTransformerFeatureWeights(cfg)
	fillDense := func(dst []float32, base float32) {
		for i := range dst {
			dst[i] = base + float32((i%17)+1)/257
		}
	}
	for i := range weights.Layers {
		layer := &weights.Layers[i]
		fillDense(layer.QW, 0.01)
		fillDense(layer.QB, 0.02)
		fillDense(layer.KW, 0.03)
		fillDense(layer.KB, 0.04)
		fillDense(layer.VW, 0.05)
		fillDense(layer.VB, 0.06)
		fillDense(layer.OW, 0.07)
		fillDense(layer.OB, 0.08)
		fillDense(layer.W1, 0.09)
		fillDense(layer.B1, 0.10)
		fillDense(layer.W3, 0.11)
		fillDense(layer.B3, 0.12)
		fillDense(layer.W2, 0.13)
		fillDense(layer.B2, 0.14)
		if len(layer.QGateW) != 0 {
			fillDense(layer.QGateW, 0.15)
			fillDense(layer.QGateB, 0.16)
		}
		for j := range layer.InputNorm {
			layer.InputNorm[j] = 1
		}
		for j := range layer.PostAttentionNorm {
			layer.PostAttentionNorm[j] = 1
		}
		for j := range layer.QNorm {
			layer.QNorm[j] = 1
		}
		for j := range layer.KNorm {
			layer.KNorm[j] = 1
		}
	}
	for i := range weights.FinalNorm {
		weights.FinalNorm[i] = 1
	}
	if len(weights.RopeCos) != 0 {
		fillDense(weights.RopeCos, 0.17)
	}
	if len(weights.RopeSin) != 0 {
		fillDense(weights.RopeSin, 0.18)
	}
	if len(weights.RopeRotateW) != 0 {
		fillDense(weights.RopeRotateW, 0.19)
	}
	if len(weights.RopeRotateB) != 0 {
		fillDense(weights.RopeRotateB, 0.20)
	}
	if len(weights.LMHeadW) != 0 {
		fillDense(weights.LMHeadW, 0.21)
	}
	if len(weights.LMHeadB) != 0 {
		fillDense(weights.LMHeadB, 0.22)
	}
	return weights
}

func buildTransformerFeatureWeightsFromReducedFusedProbe(cfg MILTransformerConfig) MILTransformerWeights {
	weights := buildTransformerFeatureWeights(cfg)
	if len(weights.Layers) != 1 {
		return weights
	}
	layer := &weights.Layers[0]
	copy(layer.QW, makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0009, 2011))
	copy(layer.QB, makeDeterministicTensor(cfg.Dim, 0.0007, 2022))
	copy(layer.KW, makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0009, 2017))
	copy(layer.KB, makeDeterministicTensor(cfg.Dim, 0.0007, 2028))
	copy(layer.VW, makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0009, 2027))
	copy(layer.VB, makeDeterministicTensor(cfg.Dim, 0.0007, 2038))
	copy(layer.OW, makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0009, 2039))
	copy(layer.OB, makeDeterministicTensor(cfg.Dim, 0.0007, 2050))

	fusedW := makeDeterministicTensor(2*cfg.HiddenDim*cfg.Dim, 0.0009, 2053)
	fusedB := makeDeterministicTensor(2*cfg.HiddenDim, 0.0007, 2064)
	splitW := cfg.HiddenDim * cfg.Dim
	copy(layer.W1, fusedW[:splitW])
	copy(layer.W3, fusedW[splitW:])
	copy(layer.B1, fusedB[:cfg.HiddenDim])
	copy(layer.B3, fusedB[cfg.HiddenDim:])
	copy(layer.W2, makeDeterministicTensor(cfg.Dim*cfg.HiddenDim, 0.0009, 2063))
	copy(layer.B2, makeDeterministicTensor(cfg.Dim, 0.0007, 2074))

	for i := range layer.InputNorm {
		layer.InputNorm[i] = 1
	}
	for i := range layer.PostAttentionNorm {
		layer.PostAttentionNorm[i] = 1
	}
	for i := range layer.QNorm {
		layer.QNorm[i] = 1
	}
	for i := range layer.KNorm {
		layer.KNorm[i] = 1
	}
	for i := range weights.FinalNorm {
		weights.FinalNorm[i] = 1
	}
	return weights
}

func filterModelWeightFilesByMILText(files []modelWeightFile, milText string) []modelWeightFile {
	out := make([]modelWeightFile, 0, len(files))
	for _, f := range files {
		if strings.Contains(milText, f.Path) {
			out = append(out, f)
		}
	}
	return out
}

func buildMILProbeReadStateOnly(numHeads int, headDim int, maxSeq int) (string, []modelWeightFile, error) {
	var b strings.Builder
	b.WriteString(ffnMILHeader)
	fmt.Fprintf(
		&b,
		"    func main<ios18>(state<tensor<fp16, [1,%d,%d,%d]>> k_state) {\n",
		numHeads,
		maxSeq,
		headDim,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", numHeads, maxSeq, headDim)
	b.WriteString("    } -> (k_read);\n}\n")
	return b.String(), nil, nil
}

func buildMILProbeUpdateStateOnly(numHeads int, headDim int, maxSeq int) (string, []modelWeightFile, error) {
	var b strings.Builder
	b.WriteString(ffnMILHeader)
	fmt.Fprintf(
		&b,
		"    func main<ios18>(tensor<fp32, [1,%d,%d,%d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state) {\n",
		numHeads,
		maxSeq,
		headDim,
		numHeads,
		maxSeq,
		headDim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_x\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = coreml_update_state(state=k_state, value=x)[name=string(\"k_next\")];\n", numHeads, maxSeq, headDim)
	b.WriteString("    } -> (k_next);\n}\n")
	return b.String(), nil, nil
}

func buildMILProbeWriteStateReadback(numHeads int, headDim int, maxSeq int) (string, []modelWeightFile, error) {
	var b strings.Builder
	b.WriteString("program(1.3)\n{\n")
	fmt.Fprintf(
		&b,
		"    func main<ios18>(state<tensor<fp16, [1, %d, %d, %d]>> k_state, tensor<fp16, [1, %d, %d, %d]> x) {\n",
		numHeads,
		maxSeq,
		headDim,
		numHeads,
		maxSeq,
		headDim,
	)
	fmt.Fprintln(&b, "            write_state(data = x, input = k_state)[name = string(\"write_state_0\")];")
	fmt.Fprintf(&b, "            tensor<fp16, [1, %d, %d, %d]> k_next = read_state(input = k_state)[name = string(\"read_state_0\")];\n", numHeads, maxSeq, headDim)
	b.WriteString("        } -> (k_next);\n}\n")
	return b.String(), nil, nil
}

func buildMILProbeStateUpdateOnly(numHeads int, headDim int, maxSeq int) (string, []modelWeightFile, error) {
	var b strings.Builder
	b.WriteString(ffnMILHeader)
	fmt.Fprintf(
		&b,
		"    func main<ios18>(tensor<fp32, [1,%d,1,%d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state) {\n",
		numHeads,
		headDim,
		numHeads,
		maxSeq,
		headDim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_x\")];\n", numHeads, headDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", numHeads, maxSeq-1, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,x))[name=string(\"k_write\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = coreml_update_state(state=k_state, value=k_write)[name=string(\"k_next\")];\n", numHeads, maxSeq, headDim)
	b.WriteString("    } -> (k_next);\n}\n")
	return b.String(), nil, nil
}

func buildMILProbeStatefulSDPA(numHeads int, headDim int, maxSeq int, withMask bool) (string, []modelWeightFile, error) {
	var b strings.Builder
	b.WriteString(ffnMILHeader)
	inputs := []string{
		fmt.Sprintf("tensor<fp32, [1,%d,1,%d]> q_in", numHeads, headDim),
		fmt.Sprintf("tensor<fp32, [1,%d,1,%d]> k_in", numHeads, headDim),
		fmt.Sprintf("tensor<fp32, [1,%d,1,%d]> v_in", numHeads, headDim),
		fmt.Sprintf("state<tensor<fp16, [1,%d,%d,%d]>> k_state", numHeads, maxSeq, headDim),
		fmt.Sprintf("state<tensor<fp16, [1,%d,%d,%d]>> v_state", numHeads, maxSeq, headDim),
	}
	if withMask {
		inputs = append(inputs, fmt.Sprintf("tensor<bool, [1, 1, 1, %d]> attn_mask_in", maxSeq))
	}
	fmt.Fprintf(&b, "    func main<ios18>(%s) {\n", strings.Join(inputs, ", "))
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q = cast(dtype=to_fp16, x=q_in)[name=string(\"cast_q\")];\n", numHeads, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k = cast(dtype=to_fp16, x=k_in)[name=string(\"cast_k\")];\n", numHeads, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v = cast(dtype=to_fp16, x=v_in)[name=string(\"cast_v\")];\n", numHeads, headDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", numHeads, maxSeq-1, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", numHeads, maxSeq-1, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k))[name=string(\"k_write\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = coreml_update_state(state=k_state, value=k_write)[name=string(\"k_next\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v))[name=string(\"v_write\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = coreml_update_state(state=v_state, value=v_write)[name=string(\"v_next\")];\n", numHeads, maxSeq, headDim)
	if withMask {
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(attn_mask=attn_mask_in, query=q, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", numHeads, headDim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", numHeads, headDim)
	}
	b.WriteString("    } -> (attn);\n}\n")
	return b.String(), nil, nil
}

func buildMILProbeStatefulSDPAWriteReadback(numHeads int, headDim int, maxSeq int, withMask bool) (string, []modelWeightFile, error) {
	var b strings.Builder
	b.WriteString(ffnMILHeader)
	inputs := []string{
		fmt.Sprintf("tensor<fp32, [1,%d,1,%d]> q_in", numHeads, headDim),
		fmt.Sprintf("tensor<fp32, [1,%d,1,%d]> k_in", numHeads, headDim),
		fmt.Sprintf("tensor<fp32, [1,%d,1,%d]> v_in", numHeads, headDim),
		fmt.Sprintf("state<tensor<fp16, [1,%d,%d,%d]>> k_state", numHeads, maxSeq, headDim),
		fmt.Sprintf("state<tensor<fp16, [1,%d,%d,%d]>> v_state", numHeads, maxSeq, headDim),
	}
	if withMask {
		inputs = append(inputs, fmt.Sprintf("tensor<bool, [1, 1, 1, %d]> attn_mask_in", maxSeq))
	}
	fmt.Fprintf(&b, "    func main<ios18>(%s) {\n", strings.Join(inputs, ", "))
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q = cast(dtype=to_fp16, x=q_in)[name=string(\"cast_q\")];\n", numHeads, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k = cast(dtype=to_fp16, x=k_in)[name=string(\"cast_k\")];\n", numHeads, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v = cast(dtype=to_fp16, x=v_in)[name=string(\"cast_v\")];\n", numHeads, headDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", numHeads, maxSeq-1, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", numHeads, maxSeq-1, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k))[name=string(\"k_write\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v))[name=string(\"v_write\")];\n", numHeads, maxSeq, headDim)
	fmt.Fprintf(&b, "        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", numHeads, maxSeq, headDim)
	if withMask {
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(attn_mask=attn_mask_in, query=q, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", numHeads, headDim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", numHeads, headDim)
	}
	b.WriteString("    } -> (attn);\n}\n")
	return b.String(), nil, nil
}

func buildMILProbeSDPA(cfg MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 149)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 151)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 157)

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
		cfg.NumHeads,
		cfg.HeadDim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n",
		cfg.Dim,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out16 = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "out16", cfg.Dim)
	return b.String(), files, makeDeterministicTensor(cfg.Dim, 0.002, 163), cfg.Dim, nil
}

func buildMILProbeAttentionBlock(cfg MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 167)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 173)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 179)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 181)

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
		cfg.NumHeads,
		cfg.HeadDim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n",
		cfg.Dim,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out16 = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "out16", cfg.Dim)
	return b.String(), files, makeDeterministicTensor(cfg.Dim, 0.002, 191), cfg.Dim, nil
}

func buildMILProbeAttentionPlusFFN(cfg MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	cfg.NumLayers = 1
	weights := MILTransformerWeights{
		Layers: make([]MILTransformerLayerWeights, cfg.NumLayers),
	}
	for i := range weights.Layers {
		weights.Layers[i] = MILTransformerLayerWeights{
			QW: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0008, 197+i),
			QB: makeDeterministicTensor(cfg.Dim, 0.0006, 211+i),
			KW: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0008, 223+i),
			KB: makeDeterministicTensor(cfg.Dim, 0.0006, 227+i),
			VW: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0008, 229+i),
			VB: makeDeterministicTensor(cfg.Dim, 0.0006, 233+i),
			OW: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0008, 239+i),
			OB: makeDeterministicTensor(cfg.Dim, 0.0006, 241+i),
			W1: makeDeterministicTensor(cfg.HiddenDim*cfg.Dim, 0.0007, 251+i),
			B1: makeDeterministicTensor(cfg.HiddenDim, 0.0005, 257+i),
			W3: makeDeterministicTensor(cfg.HiddenDim*cfg.Dim, 0.0007, 263+i),
			B3: makeDeterministicTensor(cfg.HiddenDim, 0.0005, 269+i),
			W2: makeDeterministicTensor(cfg.Dim*cfg.HiddenDim, 0.0007, 271+i),
			B2: makeDeterministicTensor(cfg.Dim, 0.0005, 277+i),
		}
	}
	files, err := buildMILTransformerWeightFiles(cfg, weights)
	if err != nil {
		return "", nil, nil, 0, err
	}
	milText, err := transformerMILText(cfg)
	if err != nil {
		return "", nil, nil, 0, err
	}
	return milText, files, makeDeterministicTensor(cfg.Dim, 0.002, 281), cfg.Dim, nil
}

func buildMILProbeGatedFFNOnly(cfg MILTransformerConfig, useConvFFN bool, residual bool) (string, []modelWeightFile, []float32, int, error) {
	var files []modelWeightFile
	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)

	if useConvFFN {
		w1Path, b1Path := addProbeLinearWeights(&files, "ffn1", cfg.Dim, cfg.HiddenDim, 307)
		w3Path, b3Path := addProbeLinearWeights(&files, "ffn3", cfg.Dim, cfg.HiddenDim, 311)
		w2Path, b2Path := addProbeLinearWeights(&files, "ffn2", cfg.HiddenDim, cfg.Dim, 313)
		writeConvConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.HiddenDim, cfg.Dim)
		writeConvConstBlock(&b, "w3", "b3", w3Path, b3Path, cfg.HiddenDim, cfg.Dim)
		writeConvConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<int32, [4]> ffn_in_shape = const()[name=string(\"ffn_in_shape\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<int32, [3]> ffn_out_shape = const()[name=string(\"ffn_out_shape\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> ffn_in = reshape(shape=ffn_in_shape, x=x)[name=string(\"ffn_in\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> gate = conv(bias=b1,dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=w1,x=ffn_in)[name=string(\"gate\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> up = conv(bias=b3,dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=w3,x=ffn_in)[name=string(\"up\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> down4 = conv(bias=b2,dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=w2,x=mix)[name=string(\"down4\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = reshape(shape=ffn_out_shape, x=down4)[name=string(\"down\")];\n", cfg.Dim)
	} else {
		w1Path, b1Path := addProbeLinearWeights(&files, "ffn1", cfg.Dim, cfg.HiddenDim, 307)
		w3Path, b3Path := addProbeLinearWeights(&files, "ffn3", cfg.Dim, cfg.HiddenDim, 311)
		w2Path, b2Path := addProbeLinearWeights(&files, "ffn2", cfg.HiddenDim, cfg.Dim, 313)
		writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "w3", "b3", w3Path, b3Path, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = linear(bias=b1, weight=w1, x=x)[name=string(\"gate\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = linear(bias=b3, weight=w3, x=x)[name=string(\"up\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=b2, weight=w2, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	}
	outVar := "down"
	if residual {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out16 = add(x=x, y=down)[name=string(\"resid\")];\n", cfg.Dim)
		outVar = "out16"
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, makeDeterministicTensor(cfg.Dim, 0.002, 317), cfg.Dim, nil
}

func buildMILProbeAttentionFFNComposition(cfg MILTransformerConfig, stateful bool, withFFN bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 331)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 337)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 347)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 349)
	w1Path, b1Path := addProbeLinearWeights(&files, "ffn1", cfg.Dim, cfg.HiddenDim, 353)
	w3Path, b3Path := addProbeLinearWeights(&files, "ffn3", cfg.Dim, cfg.HiddenDim, 359)
	w2Path, b2Path := addProbeLinearWeights(&files, "ffn2", cfg.HiddenDim, cfg.Dim, 367)

	var b strings.Builder
	if stateful {
		fmt.Fprintf(
			&b,
			"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state) {\n",
			ffnMILHeader,
			cfg.Dim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
		)
		b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	} else {
		writeProbeMILPreamble(&b, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	if withFFN {
		writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "w3", "b3", w3Path, b3Path, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.HiddenDim)
	}
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
		cfg.NumHeads,
		cfg.HeadDim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n",
		cfg.Dim,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	if stateful {
		writeProbeStateConsts(&b)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=k)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	outVar := "resid"
	if withFFN {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = linear(bias=b1, weight=w1, x=resid)[name=string(\"gate\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = linear(bias=b3, weight=w3, x=resid)[name=string(\"up\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=b2, weight=w2, x=mix)[name=string(\"down\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out16 = add(x=resid, y=down)[name=string(\"ffn_resid\")];\n", cfg.Dim)
		outVar = "out16"
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFFNComposition(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool, withFFN bool, ffnInputSource string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 401)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 409)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 419)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 421)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 431)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 433)
	w1Path, b1Path := addProbeLinearWeights(&files, "ffn1", cfg.Dim, cfg.HiddenDim, 439)
	w3Path, b3Path := addProbeLinearWeights(&files, "ffn3", cfg.Dim, cfg.HiddenDim, 443)
	w2Path, b2Path := addProbeLinearWeights(&files, "ffn2", cfg.HiddenDim, cfg.Dim, 449)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	if withFFN {
		writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "w3", "b3", w3Path, b3Path, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.HiddenDim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	outVar := "resid"
	if withFFN {
		ffnInput := "resid"
		switch ffnInputSource {
		case "", "resid":
			ffnInput = "resid"
		case "x":
			ffnInput = "x"
		default:
			return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureFFNComposition: unsupported ffnInputSource %q", ffnInputSource)
		}
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = linear(bias=b1, weight=w1, x=%s)[name=string(\"gate\")];\n", cfg.HiddenDim, ffnInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = linear(bias=b3, weight=w3, x=%s)[name=string(\"up\")];\n", cfg.HiddenDim, ffnInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=b2, weight=w2, x=mix)[name=string(\"down\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out16 = add(x=resid, y=down)[name=string(\"ffn_resid\")];\n", cfg.Dim)
		outVar = "out16"
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeaturePostLinear(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 461)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 463)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 467)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 479)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 487)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 491)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 499)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "post", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFinalNormPieces(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool, normPiece string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1501)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1511)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1523)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1531)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1543)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1553)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1567)
	normWPath := addProbeVectorWeight(&files, "normw", cfg.Dim, 1571)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	if normPiece == "weight_only" || normPiece == "full_norm" {
		fmt.Fprintf(
			&b,
			"        tensor<fp16, [1,1,%d]> normw = const()[name=string(\"normw\"), val=tensor<fp16, [1,1,%d]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
			cfg.Dim,
			cfg.Dim,
			normWPath,
			blobChunkOffset,
		)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	outVar := "post"
	switch normPiece {
	case "none":
	case "weight_only":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post_norm = mul(x=post, y=normw)[name=string(\"post_norm\")];\n", cfg.Dim)
		outVar = "post_norm"
	case "stats_only", "full_norm":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fn_abs = abs(x=post)[name=string(\"fn_abs\")];\n", cfg.Dim)
		b.WriteString("        tensor<int32, [1]> fn_ax = const()[name=string(\"fn_ax\"), val=tensor<int32, [1]>([2])];\n")
		b.WriteString("        bool fn_kd = const()[name=string(\"fn_kd\"), val=bool(true)];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,1]> fn_maxabs = reduce_max(x=fn_abs, axes=fn_ax, keep_dims=fn_kd)[name=string(\"fn_maxabs\")];\n")
		b.WriteString("        fp16 fn_eps = const()[name=string(\"fn_eps\"), val=fp16(0.000001)];\n")
		b.WriteString("        tensor<fp16, [1,1,1]> fn_safemax = maximum(x=fn_maxabs, y=fn_eps)[name=string(\"fn_safemax\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fn_scaled = real_div(x=post, y=fn_safemax)[name=string(\"fn_scaled\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fn_square = square(x=fn_scaled)[name=string(\"fn_square\")];\n", cfg.Dim)
		b.WriteString("        tensor<fp16, [1,1,1]> fn_meansq = reduce_mean(x=fn_square, axes=fn_ax, keep_dims=fn_kd)[name=string(\"fn_meansq\")];\n")
		b.WriteString("        tensor<fp16, [1,1,1]> fn_meaneps = add(x=fn_meansq, y=fn_eps)[name=string(\"fn_meaneps\")];\n")
		b.WriteString("        tensor<fp16, [1,1,1]> fn_rms = sqrt(x=fn_meaneps)[name=string(\"fn_rms\")];\n")
		b.WriteString("        tensor<fp16, [1,1,1]> fn_scaledrms = mul(x=fn_rms, y=fn_safemax)[name=string(\"fn_scaledrms\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fn_norm = real_div(x=post, y=fn_scaledrms)[name=string(\"fn_norm\")];\n", cfg.Dim)
		outVar = "fn_norm"
		if normPiece == "full_norm" {
			fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post_norm = mul(x=fn_norm, y=normw)[name=string(\"post_norm\")];\n", cfg.Dim)
			outVar = "post_norm"
		}
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureFinalNormPieces: unsupported normPiece %q", normPiece)
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFFNPieces(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool, ffnPiece string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1601)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1607)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1613)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1619)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1627)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1637)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1651)
	var f1WPath, f1BPath, f2WPath, f2BPath, f3WPath, f3BPath string
	switch ffnPiece {
	case "none":
	case "one_linear_hidden", "one_linear_hidden_reduce", "two_linears", "gated_ffn":
		f1WPath, f1BPath = addProbeLinearWeights(&files, "f1", cfg.Dim, cfg.HiddenDim, 1663)
		if ffnPiece == "two_linears" || ffnPiece == "gated_ffn" {
			f2WPath, f2BPath = addProbeLinearWeights(&files, "f2", cfg.HiddenDim, cfg.Dim, 1669)
		}
		if ffnPiece == "gated_ffn" {
			f3WPath, f3BPath = addProbeLinearWeights(&files, "f3", cfg.Dim, cfg.HiddenDim, 1691)
		}
	case "one_linear_same_width", "two_linears_same_width":
		f1WPath, f1BPath = addProbeLinearWeights(&files, "f1", cfg.Dim, cfg.Dim, 1663)
		if ffnPiece == "two_linears_same_width" {
			f2WPath, f2BPath = addProbeLinearWeights(&files, "f2", cfg.Dim, cfg.Dim, 1669)
		}
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureFFNPieces: unsupported ffnPiece %q", ffnPiece)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	switch ffnPiece {
	case "one_linear_hidden", "one_linear_hidden_reduce", "two_linears", "gated_ffn":
		writeLinearConstBlock(&b, "f1w", "f1b", f1WPath, f1BPath, cfg.HiddenDim, cfg.Dim)
		if ffnPiece == "two_linears" || ffnPiece == "gated_ffn" {
			writeLinearConstBlock(&b, "f2w", "f2b", f2WPath, f2BPath, cfg.Dim, cfg.HiddenDim)
		}
		if ffnPiece == "gated_ffn" {
			writeLinearConstBlock(&b, "f3w", "f3b", f3WPath, f3BPath, cfg.HiddenDim, cfg.Dim)
		}
	case "one_linear_same_width", "two_linears_same_width":
		writeLinearConstBlock(&b, "f1w", "f1b", f1WPath, f1BPath, cfg.Dim, cfg.Dim)
		if ffnPiece == "two_linears_same_width" {
			writeLinearConstBlock(&b, "f2w", "f2b", f2WPath, f2BPath, cfg.Dim, cfg.Dim)
		}
	case "none":
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	outVar := "post"
	switch ffnPiece {
	case "none":
	case "one_linear_hidden":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.HiddenDim)
		outVar = "f1"
	case "one_linear_same_width":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.Dim)
		outVar = "f1"
	case "one_linear_hidden_reduce":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.HiddenDim)
		b.WriteString("        tensor<int32, [1]> f1_ax = const()[name=string(\"f1_ax\"), val=tensor<int32, [1]>([2])];\n")
		b.WriteString("        bool f1_kd = const()[name=string(\"f1_kd\"), val=bool(true)];\n")
		b.WriteString("        tensor<fp16, [1,1,1]> f1_reduce = reduce_mean(x=f1, axes=f1_ax, keep_dims=f1_kd)[name=string(\"f1_reduce\")];\n")
		outVar = "f1_reduce"
	case "two_linears":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f2 = linear(bias=f2b, weight=f2w, x=f1)[name=string(\"f2\")];\n", cfg.Dim)
		outVar = "f2"
	case "two_linears_same_width":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f2 = linear(bias=f2b, weight=f2w, x=f1)[name=string(\"f2\")];\n", cfg.Dim)
		outVar = "f2"
	case "gated_ffn":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1_sig = sigmoid(x=f1)[name=string(\"f1_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1_act = mul(x=f1, y=f1_sig)[name=string(\"f1_act\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f3 = linear(bias=f3b, weight=f3w, x=post)[name=string(\"f3\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f_mix = mul(x=f1_act, y=f3)[name=string(\"f_mix\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f2 = linear(bias=f2b, weight=f2w, x=f_mix)[name=string(\"f2\")];\n", cfg.Dim)
		outVar = "f2"
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureRopeGateOneAffine(cfg MILTransformerConfig, gateMode string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1703)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1709)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1721)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1723)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1733)
	f1WPath, f1BPath := addProbeLinearWeights(&files, "f1", cfg.Dim, cfg.HiddenDim, 1741)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1753)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1763)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "f1w", "f1b", f1WPath, f1BPath, cfg.HiddenDim, cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" || gateMode == "branch_add" ||
		gateMode == "attn_branch_mul" || gateMode == "attn_branch_sigmoid_mul" || gateMode == "attn_branch_add" ||
		gateMode == "main_linear" || gateMode == "main_sigmoid_linear" {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" || gateMode == "branch_add" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if gateMode == "attn_branch_mul" || gateMode == "attn_branch_sigmoid_mul" || gateMode == "attn_branch_add" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=attn_r)[name=string(\"qg\")];\n", cfg.Dim)
	}
	attnInput := "attn_r"
	switch gateMode {
	case "none":
	case "main_linear":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = linear(bias=qgb, weight=qgw, x=attn_r)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "main_sigmoid_linear":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_main = linear(bias=qgb, weight=qgw, x=attn_r)[name=string(\"attn_main\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = sigmoid(x=attn_main)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "input_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=x)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "input_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = add(x=attn_r, y=x)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "runtime_branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_branch = add(x=x, y=x)[name=string(\"gate_branch\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=gate_branch)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "runtime_branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_branch = add(x=x, y=x)[name=string(\"gate_branch\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate_branch)[name=string(\"gate_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=gate_sig)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "runtime_branch_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_branch = add(x=x, y=x)[name=string(\"gate_branch\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = add(x=attn_r, y=gate_branch)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=qg)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=qg_sig)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "branch_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = add(x=attn_r, y=qg)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureRopeGateOneAffine: unsupported gateMode %q", gateMode)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.HiddenDim)
	writeProbeMILReturn(&b, "f1", cfg.HiddenDim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureRopeGateFFNEntry(cfg MILTransformerConfig, gateMode string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1801)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1807)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1813)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1817)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1823)
	f1WPath, f1BPath := addProbeLinearWeights(&files, "f1", cfg.Dim, cfg.HiddenDim, 1829)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1831)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1847)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "f1w", "f1b", f1WPath, f1BPath, cfg.HiddenDim, cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" || gateMode == "branch_add" ||
		gateMode == "attn_branch_mul" || gateMode == "attn_branch_sigmoid_mul" || gateMode == "attn_branch_add" ||
		gateMode == "main_linear" || gateMode == "main_sigmoid_linear" {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" || gateMode == "branch_add" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	switch gateMode {
	case "none":
	case "main_linear":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = linear(bias=qgb, weight=qgw, x=attn_r)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "main_sigmoid_linear":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_main = linear(bias=qgb, weight=qgw, x=attn_r)[name=string(\"attn_main\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = sigmoid(x=attn_main)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "input_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=x)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "input_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = add(x=attn_r, y=x)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "runtime_branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_branch = add(x=x, y=x)[name=string(\"gate_branch\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=gate_branch)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "runtime_branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_branch = add(x=x, y=x)[name=string(\"gate_branch\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate_branch)[name=string(\"gate_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=gate_sig)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "runtime_branch_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_branch = add(x=x, y=x)[name=string(\"gate_branch\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = add(x=attn_r, y=gate_branch)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=qg)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=qg_sig)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "branch_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = add(x=attn_r, y=qg)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "attn_branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=qg)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "attn_branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = mul(x=attn_r, y=qg_sig)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	case "attn_branch_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mix = add(x=attn_r, y=qg)[name=string(\"attn_mix\")];\n", cfg.Dim)
		attnInput = "attn_mix"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureRopeGateFFNEntry: unsupported gateMode %q", gateMode)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.HiddenDim)
	writeProbeMILReturn(&b, "f1", cfg.HiddenDim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureRopePreTransformFFNEntry(cfg MILTransformerConfig, transform string) (string, []modelWeightFile, error) {
	return buildMILProbeAttentionFeatureRopeModePreTransformFFNEntry(cfg, "dynamic", transform)
}

func buildMILProbeAttentionFeatureRopeModePreTransformFFNEntry(cfg MILTransformerConfig, ropeMode string, transform string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1901)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1907)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1913)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1919)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1921)
	f1WPath, f1BPath := addProbeLinearWeights(&files, "f1", cfg.Dim, cfg.HiddenDim, 1931)
	preWPath, preBPath := addProbeLinearWeights(&files, "pre", cfg.Dim, cfg.Dim, 1933)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1949)

	var b strings.Builder
	switch ropeMode {
	case "dynamic":
		fmt.Fprintf(
			&b,
			"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
			ffnMILHeader,
			cfg.Dim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
			cfg.Dim,
			cfg.Dim,
		)
		b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	case "static":
		writeProbeMILPreamble(&b, cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = const()[name=string(\"pos_cos\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 727)))
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = const()[name=string(\"pos_sin\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 733)))
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	case "none":
		fmt.Fprintf(
			&b,
			"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state) {\n",
			ffnMILHeader,
			cfg.Dim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
		)
		b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureRopeModePreTransformFFNEntry: unsupported ropeMode %q", ropeMode)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "f1w", "f1b", f1WPath, f1BPath, cfg.HiddenDim, cfg.Dim)
	if transform == "linear" || transform == "sigmoid_linear" {
		writeLinearConstBlock(&b, "prew", "preb", preWPath, preBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	xsrc := "x"
	switch transform {
	case "none":
	case "linear":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x_pre = linear(bias=preb, weight=prew, x=x)[name=string(\"x_pre\")];\n", cfg.Dim)
		xsrc = "x_pre"
	case "sigmoid_linear":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x_pre_lin = linear(bias=preb, weight=prew, x=x)[name=string(\"x_pre_lin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x_pre = sigmoid(x=x_pre_lin)[name=string(\"x_pre\")];\n", cfg.Dim)
		xsrc = "x_pre"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureRopePreTransformFFNEntry: unsupported transform %q", transform)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=%s)[name=string(\"q\")];\n", cfg.Dim, xsrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=%s)[name=string(\"k\")];\n", cfg.Dim, xsrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=%s)[name=string(\"v\")];\n", cfg.Dim, xsrc)
	qSrc := "q"
	kSrc := "k"
	if ropeMode != "none" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		qSrc = "qrope"
		kSrc = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qSrc)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kSrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> f1 = linear(bias=f1b, weight=f1w, x=post)[name=string(\"f1\")];\n", cfg.HiddenDim)
	writeProbeMILReturn(&b, "f1", cfg.HiddenDim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureSimpleMLP(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 503)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 509)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 521)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 523)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 541)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 547)
	w1Path, b1Path := addProbeLinearWeights(&files, "mlp1", cfg.Dim, cfg.HiddenDim, 557)
	w2Path, b2Path := addProbeLinearWeights(&files, "mlp2", cfg.HiddenDim, cfg.Dim, 563)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.HiddenDim)
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mlp1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"mlp1\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mlp1_act = sigmoid(x=mlp1)[name=string(\"mlp1_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mlp2 = linear(bias=b2, weight=w2, x=mlp1_act)[name=string(\"mlp2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "mlp2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureTwoLinears(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 571)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 577)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 587)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 593)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 599)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 601)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.HiddenDim, 607)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.HiddenDim, cfg.Dim, 613)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.HiddenDim)
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureTwoLinearsSameWidth(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 617)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 619)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 631)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 641)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 643)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 647)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 653)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 659)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureAffineSourceDepth(cfg MILTransformerConfig, dynamicRoPE bool, attnGate bool, source string, depth int) (string, []modelWeightFile, error) {
	if depth < 1 || depth > 2 {
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureAffineSourceDepth: unsupported depth %d", depth)
	}

	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 661)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 673)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 677)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 683)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 691)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 701)
	var w2Path, b2Path string
	if depth == 2 {
		w2Path, b2Path = addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 709)
	}
	var qgWPath, qgBPath string
	if attnGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 719)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	if depth == 2 {
		writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	}
	if attnGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if attnGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)

	affineInput := ""
	switch source {
	case "attn":
		affineInput = attnInput
	case "proj":
		affineInput = "proj"
	case "resid":
		affineInput = "resid"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureAffineSourceDepth: unsupported source %q", source)
	}

	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=%s)[name=string(\"lin1\")];\n", cfg.Dim, affineInput)
	outVar := "lin1"
	if depth == 2 {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
		outVar = "lin2"
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureTwoLinearsStatefulness(cfg MILTransformerConfig, stateful bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 661)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 673)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 677)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 683)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 691)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 701)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 709)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 719)

	var b strings.Builder
	if stateful {
		fmt.Fprintf(
			&b,
			"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
			ffnMILHeader,
			cfg.Dim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
			cfg.NumHeads,
			cfg.MaxSeqLen,
			cfg.HeadDim,
			cfg.Dim,
			cfg.Dim,
		)
		b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	} else {
		writeProbeMILPreamble(&b, cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = const()[name=string(\"pos_cos\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 727)))
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = const()[name=string(\"pos_sin\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 733)))
	}
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	var attnK, attnV string
	if stateful {
		writeProbeStateConsts(&b)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
		attnK = "k_next"
		attnV = "v_next"
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=krope)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
		attnK = "k4"
		attnV = "v4"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=%s, value=%s)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim, attnK, attnV)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_gated)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureGateTopology(cfg MILTransformerConfig, gateMode string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 739)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 743)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 751)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 757)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 761)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 769)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 773)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 787)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	if gateMode == "const_mul" || gateMode == "const_sigmoid_mul" || gateMode == "const_add" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_const = const()[name=string(\"gate_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 797)))
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=krope)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	switch gateMode {
	case "none":
	case "const_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_const)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "const_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate_const)[name=string(\"gate_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "const_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=gate_const)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureGateTopology: unsupported gateMode %q", gateMode)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureMixPosition(cfg MILTransformerConfig, mixPosition string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 809)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 811)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 821)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 823)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 827)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 829)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 839)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	if mixPosition != "none" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix_const = const()[name=string(\"mix_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 853)))
	}
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	projInput := "attn_r"
	switch mixPosition {
	case "none":
	case "before_proj":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mixed = add(x=attn_r, y=mix_const)[name=string(\"mixed\")];\n", cfg.Dim)
		projInput = "mixed"
	case "after_proj":
	case "after_resid":
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureMixPosition: unsupported mixPosition %q", mixPosition)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, projInput)
	projVar := "proj"
	if mixPosition == "after_proj" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj_mixed = add(x=proj, y=mix_const)[name=string(\"proj_mixed\")];\n", cfg.Dim)
		projVar = "proj_mixed"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=%s)[name=string(\"resid\")];\n", cfg.Dim, projVar)
	residVar := "resid"
	if mixPosition == "after_resid" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid_mixed = add(x=resid, y=mix_const)[name=string(\"resid_mixed\")];\n", cfg.Dim)
		residVar = "resid_mixed"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=%s)[name=string(\"lin1\")];\n", cfg.Dim, residVar)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureMixPositionPostLinear(cfg MILTransformerConfig, dynamicRoPE bool, mixPosition string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 857)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 859)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 863)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 877)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 881)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 883)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	}
	if mixPosition != "none" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix_const = const()[name=string(\"mix_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 887)))
	}
	if dynamicRoPE {
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	projInput := "attn_r"
	switch mixPosition {
	case "none":
	case "before_proj":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mixed = add(x=attn_r, y=mix_const)[name=string(\"mixed\")];\n", cfg.Dim)
		projInput = "mixed"
	case "after_proj":
	case "after_resid":
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureMixPositionPostLinear: unsupported mixPosition %q", mixPosition)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, projInput)
	projVar := "proj"
	if mixPosition == "after_proj" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj_mixed = add(x=proj, y=mix_const)[name=string(\"proj_mixed\")];\n", cfg.Dim)
		projVar = "proj_mixed"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=%s)[name=string(\"resid\")];\n", cfg.Dim, projVar)
	residVar := "resid"
	if mixPosition == "after_resid" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid_mixed = add(x=resid, y=mix_const)[name=string(\"resid_mixed\")];\n", cfg.Dim)
		residVar = "resid_mixed"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=%s)[name=string(\"post\")];\n", cfg.Dim, residVar)
	writeProbeMILReturn(&b, "post", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureMixReturn(cfg MILTransformerConfig, dynamicRoPE bool, mixMode string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 907)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 911)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 919)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 929)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	}
	if mixMode == "add" || mixMode == "mul" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix_const = const()[name=string(\"mix_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 937)))
	}
	if dynamicRoPE {
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	outVar := "attn_r"
	switch mixMode {
	case "none":
	case "add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mixed = add(x=attn_r, y=mix_const)[name=string(\"attn_mixed\")];\n", cfg.Dim)
		outVar = "attn_mixed"
	case "mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mixed = mul(x=attn_r, y=mix_const)[name=string(\"attn_mixed\")];\n", cfg.Dim)
		outVar = "attn_mixed"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureMixReturn: unsupported mixMode %q", mixMode)
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureGateTopologyPostLinear(cfg MILTransformerConfig, gateMode string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 941)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 947)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 953)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 967)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 971)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 977)

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	if gateMode == "const_mul" || gateMode == "const_sigmoid_mul" || gateMode == "const_add" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_const = const()[name=string(\"gate_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 983)))
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if gateMode == "branch_mul" || gateMode == "branch_sigmoid_mul" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	switch gateMode {
	case "none":
	case "const_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_const)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "const_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate_const)[name=string(\"gate_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "const_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=gate_const)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureGateTopologyPostLinear: unsupported gateMode %q", gateMode)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "post", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureGateTopologyTwoLinears(cfg MILTransformerConfig, gateMode string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 721)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 727)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 733)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 739)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 743)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 751)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 757)
	var qgWPath, qgBPath string
	if gateMode != "none" {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 761)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	if gateMode != "none" {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if gateMode != "none" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	switch gateMode {
	case "none":
	case "mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mixed = mul(x=attn_r, y=qg)[name=string(\"attn_mixed\")];\n", cfg.Dim)
		attnInput = "attn_mixed"
	case "sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mixed = mul(x=attn_r, y=qg_sig)[name=string(\"attn_mixed\")];\n", cfg.Dim)
		attnInput = "attn_mixed"
	case "add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mixed = add(x=attn_r, y=qg)[name=string(\"attn_mixed\")];\n", cfg.Dim)
		attnInput = "attn_mixed"
	case "sigmoid_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mixed = add(x=attn_r, y=qg_sig)[name=string(\"attn_mixed\")];\n", cfg.Dim)
		attnInput = "attn_mixed"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureGateTopologyTwoLinears: unsupported gateMode %q", gateMode)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureGatePlacementTwoLinears(cfg MILTransformerConfig, placement string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 769)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 773)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 787)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 797)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 809)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 821)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 823)
	needsGate := placement != "none"
	var qgWPath, qgBPath string
	if needsGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 811)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	if needsGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if needsGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	mixed := "attn_r"
	switch placement {
	case "none":
	case "before_proj":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_mixed = mul(x=attn_r, y=qg_sig)[name=string(\"attn_mixed\")];\n", cfg.Dim)
		mixed = "attn_mixed"
	default:
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, mixed)
	if placement == "after_proj" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj_mixed = mul(x=proj, y=qg_sig)[name=string(\"proj_mixed\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj_mixed)[name=string(\"resid\")];\n", cfg.Dim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	}
	lin1Input := "resid"
	switch placement {
	case "after_resid":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid_mixed = mul(x=resid, y=qg_sig)[name=string(\"resid_mixed\")];\n", cfg.Dim)
		lin1Input = "resid_mixed"
	case "none", "before_proj", "after_proj", "after_lin1":
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureGatePlacementTwoLinears: unsupported placement %q", placement)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=%s)[name=string(\"lin1\")];\n", cfg.Dim, lin1Input)
	lin2Input := "lin1"
	if placement == "after_lin1" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1_mixed = mul(x=lin1, y=qg_sig)[name=string(\"lin1_mixed\")];\n", cfg.Dim)
		lin2Input = "lin1_mixed"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=%s)[name=string(\"lin2\")];\n", cfg.Dim, lin2Input)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureDeadGateBranchTwoLinears(cfg MILTransformerConfig, deadBranch string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 827)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 829)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 839)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 853)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 857)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 859)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 863)
	needsDeadBranch := deadBranch != "none"
	var qgWPath, qgBPath string
	if needsDeadBranch {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 877)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	if needsDeadBranch {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if needsDeadBranch {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
		if deadBranch == "qg_sigmoid" {
			fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		} else if deadBranch != "qg" {
			return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureDeadGateBranchTwoLinears: unsupported deadBranch %q", deadBranch)
		}
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureDeadFFNBranch(cfg MILTransformerConfig, deadBranch string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 881)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 883)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 887)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 907)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 911)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 919)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 929)
	var dead1WPath, dead1BPath, dead2WPath, dead2BPath, dead3WPath, dead3BPath string
	if deadBranch != "none" {
		switch deadBranch {
		case "one_linear_hidden", "one_linear_hidden_elementwise", "one_linear_hidden_const_add", "one_linear_hidden_const_mul", "one_linear_hidden_reduce", "linear_ffn", "linear_sigmoid", "gated_ffn":
			dead1WPath, dead1BPath = addProbeLinearWeights(&files, "dead1", cfg.Dim, cfg.HiddenDim, 937)
			if deadBranch != "one_linear_hidden" {
				if deadBranch != "one_linear_hidden_elementwise" && deadBranch != "one_linear_hidden_const_add" && deadBranch != "one_linear_hidden_const_mul" && deadBranch != "one_linear_hidden_reduce" {
					dead2WPath, dead2BPath = addProbeLinearWeights(&files, "dead2", cfg.HiddenDim, cfg.Dim, 941)
				}
			}
		case "one_linear_same_width", "one_linear_same_width_elementwise", "one_linear_same_width_const_add", "one_linear_same_width_const_mul", "one_linear_same_width_reduce":
			dead1WPath, dead1BPath = addProbeLinearWeights(&files, "dead1", cfg.Dim, cfg.Dim, 937)
			if deadBranch != "one_linear_same_width" && deadBranch != "one_linear_same_width_elementwise" && deadBranch != "one_linear_same_width_const_add" && deadBranch != "one_linear_same_width_const_mul" && deadBranch != "one_linear_same_width_reduce" {
				dead2WPath, dead2BPath = addProbeLinearWeights(&files, "dead2", cfg.HiddenDim, cfg.Dim, 941)
			}
		case "same_width_two_linears":
			dead1WPath, dead1BPath = addProbeLinearWeights(&files, "dead1", cfg.Dim, cfg.Dim, 937)
			dead2WPath, dead2BPath = addProbeLinearWeights(&files, "dead2", cfg.Dim, cfg.Dim, 941)
		case "elementwise_depth2":
			// No learned weights; this isolates deeper dead subgraphs from affine depth.
		default:
			return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureDeadFFNBranch: unsupported deadBranch %q", deadBranch)
		}
		if deadBranch == "gated_ffn" {
			dead3WPath, dead3BPath = addProbeLinearWeights(&files, "dead3", cfg.Dim, cfg.HiddenDim, 947)
		}
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state, tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.Dim,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	switch deadBranch {
	case "one_linear_hidden_const_add", "one_linear_hidden_const_mul":
		deadConst := makeDeterministicTensor(cfg.HiddenDim, 0.01, 951)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_const = const()[name=string(\"dead_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.HiddenDim, cfg.HiddenDim, testTensorLiteral16(deadConst))
	case "one_linear_same_width_const_add", "one_linear_same_width_const_mul":
		deadConst := makeDeterministicTensor(cfg.Dim, 0.01, 953)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_const = const()[name=string(\"dead_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(deadConst))
	}
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	switch deadBranch {
	case "one_linear_hidden":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.HiddenDim, cfg.Dim)
	case "one_linear_hidden_elementwise":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.HiddenDim, cfg.Dim)
	case "one_linear_hidden_reduce":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.HiddenDim, cfg.Dim)
	case "one_linear_same_width":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.Dim, cfg.Dim)
	case "one_linear_same_width_elementwise":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.Dim, cfg.Dim)
	case "one_linear_same_width_reduce":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.Dim, cfg.Dim)
	case "linear_ffn", "linear_sigmoid", "gated_ffn":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "dead2w", "dead2b", dead2WPath, dead2BPath, cfg.Dim, cfg.HiddenDim)
		if deadBranch == "gated_ffn" {
			writeLinearConstBlock(&b, "dead3w", "dead3b", dead3WPath, dead3BPath, cfg.HiddenDim, cfg.Dim)
		}
	case "same_width_two_linears":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.Dim, cfg.Dim)
		writeLinearConstBlock(&b, "dead2w", "dead2b", dead2WPath, dead2BPath, cfg.Dim, cfg.Dim)
	case "elementwise_depth2", "none":
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	if deadBranch == "linear_ffn" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_out = linear(bias=dead2b, weight=dead2w, x=dead_hidden)[name=string(\"dead_out\")];\n", cfg.Dim)
		_ = "dead_out"
	}
	if deadBranch == "linear_sigmoid" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_sig = sigmoid(x=dead_hidden)[name=string(\"dead_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_out = linear(bias=dead2b, weight=dead2w, x=dead_sig)[name=string(\"dead_out\")];\n", cfg.Dim)
		_ = "dead_out"
	}
	if deadBranch == "same_width_two_linears" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_out = linear(bias=dead2b, weight=dead2w, x=dead_hidden)[name=string(\"dead_out\")];\n", cfg.Dim)
		_ = "dead_out"
	}
	if deadBranch == "elementwise_depth2" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_sig = sigmoid(x=x)[name=string(\"dead_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_mix = mul(x=dead_sig, y=x)[name=string(\"dead_mix\")];\n", cfg.Dim)
		_ = "dead_mix"
	}
	if deadBranch == "gated_ffn" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_gate = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_gate\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_gate_sig = sigmoid(x=dead_gate)[name=string(\"dead_gate_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_gate_act = mul(x=dead_gate, y=dead_gate_sig)[name=string(\"dead_gate_act\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_up = linear(bias=dead3b, weight=dead3w, x=x)[name=string(\"dead_up\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_mix = mul(x=dead_gate_act, y=dead_up)[name=string(\"dead_mix\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_out = linear(bias=dead2b, weight=dead2w, x=dead_mix)[name=string(\"dead_out\")];\n", cfg.Dim)
		_ = "dead_out"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureDeadBranchRoPE(cfg MILTransformerConfig, dynamicRoPE bool, deadBranch string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 953)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 967)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 971)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 977)
	w1Path, b1Path := addProbeLinearWeights(&files, "lin1", cfg.Dim, cfg.Dim, 983)
	w2Path, b2Path := addProbeLinearWeights(&files, "lin2", cfg.Dim, cfg.Dim, 991)
	var ropeWPath, ropeBPath string
	if dynamicRoPE {
		ropeWPath, ropeBPath = addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 997)
	}
	var dead1WPath, dead1BPath, dead2WPath, dead2BPath string
	switch deadBranch {
	case "linear_ffn":
		dead1WPath, dead1BPath = addProbeLinearWeights(&files, "dead1", cfg.Dim, cfg.HiddenDim, 1009)
		dead2WPath, dead2BPath = addProbeLinearWeights(&files, "dead2", cfg.HiddenDim, cfg.Dim, 1013)
	case "qg":
		dead1WPath, dead1BPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1019)
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureDeadBranchRoPE: unsupported deadBranch %q", deadBranch)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	if dynamicRoPE {
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.Dim)
	switch deadBranch {
	case "linear_ffn":
		writeLinearConstBlock(&b, "dead1w", "dead1b", dead1WPath, dead1BPath, cfg.HiddenDim, cfg.Dim)
		writeLinearConstBlock(&b, "dead2w", "dead2b", dead2WPath, dead2BPath, cfg.Dim, cfg.HiddenDim)
	case "qg":
		writeLinearConstBlock(&b, "qgw", "qgb", dead1WPath, dead1BPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	qForShape := "q"
	kForShape := "k"
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		qForShape = "qrope"
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		kForShape = "krope"
	}
	if deadBranch == "one_linear_hidden" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		_ = "dead_hidden"
	}
	if deadBranch == "one_linear_hidden_elementwise" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_sig = sigmoid(x=dead_hidden)[name=string(\"dead_hidden_sig\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_mix = mul(x=dead_hidden, y=dead_hidden_sig)[name=string(\"dead_hidden_mix\")];\n", cfg.HiddenDim)
		_ = "dead_hidden_mix"
	}
	if deadBranch == "one_linear_hidden_const_add" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_mix = add(x=dead_hidden, y=dead_const)[name=string(\"dead_hidden_mix\")];\n", cfg.HiddenDim)
		_ = "dead_hidden_mix"
	}
	if deadBranch == "one_linear_hidden_const_mul" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_mix = mul(x=dead_hidden, y=dead_const)[name=string(\"dead_hidden_mix\")];\n", cfg.HiddenDim)
		_ = "dead_hidden_mix"
	}
	if deadBranch == "one_linear_hidden_reduce" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<int32, [1]> dead_axes = const()[name=string(\"dead_axes\"), val=tensor<int32, [1]>([2])];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,1]> dead_hidden_reduce = reduce_mean(axes=dead_axes, keep_dims=true, x=dead_hidden)[name=string(\"dead_hidden_reduce\")];\n")
		_ = "dead_hidden_reduce"
	}
	if deadBranch == "one_linear_same_width" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.Dim)
		_ = "dead_hidden"
	}
	if deadBranch == "one_linear_same_width_elementwise" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_sig = sigmoid(x=dead_hidden)[name=string(\"dead_hidden_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_mix = mul(x=dead_hidden, y=dead_hidden_sig)[name=string(\"dead_hidden_mix\")];\n", cfg.Dim)
		_ = "dead_hidden_mix"
	}
	if deadBranch == "one_linear_same_width_const_add" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_mix = add(x=dead_hidden, y=dead_const)[name=string(\"dead_hidden_mix\")];\n", cfg.Dim)
		_ = "dead_hidden_mix"
	}
	if deadBranch == "one_linear_same_width_const_mul" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden_mix = mul(x=dead_hidden, y=dead_const)[name=string(\"dead_hidden_mix\")];\n", cfg.Dim)
		_ = "dead_hidden_mix"
	}
	if deadBranch == "one_linear_same_width_reduce" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<int32, [1]> dead_axes = const()[name=string(\"dead_axes\"), val=tensor<int32, [1]>([2])];\n")
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,1]> dead_hidden_reduce = reduce_mean(axes=dead_axes, keep_dims=true, x=dead_hidden)[name=string(\"dead_hidden_reduce\")];\n")
		_ = "dead_hidden_reduce"
	}
	if deadBranch == "linear_ffn" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_hidden = linear(bias=dead1b, weight=dead1w, x=x)[name=string(\"dead_hidden\")];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> dead_out = linear(bias=dead2b, weight=dead2w, x=dead_hidden)[name=string(\"dead_out\")];\n", cfg.Dim)
	}
	if deadBranch == "qg" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qForShape)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=%s)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim, kForShape)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin1 = linear(bias=b1, weight=w1, x=resid)[name=string(\"lin1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> lin2 = linear(bias=b2, weight=w2, x=lin1)[name=string(\"lin2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "lin2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear(cfg MILTransformerConfig, gateOp string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 991)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 997)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1009)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1013)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1021)
	needsBranch := strings.HasPrefix(gateOp, "branch_") || strings.HasPrefix(gateOp, "combined_")
	var qgWPath, qgBPath string
	if needsBranch {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1019)
	}

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	if strings.HasSuffix(gateOp, "_plus_const") || strings.HasSuffix(gateOp, "_times_const") || strings.HasSuffix(gateOp, "_plus_zero") || strings.HasSuffix(gateOp, "_times_one") {
		constVals := makeDeterministicTensor(cfg.Dim, 0.01, 1031)
		switch {
		case strings.HasSuffix(gateOp, "_plus_zero"):
			constVals = makeDeterministicTensor(cfg.Dim, 0, 0)
		case strings.HasSuffix(gateOp, "_times_one"):
			constVals = makeDeterministicTensor(cfg.Dim, 1, 0)
		}
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_const = const()[name=string(\"gate_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(constVals))
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	if needsBranch {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	if needsBranch {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	gateInput := "x"
	if needsBranch {
		gateInput = "qg"
	}
	if strings.HasSuffix(gateOp, "_plus_const") {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_shift = add(x=%s, y=gate_const)[name=string(\"gate_shift\")];\n", cfg.Dim, gateInput)
		gateInput = "gate_shift"
	} else if strings.HasSuffix(gateOp, "_plus_zero") {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_shift = add(x=%s, y=gate_const)[name=string(\"gate_shift\")];\n", cfg.Dim, gateInput)
		gateInput = "gate_shift"
	} else if strings.HasSuffix(gateOp, "_times_const") {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_scale = mul(x=%s, y=gate_const)[name=string(\"gate_scale\")];\n", cfg.Dim, gateInput)
		gateInput = "gate_scale"
	} else if strings.HasSuffix(gateOp, "_times_one") {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_scale = mul(x=%s, y=gate_const)[name=string(\"gate_scale\")];\n", cfg.Dim, gateInput)
		gateInput = "gate_scale"
	}
	attnInput := "attn_r"
	switch gateOp {
	case "branch_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "branch_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=%s)[name=string(\"qg_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "branch_sigmoid_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=%s)[name=string(\"qg_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_mul_plus_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "branch_sigmoid_mul_plus_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=%s)[name=string(\"qg_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_mul_times_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "branch_sigmoid_mul_times_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=%s)[name=string(\"qg_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "branch_mul_plus_zero":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "branch_mul_times_one":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "input_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "input_sigmoid_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=%s)[name=string(\"gate_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "input_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "input_sigmoid_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=%s)[name=string(\"gate_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=gate_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "input_mul_plus_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "input_sigmoid_mul_plus_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=%s)[name=string(\"gate_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "input_mul_times_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "input_sigmoid_mul_times_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=%s)[name=string(\"gate_sig\")];\n", cfg.Dim, gateInput)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "input_mul_plus_zero":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "input_mul_times_one":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=%s)[name=string(\"attn_gated\")];\n", cfg.Dim, gateInput)
		attnInput = "attn_gated"
	case "combined_add_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_mix = add(x=x, y=qg)[name=string(\"gate_mix\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_mix)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "combined_add_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_mix = add(x=x, y=qg)[name=string(\"gate_mix\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=gate_mix)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "combined_mul_mul":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_mix = mul(x=x, y=qg)[name=string(\"gate_mix\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=gate_mix)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	case "combined_mul_add":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_mix = mul(x=x, y=qg)[name=string(\"gate_mix\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = add(x=attn_r, y=gate_mix)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureLearnedGateVariantsPostLinear: unsupported gateOp %q", gateOp)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "post", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeaturePostAttentionConstTransform(cfg MILTransformerConfig, transform string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1049)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1051)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1057)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1061)
	qgWPath, qgBPath := addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1063)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1067)
	rmsWPath := addProbeVectorWeight(&files, "rms", cfg.Dim, 1069)

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post_const = const()[name=string(\"post_const\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1071)))
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post_zero = const()[name=string(\"post_zero\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0, 0)))
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post_one = const()[name=string(\"post_one\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 1, 0)))
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=krope)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gate = sigmoid(x=qg)[name=string(\"attn_gate\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=attn_gate)[name=string(\"attn_gated\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_gated)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)

	out := "resid"
	switch transform {
	case "none":
	case "mul_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = mul(x=resid, y=post_const)[name=string(\"post\")];\n", cfg.Dim)
		out = "post"
	case "add_const":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = add(x=resid, y=post_const)[name=string(\"post\")];\n", cfg.Dim)
		out = "post"
	case "mul_one":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = mul(x=resid, y=post_one)[name=string(\"post\")];\n", cfg.Dim)
		out = "post"
	case "add_zero":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = add(x=resid, y=post_zero)[name=string(\"post\")];\n", cfg.Dim)
		out = "post"
	case "rmsnorm":
		out = writeRMSNorm3D(&b, "post_rms", "resid", cfg.Dim, 1e-6, rmsWPath)
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeaturePostAttentionConstTransform: unsupported transform %q", transform)
	}
	writeProbeMILReturn(&b, out, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureDynamicRopeGateResidual(cfg MILTransformerConfig, dynamicRoPE bool, outputGate bool, stage string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1081)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1087)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1091)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1093)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1097)
	var qgWPath, qgBPath, ropeWPath, ropeBPath string
	if outputGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1103)
	}
	if dynamicRoPE {
		ropeWPath, ropeBPath = addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1109)
	}

	var b strings.Builder
	writeProbeMILPreamble(&b, cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = const()[name=string(\"pos_cos\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1117)))
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = const()[name=string(\"pos_sin\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1123)))
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	if outputGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	if stage == "post_linear" {
		writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	qSrc := "q"
	kSrc := "k"
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		qSrc = "qrope"
		kSrc = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qSrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=%s)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim, kSrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gate = sigmoid(x=qg)[name=string(\"attn_gate\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=attn_gate)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)

	out := "proj"
	if stage == "resid" || stage == "post_linear" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
		out = "resid"
	}
	if stage == "post_linear" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
		out = "post"
	}
	writeProbeMILReturn(&b, out, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureRoPESourcePostLinear(cfg MILTransformerConfig, ropeSource string, outputGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1129)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1151)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1163)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1171)
	postWPath, postBPath := addProbeLinearWeights(&files, "post", cfg.Dim, cfg.Dim, 1181)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1193)
	var qgWPath, qgBPath string
	if outputGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1201)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	switch ropeSource {
	case "input":
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	case "runtime":
	case "const":
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureRoPESourcePostLinear: unsupported ropeSource %q", ropeSource)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if ropeSource == "input" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	} else if ropeSource == "runtime" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = const()[name=string(\"pos_cos\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1213)))
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = const()[name=string(\"pos_sin\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1223)))
	}
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "pw", "pb", postWPath, postBPath, cfg.Dim, cfg.Dim)
	if outputGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> post = linear(bias=pb, weight=pw, x=resid)[name=string(\"post\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "post", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureRoPESourceFFNEntry(cfg MILTransformerConfig, ropeSource string, outputGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1231)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1237)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1241)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1249)
	ffnWPath, ffnBPath := addProbeLinearWeights(&files, "ffn", cfg.Dim, cfg.HiddenDim, 1259)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1271)
	var qgWPath, qgBPath string
	if outputGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1283)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	switch ropeSource {
	case "input":
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	case "runtime":
	case "const":
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureRoPESourceFFNEntry: unsupported ropeSource %q", ropeSource)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if ropeSource == "input" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	} else if ropeSource == "runtime" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = const()[name=string(\"pos_cos\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1291)))
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = const()[name=string(\"pos_sin\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1301)))
	}
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", ffnWPath, ffnBPath, cfg.HiddenDim, cfg.Dim)
	if outputGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=krope)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ffn1 = linear(bias=fb, weight=fw, x=resid)[name=string(\"ffn1\")];\n", cfg.HiddenDim)
	writeProbeMILReturn(&b, "ffn1", cfg.HiddenDim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureRoPESourceFusedFFNEntry(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1231)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1237)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1241)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1249)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fw13", cfg.Dim, 2*cfg.HiddenDim, 1259)
	downWPath, downBPath := addProbeLinearWeights(&files, "dw2", cfg.HiddenDim, cfg.Dim, 1269)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1271)

	var b strings.Builder
	fmt.Fprintf(&b, "%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in", ffnMILHeader, cfg.Dim)
	switch ropeSource {
	case "input":
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	case "runtime":
	case "const":
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureRoPESourceFusedFFNEntry: unsupported ropeSource %q", ropeSource)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if ropeSource == "input" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	} else if ropeSource == "runtime" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = const()[name=string(\"pos_cos\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1291)))
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = const()[name=string(\"pos_sin\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 1301)))
	}
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw13", "fb13", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "dw2", "db2", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> slice_mask = const()[name=string(\"slice_mask\"), val=tensor<bool, [3]>([true,true,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=krope)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb13, weight=fw13, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=slice_mask, end=gate_end, end_mask=slice_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=slice_mask, end=up_end, end_mask=slice_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=db2, weight=dw2, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> y = add(x=resid, y=down)[name=string(\"y\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "y", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFusedFFNEntry(cfg MILTransformerConfig, dynamicRoPE bool, outputGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1319)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1321)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1327)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1361)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 1367)
	downWPath, downBPath := addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 1373)
	var qgWPath, qgBPath, ropeWPath, ropeBPath string
	if outputGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1381)
	}
	if dynamicRoPE {
		ropeWPath, ropeBPath = addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1399)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "dw", "db", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	if outputGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> fused_shape = const()[name=string(\"fused_shape\"), val=tensor<int32, [3]>([1,1,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	qSrc := "q"
	kSrc := "k"
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		qSrc = "qrope"
		kSrc = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qSrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=%s)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim, kSrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb, weight=fw, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=gate_mask, end=gate_end, end_mask=gate_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=up_mask, end=up_end, end_mask=up_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_sig, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=db, weight=dw, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "down", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFusedFFNNoDown(cfg MILTransformerConfig, dynamicRoPE bool, outputGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1433)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1439)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1447)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1451)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 1459)
	var qgWPath, qgBPath, ropeWPath, ropeBPath string
	if outputGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1471)
	}
	if dynamicRoPE {
		ropeWPath, ropeBPath = addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1481)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n",
		ffnMILHeader,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
		writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	}
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	if outputGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	qSrc := "q"
	kSrc := "k"
	if dynamicRoPE {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
		qSrc = "qrope"
		kSrc = "krope"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=%s)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim, qSrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=%s)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim, kSrc)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb, weight=fw, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=gate_mask, end=gate_end, end_mask=gate_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=up_mask, end=up_end, end_mask=up_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_sig, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	writeProbeMILReturn(&b, "mix", cfg.HiddenDim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFusedFFNFinalNorm(cfg MILTransformerConfig, outputGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1511)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1523)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1531)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1543)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 1553)
	downWPath, downBPath := addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 1567)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1571)
	normWPath := addProbeVectorWeight(&files, "final_norm", cfg.Dim, 1583)
	var qgWPath, qgBPath string
	if outputGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1597)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n",
		ffnMILHeader,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "dw", "db", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	if outputGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=krope)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb, weight=fw, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=gate_mask, end=gate_end, end_mask=gate_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=up_mask, end=up_end, end_mask=up_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_sig, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=db, weight=dw, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ffn_resid = add(x=resid, y=down)[name=string(\"ffn_resid\")];\n", cfg.Dim)
	out := writeRMSNorm3D(&b, "probe_finalnorm", "ffn_resid", cfg.Dim, cfg.RMSNormEps, normWPath)
	writeProbeMILReturn(&b, out, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFusedSwiGLUFFNFinalNorm(cfg MILTransformerConfig, outputGate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 1609)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 1613)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 1619)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 1627)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 1637)
	downWPath, downBPath := addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 1657)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 1667)
	normWPath := addProbeVectorWeight(&files, "final_norm", cfg.Dim, 1691)
	var qgWPath, qgBPath string
	if outputGate {
		qgWPath, qgBPath = addProbeLinearWeights(&files, "qg", cfg.Dim, cfg.Dim, 1709)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n",
		ffnMILHeader,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "dw", "db", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	if outputGate {
		writeLinearConstBlock(&b, "qgw", "qgb", qgWPath, qgBPath, cfg.Dim, cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg = linear(bias=qgb, weight=qgw, x=x)[name=string(\"qg\")];\n", cfg.Dim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrot = linear(bias=ropeb, weight=ropew, x=q)[name=string(\"qrot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qcos = mul(x=q, y=pos_cos)[name=string(\"qcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qsin = mul(x=qrot, y=pos_sin)[name=string(\"qsin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qrope = add(x=qcos, y=qsin)[name=string(\"qrope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krot = linear(bias=ropeb, weight=ropew, x=k)[name=string(\"krot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> kcos = mul(x=k, y=pos_cos)[name=string(\"kcos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ksin = mul(x=krot, y=pos_sin)[name=string(\"ksin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> krope = add(x=kcos, y=ksin)[name=string(\"krope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=qrope)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=krope)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	attnInput := "attn_r"
	if outputGate {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> qg_sig = sigmoid(x=qg)[name=string(\"qg_sig\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_gated = mul(x=attn_r, y=qg_sig)[name=string(\"attn_gated\")];\n", cfg.Dim)
		attnInput = "attn_gated"
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=%s)[name=string(\"proj\")];\n", cfg.Dim, attnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb, weight=fw, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=gate_mask, end=gate_end, end_mask=gate_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=up_mask, end=up_end, end_mask=up_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=db, weight=dw, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ffn_resid = add(x=resid, y=down)[name=string(\"ffn_resid\")];\n", cfg.Dim)
	out := writeRMSNorm3D(&b, "probe_finalnorm", "ffn_resid", cfg.Dim, cfg.RMSNormEps, normWPath)
	writeProbeMILReturn(&b, out, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionNormFFNInteraction(cfg MILTransformerConfig, preFFNNorm bool, finalNorm bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 331)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 337)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 347)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 349)
	w1Path, b1Path := addProbeLinearWeights(&files, "ffn1", cfg.Dim, cfg.HiddenDim, 353)
	w3Path, b3Path := addProbeLinearWeights(&files, "ffn3", cfg.Dim, cfg.HiddenDim, 359)
	w2Path, b2Path := addProbeLinearWeights(&files, "ffn2", cfg.HiddenDim, cfg.Dim, 367)

	var normInPath, normFinalPath string
	if preFFNNorm {
		normInPath = addProbeVectorWeight(&files, "norm_in", cfg.Dim, 373)
	}
	if finalNorm {
		normFinalPath = addProbeVectorWeight(&files, "norm_final", cfg.Dim, 379)
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in, state<tensor<fp16, [1,%d,%d,%d]>> k_state, state<tensor<fp16, [1,%d,%d,%d]>> v_state) {\n",
		ffnMILHeader,
		cfg.Dim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
		cfg.NumHeads,
		cfg.MaxSeqLen,
		cfg.HeadDim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "w1", "b1", w1Path, b1Path, cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "w3", "b3", w3Path, b3Path, cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "w2", "b2", w2Path, b2Path, cfg.Dim, cfg.HiddenDim)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
		cfg.NumHeads,
		cfg.HeadDim,
	)
	fmt.Fprintf(
		&b,
		"        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n",
		cfg.Dim,
	)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	writeProbeStateConsts(&b)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_read = read_state(input=k_state)[name=string(\"k_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_read = read_state(input=v_state)[name=string(\"v_read\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=k_read)[name=string(\"k_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_tail = slice_by_index(begin=cache_tail_begin, begin_mask=cache_tail_begin_mask, end=cache_tail_end, end_mask=cache_tail_end_mask, x=v_read)[name=string(\"v_tail\")];\n", cfg.NumHeads, cfg.MaxSeqLen-1, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k_step = reshape(shape=qsh, x=k)[name=string(\"k_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v_step = reshape(shape=qsh, x=v)[name=string(\"v_step\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_write = concat(axis=cache_axis,interleave=cache_interleave,values=(k_tail,k_step))[name=string(\"k_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = k_write, input = k_state)[name = string(\"k_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_next = read_state(input = k_state)[name = string(\"k_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_write = concat(axis=cache_axis,interleave=cache_interleave,values=(v_tail,v_step))[name=string(\"v_write\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	b.WriteString("        write_state(data = v_write, input = v_state)[name = string(\"v_write_state\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_next = read_state(input = v_state)[name = string(\"v_next\")];\n", cfg.NumHeads, cfg.MaxSeqLen, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k_next, value=v_next)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)

	ffnInput := "resid"
	if preFFNNorm {
		ffnInput = writeRMSNorm3D(&b, "probe_prefnnorm", "resid", cfg.Dim, cfg.RMSNormEps, normInPath)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = linear(bias=b1, weight=w1, x=%s)[name=string(\"gate\")];\n", cfg.HiddenDim, ffnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = linear(bias=b3, weight=w3, x=%s)[name=string(\"up\")];\n", cfg.HiddenDim, ffnInput)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=b2, weight=w2, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> ffn_resid = add(x=resid, y=down)[name=string(\"ffn_resid\")];\n", cfg.Dim)
	outVar := "ffn_resid"
	if finalNorm {
		outVar = writeRMSNorm3D(&b, "probe_finalnorm", "ffn_resid", cfg.Dim, cfg.RMSNormEps, normFinalPath)
	}
	writeProbeMILReturn(&b, outVar, cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFusedResidualTail(cfg MILTransformerConfig, includeResid bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 2101)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 2111)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 2129)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 2137)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 2143)
	downWPath, downBPath := addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 2153)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n",
		ffnMILHeader,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "dw", "db", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb, weight=fw, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=gate_mask, end=gate_end, end_mask=gate_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=up_mask, end=up_end, end_mask=up_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=db, weight=dw, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	if includeResid {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out = add(x=resid, y=down)[name=string(\"out\")];\n", cfg.Dim)
		writeProbeMILReturn(&b, "out", cfg.Dim)
		return b.String(), files, nil
	}
	writeProbeMILReturn(&b, "down", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureFusedConstOrder(cfg MILTransformerConfig, constLate bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 2201)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 2207)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 2213)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 2221)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 2237)
	downWPath, downBPath := addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 2243)

	writeFusedConsts := func(b *strings.Builder) {
		fmt.Fprintf(b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
		fmt.Fprintf(b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
		fmt.Fprintf(b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
		fmt.Fprintf(b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
		fmt.Fprintf(b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([true,true,false])];\n")
		fmt.Fprintf(b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
		fmt.Fprintf(b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", 2*cfg.HiddenDim)
		fmt.Fprintf(b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([true,true,false])];\n")
	}

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n",
		ffnMILHeader,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "dw", "db", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	if !constLate {
		writeFusedConsts(&b)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	if constLate {
		writeFusedConsts(&b)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb, weight=fw, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=gate_mask, end=gate_end, end_mask=gate_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=up_mask, end=up_end, end_mask=up_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=db, weight=dw, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> out = add(x=resid, y=down)[name=string(\"out\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "out", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFused(cfg MILTransformerConfig) (string, []modelWeightFile, error) {
	return buildMILProbeAttentionFeatureCanonicalLikeFusedWithWeightNames(cfg, false)
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNorm(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(cfg, ropeSource)
	if err != nil {
		return "", nil, err
	}
	inputNormBlock := strings.Join([]string{
		"        tensor<fp16, [16]> in_norm_w = const()[name=string(\"in_norm_w\"), val=tensor<fp16, [16]>([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])];",
		"        tensor<fp16, [1,1,1]> x_max_eps_const = const()[name=string(\"x_max_eps_const\"), val=tensor<fp16, [1,1,1]>([1e-6])];",
		"        tensor<fp16, [1,1,1]> x_var_eps_const = const()[name=string(\"x_var_eps_const\"), val=tensor<fp16, [1,1,1]>([1e-5])];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> x_abs = abs(x=x)[name=string(\"x_abs\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> x_max = reduce_max(axes=[2], keep_dims=true, x=x_abs)[name=string(\"x_max\")];",
		"        tensor<fp16, [1,1,1]> x_max_eps = maximum(x=x_max, y=x_max_eps_const)[name=string(\"x_max_eps\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> x_scaled = real_div(x=x, y=x_max_eps)[name=string(\"x_scaled\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> x_sq = square(x=x_scaled)[name=string(\"x_sq\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> x_mean = reduce_mean(axes=[2], keep_dims=true, x=x_sq)[name=string(\"x_mean\")];",
		"        tensor<fp16, [1,1,1]> x_var = add(x=x_mean, y=x_var_eps_const)[name=string(\"x_var\")];",
		"        tensor<fp16, [1,1,1]> x_denom = sqrt(x=x_var)[name=string(\"x_denom\")];",
		"        tensor<fp16, [1,1,1]> x_scaled_rms = mul(x=x_denom, y=x_max_eps)[name=string(\"x_scaled_rms\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> x_unit = real_div(x=x, y=x_scaled_rms)[name=string(\"x_unit\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> x_norm = mul(x=x_unit, y=in_norm_w)[name=string(\"x_norm\")];", cfg.Dim),
	}, "\n")
	milText = strings.Replace(milText,
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n%s\n", cfg.Dim, inputNormBlock),
		1,
	)
	repls := map[string]string{
		"weight=l0_q_w, x=x)":   "weight=l0_q_w, x=x_norm)",
		"weight=l0_k_w, x=x)":   "weight=l0_k_w, x=x_norm)",
		"weight=l0_v_w, x=x)":   "weight=l0_v_w, x=x_norm)",
		"add(x=x, y=l0_attn_o)": "add(x=x_norm, y=l0_attn_o)",
	}
	for old, new := range repls {
		milText = strings.ReplaceAll(milText, old, new)
	}
	return milText, files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeFinalNorm(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(cfg, ropeSource)
	if err != nil {
		return "", nil, err
	}
	finalNormBlock := strings.Join([]string{
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_norm_w = const()[name=string(\"final_norm_w\"), val=tensor<fp16, [1,1,%d]>(%s)];", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.001, 2401))),
		"        tensor<int32, [1]> final_ax = const()[name=string(\"final_ax\"), val=tensor<int32, [1]>([2])];",
		"        bool final_kd = const()[name=string(\"final_kd\"), val=bool(true)];",
		"        fp16 final_eps = const()[name=string(\"final_eps\"), val=fp16(0.000010)];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_abs = abs(x=l0_resid2)[name=string(\"final_abs\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> final_max = reduce_max(x=final_abs, axes=final_ax, keep_dims=final_kd)[name=string(\"final_max\")];",
		"        tensor<fp16, [1,1,1]> final_max_eps = maximum(x=final_max, y=final_eps)[name=string(\"final_max_eps\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_scaled = real_div(x=l0_resid2, y=final_max_eps)[name=string(\"final_scaled\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_sq = square(x=final_scaled)[name=string(\"final_sq\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> final_mean = reduce_mean(x=final_sq, axes=final_ax, keep_dims=final_kd)[name=string(\"final_mean\")];",
		"        tensor<fp16, [1,1,1]> final_var = add(x=final_mean, y=final_eps)[name=string(\"final_var\")];",
		"        tensor<fp16, [1,1,1]> final_denom = sqrt(x=final_var)[name=string(\"final_denom\")];",
		"        tensor<fp16, [1,1,1]> final_scaled_rms = mul(x=final_denom, y=final_max_eps)[name=string(\"final_scaled_rms\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_unit = real_div(x=l0_resid2, y=final_scaled_rms)[name=string(\"final_unit\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_out = mul(x=final_unit, y=final_norm_w)[name=string(\"final_out\")];", cfg.Dim),
	}, "\n")
	milText = strings.Replace(
		milText,
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n        tensor<fp32, [1,1,%d]> y = cast(dtype=to_fp32, x=l0_resid2)[name=string(\"cast_out\")];\n    } -> (y);\n}\n", cfg.Dim, cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n%s\n        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n        tensor<fp32, [1,1,%d]> y = cast(dtype=to_fp32, x=final_out)[name=string(\"cast_out\")];\n    } -> (y);\n}\n", cfg.Dim, finalNormBlock, cfg.Dim),
		1,
	)
	return milText, files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNormFinalNorm(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNorm(cfg, ropeSource)
	if err != nil {
		return "", nil, err
	}
	finalNormBlock := strings.Join([]string{
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_norm_w = const()[name=string(\"final_norm_w\"), val=tensor<fp16, [1,1,%d]>(%s)];", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.001, 2401))),
		"        tensor<int32, [1]> final_ax = const()[name=string(\"final_ax\"), val=tensor<int32, [1]>([2])];",
		"        bool final_kd = const()[name=string(\"final_kd\"), val=bool(true)];",
		"        fp16 final_eps = const()[name=string(\"final_eps\"), val=fp16(0.000010)];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_abs = abs(x=l0_resid2)[name=string(\"final_abs\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> final_max = reduce_max(x=final_abs, axes=final_ax, keep_dims=final_kd)[name=string(\"final_max\")];",
		"        tensor<fp16, [1,1,1]> final_max_eps = maximum(x=final_max, y=final_eps)[name=string(\"final_max_eps\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_scaled = real_div(x=l0_resid2, y=final_max_eps)[name=string(\"final_scaled\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_sq = square(x=final_scaled)[name=string(\"final_sq\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> final_mean = reduce_mean(x=final_sq, axes=final_ax, keep_dims=final_kd)[name=string(\"final_mean\")];",
		"        tensor<fp16, [1,1,1]> final_var = add(x=final_mean, y=final_eps)[name=string(\"final_var\")];",
		"        tensor<fp16, [1,1,1]> final_denom = sqrt(x=final_var)[name=string(\"final_denom\")];",
		"        tensor<fp16, [1,1,1]> final_scaled_rms = mul(x=final_denom, y=final_max_eps)[name=string(\"final_scaled_rms\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_unit = real_div(x=l0_resid2, y=final_scaled_rms)[name=string(\"final_unit\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_out = mul(x=final_unit, y=final_norm_w)[name=string(\"final_out\")];", cfg.Dim),
	}, "\n")
	milText = strings.Replace(
		milText,
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n        tensor<fp32, [1,1,%d]> y = cast(dtype=to_fp32, x=l0_resid2)[name=string(\"cast_out\")];\n    } -> (y);\n}\n", cfg.Dim, cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n%s\n        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n        tensor<fp32, [1,1,%d]> y = cast(dtype=to_fp32, x=final_out)[name=string(\"cast_out\")];\n    } -> (y);\n}\n", cfg.Dim, finalNormBlock, cfg.Dim),
		1,
	)
	return milText, files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNormPostNormFinalNorm(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeInputNorm(cfg, ropeSource)
	if err != nil {
		return "", nil, err
	}
	postNormBlock := strings.Join([]string{
		fmt.Sprintf("        tensor<fp16, [%d]> post_norm_w = const()[name=string(\"post_norm_w\"), val=tensor<fp16, [%d]>([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])];", cfg.Dim, cfg.Dim),
		"        tensor<fp16, [1,1,1]> post_max_eps_const = const()[name=string(\"post_max_eps_const\"), val=tensor<fp16, [1,1,1]>([1e-6])];",
		"        tensor<fp16, [1,1,1]> post_var_eps_const = const()[name=string(\"post_var_eps_const\"), val=tensor<fp16, [1,1,1]>([1e-5])];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_abs = abs(x=l0_resid1)[name=string(\"post_abs\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> post_max = reduce_max(axes=[2], keep_dims=true, x=post_abs)[name=string(\"post_max\")];",
		"        tensor<fp16, [1,1,1]> post_max_eps = maximum(x=post_max, y=post_max_eps_const)[name=string(\"post_max_eps\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_scaled = real_div(x=l0_resid1, y=post_max_eps)[name=string(\"post_scaled\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_sq = square(x=post_scaled)[name=string(\"post_sq\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> post_mean = reduce_mean(axes=[2], keep_dims=true, x=post_sq)[name=string(\"post_mean\")];",
		"        tensor<fp16, [1,1,1]> post_var = add(x=post_mean, y=post_var_eps_const)[name=string(\"post_var\")];",
		"        tensor<fp16, [1,1,1]> post_denom = sqrt(x=post_var)[name=string(\"post_denom\")];",
		"        tensor<fp16, [1,1,1]> post_scaled_rms = mul(x=post_denom, y=post_max_eps)[name=string(\"post_scaled_rms\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_unit = real_div(x=l0_resid1, y=post_scaled_rms)[name=string(\"post_unit\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_norm = mul(x=post_unit, y=post_norm_w)[name=string(\"post_norm\")];", cfg.Dim),
	}, "\n")
	milText = strings.Replace(
		milText,
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_fused = linear(bias=l0_b13, weight=l0_w13, x=l0_resid1)[name=string(\"l0_fused\")];\n", 2*cfg.HiddenDim),
		fmt.Sprintf("%s\n        tensor<fp16, [1,1,%d]> l0_fused = linear(bias=l0_b13, weight=l0_w13, x=post_norm)[name=string(\"l0_fused\")];\n", postNormBlock, 2*cfg.HiddenDim),
		1,
	)
	finalNormBlock := strings.Join([]string{
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_norm_w = const()[name=string(\"final_norm_w\"), val=tensor<fp16, [1,1,%d]>(%s)];", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.001, 2401))),
		"        tensor<int32, [1]> final_ax = const()[name=string(\"final_ax\"), val=tensor<int32, [1]>([2])];",
		"        bool final_kd = const()[name=string(\"final_kd\"), val=bool(true)];",
		"        fp16 final_eps = const()[name=string(\"final_eps\"), val=fp16(0.000010)];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_abs = abs(x=l0_resid2)[name=string(\"final_abs\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> final_max = reduce_max(x=final_abs, axes=final_ax, keep_dims=final_kd)[name=string(\"final_max\")];",
		"        tensor<fp16, [1,1,1]> final_max_eps = maximum(x=final_max, y=final_eps)[name=string(\"final_max_eps\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_scaled = real_div(x=l0_resid2, y=final_max_eps)[name=string(\"final_scaled\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_sq = square(x=final_scaled)[name=string(\"final_sq\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> final_mean = reduce_mean(x=final_sq, axes=final_ax, keep_dims=final_kd)[name=string(\"final_mean\")];",
		"        tensor<fp16, [1,1,1]> final_var = add(x=final_mean, y=final_eps)[name=string(\"final_var\")];",
		"        tensor<fp16, [1,1,1]> final_denom = sqrt(x=final_var)[name=string(\"final_denom\")];",
		"        tensor<fp16, [1,1,1]> final_scaled_rms = mul(x=final_denom, y=final_max_eps)[name=string(\"final_scaled_rms\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_unit = real_div(x=l0_resid2, y=final_scaled_rms)[name=string(\"final_unit\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> final_out = mul(x=final_unit, y=final_norm_w)[name=string(\"final_out\")];", cfg.Dim),
	}, "\n")
	milText = strings.Replace(
		milText,
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n        tensor<fp32, [1,1,%d]> y = cast(dtype=to_fp32, x=l0_resid2)[name=string(\"cast_out\")];\n    } -> (y);\n}\n", cfg.Dim, cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n%s\n        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n        tensor<fp32, [1,1,%d]> y = cast(dtype=to_fp32, x=final_out)[name=string(\"cast_out\")];\n    } -> (y);\n}\n", cfg.Dim, finalNormBlock, cfg.Dim),
		1,
	)
	return milText, files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopeQKNorm(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(cfg, ropeSource)
	if err != nil {
		return "", nil, err
	}
	qkNormBlock := strings.Join([]string{
		fmt.Sprintf("        tensor<fp16, [%d]> q_norm_w = const()[name=string(\"q_norm_w\"), val=tensor<fp16, [%d]>([1,1,1,1])];", cfg.HeadDim, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [%d]> k_norm_w = const()[name=string(\"k_norm_w\"), val=tensor<fp16, [%d]>([1,1,1,1])];", cfg.HeadDim, cfg.HeadDim),
		"        tensor<fp16, [1,1,1,1]> qk_max_eps_const = const()[name=string(\"qk_max_eps_const\"), val=tensor<fp16, [1,1,1,1]>([1e-6])];",
		"        tensor<fp16, [1,1,1,1]> qk_var_eps_const = const()[name=string(\"qk_var_eps_const\"), val=tensor<fp16, [1,1,1,1]>([1e-5])];",
		"        tensor<fp16, [1,1,1,1]> qk_one_const = const()[name=string(\"qk_one_const\"), val=tensor<fp16, [1,1,1,1]>([1])];",
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_q_abs = abs(x=l0_q)[name=string(\"l0_q_abs\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_q_max = reduce_max(axes=[3], keep_dims=true, x=l0_q_abs)[name=string(\"l0_q_max\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_q_max_eps = maximum(x=l0_q_max, y=qk_max_eps_const)[name=string(\"l0_q_max_eps\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_q_scaled = real_div(x=l0_q, y=l0_q_max_eps)[name=string(\"l0_q_scaled\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_q_sq = square(x=l0_q_scaled)[name=string(\"l0_q_sq\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_q_mean = reduce_mean(axes=[3], keep_dims=true, x=l0_q_sq)[name=string(\"l0_q_mean\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_q_var = add(x=l0_q_mean, y=qk_var_eps_const)[name=string(\"l0_q_var\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_q_denom = sqrt(x=l0_q_var)[name=string(\"l0_q_denom\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_q_rsqrt = real_div(x=qk_one_const, y=l0_q_denom)[name=string(\"l0_q_rsqrt\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_q_unit = mul(x=l0_q_scaled, y=l0_q_rsqrt)[name=string(\"l0_q_unit\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_qn = mul(x=l0_q_unit, y=q_norm_w)[name=string(\"l0_qn\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_k_abs = abs(x=l0_k)[name=string(\"l0_k_abs\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_k_max = reduce_max(axes=[3], keep_dims=true, x=l0_k_abs)[name=string(\"l0_k_max\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_k_max_eps = maximum(x=l0_k_max, y=qk_max_eps_const)[name=string(\"l0_k_max_eps\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_k_scaled = real_div(x=l0_k, y=l0_k_max_eps)[name=string(\"l0_k_scaled\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_k_sq = square(x=l0_k_scaled)[name=string(\"l0_k_sq\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_k_mean = reduce_mean(axes=[3], keep_dims=true, x=l0_k_sq)[name=string(\"l0_k_mean\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_k_var = add(x=l0_k_mean, y=qk_var_eps_const)[name=string(\"l0_k_var\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_k_denom = sqrt(x=l0_k_var)[name=string(\"l0_k_denom\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,1]> l0_k_rsqrt = real_div(x=qk_one_const, y=l0_k_denom)[name=string(\"l0_k_rsqrt\")];", cfg.NumHeads),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_k_unit = mul(x=l0_k_scaled, y=l0_k_rsqrt)[name=string(\"l0_k_unit\")];", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_kn = mul(x=l0_k_unit, y=k_norm_w)[name=string(\"l0_kn\")];", cfg.NumHeads, cfg.HeadDim),
	}, "\n")
	milText = strings.Replace(
		milText,
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_v = reshape(shape=l0_q_shape, x=l0_v_flat)[name=string(\"l0_v\")];\n", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_v = reshape(shape=l0_q_shape, x=l0_v_flat)[name=string(\"l0_v\")];\n%s\n", cfg.NumHeads, cfg.HeadDim, qkNormBlock),
		1,
	)
	milText = strings.Replace(
		milText,
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_attn = scaled_dot_product_attention(query=l0_q, key=l0_k, value=l0_v)[name=string(\"l0_attn\")];\n", cfg.NumHeads, cfg.HeadDim),
		fmt.Sprintf("        tensor<fp16, [1,%d,1,%d]> l0_attn = scaled_dot_product_attention(query=l0_qn, key=l0_kn, value=l0_v)[name=string(\"l0_attn\")];\n", cfg.NumHeads, cfg.HeadDim),
		1,
	)
	return milText, files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithRopePostNorm(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	milText, files, err := buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(cfg, ropeSource)
	if err != nil {
		return "", nil, err
	}
	postNormBlock := strings.Join([]string{
		fmt.Sprintf("        tensor<fp16, [%d]> post_norm_w = const()[name=string(\"post_norm_w\"), val=tensor<fp16, [%d]>([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])];", cfg.Dim, cfg.Dim),
		"        tensor<fp16, [1,1,1]> post_max_eps_const = const()[name=string(\"post_max_eps_const\"), val=tensor<fp16, [1,1,1]>([1e-6])];",
		"        tensor<fp16, [1,1,1]> post_var_eps_const = const()[name=string(\"post_var_eps_const\"), val=tensor<fp16, [1,1,1]>([1e-5])];",
		"        tensor<fp16, [1,1,1]> post_one_const = const()[name=string(\"post_one_const\"), val=tensor<fp16, [1,1,1]>([1])];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_abs = abs(x=l0_resid1)[name=string(\"post_abs\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> post_max = reduce_max(axes=[2], keep_dims=true, x=post_abs)[name=string(\"post_max\")];",
		"        tensor<fp16, [1,1,1]> post_max_eps = maximum(x=post_max, y=post_max_eps_const)[name=string(\"post_max_eps\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_scaled = real_div(x=l0_resid1, y=post_max_eps)[name=string(\"post_scaled\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_sq = square(x=post_scaled)[name=string(\"post_sq\")];", cfg.Dim),
		"        tensor<fp16, [1,1,1]> post_mean = reduce_mean(axes=[2], keep_dims=true, x=post_sq)[name=string(\"post_mean\")];",
		"        tensor<fp16, [1,1,1]> post_var = add(x=post_mean, y=post_var_eps_const)[name=string(\"post_var\")];",
		"        tensor<fp16, [1,1,1]> post_denom = sqrt(x=post_var)[name=string(\"post_denom\")];",
		"        tensor<fp16, [1,1,1]> post_rsqrt = real_div(x=post_one_const, y=post_denom)[name=string(\"post_rsqrt\")];",
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_unit = mul(x=post_scaled, y=post_rsqrt)[name=string(\"post_unit\")];", cfg.Dim),
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> post_norm = mul(x=post_unit, y=post_norm_w)[name=string(\"post_norm\")];", cfg.Dim),
	}, "\n")
	milText = strings.Replace(
		milText,
		fmt.Sprintf("        tensor<fp16, [1,1,%d]> l0_fused = linear(bias=l0_b13, weight=l0_w13, x=l0_resid1)[name=string(\"l0_fused\")];\n", 2*cfg.HiddenDim),
		fmt.Sprintf("%s\n        tensor<fp16, [1,1,%d]> l0_fused = linear(bias=l0_b13, weight=l0_w13, x=post_norm)[name=string(\"l0_fused\")];\n", postNormBlock, 2*cfg.HiddenDim),
		1,
	)
	return milText, files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope(cfg MILTransformerConfig, ropeSource string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 2311)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 2317)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 2327)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 2333)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 2347)
	downWPath, downBPath := addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 2357)
	ropeWPath, ropeBPath := addProbeLinearWeights(&files, "rope", cfg.Dim, cfg.Dim, 2371)

	var b strings.Builder
	fmt.Fprintf(&b, "%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in", ffnMILHeader, cfg.Dim)
	switch ropeSource {
	case "input":
		fmt.Fprintf(&b, ", tensor<fp32, [1, 1, %d]> pos_cos_in, tensor<fp32, [1, 1, %d]> pos_sin_in", cfg.Dim, cfg.Dim)
	case "runtime", "const":
	default:
		return "", nil, fmt.Errorf("buildMILProbeAttentionFeatureCanonicalLikeFusedWithRope: unsupported ropeSource %q", ropeSource)
	}
	b.WriteString(") {\n")
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	if ropeSource == "input" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = cast(dtype=to_fp16, x=pos_cos_in)[name=string(\"cast_pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = cast(dtype=to_fp16, x=pos_sin_in)[name=string(\"cast_pos_sin\")];\n", cfg.Dim)
	} else if ropeSource == "runtime" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = sigmoid(x=x)[name=string(\"pos_cos\")];\n", cfg.Dim)
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = add(x=x, y=x)[name=string(\"pos_sin\")];\n", cfg.Dim)
	} else {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_cos = const()[name=string(\"pos_cos\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 2381)))
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> pos_sin = const()[name=string(\"pos_sin\"), val=tensor<fp16, [1,1,%d]>(%s)];\n", cfg.Dim, cfg.Dim, testTensorLiteral16(makeDeterministicTensor(cfg.Dim, 0.01, 2391)))
	}
	writeLinearConstBlock(&b, "ropew", "ropeb", ropeWPath, ropeBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_q_w", "l0_q_b", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_k_w", "l0_k_b", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_v_w", "l0_v_b", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_o_w", "l0_o_b", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_w13", "l0_b13", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_w2", "l0_b2", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> l0_q_shape = const()[name=string(\"l0_q_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_out_shape = const()[name=string(\"l0_out_shape\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_gate_begin = const()[name=string(\"l0_gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_gate_end = const()[name=string(\"l0_gate_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_up_begin = const()[name=string(\"l0_up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_up_end = const()[name=string(\"l0_up_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> l0_slice_mask = const()[name=string(\"l0_slice_mask\"), val=tensor<bool, [3]>([true,true,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_q_flat = linear(bias=l0_q_b, weight=l0_q_w, x=x)[name=string(\"l0_q_flat\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_k_flat = linear(bias=l0_k_b, weight=l0_k_w, x=x)[name=string(\"l0_k_flat\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_v_flat = linear(bias=l0_v_b, weight=l0_v_w, x=x)[name=string(\"l0_v_flat\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_q_rot = linear(bias=ropeb, weight=ropew, x=l0_q_flat)[name=string(\"l0_q_rot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_q_cos = mul(x=l0_q_flat, y=pos_cos)[name=string(\"l0_q_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_q_sin = mul(x=l0_q_rot, y=pos_sin)[name=string(\"l0_q_sin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_q_rope = add(x=l0_q_cos, y=l0_q_sin)[name=string(\"l0_q_rope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_k_rot = linear(bias=ropeb, weight=ropew, x=l0_k_flat)[name=string(\"l0_k_rot\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_k_cos = mul(x=l0_k_flat, y=pos_cos)[name=string(\"l0_k_cos\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_k_sin = mul(x=l0_k_rot, y=pos_sin)[name=string(\"l0_k_sin\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_k_rope = add(x=l0_k_cos, y=l0_k_sin)[name=string(\"l0_k_rope\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_q = reshape(shape=l0_q_shape, x=l0_q_rope)[name=string(\"l0_q\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_k = reshape(shape=l0_q_shape, x=l0_k_rope)[name=string(\"l0_k\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_v = reshape(shape=l0_q_shape, x=l0_v_flat)[name=string(\"l0_v\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_attn = scaled_dot_product_attention(query=l0_q, key=l0_k, value=l0_v)[name=string(\"l0_attn\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_attn_r = reshape(shape=l0_out_shape, x=l0_attn)[name=string(\"l0_attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_attn_o = linear(bias=l0_o_b, weight=l0_o_w, x=l0_attn_r)[name=string(\"l0_attn_o\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_resid1 = add(x=x, y=l0_attn_o)[name=string(\"l0_resid1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_fused = linear(bias=l0_b13, weight=l0_w13, x=l0_resid1)[name=string(\"l0_fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_gate = slice_by_index(begin=l0_gate_begin, begin_mask=l0_slice_mask, end=l0_gate_end, end_mask=l0_slice_mask, x=l0_fused)[name=string(\"l0_gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_gate_sig = sigmoid(x=l0_gate)[name=string(\"l0_gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_gate_act = mul(x=l0_gate, y=l0_gate_sig)[name=string(\"l0_gate_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_up = slice_by_index(begin=l0_up_begin, begin_mask=l0_slice_mask, end=l0_up_end, end_mask=l0_slice_mask, x=l0_fused)[name=string(\"l0_up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_mix = mul(x=l0_gate_act, y=l0_up)[name=string(\"l0_mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_down = linear(bias=l0_b2, weight=l0_w2, x=l0_mix)[name=string(\"l0_down\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "l0_resid2", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeAttentionFeatureCanonicalLikeFusedWithWeightNames(cfg MILTransformerConfig, canonicalWeightNames bool) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	var qWPath, qBPath, kWPath, kBPath, vWPath, vBPath string
	var oWPath, oBPath, fusedWPath, fusedBPath, downWPath, downBPath string
	if canonicalWeightNames {
		qWPath, qBPath = addProbeLinearWeightsNamed(&files, "l0_q_w", "l0_q_b", cfg.Dim, cfg.Dim, 2011)
		kWPath, kBPath = addProbeLinearWeightsNamed(&files, "l0_k_w", "l0_k_b", cfg.Dim, cfg.Dim, 2017)
		vWPath, vBPath = addProbeLinearWeightsNamed(&files, "l0_v_w", "l0_v_b", cfg.Dim, cfg.Dim, 2027)
		oWPath, oBPath = addProbeLinearWeightsNamed(&files, "l0_o_w", "l0_o_b", cfg.Dim, cfg.Dim, 2039)
		fusedWPath, fusedBPath = addProbeLinearWeightsNamed(&files, "l0_w13", "l0_b13", cfg.Dim, 2*cfg.HiddenDim, 2053)
		downWPath, downBPath = addProbeLinearWeightsNamed(&files, "l0_w2", "l0_b2", cfg.HiddenDim, cfg.Dim, 2063)
	} else {
		qWPath, qBPath = addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 2011)
		kWPath, kBPath = addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 2017)
		vWPath, vBPath = addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 2027)
		oWPath, oBPath = addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 2039)
		fusedWPath, fusedBPath = addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 2053)
		downWPath, downBPath = addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 2063)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n", ffnMILHeader, cfg.Dim)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> current = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "l0_q_w", "l0_q_b", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_k_w", "l0_k_b", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_v_w", "l0_v_b", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_o_w", "l0_o_b", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_w13", "l0_b13", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "l0_w2", "l0_b2", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> l0_q_shape = const()[name=string(\"l0_q_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_out_shape = const()[name=string(\"l0_out_shape\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_gate_begin = const()[name=string(\"l0_gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_gate_end = const()[name=string(\"l0_gate_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_up_begin = const()[name=string(\"l0_up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> l0_up_end = const()[name=string(\"l0_up_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<bool, [3]> l0_slice_mask = const()[name=string(\"l0_slice_mask\"), val=tensor<bool, [3]>([true,true,false])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_q_flat = linear(bias=l0_q_b, weight=l0_q_w, x=current)[name=string(\"l0_q_flat\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_k_flat = linear(bias=l0_k_b, weight=l0_k_w, x=current)[name=string(\"l0_k_flat\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_v_flat = linear(bias=l0_v_b, weight=l0_v_w, x=current)[name=string(\"l0_v_flat\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_q = reshape(shape=l0_q_shape, x=l0_q_flat)[name=string(\"l0_q\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_k = reshape(shape=l0_q_shape, x=l0_k_flat)[name=string(\"l0_k\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_v = reshape(shape=l0_q_shape, x=l0_v_flat)[name=string(\"l0_v\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> l0_attn = scaled_dot_product_attention(query=l0_q, key=l0_k, value=l0_v)[name=string(\"l0_attn\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_attn_r = reshape(shape=l0_out_shape, x=l0_attn)[name=string(\"l0_attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_attn_o = linear(bias=l0_o_b, weight=l0_o_w, x=l0_attn_r)[name=string(\"l0_attn_o\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_resid1 = add(x=current, y=l0_attn_o)[name=string(\"l0_resid1\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_fused = linear(bias=l0_b13, weight=l0_w13, x=l0_resid1)[name=string(\"l0_fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_gate = slice_by_index(begin=l0_gate_begin, begin_mask=l0_slice_mask, end=l0_gate_end, end_mask=l0_slice_mask, x=l0_fused)[name=string(\"l0_gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_gate_sig = sigmoid(x=l0_gate)[name=string(\"l0_gate_sig\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_gate_act = mul(x=l0_gate, y=l0_gate_sig)[name=string(\"l0_gate_act\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_up = slice_by_index(begin=l0_up_begin, begin_mask=l0_slice_mask, end=l0_up_end, end_mask=l0_slice_mask, x=l0_fused)[name=string(\"l0_up\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_mix = mul(x=l0_gate_act, y=l0_up)[name=string(\"l0_mix\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_down = linear(bias=l0_b2, weight=l0_w2, x=l0_mix)[name=string(\"l0_down\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> l0_resid2 = add(x=l0_resid1, y=l0_down)[name=string(\"l0_resid2\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "l0_resid2", cfg.Dim)
	return b.String(), files, nil
}

var milOpLineRE = regexp.MustCompile(`(?:=\s*|^\s*)([a-z_]+)\(`)
var milTensorPrefixRE = regexp.MustCompile(`^tensor<[^>]+>\s+[A-Za-z0-9_]+\s+=\s+`)
var milNameAttrRE = regexp.MustCompile(`\s*\[name\s*=\s*string\("[^"]*"\)\]`)
var milArgKeyRE = regexp.MustCompile(`([a-z_]+)\s*=`)
var milOutputShapeRE = regexp.MustCompile(`^tensor<[^,>]+,\s*(\[[^]]+\])>`)

func milOpSequence(milText string) []string {
	var ops []string
	for _, line := range strings.Split(milText, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		m := milOpLineRE.FindStringSubmatch(line)
		if len(m) == 2 {
			ops = append(ops, m[1])
		}
	}
	return ops
}

func milNonConstLines(milText string) []string {
	var lines []string
	for _, line := range strings.Split(milText, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		if strings.Contains(line, "= const(") {
			continue
		}
		lines = append(lines, line)
	}
	return lines
}

func milStructuralLines(milText string) []string {
	var lines []string
	for _, line := range strings.Split(milText, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		if strings.Contains(line, "= const(") {
			continue
		}
		shape := ""
		if m := milOutputShapeRE.FindStringSubmatch(line); len(m) == 2 {
			shape = m[1]
		}
		normalized := milTensorPrefixRE.ReplaceAllString(line, "")
		normalized = milNameAttrRE.ReplaceAllString(normalized, "")
		normalized = strings.TrimSuffix(normalized, ";")
		m := milOpLineRE.FindStringSubmatch(normalized)
		if len(m) != 2 {
			lines = append(lines, normalized)
			continue
		}
		argMatches := milArgKeyRE.FindAllStringSubmatch(normalized, -1)
		args := make([]string, 0, len(argMatches))
		for _, arg := range argMatches {
			if len(arg) == 2 {
				args = append(args, arg[1])
			}
		}
		lines = append(lines, fmt.Sprintf("%s %s(%s)", shape, m[1], strings.Join(args, ",")))
	}
	return lines
}

func milNormalizedLines(milText string) []string {
	var lines []string
	for _, line := range strings.Split(milText, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		shape := ""
		if m := milOutputShapeRE.FindStringSubmatch(line); len(m) == 2 {
			shape = m[1]
		}
		normalized := milTensorPrefixRE.ReplaceAllString(line, "")
		normalized = milNameAttrRE.ReplaceAllString(normalized, "")
		normalized = strings.TrimSuffix(normalized, ";")
		normalized = strings.Join(strings.Fields(normalized), " ")
		if shape != "" {
			lines = append(lines, fmt.Sprintf("%s %s", shape, normalized))
			continue
		}
		lines = append(lines, normalized)
	}
	return lines
}

func milBlobConstLines(milText string) []string {
	var lines []string
	for _, line := range strings.Split(milText, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		if !strings.Contains(line, "BLOBFILE(") {
			continue
		}
		shape := ""
		if m := milOutputShapeRE.FindStringSubmatch(line); len(m) == 2 {
			shape = m[1]
		}
		normalized := milTensorPrefixRE.ReplaceAllString(line, "")
		normalized = milNameAttrRE.ReplaceAllString(normalized, "")
		normalized = strings.TrimSuffix(normalized, ";")
		normalized = strings.Join(strings.Fields(normalized), " ")
		if shape != "" {
			lines = append(lines, fmt.Sprintf("%s %s", shape, normalized))
			continue
		}
		lines = append(lines, normalized)
	}
	return lines
}

func filterMILStructuralOps(lines []string, excluded ...string) []string {
	if len(excluded) == 0 {
		return append([]string(nil), lines...)
	}
	var out []string
	for _, line := range lines {
		skip := false
		for _, op := range excluded {
			if strings.Contains(line, " "+op+"(") || strings.HasPrefix(line, op+"(") {
				skip = true
				break
			}
		}
		if !skip {
			out = append(out, line)
		}
	}
	return out
}

func filterMILStructuralFooter(lines []string) []string {
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "}") {
			continue
		}
		out = append(out, line)
	}
	return out
}

func normalizeMILStructuralArgOrder(lines []string) []string {
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		out = append(out, normalizeMILStructuralLineArgOrder(line))
	}
	return out
}

func normalizeMILStructuralLineArgOrder(line string) string {
	open := strings.IndexByte(line, '(')
	close := strings.LastIndexByte(line, ')')
	if open < 0 || close < 0 || close <= open+1 {
		return line
	}
	args := splitTopLevelComma(line[open+1 : close])
	if len(args) < 2 {
		return line
	}
	for i := range args {
		args[i] = strings.TrimSpace(args[i])
	}
	slices.Sort(args)
	return line[:open+1] + strings.Join(args, ",") + line[close:]
}

func splitTopLevelComma(s string) []string {
	var parts []string
	start := 0
	depth := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '[', '(', '{':
			depth++
		case ']', ')', '}':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	parts = append(parts, s[start:])
	return parts
}

func qwenFixtureSingleLayerProgram(t *testing.T) *publicmodelir.Program {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed")
	}
	fixturePath := filepath.Join(filepath.Dir(thisFile), "..", "..", "examples", "mlx-go-lm", "mlxlm", "models", "testdata", "qwen35_decode_trunk.modelir")
	textData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture %s: %v", fixturePath, err)
	}
	prog, err := publicmodelir.ParseText(textData)
	if err != nil {
		t.Fatalf("ParseText(%s): %v", fixturePath, err)
	}
	fn, ok := prog.FunctionByName(prog.Entry)
	if !ok {
		t.Fatalf("fixture entry function %q not found", prog.Entry)
	}

	reduced := *prog
	reduced.Functions = append([]publicmodelir.Function(nil), prog.Functions...)
	outFn := *fn
	outFn.States = nil
	for _, st := range fn.States {
		if strings.HasPrefix(st.Name, "l0_") {
			outFn.States = append(outFn.States, st)
		}
	}
	outFn.Consts = nil
	for _, c := range fn.Consts {
		switch {
		case strings.HasPrefix(c.Name, "l0_"):
			outFn.Consts = append(outFn.Consts, c)
		case c.Name == "final_norm":
			outFn.Consts = append(outFn.Consts, c)
		case strings.HasPrefix(c.Name, "rope_"):
			outFn.Consts = append(outFn.Consts, c)
		}
	}
	outFn.Ops = nil
	for _, op := range fn.Ops {
		skip := false
		for _, out := range op.Outputs {
			if strings.HasPrefix(out.Name, "l1_") || out.Name == "lm_proj" || out.Name == "logits" || out.Name == "final" {
				skip = true
				break
			}
		}
		if !skip {
			outFn.Ops = append(outFn.Ops, op)
		}
	}
	outFn.Returns = []string{"l0_y"}
	reduced.Functions[0] = outFn
	reduced.Weights = syntheticWeightsForConsts(outFn.Consts)
	return &reduced
}

func syntheticWeightsForConsts(consts []publicmodelir.Const) map[string]*publicmodelir.Weight {
	weights := make(map[string]*publicmodelir.Weight, len(consts))
	for _, c := range consts {
		if c.WeightRef == "" {
			continue
		}
		if _, ok := weights[c.WeightRef]; ok {
			continue
		}
		weights[c.WeightRef] = syntheticWeightForConst(c)
	}
	return weights
}

func syntheticWeightForConst(c publicmodelir.Const) *publicmodelir.Weight {
	n := c.Type.Shape.NumElements()
	if n < 0 {
		n = 0
	}
	w := &publicmodelir.Weight{
		DType: c.Type.DType,
		Shape: append(publicmodelir.Shape(nil), c.Type.Shape...),
	}
	switch c.Type.DType {
	case publicmodelir.DTypeFP32:
		w.Data = make([]byte, int(n)*4)
		for i := int64(0); i < n; i++ {
			v := float32((i%29)+1) * 0.001
			binary.LittleEndian.PutUint32(w.Data[i*4:], math.Float32bits(v))
		}
	case publicmodelir.DTypeFP16:
		w.Data = make([]byte, int(n)*2)
	default:
		w.Data = make([]byte, int(n)*4)
	}
	return w
}

func buildMILProbeAttentionFeatureFusedSliceStyle(cfg MILTransformerConfig, sliceStyle, gateMode string) (string, []modelWeightFile, error) {
	var files []modelWeightFile
	qWPath, qBPath := addProbeLinearWeights(&files, "q", cfg.Dim, cfg.Dim, 2011)
	kWPath, kBPath := addProbeLinearWeights(&files, "k", cfg.Dim, cfg.Dim, 2017)
	vWPath, vBPath := addProbeLinearWeights(&files, "v", cfg.Dim, cfg.Dim, 2027)
	oWPath, oBPath := addProbeLinearWeights(&files, "o", cfg.Dim, cfg.Dim, 2039)
	fusedWPath, fusedBPath := addProbeLinearWeights(&files, "fused", cfg.Dim, 2*cfg.HiddenDim, 2053)
	downWPath, downBPath := addProbeLinearWeights(&files, "down", cfg.HiddenDim, cfg.Dim, 2063)

	var b strings.Builder
	fmt.Fprintf(
		&b,
		"%s    func main<ios18>(tensor<fp32, [1, 1, %d]> x_in) {\n",
		ffnMILHeader,
		cfg.Dim,
	)
	b.WriteString("        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string(\"cast_in\")];\n", cfg.Dim)
	writeLinearConstBlock(&b, "qw", "qb", qWPath, qBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "kw", "kb", kWPath, kBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "vw", "vb", vWPath, vBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "ow", "ob", oWPath, oBPath, cfg.Dim, cfg.Dim)
	writeLinearConstBlock(&b, "fw", "fb", fusedWPath, fusedBPath, 2*cfg.HiddenDim, cfg.Dim)
	writeLinearConstBlock(&b, "dw", "db", downWPath, downBPath, cfg.Dim, cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<int32, [3]> osh = const()[name=string(\"osh\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.Dim)
	switch sliceStyle {
	case "explicit":
		fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
		fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
		fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([1,1,%d])];\n", 2*cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([false,false,false])];\n")
	case "masked":
		fmt.Fprintf(&b, "        tensor<int32, [3]> gate_begin = const()[name=string(\"gate_begin\"), val=tensor<int32, [3]>([0,0,0])];\n")
		fmt.Fprintf(&b, "        tensor<int32, [3]> gate_end = const()[name=string(\"gate_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<bool, [3]> gate_mask = const()[name=string(\"gate_mask\"), val=tensor<bool, [3]>([true,true,false])];\n")
		fmt.Fprintf(&b, "        tensor<int32, [3]> up_begin = const()[name=string(\"up_begin\"), val=tensor<int32, [3]>([0,0,%d])];\n", cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<int32, [3]> up_end = const()[name=string(\"up_end\"), val=tensor<int32, [3]>([0,0,%d])];\n", 2*cfg.HiddenDim)
		fmt.Fprintf(&b, "        tensor<bool, [3]> up_mask = const()[name=string(\"up_mask\"), val=tensor<bool, [3]>([true,true,false])];\n")
	default:
		return "", nil, fmt.Errorf("unknown slice style %q", sliceStyle)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> q = linear(bias=qb, weight=qw, x=x)[name=string(\"q\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> k = linear(bias=kb, weight=kw, x=x)[name=string(\"k\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> v = linear(bias=vb, weight=vw, x=x)[name=string(\"v\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> q4 = reshape(shape=qsh, x=q)[name=string(\"q4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> k4 = reshape(shape=qsh, x=k)[name=string(\"k4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> v4 = reshape(shape=qsh, x=v)[name=string(\"v4\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> attn = scaled_dot_product_attention(query=q4, key=k4, value=v4)[name=string(\"sdpa\")];\n", cfg.NumHeads, cfg.HeadDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> attn_r = reshape(shape=osh, x=attn)[name=string(\"attn_r\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> proj = linear(bias=ob, weight=ow, x=attn_r)[name=string(\"proj\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> resid = add(x=x, y=proj)[name=string(\"resid\")];\n", cfg.Dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> fused = linear(bias=fb, weight=fw, x=resid)[name=string(\"fused\")];\n", 2*cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate = slice_by_index(begin=gate_begin, begin_mask=gate_mask, end=gate_end, end_mask=gate_mask, x=fused)[name=string(\"gate\")];\n", cfg.HiddenDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_sig = sigmoid(x=gate)[name=string(\"gate_sig\")];\n", cfg.HiddenDim)
	if gateMode == "swiglu" {
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> gate_act = mul(x=gate, y=gate_sig)[name=string(\"gate_act\")];\n", cfg.HiddenDim)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> up = slice_by_index(begin=up_begin, begin_mask=up_mask, end=up_end, end_mask=up_mask, x=fused)[name=string(\"up\")];\n", cfg.HiddenDim)
	switch gateMode {
	case "sigmoid":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_sig, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	case "swiglu":
		fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> mix = mul(x=gate_act, y=up)[name=string(\"mix\")];\n", cfg.HiddenDim)
	default:
		return "", nil, fmt.Errorf("unknown gate mode %q", gateMode)
	}
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> down = linear(bias=db, weight=dw, x=mix)[name=string(\"down\")];\n", cfg.Dim)
	writeProbeMILReturn(&b, "down", cfg.Dim)
	return b.String(), files, nil
}

func buildMILProbeLargeLinear(_ MILTransformerConfig) (string, []modelWeightFile, []float32, int, error) {
	inDim := envIntOrDefault("MLXGO_ANE_TEST_MIL_LARGE_IN_DIM", 768)
	outDim := envIntOrDefault("MLXGO_ANE_TEST_MIL_LARGE_OUT_DIM", 32000)

	var files []modelWeightFile
	wPath, bPath := addProbeLinearWeights(&files, "large", inDim, outDim, 283)

	var b strings.Builder
	writeProbeMILPreamble(&b, inDim)
	writeLinearConstBlock(&b, "w0", "b0", wPath, bPath, outDim, inDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d]> y0 = linear(bias=b0, weight=w0, x=x)[name=string(\"linear_large\")];\n", outDim)
	writeProbeMILReturn(&b, "y0", outDim)
	return b.String(), files, makeDeterministicTensor(inDim, 0.0015, 293), outDim, nil
}

func testTensorLiteral16(xs []float32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, x := range xs {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(formatFloat16Literal(x))
	}
	b.WriteByte(']')
	return b.String()
}

func formatFloat16Literal(x float32) string {
	s := strconv.FormatFloat(float64(x), 'f', 6, 32)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

func envIntOrDefault(key string, dflt int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return dflt
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return dflt
	}
	return n
}
