//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/private/appleneuralengine"
	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
)

func TestANEDraftModelTinyQwenAttentionInitMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run ANE draft tiny qwen attention init matrix", milTransformerFeatureMatrixEnv)
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
		t.Fatalf("validate base cfg: %v", err)
	}

	type initCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []initCase{
		{name: "baseline_norms", cfg: baseCfg},
		{name: "dynamic_rope_and_gate", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
		})},
		{name: "dynamic_rope_and_gate_no_input_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableInputNormOps = true
		})},
		{name: "dynamic_rope_and_gate_no_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableQKNormOps = true
		})},
		{name: "dynamic_rope_and_gate_no_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableFinalNormOp = true
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
			model, err := NewANEDraftModelFromMILTransformer(
				tc.cfg,
				buildTransformerFeatureWeights(tc.cfg),
				tc.cfg.Dim,
				64,
				tc.cfg.Dim,
				nil,
			)
			if err != nil {
				t.Logf("%s init failed: %v", tc.name, err)
				return
			}
			t.Logf("%s init succeeded", tc.name)
			model.Close()
		})
	}
}

func TestANEDraftModelTinyQwenAttentionEvalMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run ANE draft tiny qwen attention eval matrix", milTransformerFeatureMatrixEnv)
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
		t.Fatalf("validate base cfg: %v", err)
	}

	type evalCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []evalCase{
		{name: "baseline_stateless", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.KVCacheState = false
			cfg.KVCacheMaxLen = 0
		})},
		{name: "baseline_norms", cfg: baseCfg},
		{name: "baseline_norms_identity", cfg: baseCfg},
		{name: "dynamic_rope_and_gate_no_input_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableInputNormOps = true
		})},
		{name: "dynamic_rope_and_gate_no_qk_norms", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableQKNormOps = true
		})},
		{name: "dynamic_rope_and_gate_no_final_norm", cfg: withTransformerFeature(baseCfg, func(cfg *MILTransformerConfig) {
			cfg.DynamicRoPEInputs = true
			cfg.AttentionOutputGate = true
			cfg.DisableFinalNormOp = true
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
			if tc.name == "baseline_norms_identity" {
				weights = testTransformerWeightsIdentity(tc.cfg)
			}
			model, err := NewANEDraftModelFromMILTransformer(
				tc.cfg,
				weights,
				tc.cfg.Dim,
				64,
				tc.cfg.Dim,
				nil,
			)
			if err != nil {
				t.Logf("%s init failed: %v", tc.name, err)
				return
			}
			defer model.Close()

			input := make([]float32, tc.cfg.Dim)
			for i := range input {
				input[i] = 0.01 * float32(i+1)
			}
			out, err := model.EvalToken(input)
			if err != nil {
				t.Logf("%s eval failed: %v", tc.name, err)
				return
			}
			if len(out) != tc.cfg.Dim {
				t.Fatalf("%s len(out)=%d want %d", tc.name, len(out), tc.cfg.Dim)
			}
			t.Logf("%s eval succeeded", tc.name)
		})
	}
}

func TestANEDraftModelTinyQwenFullTrunkInitMatrix(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run ANE draft tiny qwen full trunk init matrix", milTransformerFeatureMatrixEnv)
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
		t.Fatalf("validate base cfg: %v", err)
	}

	type initCase struct {
		name string
		cfg  MILTransformerConfig
	}
	cases := []initCase{
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
			model, err := NewANEDraftModelFromMILTransformer(
				tc.cfg,
				buildTransformerFeatureWeights(tc.cfg),
				tc.cfg.Dim,
				64,
				tc.cfg.Dim,
				nil,
			)
			if err != nil {
				t.Logf("%s init failed: %v", tc.name, err)
				return
			}
			t.Logf("%s init succeeded", tc.name)
			model.Close()
		})
	}
}

func TestANEDraftModelTinyQwenAttentionCompareGenericRuntime(t *testing.T) {
	if os.Getenv(milTransformerFeatureMatrixEnv) == "" {
		t.Skipf("set %s=1 to run ANE draft tiny qwen attention runtime comparison", milTransformerFeatureMatrixEnv)
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
		DisableNormOps:     false,
		IncludeLMHead:      false,
		AttentionMaskInput: false,
	}
	if err := validateMILTransformerConfig(cfg); err != nil {
		t.Fatalf("validate cfg: %v", err)
	}

	model, err := NewANEDraftModelFromMILTransformer(
		cfg,
		testTransformerWeightsIdentity(cfg),
		cfg.Dim,
		64,
		cfg.Dim,
		nil,
	)
	if err != nil {
		t.Fatalf("NewANEDraftModelFromMILTransformer: %v", err)
	}
	defer model.Close()

	input := make([]float32, cfg.Dim)
	for i := range input {
		input[i] = 0.01 * float32(i+1)
	}

	draftOut, draftErr := model.EvalToken(input)
	target := compiledModelIRRuntimeTarget{inMemoryModel: model.inMemoryMIL}
	genericOuts, genericErr := evalModelIRRuntimeSteps(context.Background(), target, [][]float32{input}, cfg.Dim)

	if genericErr != nil {
		t.Logf("generic runtime eval failed: %v", genericErr)
	}
	if draftErr != nil {
		t.Logf("draft eval failed: %v", draftErr)
	}
	if draftErr == nil && len(draftOut) != cfg.Dim {
		t.Fatalf("draft len(out)=%d want %d", len(draftOut), cfg.Dim)
	}
	if genericErr == nil {
		if len(genericOuts) != 1 {
			t.Fatalf("generic runtime outputs=%d want 1", len(genericOuts))
		}
		if len(genericOuts[0]) != cfg.Dim {
			t.Fatalf("generic runtime len(out)=%d want %d", len(genericOuts[0]), cfg.Dim)
		}
	}

	switch {
	case draftErr == nil && genericErr == nil:
		t.Log("draft and generic runtime eval both succeeded")
	case draftErr != nil && genericErr == nil:
		t.Fatalf("draft eval failed while generic runtime eval succeeded: draft=%v", draftErr)
	case draftErr == nil && genericErr != nil:
		t.Fatalf("draft eval succeeded while generic runtime eval failed: generic=%v", genericErr)
	default:
		t.Log("draft and generic runtime eval both failed")
	}
}

func TestANEDraftModelStateSyncProbe(t *testing.T) {
	if os.Getenv("MLXGO_ANE_TEST_DRAFT_STATE_SYNC_PROBE") == "" {
		t.Skip("set MLXGO_ANE_TEST_DRAFT_STATE_SYNC_PROBE=1 to run ANE draft state sync probe")
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
		DisableNormOps:     false,
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
	input := make([]float32, cfg.Dim)
	for i := range input {
		input[i] = 0.01 * float32(i+1)
	}
	modes := []string{
		"",
		"copy_model_to_inmem",
		"copy_inmem_to_model",
		"refresh_model_attrs",
		"copy_and_refresh",
	}
	for _, mode := range modes {
		mode := mode
		name := mode
		if name == "" {
			name = "none"
		}
		t.Run(name, func(t *testing.T) {
			model, err := buildModelFromMILTextWithDescriptorFallback(
				"ane draft state sync probe "+name,
				milText,
				fromANEModelWeightFiles(files),
			)
			if err != nil {
				t.Fatalf("buildModelFromMILTextWithDescriptorFallback(%s): %v", name, err)
			}
			defer unloadMILProbeModel(t, name, model)

			before := snapshotModelState(model)
			if mode != "" {
				if err := applyModelStateSync(mode, model); err != nil {
					t.Logf("state sync mode=%s error=%v before=%+v", mode, err, before)
					return
				}
			}
			after := snapshotModelState(model)
			outs, err := evalModelIRRuntimeSteps(
				context.Background(),
				compiledModelIRRuntimeTarget{inMemoryModel: model},
				[][]float32{input},
				cfg.Dim,
			)
			if err != nil {
				t.Logf("state sync mode=%s eval error=%v before=%+v after=%+v", name, err, before, after)
				return
			}
			if len(outs) != 1 || len(outs[0]) != cfg.Dim {
				t.Fatalf("state sync mode=%s outputs=%d len0=%d want 1/%d", name, len(outs), len(outs[0]), cfg.Dim)
			}
			t.Logf("state sync mode=%s eval ok before=%+v after=%+v", name, before, after)
		})
	}
}

func TestANEDraftModelSharedClientPrimeProbe(t *testing.T) {
	if os.Getenv("MLXGO_ANE_TEST_DRAFT_SHARED_CLIENT_PRIME") == "" {
		t.Skip("set MLXGO_ANE_TEST_DRAFT_SHARED_CLIENT_PRIME=1 to run ANE draft shared-client prime probe")
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
	model, err := buildModelFromMILTextWithDescriptorFallback(
		"ane draft shared client prime probe",
		milText,
		fromANEModelWeightFiles(files),
	)
	if err != nil {
		t.Fatalf("buildModelFromMILTextWithDescriptorFallback: %v", err)
	}
	defer unloadMILProbeModel(t, "shared_client_prime_probe", model)

	input := make([]float32, cfg.Dim)
	for i := range input {
		input[i] = 0.01 * float32(i+1)
	}
	_, beforeErr := evalModelIRRuntimeSteps(
		context.Background(),
		compiledModelIRRuntimeTarget{inMemoryModel: model},
		[][]float32{input},
		cfg.Dim,
	)
	t.Logf("before prime err=%v state=%+v", beforeErr, snapshotModelState(model))

	shared := model.SharedConnection()
	base := model.Model()
	if shared.GetID() == 0 || base.GetID() == 0 {
		t.Skipf("shared/base unavailable shared=%#x base=%#x", shared.GetID(), base.GetID())
	}

	opts := foundation.NewMutableDictionaryWithCapacity(0)
	strategies := []struct {
		name string
		run  func() error
	}{
		{
			name: "load_only",
			run: func() error {
				_, err := shared.LoadModelOptionsQosError(base, opts, defaultANEQoS)
				return err
			},
		},
		{
			name: "compile_and_load",
			run: func() error {
				if _, err := shared.CompileModelOptionsQosError(base, opts, defaultANEQoS); err != nil {
					return err
				}
				_, err := shared.LoadModelOptionsQosError(base, opts, defaultANEQoS)
				return err
			},
		},
	}
	for _, tc := range strategies {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err != nil {
				t.Logf("prime strategy=%s setup error=%v", tc.name, err)
				return
			}
			defer func() {
				_, _ = shared.UnloadModelOptionsQosError(base, opts, defaultANEQoS)
			}()
			_, afterErr := evalModelIRRuntimeSteps(
				context.Background(),
				compiledModelIRRuntimeTarget{inMemoryModel: model},
				[][]float32{input},
				cfg.Dim,
			)
			t.Logf("prime strategy=%s after err=%v state=%+v", tc.name, afterErr, snapshotModelState(model))
		})
	}
}

func BenchmarkANEDraftGenerate(b *testing.B) {
	if os.Getenv("MLXGO_ANE_BENCH_DRAFT") == "" {
		b.Skip("set MLXGO_ANE_BENCH_DRAFT=1 to run ANE draft benchmark")
	}

	modelPath := os.Getenv("MLXGO_ANE_DRAFT_MODEL_PATH")
	if modelPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			b.Skipf("resolve home dir: %v", err)
		}
		modelPath = filepath.Join(home, "go", "src", "github.com", "maderix", "ANE", "testdata", "ffn", "draft125m_ffn.mlmodelc")
	}
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("draft model path unavailable: %s (%v)", modelPath, err)
	}

	hiddenDim := envInt("MLXGO_ANE_DRAFT_HIDDEN_DIM", 768)
	vocabSize := envInt("MLXGO_ANE_DRAFT_VOCAB_SIZE", 32000)
	k := envInt("MLXGO_ANE_DRAFT_K", 8)
	if hiddenDim <= 0 || vocabSize <= 0 || k <= 0 {
		b.Fatalf("invalid draft bench params hidden=%d vocab=%d k=%d", hiddenDim, vocabSize, k)
	}

	embeddings := makeDraftEmbeddings(vocabSize, hiddenDim)
	draft, err := NewANEDraftModel(modelPath, hiddenDim, vocabSize, embeddings)
	if err != nil {
		b.Fatalf("NewANEDraftModel: %v", err)
	}
	defer draft.Close()

	input := append([]float32(nil), embeddings[:hiddenDim]...)
	totalTokens := b.N * k

	b.ReportAllocs()
	start := time.Now()
	var aneDur time.Duration
	var cpuDur time.Duration
	var genDur time.Duration
	for i := 0; i < b.N; i++ {
		tokenIDs, allLogits, timing, err := draft.GenerateDraftWithTiming(input, k)
		if err != nil {
			b.Fatalf("GenerateDraft: %v", err)
		}
		if len(tokenIDs) != k || len(allLogits) != k {
			b.Fatalf("GenerateDraft sizes ids=%d logits=%d want=%d", len(tokenIDs), len(allLogits), k)
		}
		last := tokenIDs[len(tokenIDs)-1]
		rowStart := last * hiddenDim
		rowEnd := rowStart + hiddenDim
		copy(input, embeddings[rowStart:rowEnd])
		aneDur += timing.ANE
		cpuDur += timing.CPU
		genDur += timing.Total
	}
	elapsed := time.Since(start)
	if totalTokens > 0 {
		b.ReportMetric(float64(elapsed.Microseconds())/float64(totalTokens), "per_token_us")
		b.ReportMetric(float64(aneDur.Microseconds())/float64(totalTokens), "ane_us/token")
		b.ReportMetric(float64(cpuDur.Microseconds())/float64(totalTokens), "cpu_us/token")
		b.ReportMetric(float64(genDur.Microseconds())/float64(totalTokens), "gen_us/token")
		b.ReportMetric(float64(totalTokens)/elapsed.Seconds(), "tokens_per_sec")
	}
	b.ReportMetric(float64(elapsed.Microseconds())/float64(b.N), "total_us/op")
}

func makeDraftEmbeddings(vocabSize, hiddenDim int) []float32 {
	n := vocabSize * hiddenDim
	emb := make([]float32, n)
	for tok := 0; tok < vocabSize; tok++ {
		row := tok * hiddenDim
		base := float32((tok%53)-26) * 0.001
		for d := 0; d < hiddenDim; d++ {
			emb[row+d] = base + float32((d%29)-14)*0.0004
		}
	}
	return emb
}

func TestMergeDraftWithTargetGreedy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		draft       []int
		logits      [][]float32
		want        []int
		wantErrPart string
	}{
		{
			name:   "mismatch at first position",
			draft:  []int{3, 4, 5},
			logits: [][]float32{{0, 9, 0}, {0, 0, 8}, {0, 0, 7}, {0, 6, 0}},
			want:   []int{1},
		},
		{
			name:   "mismatch in middle",
			draft:  []int{1, 2, 3},
			logits: [][]float32{{0, 5, 0, 0}, {0, 0, 0, 9}, {0, 0, 8, 0}, {0, 0, 0, 7}},
			want:   []int{1, 3},
		},
		{
			name:   "all match plus one extra",
			draft:  []int{1, 2},
			logits: [][]float32{{0, 7, 0, 0}, {0, 0, 9, 0}, {0, 0, 0, 8}},
			want:   []int{1, 2, 3},
		},
		{
			name:        "empty draft rejected",
			draft:       nil,
			logits:      [][]float32{{1}},
			wantErrPart: "empty draft token sequence",
		},
		{
			name:        "insufficient target logits rejected",
			draft:       []int{1, 2, 3},
			logits:      [][]float32{{1}, {1}, {1}},
			wantErrPart: "target logits length",
		},
		{
			name:        "empty target row rejected",
			draft:       []int{1},
			logits:      [][]float32{{}, {1}},
			wantErrPart: "target logits[0] is empty",
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := mergeDraftWithTargetGreedy(tc.draft, tc.logits)
			if tc.wantErrPart != "" {
				if err == nil {
					t.Fatalf("mergeDraftWithTargetGreedy error=nil want contains %q", tc.wantErrPart)
				}
				if !strings.Contains(err.Error(), tc.wantErrPart) {
					t.Fatalf("mergeDraftWithTargetGreedy err=%q want contains %q", err, tc.wantErrPart)
				}
				return
			}
			if err != nil {
				t.Fatalf("mergeDraftWithTargetGreedy: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("mergeDraftWithTargetGreedy=%v want=%v", got, tc.want)
			}
		})
	}
}

func TestANEDraftEmbeddingForToken(t *testing.T) {
	t.Parallel()

	model := &ANEDraftModel{
		hiddenDim:  4,
		vocabSize:  3,
		embeddings: []float32{10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33},
	}
	got, err := model.EmbeddingForToken(1)
	if err != nil {
		t.Fatalf("embeddingForToken: %v", err)
	}
	want := []float32{20, 21, 22, 23}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("embeddingForToken=%v want=%v", got, want)
	}
	got[0] = -1
	if model.embeddings[4] != 20 {
		t.Fatalf("embeddingForToken returned alias into backing embeddings")
	}
}

func TestANEDraftEmbeddingForTokenRejectsOutOfRange(t *testing.T) {
	t.Parallel()
	model := &ANEDraftModel{
		hiddenDim:  2,
		vocabSize:  2,
		embeddings: []float32{1, 2, 3, 4},
	}
	if _, err := model.EmbeddingForToken(-1); err == nil {
		t.Fatal("embeddingForToken(-1) error=nil")
	}
	if _, err := model.EmbeddingForToken(2); err == nil {
		t.Fatal("embeddingForToken(2) error=nil")
	}
}

func TestANEDraftModelSharedEventAccessorsNil(t *testing.T) {
	t.Parallel()

	var model *ANEDraftModel
	if model.WaitEvent() != nil {
		t.Fatal("nil model WaitEvent != nil")
	}
	if model.SignalEvent() != nil {
		t.Fatal("nil model SignalEvent != nil")
	}
	if model.InputSurface() != nil {
		t.Fatal("nil model InputSurface != nil")
	}
	if model.PosCosSurface() != nil {
		t.Fatal("nil model PosCosSurface != nil")
	}
	if model.PosSinSurface() != nil {
		t.Fatal("nil model PosSinSurface != nil")
	}
	if model.WaitValue() != 0 {
		t.Fatal("nil model WaitValue != 0")
	}
	if model.SignalValue() != 0 {
		t.Fatal("nil model SignalValue != 0")
	}
	if model.OutputSurface() != nil {
		t.Fatal("nil model OutputSurface != nil")
	}
	if _, err := model.NewDefaultOutputMetalBufferBinding(); err == nil {
		t.Fatal("nil model NewDefaultOutputMetalBufferBinding error=nil")
	}
	if _, err := model.NewDefaultWaitMetalSharedEvent(); err == nil {
		t.Fatal("nil model NewDefaultWaitMetalSharedEvent error=nil")
	}
	if _, err := model.NewDefaultSignalMetalSharedEvent(); err == nil {
		t.Fatal("nil model NewDefaultSignalMetalSharedEvent error=nil")
	}
	if err := model.EvalPreparedSurface(context.Background()); err == nil {
		t.Fatal("nil model EvalPreparedSurface error=nil")
	}

	model = &ANEDraftModel{}
	if model.WaitEvent() != nil {
		t.Fatal("empty model WaitEvent != nil")
	}
	if model.SignalEvent() != nil {
		t.Fatal("empty model SignalEvent != nil")
	}
	if model.InputSurface() != nil {
		t.Fatal("empty model InputSurface != nil")
	}
	if model.PosCosSurface() != nil {
		t.Fatal("empty model PosCosSurface != nil")
	}
	if model.PosSinSurface() != nil {
		t.Fatal("empty model PosSinSurface != nil")
	}
	if model.WaitValue() != 0 {
		t.Fatal("empty model WaitValue != 0")
	}
	if model.SignalValue() != 0 {
		t.Fatal("empty model SignalValue != 0")
	}
	if model.OutputSurface() != nil {
		t.Fatal("empty model OutputSurface != nil")
	}
	if _, err := model.NewDefaultOutputMetalBufferBinding(); err == nil {
		t.Fatal("empty model NewDefaultOutputMetalBufferBinding error=nil")
	}
	if _, err := model.NewDefaultWaitMetalSharedEvent(); err == nil {
		t.Fatal("empty model NewDefaultWaitMetalSharedEvent error=nil")
	}
	if _, err := model.NewDefaultSignalMetalSharedEvent(); err == nil {
		t.Fatal("empty model NewDefaultSignalMetalSharedEvent error=nil")
	}
	if err := model.EvalPreparedSurface(context.Background()); err == nil {
		t.Fatal("empty model EvalPreparedSurface error=nil")
	}
}

func TestANEDraftModelRoPESliceTracksDecodePosition(t *testing.T) {
	t.Parallel()
	model := &ANEDraftModel{}
	cos := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	sin := []float32{
		-1, -2, -3, -4,
		-5, -6, -7, -8,
	}
	if err := model.SetRoPETables(cos, sin, 4, 2); err != nil {
		t.Fatalf("SetRoPETables: %v", err)
	}
	// Without KV state, position defaults to 0.
	gotCos, gotSin, err := model.CurrentRoPESlice()
	if err != nil {
		t.Fatalf("CurrentRoPESlice: %v", err)
	}
	if !reflect.DeepEqual(gotCos, []float32{1, 2, 3, 4}) || !reflect.DeepEqual(gotSin, []float32{-1, -2, -3, -4}) {
		t.Fatalf("row0 mismatch cos=%v sin=%v", gotCos, gotSin)
	}

	if err := model.AdvanceDecodePosition(); err != nil {
		t.Fatalf("AdvanceDecodePosition: %v", err)
	}
	gotCos, gotSin, err = model.CurrentRoPESlice()
	if err != nil {
		t.Fatalf("CurrentRoPESlice at pos1: %v", err)
	}
	if !reflect.DeepEqual(gotCos, []float32{5, 6, 7, 8}) || !reflect.DeepEqual(gotSin, []float32{-5, -6, -7, -8}) {
		t.Fatalf("row1 mismatch cos=%v sin=%v", gotCos, gotSin)
	}
	model.Close()
}

func TestANEDraftModelRewindDecodePosition(t *testing.T) {
	t.Parallel()
	model := &ANEDraftModel{}
	cos := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	sin := []float32{
		-1, -2, -3, -4,
		-5, -6, -7, -8,
		-9, -10, -11, -12,
	}
	if err := model.SetRoPETables(cos, sin, 4, 3); err != nil {
		t.Fatalf("SetRoPETables: %v", err)
	}
	if err := model.AdvanceDecodePosition(); err != nil {
		t.Fatalf("AdvanceDecodePosition #1: %v", err)
	}
	if err := model.AdvanceDecodePosition(); err != nil {
		t.Fatalf("AdvanceDecodePosition #2: %v", err)
	}
	if got := model.DecodePosition(); got != 2 {
		t.Fatalf("DecodePosition=%d want=2", got)
	}
	if err := model.RewindDecodePosition(1); err != nil {
		t.Fatalf("RewindDecodePosition: %v", err)
	}
	if got := model.DecodePosition(); got != 1 {
		t.Fatalf("DecodePosition after rewind=%d want=1", got)
	}
	gotCos, gotSin, err := model.CurrentRoPESlice()
	if err != nil {
		t.Fatalf("CurrentRoPESlice after rewind: %v", err)
	}
	if !reflect.DeepEqual(gotCos, []float32{5, 6, 7, 8}) || !reflect.DeepEqual(gotSin, []float32{-5, -6, -7, -8}) {
		t.Fatalf("rewound row mismatch cos=%v sin=%v", gotCos, gotSin)
	}
	model.Close()
}

func TestANEDraftModelRewindDecodePositionRejectsStatefulMIL(t *testing.T) {
	t.Parallel()
	model := &ANEDraftModel{
		statefulMIL: &statefulMILRuntime{},
	}
	if err := model.RewindDecodePosition(1); err == nil {
		t.Fatal("RewindDecodePosition error=nil for stateful MIL runtime")
	}
}

func TestANEDraftModelSnapshotRestoreDecodeState(t *testing.T) {
	t.Parallel()

	layout := compiledTensorLayout{
		Channels:    2,
		Width:       4,
		Height:      3,
		ElemSize:    4,
		RowStride:   64,
		PlaneStride: 3 * 64,
		Name:        "state",
		Symbol:      "state",
		SymbolIndex: 0,
	}
	state0, err := newIOSurfaceFloat32WithLayout(layout)
	if err != nil {
		t.Fatalf("newIOSurfaceFloat32WithLayout(state0): %v", err)
	}
	defer state0.Close()
	state1, err := newIOSurfaceFloat32WithLayout(layout)
	if err != nil {
		t.Fatalf("newIOSurfaceFloat32WithLayout(state1): %v", err)
	}
	defer state1.Close()

	want0 := make([]float32, layout.logicalCount())
	want1 := make([]float32, layout.logicalCount())
	for i := range want0 {
		want0[i] = float32(i + 1)
		want1[i] = float32(100 + i + 1)
	}
	if err := state0.Write(want0); err != nil {
		t.Fatalf("state0.Write: %v", err)
	}
	if err := state1.Write(want1); err != nil {
		t.Fatalf("state1.Write: %v", err)
	}

	model := &ANEDraftModel{
		statefulMIL: &statefulMILRuntime{label: "test"},
		milState:    []*IOSurfaceFloat32{state0, state1},
		decodePos:   3,
	}
	snap, err := model.SnapshotDecodeState()
	if err != nil {
		t.Fatalf("SnapshotDecodeState: %v", err)
	}
	if got := snap.decodePos; got != 3 {
		t.Fatalf("snapshot decodePos=%d want=3", got)
	}

	if err := state0.Fill(0); err != nil {
		t.Fatalf("state0.Fill: %v", err)
	}
	if err := state1.Fill(0); err != nil {
		t.Fatalf("state1.Fill: %v", err)
	}
	model.decodePos = 0

	if err := model.RestoreDecodeState(snap); err != nil {
		t.Fatalf("RestoreDecodeState: %v", err)
	}
	if got := model.DecodePosition(); got != 3 {
		t.Fatalf("DecodePosition after restore=%d want=3", got)
	}
	got0, err := state0.Read()
	if err != nil {
		t.Fatalf("state0.Read: %v", err)
	}
	got1, err := state1.Read()
	if err != nil {
		t.Fatalf("state1.Read: %v", err)
	}
	if !reflect.DeepEqual(got0, want0) {
		t.Fatalf("state0 restore mismatch got=%v want=%v", got0, want0)
	}
	if !reflect.DeepEqual(got1, want1) {
		t.Fatalf("state1 restore mismatch got=%v want=%v", got1, want1)
	}
}

func TestANEDraftModelRestoreStatefulMILState(t *testing.T) {
	t.Parallel()

	layout := compiledTensorLayout{
		Channels:    2,
		Width:       4,
		Height:      3,
		ElemSize:    4,
		RowStride:   64,
		PlaneStride: 3 * 64,
		Name:        "state",
		Symbol:      "state",
		SymbolIndex: 0,
	}
	state0, err := newIOSurfaceFloat32WithLayout(layout)
	if err != nil {
		t.Fatalf("newIOSurfaceFloat32WithLayout(state0): %v", err)
	}
	defer state0.Close()
	state1, err := newIOSurfaceFloat32WithLayout(layout)
	if err != nil {
		t.Fatalf("newIOSurfaceFloat32WithLayout(state1): %v", err)
	}
	defer state1.Close()

	want0 := make([]float32, layout.logicalCount())
	want1 := make([]float32, layout.logicalCount())
	for i := range want0 {
		want0[i] = float32(i + 11)
		want1[i] = float32(200 + i)
	}

	model := &ANEDraftModel{
		statefulMIL: &statefulMILRuntime{label: "test"},
		milState:    []*IOSurfaceFloat32{state0, state1},
	}
	if err := model.RestoreStatefulMILState(7, [][]float32{want0, want1}); err != nil {
		t.Fatalf("RestoreStatefulMILState: %v", err)
	}
	if got := model.DecodePosition(); got != 7 {
		t.Fatalf("DecodePosition=%d want=7", got)
	}
	got0, err := state0.Read()
	if err != nil {
		t.Fatalf("state0.Read: %v", err)
	}
	got1, err := state1.Read()
	if err != nil {
		t.Fatalf("state1.Read: %v", err)
	}
	if !reflect.DeepEqual(got0, want0) {
		t.Fatalf("state0 restore mismatch got=%v want=%v", got0, want0)
	}
	if !reflect.DeepEqual(got1, want1) {
		t.Fatalf("state1 restore mismatch got=%v want=%v", got1, want1)
	}
}

func TestANEDraftModelConfigureKVStateRejectsBoundAttentionDecodePlan(t *testing.T) {
	t.Parallel()
	model := &ANEDraftModel{
		milMultiPlan: &MultiSurfaceEvalPlan{},
		milKCurr:     &IOSurfaceFloat32{},
	}
	if err := model.ConfigureKVState(1, 1, 1, 8); err == nil {
		t.Fatal("ConfigureKVState error=nil for bound attention decode plan")
	}
}

func TestNewANEDraftModelFromMILTransformerRejectsStatefulMaskInput(t *testing.T) {
	t.Parallel()

	cfg := testTransformerConfig(1)
	cfg.Dim = 8
	cfg.NumHeads = 2
	cfg.HeadDim = 4
	cfg.HiddenDim = 16
	cfg.MaxSeqLen = 8
	cfg.KVCacheState = true
	cfg.AttentionMaskInput = true

	_, err := NewANEDraftModelFromMILTransformer(cfg, testTransformerWeights(cfg), cfg.Dim, 32, cfg.Dim, nil)
	if err == nil || !strings.Contains(err.Error(), "stateful attention mask input is not supported") {
		t.Fatalf("NewANEDraftModelFromMILTransformer error=%v want unsupported stateful attention mask input", err)
	}
}

func TestANEDraftModelStatefulMILResetSmoke(t *testing.T) {
	if os.Getenv("MLXGO_ANE_TEST_DRAFT_STATEFUL_RESET") == "" {
		t.Skip("set MLXGO_ANE_TEST_DRAFT_STATEFUL_RESET=1 to run stateful draft reset smoke test")
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
	cfg.SkipFFN = true
	weights := testTransformerWeightsIdentity(cfg)

	stateful, err := NewANEDraftModelFromMILTransformer(cfg, weights, cfg.Dim, 32, cfg.Dim, nil)
	if err != nil {
		if strings.Contains(err.Error(), "ANECCompile() FAILED") {
			t.Skipf("stateful draft in-memory compile unavailable on this host: %v", err)
		}
		t.Fatalf("NewANEDraftModelFromMILTransformer(stateful): %v", err)
	}
	defer stateful.Close()

	statelessCfg := cfg
	statelessCfg.KVCacheState = false
	stateless, err := NewANEDraftModelFromMILTransformer(statelessCfg, weights, cfg.Dim, 32, cfg.Dim, nil)
	if err != nil {
		t.Fatalf("NewANEDraftModelFromMILTransformer(stateless): %v", err)
	}
	defer stateless.Close()

	step1 := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	step2 := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	if _, err := stateful.evalTokenLocked(step1); err != nil {
		t.Fatalf("stateful eval step1: %v", err)
	}
	stateful2, err := stateful.evalTokenLocked(step2)
	if err != nil {
		t.Fatalf("stateful eval step2: %v", err)
	}
	if err := stateful.Reset(); err != nil {
		t.Fatalf("stateful Reset: %v", err)
	}
	afterReset, err := stateful.evalTokenLocked(step2)
	if err != nil {
		t.Fatalf("stateful eval after reset: %v", err)
	}
	stateless2, err := stateless.evalTokenLocked(step2)
	if err != nil {
		t.Fatalf("stateless eval step2: %v", err)
	}
	if nearlyEqualFloat32s(stateful2, afterReset, 1e-3) {
		t.Fatal("stateful output after reset unexpectedly matched stale stateful output")
	}
	if !nearlyEqualFloat32s(afterReset, stateless2, 1e-3) {
		t.Fatal("stateful output after reset did not match stateless baseline")
	}
}
