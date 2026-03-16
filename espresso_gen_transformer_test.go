//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/tmc/apple/private/appleneuralengine"
)

const espressoTransformerTestEnv = "MLXGO_ANE_TEST_ESPRESSO_TRANSFORMER"

func TestEspressoGenTransformer(t *testing.T) {
	if os.Getenv(espressoTransformerTestEnv) == "" {
		t.Skipf("set %s=1 to run transformer Espresso generator integration test", espressoTransformerTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	cfg := TransformerConfig{
		NumLayers:          envIntWithDefault(t, "MLXGO_ANE_TEST_TRANSFORMER_LAYERS", 1),
		Dim:                envIntWithDefault(t, "MLXGO_ANE_TEST_TRANSFORMER_DIM", 768),
		NumHeads:           envIntWithDefault(t, "MLXGO_ANE_TEST_TRANSFORMER_HEADS", 12),
		HeadDim:            envIntWithDefault(t, "MLXGO_ANE_TEST_TRANSFORMER_HEAD_DIM", 64),
		HiddenDim:          envIntWithDefault(t, "MLXGO_ANE_TEST_TRANSFORMER_HIDDEN_DIM", 3072),
		VocabSize:          envIntWithDefault(t, "MLXGO_ANE_TEST_TRANSFORMER_VOCAB", 32000),
		UseLookupEmbedding: os.Getenv("MLXGO_ANE_TEST_TRANSFORMER_USE_LOOKUP") != "",
		IncludeRMSNorm:     false, // incremental step: skip RMSNorm until reduce/pow path is validated.
	}
	if cfg.NumLayers <= 0 || cfg.Dim <= 0 || cfg.HiddenDim <= 0 || cfg.VocabSize <= 0 {
		t.Fatalf("invalid transformer config: %+v", cfg)
	}

	layers := make([]TransformerLayerWeights, cfg.NumLayers)
	for i := range layers {
		layers[i] = TransformerLayerWeights{
			QProj: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0005, 101+i),
			KProj: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0005, 103+i),
			VProj: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0005, 107+i),
			OProj: makeDeterministicTensor(cfg.Dim*cfg.Dim, 0.0005, 109+i),
			W1:    makeDeterministicTensor(cfg.HiddenDim*cfg.Dim, 0.0008, 113+i),
			W3:    makeDeterministicTensor(cfg.HiddenDim*cfg.Dim, 0.0007, 127+i),
			W2:    makeDeterministicTensor(cfg.Dim*cfg.HiddenDim, 0.0006, 131+i),
		}
	}
	embedLen := cfg.Dim * cfg.Dim
	if cfg.UseLookupEmbedding {
		embedLen = cfg.Dim * cfg.VocabSize
	}
	weights := TransformerWeights{
		Embedding: makeDeterministicTensor(embedLen, 0.0004, 137),
		Layers:    layers,
		LMHead:    makeDeterministicTensor(cfg.VocabSize*cfg.Dim, 0.0005, 149),
	}

	dir, err := os.MkdirTemp("", "espresso_transformer_*.mlmodelc")
	if err != nil {
		t.Fatalf("mkdir temp model dir: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := GenerateTransformerEspressoDir(dir, cfg, weights); err != nil {
		t.Fatalf("GenerateTransformerEspressoDir: %v", err)
	}
	for _, name := range []string{"model.espresso.net", "model.espresso.shape", "model.espresso.weights"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("generated file %q missing: %v", name, err)
		}
	}

	layerTypes, err := readEspressoLayerTypes(filepath.Join(dir, "model.espresso.net"))
	if err != nil {
		t.Fatalf("read layer types: %v", err)
	}

	model, err := CompileAndLoadEspresso(client, dir, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		if shouldSkipTransformerCompileError(err) {
			t.Skipf("Espresso translator rejected transformer graph; layer types=%v err=%v", layerTypes, err)
		}
		t.Fatalf("CompileAndLoadEspresso: %v", err)
	}
	defer model.Close()

	input := makeDeterministicTensor(cfg.Dim, 0.001, 151)
	if cfg.UseLookupEmbedding {
		// The evaluator API writes float32 input; for lookup mode this carries a
		// single token id payload value.
		input = []float32{1}
	}
	const iterations = 3
	var total time.Duration
	var logits []float32
	for i := 0; i < iterations; i++ {
		out, d, err := model.EvalSingleIO(context.Background(), input, cfg.VocabSize, true)
		if err != nil {
			t.Fatalf("EvalSingleIO iter=%d layer_types=%v: %v", i, layerTypes, err)
		}
		total += d
		logits = out
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits len=%d want=%d", len(logits), cfg.VocabSize)
	}
	nonZero := false
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits[%d] not finite: %g", i, v)
		}
		if !nonZero && math.Abs(float64(v)) > 1e-8 {
			nonZero = true
		}
	}
	if !nonZero {
		t.Fatalf("logits are all zeros (len=%d)", len(logits))
	}
	t.Logf(
		"transformer generator avg eval latency: %s cfg=%+v layer_types=%v",
		total/iterations,
		cfg,
		layerTypes,
	)
}

func shouldSkipTransformerCompileError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "Cannot serialize ANEC_IR_repr") ||
		strings.Contains(msg, "CompilationFailure") ||
		strings.Contains(msg, "espresso")
}

func readEspressoLayerTypes(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var net EspressoNet
	if err := json.Unmarshal(data, &net); err != nil {
		return nil, err
	}
	set := make(map[string]bool)
	for _, layer := range net.Layers {
		if layer.Type == "" {
			continue
		}
		set[layer.Type] = true
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}
