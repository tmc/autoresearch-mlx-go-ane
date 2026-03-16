//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/tmc/apple/private/appleneuralengine"
)

const espressoGenTestEnv = "MLXGO_ANE_TEST_ESPRESSO_GEN"
const espressoParityTestEnv = "MLXGO_ANE_TEST_ESPRESSO_PARITY"

func TestEspressoGenFFN(t *testing.T) {
	if os.Getenv(espressoGenTestEnv) == "" {
		t.Skipf("set %s=1 to run Espresso generator integration test", espressoGenTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	const (
		defaultDim       = 64
		defaultHiddenDim = 256
	)
	dim := envIntWithDefault(t, "MLXGO_ANE_TEST_ESPRESSO_DIM", defaultDim)
	hiddenDim := envIntWithDefault(t, "MLXGO_ANE_TEST_ESPRESSO_HIDDEN_DIM", defaultHiddenDim)
	w1 := makeDeterministicTensor(hiddenDim*dim, 0.005, 29)
	w3 := makeDeterministicTensor(hiddenDim*dim, 0.004, 31)
	w2 := makeDeterministicTensor(dim*hiddenDim, 0.003, 37)

	dir, err := os.MkdirTemp("", "espresso_ffn_*.mlmodelc")
	if err != nil {
		t.Fatalf("mkdir temp model dir: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := GenerateFFNEspressoDir(dir, dim, hiddenDim, w1, w3, w2); err != nil {
		t.Fatalf("GenerateFFNEspressoDir: %v", err)
	}
	for _, name := range []string{"model.espresso.net", "model.espresso.shape", "model.espresso.weights"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("generated file %q missing: %v", name, err)
		}
	}

	model, err := CompileAndLoadEspresso(client, dir, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		if strings.Contains(err.Error(), "Cannot serialize ANEC_IR_repr") {
			t.Skipf(
				"Espresso translator rejected dim=%d hidden=%d on this runtime; try MLXGO_ANE_TEST_ESPRESSO_DIM=768 MLXGO_ANE_TEST_ESPRESSO_HIDDEN_DIM=3072: %v",
				dim,
				hiddenDim,
				err,
			)
		}
		t.Fatalf("CompileAndLoadEspresso: %v", err)
	}
	defer model.Close()

	input := makeDeterministicTensor(dim, 0.02, 23)
	var total time.Duration
	var out []float32
	for i := 0; i < 10; i++ {
		got, d, err := model.EvalSingleIO(context.Background(), input, dim, true)
		if err != nil {
			t.Fatalf("EvalSingleIO iter=%d: %v", i, err)
		}
		total += d
		out = got
	}
	if len(out) != dim {
		t.Fatalf("output len=%d want=%d", len(out), dim)
	}

	nonZero := false
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] not finite: %g", i, v)
		}
		if !nonZero && math.Abs(float64(v)) > 1e-7 {
			nonZero = true
		}
	}
	if !nonZero {
		t.Fatalf("output is all zeros (len=%d)", len(out))
	}
	avg := total / 10
	t.Logf("espresso generated FFN avg eval latency: %s", avg)
}

func TestEspressoGenFFNParity(t *testing.T) {
	if os.Getenv(espressoParityTestEnv) == "" {
		t.Skipf("set %s=1 to run Espresso parity integration test", espressoParityTestEnv)
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		dim          = 768
		hiddenDim    = 3072
		wantFileSize = 28422400
		wantBias0Off = 56
		wantW0Off    = 49408
		wantB1Off    = 9486592
		wantW1Off    = 9535744
		wantB2Off    = 18972928
		wantW2Off    = 18985216
	)

	w1 := makeDeterministicTensor(hiddenDim*dim, 0.0008, 29)
	w3 := makeDeterministicTensor(hiddenDim*dim, 0.0007, 31)
	w2 := makeDeterministicTensor(dim*hiddenDim, 0.0006, 37)

	dir, err := os.MkdirTemp("", "espresso_ffn_parity_*.mlmodelc")
	if err != nil {
		t.Fatalf("mkdir temp model dir: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := GenerateFFNEspressoDir(dir, dim, hiddenDim, w1, w3, w2); err != nil {
		t.Fatalf("GenerateFFNEspressoDir: %v", err)
	}

	weightsPath := filepath.Join(dir, "model.espresso.weights")
	gotWeights, err := os.ReadFile(weightsPath)
	if err != nil {
		t.Fatalf("read generated weights: %v", err)
	}
	if len(gotWeights) != wantFileSize {
		t.Fatalf("weights file size=%d want=%d", len(gotWeights), wantFileSize)
	}

	layout := computeEspressoFFNWeightLayout(dim, hiddenDim)
	if layout.Bias0Off != wantBias0Off {
		t.Fatalf("bias0 offset=%d want=%d", layout.Bias0Off, wantBias0Off)
	}
	if layout.Weight0Off != wantW0Off {
		t.Fatalf("weight0 offset=%d want=%d", layout.Weight0Off, wantW0Off)
	}
	if layout.Bias1Off != wantB1Off {
		t.Fatalf("bias1 offset=%d want=%d", layout.Bias1Off, wantB1Off)
	}
	if layout.Weight1Off != wantW1Off {
		t.Fatalf("weight1 offset=%d want=%d", layout.Weight1Off, wantW1Off)
	}
	if layout.Bias2Off != wantB2Off {
		t.Fatalf("bias2 offset=%d want=%d", layout.Bias2Off, wantB2Off)
	}
	if layout.Weight2Off != wantW2Off {
		t.Fatalf("weight2 offset=%d want=%d", layout.Weight2Off, wantW2Off)
	}
	if layout.TotalBytes != wantFileSize {
		t.Fatalf("layout total=%d want=%d", layout.TotalBytes, wantFileSize)
	}

	wantWords := espressoFFNHeaderWords(layout)
	for i, want := range wantWords {
		base := i * 8
		got := binary.LittleEndian.Uint64(gotWeights[base : base+8])
		if got != want {
			t.Fatalf("weights header word[%d]=%d want=%d", i, got, want)
		}
	}

	if binary.LittleEndian.Uint32(gotWeights[wantW0Off:wantW0Off+4]) != math.Float32bits(w1[0]) {
		t.Fatalf("weight0 payload start mismatch at offset=%d", wantW0Off)
	}
	if binary.LittleEndian.Uint32(gotWeights[wantW1Off:wantW1Off+4]) != math.Float32bits(w3[0]) {
		t.Fatalf("weight1 payload start mismatch at offset=%d", wantW1Off)
	}
	if binary.LittleEndian.Uint32(gotWeights[wantW2Off:wantW2Off+4]) != math.Float32bits(w2[0]) {
		t.Fatalf("weight2 payload start mismatch at offset=%d", wantW2Off)
	}

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	model, err := CompileAndLoadEspresso(client, dir, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		t.Fatalf("CompileAndLoadEspresso: %v", err)
	}
	defer model.Close()

	input := makeDeterministicTensor(dim, 0.02, 23)
	const iterations = 30
	durs := make([]time.Duration, 0, iterations)
	var out []float32
	for i := 0; i < iterations; i++ {
		got, d, err := model.EvalSingleIO(context.Background(), input, dim, true)
		if err != nil {
			t.Fatalf("EvalSingleIO iter=%d: %v", i, err)
		}
		out = got
		durs = append(durs, d)
	}
	if len(out) != dim {
		t.Fatalf("output len=%d want=%d", len(out), dim)
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] not finite: %g", i, v)
		}
	}
	sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
	p50 := durs[len(durs)/2]
	t.Logf("espresso parity FFN latency p50=%s (target <250µs)", p50)
	if p50 >= 250*time.Microsecond {
		t.Fatalf("p50 latency=%s exceeds target 250µs", p50)
	}
}

func BenchmarkGenerateFFNEspressoDir(b *testing.B) {
	const (
		dim       = 64
		hiddenDim = 256
	)
	w1 := makeDeterministicTensor(hiddenDim*dim, 0.005, 29)
	w3 := makeDeterministicTensor(hiddenDim*dim, 0.004, 31)
	w2 := makeDeterministicTensor(dim*hiddenDim, 0.003, 37)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dir := filepath.Join(b.TempDir(), fmt.Sprintf("ffn_%d.mlmodelc", i))
		if err := GenerateFFNEspressoDir(dir, dim, hiddenDim, w1, w3, w2); err != nil {
			b.Fatalf("GenerateFFNEspressoDir: %v", err)
		}
	}
}
