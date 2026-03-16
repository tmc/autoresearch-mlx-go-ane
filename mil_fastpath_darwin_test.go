//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

const milFastPathTestEnv = "MLXGO_ANE_TEST_MIL_FASTPATH"

func TestMILModelFastPath(t *testing.T) {
	t.Helper()
	if os.Getenv(milFastPathTestEnv) == "" {
		t.Skipf("set %s=1 to run MIL compile/load fast-path test", milFastPathTestEnv)
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
		batch  = 1
		inDim  = 64
		outDim = 64
	)
	milText, err := linearMILText(batch, inDim, outDim)
	if err != nil {
		t.Fatalf("linearMILText: %v", err)
	}

	weights := make([]float32, outDim*inDim)
	for o := 0; o < outDim; o++ {
		for i := 0; i < inDim; i++ {
			if o == i {
				weights[o*inDim+i] = 1
			} else {
				weights[o*inDim+i] = float32(((o+i)%11)-5) * 0.01
			}
		}
	}
	weightBlob, err := buildLinearWeightsBlob(weights, outDim, inDim)
	if err != nil {
		t.Fatalf("buildLinearWeightsBlob: %v", err)
	}

	model, err := CompileAndLoadMIL(client, milText, weightBlob, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		t.Fatalf("CompileAndLoadMIL: %v", err)
	}
	defer model.Close()

	input := make([]float32, inDim)
	for i := range input {
		input[i] = float32(i%17-8) * 0.125
	}
	expected := evalLinearReference(input, weights, outDim, inDim)

	var best time.Duration
	for i := 0; i < 5; i++ {
		out, dur, err := model.EvalSingleIO(context.Background(), input, outDim, true)
		if err != nil {
			t.Fatalf("EvalSingleIO iter=%d: %v", i, err)
		}
		if len(out) != len(expected) {
			t.Fatalf("EvalSingleIO iter=%d len(out)=%d want=%d", i, len(out), len(expected))
		}
		if err := maxAbsDiffWithin(out, expected, 2e-2); err != nil {
			t.Fatalf("EvalSingleIO iter=%d output mismatch: %v", i, err)
		}
		if i == 0 || dur < best {
			best = dur
		}
	}
	t.Logf("MIL fast path best eval latency: %s", best)
}

func TestNewMILTextDescriptor(t *testing.T) {
	milText, err := linearMILText(1, 1, 1)
	if err != nil {
		t.Fatalf("linearMILText: %v", err)
	}
	blob, err := buildLinearWeightsBlob([]float32{1}, 1, 1)
	if err != nil {
		t.Fatalf("buildLinearWeightsBlob: %v", err)
	}
	weights := foundation.NewMutableDictionaryWithCapacity(1)
	info := foundation.NewMutableDictionaryWithCapacity(2)
	info.SetObjectForKey(foundation.NewNumberWithInt(0), foundation.NewStringWithString("offset"))
	info.SetObjectForKey(foundation.NewDataWithBytesLength(blob), foundation.NewStringWithString("data"))
	weights.SetObjectForKey(info, foundation.NewStringWithString(linearWeightBlobPathInMIL))

	descObj, usedMILInit, err := newMILTextDescriptor(milText, weights, objectivec.Object{})
	if err != nil {
		t.Fatalf("newMILTextDescriptor: %v", err)
	}
	if descObj.GetID() == 0 {
		t.Fatal("newMILTextDescriptor returned nil descriptor")
	}
	desc := appleneuralengine.ANEInMemoryModelDescriptorFromID(descObj.GetID())
	if usedMILInit && !desc.IsMILModel() {
		t.Fatalf("descriptor reports isMILModel=false after MIL init path")
	}
}

func evalLinearReference(input, weights []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		var sum float32
		row := weights[o*inDim : (o+1)*inDim]
		for i, v := range input {
			sum += row[i] * v
		}
		out[o] = sum
	}
	return out
}

func maxAbsDiffWithin(got, want []float32, tol float64) error {
	var (
		maxDiff float64
		maxIdx  int
	)
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	if maxDiff > tol {
		return fmt.Errorf("max abs diff %.6f at index %d exceeds tolerance %.6f", maxDiff, maxIdx, tol)
	}
	return nil
}
