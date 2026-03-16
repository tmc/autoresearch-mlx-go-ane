//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/apple/private/appleneuralengine"
)

// TestEspressoFFNWeightScale tests the Espresso FFN at various weight
// magnitudes to find the overflow threshold.
func TestEspressoFFNWeightScale(t *testing.T) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable")
	}
	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("ANE client unavailable")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	dim := 2048
	hidden := 11008

	scales := []float64{0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005}
	for _, scale := range scales {
		t.Run(fmt.Sprintf("scale=%.3f", scale), func(t *testing.T) {
			w1 := makeDeterministicTensor(hidden*dim, float32(scale), 29)
			w3 := makeDeterministicTensor(hidden*dim, float32(scale), 31)
			w2 := makeDeterministicTensor(dim*hidden, float32(scale), 37)

			dir := filepath.Join(t.TempDir(), "ffn")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := GenerateFFNEspressoDir(dir, dim, hidden, w1, w3, w2); err != nil {
				t.Fatal(err)
			}
			model, err := CompileAndLoadEspresso(client, dir, "", 0)
			if err != nil {
				t.Fatal(err)
			}
			defer model.Close()

			input := makeDeterministicTensor(dim, 0.02, 23)
			out, _, err := model.EvalSingleIO(context.Background(), input, dim, true)
			if err != nil {
				t.Fatal(err)
			}

			var maxAbs float64
			for _, v := range out {
				a := math.Abs(float64(v))
				if a > maxAbs {
					maxAbs = a
				}
			}
			hasInf := false
			hasNaN := false
			for _, v := range out {
				if math.IsInf(float64(v), 0) {
					hasInf = true
				}
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
			}
			t.Logf("first 5: %v", out[:min(5, len(out))])
			t.Logf("max abs: %.6g inf=%v nan=%v", maxAbs, hasInf, hasNaN)

			if maxAbs > 1e10 {
				t.Errorf("output overflow: max_abs=%.6g", maxAbs)
			}
		})
	}
}
