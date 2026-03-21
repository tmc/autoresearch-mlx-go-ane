package mlxgoane

import (
	"context"
	"testing"

	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
)

func TestInstallModelLinearHookRoutesStandardLinear(t *testing.T) {
	rt := NewRuntime(fakeExecutor{y: []float32{10, 20, 30, 40}})
	restore := InstallModelLinearHookWithOptions(rt, LinearHookOptions{})
	t.Cleanup(restore)

	weight, err := mlx.FromSlice([]float32{1, 0, 0, 1}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice weight: %v", err)
	}
	defer weight.Free()

	layer := models.NewStandardLinear(weight, nil)

	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()

	y := layer.Forward(context.Background(), x)
	defer y.Free()

	got := mlx.MustToSlice[float32](y)
	want := []float32{10, 20, 30, 40}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("y[%d]=%g want=%g", i, got[i], want[i])
		}
	}
}

func TestInstallModelLinearHookDoesNotDoubleApplyBias(t *testing.T) {
	rt := NewRuntime(fakeExecutor{y: []float32{1, 2, 3, 4}})
	restore := InstallModelLinearHookWithOptions(rt, LinearHookOptions{})
	t.Cleanup(restore)

	weight, err := mlx.FromSlice([]float32{1, 0, 0, 1}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice weight: %v", err)
	}
	defer weight.Free()

	bias, err := mlx.FromSlice([]float32{10, 20}, []int{2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice bias: %v", err)
	}
	defer bias.Free()

	layer := models.NewStandardLinear(weight, bias)

	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()

	y := layer.Forward(context.Background(), x)
	defer y.Free()

	got := mlx.MustToSlice[float32](y)
	want := []float32{11, 22, 13, 24}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("y[%d]=%g want=%g", i, got[i], want[i])
		}
	}
}
