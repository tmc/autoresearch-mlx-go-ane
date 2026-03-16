package mlxgoane

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

type fakeExecutor struct {
	y   []float32
	err error
}

func (f fakeExecutor) Linear(context.Context, []float32, []float32, int, int, int) ([]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	return append([]float32(nil), f.y...), nil
}

func TestLinearFallbackMatchesMLX(t *testing.T) {
	x, err := mlx.FromSlice([]float32{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()

	w, err := mlx.FromSlice([]float32{
		1, 0, 0,
		0, 1, 0,
	}, []int{2, 3}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	rt := NewRuntime(nil)
	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if got.Backend != BackendMLX {
		t.Fatalf("backend=%q want=%q", got.Backend, BackendMLX)
	}

	data, err := mlx.ToSlice[float32](got.Y)
	if err != nil {
		t.Fatalf("ToSlice: %v", err)
	}
	want := []float32{1, 2, 4, 5}
	for i := range want {
		if !almostEqual(data[i], want[i]) {
			t.Fatalf("y[%d]=%g want=%g (full=%v)", i, data[i], want[i], data)
		}
	}
}

func TestLinearANEPath(t *testing.T) {
	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	w, err := mlx.FromSlice([]float32{1, 0, 0, 1}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	rt := NewRuntime(fakeExecutor{y: []float32{10, 20, 30, 40}})
	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if got.Backend != BackendANE {
		t.Fatalf("backend=%q want=%q", got.Backend, BackendANE)
	}
	data := mlx.MustToSlice[float32](got.Y)
	want := []float32{10, 20, 30, 40}
	for i := range want {
		if data[i] != want[i] {
			t.Fatalf("y[%d]=%g want=%g", i, data[i], want[i])
		}
	}
}

func TestLinearANEFallbackOnError(t *testing.T) {
	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	w, err := mlx.FromSlice([]float32{1, 0, 0, 1}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	rt := NewRuntime(fakeExecutor{err: errors.New("boom")})
	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if got.Backend != BackendMLX {
		t.Fatalf("backend=%q want=%q", got.Backend, BackendMLX)
	}
	if got.FallbackReason == "" {
		t.Fatal("FallbackReason is empty")
	}
}

func TestLinearRejectsShapeMismatch(t *testing.T) {
	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	w, err := mlx.FromSlice([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	rt := NewRuntime(nil)
	_, err = rt.Linear(context.Background(), x, w)
	if err == nil {
		t.Fatal("Linear returned nil error for mismatched in-dim")
	}
}

func almostEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-5
}
