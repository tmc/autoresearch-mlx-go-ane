package mlxgoane

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

type fakeProbeExecutor struct {
	y       []float32
	err     error
	called  int
	cache   map[probeLinearModelKey]bool
	cacheSz int
	route   *probeLinearModelKey
}

type probeLinearModelKey struct {
	batch      int
	inDim      int
	outDim     int
	weightHash uint64
}

type probeLinearShapeKey struct {
	batch  int
	inDim  int
	outDim int
}

type fakeRouteProbeExecutor struct {
	*fakeProbeExecutor
	routeCache map[probeLinearShapeKey]bool
}

func (f *fakeProbeExecutor) Linear(context.Context, []float32, []float32, int, int, int) ([]float32, error) {
	f.called++
	if f.err != nil {
		return nil, f.err
	}
	return append([]float32(nil), f.y...), nil
}

func (f *fakeProbeExecutor) HasLinearModel(batch, inDim, outDim int, weightHash uint64) bool {
	if f.cache == nil {
		return false
	}
	return f.cache[probeLinearModelKey{
		batch:      batch,
		inDim:      inDim,
		outDim:     outDim,
		weightHash: weightHash,
	}]
}

func (f *fakeRouteProbeExecutor) HasLinearRouteModel(batch, inDim, outDim int) bool {
	if f == nil || f.routeCache == nil {
		return false
	}
	return f.routeCache[probeLinearShapeKey{
		batch:  batch,
		inDim:  inDim,
		outDim: outDim,
	}]
}

func (f *fakeProbeExecutor) LinearCacheSize() int {
	return f.cacheSz
}

func (f *fakeProbeExecutor) LinearRouteShape(batch, inDim, outDim int) (int, int, int) {
	if f.route == nil {
		return batch, inDim, outDim
	}
	return f.route.batch, f.route.inDim, f.route.outDim
}

func TestRuntimeRouterSmallSpatialFallsBackWithoutANECall(t *testing.T) {
	x, err := mlx.FromSlice(make([]float32, 8*64), []int{8, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	w, err := mlx.FromSlice(make([]float32, 64*64), []int{64, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	exec := &fakeProbeExecutor{
		y: make([]float32, 8*64),
	}
	rt := NewRuntime(exec)
	rt.Router = NewLinearRouter(DefaultLinearRouteConfig())

	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if exec.called != 0 {
		t.Fatalf("ANE executor called=%d want=0", exec.called)
	}
	if got.Backend != BackendMLX {
		t.Fatalf("backend=%q want=%q", got.Backend, BackendMLX)
	}
	if !strings.Contains(got.FallbackReason, routeReasonSmallSpatial) {
		t.Fatalf("FallbackReason=%q want contains %q", got.FallbackReason, routeReasonSmallSpatial)
	}
}

func TestRuntimeRouterCompileBudgetBlocksColdMiss(t *testing.T) {
	x, err := mlx.FromSlice(make([]float32, 16*64), []int{16, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	w, err := mlx.FromSlice(make([]float32, 64*64), []int{64, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	exec := &fakeProbeExecutor{
		y:       make([]float32, 16*64),
		cacheSz: 100,
	}
	rt := NewRuntime(exec)
	rt.Router = NewLinearRouter(DefaultLinearRouteConfig())

	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if exec.called != 0 {
		t.Fatalf("ANE executor called=%d want=0", exec.called)
	}
	if !strings.Contains(got.FallbackReason, routeReasonCompileBudget) {
		t.Fatalf("FallbackReason=%q want contains %q", got.FallbackReason, routeReasonCompileBudget)
	}
}

func TestRuntimeRouterAllowsCachedANECall(t *testing.T) {
	x, err := mlx.FromSlice(make([]float32, 16*64), []int{16, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	wData := make([]float32, 64*64)
	w, err := mlx.FromSlice(wData, []int{64, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	key := probeLinearModelKey{
		batch:      16,
		inDim:      64,
		outDim:     64,
		weightHash: hashFloat32Slice(wData),
	}
	exec := &fakeProbeExecutor{
		y: make([]float32, 16*64),
		cache: map[probeLinearModelKey]bool{
			key: true,
		},
		cacheSz: 100,
	}
	rt := NewRuntime(exec)
	rt.Router = NewLinearRouter(DefaultLinearRouteConfig())

	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if exec.called != 1 {
		t.Fatalf("ANE executor called=%d want=1", exec.called)
	}
	if got.Backend != BackendANE {
		t.Fatalf("backend=%q want=%q", got.Backend, BackendANE)
	}
}

func TestRuntimeRouterAllowsRouteCachedANECall(t *testing.T) {
	x, err := mlx.FromSlice(make([]float32, 16*64), []int{16, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	w, err := mlx.FromSlice(make([]float32, 64*64), []int{64, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	exec := &fakeRouteProbeExecutor{
		fakeProbeExecutor: &fakeProbeExecutor{
			y:       make([]float32, 16*64),
			cacheSz: 100,
		},
		routeCache: map[probeLinearShapeKey]bool{
			{batch: 16, inDim: 64, outDim: 64}: true,
		},
	}
	rt := NewRuntime(exec)
	rt.Router = NewLinearRouter(DefaultLinearRouteConfig())

	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if exec.called != 1 {
		t.Fatalf("ANE executor called=%d want=1", exec.called)
	}
	if got.Backend != BackendANE {
		t.Fatalf("backend=%q want=%q", got.Backend, BackendANE)
	}
}

func TestRuntimeRouterNoFallbackReturnsError(t *testing.T) {
	x, err := mlx.FromSlice(make([]float32, 8*64), []int{8, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	w, err := mlx.FromSlice(make([]float32, 64*64), []int{64, 64}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	exec := &fakeProbeExecutor{
		y: make([]float32, 8*64),
	}
	rt := NewRuntime(exec)
	rt.AllowFallback = false
	rt.Router = NewLinearRouter(DefaultLinearRouteConfig())

	_, err = rt.Linear(context.Background(), x, w)
	if err == nil {
		t.Fatal("Linear returned nil error with fallback disabled")
	}
	if !strings.Contains(err.Error(), routeReasonSmallSpatial) {
		t.Fatalf("Linear error=%q want contains %q", err.Error(), routeReasonSmallSpatial)
	}
}

func TestRuntimeRouterUsesExecutorRouteShape(t *testing.T) {
	x, err := mlx.FromSlice(make([]float32, 32*32), []int{32, 32}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()
	wData := make([]float32, 3*32)
	w, err := mlx.FromSlice(wData, []int{3, 32}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice w: %v", err)
	}
	defer w.Free()

	exec := &fakeProbeExecutor{
		y:     make([]float32, 32*3),
		route: &probeLinearModelKey{batch: 32, inDim: 32, outDim: 8},
	}
	rt := NewRuntime(exec)
	rt.Router = NewLinearRouter(DefaultLinearRouteConfig())

	got, err := rt.Linear(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}
	defer got.Y.Free()

	if exec.called != 1 {
		t.Fatalf("ANE executor called=%d want=1", exec.called)
	}
	if got.Backend != BackendANE {
		t.Fatalf("backend=%q want=%q", got.Backend, BackendANE)
	}
}
