package mlxgoane

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlxc"
)

// Backend identifies which engine produced the result.
type Backend string

const (
	BackendANE Backend = "ane"
	BackendMLX Backend = "mlx"
)

// LinearExecutor runs a single linear forward pass using ANE-compatible data.
//
// Inputs:
//   - x shape is [batch, inDim]
//   - w shape is [outDim, inDim]
//
// Output:
//   - y shape is [batch, outDim]
type LinearExecutor interface {
	Linear(ctx context.Context, x, w []float32, batch, inDim, outDim int) ([]float32, error)
}

// Runtime executes selected ops on ANE and falls back to mlx-go when needed.
type Runtime struct {
	Executor      LinearExecutor
	AllowFallback bool
	Router        *LinearRouter
}

// LinearResult contains a linear forward output and execution details.
type LinearResult struct {
	Y              *mlx.Array
	Backend        Backend
	FallbackReason string
}

// NewRuntime returns a runtime with fallback enabled.
func NewRuntime(executor LinearExecutor) *Runtime {
	return &Runtime{Executor: executor, AllowFallback: true}
}

// Linear computes x * w^T using ANE when available, else mlx-go fallback.
//
// x must be float32 [batch, inDim], w must be float32 [outDim, inDim].
func (r *Runtime) Linear(ctx context.Context, x, w *mlx.Array) (*LinearResult, error) {
	batch, inDim, outDim, err := validateLinearInputs(x, w)
	if err != nil {
		return nil, err
	}

	if r != nil && r.Executor != nil {
		if r.Router != nil {
			decision, err := r.routeLinear(w, batch, inDim, outDim)
			if err != nil {
				return nil, err
			}
			if !decision.UseANE {
				if !r.AllowFallback {
					return nil, fmt.Errorf("ane routing: %s", decision.Reason)
				}
				y, fbErr := linearMLX(x, w)
				if fbErr != nil {
					return nil, fmt.Errorf("ane route blocked (%s) and mlx fallback failed: %w", decision.Reason, fbErr)
				}
				return &LinearResult{
					Y:              y,
					Backend:        BackendMLX,
					FallbackReason: "router: " + decision.Reason,
				}, nil
			}
		}
		y, err := r.linearANE(ctx, x, w, batch, inDim, outDim)
		if err == nil {
			return &LinearResult{Y: y, Backend: BackendANE}, nil
		}
		if r.AllowFallback {
			y, fbErr := linearMLX(x, w)
			if fbErr != nil {
				return nil, fmt.Errorf("ane failed (%v) and mlx fallback failed: %w", err, fbErr)
			}
			return &LinearResult{
				Y:              y,
				Backend:        BackendMLX,
				FallbackReason: err.Error(),
			}, nil
		}
		return nil, fmt.Errorf("ane linear: %w", err)
	}

	y, err := linearMLX(x, w)
	if err != nil {
		return nil, err
	}
	return &LinearResult{Y: y, Backend: BackendMLX}, nil
}

type linearModelProbe interface {
	HasLinearModel(batch, inDim, outDim int, weightHash uint64) bool
}

type linearCacheSizer interface {
	LinearCacheSize() int
}

type linearRouteModelProbe interface {
	HasLinearRouteModel(batch, inDim, outDim int) bool
}

type linearRouteShapeProvider interface {
	LinearRouteShape(batch, inDim, outDim int) (routeBatch, routeInDim, routeOutDim int)
}

func (r *Runtime) routeLinear(w *mlx.Array, batch, inDim, outDim int) (RouteDecision, error) {
	routeBatch, routeInDim, routeOutDim := batch, inDim, outDim
	if provider, ok := r.Executor.(linearRouteShapeProvider); ok {
		routeBatch, routeInDim, routeOutDim = provider.LinearRouteShape(batch, inDim, outDim)
	}
	in := LinearRouteInput{
		Batch:  routeBatch,
		InDim:  routeInDim,
		OutDim: routeOutDim,
	}

	if r == nil || r.Executor == nil || r.Router == nil {
		return RouteDecision{UseANE: true, Reason: routeReasonEligible}, nil
	}

	if sizer, ok := r.Executor.(linearCacheSizer); ok {
		in.CacheKnown = true
		in.CacheSize = sizer.LinearCacheSize()
	}
	if probe, ok := r.Executor.(linearRouteModelProbe); ok {
		in.CacheKnown = true
		in.CacheHit = probe.HasLinearRouteModel(routeBatch, routeInDim, routeOutDim)
		return r.Router.DecideLinear(in), nil
	}
	if probe, ok := r.Executor.(linearModelProbe); ok {
		wData, err := mlx.ToSlice[float32](w)
		if err != nil {
			return RouteDecision{}, fmt.Errorf("route linear: extract w: %w", err)
		}
		in.CacheKnown = true
		in.CacheHit = probe.HasLinearModel(batch, inDim, outDim, hashFloat32Slice(wData))
	}

	return r.Router.DecideLinear(in), nil
}

func validateLinearInputs(x, w *mlx.Array) (batch, inDim, outDim int, err error) {
	if x == nil || x.IsNil() {
		return 0, 0, 0, fmt.Errorf("linear: x is nil")
	}
	if w == nil || w.IsNil() {
		return 0, 0, 0, fmt.Errorf("linear: w is nil")
	}
	if x.Dtype() != mlx.Float32 {
		return 0, 0, 0, fmt.Errorf("linear: x dtype=%v want float32", x.Dtype())
	}
	if w.Dtype() != mlx.Float32 {
		return 0, 0, 0, fmt.Errorf("linear: w dtype=%v want float32", w.Dtype())
	}

	xShape := x.Shape()
	wShape := w.Shape()
	if len(xShape) != 2 {
		return 0, 0, 0, fmt.Errorf("linear: x shape=%v want rank-2", xShape)
	}
	if len(wShape) != 2 {
		return 0, 0, 0, fmt.Errorf("linear: w shape=%v want rank-2", wShape)
	}
	if xShape[1] != wShape[1] {
		return 0, 0, 0, fmt.Errorf("linear: in-dim mismatch x=%v w=%v", xShape, wShape)
	}
	return xShape[0], xShape[1], wShape[0], nil
}

func (r *Runtime) linearANE(ctx context.Context, x, w *mlx.Array, batch, inDim, outDim int) (*mlx.Array, error) {
	xData, releaseX, err := contiguousFloat32View(x)
	if err != nil {
		return nil, fmt.Errorf("extract x: %w", err)
	}
	defer releaseX()
	wData, releaseW, err := contiguousFloat32View(w)
	if err != nil {
		return nil, fmt.Errorf("extract w: %w", err)
	}
	defer releaseW()
	yData, err := r.Executor.Linear(ctx, xData, wData, batch, inDim, outDim)
	if err != nil {
		return nil, err
	}
	if len(yData) != batch*outDim {
		return nil, fmt.Errorf("ane output len=%d want=%d", len(yData), batch*outDim)
	}
	y, err := mlx.FromSlice(yData, []int{batch, outDim}, mlx.Float32)
	if err != nil {
		return nil, fmt.Errorf("materialize output: %w", err)
	}
	return y, nil
}

func contiguousFloat32View(a *mlx.Array) ([]float32, func(), error) {
	if a == nil || a.IsNil() {
		return nil, func() {}, fmt.Errorf("array is nil")
	}
	if a.Dtype() != mlx.Float32 {
		return nil, func() {}, fmt.Errorf("array dtype=%v want float32", a.Dtype())
	}

	view := a
	owned := false
	isContig, err := a.MLXArrayIsContiguous()
	if err != nil {
		return nil, func() {}, fmt.Errorf("check contiguous: %w", err)
	}
	if !isContig {
		view, err = mlx.Contiguous(a, false, nil)
		if err != nil {
			return nil, func() {}, fmt.Errorf("make contiguous: %w", err)
		}
		owned = true
	}
	if err := mlx.Eval(view); err != nil {
		if owned {
			_ = view.Free()
		}
		return nil, func() {}, fmt.Errorf("eval array: %w", err)
	}
	ptr := mlxc.ArrayDataFloat32(view.MlxcArray())
	if ptr == nil {
		if owned {
			_ = view.Free()
		}
		return nil, func() {}, fmt.Errorf("array data is nil")
	}
	data := unsafe.Slice(ptr, view.Size())
	release := func() {
		runtime.KeepAlive(view)
		if owned {
			_ = view.Free()
		}
	}
	return data, release, nil
}

func linearMLX(x, w *mlx.Array) (*mlx.Array, error) {
	wT, err := mlx.Transpose(w, nil)
	if err != nil {
		return nil, fmt.Errorf("transpose weights: %w", err)
	}
	defer wT.Free()

	y, err := mlx.Matmul(x, wT, nil)
	if err != nil {
		return nil, fmt.Errorf("mlx matmul: %w", err)
	}
	return y, nil
}
