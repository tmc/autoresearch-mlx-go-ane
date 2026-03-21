package mlxgoane

import (
	"context"
	"fmt"
	"sync"

	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go/mlx"
	mlxraw "github.com/tmc/mlx-go/mlx/raw"
	"github.com/tmc/mlx-go/mlxc"
)

// linearVJPHandle holds the mlxc closures that must stay alive for the duration
// of compilation and execution. mlx.Compile records a pointer to these closures
// internally; freeing them while the compiled graph is active causes a crash.
type linearVJPHandle struct {
	fwdClosure  mlxc.Closure
	vjpClosure  mlxc.ClosureCustom
	wrapClosure mlxc.Closure
}

func (h *linearVJPHandle) free() {
	if !h.wrapClosure.IsNil() {
		mlxc.ClosureFree(h.wrapClosure)
		h.wrapClosure = mlxc.Closure{}
	}
	if !h.fwdClosure.IsNil() {
		mlxc.ClosureFree(h.fwdClosure)
		h.fwdClosure = mlxc.Closure{}
	}
	if !h.vjpClosure.IsNil() {
		mlxc.ClosureCustomFree(h.vjpClosure)
		h.vjpClosure = mlxc.ClosureCustom{}
	}
}

// linearVJPCache caches a single VJP-wrapped closure per Runtime instance.
// The cached closure is built once and kept alive for the training session.
type linearVJPCache struct {
	mu     sync.Mutex
	handle *linearVJPHandle
}

func (c *linearVJPCache) free() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle != nil {
		c.handle.free()
		c.handle = nil
	}
}

// initVJPHandle builds the persistent wrappedFn closure (called once, under lock).
func (c *linearVJPCache) initVJPHandle(r *Runtime) (*linearVJPHandle, error) {
	// Forward closure: inputs=[x, w], output=[y].
	//
	// Key design: when inside an mlx.Compile trace (kvcache.IsCompiling()==true),
	// we must return symbolic MLX arrays so the graph tracer can record operations.
	// When evaluating for real, we run ANE.
	fwdFn := mlxc.NewClosureFunc(func(output *mlxc.VectorArray, input mlxc.VectorArray) error {
		var cx, cw mlxc.Array
		if status := mlxc.VectorArrayGet(&cx, input, 0); status != 0 {
			return fmt.Errorf("vjp fwd: get x: status %d", status)
		}
		if status := mlxc.VectorArrayGet(&cw, input, 1); status != 0 {
			return fmt.Errorf("vjp fwd: get w: status %d", status)
		}
		xArr := mlx.NewArrayFromMlxc(cx)
		wArr := mlx.NewArrayFromMlxc(cw)
		defer xArr.Free()
		defer wArr.Free()

		xShape := xArr.Shape()
		wShape := wArr.Shape()
		if len(xShape) != 2 || len(wShape) != 2 {
			return fmt.Errorf("vjp fwd: rank-2 required, got x=%v w=%v", xShape, wShape)
		}
		batch, inDim, outDim := xShape[0], xShape[1], wShape[0]

		var (
			yArr *mlx.Array
			err  error
		)

		// During Compile tracing we must return symbolic (MLX) arrays.
		// ANE would return a concrete mlx.FromSlice array that breaks graph tracing.
		if !kvcache.IsCompiling() && r != nil && r.Executor != nil {
			yArr, err = r.linearANE(context.Background(), xArr, wArr, batch, inDim, outDim)
			if err == nil && yArr != nil {
				r.setVJPResult(BackendANE, "")
			}
		}
		if err != nil || yArr == nil {
			fallback := "compile trace"
			if err != nil {
				fallback = err.Error()
			}
			var fbErr error
			yArr, fbErr = linearMLX(xArr, wArr)
			if fbErr != nil {
				panic("vjp fwd: mlx fallback failed")
			}
			if r != nil {
				r.setVJPResult(BackendMLX, fallback)
			}
			err = nil // Clear error since fallback succeeded
		}
		_ = inDim // used implicitly via linearMLX / linearANE
		_ = outDim

		*output = mlxc.NewVectorArray()
		mlxc.VectorArrayAppendValue(*output, yArr.MlxcArray())
		return nil
	})

	// VJP closure: primals=[x,w], cotangents=[dL/dy], outputs=[y]
	// Returns [dL/dx, dL/dw] via standard MLX matmul (always symbolic – correct
	// for both tracing and real evaluation).
	vjpFn := mlxc.NewClosureCustomFunc(func(output *mlxc.VectorArray, primals, cotangents, _ mlxc.VectorArray) error {
		var cx, cw, cCot mlxc.Array
		if status := mlxc.VectorArrayGet(&cx, primals, 0); status != 0 {
			return fmt.Errorf("vjp: get x: status %d", status)
		}
		if status := mlxc.VectorArrayGet(&cw, primals, 1); status != 0 {
			return fmt.Errorf("vjp: get w: status %d", status)
		}
		if status := mlxc.VectorArrayGet(&cCot, cotangents, 0); status != 0 {
			return fmt.Errorf("vjp: get cotangent: status %d", status)
		}
		xArr := mlx.NewArrayFromMlxc(cx)
		wArr := mlx.NewArrayFromMlxc(cw)
		cotArr := mlx.NewArrayFromMlxc(cCot)

		// dL/dx = dL/dy @ w
		dldx, err := mlxraw.Matmul(cotArr, wArr, nil)
		if err != nil {
			return fmt.Errorf("vjp: dL/dx: %w", err)
		}
		// dL/dw = dL/dy^T @ x  → [outDim, inDim]
		cotT, err := mlxraw.Transpose(cotArr, nil)
		if err != nil {
			dldx.Free()
			return fmt.Errorf("vjp: transpose cotangent: %w", err)
		}
		dldw, err := mlxraw.Matmul(cotT, xArr, nil)
		cotT.Free()
		if err != nil {
			dldx.Free()
			return fmt.Errorf("vjp: dL/dw: %w", err)
		}

		*output = mlxc.NewVectorArray()
		mlxc.VectorArrayAppendValue(*output, dldx.MlxcArray())
		mlxc.VectorArrayAppendValue(*output, dldw.MlxcArray())
		return nil
	})

	wrap, err := mlxc.CustomVjp(fwdFn, vjpFn)
	if err != nil {
		mlxc.ClosureFree(fwdFn)
		mlxc.ClosureCustomFree(vjpFn)
		return nil, fmt.Errorf("customvjp create: %w", err)
	}
	return &linearVJPHandle{
		fwdClosure:  fwdFn,
		vjpClosure:  vjpFn,
		wrapClosure: wrap,
	}, nil
}

// linearANEWithVJP executes the ANE linear forward pass through an mlxc.CustomVjp
// closure cached for the lifetime of the Runtime.
//
// The closure is built once so mlx.Compile can safely record its pointer.
// Inside a Compile trace the forward body uses MLX matmul (symbolic); outside
// compilation (real eval) it runs ANE.
func (r *Runtime) linearANEWithVJP(ctx context.Context, x, w *mlx.Array, batch, inDim, outDim int) (*mlx.Array, error) {
	if r.vjpCache == nil {
		r.vjpCache = &linearVJPCache{}
	}
	cache := r.vjpCache

	cache.mu.Lock()
	if cache.handle == nil {
		h, err := cache.initVJPHandle(r)
		if err != nil {
			cache.mu.Unlock()
			return nil, err
		}
		cache.handle = h
	}
	wrap := cache.handle.wrapClosure
	cache.mu.Unlock()

	inputVec := mlxc.NewVectorArray()
	mlxc.VectorArrayAppendValue(inputVec, x.MlxcArray())
	mlxc.VectorArrayAppendValue(inputVec, w.MlxcArray())
	defer mlxc.VectorArrayFree(inputVec)

	var outVec mlxc.VectorArray
	if status := mlxc.ClosureApply(&outVec, wrap, inputVec); status != 0 {
		if msg := mlxc.GetLastError(); msg != "" {
			return nil, fmt.Errorf("customvjp apply: %s", msg)
		}
		return nil, fmt.Errorf("customvjp apply: status %d", status)
	}
	defer mlxc.VectorArrayFree(outVec)

	var cy mlxc.Array
	if status := mlxc.VectorArrayGet(&cy, outVec, 0); status != 0 {
		return nil, fmt.Errorf("customvjp: get output: status %d", status)
	}
	return mlx.NewArrayFromMlxc(cy), nil
}
