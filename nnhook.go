package mlxgoane

import (
	"context"
	"fmt"
	"sync"

	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/nn"
	mlxraw "github.com/tmc/mlx-go/mlx/raw"
	"github.com/tmc/mlx-go/mlxc"
)

// LinearHookOptions configures nn.Linear hook installation.
type LinearHookOptions struct {
	Stats *LinearHookStats
	Mode  *nn.LinearMode
}

type linearHookSession struct {
	rt      *Runtime
	stats   *LinearHookStats
	mode    nn.LinearMode
	modeSet bool

	vjpMu  sync.Mutex
	handle *linearHookVJPHandle
}

type linearHookVJPHandle struct {
	fwdClosure  mlxc.Closure
	vjpClosure  mlxc.ClosureCustom
	wrapClosure mlxc.Closure
}

func (h *linearHookVJPHandle) free() {
	if h == nil {
		return
	}
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

func (s *linearHookSession) free() {
	s.vjpMu.Lock()
	defer s.vjpMu.Unlock()
	if s.handle != nil {
		s.handle.free()
		s.handle = nil
	}
}

type linearHookManagerState struct {
	mu        sync.Mutex
	installed bool
	baseHook  nn.LinearForwardHook
	baseMode  nn.LinearMode
	sessions  []*linearHookSession
}

var linearHookManager linearHookManagerState

func newLinearHookSession(rt *Runtime, opts LinearHookOptions) *linearHookSession {
	session := &linearHookSession{
		rt:    rt,
		stats: opts.Stats,
	}
	if opts.Mode != nil {
		session.mode = *opts.Mode
		session.modeSet = true
	}
	return session
}

// InstallNNLinearHook installs a process-wide nn.Linear hook that routes
// linear forward passes through Runtime.
//
// The returned function restores the previous hook.
func InstallNNLinearHook(rt *Runtime) (restore func()) {
	return InstallNNLinearHookWithOptions(rt, LinearHookOptions{})
}

// InstallNNLinearHookWithStats installs a process-wide nn.Linear hook that
// routes linear forward passes through Runtime and optionally records summary
// stats about backend selection and ANE executor telemetry.
//
// The returned function restores the previous hook.
func InstallNNLinearHookWithStats(rt *Runtime, stats *LinearHookStats) (restore func()) {
	return InstallNNLinearHookWithOptions(rt, LinearHookOptions{Stats: stats})
}

// InstallNNLinearHookWithOptions installs a managed nn.Linear hook session.
//
// Sessions compose within this package: the most recently installed session is
// active, and restoring it reveals the previous session or the original hook.
func InstallNNLinearHookWithOptions(rt *Runtime, opts LinearHookOptions) (restore func()) {
	session := newLinearHookSession(rt, opts)

	linearHookManager.mu.Lock()
	if !linearHookManager.installed {
		linearHookManager.baseMode = nn.CurrentLinearMode()
		linearHookManager.baseHook = nn.SetLinearForwardHook(linearHookDispatch)
		linearHookManager.installed = true
	}
	linearHookManager.sessions = append(linearHookManager.sessions, session)
	linearHookManager.applyTopModeLocked()
	linearHookManager.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			linearHookManager.mu.Lock()
			for i := len(linearHookManager.sessions) - 1; i >= 0; i-- {
				if linearHookManager.sessions[i] != session {
					continue
				}
				copy(linearHookManager.sessions[i:], linearHookManager.sessions[i+1:])
				linearHookManager.sessions = linearHookManager.sessions[:len(linearHookManager.sessions)-1]
				break
			}
			session.free()
			if len(linearHookManager.sessions) == 0 {
				nn.SetLinearForwardHook(linearHookManager.baseHook)
				nn.SetLinearMode(linearHookManager.baseMode)
				linearHookManager.baseHook = nil
				linearHookManager.sessions = nil
				linearHookManager.installed = false
				linearHookManager.mu.Unlock()
				return
			}
			linearHookManager.applyTopModeLocked()
			linearHookManager.mu.Unlock()
		})
	}
}

func linearHookDispatch(x, weight, bias *mlx.Array) (*mlx.Array, bool, error) {
	linearHookManager.mu.Lock()
	var (
		session *linearHookSession
		base    nn.LinearForwardHook
	)
	if n := len(linearHookManager.sessions); n > 0 {
		session = linearHookManager.sessions[n-1]
	}
	base = linearHookManager.baseHook
	linearHookManager.mu.Unlock()

	if session == nil {
		if base != nil {
			return base(x, weight, bias)
		}
		return nil, false, nil
	}

	y, handled, err := session.runContext(context.Background(), x, weight, bias, true)
	if handled || err != nil {
		return y, handled, err
	}
	if base != nil {
		return base(x, weight, bias)
	}
	return nil, false, nil
}

func (m *linearHookManagerState) applyTopModeLocked() {
	mode := m.baseMode
	for i := len(m.sessions) - 1; i >= 0; i-- {
		if !m.sessions[i].modeSet {
			continue
		}
		mode = m.sessions[i].mode
		break
	}
	nn.SetLinearMode(mode)
}

func (s *linearHookSession) runContext(ctx context.Context, x, weight, bias *mlx.Array, applyBias bool) (*mlx.Array, bool, error) {
	if s == nil || s.rt == nil {
		return nil, false, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	origShape := x.Shape()
	needsReshape := len(origShape) > 2

	evalX := x
	ownX := false
	if needsReshape {
		inDim := weight.Shape()[1]
		flatX, err := mlxraw.Reshape(x, []int{-1, inDim}, nil)
		if err != nil {
			return nil, false, err
		}
		evalX = flatX
		ownX = true
	}
	if evalX.Dtype() != mlx.Float32 {
		castX, err := mlxraw.Astype(evalX, mlx.Float32, nil)
		if err != nil {
			if ownX {
				evalX.Free()
			}
			return nil, false, err
		}
		if ownX {
			evalX.Free()
		}
		evalX = castX
		ownX = true
	}
	defer func() {
		if ownX {
			evalX.Free()
		}
	}()

	evalW := weight
	ownW := false
	if weight.Dtype() != mlx.Float32 {
		castW, err := mlxraw.Astype(weight, mlx.Float32, nil)
		if err != nil {
			return nil, false, err
		}
		evalW = castW
		ownW = true
	}
	defer func() {
		if ownW {
			evalW.Free()
		}
	}()

	var (
		y   *mlx.Array
		err error
	)
	if nn.CurrentLinearMode() == nn.LinearModeTraining {
		y, err = s.trainingLinear(evalX, evalW)
	} else {
		var res *LinearResult
		res, err = s.rt.Linear(ctx, evalX, evalW)
		if err == nil && s.stats != nil {
			s.stats.record(s.rt, res)
		}
		if err == nil {
			y = res.Y
		}
	}
	if err != nil {
		return nil, false, err
	}
	if needsReshape {
		outDim := weight.Shape()[0]
		newShape := append([]int{}, origShape[:len(origShape)-1]...)
		newShape = append(newShape, outDim)
		unflatY, err := mlxraw.Reshape(y, newShape, nil)
		if err != nil {
			y.Free()
			return nil, false, err
		}
		y.Free()
		y = unflatY
	}

	if !applyBias || bias == nil || bias.IsNil() {
		return y, true, nil
	}
	withBias, err := mlxraw.Add(y, bias, nil)
	y.Free()
	if err != nil {
		return nil, false, err
	}
	return withBias, true, nil
}

// InstallModelLinearHookWithOptions installs a default models.LinearHook that
// routes StandardLinear forward passes through Runtime when no per-context hook
// is present.
func InstallModelLinearHookWithOptions(rt *Runtime, opts LinearHookOptions) (restore func()) {
	session := newLinearHookSession(rt, opts)
	restoreDefault := models.InstallDefaultLinearHook(func(ctx context.Context, x, weight, bias *mlx.Array) (*mlx.Array, bool) {
		y, handled, err := session.runContext(ctx, x, weight, bias, false)
		if err != nil {
			return nil, false
		}
		return y, handled
	})

	var once sync.Once
	return func() {
		once.Do(func() {
			restoreDefault()
			session.free()
		})
	}
}

func (s *linearHookSession) trainingLinear(x, w *mlx.Array) (*mlx.Array, error) {
	s.vjpMu.Lock()
	if s.handle == nil {
		handle, err := s.initTrainingHandle()
		if err != nil {
			s.vjpMu.Unlock()
			return nil, err
		}
		s.handle = handle
	}
	wrap := s.handle.wrapClosure
	s.vjpMu.Unlock()

	inputVec := mlxc.NewVectorArray()
	mlxc.VectorArrayAppendValue(inputVec, x.MlxcArray())
	mlxc.VectorArrayAppendValue(inputVec, w.MlxcArray())
	defer mlxc.VectorArrayFree(inputVec)

	var outVec mlxc.VectorArray
	if status := mlxc.ClosureApply(&outVec, wrap, inputVec); status != 0 {
		if msg := mlxc.GetLastError(); msg != "" {
			return nil, fmt.Errorf("hook customvjp apply: %s", msg)
		}
		return nil, fmt.Errorf("hook customvjp apply: status %d", status)
	}
	defer mlxc.VectorArrayFree(outVec)

	var cy mlxc.Array
	if status := mlxc.VectorArrayGet(&cy, outVec, 0); status != 0 {
		return nil, fmt.Errorf("hook customvjp: get output: status %d", status)
	}
	return mlx.NewArrayFromMlxc(cy), nil
}

func (s *linearHookSession) initTrainingHandle() (*linearHookVJPHandle, error) {
	fwdFn := mlxc.NewClosureFunc(func(output *mlxc.VectorArray, input mlxc.VectorArray) error {
		var cx, cw mlxc.Array
		if status := mlxc.VectorArrayGet(&cx, input, 0); status != 0 {
			return fmt.Errorf("hook vjp fwd: get x: status %d", status)
		}
		if status := mlxc.VectorArrayGet(&cw, input, 1); status != 0 {
			return fmt.Errorf("hook vjp fwd: get w: status %d", status)
		}

		gx := mlx.NewArrayFromMlxc(cx)
		gw := mlx.NewArrayFromMlxc(cw)
		res, err := s.rt.linearForward(context.Background(), gx, gw)
		if err != nil {
			y, fbErr := linearMLX(gx, gw)
			if fbErr != nil {
				return fmt.Errorf("hook vjp fwd: ane=%v mlx=%w", err, fbErr)
			}
			res = &LinearResult{
				Y:              y,
				Backend:        BackendMLX,
				FallbackReason: err.Error(),
			}
		}
		if s.stats != nil {
			s.stats.record(s.rt, res)
		}
		*output = mlxc.NewVectorArrayValue(res.Y.MlxcArray())
		return nil
	})

	vjpFn := mlxc.NewClosureCustomFunc(func(output *mlxc.VectorArray, primals, cotangents, _ mlxc.VectorArray) error {
		var cx, cw, cdy mlxc.Array
		if status := mlxc.VectorArrayGet(&cx, primals, 0); status != 0 {
			return fmt.Errorf("hook vjp bwd: get x: status %d", status)
		}
		if status := mlxc.VectorArrayGet(&cw, primals, 1); status != 0 {
			return fmt.Errorf("hook vjp bwd: get w: status %d", status)
		}
		if status := mlxc.VectorArrayGet(&cdy, cotangents, 0); status != 0 {
			return fmt.Errorf("hook vjp bwd: get dy: status %d", status)
		}

		gx := mlx.NewArrayFromMlxc(cx)
		gw := mlx.NewArrayFromMlxc(cw)
		gdy := mlx.NewArrayFromMlxc(cdy)

		dx, err := mlxraw.Matmul(gdy, gw, nil)
		if err != nil {
			return err
		}
		inDim := gw.Shape()[1]
		outDim := gw.Shape()[0]
		flatX, err := mlxraw.Reshape(gx, []int{-1, inDim}, nil)
		if err != nil {
			dx.Free()
			return err
		}
		flatDy, err := mlxraw.Reshape(gdy, []int{-1, outDim}, nil)
		if err != nil {
			dx.Free()
			flatX.Free()
			return err
		}
		transposedDy, err := mlxraw.Transpose(flatDy, nil)
		if err != nil {
			dx.Free()
			flatX.Free()
			flatDy.Free()
			return err
		}
		dw, err := mlxraw.Matmul(transposedDy, flatX, nil)
		transposedDy.Free()
		flatDy.Free()
		flatX.Free()
		if err != nil {
			dx.Free()
			return err
		}

		*output = mlxc.NewVectorArray()
		mlxc.VectorArrayAppendValue(*output, dx.MlxcArray())
		mlxc.VectorArrayAppendValue(*output, dw.MlxcArray())
		return nil
	})

	wrap, err := mlxc.CustomVjp(fwdFn, vjpFn)
	if err != nil {
		mlxc.ClosureFree(fwdFn)
		mlxc.ClosureCustomFree(vjpFn)
		return nil, err
	}
	return &linearHookVJPHandle{
		fwdClosure:  fwdFn,
		vjpClosure:  vjpFn,
		wrapClosure: wrap,
	}, nil
}
