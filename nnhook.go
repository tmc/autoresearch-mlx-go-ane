package mlxgoane

import (
	"context"
	"fmt"

	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/nn"
)

// InstallNNLinearHook installs a process-wide nn.Linear hook that routes
// linear forward passes through Runtime.
//
// The returned function restores the previous hook.
func InstallNNLinearHook(rt *Runtime) (restore func()) {
	return InstallNNLinearHookWithStats(rt, nil)
}

// InstallNNLinearHookWithStats installs a process-wide nn.Linear hook that
// routes linear forward passes through Runtime and optionally records summary
// stats about backend selection and ANE executor telemetry.
//
// The returned function restores the previous hook.
func InstallNNLinearHookWithStats(rt *Runtime, stats *LinearHookStats) (restore func()) {
	prev := nn.SetLinearForwardHook(func(x, weight, bias *mlx.Array) (*mlx.Array, bool, error) {
		if rt == nil {
			return nil, false, nil
		}
		res, err := rt.Linear(context.Background(), x, weight)
		if err != nil {
			return nil, false, fmt.Errorf("runtime linear: %w", err)
		}
		if stats != nil {
			stats.record(rt, res)
		}
		if bias == nil || bias.IsNil() {
			return res.Y, true, nil
		}
		withBias, err := mlx.Add(res.Y, bias, nil)
		if err != nil {
			res.Y.Free()
			return nil, false, fmt.Errorf("add bias: %w", err)
		}
		res.Y.Free()
		return withBias, true, nil
	})
	return func() {
		nn.SetLinearForwardHook(prev)
	}
}
