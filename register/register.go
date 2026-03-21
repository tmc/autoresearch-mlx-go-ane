//go:build darwin

package register

import (
	"context"
	"fmt"
	"time"

	mlxgoane "github.com/tmc/mlx-go-ane"
	_ "github.com/tmc/mlx-go-ane/anedraftimpl"
	"github.com/tmc/mlx-go-ane/decode"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go-lm/offload"
	"github.com/tmc/mlx-go/mlx/nn"
)

func init() {
	offload.RegisterTrainingBackend(trainingBackend{})
	offload.RegisterSpeculativeBackend(speculativeBackend{})
	offload.RegisterDecodePlaneRuntime(decodePlaneRuntime{})
}

// trainingBackend implements the SetupRouting method that
// offload.SetupTrainingRouting type-asserts to.
type trainingBackend struct{}

func (trainingBackend) SetupRouting(ctx context.Context, modeRaw, profileRaw string, allowFallback bool) (offload.TrainingRouting, error) {
	profile, err := parseLinearRouteProfile(profileRaw)
	if err != nil {
		return nil, err
	}
	exec, err := mlxgoane.NewApplePrivateDynamicLinearExecutor()
	if err != nil {
		return nil, err
	}
	rt := mlxgoane.NewRuntimeWithOptions(mlxgoane.RuntimeOptions{
		Executor:                   exec,
		AllowFallback:              &allowFallback,
		LinearRouteProfile:         profile,
		TrainingLinearRouteProfile: profile,
	})
	stats := mlxgoane.NewLinearHookStats()
	mode := nn.LinearModeTraining
	restoreNN := mlxgoane.InstallNNLinearHookWithOptions(rt, mlxgoane.LinearHookOptions{
		Stats: stats,
		Mode:  &mode,
	})
	restoreModel := mlxgoane.InstallModelLinearHookWithOptions(rt, mlxgoane.LinearHookOptions{
		Stats: stats,
	})
	restore := func() {
		restoreModel()
		restoreNN()
	}
	if restoreNN == nil && restoreModel == nil {
		restore = func() {}
	}
	return &trainingRouting{
		restore: restore,
		stats:   stats,
		mode:    modeRaw,
		profile: profile,
	}, nil
}

type trainingRouting struct {
	restore func()
	stats   *mlxgoane.LinearHookStats
	mode    string
	profile mlxgoane.LinearRouteProfile
	last    mlxgoane.LinearHookStatsSnapshot
}

func (r *trainingRouting) Close() {
	if r != nil && r.restore != nil {
		r.restore()
	}
}

func (r *trainingRouting) Report() {
	if r == nil || r.stats == nil {
		return
	}
	s := r.stats.Snapshot()
	fmt.Printf(
		"ANE linear hook summary: total=%d ane=%d mlx=%d ane_frac=%.2f cache_hits=%d cache_misses=%d ane_avg_wall=%s ane_avg_overhead=%s mlx_avg_wall=%s fallback_reasons=%s\n",
		s.TotalCalls,
		s.ANECalls,
		s.MLXCalls,
		s.ANEFraction(),
		s.CacheHits,
		s.CacheMisses,
		s.AvgANEWall().Round(time.Microsecond),
		s.AvgANEOverhead().Round(time.Microsecond),
		s.AvgMLXWall().Round(time.Microsecond),
		s.FormatFallbackReasons(),
	)
	r.last = s
}

func (r *trainingRouting) ReportWindow(label string) {
	if r == nil || r.stats == nil {
		return
	}
	s := r.stats.Snapshot()
	window := diffLinearHookStatsSnapshot(s, r.last)
	deltaTotal := s.TotalCalls - r.last.TotalCalls
	deltaANE := s.ANECalls - r.last.ANECalls
	deltaMLX := s.MLXCalls - r.last.MLXCalls
	if deltaTotal == 0 {
		return
	}
	fmt.Printf(
		"ANE linear hook window[%s]: total=%d ane=%d mlx=%d ane_frac=%.2f ane_avg_wall=%s ane_avg_overhead=%s mlx_avg_wall=%s\n",
		label,
		deltaTotal,
		deltaANE,
		deltaMLX,
		float64(deltaANE)/float64(deltaTotal),
		window.AvgANEWall().Round(time.Microsecond),
		window.AvgANEOverhead().Round(time.Microsecond),
		window.AvgMLXWall().Round(time.Microsecond),
	)
	r.last = s
}

func diffLinearHookStatsSnapshot(cur, prev mlxgoane.LinearHookStatsSnapshot) mlxgoane.LinearHookStatsSnapshot {
	return mlxgoane.LinearHookStatsSnapshot{
		TotalCalls:    cur.TotalCalls - prev.TotalCalls,
		ANECalls:      cur.ANECalls - prev.ANECalls,
		MLXCalls:      cur.MLXCalls - prev.MLXCalls,
		CacheHits:     cur.CacheHits - prev.CacheHits,
		CacheMisses:   cur.CacheMisses - prev.CacheMisses,
		BuildTotal:    cur.BuildTotal - prev.BuildTotal,
		CompileTotal:  cur.CompileTotal - prev.CompileTotal,
		LoadTotal:     cur.LoadTotal - prev.LoadTotal,
		EvaluateTotal: cur.EvaluateTotal - prev.EvaluateTotal,
		ANEWallTotal:  cur.ANEWallTotal - prev.ANEWallTotal,
		MLXWallTotal:  cur.MLXWallTotal - prev.MLXWallTotal,
	}
}

// speculativeBackend registers ANE speculative decoding support.
// Consumers type-assert to a local interface with NewRuntime().
type speculativeBackend struct{}

func (speculativeBackend) NewRuntime() (*speculativeRuntime, error) {
	exec, err := mlxgoane.NewApplePrivateExecutor()
	if err != nil {
		return nil, err
	}
	rt := mlxgoane.NewRuntime(exec)
	rt.AllowFallback = true
	var telemetry linearTelemetryProvider
	if p, ok := exec.(mlxgoane.LinearTelemetryProvider); ok {
		telemetry = linearTelemetryAdapter{provider: p}
	}
	return &speculativeRuntime{
		runtime:   rt,
		telemetry: telemetry,
	}, nil
}

// linearTelemetryProvider is the consumer-local interface for telemetry.
type linearTelemetryProvider interface {
	LastLinearTelemetry() offload.LinearTelemetry
	LinearCacheSize() int
}

type speculativeRuntime struct {
	runtime   *mlxgoane.Runtime
	telemetry linearTelemetryProvider
}

func (r *speculativeRuntime) InstallLinearHook() func() {
	mode := nn.LinearModeInference
	restoreNN := mlxgoane.InstallNNLinearHookWithOptions(r.runtime, mlxgoane.LinearHookOptions{Mode: &mode})
	restoreModel := mlxgoane.InstallModelLinearHookWithOptions(r.runtime, mlxgoane.LinearHookOptions{})
	return func() {
		restoreModel()
		restoreNN()
	}
}

func (r *speculativeRuntime) Telemetry() linearTelemetryProvider {
	return r.telemetry
}

type linearTelemetryAdapter struct {
	provider mlxgoane.LinearTelemetryProvider
}

func (a linearTelemetryAdapter) LastLinearTelemetry() offload.LinearTelemetry {
	t := a.provider.LastLinearTelemetry()
	return offload.LinearTelemetry{
		CacheHit: t.CacheHit,
		Build:    t.Build,
		Compile:  t.Compile,
		Load:     t.Load,
		Evaluate: t.Evaluate,
	}
}

func (a linearTelemetryAdapter) LinearCacheSize() int {
	return a.provider.LinearCacheSize()
}

// decodePlaneRuntime registers the ANE decode plane backend.
// Consumers type-assert to local interfaces for the methods they need.
type decodePlaneRuntime struct{}

func (decodePlaneRuntime) SetModelMirrorRoot(cacheDir string) {
	mlxgoane.SetModelMirrorRoot(cacheDir)
}

// Available reports whether the ANE decode plane runtime is functional.
func (decodePlaneRuntime) Available() bool { return true }

// WrapModel wraps a LanguageModel with the ANE decode plane engine.
// This is the method that decodeplane.Wrap() type-asserts to via the registry.
func (decodePlaneRuntime) WrapModel(ctx context.Context, model models.LanguageModel, mode, modelPath, cacheDir string, warn func(string, ...any)) (models.LanguageModel, error) {
	return decode.Wrap(model, decode.Options{
		Mode:      mode,
		ModelPath: modelPath,
		CacheDir:  cacheDir,
		Warn:      warn,
	})
}

func parseLinearRouteProfile(raw string) (mlxgoane.LinearRouteProfile, error) {
	switch raw {
	case "", "balanced":
		return mlxgoane.LinearRouteProfileBalanced, nil
	case "conservative":
		return mlxgoane.LinearRouteProfileConservative, nil
	case "aggressive":
		return mlxgoane.LinearRouteProfileAggressive, nil
	default:
		return "", fmt.Errorf("unsupported ANE route profile %q", raw)
	}
}
