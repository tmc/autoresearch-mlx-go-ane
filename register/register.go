//go:build darwin && ane_appleneuralengine

package register

import (
	"fmt"

	"github.com/tmc/mlx-go-lm/exp/anehooks"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	mlxgoane "github.com/tmc/mlx-go-ane"
	"github.com/tmc/mlx-go-ane/decode"
	_ "github.com/tmc/mlx-go-ane/anedraftimpl"
)

func init() {
	anehooks.RegisterTrainingBackend(trainingBackend{})
	anehooks.RegisterSpeculativeBackend(speculativeBackend{})
	anehooks.RegisterDecodePlaneRuntime(decodePlaneRuntime{})
}

// trainingBackend implements the SetupRouting method that
// anehooks.SetupTrainingRouting type-asserts to.
type trainingBackend struct{}

func (trainingBackend) SetupRouting(modeRaw, profileRaw string, allowFallback bool) (anehooks.TrainingRouting, error) {
	profile, err := parseLinearRouteProfile(profileRaw)
	if err != nil {
		return nil, err
	}
	exec, err := mlxgoane.NewApplePrivateExecutor()
	if err != nil {
		return nil, err
	}
	rt := mlxgoane.NewRuntimeWithOptions(mlxgoane.RuntimeOptions{
		Executor:           exec,
		AllowFallback:      &allowFallback,
		LinearRouteProfile: profile,
	})
	stats := mlxgoane.NewLinearHookStats()
	restore := mlxgoane.InstallNNLinearHookWithStats(rt, stats)
	if restore == nil {
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
	if s.TotalCalls == 0 {
		return
	}
}

func (r *trainingRouting) ReportWindow(string) {}

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
	LastLinearTelemetry() anehooks.LinearTelemetry
	LinearCacheSize() int
}

type speculativeRuntime struct {
	runtime   *mlxgoane.Runtime
	telemetry linearTelemetryProvider
}

func (r *speculativeRuntime) InstallLinearHook() func() {
	return mlxgoane.InstallNNLinearHook(r.runtime)
}

func (r *speculativeRuntime) Telemetry() linearTelemetryProvider {
	return r.telemetry
}

type linearTelemetryAdapter struct {
	provider mlxgoane.LinearTelemetryProvider
}

func (a linearTelemetryAdapter) LastLinearTelemetry() anehooks.LinearTelemetry {
	t := a.provider.LastLinearTelemetry()
	return anehooks.LinearTelemetry{
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
// This is the method that anedecode.Wrap() type-asserts to via the registry.
func (decodePlaneRuntime) WrapModel(model models.LanguageModel, mode, modelPath, cacheDir string, warn func(string, ...any)) (models.LanguageModel, error) {
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
