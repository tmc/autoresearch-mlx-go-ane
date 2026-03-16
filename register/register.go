//go:build darwin && ane_appleneuralengine

package register

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/mlx-go-lm/anehooks"
	mlxgoane "github.com/tmc/mlx-go-ane"
	_ "github.com/tmc/mlx-go-ane/anedraftimpl"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/modelir"
)

func init() {
	anehooks.RegisterTrainingBackend(trainingBackend{})
	anehooks.RegisterSpeculativeBackend(speculativeBackend{})
	anehooks.RegisterDecodePlaneRuntime(decodePlaneRuntime{})
}

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

type speculativeBackend struct{}

func (speculativeBackend) NewRuntime() (anehooks.SpeculativeRuntime, error) {
	exec, err := mlxgoane.NewApplePrivateExecutor()
	if err != nil {
		return nil, err
	}
	rt := mlxgoane.NewRuntime(exec)
	rt.AllowFallback = true
	var telemetry anehooks.LinearTelemetryProvider
	if p, ok := exec.(mlxgoane.LinearTelemetryProvider); ok {
		telemetry = linearTelemetryAdapter{provider: p}
	}
	return speculativeRuntime{
		runtime:   rt,
		telemetry: telemetry,
	}, nil
}

type speculativeRuntime struct {
	runtime   *mlxgoane.Runtime
	telemetry anehooks.LinearTelemetryProvider
}

func (r speculativeRuntime) InstallLinearHook() func() {
	return mlxgoane.InstallNNLinearHook(r.runtime)
}

func (r speculativeRuntime) Telemetry() anehooks.LinearTelemetryProvider {
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

type decodePlaneRuntime struct{}

func (decodePlaneRuntime) SetModelMirrorRoot(cacheDir string) {
	mlxgoane.SetModelMirrorRoot(cacheDir)
}

func (decodePlaneRuntime) NewQwen35Stage(dim, hidden int, cacheDir string, w1, w3, w2 []float32) (anehooks.DecodePlaneStage, anehooks.DecodePlaneBridge, error) {
	cfg := mlxgoane.DefaultMultiSurfaceEvalPlanConfig()
	cfg.EnableMetalWait = true
	cfg.EnableMetalSignal = true
	cfg.WaitValue = 1
	cfg.SignalValue = 1
	stage, err := mlxgoane.NewSurfaceEspressoFFN(dim, hidden, 1, cacheDir, w1, w3, w2, cfg)
	if err != nil {
		return nil, nil, err
	}
	bridge, err := mlxgoane.NewMLXSurfaceSyncBridge(stage.WaitEvent(), stage.SignalEvent())
	if err != nil {
		stage.Close()
		return nil, nil, err
	}
	return decodePlaneStageAdapter{stage: stage}, decodePlaneBridgeAdapter{bridge: bridge}, nil
}

func (decodePlaneRuntime) NewQwen35DirectBlock(prog *modelir.Program, cfg anehooks.DecodePlaneDirectBlockConfig) (anehooks.DecodePlaneDirectBlock, anehooks.DecodePlaneBridge, error) {
	if prog == nil {
		return nil, nil, fmt.Errorf("ane direct block: program is nil")
	}
	if cfg.HiddenDim <= 0 || cfg.VocabSize <= 0 || cfg.OutputDim <= 0 || cfg.MaxSeqLen <= 0 {
		return nil, nil, fmt.Errorf(
			"ane direct block: invalid dims hidden=%d vocab=%d output=%d max_seq=%d",
			cfg.HiddenDim,
			cfg.VocabSize,
			cfg.OutputDim,
			cfg.MaxSeqLen,
		)
	}
	selectedLayers := cfg.SelectedLayers
	if selectedLayers <= 0 {
		selectedLayers = 1
	}
	reifyOpts := mlxgoane.ReifyOptions{
		TransformerConfig: mlxgoane.MILTransformerConfig{
			NumLayers:          selectedLayers,
			MaxSeqLen:          cfg.MaxSeqLen,
			KVCacheState:       true,
			KVCacheMaxLen:      cfg.MaxSeqLen,
			AttentionMaskInput: false,
		},
		RequestedLayers: selectedLayers,
		SelectedLayers:  selectedLayers,
	}
	planCfg := mlxgoane.DefaultMultiSurfaceEvalPlanConfig()
	planCfg.EnableMetalWait = true
	planCfg.EnableMetalSignal = true
	planCfg.WaitValue = 1
	planCfg.SignalValue = 1
	var (
		draft *mlxgoane.ANEDraftModel
		err   error
	)
	if selectedLayers > 1 {
		draft, _, err = mlxgoane.NewANEDraftModelFromModelIRProgramLayerStackWithPlanConfig(
			prog,
			reifyOpts,
			cfg.HiddenDim,
			cfg.VocabSize,
			cfg.OutputDim,
			nil,
			&planCfg,
		)
	} else {
		draft, _, err = mlxgoane.NewANEDraftModelFromModelIRProgramWithPlanConfig(
			prog,
			reifyOpts,
			cfg.HiddenDim,
			cfg.VocabSize,
			cfg.OutputDim,
			nil,
			&planCfg,
		)
	}
	if err != nil {
		return nil, nil, err
	}
	bridge, err := mlxgoane.NewMLXSurfaceSyncBridge(draft.WaitEvent(), draft.SignalEvent())
	if err != nil {
		draft.Close()
		return nil, nil, err
	}
	return decodePlaneDirectBlockAdapter{draft: draft}, decodePlaneBridgeAdapter{bridge: bridge}, nil
}

type decodePlaneStageAdapter struct {
	stage *mlxgoane.SurfaceEspressoFFN
}

func (a decodePlaneStageAdapter) Close() { a.stage.Close() }

func (a decodePlaneStageAdapter) InitStats() anehooks.DecodePlaneInitStats {
	s := a.stage.InitStats()
	return anehooks.DecodePlaneInitStats{
		ArtifactCacheHit: s.ArtifactCacheHit,
		ArtifactReady:    s.ArtifactReady,
		CompileLoad:      s.CompileLoad,
		Map:              s.Map,
	}
}

func (a decodePlaneStageAdapter) ModelDim() int { return a.stage.ModelDim() }
func (a decodePlaneStageAdapter) MapSeq() int   { return a.stage.MapSeq() }
func (a decodePlaneStageAdapter) WaitEvent() anehooks.DecodePlaneEvent {
	return decodePlaneEventAdapter{event: a.stage.WaitEvent()}
}
func (a decodePlaneStageAdapter) SignalEvent() anehooks.DecodePlaneEvent {
	return decodePlaneEventAdapter{event: a.stage.SignalEvent()}
}
func (a decodePlaneStageAdapter) WaitValue() uint64   { return a.stage.WaitValue() }
func (a decodePlaneStageAdapter) SignalValue() uint64 { return a.stage.SignalValue() }
func (a decodePlaneStageAdapter) InputShape() []int   { return a.stage.InputShape() }
func (a decodePlaneStageAdapter) InputSurface() anehooks.DecodePlaneInputSurface {
	return decodePlaneInputSurfaceAdapter{surface: a.stage.InputSurface()}
}
func (a decodePlaneStageAdapter) OutputSurface() any { return a.stage.OutputSurface() }

func (a decodePlaneStageAdapter) EvalPreparedSurfaceAsync(ctx context.Context) <-chan anehooks.DecodePlaneAsyncResult {
	src := a.stage.EvalPreparedSurfaceAsync(ctx)
	dst := make(chan anehooks.DecodePlaneAsyncResult, 1)
	go func() {
		res := <-src
		dst <- anehooks.DecodePlaneAsyncResult{Err: res.Err}
	}()
	return dst
}

func (a decodePlaneStageAdapter) EvalPreparedSurface(ctx context.Context) error {
	_, err := a.stage.EvalPreparedSurface(ctx)
	return err
}

type decodePlaneEventAdapter struct {
	event *mlxgoane.SharedEvent
}

func (a decodePlaneEventAdapter) SetSignaledValue(value uint64) error {
	if a.event == nil {
		return fmt.Errorf("ane decode plane event is nil")
	}
	return a.event.SetSignaledValue(value)
}

func (a decodePlaneEventAdapter) WaitCPU(value uint64, timeout time.Duration) error {
	if a.event == nil {
		return fmt.Errorf("ane decode plane event is nil")
	}
	return a.event.WaitCPU(value, timeout)
}

type decodePlaneInputSurfaceAdapter struct {
	surface *mlxgoane.IOSurfaceFloat32
}

func (a decodePlaneInputSurfaceAdapter) Write(v []float32) error {
	if a.surface == nil {
		return fmt.Errorf("ane decode plane input surface is nil")
	}
	return a.surface.Write(v)
}

func (a decodePlaneInputSurfaceAdapter) RawSurface() any {
	return a.surface
}

type decodePlaneBridgeAdapter struct {
	bridge *mlxgoane.MLXSurfaceSyncBridge
}

type decodePlaneDirectBlockAdapter struct {
	draft *mlxgoane.ANEDraftModel
}

func (a decodePlaneDirectBlockAdapter) Close() { a.draft.Close() }

func (a decodePlaneDirectBlockAdapter) InitStats() anehooks.DecodePlaneInitStats {
	return anehooks.DecodePlaneInitStats{}
}

func (a decodePlaneDirectBlockAdapter) InputSurface() anehooks.DecodePlaneInputSurface {
	return decodePlaneInputSurfaceAdapter{surface: a.draft.InputSurface()}
}

func (a decodePlaneDirectBlockAdapter) PosCosSurface() anehooks.DecodePlaneInputSurface {
	return decodePlaneInputSurfaceAdapter{surface: a.draft.PosCosSurface()}
}

func (a decodePlaneDirectBlockAdapter) PosSinSurface() anehooks.DecodePlaneInputSurface {
	return decodePlaneInputSurfaceAdapter{surface: a.draft.PosSinSurface()}
}

func (a decodePlaneDirectBlockAdapter) OutputSurface() any {
	return a.draft.OutputSurface()
}

func (a decodePlaneDirectBlockAdapter) WaitEvent() anehooks.DecodePlaneEvent {
	return decodePlaneEventAdapter{event: a.draft.WaitEvent()}
}

func (a decodePlaneDirectBlockAdapter) SignalEvent() anehooks.DecodePlaneEvent {
	return decodePlaneEventAdapter{event: a.draft.SignalEvent()}
}

func (a decodePlaneDirectBlockAdapter) WaitValue() uint64 {
	return a.draft.WaitValue()
}

func (a decodePlaneDirectBlockAdapter) SignalValue() uint64 {
	return a.draft.SignalValue()
}

func (a decodePlaneDirectBlockAdapter) EvalPreparedSurface(ctx context.Context) error {
	return a.draft.EvalPreparedSurface(ctx)
}

func (a decodePlaneDirectBlockAdapter) Reset() error {
	return a.draft.Reset()
}

func (a decodePlaneDirectBlockAdapter) DecodePosition() int {
	return a.draft.DecodePosition()
}

func (a decodePlaneDirectBlockAdapter) AdvanceDecodePosition() error {
	return a.draft.AdvanceDecodePosition()
}

func (a decodePlaneDirectBlockAdapter) SetRoPETables(cosTable, sinTable []float32, headDim, maxSeqLen int) error {
	return a.draft.SetRoPETables(cosTable, sinTable, headDim, maxSeqLen)
}

func (a decodePlaneDirectBlockAdapter) CurrentRoPESlice() ([]float32, []float32, error) {
	return a.draft.CurrentRoPESlice()
}

func (a decodePlaneDirectBlockAdapter) RestoreStatefulMILState(decodePos int, milState [][]float32) error {
	return a.draft.RestoreStatefulMILState(decodePos, milState)
}

func (a decodePlaneBridgeAdapter) CopyIntoSignalReady(dst, src *mlx.Array, stream any, waitValue uint64) error {
	return a.bridge.CopyIntoSignalReady(dst, src, streamAsMLX(stream), waitValue)
}

func (a decodePlaneBridgeAdapter) WaitForANE(stream any, signalValue uint64) error {
	return a.bridge.WaitForANE(streamAsMLX(stream), signalValue)
}

func (a decodePlaneBridgeAdapter) CopyInto(dst, src *mlx.Array, stream any) error {
	return a.bridge.CopyInto(dst, src, streamAsMLX(stream))
}

func (a decodePlaneBridgeAdapter) RMSNormInto(dst, src, weight *mlx.Array, stream any, eps float32) error {
	return a.bridge.RMSNormInto(dst, src, weight, streamAsMLX(stream), eps)
}

func (a decodePlaneBridgeAdapter) AddInto(dst, x, y *mlx.Array, stream any) error {
	return a.bridge.AddInto(dst, x, y, streamAsMLX(stream))
}

func (a decodePlaneBridgeAdapter) AddRMSNormInto(dst, x, y, weight *mlx.Array, stream any, eps float32) error {
	return a.bridge.AddRMSNormInto(dst, x, y, weight, streamAsMLX(stream), eps)
}

func (a decodePlaneBridgeAdapter) RMSNormIntoSignalReady(dst, src, weight *mlx.Array, stream any, eps float32, waitValue uint64) error {
	return a.bridge.RMSNormIntoSignalReady(dst, src, weight, streamAsMLX(stream), eps, waitValue)
}

func (a decodePlaneBridgeAdapter) AddRMSNormIntoSignalReady(dst, x, y, weight *mlx.Array, stream any, eps float32, waitValue uint64) error {
	return a.bridge.AddRMSNormIntoSignalReady(dst, x, y, weight, streamAsMLX(stream), eps, waitValue)
}

func (a decodePlaneBridgeAdapter) SignalMLXReady(stream any, value uint64) error {
	return a.bridge.SignalMLXReady(streamAsMLX(stream), value)
}

func (a decodePlaneBridgeAdapter) AliasWritableFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	s, ok := unwrapIOSurface(surface)
	if !ok {
		return nil, nil, fmt.Errorf("ane decode plane writable surface has type %T", surface)
	}
	return a.bridge.AliasWritableFloat32(s, shape)
}

func (a decodePlaneBridgeAdapter) AliasReadOnlyFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	s, ok := unwrapIOSurface(surface)
	if !ok {
		return nil, nil, fmt.Errorf("ane decode plane readonly surface has type %T", surface)
	}
	return a.bridge.AliasReadOnlyFloat32(s, shape)
}

func (a decodePlaneBridgeAdapter) FinalizeStream(stream any) error {
	return a.bridge.FinalizeStream(streamAsMLX(stream))
}

func streamAsMLX(stream any) *mlx.Stream {
	if stream == nil {
		return nil
	}
	s, ok := stream.(*mlx.Stream)
	if !ok {
		return nil
	}
	return s
}

func unwrapIOSurface(surface any) (*mlxgoane.IOSurfaceFloat32, bool) {
	if s, ok := surface.(*mlxgoane.IOSurfaceFloat32); ok {
		return s, true
	}
	type rawSurfaceProvider interface {
		RawSurface() any
	}
	p, ok := surface.(rawSurfaceProvider)
	if !ok {
		return nil, false
	}
	s, ok := p.RawSurface().(*mlxgoane.IOSurfaceFloat32)
	return s, ok
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

func durationMillis(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}
