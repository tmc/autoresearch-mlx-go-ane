//go:build darwin && ane_appleneuralengine

package register

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/mlx-go-lm/exp/anehooks"
	mlxgoane "github.com/tmc/mlx-go-ane"
	"github.com/tmc/mlx-go/mlx"
)

// ffnStage wraps a compiled FFN MIL model with IOSurface I/O.
// Satisfies stageEvaluator + synchronizer via Go structural typing.
type ffnStage struct {
	milModel *mlxgoane.FFNMILModel
	plan     *mlxgoane.SurfaceEvalPlan
	input    *mlxgoane.IOSurfaceFloat32
	output   *mlxgoane.IOSurfaceFloat32
	dim      int
	mapSeq   int
	initDur  time.Duration
}

func (s *ffnStage) InputSurface() anehooks.InputSurface {
	return &ioSurfaceInput{surface: s.input}
}

func (s *ffnStage) InputShape() []int {
	return []int{s.dim, s.mapSeq}
}

func (s *ffnStage) OutputSurface() any {
	return s.output
}

func (s *ffnStage) ModelDim() int {
	return s.dim
}

func (s *ffnStage) MapSeq() int {
	return s.mapSeq
}

func (s *ffnStage) EvalPreparedSurfaceAsync(ctx context.Context) <-chan anehooks.AsyncResult {
	ch := make(chan anehooks.AsyncResult, 1)
	go func() {
		err := s.plan.Eval(ctx)
		ch <- anehooks.AsyncResult{Err: err}
	}()
	return ch
}

func (s *ffnStage) EvalPreparedSurface(ctx context.Context) error {
	return s.plan.Eval(ctx)
}

func (s *ffnStage) WaitEvent() anehooks.Event {
	ev := s.plan.WaitEvent()
	if ev == nil {
		return nil
	}
	return ev
}

func (s *ffnStage) SignalEvent() anehooks.Event {
	ev := s.plan.SignalEvent()
	if ev == nil {
		return nil
	}
	return ev
}

func (s *ffnStage) WaitValue() uint64  { return s.plan.WaitValue() }
func (s *ffnStage) SignalValue() uint64 { return s.plan.SignalValue() }

func (s *ffnStage) InitStats() anehooks.InitStats {
	return anehooks.InitStats{CompileLoad: s.initDur}
}

func (s *ffnStage) Close() {
	if s.plan != nil {
		s.plan.Close()
	}
	if s.milModel != nil {
		s.milModel.Close()
	}
}

// ioSurfaceInput adapts IOSurfaceFloat32 to anehooks.InputSurface.
type ioSurfaceInput struct {
	surface *mlxgoane.IOSurfaceFloat32
}

func (s *ioSurfaceInput) Write(data []float32) error { return s.surface.Write(data) }
func (s *ioSurfaceInput) RawSurface() any            { return s.surface }

// ffnBridge wraps MLXSurfaceSyncBridge with type assertions for the decode
// engine's bridge interfaces (bridgeAliaser, bridgeTransfer, bridgeSyncer).
type ffnBridge struct {
	bridge *mlxgoane.MLXSurfaceSyncBridge
}

// bridgeSyncer methods.
func (b *ffnBridge) WaitForANE(stream *mlx.Stream, signalValue uint64) error {
	return b.bridge.WaitForANE(stream, signalValue)
}

func (b *ffnBridge) FinalizeStream(stream *mlx.Stream) error {
	return b.bridge.FinalizeStream(stream)
}

// bridgeTransfer methods.
func (b *ffnBridge) SignalMLXReady(stream *mlx.Stream, value uint64) error {
	return b.bridge.SignalMLXReady(stream, value)
}

func (b *ffnBridge) CopyInto(dst, src *mlx.Array, stream *mlx.Stream) error {
	return b.bridge.CopyInto(dst, src, stream)
}

func (b *ffnBridge) CopyIntoSignalReady(dst, src *mlx.Array, stream *mlx.Stream, waitValue uint64) error {
	return b.bridge.CopyIntoSignalReady(dst, src, stream, waitValue)
}

// unwrapSurface extracts *IOSurfaceFloat32 from any, handling both direct
// surfaces and InputSurface wrappers that provide RawSurface().
func unwrapSurface(surface any) (*mlxgoane.IOSurfaceFloat32, error) {
	if surf, ok := surface.(*mlxgoane.IOSurfaceFloat32); ok {
		return surf, nil
	}
	type rawSurfacer interface {
		RawSurface() any
	}
	if rs, ok := surface.(rawSurfacer); ok {
		if surf, ok := rs.RawSurface().(*mlxgoane.IOSurfaceFloat32); ok {
			return surf, nil
		}
	}
	return nil, fmt.Errorf("expected *IOSurfaceFloat32, got %T", surface)
}

// bridgeAliaser methods — surface may be *IOSurfaceFloat32 or an InputSurface wrapper.
func (b *ffnBridge) AliasWritableFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	surf, err := unwrapSurface(surface)
	if err != nil {
		return nil, nil, fmt.Errorf("AliasWritableFloat32: %w", err)
	}
	return b.bridge.AliasWritableFloat32(surf, shape)
}

func (b *ffnBridge) AliasReadOnlyFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	surf, err := unwrapSurface(surface)
	if err != nil {
		return nil, nil, fmt.Errorf("AliasReadOnlyFloat32: %w", err)
	}
	return b.bridge.AliasReadOnlyFloat32(surf, shape)
}

// bridgeAdder (optional, used by denseMetalResidualAdd).
func (b *ffnBridge) AddInto(dst, x, y *mlx.Array, stream *mlx.Stream) error {
	return b.bridge.AddInto(dst, x, y, stream)
}

// NewQwen35Stage builds an FFN stage for a single dense layer.
//
// Uses MIL-based conv with fp16 weights instead of Espresso inner_product
// to avoid float16 accumulation overflow at model-scale dimensions.
func (decodePlaneRuntime) NewQwen35Stage(dim, hidden int, cacheDir string, w1, w3, w2 []float32) (any, any, error) {
	if dim <= 0 || hidden <= 0 {
		return nil, nil, fmt.Errorf("invalid dimensions: dim=%d hidden=%d", dim, hidden)
	}

	start := time.Now()

	// Compile MIL-based SwiGLU FFN with fp16 weights.
	milModel, err := mlxgoane.CompileFFNMIL(dim, hidden, w1, w3, w2)
	if err != nil {
		return nil, nil, fmt.Errorf("compile FFN MIL: %w", err)
	}

	// Build eval plan with Metal shared events.
	mapSeq := 1
	plan, err := mlxgoane.NewSurfaceEvalPlan(milModel.Model, milModel.Input, milModel.Output, mlxgoane.SurfaceEvalPlanConfig{
		EnableMetalWait:   true,
		EnableMetalSignal: true,
	})
	if err != nil {
		milModel.Close()
		return nil, nil, fmt.Errorf("create eval plan: %w", err)
	}

	stage := &ffnStage{
		milModel: milModel,
		plan:     plan,
		input:    milModel.Input,
		output:   milModel.Output,
		dim:      dim,
		mapSeq:   mapSeq,
		initDur:  time.Since(start),
	}

	// Build MLX↔ANE bridge.
	bridge, err := mlxgoane.NewMLXSurfaceSyncBridge(plan.WaitEvent(), plan.SignalEvent())
	if err != nil {
		stage.Close()
		return nil, nil, fmt.Errorf("create bridge: %w", err)
	}

	return stage, &ffnBridge{bridge: bridge}, nil
}

// NewQwen35DirectBlock is not yet implemented.
func (decodePlaneRuntime) NewQwen35DirectBlock(prog any, cfg anehooks.DirectBlockConfig) (any, any, error) {
	return nil, nil, fmt.Errorf("NewQwen35DirectBlock: not yet implemented")
}
