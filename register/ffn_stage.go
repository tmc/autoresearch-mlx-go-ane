//go:build darwin && ane_appleneuralengine

package register

import (
	"context"
	"fmt"
	"time"
	"unsafe"

	mlxgoane "github.com/tmc/mlx-go-ane"
	"github.com/tmc/mlx-go-lm/exp/anehooks"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/modelir"
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
		if err == nil {
			err = compactOutput(s.output)
		}
		ch <- anehooks.AsyncResult{Err: err}
	}()
	return ch
}

func (s *ffnStage) EvalPreparedSurface(ctx context.Context) error {
	if err := s.plan.Eval(ctx); err != nil {
		return err
	}
	if err := compactOutput(s.output); err != nil {
		return fmt.Errorf("compact output: %w", err)
	}
	return nil
}

// WaitEvent returns nil because MIL FFN stages do not use Metal shared
// events. Input is written via CPU (GPU aliasing is disabled for planar
// IOSurface layouts), so there is no Metal command buffer to signal from.
// CPU-side SetSignaledValue does not reliably trigger ANE-side waits.
func (s *ffnStage) WaitEvent() anehooks.Event { return nil }

// SignalEvent returns nil for the same reason as WaitEvent.
func (s *ffnStage) SignalEvent() anehooks.Event { return nil }

func (s *ffnStage) WaitValue() uint64   { return 0 }
func (s *ffnStage) SignalValue() uint64 { return 0 }

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

// ffnBridge wraps MLXSurfaceSyncBridge with layout-aware I/O for MIL FFN
// stages. The compiled MIL model uses planar IOSurface layouts (64-byte
// aligned planes per channel), but GPU aliasing assumes contiguous memory.
//
// Input path: GPU aliasing is disabled; the decode engine falls back to the
// CPU Write() method which handles layout scattering via layoutByteOffset.
//
// Output path: After ANE completes, compactOutput gathers planar data into
// contiguous order at the IOSurface base address so the cached GPU alias
// reads correct values.
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

// bridgeAliaser methods.
//
// AliasWritableFloat32 is disabled for input because the IOSurface uses a
// planar layout. The decode engine falls back to CPU Write() which handles
// layout scattering correctly.
//
// AliasReadOnlyFloat32 creates a standard alias. The data is correct because
// compactOutput (called from WaitForANE) linearizes the planar output before
// the alias is read.
func (b *ffnBridge) AliasWritableFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	return nil, nil, fmt.Errorf("ffn bridge: GPU aliasing disabled for planar input surface")
}

func (b *ffnBridge) AliasReadOnlyFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	surf, err := extractIOSurface(surface)
	if err != nil {
		return nil, nil, fmt.Errorf("AliasReadOnlyFloat32: %w", err)
	}
	return b.bridge.AliasReadOnlyFloat32(surf, shape)
}

// extractIOSurface extracts *IOSurfaceFloat32 from any, handling both direct
// surfaces and InputSurface wrappers that provide RawSurface().
func extractIOSurface(surface any) (*mlxgoane.IOSurfaceFloat32, error) {
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

// compactOutput reads the output IOSurface using layout-aware gathering
// (planar → contiguous) and writes the contiguous data back to the surface's
// base address. This lets the GPU alias (which reads linearly) see correct data.
func compactOutput(surf *mlxgoane.IOSurfaceFloat32) error {
	// Read via layout-aware path (gathers from scattered planes).
	data, err := surf.Read()
	if err != nil {
		return fmt.Errorf("compact output: read: %w", err)
	}
	// Write contiguously to base address.
	base, byteLen, err := surf.LockWritable()
	if err != nil {
		return fmt.Errorf("compact output: lock: %w", err)
	}
	need := len(data) * 4
	if byteLen < need {
		_ = surf.UnlockWritable()
		return fmt.Errorf("compact output: alloc=%d need=%d", byteLen, need)
	}
	dst := unsafe.Slice((*float32)(base), len(data))
	copy(dst, data)
	if err := surf.UnlockWritable(); err != nil {
		return fmt.Errorf("compact output: unlock: %w", err)
	}
	return nil
}

// bridgeAdder (optional, used by denseMetalResidualAdd).
func (b *ffnBridge) AddInto(dst, x, y *mlx.Array, stream *mlx.Stream) error {
	return b.bridge.AddInto(dst, x, y, stream)
}

type directBlock struct {
	draft *mlxgoane.ANEDraftModel
}

func (b *directBlock) Close() {
	if b != nil && b.draft != nil {
		b.draft.Close()
	}
}

func (b *directBlock) InitStats() anehooks.InitStats {
	return anehooks.InitStats{}
}

func (b *directBlock) InputSurface() anehooks.InputSurface {
	return &ioSurfaceInput{surface: b.draft.InputSurface()}
}

func (b *directBlock) PosCosSurface() anehooks.InputSurface {
	return &ioSurfaceInput{surface: b.draft.PosCosSurface()}
}

func (b *directBlock) PosSinSurface() anehooks.InputSurface {
	return &ioSurfaceInput{surface: b.draft.PosSinSurface()}
}

func (b *directBlock) OutputSurface() any {
	return b.draft.OutputSurface()
}

func (b *directBlock) EvalPreparedSurface(ctx context.Context) error {
	return b.draft.EvalPreparedSurface(ctx)
}

func (b *directBlock) WaitEvent() anehooks.Event {
	ev := b.draft.WaitEvent()
	if ev == nil {
		return nil
	}
	return ev
}

func (b *directBlock) SignalEvent() anehooks.Event {
	ev := b.draft.SignalEvent()
	if ev == nil {
		return nil
	}
	return ev
}

func (b *directBlock) WaitValue() uint64 {
	return b.draft.WaitValue()
}

func (b *directBlock) SignalValue() uint64 {
	return b.draft.SignalValue()
}

func (b *directBlock) Reset() error {
	return b.draft.Reset()
}

func (b *directBlock) SetRoPETables(cosTable, sinTable []float32, headDim, maxSeqLen int) error {
	return b.draft.SetRoPETables(cosTable, sinTable, headDim, maxSeqLen)
}

func (b *directBlock) RestoreStatefulMILState(decodePos int, milState [][]float32) error {
	return b.draft.RestoreStatefulMILState(decodePos, milState)
}

func (b *directBlock) DecodePosition() int {
	return b.draft.DecodePosition()
}

func (b *directBlock) AdvanceDecodePosition() error {
	return b.draft.AdvanceDecodePosition()
}

func (b *directBlock) CurrentRoPESlice() ([]float32, []float32, error) {
	return b.draft.CurrentRoPESlice()
}

type directBlockBridge struct {
	bridge *mlxgoane.MLXSurfaceSyncBridge
}

func (b *directBlockBridge) WaitForANE(stream *mlx.Stream, signalValue uint64) error {
	return b.bridge.WaitForANE(stream, signalValue)
}

func (b *directBlockBridge) FinalizeStream(stream *mlx.Stream) error {
	return b.bridge.FinalizeStream(stream)
}

func (b *directBlockBridge) SignalMLXReady(stream *mlx.Stream, value uint64) error {
	return b.bridge.SignalMLXReady(stream, value)
}

func (b *directBlockBridge) CopyInto(dst, src *mlx.Array, stream *mlx.Stream) error {
	return b.bridge.CopyInto(dst, src, stream)
}

func (b *directBlockBridge) CopyIntoSignalReady(dst, src *mlx.Array, stream *mlx.Stream, waitValue uint64) error {
	return b.bridge.CopyIntoSignalReady(dst, src, stream, waitValue)
}

func (b *directBlockBridge) AliasWritableFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	surf, err := extractIOSurface(surface)
	if err != nil {
		return nil, nil, fmt.Errorf("AliasWritableFloat32: %w", err)
	}
	return b.bridge.AliasWritableFloat32(surf, shape)
}

func (b *directBlockBridge) AliasReadOnlyFloat32(surface any, shape []int) (*mlx.Array, func() error, error) {
	surf, err := extractIOSurface(surface)
	if err != nil {
		return nil, nil, fmt.Errorf("AliasReadOnlyFloat32: %w", err)
	}
	return b.bridge.AliasReadOnlyFloat32(surf, shape)
}

func (b *directBlockBridge) RMSNormIntoSignalReady(dst, src, weight *mlx.Array, stream *mlx.Stream, eps float32, waitValue uint64) error {
	return b.bridge.RMSNormIntoSignalReady(dst, src, weight, stream, eps, waitValue)
}

func (b *directBlockBridge) AddRMSNormIntoSignalReady(dst, x, y, weight *mlx.Array, stream *mlx.Stream, eps float32, waitValue uint64) error {
	return b.bridge.AddRMSNormIntoSignalReady(dst, x, y, weight, stream, eps, waitValue)
}

func (b *directBlockBridge) AddRMSNormInto(dst, x, y, weight *mlx.Array, stream *mlx.Stream, eps float32) error {
	return b.bridge.AddRMSNormInto(dst, x, y, weight, stream, eps)
}

func (b *directBlockBridge) AddInto(dst, x, y *mlx.Array, stream *mlx.Stream) error {
	return b.bridge.AddInto(dst, x, y, stream)
}

// transposeRowMajor transposes a [rows, cols] row-major matrix to [cols, rows].
func transposeRowMajor(m []float32, rows, cols int) []float32 {
	out := make([]float32, len(m))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = m[r*cols+c]
		}
	}
	return out
}

// NewQwen35Stage builds an FFN stage for a single dense layer.
//
// Uses MIL-based conv with fp16 weights instead of Espresso inner_product
// to avoid float16 accumulation overflow at model-scale dimensions.
//
// Weights w1 (gate), w3 (up), w2 (down) arrive in [in, out] row-major order
// from linearWeightRows (which transposes quantized weights). CompileFFNMIL
// expects [out, in] for conv weight layout, so we transpose here.
func (decodePlaneRuntime) NewQwen35Stage(dim, hidden int, cacheDir string, w1, w3, w2 []float32) (any, any, error) {
	if dim <= 0 || hidden <= 0 {
		return nil, nil, fmt.Errorf("invalid dimensions: dim=%d hidden=%d", dim, hidden)
	}

	start := time.Now()

	// Transpose weights from [in, out] to [out, in] for conv layout.
	// w1 (gate): [dim, hidden] → [hidden, dim]
	// w3 (up):   [dim, hidden] → [hidden, dim]
	// w2 (down): [hidden, dim] → [dim, hidden]
	gate := transposeRowMajor(w1, dim, hidden)
	up := transposeRowMajor(w3, dim, hidden)
	down := transposeRowMajor(w2, hidden, dim)

	// Compile MIL-based SwiGLU FFN with fp16 weights.
	milModel, err := mlxgoane.CompileFFNMIL(dim, hidden, gate, up, down)
	if err != nil {
		return nil, nil, fmt.Errorf("compile FFN MIL: %w", err)
	}

	// Build eval plan with Metal shared events.
	mapSeq := 1
	// Metal shared events are disabled for MIL FFN stages because the
	// input path uses CPU-side IOSurface writes (GPU aliasing is disabled
	// for planar layouts). CPU-side SetSignaledValue does not reliably
	// trigger ANE-side wait events, causing the ANE to execute with
	// stale/zero input data. Without events, plan.Eval runs synchronously
	// which is correct for the CPU-write → ANE-eval → CPU-read pattern.
	plan, err := mlxgoane.NewSurfaceEvalPlan(milModel.Model, milModel.Input, milModel.Output, mlxgoane.SurfaceEvalPlanConfig{})
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

func (decodePlaneRuntime) NewQwen35DirectBlock(prog any, cfg anehooks.DirectBlockConfig) (any, any, error) {
	modelProg, ok := prog.(*modelir.Program)
	if !ok {
		return nil, nil, fmt.Errorf("ane direct block: expected *modelir.Program, got %T", prog)
	}
	if modelProg == nil {
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
			modelProg,
			reifyOpts,
			cfg.HiddenDim,
			cfg.VocabSize,
			cfg.OutputDim,
			nil,
			&planCfg,
		)
	} else {
		draft, _, err = mlxgoane.NewANEDraftModelFromModelIRProgramWithPlanConfig(
			modelProg,
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
		return nil, nil, fmt.Errorf("create direct block bridge: %w", err)
	}
	return &directBlock{draft: draft}, &directBlockBridge{bridge: bridge}, nil
}
