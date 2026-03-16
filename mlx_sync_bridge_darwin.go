//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"

	"github.com/tmc/mlx-go-ane/internal/mlxaneext"
	"github.com/tmc/mlx-go/exp/mlxcext"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlxc"
)

// MLXSurfaceSyncBridge pairs ANE-side shared-event ports with MLX-side Metal
// stream interop and shared IOSurface-backed MLX arrays.
type MLXSurfaceSyncBridge struct {
	waitPort   uint32
	signalPort uint32
}

// NewMLXSurfaceSyncBridge records the provided ANE shared-event ports for
// MLX-side Metal stream interop.
func NewMLXSurfaceSyncBridge(wait, signal *SharedEvent) (*MLXSurfaceSyncBridge, error) {
	needInterop := (wait != nil && wait.Port() != 0) || (signal != nil && signal.Port() != 0)
	if needInterop && !mlxcext.HasMetalInterop() {
		return nil, fmt.Errorf("mlx surface sync bridge: mlxcext Metal interop unavailable")
	}
	return &MLXSurfaceSyncBridge{
		waitPort:   portOf(wait),
		signalPort: portOf(signal),
	}, nil
}

// Close releases bridge-owned resources.
func (b *MLXSurfaceSyncBridge) Close() {}

func portOf(ev *SharedEvent) uint32 {
	if ev == nil {
		return 0
	}
	return ev.Port()
}

func mlxcStreamOrDefault(stream *mlx.Stream) mlxc.Stream {
	if stream != nil {
		return stream.MlxcStream()
	}
	return mlx.DefaultStream().MlxcStream()
}

func shapeToInt32(shape []int) ([]int32, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("shape is empty")
	}
	out := make([]int32, len(shape))
	total := 1
	for i, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid shape dim=%d", dim)
		}
		total *= dim
		out[i] = int32(dim)
	}
	return out, nil
}

// WaitForANE encodes a wait on the ANE->MLX signal event.
func (b *MLXSurfaceSyncBridge) WaitForANE(stream *mlx.Stream, value uint64) error {
	if b == nil || b.signalPort == 0 {
		return fmt.Errorf("mlx surface sync bridge: signal event is unavailable")
	}
	return mlxcext.WaitSharedEvent(mlxcStreamOrDefault(stream), b.signalPort, value)
}

// SignalMLXReady encodes a signal on the MLX->ANE wait event.
func (b *MLXSurfaceSyncBridge) SignalMLXReady(stream *mlx.Stream, value uint64) error {
	if b == nil || b.waitPort == 0 {
		return fmt.Errorf("mlx surface sync bridge: wait event is unavailable")
	}
	return mlxcext.SignalSharedEvent(mlxcStreamOrDefault(stream), b.waitPort, value)
}

// FinalizeStream ends encoding and commits the current MLX GPU command buffer.
func (b *MLXSurfaceSyncBridge) FinalizeStream(stream *mlx.Stream) error {
	return mlxcext.FinalizeMetalStream(mlxcStreamOrDefault(stream))
}

// AliasReadOnlyFloat32 aliases a read-only IOSurface view into an MLX array.
func (b *MLXSurfaceSyncBridge) AliasReadOnlyFloat32(surface *IOSurfaceFloat32, shape []int) (*mlx.Array, func() error, error) {
	if surface == nil {
		return nil, nil, fmt.Errorf("mlx surface sync bridge: surface is nil")
	}
	shape32, err := shapeToInt32(shape)
	if err != nil {
		return nil, nil, err
	}
	arr, err := mlxaneext.ImportIOSurfaceFloat32ReadOnly(uint64(surface.Ref()), shape32)
	if err != nil {
		return nil, nil, err
	}
	wrapped := mlx.NewArrayFromMlxc(arr)
	cleanup := func() error {
		if wrapped == nil {
			return nil
		}
		err := wrapped.Free()
		wrapped = nil
		return err
	}
	return wrapped, cleanup, nil
}

// AliasWritableFloat32 aliases a writable IOSurface view into an MLX array.
func (b *MLXSurfaceSyncBridge) AliasWritableFloat32(surface *IOSurfaceFloat32, shape []int) (*mlx.Array, func() error, error) {
	if surface == nil {
		return nil, nil, fmt.Errorf("mlx surface sync bridge: surface is nil")
	}
	shape32, err := shapeToInt32(shape)
	if err != nil {
		return nil, nil, err
	}
	arr, err := mlxaneext.ImportIOSurfaceFloat32(uint64(surface.Ref()), shape32)
	if err != nil {
		return nil, nil, err
	}
	wrapped := mlx.NewArrayFromMlxc(arr)
	cleanup := func() error {
		if wrapped == nil {
			return nil
		}
		err := wrapped.Free()
		wrapped = nil
		return err
	}
	return wrapped, cleanup, nil
}

// CopyInto copies src into the storage backing dst on stream.
func (b *MLXSurfaceSyncBridge) CopyInto(dst, src *mlx.Array, stream *mlx.Stream) error {
	if dst == nil || dst.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: dst is nil")
	}
	if src == nil || src.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: src is nil")
	}
	if err := mlxcext.PrepareMetalArray(src.MlxcArray()); err != nil {
		return fmt.Errorf("mlx surface sync bridge: prepare src: %w", err)
	}
	return mlxcext.CopyContiguous(dst.MlxcArray(), src.MlxcArray(), mlxcStreamOrDefault(stream))
}

// AddInto computes dst = x + y on the MLX GPU stream into a float32 dst.
func (b *MLXSurfaceSyncBridge) AddInto(dst, x, y *mlx.Array, stream *mlx.Stream) error {
	if dst == nil || dst.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: dst is nil")
	}
	if x == nil || x.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: x is nil")
	}
	if y == nil || y.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: y is nil")
	}
	if err := mlxcext.PrepareMetalArray(x.MlxcArray()); err != nil {
		return fmt.Errorf("mlx surface sync bridge: prepare x: %w", err)
	}
	if err := mlxcext.PrepareMetalArray(y.MlxcArray()); err != nil {
		return fmt.Errorf("mlx surface sync bridge: prepare y: %w", err)
	}
	return mlxcext.AddInto(dst.MlxcArray(), x.MlxcArray(), y.MlxcArray(), mlxcStreamOrDefault(stream))
}

// CopyIntoSignalReady copies src into dst and signals the MLX->ANE wait event
// on the same MLX GPU command buffer.
func (b *MLXSurfaceSyncBridge) CopyIntoSignalReady(dst, src *mlx.Array, stream *mlx.Stream, value uint64) error {
	if b == nil || b.waitPort == 0 {
		return fmt.Errorf("mlx surface sync bridge: wait event is unavailable")
	}
	if dst == nil || dst.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: dst is nil")
	}
	if src == nil || src.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: src is nil")
	}
	if err := mlxcext.PrepareMetalArray(src.MlxcArray()); err != nil {
		return fmt.Errorf("mlx surface sync bridge: prepare src: %w", err)
	}
	return mlxcext.CopyContiguousSignalEvent(dst.MlxcArray(), src.MlxcArray(), mlxcStreamOrDefault(stream), b.waitPort, value)
}

// RMSNormInto computes a seq=1 contiguous float32 RMSNorm from src into dst on
// stream using weight and eps.
func (b *MLXSurfaceSyncBridge) RMSNormInto(dst, src, weight *mlx.Array, stream *mlx.Stream, eps float32) error {
	if dst == nil || dst.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: dst is nil")
	}
	if src == nil || src.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: src is nil")
	}
	if weight == nil || weight.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: weight is nil")
	}
	return mlxcext.RMSNormInto(dst.MlxcArray(), src.MlxcArray(), weight.MlxcArray(), mlxcStreamOrDefault(stream), eps)
}

// AddRMSNormInto computes dst = RMSNorm(x + y, weight) on the MLX GPU stream.
func (b *MLXSurfaceSyncBridge) AddRMSNormInto(dst, x, y, weight *mlx.Array, stream *mlx.Stream, eps float32) error {
	if dst == nil || dst.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: dst is nil")
	}
	if x == nil || x.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: x is nil")
	}
	if y == nil || y.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: y is nil")
	}
	if weight == nil || weight.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: weight is nil")
	}
	if err := mlxcext.PrepareMetalArray(x.MlxcArray()); err != nil {
		return fmt.Errorf("mlx surface sync bridge: prepare x: %w", err)
	}
	if err := mlxcext.PrepareMetalArray(y.MlxcArray()); err != nil {
		return fmt.Errorf("mlx surface sync bridge: prepare y: %w", err)
	}
	if err := mlxcext.PrepareMetalArray(weight.MlxcArray()); err != nil {
		return fmt.Errorf("mlx surface sync bridge: prepare weight: %w", err)
	}
	return mlxcext.AddRMSNormInto(dst.MlxcArray(), x.MlxcArray(), y.MlxcArray(), weight.MlxcArray(), mlxcStreamOrDefault(stream), eps)
}

// RMSNormIntoSignalReady computes the same RMSNorm and signals the MLX->ANE
// wait event on the same MLX GPU command buffer.
func (b *MLXSurfaceSyncBridge) RMSNormIntoSignalReady(dst, src, weight *mlx.Array, stream *mlx.Stream, eps float32, value uint64) error {
	if b == nil || b.waitPort == 0 {
		return fmt.Errorf("mlx surface sync bridge: wait event is unavailable")
	}
	if dst == nil || dst.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: dst is nil")
	}
	if src == nil || src.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: src is nil")
	}
	if weight == nil || weight.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: weight is nil")
	}
	return mlxcext.RMSNormIntoSignalEvent(dst.MlxcArray(), src.MlxcArray(), weight.MlxcArray(), mlxcStreamOrDefault(stream), eps, b.waitPort, value)
}

// AddRMSNormIntoSignalReady computes dst = RMSNorm(x + y, weight) on the MLX
// GPU stream and signals the MLX->ANE wait event on the same command buffer.
func (b *MLXSurfaceSyncBridge) AddRMSNormIntoSignalReady(dst, x, y, weight *mlx.Array, stream *mlx.Stream, eps float32, value uint64) error {
	if b == nil || b.waitPort == 0 {
		return fmt.Errorf("mlx surface sync bridge: wait event is unavailable")
	}
	if dst == nil || dst.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: dst is nil")
	}
	if x == nil || x.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: x is nil")
	}
	if y == nil || y.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: y is nil")
	}
	if weight == nil || weight.IsNil() {
		return fmt.Errorf("mlx surface sync bridge: weight is nil")
	}
	return mlxcext.AddRMSNormIntoSignalEvent(dst.MlxcArray(), x.MlxcArray(), y.MlxcArray(), weight.MlxcArray(), mlxcStreamOrDefault(stream), eps, b.waitPort, value)
}
