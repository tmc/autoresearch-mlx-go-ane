//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/iosurface"
)

func newIOSurfaceFloat32WithLayout(layout compiledTensorLayout) (*IOSurfaceFloat32, error) {
	surf, err := newFloatSurfaceForLayout(layout)
	if err != nil {
		return nil, err
	}
	layoutCopy := layout
	return &IOSurfaceFloat32{
		surface: surf,
		count:   layout.logicalCount(),
		layout:  &layoutCopy,
		owned:   true,
	}, nil
}

func wrapIOSurfaceFloat32WithLayout(surface coregraphics.IOSurfaceRef, layout compiledTensorLayout) (*IOSurfaceFloat32, error) {
	if surface == 0 {
		return nil, fmt.Errorf("wrap IOSurface: surface is nil")
	}
	if err := layout.valid(); err != nil {
		return nil, fmt.Errorf("wrap IOSurface: %w", err)
	}
	layoutCopy := layout
	return &IOSurfaceFloat32{
		surface: surface,
		count:   layout.logicalCount(),
		layout:  &layoutCopy,
		owned:   false,
	}, nil
}

func writeFloat32IOSurfaceWithLayout(
	surface coregraphics.IOSurfaceRef,
	data []float32,
	layout compiledTensorLayout,
) error {
	return writeFloat32IOSurfaceAtWithLayout(surface, 0, data, layout.logicalCount(), layout)
}

func writeFloat32IOSurfaceAtWithLayout(
	surface coregraphics.IOSurfaceRef,
	offset int,
	data []float32,
	count int,
	layout compiledTensorLayout,
) error {
	if err := layout.valid(); err != nil {
		return err
	}
	if offset < 0 {
		return fmt.Errorf("write IOSurface at: invalid offset=%d", offset)
	}
	if offset+len(data) > count {
		return fmt.Errorf(
			"write IOSurface at: range [%d,%d) out of bounds [0,%d)",
			offset,
			offset+len(data),
			count,
		)
	}
	raw := iosurface.IOSurfaceRef(surface)
	if rc := iosurface.IOSurfaceLock(raw, 0, nil); rc != 0 {
		return fmt.Errorf("write IOSurface at: IOSurfaceLock rc=%d", rc)
	}
	defer iosurface.IOSurfaceUnlock(raw, 0, nil)
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		return fmt.Errorf("write IOSurface at: base address is nil")
	}
	alloc := layout.allocSize()
	if got := int(iosurface.IOSurfaceGetAllocSize(raw)); got < alloc {
		return fmt.Errorf("write IOSurface at: alloc size=%d want>=%d", got, alloc)
	}
	dst := unsafe.Slice((*byte)(base), alloc)
	for i, v := range data {
		off, err := layoutByteOffset(layout, offset+i, count)
		if err != nil {
			return err
		}
		switch layout.ElemSize {
		case 2:
			bits := float32ToFP16(v)
			dst[off] = byte(bits)
			dst[off+1] = byte(bits >> 8)
		case 4:
			*(*float32)(unsafe.Pointer(&dst[off])) = v
		default:
			return fmt.Errorf("write IOSurface at: unsupported elem size=%d", layout.ElemSize)
		}
	}
	return nil
}

func readFloat32IOSurfaceWithLayout(
	surface coregraphics.IOSurfaceRef,
	count int,
	layout compiledTensorLayout,
) ([]float32, error) {
	return readFloat32IOSurfaceAtWithLayout(surface, 0, count, count, layout)
}

func readFloat32IOSurfaceAtWithLayout(
	surface coregraphics.IOSurfaceRef,
	offset int,
	n int,
	count int,
	layout compiledTensorLayout,
) ([]float32, error) {
	if err := layout.valid(); err != nil {
		return nil, err
	}
	if offset < 0 || n < 0 {
		return nil, fmt.Errorf("read IOSurface at: invalid offset=%d n=%d", offset, n)
	}
	if offset+n > count {
		return nil, fmt.Errorf(
			"read IOSurface at: range [%d,%d) out of bounds [0,%d)",
			offset,
			offset+n,
			count,
		)
	}
	raw := iosurface.IOSurfaceRef(surface)
	if rc := iosurface.IOSurfaceLock(raw, iosurface.KIOSurfaceLockReadOnly, nil); rc != 0 {
		return nil, fmt.Errorf("read IOSurface at: IOSurfaceLock rc=%d", rc)
	}
	defer iosurface.IOSurfaceUnlock(raw, iosurface.KIOSurfaceLockReadOnly, nil)
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		return nil, fmt.Errorf("read IOSurface at: base address is nil")
	}
	alloc := layout.allocSize()
	if got := int(iosurface.IOSurfaceGetAllocSize(raw)); got < alloc {
		return nil, fmt.Errorf("read IOSurface at: alloc size=%d want>=%d", got, alloc)
	}
	src := unsafe.Slice((*byte)(base), alloc)
	out := make([]float32, n)
	for i := range out {
		off, err := layoutByteOffset(layout, offset+i, count)
		if err != nil {
			return nil, err
		}
		switch layout.ElemSize {
		case 2:
			bits := uint16(src[off]) | uint16(src[off+1])<<8
			out[i] = fp16ToFloat32(bits)
		case 4:
			out[i] = *(*float32)(unsafe.Pointer(&src[off]))
		default:
			return nil, fmt.Errorf("read IOSurface at: unsupported elem size=%d", layout.ElemSize)
		}
	}
	return out, nil
}

func zeroIOSurfaceWithLayout(surface coregraphics.IOSurfaceRef, layout compiledTensorLayout) error {
	if err := layout.valid(); err != nil {
		return err
	}
	raw := iosurface.IOSurfaceRef(surface)
	if rc := iosurface.IOSurfaceLock(raw, 0, nil); rc != 0 {
		return fmt.Errorf("fill IOSurface: IOSurfaceLock rc=%d", rc)
	}
	defer iosurface.IOSurfaceUnlock(raw, 0, nil)
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		return fmt.Errorf("fill IOSurface: base address is nil")
	}
	alloc := layout.allocSize()
	if got := int(iosurface.IOSurfaceGetAllocSize(raw)); got < alloc {
		return fmt.Errorf("fill IOSurface: alloc size=%d want>=%d", got, alloc)
	}
	clear(unsafe.Slice((*byte)(base), alloc))
	return nil
}

func layoutByteOffset(layout compiledTensorLayout, logicalIndex, count int) (int, error) {
	if logicalIndex < 0 || logicalIndex >= count {
		return 0, fmt.Errorf("logical index=%d out of bounds [0,%d)", logicalIndex, count)
	}
	rowSize := layout.Channels * layout.Width
	h := logicalIndex / rowSize
	rem := logicalIndex % rowSize
	c := rem / layout.Width
	w := rem % layout.Width
	return c*layout.PlaneStride + h*layout.RowStride + w*layout.ElemSize, nil
}

func float32ToFP16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := (b >> 16) & 0x8000
	exp := int((b>>23)&0xFF) - 127 + 15
	frac := b & 0x7FFFFF

	switch {
	case exp <= 0:
		return uint16(sign)
	case exp >= 31:
		return uint16(sign | 0x7C00)
	default:
		return uint16(sign | uint32(exp)<<10 | (frac >> 13))
	}
}

func fp16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h) & 0x3FF

	switch {
	case exp == 0:
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3FF
		fallthrough
	case exp < 31:
		return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | frac<<13)
	default:
		return math.Float32frombits(sign<<31 | 0x7F800000 | frac<<13)
	}
}
