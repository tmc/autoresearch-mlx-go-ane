//go:build darwin && cgo

package mlxaneext

import (
	"fmt"
	"unsafe"

	"github.com/tmc/apple/corefoundation"
	"github.com/tmc/apple/iosurface"
	"github.com/tmc/mlx-go/mlxc"
)

// ImportIOSurfaceFloat32 aliases an IOSurface-backed float32 buffer into an MLX
// array with writable access.
func ImportIOSurfaceFloat32(surfaceRef uint64, shape []int32) (mlxc.Array, error) {
	return importIOSurfaceFloat32(surfaceRef, shape, false)
}

// ImportIOSurfaceFloat32ReadOnly aliases an IOSurface-backed float32 buffer
// into an MLX array with a read-only IOSurface lock.
func ImportIOSurfaceFloat32ReadOnly(surfaceRef uint64, shape []int32) (mlxc.Array, error) {
	return importIOSurfaceFloat32(surfaceRef, shape, true)
}

func importIOSurfaceFloat32(surfaceRef uint64, shape []int32, readOnly bool) (mlxc.Array, error) {
	if surfaceRef == 0 {
		return mlxc.Array{}, fmt.Errorf("import iosurface float32: surface ref is zero")
	}
	if len(shape) == 0 {
		return mlxc.Array{}, fmt.Errorf("import iosurface float32: shape is empty")
	}
	raw := iosurface.IOSurfaceRef(surfaceRef)
	byteLen, err := float32ShapeBytes(shape)
	if err != nil {
		return mlxc.Array{}, fmt.Errorf("import iosurface float32: %w", err)
	}
	corefoundation.CFRetain(corefoundation.CFTypeRef(raw))
	iosurface.IOSurfaceIncrementUseCount(raw)
	lockOpts := iosurface.IOSurfaceLockOptions(0)
	if readOnly {
		lockOpts = iosurface.KIOSurfaceLockReadOnly
	}
	if rc := iosurface.IOSurfaceLock(raw, lockOpts, nil); rc != 0 {
		iosurface.IOSurfaceDecrementUseCount(raw)
		corefoundation.CFRelease(corefoundation.CFTypeRef(raw))
		return mlxc.Array{}, fmt.Errorf("import iosurface float32: IOSurfaceLock rc=%d", rc)
	}
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		iosurface.IOSurfaceUnlock(raw, lockOpts, nil)
		iosurface.IOSurfaceDecrementUseCount(raw)
		corefoundation.CFRelease(corefoundation.CFTypeRef(raw))
		return mlxc.Array{}, fmt.Errorf("import iosurface float32: base address is nil")
	}
	if got := iosurface.IOSurfaceGetAllocSize(raw); got < uintptr(byteLen) {
		iosurface.IOSurfaceUnlock(raw, lockOpts, nil)
		iosurface.IOSurfaceDecrementUseCount(raw)
		corefoundation.CFRelease(corefoundation.CFTypeRef(raw))
		return mlxc.Array{}, fmt.Errorf("import iosurface float32: alloc size=%d want>=%d", got, byteLen)
	}
	arr := mlxc.ArrayNewDataManaged(base, shape, mlxc.Float32, func(_ unsafe.Pointer) {
		_ = iosurface.IOSurfaceUnlock(raw, lockOpts, nil)
		iosurface.IOSurfaceDecrementUseCount(raw)
		corefoundation.CFRelease(corefoundation.CFTypeRef(raw))
	})
	return arr, nil
}

func float32ShapeBytes(shape []int32) (int, error) {
	total := int64(1)
	for _, dim := range shape {
		if dim <= 0 {
			return 0, fmt.Errorf("invalid dim=%d", dim)
		}
		total *= int64(dim)
	}
	if total <= 0 {
		return 0, fmt.Errorf("invalid element count")
	}
	if total > int64(^uint(0)>>1)/4 {
		return 0, fmt.Errorf("shape is too large")
	}
	return int(total) * int(unsafe.Sizeof(float32(0))), nil
}
