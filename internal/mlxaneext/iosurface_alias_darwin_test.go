//go:build darwin

package mlxaneext

import (
	"testing"
	"unsafe"

	"github.com/tmc/apple/corefoundation"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/iosurface"
	"github.com/tmc/mlx-go/exp/mlxcext"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlxc"
)

func newTestFloatSurface(t *testing.T, count int) coregraphics.IOSurfaceRef {
	t.Helper()
	bytes := count * 4
	props := foundation.NewMutableDictionaryWithCapacity(6)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(bytes),
		foundation.NewStringWithString(iosurface.KIOSurfaceWidth),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(1),
		foundation.NewStringWithString(iosurface.KIOSurfaceHeight),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(1),
		foundation.NewStringWithString(iosurface.KIOSurfaceBytesPerElement),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(bytes),
		foundation.NewStringWithString(iosurface.KIOSurfaceBytesPerRow),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(bytes),
		foundation.NewStringWithString(iosurface.KIOSurfaceAllocSize),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(0),
		foundation.NewStringWithString(iosurface.KIOSurfacePixelFormat),
	)
	raw := iosurface.IOSurfaceCreate(corefoundation.CFDictionaryRef(props.GetID()))
	if raw == 0 {
		t.Fatal("IOSurfaceCreate returned nil")
	}
	return coregraphics.IOSurfaceRef(raw)
}

func TestImportIOSurfaceFloat32(t *testing.T) {
	if !mlxcext.HasMetalInterop() {
		t.Skip("ANE interop unavailable")
	}

	surf := newTestFloatSurface(t, 4)
	defer corefoundation.CFRelease(corefoundation.CFTypeRef(surf))
	if rc := iosurface.IOSurfaceLock(iosurface.IOSurfaceRef(surf), 0, nil); rc != 0 {
		t.Fatalf("IOSurfaceLock: rc=%d", rc)
	}
	base := (*[4]float32)(iosurface.IOSurfaceGetBaseAddress(iosurface.IOSurfaceRef(surf)))
	base[0], base[1], base[2], base[3] = 1, 2, 3, 4
	_ = iosurface.IOSurfaceUnlock(iosurface.IOSurfaceRef(surf), 0, nil)

	arr, err := ImportIOSurfaceFloat32(uint64(surf), []int32{2, 2})
	if err != nil {
		t.Fatalf("ImportIOSurfaceFloat32: %v", err)
	}
	defer mlxc.ArrayFree(arr)

	wrapped := mlx.NewArrayFromMlxc(arr)
	slice, err := mlx.ToSlice[float32](wrapped)
	if err != nil {
		t.Fatalf("ToSlice: %v", err)
	}
	want := []float32{1, 2, 3, 4}
	for i, got := range slice {
		if got != want[i] {
			t.Fatalf("slice[%d]=%v want %v", i, got, want[i])
		}
	}
}

func TestCopyContiguousSignalEvent(t *testing.T) {
	if !mlxcext.HasMetalInterop() {
		t.Skip("ANE interop unavailable")
	}
	ev := iosurface.NewIOSurfaceSharedEvent()
	if ev.ID == 0 {
		t.Skip("IOSurfaceSharedEvent unavailable")
	}
	defer ev.Release()
	port := ev.EventPort()
	if port == 0 {
		t.Skip("IOSurfaceSharedEvent port unavailable")
	}

	surf := newTestFloatSurface(t, 4)
	defer corefoundation.CFRelease(corefoundation.CFTypeRef(surf))

	dst, err := ImportIOSurfaceFloat32(uint64(surf), []int32{2, 2})
	if err != nil {
		t.Fatalf("ImportIOSurfaceFloat32: %v", err)
	}
	defer mlxc.ArrayFree(dst)

	srcBuf := []float32{5, 6, 7, 8}
	src, err := mlxcext.NewManagedArray(unsafe.Pointer(&srcBuf[0]), []int32{2, 2}, mlxc.Float32, nil)
	if err != nil {
		t.Fatalf("NewManagedArray: %v", err)
	}
	defer mlxc.ArrayFree(src)

	stream := mlx.DefaultStream().MlxcStream()
	if err := mlxcext.CopyContiguousSignalEvent(dst, src, stream, port, 2); err != nil {
		t.Fatalf("CopyContiguousSignalEvent: %v", err)
	}
}
