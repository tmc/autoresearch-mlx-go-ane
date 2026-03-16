//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"math"
	"testing"
	"time"

	"github.com/tmc/apple/iosurface"
	"github.com/tmc/mlx-go/exp/mlxcext"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/fast"
)

func TestMLXSurfaceSyncBridgeAliasReadOnly(t *testing.T) {
	surf, err := NewIOSurfaceFloat32(4)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()
	if err := surf.Write([]float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Write: %v", err)
	}

	bridge, err := NewMLXSurfaceSyncBridge(nil, nil)
	if err != nil {
		t.Fatalf("NewMLXSurfaceSyncBridge: %v", err)
	}
	defer bridge.Close()

	arr, cleanup, err := bridge.AliasReadOnlyFloat32(surf, []int{2, 2})
	if err != nil {
		t.Fatalf("AliasReadOnlyFloat32: %v", err)
	}
	defer cleanup()

	got, err := mlx.ToSlice[float32](arr)
	if err != nil {
		t.Fatalf("ToSlice: %v", err)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestMLXSurfaceSyncBridgeCopyInto(t *testing.T) {
	bridge, err := NewMLXSurfaceSyncBridge(nil, nil)
	if err != nil {
		t.Fatalf("NewMLXSurfaceSyncBridge: %v", err)
	}
	defer bridge.Close()

	dst := mlx.NewArray([]float32{0, 0, 0, 0}, 2, 2)
	defer dst.Free()
	src := mlx.NewArray([]float32{5, 6, 7, 8}, 2, 2)
	defer src.Free()
	if err := bridge.CopyInto(dst, src, nil); err != nil {
		t.Fatalf("CopyInto: %v", err)
	}
	if err := mlx.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	got, err := mlx.ToSlice[float32](dst)
	if err != nil {
		t.Fatalf("ToSlice: %v", err)
	}
	want := []float32{5, 6, 7, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestMLXSurfaceSyncBridgeCopyIntoSignalReady(t *testing.T) {
	if !mlxcext.HasMetalInterop() {
		t.Skip("mlxcext Metal interop unavailable")
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
	wait, err := WrapSharedEventPort(port, 0)
	if err != nil {
		t.Fatalf("WrapSharedEventPort: %v", err)
	}
	bridge, err := NewMLXSurfaceSyncBridge(wait, nil)
	if err != nil {
		t.Fatalf("NewMLXSurfaceSyncBridge: %v", err)
	}
	defer bridge.Close()

	surf, err := NewIOSurfaceFloat32(4)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()
	dst, cleanup, err := bridge.AliasWritableFloat32(surf, []int{2, 2})
	if err != nil {
		t.Fatalf("AliasWritableFloat32: %v", err)
	}
	defer cleanup()
	src := mlx.NewArray([]float32{5, 6, 7, 8}, 2, 2)
	defer src.Free()
	if err := bridge.CopyIntoSignalReady(dst, src, nil, 2); err != nil {
		t.Fatalf("CopyIntoSignalReady: %v", err)
	}
	if err := bridge.FinalizeStream(nil); err != nil {
		t.Fatalf("FinalizeStream: %v", err)
	}
	if err := wait.WaitCPU(2, time.Second); err != nil {
		t.Fatalf("WaitCPU: %v", err)
	}
	got, err := surf.Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	want := []float32{5, 6, 7, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestMLXSurfaceSyncBridgeCopyIntoSignalReadyLazySource(t *testing.T) {
	if !mlxcext.HasMetalInterop() {
		t.Skip("mlxcext Metal interop unavailable")
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
	wait, err := WrapSharedEventPort(port, 0)
	if err != nil {
		t.Fatalf("WrapSharedEventPort: %v", err)
	}
	bridge, err := NewMLXSurfaceSyncBridge(wait, nil)
	if err != nil {
		t.Fatalf("NewMLXSurfaceSyncBridge: %v", err)
	}
	defer bridge.Close()

	surf, err := NewIOSurfaceFloat32(4)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()
	dst, cleanup, err := bridge.AliasWritableFloat32(surf, []int{2, 2})
	if err != nil {
		t.Fatalf("AliasWritableFloat32: %v", err)
	}
	defer cleanup()

	base := mlx.NewArray([]float32{1, 2, 3, 4}, 2, 2)
	defer base.Free()
	src, err := mlx.Add(base, base, nil)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	defer src.Free()

	if err := bridge.CopyIntoSignalReady(dst, src, nil, 2); err != nil {
		t.Fatalf("CopyIntoSignalReady lazy source: %v", err)
	}
	if err := bridge.FinalizeStream(nil); err != nil {
		t.Fatalf("FinalizeStream: %v", err)
	}
	if err := wait.WaitCPU(2, time.Second); err != nil {
		t.Fatalf("WaitCPU: %v", err)
	}
	got, err := surf.Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	want := []float32{2, 4, 6, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestMLXSurfaceSyncBridgeSignalMLXReady(t *testing.T) {
	if !mlxcext.HasMetalInterop() {
		t.Skip("mlxcext Metal interop unavailable")
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
	wait, err := WrapSharedEventPort(port, 0)
	if err != nil {
		t.Fatalf("WrapSharedEventPort: %v", err)
	}
	bridge, err := NewMLXSurfaceSyncBridge(wait, nil)
	if err != nil {
		t.Fatalf("NewMLXSurfaceSyncBridge: %v", err)
	}
	defer bridge.Close()
	if err := bridge.SignalMLXReady(nil, 2); err != nil {
		t.Fatalf("SignalMLXReady: %v", err)
	}
	if err := bridge.FinalizeStream(nil); err != nil {
		t.Fatalf("FinalizeStream: %v", err)
	}
	if err := wait.WaitCPU(2, time.Second); err != nil {
		t.Fatalf("WaitCPU: %v", err)
	}
}

func TestMLXSurfaceSyncBridgeRMSNormIntoSignalReady(t *testing.T) {
	if !mlxcext.HasMetalInterop() {
		t.Skip("mlxcext Metal interop unavailable")
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
	wait, err := WrapSharedEventPort(port, 0)
	if err != nil {
		t.Fatalf("WrapSharedEventPort: %v", err)
	}
	bridge, err := NewMLXSurfaceSyncBridge(wait, nil)
	if err != nil {
		t.Fatalf("NewMLXSurfaceSyncBridge: %v", err)
	}
	defer bridge.Close()

	surf, err := NewIOSurfaceFloat32(4)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()
	dst, cleanup, err := bridge.AliasWritableFloat32(surf, []int{1, 1, 4})
	if err != nil {
		t.Fatalf("AliasWritableFloat32: %v", err)
	}
	defer cleanup()
	src := mlx.NewArray([]float32{1, 2, 3, 4}, 1, 1, 4)
	defer src.Free()
	weight := mlx.NewArray([]float32{1, 1, 1, 1}, 4)
	defer weight.Free()

	wantArr, err := fast.RMSNorm(src, weight, 1e-5, nil)
	if err != nil {
		t.Fatalf("fast.RMSNorm: %v", err)
	}
	defer wantArr.Free()
	want, err := mlx.ToSlice[float32](wantArr)
	if err != nil {
		t.Fatalf("ToSlice want: %v", err)
	}

	if err := bridge.RMSNormIntoSignalReady(dst, src, weight, nil, 1e-5, 2); err != nil {
		t.Fatalf("RMSNormIntoSignalReady: %v", err)
	}
	if err := wait.WaitCPU(2, time.Second); err != nil {
		t.Fatalf("WaitCPU: %v", err)
	}
	got, err := surf.Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("got[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestMLXSurfaceSyncBridgeAddRMSNormIntoSignalReady(t *testing.T) {
	if !mlxcext.HasMetalInterop() {
		t.Skip("mlxcext Metal interop unavailable")
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
	wait, err := WrapSharedEventPort(port, 0)
	if err != nil {
		t.Fatalf("WrapSharedEventPort: %v", err)
	}
	bridge, err := NewMLXSurfaceSyncBridge(wait, nil)
	if err != nil {
		t.Fatalf("NewMLXSurfaceSyncBridge: %v", err)
	}
	defer bridge.Close()

	surf, err := NewIOSurfaceFloat32(4)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()
	dst, cleanup, err := bridge.AliasWritableFloat32(surf, []int{1, 1, 4})
	if err != nil {
		t.Fatalf("AliasWritableFloat32: %v", err)
	}
	defer cleanup()
	x := mlx.NewArray([]float32{1, 2, 3, 4}, 1, 1, 4)
	defer x.Free()
	y := mlx.NewArray([]float32{0.5, 0.5, 0.5, 0.5}, 1, 1, 4)
	defer y.Free()
	weight := mlx.NewArray([]float32{1, 1, 1, 1}, 4)
	defer weight.Free()

	sum, err := mlx.Add(x, y, nil)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	defer sum.Free()
	wantArr, err := fast.RMSNorm(sum, weight, 1e-5, nil)
	if err != nil {
		t.Fatalf("fast.RMSNorm: %v", err)
	}
	defer wantArr.Free()
	want, err := mlx.ToSlice[float32](wantArr)
	if err != nil {
		t.Fatalf("ToSlice want: %v", err)
	}

	if err := bridge.AddRMSNormIntoSignalReady(dst, x, y, weight, nil, 1e-5, 2); err != nil {
		t.Fatalf("AddRMSNormIntoSignalReady: %v", err)
	}
	if err := wait.WaitCPU(2, time.Second); err != nil {
		t.Fatalf("WaitCPU: %v", err)
	}
	got, err := surf.Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("got[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}
