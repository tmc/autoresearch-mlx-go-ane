//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/tmc/apple/objc"
)

// SharedEvent wraps an IOSurfaceSharedEvent exposed by an eval plan.
//
// A SharedEvent provides the event Mach port used by Metal<->ANE request
// wiring and, when the underlying shared-event object is retained by the plan,
// a direct way to update its signaled value from Go.
type SharedEvent struct {
	port     uint32
	value    *uint64
	sharedID objc.ID
}

// WrapSharedEventPort creates a SharedEvent wrapper for an existing Mach port.
//
// The returned wrapper can be used for CPU wait/signal helpers but cannot call
// SetSignaledValue because no retained shared-event object is available.
func WrapSharedEventPort(port uint32, value uint64) (*SharedEvent, error) {
	if port == 0 {
		return nil, fmt.Errorf("wrap shared event: event port is zero")
	}
	v := value
	return &SharedEvent{port: port, value: &v}, nil
}

func newSharedEvent(port uint32, value *uint64, sharedID objc.ID, _ string) *SharedEvent {
	if port == 0 {
		return nil
	}
	return &SharedEvent{port: port, value: value, sharedID: sharedID}
}

// Port returns the IOSurfaceSharedEvent Mach port.
func (e *SharedEvent) Port() uint32 {
	if e == nil {
		return 0
	}
	return e.port
}

// Value returns the current target/signaled value tracked by the wrapper.
func (e *SharedEvent) Value() uint64 {
	if e == nil || e.value == nil {
		return 0
	}
	return *e.value
}

// SetSignaledValue updates the underlying IOSurfaceSharedEvent object.
func (e *SharedEvent) SetSignaledValue(value uint64) error {
	if e == nil {
		return fmt.Errorf("shared event: event is nil")
	}
	if e.sharedID == 0 {
		return fmt.Errorf("shared event: no retained shared-event object attached")
	}
	if err := objcShimSetSharedEventValue(e.sharedID, value); err != nil {
		return err
	}
	if e.value != nil {
		*e.value = value
	}
	return nil
}

// SignalCPU signals the shared event from the host CPU.
//
// When the optional bridge runtime is available it uses the bridge's direct
// CPU-side shared-event signaling path. Otherwise it falls back to the ObjC
// Metal helper used by the existing integration tests.
func (e *SharedEvent) SignalCPU(value uint64) error {
	if e == nil {
		return fmt.Errorf("shared event: event is nil")
	}
	if e.port == 0 {
		return fmt.Errorf("shared event: event port is zero")
	}
	if aneBridgeSignalEventCPU != nil {
		if rc := aneBridgeSignalEventCPU(e.port, value); rc != 0 {
			return fmt.Errorf("shared event: bridge cpu signal rc=%d", rc)
		}
	} else {
		if err := objcShimMetalSignalSharedEvent(e.port, value, 1000); err != nil {
			return err
		}
	}
	if e.value != nil {
		*e.value = value
	}
	return nil
}

// WaitCPU blocks until the shared event reaches value or timeout expires.
//
// When the optional bridge runtime is available it uses the bridge's direct
// CPU-side wait path. Otherwise it falls back to the ObjC Metal helper used by
// the current integration tests.
func (e *SharedEvent) WaitCPU(value uint64, timeout time.Duration) error {
	if e == nil {
		return fmt.Errorf("shared event: event is nil")
	}
	if e.port == 0 {
		return fmt.Errorf("shared event: event port is zero")
	}
	if timeout < 0 {
		return fmt.Errorf("shared event: invalid timeout=%s", timeout)
	}
	ms := uint32(timeout / time.Millisecond)
	if timeout > 0 && ms == 0 {
		ms = 1
	}
	if aneBridgeWaitEventCPU != nil {
		if rc := aneBridgeWaitEventCPU(e.port, value, ms); rc != 0 {
			return fmt.Errorf("shared event: bridge cpu wait rc=%d", rc)
		}
		return nil
	}
	return objcShimMetalWaitSharedEvent(e.port, value, uint64(ms))
}

// IOSurfaceReadOnlyView is a scoped read-only lock on an IOSurface.
//
// It is intended for zero-copy consumers such as Metal
// newBufferWithBytesNoCopy call sites.
type IOSurfaceReadOnlyView struct {
	surface *IOSurfaceFloat32
	base    unsafe.Pointer
	byteLen int
}

// IOSurfaceWritableView is a scoped read-write lock on an IOSurface.
type IOSurfaceWritableView struct {
	surface *IOSurfaceFloat32
	base    unsafe.Pointer
	byteLen int
}

// ReadOnlyView locks the IOSurface for read-only access and returns a scoped
// view that must be closed.
func (s *IOSurfaceFloat32) ReadOnlyView() (*IOSurfaceReadOnlyView, error) {
	base, n, err := s.LockReadOnly()
	if err != nil {
		return nil, err
	}
	return &IOSurfaceReadOnlyView{surface: s, base: base, byteLen: n}, nil
}

// Pointer returns the base address of the locked IOSurface.
func (v *IOSurfaceReadOnlyView) Pointer() unsafe.Pointer {
	if v == nil {
		return nil
	}
	return v.base
}

// ByteLen returns the mapped byte length.
func (v *IOSurfaceReadOnlyView) ByteLen() int {
	if v == nil {
		return 0
	}
	return v.byteLen
}

// Float32s returns a read-only float32 slice over the locked IOSurface.
func (v *IOSurfaceReadOnlyView) Float32s() []float32 {
	if v == nil || v.base == nil || v.byteLen == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(v.base), v.byteLen/4)
}

// Close releases the underlying IOSurface read lock.
func (v *IOSurfaceReadOnlyView) Close() error {
	if v == nil || v.surface == nil {
		return nil
	}
	err := v.surface.UnlockReadOnly()
	v.surface = nil
	v.base = nil
	v.byteLen = 0
	return err
}

// WritableView locks the IOSurface for read-write access and returns a scoped
// view that must be closed.
func (s *IOSurfaceFloat32) WritableView() (*IOSurfaceWritableView, error) {
	base, n, err := s.LockWritable()
	if err != nil {
		return nil, err
	}
	return &IOSurfaceWritableView{surface: s, base: base, byteLen: n}, nil
}

// Pointer returns the base address of the locked IOSurface.
func (v *IOSurfaceWritableView) Pointer() unsafe.Pointer {
	if v == nil {
		return nil
	}
	return v.base
}

// ByteLen returns the mapped byte length.
func (v *IOSurfaceWritableView) ByteLen() int {
	if v == nil {
		return 0
	}
	return v.byteLen
}

// Float32s returns a writable float32 slice over the locked IOSurface.
func (v *IOSurfaceWritableView) Float32s() []float32 {
	if v == nil || v.base == nil || v.byteLen == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(v.base), v.byteLen/4)
}

// Close releases the underlying IOSurface write lock.
func (v *IOSurfaceWritableView) Close() error {
	if v == nil || v.surface == nil {
		return nil
	}
	err := v.surface.UnlockWritable()
	v.surface = nil
	v.base = nil
	v.byteLen = 0
	return err
}
