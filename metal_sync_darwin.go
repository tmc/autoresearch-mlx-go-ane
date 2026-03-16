//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
)

var (
	defaultMetalDeviceOnce sync.Once
	defaultMetalDevice     metal.MTLDeviceObject
	defaultMetalDeviceErr  error
)

func metalDeviceObject(device metal.MTLDevice) (metal.MTLDeviceObject, error) {
	if device == nil || device.GetID() == 0 {
		return metal.MTLDeviceObject{}, fmt.Errorf("metal device is nil")
	}
	obj, ok := device.(metal.MTLDeviceObject)
	if !ok {
		return metal.MTLDeviceObject{}, fmt.Errorf("metal device has unexpected concrete type")
	}
	return obj, nil
}

// DefaultMetalDevice returns the system default Metal device.
func DefaultMetalDevice() (metal.MTLDeviceObject, error) {
	defaultMetalDeviceOnce.Do(func() {
		ptr := metal.MTLCreateSystemDefaultDevice()
		if ptr == nil {
			defaultMetalDeviceErr = fmt.Errorf("default metal device: unavailable")
			return
		}
		device := metal.MTLDeviceObjectFromID(objc.ID(uintptr(ptr)))
		if device.GetID() == 0 {
			defaultMetalDeviceErr = fmt.Errorf("default metal device: zero object id")
			return
		}
		defaultMetalDevice = device
	})
	if defaultMetalDeviceErr != nil {
		return metal.MTLDeviceObject{}, defaultMetalDeviceErr
	}
	return defaultMetalDevice, nil
}

// MetalSharedEvent wraps a Metal shared event imported from an IOSurface event port.
type MetalSharedEvent struct {
	event metal.MTLSharedEvent
}

// Event returns the underlying Metal shared event object.
func (e *MetalSharedEvent) Event() metal.MTLSharedEvent {
	if e == nil {
		return nil
	}
	return e.event
}

// SignaledValue returns the Metal event's current signaled value.
func (e *MetalSharedEvent) SignaledValue() uint64 {
	if e == nil || e.event == nil {
		return 0
	}
	return e.event.SignaledValue()
}

// SetSignaledValue updates the Metal event's current signaled value.
func (e *MetalSharedEvent) SetSignaledValue(value uint64) error {
	if e == nil || e.event == nil {
		return fmt.Errorf("metal shared event: event is nil")
	}
	e.event.SetSignaledValue(value)
	return nil
}

// EncodeWait encodes a wait on a command buffer.
func (e *MetalSharedEvent) EncodeWait(cmd metal.MTLCommandBuffer, value uint64) error {
	if e == nil || e.event == nil {
		return fmt.Errorf("metal shared event: event is nil")
	}
	if cmd == nil || cmd.GetID() == 0 {
		return fmt.Errorf("metal shared event: command buffer is nil")
	}
	cmd.EncodeWaitForEventValue(e.event, value)
	return nil
}

// EncodeSignal encodes a signal on a command buffer.
func (e *MetalSharedEvent) EncodeSignal(cmd metal.MTLCommandBuffer, value uint64) error {
	if e == nil || e.event == nil {
		return fmt.Errorf("metal shared event: event is nil")
	}
	if cmd == nil || cmd.GetID() == 0 {
		return fmt.Errorf("metal shared event: command buffer is nil")
	}
	cmd.EncodeSignalEventValue(e.event, value)
	return nil
}

// Close releases the underlying Metal shared event.
func (e *MetalSharedEvent) Close() {
	if e == nil || e.event == nil {
		return
	}
	objectivec.ObjectFromID(e.event.GetID()).Release()
	e.event = nil
}

// NewMetalSharedEvent imports the shared event into Metal using its Mach port.
func (e *SharedEvent) NewMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if e == nil {
		return nil, fmt.Errorf("shared event: event is nil")
	}
	if e.port == 0 {
		return nil, fmt.Errorf("shared event: event port is zero")
	}
	dev, err := metalDeviceObject(device)
	if err != nil {
		return nil, fmt.Errorf("shared event: %w", err)
	}
	event := dev.NewSharedEventWithMachPort(e.port)
	if event == nil || event.GetID() == 0 {
		return nil, fmt.Errorf("shared event: import into metal failed")
	}
	return &MetalSharedEvent{event: event}, nil
}

// NewDefaultMetalSharedEvent imports the shared event using the default Metal device.
func (e *SharedEvent) NewDefaultMetalSharedEvent() (*MetalSharedEvent, error) {
	device, err := DefaultMetalDevice()
	if err != nil {
		return nil, err
	}
	return e.NewMetalSharedEvent(device)
}

// MetalBuffer wraps a no-copy Metal buffer view over a locked IOSurface.
//
// The IOSurface read lock remains held until Close. Callers must not close the
// buffer until all Metal work that reads from it has completed.
type MetalBuffer struct {
	buffer metal.MTLBuffer
	view   *IOSurfaceReadOnlyView
}

// Buffer returns the underlying Metal buffer object.
func (b *MetalBuffer) Buffer() metal.MTLBuffer {
	if b == nil {
		return nil
	}
	return b.buffer
}

// Pointer returns the IOSurface base address used for the Metal buffer.
func (b *MetalBuffer) Pointer() unsafe.Pointer {
	if b == nil || b.view == nil {
		return nil
	}
	return b.view.Pointer()
}

// ByteLen returns the no-copy mapped byte length.
func (b *MetalBuffer) ByteLen() int {
	if b == nil || b.view == nil {
		return 0
	}
	return b.view.ByteLen()
}

// Close releases the Metal buffer and unlocks the IOSurface view.
func (b *MetalBuffer) Close() error {
	if b == nil {
		return nil
	}
	if b.buffer != nil {
		objectivec.ObjectFromID(b.buffer.GetID()).Release()
		b.buffer = nil
	}
	if b.view == nil {
		return nil
	}
	err := b.view.Close()
	b.view = nil
	return err
}

// MetalBufferBinding wraps a reusable no-copy Metal buffer over an IOSurface.
//
// Unlike MetalBuffer, the IOSurface is not held read-locked for the lifetime of
// the binding. Callers must bracket each Metal read with LockReadOnly and
// UnlockReadOnly so ANE writes and Metal reads remain coherent without paying
// the per-step newBufferWithBytesNoCopy cost.
type MetalBufferBinding struct {
	buffer  metal.MTLBuffer
	surface *IOSurfaceFloat32
	ptr     unsafe.Pointer
	byteLen int
}

// Buffer returns the underlying Metal buffer object.
func (b *MetalBufferBinding) Buffer() metal.MTLBuffer {
	if b == nil {
		return nil
	}
	return b.buffer
}

// Pointer returns the IOSurface base address used for the Metal buffer.
func (b *MetalBufferBinding) Pointer() unsafe.Pointer {
	if b == nil {
		return nil
	}
	return b.ptr
}

// ByteLen returns the no-copy mapped byte length.
func (b *MetalBufferBinding) ByteLen() int {
	if b == nil {
		return 0
	}
	return b.byteLen
}

// LockReadOnly locks the IOSurface for read-only access before Metal reads.
func (b *MetalBufferBinding) LockReadOnly() error {
	if b == nil {
		return fmt.Errorf("metal buffer binding: binding is nil")
	}
	if b.surface == nil {
		return fmt.Errorf("metal buffer binding: surface is nil")
	}
	ptr, n, err := b.surface.LockReadOnly()
	if err != nil {
		return err
	}
	if ptr != b.ptr || n != b.byteLen {
		_ = b.surface.UnlockReadOnly()
		return fmt.Errorf("metal buffer binding: IOSurface mapping changed")
	}
	return nil
}

// UnlockReadOnly releases a previous LockReadOnly call.
func (b *MetalBufferBinding) UnlockReadOnly() error {
	if b == nil || b.surface == nil {
		return nil
	}
	return b.surface.UnlockReadOnly()
}

// Close releases the underlying Metal buffer and unlocks the IOSurface if
// needed.
func (b *MetalBufferBinding) Close() error {
	if b == nil {
		return nil
	}
	if b.buffer != nil {
		objectivec.ObjectFromID(b.buffer.GetID()).Release()
		b.buffer = nil
	}
	err := b.UnlockReadOnly()
	b.surface = nil
	b.ptr = nil
	b.byteLen = 0
	return err
}

func newMetalBufferFromPointer(device metal.MTLDeviceObject, ptr unsafe.Pointer, n int) (metal.MTLBuffer, error) {
	buffer := device.NewBufferWithBytesNoCopyLengthOptionsDeallocator(
		ptr,
		uint(n),
		metal.MTLResourceStorageModeShared,
		0,
	)
	if buffer == nil || buffer.GetID() == 0 {
		return nil, fmt.Errorf("metal buffer: newBufferWithBytesNoCopy failed")
	}
	return buffer, nil
}

// NewMetalBufferNoCopy wraps the IOSurface in a no-copy Metal buffer.
func (s *IOSurfaceFloat32) NewMetalBufferNoCopy(device metal.MTLDevice) (*MetalBuffer, error) {
	if s == nil {
		return nil, fmt.Errorf("metal buffer: surface wrapper is nil")
	}
	dev, err := metalDeviceObject(device)
	if err != nil {
		return nil, fmt.Errorf("metal buffer: %w", err)
	}
	view, err := s.ReadOnlyView()
	if err != nil {
		return nil, err
	}
	buffer, err := newMetalBufferFromPointer(dev, view.Pointer(), view.ByteLen())
	if err != nil {
		_ = view.Close()
		return nil, err
	}
	return &MetalBuffer{buffer: buffer, view: view}, nil
}

// NewDefaultMetalBufferNoCopy wraps the IOSurface in a no-copy buffer on the default Metal device.
func (s *IOSurfaceFloat32) NewDefaultMetalBufferNoCopy() (*MetalBuffer, error) {
	device, err := DefaultMetalDevice()
	if err != nil {
		return nil, err
	}
	return s.NewMetalBufferNoCopy(device)
}

// NewMetalBufferBinding creates a reusable no-copy Metal buffer over the
// IOSurface.
func (s *IOSurfaceFloat32) NewMetalBufferBinding(device metal.MTLDevice) (*MetalBufferBinding, error) {
	if s == nil {
		return nil, fmt.Errorf("metal buffer binding: surface wrapper is nil")
	}
	dev, err := metalDeviceObject(device)
	if err != nil {
		return nil, fmt.Errorf("metal buffer binding: %w", err)
	}
	ptr, n, err := s.LockReadOnly()
	if err != nil {
		return nil, err
	}
	if err := s.UnlockReadOnly(); err != nil {
		return nil, err
	}
	buffer, err := newMetalBufferFromPointer(dev, ptr, n)
	if err != nil {
		return nil, err
	}
	return &MetalBufferBinding{
		buffer:  buffer,
		surface: s,
		ptr:     ptr,
		byteLen: n,
	}, nil
}

// NewDefaultMetalBufferBinding creates a reusable no-copy Metal buffer over the
// IOSurface using the default Metal device.
func (s *IOSurfaceFloat32) NewDefaultMetalBufferBinding() (*MetalBufferBinding, error) {
	device, err := DefaultMetalDevice()
	if err != nil {
		return nil, err
	}
	return s.NewMetalBufferBinding(device)
}
