//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import "testing"

func TestDefaultMetalDevice(t *testing.T) {
	device, err := DefaultMetalDevice()
	if err != nil {
		t.Fatalf("DefaultMetalDevice: %v", err)
	}
	if got := device.GetID(); got == 0 {
		t.Fatal("DefaultMetalDevice returned zero object id")
	}
}

func TestIOSurfaceNewMetalBufferNoCopy(t *testing.T) {
	device, err := DefaultMetalDevice()
	if err != nil {
		t.Fatalf("DefaultMetalDevice: %v", err)
	}
	surf, err := NewIOSurfaceFloat32(8)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()
	if err := surf.Write([]float32{1, 2, 3, 4, 5, 6, 7, 8}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	buf, err := surf.NewMetalBufferNoCopy(device)
	if err != nil {
		t.Fatalf("NewMetalBufferNoCopy: %v", err)
	}
	defer buf.Close()
	if buf.Buffer() == nil || buf.Buffer().GetID() == 0 {
		t.Fatal("Buffer returned nil metal buffer")
	}
	if got, want := buf.ByteLen(), surf.ByteLen(); got != want {
		t.Fatalf("ByteLen=%d want=%d", got, want)
	}
	if got, want := int(buf.Buffer().Length()), surf.ByteLen(); got != want {
		t.Fatalf("buffer length=%d want=%d", got, want)
	}
	if buf.Pointer() == nil {
		t.Fatal("Pointer=nil")
	}
}

func TestIOSurfaceNewMetalBufferBinding(t *testing.T) {
	device, err := DefaultMetalDevice()
	if err != nil {
		t.Fatalf("DefaultMetalDevice: %v", err)
	}
	surf, err := NewIOSurfaceFloat32(8)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()
	if err := surf.Write([]float32{1, 2, 3, 4, 5, 6, 7, 8}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	buf, err := surf.NewMetalBufferBinding(device)
	if err != nil {
		t.Fatalf("NewMetalBufferBinding: %v", err)
	}
	defer buf.Close()
	if buf.Buffer() == nil || buf.Buffer().GetID() == 0 {
		t.Fatal("Buffer returned nil metal buffer")
	}
	if got, want := buf.ByteLen(), surf.ByteLen(); got != want {
		t.Fatalf("ByteLen=%d want=%d", got, want)
	}
	if got, want := int(buf.Buffer().Length()), surf.ByteLen(); got != want {
		t.Fatalf("buffer length=%d want=%d", got, want)
	}
	if buf.Pointer() == nil {
		t.Fatal("Pointer=nil")
	}
	if err := buf.LockReadOnly(); err != nil {
		t.Fatalf("LockReadOnly: %v", err)
	}
	if err := buf.UnlockReadOnly(); err != nil {
		t.Fatalf("UnlockReadOnly: %v", err)
	}
}

func TestSharedEventNewMetalSharedEventErrors(t *testing.T) {
	device, err := DefaultMetalDevice()
	if err != nil {
		t.Fatalf("DefaultMetalDevice: %v", err)
	}
	var event *SharedEvent
	if _, err := event.NewMetalSharedEvent(device); err == nil {
		t.Fatal("NewMetalSharedEvent error=nil for nil event")
	}
	event = &SharedEvent{}
	if _, err := event.NewMetalSharedEvent(device); err == nil {
		t.Fatal("NewMetalSharedEvent error=nil for zero-port event")
	}
}
