//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

// SurfaceOutputRuntime is the narrow output-plane interface implemented by the
// current block-level ANE runtimes.
//
// Downstream code in mlx-go should depend on this interface when it only needs
// zero-copy access to the ANE output surface and a reusable Metal buffer
// binding, instead of naming a concrete runtime type.
type SurfaceOutputRuntime interface {
	OutputSurface() *IOSurfaceFloat32
	NewDefaultOutputMetalBufferBinding() (*MetalBufferBinding, error)
}

// SharedEventRuntime is the narrow event-plane interface implemented by the
// current block-level ANE runtimes.
//
// Downstream code should prefer this interface over raw Mach ports when
// coordinating Metal and ANE work.
type SharedEventRuntime interface {
	WaitEvent() *SharedEvent
	SignalEvent() *SharedEvent
	NewDefaultWaitMetalSharedEvent() (*MetalSharedEvent, error)
	NewDefaultSignalMetalSharedEvent() (*MetalSharedEvent, error)
}

// SurfaceSyncRuntime is the preferred downstream extension point for block
// runtimes that expose both a reusable IOSurface output plane and plan-owned
// shared events.
//
// ANEDraftModel satisfies this interface. Callers that only need Metal/ANE
// handoff should depend on this interface instead of the concrete runtime.
type SurfaceSyncRuntime interface {
	SurfaceOutputRuntime
	SharedEventRuntime
	Close()
}

var _ SurfaceSyncRuntime = (*ANEDraftModel)(nil)
