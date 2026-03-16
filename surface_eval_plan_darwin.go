//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"

	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/iosurface"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

// IOSurfaceFloat32 wraps an IOSurface storing float32 values.
//
// The type is intentionally minimal: it owns allocation/release when created by
// NewIOSurfaceFloat32 and provides lock/read/write helpers used by ANE and
// Metal no-copy paths.
type IOSurfaceFloat32 struct {
	mu          sync.Mutex
	surface     coregraphics.IOSurfaceRef
	count       int
	layout      *compiledTensorLayout
	owned       bool
	readLocked  bool
	writeLocked bool
}

// NewIOSurfaceFloat32 allocates a float32 IOSurface with count elements.
func NewIOSurfaceFloat32(count int) (*IOSurfaceFloat32, error) {
	surf, err := newFloatSurface(count)
	if err != nil {
		return nil, err
	}
	return &IOSurfaceFloat32{
		surface: surf,
		count:   count,
		owned:   true,
	}, nil
}

// WrapIOSurfaceFloat32 creates a wrapper for an existing IOSurface.
//
// The returned wrapper does not own the underlying IOSurface and Close will not
// release it.
func WrapIOSurfaceFloat32(surface coregraphics.IOSurfaceRef, count int) (*IOSurfaceFloat32, error) {
	if surface == 0 {
		return nil, fmt.Errorf("wrap IOSurface: surface is nil")
	}
	if count <= 0 {
		return nil, fmt.Errorf("wrap IOSurface: invalid count=%d", count)
	}
	return &IOSurfaceFloat32{
		surface: surface,
		count:   count,
		owned:   false,
	}, nil
}

// Ref returns the underlying IOSurfaceRef.
func (s *IOSurfaceFloat32) Ref() coregraphics.IOSurfaceRef {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.surface
}

// ID returns IOSurfaceGetID(surface).
func (s *IOSurfaceFloat32) ID() uint32 {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.surface == 0 {
		return 0
	}
	return iosurface.IOSurfaceGetID(iosurface.IOSurfaceRef(s.surface))
}

// Count returns the float32 element count.
func (s *IOSurfaceFloat32) Count() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}

// ByteLen returns the backing IOSurface allocation size in bytes.
func (s *IOSurfaceFloat32) ByteLen() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.surface == 0 {
		return s.count * 4
	}
	return int(iosurface.IOSurfaceGetAllocSize(iosurface.IOSurfaceRef(s.surface)))
}

// Write copies float32 values into the IOSurface.
func (s *IOSurfaceFloat32) Write(data []float32) error {
	if s == nil {
		return fmt.Errorf("write IOSurface: surface wrapper is nil")
	}
	s.mu.Lock()
	surf := s.surface
	count := s.count
	layout := s.layout
	s.mu.Unlock()
	if surf == 0 {
		return fmt.Errorf("write IOSurface: surface is closed")
	}
	if len(data) != count {
		return fmt.Errorf("write IOSurface: len=%d want=%d", len(data), count)
	}
	if layout != nil {
		return writeFloat32IOSurfaceWithLayout(surf, data, *layout)
	}
	return writeFloat32IOSurface(surf, data)
}

// WriteAt copies float32 values into the IOSurface at element offset.
func (s *IOSurfaceFloat32) WriteAt(offset int, data []float32) error {
	if s == nil {
		return fmt.Errorf("write IOSurface at: surface wrapper is nil")
	}
	if offset < 0 {
		return fmt.Errorf("write IOSurface at: invalid offset=%d", offset)
	}
	if len(data) == 0 {
		return nil
	}
	s.mu.Lock()
	surf := s.surface
	count := s.count
	layout := s.layout
	s.mu.Unlock()
	if surf == 0 {
		return fmt.Errorf("write IOSurface at: surface is closed")
	}
	if offset+len(data) > count {
		return fmt.Errorf(
			"write IOSurface at: range [%d,%d) out of bounds [0,%d)",
			offset,
			offset+len(data),
			count,
		)
	}
	if layout != nil {
		return writeFloat32IOSurfaceAtWithLayout(surf, offset, data, count, *layout)
	}
	raw := iosurface.IOSurfaceRef(surf)
	if rc := iosurface.IOSurfaceLock(raw, 0, nil); rc != 0 {
		return fmt.Errorf("write IOSurface at: IOSurfaceLock rc=%d", rc)
	}
	defer iosurface.IOSurfaceUnlock(raw, 0, nil)
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		return fmt.Errorf("write IOSurface at: base address is nil")
	}
	endByte := (offset + len(data)) * 4
	if got := iosurface.IOSurfaceGetAllocSize(raw); got < uintptr(endByte) {
		return fmt.Errorf("write IOSurface at: alloc size=%d want>=%d", got, endByte)
	}
	dst := unsafe.Slice((*float32)(base), count)
	copy(dst[offset:offset+len(data)], data)
	return nil
}

// Read copies float32 values from the IOSurface.
func (s *IOSurfaceFloat32) Read() ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("read IOSurface: surface wrapper is nil")
	}
	s.mu.Lock()
	surf := s.surface
	count := s.count
	layout := s.layout
	s.mu.Unlock()
	if surf == 0 {
		return nil, fmt.Errorf("read IOSurface: surface is closed")
	}
	if layout != nil {
		return readFloat32IOSurfaceWithLayout(surf, count, *layout)
	}
	return readFloat32IOSurface(surf, count)
}

// ReadAt copies n float32 values from the IOSurface starting at element offset.
func (s *IOSurfaceFloat32) ReadAt(offset, n int) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("read IOSurface at: surface wrapper is nil")
	}
	if offset < 0 || n < 0 {
		return nil, fmt.Errorf("read IOSurface at: invalid offset=%d n=%d", offset, n)
	}
	s.mu.Lock()
	surf := s.surface
	count := s.count
	layout := s.layout
	s.mu.Unlock()
	if surf == 0 {
		return nil, fmt.Errorf("read IOSurface at: surface is closed")
	}
	if offset+n > count {
		return nil, fmt.Errorf(
			"read IOSurface at: range [%d,%d) out of bounds [0,%d)",
			offset,
			offset+n,
			count,
		)
	}
	if layout != nil {
		return readFloat32IOSurfaceAtWithLayout(surf, offset, n, count, *layout)
	}
	out := make([]float32, n)
	if n == 0 {
		return out, nil
	}
	raw := iosurface.IOSurfaceRef(surf)
	if rc := iosurface.IOSurfaceLock(raw, iosurface.KIOSurfaceLockReadOnly, nil); rc != 0 {
		return nil, fmt.Errorf("read IOSurface at: IOSurfaceLock rc=%d", rc)
	}
	defer iosurface.IOSurfaceUnlock(raw, iosurface.KIOSurfaceLockReadOnly, nil)
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		return nil, fmt.Errorf("read IOSurface at: base address is nil")
	}
	endByte := (offset + n) * 4
	if got := iosurface.IOSurfaceGetAllocSize(raw); got < uintptr(endByte) {
		return nil, fmt.Errorf("read IOSurface at: alloc size=%d want>=%d", got, endByte)
	}
	src := unsafe.Slice((*float32)(base), count)
	copy(out, src[offset:offset+n])
	return out, nil
}

// Fill overwrites all elements with one value.
func (s *IOSurfaceFloat32) Fill(v float32) error {
	if s == nil {
		return fmt.Errorf("fill IOSurface: surface wrapper is nil")
	}
	if v == 0 {
		s.mu.Lock()
		surf := s.surface
		count := s.count
		layout := s.layout
		s.mu.Unlock()
		if layout != nil {
			return zeroIOSurfaceWithLayout(surf, *layout)
		}
		return s.Write(make([]float32, count))
	}
	s.mu.Lock()
	count := s.count
	s.mu.Unlock()
	data := make([]float32, count)
	for i := range data {
		data[i] = v
	}
	return s.Write(data)
}

// FillNaN is useful for stale-read detection in debug flows.
func (s *IOSurfaceFloat32) FillNaN() error {
	return s.Fill(float32(math.NaN()))
}

// LockReadOnly locks the IOSurface for read-only access and returns base
// pointer + byte length for zero-copy handoff (for example Metal
// newBufferWithBytesNoCopy).
func (s *IOSurfaceFloat32) LockReadOnly() (unsafe.Pointer, int, error) {
	if s == nil {
		return nil, 0, fmt.Errorf("lock IOSurface: surface wrapper is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.surface == 0 {
		return nil, 0, fmt.Errorf("lock IOSurface: surface is closed")
	}
	if s.readLocked {
		return nil, 0, fmt.Errorf("lock IOSurface: surface already read-locked")
	}
	if s.writeLocked {
		return nil, 0, fmt.Errorf("lock IOSurface: surface already write-locked")
	}
	raw := iosurface.IOSurfaceRef(s.surface)
	if rc := iosurface.IOSurfaceLock(raw, iosurface.KIOSurfaceLockReadOnly, nil); rc != 0 {
		return nil, 0, fmt.Errorf("lock IOSurface: IOSurfaceLock rc=%d", rc)
	}
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		iosurface.IOSurfaceUnlock(raw, iosurface.KIOSurfaceLockReadOnly, nil)
		return nil, 0, fmt.Errorf("lock IOSurface: base address is nil")
	}
	n := int(iosurface.IOSurfaceGetAllocSize(raw))
	if got := iosurface.IOSurfaceGetAllocSize(raw); got < uintptr(n) {
		iosurface.IOSurfaceUnlock(raw, iosurface.KIOSurfaceLockReadOnly, nil)
		return nil, 0, fmt.Errorf("lock IOSurface: alloc size=%d want>=%d", got, n)
	}
	s.readLocked = true
	return base, n, nil
}

// UnlockReadOnly unlocks a previous LockReadOnly call.
func (s *IOSurfaceFloat32) UnlockReadOnly() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.surface == 0 || !s.readLocked {
		return nil
	}
	raw := iosurface.IOSurfaceRef(s.surface)
	if rc := iosurface.IOSurfaceUnlock(raw, iosurface.KIOSurfaceLockReadOnly, nil); rc != 0 {
		return fmt.Errorf("unlock IOSurface: IOSurfaceUnlock rc=%d", rc)
	}
	s.readLocked = false
	return nil
}

// LockWritable locks the IOSurface for read-write access and returns base
// pointer + byte length for shared-storage producers.
func (s *IOSurfaceFloat32) LockWritable() (unsafe.Pointer, int, error) {
	if s == nil {
		return nil, 0, fmt.Errorf("lock writable IOSurface: surface wrapper is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.surface == 0 {
		return nil, 0, fmt.Errorf("lock writable IOSurface: surface is closed")
	}
	if s.readLocked {
		return nil, 0, fmt.Errorf("lock writable IOSurface: surface already read-locked")
	}
	if s.writeLocked {
		return nil, 0, fmt.Errorf("lock writable IOSurface: surface already write-locked")
	}
	raw := iosurface.IOSurfaceRef(s.surface)
	if rc := iosurface.IOSurfaceLock(raw, 0, nil); rc != 0 {
		return nil, 0, fmt.Errorf("lock writable IOSurface: IOSurfaceLock rc=%d", rc)
	}
	base := iosurface.IOSurfaceGetBaseAddress(raw)
	if base == nil {
		iosurface.IOSurfaceUnlock(raw, 0, nil)
		return nil, 0, fmt.Errorf("lock writable IOSurface: base address is nil")
	}
	n := int(iosurface.IOSurfaceGetAllocSize(raw))
	if got := iosurface.IOSurfaceGetAllocSize(raw); got < uintptr(n) {
		iosurface.IOSurfaceUnlock(raw, 0, nil)
		return nil, 0, fmt.Errorf("lock writable IOSurface: alloc size=%d want>=%d", got, n)
	}
	s.writeLocked = true
	return base, n, nil
}

// UnlockWritable unlocks a previous LockWritable call.
func (s *IOSurfaceFloat32) UnlockWritable() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.surface == 0 || !s.writeLocked {
		return nil
	}
	raw := iosurface.IOSurfaceRef(s.surface)
	if rc := iosurface.IOSurfaceUnlock(raw, 0, nil); rc != 0 {
		return fmt.Errorf("unlock writable IOSurface: IOSurfaceUnlock rc=%d", rc)
	}
	s.writeLocked = false
	return nil
}

// Close releases the underlying IOSurface when owned by the wrapper.
func (s *IOSurfaceFloat32) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.surface == 0 {
		return
	}
	if s.readLocked {
		_ = iosurface.IOSurfaceUnlock(iosurface.IOSurfaceRef(s.surface), iosurface.KIOSurfaceLockReadOnly, nil)
		s.readLocked = false
	}
	if s.writeLocked {
		_ = iosurface.IOSurfaceUnlock(iosurface.IOSurfaceRef(s.surface), 0, nil)
		s.writeLocked = false
	}
	if s.owned {
		releaseIOSurface(s.surface)
	}
	s.surface = 0
}

// SurfaceEvalPlanConfig controls request construction and execution.
type SurfaceEvalPlanConfig struct {
	InputSymbolIndex    int
	OutputSymbolIndex   int
	ProcedureIndex      int
	QoS                 uint32
	DisableCacheMapping bool
	PreferDirect        bool
	EnableMetalWait     bool
	EnableMetalSignal   bool
	WaitEventPort       uint32
	SignalEventPort     uint32
	WaitValue           uint64
	SignalValue         uint64
	EnableFWToFWSignal  bool
	// Deprecated: bridge-backed eval is unsupported in bridgeless mode.
	BridgeModelPath string
	// Deprecated: bridge-backed eval is unsupported in bridgeless mode.
	BridgeModelKey string
	// Deprecated: bridge-backed eval is unsupported in bridgeless mode.
	BridgeClientHandle uintptr
}

// DefaultSurfaceEvalPlanConfig returns conservative defaults for production.
func DefaultSurfaceEvalPlanConfig() SurfaceEvalPlanConfig {
	return SurfaceEvalPlanConfig{
		InputSymbolIndex:    0,
		OutputSymbolIndex:   0,
		ProcedureIndex:      0,
		QoS:                 defaultANEQoS,
		DisableCacheMapping: false,
		PreferDirect:        true,
		EnableMetalWait:     false,
		EnableMetalSignal:   false,
		WaitEventPort:       0,
		SignalEventPort:     0,
		WaitValue:           1,
		SignalValue:         1,
		EnableFWToFWSignal:  false,
		BridgeModelPath:     "",
		BridgeModelKey:      "s",
		BridgeClientHandle:  0,
	}
}

// EvalResult reports the outcome of EvalAsync.
type EvalResult struct {
	Duration time.Duration
	Err      error
}

// SurfaceEvalPlan is a mapped request that can be re-evaluated.
//
// Reusing a plan avoids rebuilding/mapping request state each iteration and
// enables manual IOSurface chaining between model stages.
type SurfaceEvalPlan struct {
	model           appleneuralengine.ANEInMemoryModel
	clientModel     *ANEClientMILModel
	input           *IOSurfaceFloat32
	request         objectivec.IObject
	options         foundation.NSMutableDictionary
	qos             uint32
	preferDirect    bool
	output          *IOSurfaceFloat32
	waitEventPort   uint32
	waitValue       uint64
	waitSharedID    objc.ID
	signalEventPort uint32
	signalValue     uint64
	signalSharedID  objc.ID
	retainedEvents  []objc.ID
	unmap           func()
	closeOnce       sync.Once
}

// NewSurfaceEvalPlan builds and maps a reusable request.
func NewSurfaceEvalPlan(
	model appleneuralengine.ANEInMemoryModel,
	input *IOSurfaceFloat32,
	output *IOSurfaceFloat32,
	cfg SurfaceEvalPlanConfig,
) (*SurfaceEvalPlan, error) {
	if model.GetID() == 0 {
		return nil, fmt.Errorf("new surface eval plan: model is nil")
	}
	return newSurfaceEvalPlan(model, nil, input, output, cfg)
}

// NewSurfaceEvalPlanWithClientModel builds and maps a reusable request for a
// file-backed _ANEClient-loaded model.
func NewSurfaceEvalPlanWithClientModel(
	model *ANEClientMILModel,
	input *IOSurfaceFloat32,
	output *IOSurfaceFloat32,
	cfg SurfaceEvalPlanConfig,
) (*SurfaceEvalPlan, error) {
	if model == nil {
		return nil, fmt.Errorf("new surface eval plan: client model is nil")
	}
	if model.client.ID == 0 || model.model.GetID() == 0 {
		return nil, fmt.Errorf("new surface eval plan: client model is not loaded")
	}
	return newSurfaceEvalPlan(appleneuralengine.ANEInMemoryModel{}, model, input, output, cfg)
}

func newSurfaceEvalPlan(
	model appleneuralengine.ANEInMemoryModel,
	clientModel *ANEClientMILModel,
	input *IOSurfaceFloat32,
	output *IOSurfaceFloat32,
	cfg SurfaceEvalPlanConfig,
) (*SurfaceEvalPlan, error) {
	if input == nil || output == nil {
		return nil, fmt.Errorf("new surface eval plan: input/output surface wrapper is nil")
	}
	inputSurf := input.Ref()
	outputSurf := output.Ref()
	if inputSurf == 0 || outputSurf == 0 {
		return nil, fmt.Errorf("new surface eval plan: input/output surface is nil")
	}
	if cfg.BridgeClientHandle != 0 || cfg.BridgeModelPath != "" {
		return nil, fmt.Errorf("new surface eval plan: bridge-backed eval is unsupported in bridgeless mode")
	}

	dflt := DefaultSurfaceEvalPlanConfig()
	if cfg == (SurfaceEvalPlanConfig{}) {
		cfg = dflt
	} else {
		if cfg.QoS == 0 {
			cfg.QoS = dflt.QoS
		}
		if cfg.WaitValue == 0 {
			cfg.WaitValue = dflt.WaitValue
		}
		if cfg.SignalValue == 0 {
			cfg.SignalValue = dflt.SignalValue
		}
		if cfg.WaitEventPort != 0 {
			cfg.EnableMetalWait = true
		}
		if cfg.SignalEventPort != 0 {
			cfg.EnableMetalSignal = true
		}
	}
	if cfg.WaitEventPort != 0 || cfg.SignalEventPort != 0 {
		return nil, fmt.Errorf(
			"new surface eval plan: explicit event ports are unsupported (ports are produced by the plan after shared-event attach)",
		)
	}
	if cfg.EnableMetalWait {
		// Wait graphs require FW-to-FW signaling enabled on current firmware.
		cfg.EnableFWToFWSignal = true
	}

	inObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(inputSurf)
	outObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(outputSurf)
	if inObj.GetID() == 0 || outObj.GetID() == 0 {
		return nil, fmt.Errorf("new surface eval plan: create IOSurface object failed")
	}

	requestObj, err := newSurfaceEvalRequest(inObj, outObj, cfg)
	if err != nil {
		return nil, err
	}
	request := appleneuralengine.ANERequestFromID(requestObj.GetID())
	if !request.Validate() {
		return nil, fmt.Errorf("new surface eval plan: request validation failed")
	}

	options := foundation.NewMutableDictionaryWithCapacity(4)
	var retained []objc.ID
	waitEventPort := cfg.WaitEventPort
	var waitSharedID objc.ID
	signalEventPort := cfg.SignalEventPort
	var signalSharedID objc.ID
	enableSharedEvents := cfg.EnableMetalWait || cfg.EnableMetalSignal
	if enableSharedEvents {
		setANEOptionBool(options, appleneuralengine.KANEFDisableIOFencesUseSharedEventsKey, "kANEFDisableIOFencesUseSharedEventsKey", true)
		setANEOptionBool(options, appleneuralengine.KANEFEnableFWToFWSignal, "kANEFEnableFWToFWSignal", cfg.EnableFWToFWSignal)
	}

	var unmap func()
	if clientModel != nil {
		if err := callObjCBoolWithNSError(
			"appleneuralengine map IOSurfaces",
			clientModel.client.ID,
			"mapIOSurfacesWithModel:request:cacheInference:error:",
			clientModel.model,
			requestObj,
			!cfg.DisableCacheMapping,
		); err != nil {
			objcShimRelease(retained)
			return nil, err
		}
		unmap = func() {
			objc.Send[objc.ID](clientModel.client.ID, objc.Sel("unmapIOSurfacesWithModel:request:"), clientModel.model, requestObj)
		}
	} else {
		unmap, err = mapModelRequestWithFallback(
			"appleneuralengine map IOSurfaces",
			model,
			requestObj,
			!cfg.DisableCacheMapping,
		)
		if err != nil {
			objcShimRelease(retained)
			return nil, err
		}
	}
	if enableSharedEvents {
		switch {
		case cfg.EnableMetalWait && cfg.EnableMetalSignal:
			graph, attachErr := objcShimAttachWaitSignalEvents(
				requestObj.GetID(),
				cfg.WaitValue,
				uint32(cfg.OutputSymbolIndex),
				cfg.SignalValue,
			)
			if attachErr != nil {
				unmap()
				return nil, fmt.Errorf("new surface eval plan: attach wait+signal events: %w", attachErr)
			}
			retained = graph.Retained
			waitEventPort = graph.WaitEventPort
			waitSharedID = graph.WaitSharedEvent
			signalEventPort = graph.SignalEventPort
			signalSharedID = graph.SignalSharedEvent
		case cfg.EnableMetalWait:
			graph, attachErr := objcShimAttachWaitEvents(requestObj.GetID(), cfg.WaitValue)
			if attachErr != nil {
				unmap()
				return nil, fmt.Errorf("new surface eval plan: attach wait event: %w", attachErr)
			}
			retained = graph.Retained
			waitEventPort = graph.EventPort
			waitSharedID = graph.SharedEvent
		case cfg.EnableMetalSignal:
			graph, attachErr := objcShimAttachSignalEvents(
				requestObj.GetID(),
				uint32(cfg.OutputSymbolIndex),
				cfg.SignalValue,
			)
			if attachErr != nil {
				unmap()
				return nil, fmt.Errorf("new surface eval plan: attach signal event: %w", attachErr)
			}
			retained = graph.Retained
			signalEventPort = graph.EventPort
			signalSharedID = graph.SharedEvent
		}
		request.SetTransactionHandle(foundation.NewNumberWithUnsignedLongLong(1))
		if !request.Validate() {
			unmap()
			if len(retained) > 0 {
				objcShimRelease(retained)
			}
			return nil, fmt.Errorf("new surface eval plan: request validation failed after shared-event attach")
		}
	}

	return &SurfaceEvalPlan{
		model:           model,
		clientModel:     clientModel,
		input:           input,
		request:         requestObj,
		options:         options,
		qos:             cfg.QoS,
		preferDirect:    cfg.PreferDirect,
		output:          output,
		waitEventPort:   waitEventPort,
		waitValue:       cfg.WaitValue,
		waitSharedID:    waitSharedID,
		signalEventPort: signalEventPort,
		signalValue:     cfg.SignalValue,
		signalSharedID:  signalSharedID,
		retainedEvents:  retained,
		unmap:           unmap,
	}, nil
}

func newSurfaceEvalRequest(
	inObj objectivec.IObject,
	outObj objectivec.IObject,
	cfg SurfaceEvalPlanConfig,
) (objectivec.IObject, error) {
	inputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{inObj}))
	outputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{outObj}))
	inputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(cfg.InputSymbolIndex),
	}))
	outputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(cfg.OutputSymbolIndex),
	}))
	procedure := foundation.NewNumberWithInt(cfg.ProcedureIndex)

	req := appleneuralengine.GetANERequestClass().RequestWithInputsInputIndicesOutputsOutputIndicesProcedureIndex(
		inputs,
		inputIndices,
		outputs,
		outputIndices,
		procedure,
	)
	if req.GetID() == 0 {
		return objectivec.Object{}, fmt.Errorf("new surface eval plan: create request failed")
	}
	return req, nil
}

func setANEOptionBool(
	options foundation.NSMutableDictionary,
	key objectivec.IObject,
	fallbackKey string,
	value bool,
) {
	if key != nil && key.GetID() != 0 {
		options.SetObjectForKey(foundation.NewNumberWithBool(value), key)
		return
	}
	options.SetObjectForKey(
		foundation.NewNumberWithBool(value),
		foundation.NewStringWithString(fallbackKey),
	)
}

// Eval runs one synchronous ANE evaluation.
func (p *SurfaceEvalPlan) Eval(ctx context.Context) error {
	if p == nil {
		return fmt.Errorf("surface eval plan: plan is nil")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	if p.clientModel != nil {
		if p.preferDirect &&
			objc.Send[bool](p.clientModel.client.ID, objc.Sel("respondsToSelector:"), objc.Sel("doEvaluateDirectWithModel:options:request:qos:error:")) {
			if _, err := p.clientModel.client.DoEvaluateDirectWithModelOptionsRequestQosError(
				p.clientModel.model,
				p.clientModel.options,
				p.request,
				p.clientModel.qos,
			); err != nil {
				return err
			}
			return nil
		}
		if _, err := p.clientModel.client.EvaluateWithModelOptionsRequestQosError(
			p.clientModel.model,
			p.clientModel.options,
			p.request,
			p.clientModel.qos,
		); err != nil {
			return err
		}
		return nil
	}

	if p.waitEventPort != 0 || p.signalEventPort != 0 {
		shared := p.model.SharedConnection()
		base := p.model.Model()
		if shared.GetID() != 0 && base.GetID() != 0 && p.options.GetID() != 0 && p.request.GetID() != 0 {
			return objcShimEvaluate(
				shared.GetID(),
				base.GetID(),
				p.options.GetID(),
				p.request.GetID(),
				p.qos,
				p.preferDirect,
			)
		}
	}

	if p.preferDirect {
		if err := callModelEvaluateWithFallback(
			"appleneuralengine evaluate",
			p.model,
			p.qos,
			p.options,
			p.request,
		); err != nil {
			return err
		}
		return nil
	}

	return callObjCBoolWithNSError(
		"appleneuralengine evaluate",
		p.model.ID,
		"evaluateWithQoS:options:request:error:",
		p.qos,
		p.options,
		p.request,
	)
}

// EvalAsync dispatches Eval on a background goroutine and returns a result
// channel suitable for CPU+ANE overlap.
func (p *SurfaceEvalPlan) EvalAsync(ctx context.Context) <-chan EvalResult {
	ch := make(chan EvalResult, 1)
	go func() {
		start := time.Now()
		err := p.Eval(ctx)
		ch <- EvalResult{
			Duration: time.Since(start),
			Err:      err,
		}
		close(ch)
	}()
	return ch
}

// Output returns the output IOSurface wrapper bound to this plan.
func (p *SurfaceEvalPlan) Output() *IOSurfaceFloat32 {
	if p == nil {
		return nil
	}
	return p.output
}

// WaitEventPort returns the IOSurfaceSharedEvent Mach port for Metal->ANE wait
// synchronization. Zero means no wait-event graph is attached.
func (p *SurfaceEvalPlan) WaitEventPort() uint32 {
	if p == nil {
		return 0
	}
	return p.waitEventPort
}

// WaitEvent returns the attached Metal->ANE shared event wrapper.
func (p *SurfaceEvalPlan) WaitEvent() *SharedEvent {
	if p == nil {
		return nil
	}
	return newSharedEvent(p.waitEventPort, &p.waitValue, p.waitSharedID, "wait")
}

// NewWaitMetalSharedEvent imports the wait event into Metal.
func (p *SurfaceEvalPlan) NewWaitMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("surface eval plan wait metal event: plan is nil")
	}
	ev := p.WaitEvent()
	if ev == nil {
		return nil, fmt.Errorf("surface eval plan wait metal event: wait event unavailable")
	}
	return ev.NewMetalSharedEvent(device)
}

// NewDefaultWaitMetalSharedEvent imports the wait event into Metal using the
// default Metal device.
func (p *SurfaceEvalPlan) NewDefaultWaitMetalSharedEvent() (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("surface eval plan wait metal event: plan is nil")
	}
	ev := p.WaitEvent()
	if ev == nil {
		return nil, fmt.Errorf("surface eval plan wait metal event: wait event unavailable")
	}
	return ev.NewDefaultMetalSharedEvent()
}

// WaitValue returns the configured wait-event target value.
func (p *SurfaceEvalPlan) WaitValue() uint64 {
	if p == nil {
		return 0
	}
	return p.waitValue
}

// SignalEventPort returns the IOSurfaceSharedEvent Mach port for ANE->Metal
// signal synchronization. Zero means no signal-event graph is attached.
func (p *SurfaceEvalPlan) SignalEventPort() uint32 {
	if p == nil {
		return 0
	}
	return p.signalEventPort
}

// SignalEvent returns the attached ANE->Metal shared event wrapper.
func (p *SurfaceEvalPlan) SignalEvent() *SharedEvent {
	if p == nil {
		return nil
	}
	return newSharedEvent(p.signalEventPort, &p.signalValue, p.signalSharedID, "signal")
}

// NewSignalMetalSharedEvent imports the signal event into Metal.
func (p *SurfaceEvalPlan) NewSignalMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("surface eval plan signal metal event: plan is nil")
	}
	ev := p.SignalEvent()
	if ev == nil {
		return nil, fmt.Errorf("surface eval plan signal metal event: signal event unavailable")
	}
	return ev.NewMetalSharedEvent(device)
}

// NewDefaultSignalMetalSharedEvent imports the signal event into Metal using
// the default Metal device.
func (p *SurfaceEvalPlan) NewDefaultSignalMetalSharedEvent() (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("surface eval plan signal metal event: plan is nil")
	}
	ev := p.SignalEvent()
	if ev == nil {
		return nil, fmt.Errorf("surface eval plan signal metal event: signal event unavailable")
	}
	return ev.NewDefaultMetalSharedEvent()
}

// SignalValue returns the configured signal-event target value.
func (p *SurfaceEvalPlan) SignalValue() uint64 {
	if p == nil {
		return 0
	}
	return p.signalValue
}

// SetWaitEventSignaledValue updates the attached wait-event shared value.
//
// This is mainly useful for tests that do not have a Metal producer.
func (p *SurfaceEvalPlan) SetWaitEventSignaledValue(value uint64) error {
	e := p.WaitEvent()
	if e == nil {
		return fmt.Errorf("surface eval plan: no wait-event shared object attached")
	}
	return e.SetSignaledValue(value)
}

// SetSignalEventSignaledValue updates the attached signal-event shared value.
//
// This is mainly useful for tests that do not have an ANE producer.
func (p *SurfaceEvalPlan) SetSignalEventSignaledValue(value uint64) error {
	e := p.SignalEvent()
	if e == nil {
		return fmt.Errorf("surface eval plan: no signal-event shared object attached")
	}
	return e.SetSignaledValue(value)
}

// Close unmaps the request and releases retained shared-event objects.
func (p *SurfaceEvalPlan) Close() {
	if p == nil {
		return
	}
	p.closeOnce.Do(func() {
		if p.unmap != nil {
			p.unmap()
			p.unmap = nil
		}
		if len(p.retainedEvents) > 0 {
			objcShimRelease(p.retainedEvents)
			p.retainedEvents = nil
		}
	})
}
