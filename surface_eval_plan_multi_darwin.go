//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"sync"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

// SurfaceBinding binds one IOSurface wrapper to a model symbol index.
type SurfaceBinding struct {
	Surface     *IOSurfaceFloat32
	SymbolIndex int
}

// MultiSurfaceEvalPlanConfig controls request construction and execution.
type MultiSurfaceEvalPlanConfig struct {
	ProcedureIndex      int
	QoS                 uint32
	DisableCacheMapping bool
	PreferDirect        bool
	EnableMetalWait     bool
	EnableMetalSignal   bool
	WaitValue           uint64
	SignalValue         uint64
	EnableFWToFWSignal  bool
	SignalOutputBinding int
}

// DefaultMultiSurfaceEvalPlanConfig returns conservative defaults.
func DefaultMultiSurfaceEvalPlanConfig() MultiSurfaceEvalPlanConfig {
	return MultiSurfaceEvalPlanConfig{
		ProcedureIndex:      0,
		QoS:                 defaultANEQoS,
		DisableCacheMapping: false,
		PreferDirect:        true,
		EnableMetalWait:     false,
		EnableMetalSignal:   false,
		WaitValue:           1,
		SignalValue:         1,
		EnableFWToFWSignal:  false,
		SignalOutputBinding: 0,
	}
}

// MultiSurfaceEvalPlan is a mapped request for multi-input/multi-output models.
//
// It keeps request + IOSurface mappings alive across Eval calls so callers can
// mutate IOSurface contents directly between dispatches (for example KV cache
// splice writes during decode).
type MultiSurfaceEvalPlan struct {
	model           appleneuralengine.ANEInMemoryModel
	clientModel     *ANEClientMILModel
	request         objectivec.IObject
	options         foundation.NSMutableDictionary
	qos             uint32
	preferDirect    bool
	waitEventPort   uint32
	waitValue       uint64
	waitSharedID    objc.ID
	signalEventPort uint32
	signalValue     uint64
	signalSharedID  objc.ID
	retainedEvents  []objc.ID
	inputs          []SurfaceBinding
	outputs         []SurfaceBinding
	unmap           func()
	closeOnce       sync.Once
}

// NewMultiSurfaceEvalPlan builds and maps a reusable multi-IO request.
func NewMultiSurfaceEvalPlan(
	model appleneuralengine.ANEInMemoryModel,
	inputs []SurfaceBinding,
	outputs []SurfaceBinding,
	cfg MultiSurfaceEvalPlanConfig,
) (*MultiSurfaceEvalPlan, error) {
	if model.GetID() == 0 {
		return nil, fmt.Errorf("new multi-surface eval plan: model is nil")
	}
	return newMultiSurfaceEvalPlan(model, nil, inputs, outputs, cfg)
}

// NewMultiSurfaceEvalPlanWithClientModel builds and maps a reusable multi-IO
// request for a file-backed _ANEClient-loaded model.
func NewMultiSurfaceEvalPlanWithClientModel(
	model *ANEClientMILModel,
	inputs []SurfaceBinding,
	outputs []SurfaceBinding,
	cfg MultiSurfaceEvalPlanConfig,
) (*MultiSurfaceEvalPlan, error) {
	if model == nil {
		return nil, fmt.Errorf("new multi-surface eval plan: client model is nil")
	}
	if model.client.ID == 0 || model.model.GetID() == 0 {
		return nil, fmt.Errorf("new multi-surface eval plan: client model is not loaded")
	}
	return newMultiSurfaceEvalPlan(appleneuralengine.ANEInMemoryModel{}, model, inputs, outputs, cfg)
}

func newMultiSurfaceEvalPlan(
	model appleneuralengine.ANEInMemoryModel,
	clientModel *ANEClientMILModel,
	inputs []SurfaceBinding,
	outputs []SurfaceBinding,
	cfg MultiSurfaceEvalPlanConfig,
) (*MultiSurfaceEvalPlan, error) {
	trace := func(phase string, args ...any) {
		if !envTruthy("MLXGO_ANE_DEBUG_DRAFT_PLAN") {
			return
		}
		slog.Info("ANE multi-surface plan phase", append([]any{"phase", phase}, args...)...)
	}
	if len(inputs) == 0 || len(outputs) == 0 {
		return nil, fmt.Errorf("new multi-surface eval plan: inputs/outputs must be non-empty")
	}
	dflt := DefaultMultiSurfaceEvalPlanConfig()
	if cfg == (MultiSurfaceEvalPlanConfig{}) {
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
	}
	if cfg.EnableMetalWait {
		// Wait graphs require FW-to-FW signaling enabled on current firmware.
		cfg.EnableFWToFWSignal = true
	}
	if cfg.SignalOutputBinding < 0 || cfg.SignalOutputBinding >= len(outputs) {
		return nil, fmt.Errorf(
			"new multi-surface eval plan: signal output binding=%d out of range [0,%d)",
			cfg.SignalOutputBinding,
			len(outputs),
		)
	}

	inObjs, inIndices, err := surfaceBindingsToNSArray(inputs, "input")
	if err != nil {
		return nil, err
	}
	outObjs, outIndices, err := surfaceBindingsToNSArray(outputs, "output")
	if err != nil {
		return nil, err
	}
	requestObj := appleneuralengine.GetANERequestClass().RequestWithInputsInputIndicesOutputsOutputIndicesProcedureIndex(
		inObjs,
		inIndices,
		outObjs,
		outIndices,
		foundation.NewNumberWithInt(cfg.ProcedureIndex),
	)
	if requestObj.GetID() == 0 {
		return nil, fmt.Errorf("new multi-surface eval plan: create request failed")
	}
	request := appleneuralengine.ANERequestFromID(requestObj.GetID())
	if !request.Validate() {
		return nil, fmt.Errorf("new multi-surface eval plan: request validation failed")
	}
	trace("request_ready", "inputs", len(inputs), "outputs", len(outputs), "procedure", cfg.ProcedureIndex)

	options := foundation.NewMutableDictionaryWithCapacity(4)
	var retained []objc.ID
	waitEventPort := uint32(0)
	waitSharedID := objc.ID(0)
	signalEventPort := uint32(0)
	signalSharedID := objc.ID(0)
	enableSharedEvents := cfg.EnableMetalWait || cfg.EnableMetalSignal
	if enableSharedEvents {
		setANEOptionBool(options, appleneuralengine.KANEFDisableIOFencesUseSharedEventsKey, "kANEFDisableIOFencesUseSharedEventsKey", true)
		setANEOptionBool(options, appleneuralengine.KANEFEnableFWToFWSignal, "kANEFEnableFWToFWSignal", cfg.EnableFWToFWSignal)
	}

	var unmap func()
	if clientModel != nil {
		trace("map_begin", "path", "client", "disable_cache_mapping", cfg.DisableCacheMapping)
		if err := callObjCBoolWithNSError(
			"appleneuralengine map IOSurfaces (multi client)",
			clientModel.client.ID,
			"mapIOSurfacesWithModel:request:cacheInference:error:",
			clientModel.model,
			requestObj,
			!cfg.DisableCacheMapping,
		); err != nil {
			return nil, err
		}
		unmap = func() {
			objc.Send[objc.ID](clientModel.client.ID, objc.Sel("unmapIOSurfacesWithModel:request:"), clientModel.model, requestObj)
		}
	} else {
		trace("map_begin", "path", "in_memory", "disable_cache_mapping", cfg.DisableCacheMapping)
		var err error
		unmap, err = mapModelRequestWithFallback(
			"appleneuralengine map IOSurfaces (multi)",
			model,
			requestObj,
			!cfg.DisableCacheMapping,
		)
		if err != nil {
			return nil, err
		}
	}
	trace("map_done", "path", func() string {
		if clientModel != nil {
			return "client"
		}
		return "in_memory"
	}())
	if enableSharedEvents {
		trace("shared_events_begin", "wait", cfg.EnableMetalWait, "signal", cfg.EnableMetalSignal)
		switch {
		case cfg.EnableMetalWait && cfg.EnableMetalSignal:
			graph, attachErr := objcShimAttachWaitSignalEvents(
				requestObj.GetID(),
				cfg.WaitValue,
				uint32(outputs[cfg.SignalOutputBinding].SymbolIndex),
				cfg.SignalValue,
			)
			if attachErr != nil {
				unmap()
				return nil, fmt.Errorf("new multi-surface eval plan: attach wait+signal events: %w", attachErr)
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
				return nil, fmt.Errorf("new multi-surface eval plan: attach wait event: %w", attachErr)
			}
			retained = graph.Retained
			waitEventPort = graph.EventPort
			waitSharedID = graph.SharedEvent
		case cfg.EnableMetalSignal:
			graph, attachErr := objcShimAttachSignalEvents(
				requestObj.GetID(),
				uint32(outputs[cfg.SignalOutputBinding].SymbolIndex),
				cfg.SignalValue,
			)
			if attachErr != nil {
				unmap()
				return nil, fmt.Errorf("new multi-surface eval plan: attach signal event: %w", attachErr)
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
			return nil, fmt.Errorf("new multi-surface eval plan: request validation failed after shared-event attach")
		}
		trace("shared_events_done", "wait_port", waitEventPort, "signal_port", signalEventPort)
	}

	trace("plan_ready", "prefer_direct", cfg.PreferDirect)
	return &MultiSurfaceEvalPlan{
		model:           model,
		clientModel:     clientModel,
		request:         requestObj,
		options:         options,
		qos:             cfg.QoS,
		preferDirect:    cfg.PreferDirect,
		waitEventPort:   waitEventPort,
		waitValue:       cfg.WaitValue,
		waitSharedID:    waitSharedID,
		signalEventPort: signalEventPort,
		signalValue:     cfg.SignalValue,
		signalSharedID:  signalSharedID,
		retainedEvents:  retained,
		inputs:          append([]SurfaceBinding(nil), inputs...),
		outputs:         append([]SurfaceBinding(nil), outputs...),
		unmap:           unmap,
	}, nil
}

func surfaceBindingsToNSArray(bindings []SurfaceBinding, label string) (objectivec.IObject, objectivec.IObject, error) {
	bindings = append([]SurfaceBinding(nil), bindings...)
	slices.SortStableFunc(bindings, func(a, b SurfaceBinding) int {
		switch {
		case a.SymbolIndex < b.SymbolIndex:
			return -1
		case a.SymbolIndex > b.SymbolIndex:
			return 1
		default:
			return 0
		}
	})
	objs := make([]objectivec.IObject, 0, len(bindings))
	indices := make([]objectivec.IObject, 0, len(bindings))
	for i, b := range bindings {
		if b.Surface == nil {
			return objectivec.Object{}, objectivec.Object{}, fmt.Errorf("%s binding[%d]: surface is nil", label, i)
		}
		surf := b.Surface.Ref()
		if surf == 0 {
			return objectivec.Object{}, objectivec.Object{}, fmt.Errorf("%s binding[%d]: surface is closed", label, i)
		}
		obj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(surf)
		if obj.GetID() == 0 {
			return objectivec.Object{}, objectivec.Object{}, fmt.Errorf("%s binding[%d]: create IOSurface object failed", label, i)
		}
		objs = append(objs, obj)
		indices = append(indices, foundation.NewNumberWithInt(b.SymbolIndex))
	}
	return objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray(objs)), objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray(indices)), nil
}

// Eval runs one synchronous ANE evaluation.
func (p *MultiSurfaceEvalPlan) Eval(ctx context.Context) error {
	if p == nil {
		return fmt.Errorf("multi-surface eval plan: plan is nil")
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
			"appleneuralengine evaluate (multi)",
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
		"appleneuralengine evaluate (multi)",
		p.model.ID,
		"evaluateWithQoS:options:request:error:",
		p.qos,
		p.options,
		p.request,
	)
}

// WaitEventPort returns the IOSurfaceSharedEvent Mach port for Metal->ANE wait
// synchronization. Zero means no wait-event graph is attached.
func (p *MultiSurfaceEvalPlan) WaitEventPort() uint32 {
	if p == nil {
		return 0
	}
	return p.waitEventPort
}

// WaitEvent returns the attached Metal->ANE shared event wrapper.
func (p *MultiSurfaceEvalPlan) WaitEvent() *SharedEvent {
	if p == nil {
		return nil
	}
	return newSharedEvent(p.waitEventPort, &p.waitValue, p.waitSharedID, "wait")
}

// NewWaitMetalSharedEvent imports the wait event into Metal.
func (p *MultiSurfaceEvalPlan) NewWaitMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("multi-surface eval plan wait metal event: plan is nil")
	}
	ev := p.WaitEvent()
	if ev == nil {
		return nil, fmt.Errorf("multi-surface eval plan wait metal event: wait event unavailable")
	}
	return ev.NewMetalSharedEvent(device)
}

// NewDefaultWaitMetalSharedEvent imports the wait event into Metal using the
// default Metal device.
func (p *MultiSurfaceEvalPlan) NewDefaultWaitMetalSharedEvent() (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("multi-surface eval plan wait metal event: plan is nil")
	}
	ev := p.WaitEvent()
	if ev == nil {
		return nil, fmt.Errorf("multi-surface eval plan wait metal event: wait event unavailable")
	}
	return ev.NewDefaultMetalSharedEvent()
}

// WaitValue returns the configured wait-event target value.
func (p *MultiSurfaceEvalPlan) WaitValue() uint64 {
	if p == nil {
		return 0
	}
	return p.waitValue
}

// SignalEventPort returns the IOSurfaceSharedEvent Mach port for ANE->Metal
// signal synchronization. Zero means no signal-event graph is attached.
func (p *MultiSurfaceEvalPlan) SignalEventPort() uint32 {
	if p == nil {
		return 0
	}
	return p.signalEventPort
}

// SignalEvent returns the attached ANE->Metal shared event wrapper.
func (p *MultiSurfaceEvalPlan) SignalEvent() *SharedEvent {
	if p == nil {
		return nil
	}
	return newSharedEvent(p.signalEventPort, &p.signalValue, p.signalSharedID, "signal")
}

// NewSignalMetalSharedEvent imports the signal event into Metal.
func (p *MultiSurfaceEvalPlan) NewSignalMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("multi-surface eval plan signal metal event: plan is nil")
	}
	ev := p.SignalEvent()
	if ev == nil {
		return nil, fmt.Errorf("multi-surface eval plan signal metal event: signal event unavailable")
	}
	return ev.NewMetalSharedEvent(device)
}

// NewDefaultSignalMetalSharedEvent imports the signal event into Metal using
// the default Metal device.
func (p *MultiSurfaceEvalPlan) NewDefaultSignalMetalSharedEvent() (*MetalSharedEvent, error) {
	if p == nil {
		return nil, fmt.Errorf("multi-surface eval plan signal metal event: plan is nil")
	}
	ev := p.SignalEvent()
	if ev == nil {
		return nil, fmt.Errorf("multi-surface eval plan signal metal event: signal event unavailable")
	}
	return ev.NewDefaultMetalSharedEvent()
}

// SignalValue returns the configured signal-event target value.
func (p *MultiSurfaceEvalPlan) SignalValue() uint64 {
	if p == nil {
		return 0
	}
	return p.signalValue
}

// SetWaitEventSignaledValue updates the attached wait-event shared value.
func (p *MultiSurfaceEvalPlan) SetWaitEventSignaledValue(value uint64) error {
	e := p.WaitEvent()
	if e == nil {
		return fmt.Errorf("multi-surface eval plan: no wait-event shared object attached")
	}
	return e.SetSignaledValue(value)
}

// SetSignalEventSignaledValue updates the attached signal-event shared value.
func (p *MultiSurfaceEvalPlan) SetSignalEventSignaledValue(value uint64) error {
	e := p.SignalEvent()
	if e == nil {
		return fmt.Errorf("multi-surface eval plan: no signal-event shared object attached")
	}
	return e.SetSignaledValue(value)
}

// Inputs returns a copy of input bindings.
func (p *MultiSurfaceEvalPlan) Inputs() []SurfaceBinding {
	if p == nil {
		return nil
	}
	return append([]SurfaceBinding(nil), p.inputs...)
}

// Outputs returns a copy of output bindings.
func (p *MultiSurfaceEvalPlan) Outputs() []SurfaceBinding {
	if p == nil {
		return nil
	}
	return append([]SurfaceBinding(nil), p.outputs...)
}

// Close unmaps the mapped request.
func (p *MultiSurfaceEvalPlan) Close() {
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
