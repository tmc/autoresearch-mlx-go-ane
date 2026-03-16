//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

type dynamicLinearModelKey struct {
	batch        int
	paddedInDim  int
	paddedOutDim int
}

type dynamicLinearModel struct {
	mu       sync.Mutex
	key      dynamicLinearModelKey
	inMemory appleneuralengine.ANEInMemoryModel
	plan     *MultiSurfaceEvalPlan
	packedIn *IOSurfaceFloat32
	yOut     *IOSurfaceFloat32
}

func (m *dynamicLinearModel) Eval(ctx context.Context, x, w []float32, inDim, outDim int) ([]float32, time.Duration, error) {
	if m == nil {
		return nil, 0, fmt.Errorf("dynamic linear model: model is nil")
	}
	y := make([]float32, m.key.batch*outDim)
	evalDur, err := m.EvalInto(ctx, x, w, inDim, outDim, y)
	if err != nil {
		return nil, evalDur, err
	}
	return y, evalDur, nil
}

func (m *dynamicLinearModel) EvalInto(ctx context.Context, x, w []float32, inDim, outDim int, dst []float32) (time.Duration, error) {
	if m == nil {
		return 0, fmt.Errorf("dynamic linear model: model is nil")
	}
	if len(dst) != m.key.batch*outDim {
		return 0, fmt.Errorf("dynamic linear model: dst len=%d want=%d", len(dst), m.key.batch*outDim)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	base, n, err := m.packedIn.LockWritable()
	if err != nil {
		return 0, fmt.Errorf("dynamic linear model: lock packed input: %w", err)
	}
	defer func() {
		_ = m.packedIn.UnlockWritable()
	}()
	wantPacked := m.key.batch*m.key.paddedInDim + m.key.paddedOutDim*m.key.paddedInDim
	if n < wantPacked*4 {
		return 0, fmt.Errorf("dynamic linear model: packed input bytes=%d want>=%d", n, wantPacked*4)
	}
	packed := unsafe.Slice((*float32)(base), wantPacked)
	if err := dynamicLinearPackedHostToANEWithPaddingInto(
		packed,
		x,
		w,
		m.key.batch,
		inDim,
		outDim,
		m.key.paddedInDim,
		m.key.paddedOutDim,
	); err != nil {
		return 0, err
	}

	return m.evalLocked(ctx, outDim, dst)
}

func (m *dynamicLinearModel) evalLocked(ctx context.Context, outDim int, dst []float32) (time.Duration, error) {
	if m == nil {
		return 0, fmt.Errorf("dynamic linear model: model is nil")
	}
	if len(dst) != m.key.batch*outDim {
		return 0, fmt.Errorf("dynamic linear model: dst len=%d want=%d", len(dst), m.key.batch*outDim)
	}

	start := time.Now()
	if err := m.plan.Eval(ctx); err != nil {
		return time.Since(start), fmt.Errorf("dynamic linear model: eval: %w", err)
	}
	evalDur := time.Since(start)

	base, n, err := m.yOut.LockReadOnly()
	if err != nil {
		return evalDur, fmt.Errorf("dynamic linear model: lock y: %w", err)
	}
	defer func() {
		_ = m.yOut.UnlockReadOnly()
	}()
	wantY := m.key.batch * m.key.paddedOutDim
	if n < wantY*4 {
		return evalDur, fmt.Errorf("dynamic linear model: y bytes=%d want>=%d", n, wantY*4)
	}
	y := unsafe.Slice((*float32)(base), wantY)
	if err := dynamicLinearOutputANEToHostWithPaddingInto(dst, y, m.key.batch, outDim, m.key.paddedOutDim); err != nil {
		return evalDur, err
	}
	return evalDur, nil
}

func (m *dynamicLinearModel) Close() {
	if m == nil {
		return
	}
	if m.plan != nil {
		m.plan.Close()
		m.plan = nil
	}
	if m.packedIn != nil {
		m.packedIn.Close()
		m.packedIn = nil
	}
	if m.yOut != nil {
		m.yOut.Close()
		m.yOut = nil
	}
	if m.inMemory.ID != 0 {
		_ = callObjCBoolWithNSError(
			"appleneuralengine dynamic linear unload",
			m.inMemory.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
		m.inMemory = appleneuralengine.ANEInMemoryModel{}
	}
}

type applePrivateDynamicLinearExecutor struct {
	mu     sync.Mutex
	linear map[dynamicLinearModelKey]*dynamicLinearModel
	last   linearTelemetry
}

// NewApplePrivateDynamicLinearExecutor returns a training-oriented executor
// backed by private ANE bindings that treats weights as runtime inputs and
// caches compiled models by shape only.
func NewApplePrivateDynamicLinearExecutor() (LinearExecutor, error) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		return nil, fmt.Errorf("appleneuralengine: ANE not available on this host")
	}
	return &applePrivateDynamicLinearExecutor{
		linear: make(map[dynamicLinearModelKey]*dynamicLinearModel),
	}, nil
}

func (e *applePrivateDynamicLinearExecutor) Linear(ctx context.Context, x, w []float32, batch, inDim, outDim int) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("appleneuralengine dynamic linear: executor is nil")
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if len(x) != batch*inDim {
		return nil, fmt.Errorf("appleneuralengine dynamic linear: x len=%d want=%d", len(x), batch*inDim)
	}
	if len(w) != outDim*inDim {
		return nil, fmt.Errorf("appleneuralengine dynamic linear: w len=%d want=%d", len(w), outDim*inDim)
	}

	entry, metrics, err := e.lookupLinearModel(batch, inDim, outDim)
	if err != nil {
		return nil, err
	}

	y, evalDur, err := entry.Eval(ctx, x, w, inDim, outDim)
	e.mu.Lock()
	metrics.Evaluate = evalDur
	e.last = metrics
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return y, nil
}

func (e *applePrivateDynamicLinearExecutor) lookupLinearModel(batch, inDim, outDim int) (*dynamicLinearModel, linearTelemetry, error) {
	paddedInDim, paddedOutDim := dynamicLinearPaddedDims(inDim, outDim)
	key := dynamicLinearModelKey{
		batch:        batch,
		paddedInDim:  paddedInDim,
		paddedOutDim: paddedOutDim,
	}
	metrics := linearTelemetry{}

	e.mu.Lock()
	entry, ok := e.linear[key]
	metrics.CacheHit = ok
	if !ok {
		built, builtMetrics, err := buildDynamicLinearModel(key)
		if err != nil {
			e.mu.Unlock()
			return nil, linearTelemetry{}, err
		}
		metrics.Build = builtMetrics.Build
		metrics.Compile = builtMetrics.Compile
		metrics.Load = builtMetrics.Load
		entry = built
		e.linear[key] = entry
	}
	e.last = metrics
	e.mu.Unlock()
	return entry, metrics, nil
}

func buildDynamicLinearModel(key dynamicLinearModelKey) (*dynamicLinearModel, linearTelemetry, error) {
	start := time.Now()

	milText, err := dynamicLinearPackedMILText(key.batch, key.paddedInDim, key.paddedOutDim)
	if err != nil {
		return nil, linearTelemetry{}, fmt.Errorf("appleneuralengine dynamic linear build: generate MIL text: %w", err)
	}
	inMemory, err := buildModelFromMILTextWithDescriptorFallback(
		"appleneuralengine dynamic linear",
		milText,
		nil,
	)
	if err != nil {
		return nil, linearTelemetry{}, err
	}

	packedIn, err := NewIOSurfaceFloat32(key.batch*key.paddedInDim + key.paddedOutDim*key.paddedInDim)
	if err != nil {
		_ = callObjCBoolWithNSError("appleneuralengine dynamic linear unload", inMemory.ID, "unloadWithQoS:error:", defaultANEQoS)
		return nil, linearTelemetry{Build: time.Since(start)}, fmt.Errorf(
			"appleneuralengine dynamic linear build: alloc packed surface: %w",
			err,
		)
	}
	yOut, err := NewIOSurfaceFloat32(key.batch * key.paddedOutDim)
	if err != nil {
		packedIn.Close()
		_ = callObjCBoolWithNSError("appleneuralengine dynamic linear unload", inMemory.ID, "unloadWithQoS:error:", defaultANEQoS)
		return nil, linearTelemetry{Build: time.Since(start)}, fmt.Errorf(
			"appleneuralengine dynamic linear build: alloc y surface: %w",
			err,
		)
	}

	base := inMemory.Model()
	inputSyms, err := dynamicLinearProcedureSymbolIndices(base, true)
	if err != nil {
		packedIn.Close()
		yOut.Close()
		_ = callObjCBoolWithNSError("appleneuralengine dynamic linear unload", inMemory.ID, "unloadWithQoS:error:", defaultANEQoS)
		return nil, linearTelemetry{Build: time.Since(start)}, fmt.Errorf("appleneuralengine dynamic linear build: input symbols: %w", err)
	}
	if len(inputSyms) < 1 {
		packedIn.Close()
		yOut.Close()
		_ = callObjCBoolWithNSError("appleneuralengine dynamic linear unload", inMemory.ID, "unloadWithQoS:error:", defaultANEQoS)
		return nil, linearTelemetry{Build: time.Since(start)}, fmt.Errorf(
			"appleneuralengine dynamic linear build: input symbols=%v want at least 1",
			inputSyms,
		)
	}
	outputSyms, err := dynamicLinearProcedureSymbolIndices(base, false)
	if err != nil {
		packedIn.Close()
		yOut.Close()
		_ = callObjCBoolWithNSError("appleneuralengine dynamic linear unload", inMemory.ID, "unloadWithQoS:error:", defaultANEQoS)
		return nil, linearTelemetry{Build: time.Since(start)}, fmt.Errorf("appleneuralengine dynamic linear build: output symbols: %w", err)
	}

	cfg := DefaultMultiSurfaceEvalPlanConfig()
	plan, err := NewMultiSurfaceEvalPlan(
		inMemory,
		[]SurfaceBinding{
			{Surface: packedIn, SymbolIndex: inputSyms[0]},
		},
		[]SurfaceBinding{
			{Surface: yOut, SymbolIndex: outputSyms[0]},
		},
		cfg,
	)
	if err != nil && !cfg.DisableCacheMapping {
		retry := cfg
		retry.DisableCacheMapping = true
		plan, err = NewMultiSurfaceEvalPlan(
			inMemory,
			[]SurfaceBinding{
				{Surface: packedIn, SymbolIndex: inputSyms[0]},
			},
			[]SurfaceBinding{
				{Surface: yOut, SymbolIndex: outputSyms[0]},
			},
			retry,
		)
	}
	if err != nil {
		packedIn.Close()
		yOut.Close()
		_ = callObjCBoolWithNSError("appleneuralengine dynamic linear unload", inMemory.ID, "unloadWithQoS:error:", defaultANEQoS)
		return nil, linearTelemetry{Build: time.Since(start)}, fmt.Errorf(
			"appleneuralengine dynamic linear build: create eval plan: %w",
			err,
		)
	}

	return &dynamicLinearModel{
			key:      key,
			inMemory: inMemory,
			plan:     plan,
			packedIn: packedIn,
			yOut:     yOut,
		}, linearTelemetry{
			Build: time.Since(start),
		}, nil
}

func dynamicLinearProcedureSymbolIndices(base objectivec.IObject, input bool) ([]int, error) {
	if base == nil || base.GetID() == 0 {
		return nil, fmt.Errorf("model is nil")
	}
	sel := "outputSymbolIndicesForProcedureIndex:"
	if input {
		sel = "inputSymbolIndicesForProcedureIndex:"
	}
	containerID := objc.Send[objc.ID](base.GetID(), objc.Sel(sel), uint32(0))
	if containerID == 0 {
		return nil, fmt.Errorf("%s returned nil", sel)
	}
	if objc.Send[bool](containerID, objc.Sel("respondsToSelector:"), objc.Sel("count")) &&
		objc.Send[bool](containerID, objc.Sel("respondsToSelector:"), objc.Sel("firstIndex")) {
		count := int(objc.Send[uint](containerID, objc.Sel("count")))
		if count == 0 {
			return nil, fmt.Errorf("%s returned no indices", sel)
		}
		first := objc.Send[uint](containerID, objc.Sel("firstIndex"))
		if first == ^uint(0) {
			return nil, fmt.Errorf("%s returned no indices", sel)
		}
		out := make([]int, count)
		for i := range out {
			out[i] = int(first) + i
		}
		return out, nil
	}
	if objc.Send[bool](containerID, objc.Sel("respondsToSelector:"), objc.Sel("count")) &&
		objc.Send[bool](containerID, objc.Sel("respondsToSelector:"), objc.Sel("objectAtIndex:")) {
		count := int(objc.Send[uint](containerID, objc.Sel("count")))
		if count == 0 {
			return nil, fmt.Errorf("%s returned no indices", sel)
		}
		out := make([]int, 0, count)
		for i := 0; i < count; i++ {
			valueID := objc.Send[objc.ID](containerID, objc.Sel("objectAtIndex:"), uint(i))
			if valueID == 0 {
				return nil, fmt.Errorf("%s value[%d] is nil", sel, i)
			}
			out = append(out, int(objc.Send[uint32](valueID, objc.Sel("unsignedIntValue"))))
		}
		return out, nil
	}
	return nil, fmt.Errorf("%s returned unsupported container type", sel)
}

func (e *applePrivateDynamicLinearExecutor) Close() {
	if e == nil {
		return
	}
	e.mu.Lock()
	entries := make([]*dynamicLinearModel, 0, len(e.linear))
	for _, entry := range e.linear {
		entries = append(entries, entry)
	}
	e.linear = make(map[dynamicLinearModelKey]*dynamicLinearModel)
	e.mu.Unlock()

	for _, entry := range entries {
		entry.Close()
	}
}

func (e *applePrivateDynamicLinearExecutor) HasLinearModel(batch, inDim, outDim int, _ uint64) bool {
	if e == nil {
		return false
	}
	paddedInDim, paddedOutDim := dynamicLinearPaddedDims(inDim, outDim)
	key := dynamicLinearModelKey{
		batch:        batch,
		paddedInDim:  paddedInDim,
		paddedOutDim: paddedOutDim,
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.linear[key]
	return ok
}

func (e *applePrivateDynamicLinearExecutor) HasLinearRouteModel(batch, inDim, outDim int) bool {
	if e == nil {
		return false
	}
	key := dynamicLinearModelKey{
		batch:        batch,
		paddedInDim:  inDim,
		paddedOutDim: outDim,
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.linear[key]
	return ok
}

func (e *applePrivateDynamicLinearExecutor) LinearRouteShape(batch, inDim, outDim int) (int, int, int) {
	paddedInDim, paddedOutDim := dynamicLinearPaddedDims(inDim, outDim)
	return batch, paddedInDim, paddedOutDim
}

func (e *applePrivateDynamicLinearExecutor) LinearCacheSize() int {
	if e == nil {
		return 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.linear)
}

func (e *applePrivateDynamicLinearExecutor) LastLinearTelemetry() LinearTelemetry {
	if e == nil {
		return LinearTelemetry{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return LinearTelemetry{
		CacheHit: e.last.CacheHit,
		Build:    e.last.Build,
		Compile:  e.last.Compile,
		Load:     e.last.Load,
		Evaluate: e.last.Evaluate,
	}
}
