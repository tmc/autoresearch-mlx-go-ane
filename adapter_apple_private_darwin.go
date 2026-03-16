//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/tmc/apple/corefoundation"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/iosurface"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

const defaultANEQoS uint32 = 21

type linearModelKey struct {
	batch      int
	inDim      int
	outDim     int
	weightHash uint64
}

type applePrivateExecutor struct {
	mu          sync.Mutex
	linear      map[linearModelKey]appleneuralengine.ANEInMemoryModel
	ffnForward  map[ffnForwardModelKey]appleneuralengine.ANEInMemoryModel
	ffnBackward map[ffnBackwardModelKey]appleneuralengine.ANEInMemoryModel
	last        linearTelemetry
}

type linearTelemetry struct {
	CacheHit bool
	Build    time.Duration
	Compile  time.Duration
	Load     time.Duration
	Evaluate time.Duration
}

// NewApplePrivateExecutor returns a LinearExecutor backed by
// github.com/tmc/apple/private/appleneuralengine bindings.
func NewApplePrivateExecutor() (LinearExecutor, error) {
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		return nil, fmt.Errorf("appleneuralengine: ANE not available on this host")
	}
	return &applePrivateExecutor{
		linear:      make(map[linearModelKey]appleneuralengine.ANEInMemoryModel),
		ffnForward:  make(map[ffnForwardModelKey]appleneuralengine.ANEInMemoryModel),
		ffnBackward: make(map[ffnBackwardModelKey]appleneuralengine.ANEInMemoryModel),
	}, nil
}

func (e *applePrivateExecutor) Linear(ctx context.Context, x, w []float32, batch, inDim, outDim int) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("appleneuralengine linear: executor is nil")
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if len(x) != batch*inDim {
		return nil, fmt.Errorf("appleneuralengine linear: x len=%d want=%d", len(x), batch*inDim)
	}
	if len(w) != outDim*inDim {
		return nil, fmt.Errorf("appleneuralengine linear: w len=%d want=%d", len(w), outDim*inDim)
	}

	key := linearModelKey{
		batch:      batch,
		inDim:      inDim,
		outDim:     outDim,
		weightHash: hashFloat32Slice(w),
	}

	metrics := linearTelemetry{}

	e.mu.Lock()
	model, ok := e.linear[key]
	metrics.CacheHit = ok
	if !ok {
		built, builtMetrics, err := buildLinearModel(key, w)
		if err != nil {
			e.mu.Unlock()
			return nil, err
		}
		metrics.Build = builtMetrics.Build
		metrics.Compile = builtMetrics.Compile
		metrics.Load = builtMetrics.Load
		model = built
		e.linear[key] = model
	}
	e.last = metrics
	e.mu.Unlock()

	y, evalDur, err := evalLinearModel(ctx, model, x, batch, inDim, outDim)
	e.mu.Lock()
	metrics.Evaluate = evalDur
	e.last = metrics
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return y, nil
}

func buildLinearModel(key linearModelKey, w []float32) (appleneuralengine.ANEInMemoryModel, linearTelemetry, error) {
	start := time.Now()

	blob, err := buildLinearWeightsBlob(w, key.outDim, key.inDim)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, linearTelemetry{}, fmt.Errorf(
			"appleneuralengine linear model build: create weight blob: %w", err,
		)
	}
	weights := foundation.NewMutableDictionaryWithCapacity(1)
	weightInfo := foundation.NewMutableDictionaryWithCapacity(2)
	weightInfo.SetObjectForKey(
		foundation.NewNumberWithInt(0),
		foundation.NewStringWithString("offset"),
	)
	weightInfo.SetObjectForKey(
		foundation.NewDataWithBytesLength(blob),
		foundation.NewStringWithString("data"),
	)
	weights.SetObjectForKey(
		weightInfo,
		foundation.NewStringWithString(linearWeightBlobPathInMIL),
	)

	milText, err := linearMILText(key.batch, key.inDim, key.outDim)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, linearTelemetry{}, fmt.Errorf(
			"appleneuralengine linear model build: generate MIL text: %w", err,
		)
	}
	descObj, usedMILInit, err := newMILTextDescriptor(milText, weights, objectivec.Object{})
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, linearTelemetry{}, fmt.Errorf(
			"appleneuralengine linear model build: %w",
			err,
		)
	}
	if err := mirrorDescriptorFiles(descObj, milText, blob); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, linearTelemetry{}, err
	}
	sourceLabel := "MIL text"
	if usedMILInit {
		sourceLabel = "MIL text (isMILModel init)"
	}
	model, compileDur, loadDur, err := compileAndLoadLinearModel(sourceLabel, descObj)
	metrics := linearTelemetry{
		Build:   time.Since(start),
		Compile: compileDur,
		Load:    loadDur,
	}
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, metrics, err
	}
	return model, metrics, nil
}

func mirrorDescriptorFiles(descObj objectivec.IObject, milText string, blob []byte) error {
	desc := appleneuralengine.ANEInMemoryModelDescriptorFromID(descObj.GetID())
	hexIDObj := desc.HexStringIdentifier()
	if hexIDObj.GetID() == 0 {
		return fmt.Errorf("appleneuralengine linear model build: descriptor hex string identifier is nil")
	}
	hexID := foundation.NSStringFromID(hexIDObj.GetID()).String()
	if hexID == "" {
		return fmt.Errorf("appleneuralengine linear model build: descriptor hex string identifier is empty")
	}

	modelDir := filepath.Join(os.TempDir(), hexID)
	weightsDir := filepath.Join(modelDir, "weights")
	if err := os.MkdirAll(weightsDir, 0o755); err != nil {
		return fmt.Errorf("appleneuralengine linear model build: create mirrored temp dirs: %w", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.mil"), []byte(milText), 0o644); err != nil {
		return fmt.Errorf("appleneuralengine linear model build: write mirrored model.mil: %w", err)
	}
	if err := os.WriteFile(filepath.Join(weightsDir, "weight.bin"), blob, 0o644); err != nil {
		return fmt.Errorf("appleneuralengine linear model build: write mirrored weight.bin: %w", err)
	}
	return nil
}

func compileAndLoadLinearModel(source string, descObj objectivec.IObject) (appleneuralengine.ANEInMemoryModel, time.Duration, time.Duration, error) {
	modelObj := appleneuralengine.GetANEInMemoryModelClass().InMemoryModelWithDescriptor(descObj)
	if modelObj.GetID() == 0 {
		return appleneuralengine.ANEInMemoryModel{}, 0, 0, fmt.Errorf(
			"appleneuralengine linear model (%s): create in-memory model failed", source,
		)
	}
	model := appleneuralengine.ANEInMemoryModelFromID(modelObj.GetID())
	if mask := benchmarkPerfStatsMask(); mask != 0 {
		model.SetPerfStatsMask(mask)
	}
	options := foundation.NewMutableDictionaryWithCapacity(0)

	compileStart := time.Now()
	if err := compileWithFallbackProfiles(model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, time.Since(compileStart), 0, err
	}
	compileDur := time.Since(compileStart)

	loadDur, err := loadInMemoryModelWithRetry(
		fmt.Sprintf("appleneuralengine linear model (%s)", source),
		model,
		options,
	)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, compileDur, loadDur, err
	}
	return model, compileDur, loadDur, nil
}

func loadInMemoryModelWithRetry(
	label string,
	model appleneuralengine.ANEInMemoryModel,
	options foundation.NSMutableDictionary,
) (time.Duration, error) {
	start := time.Now()
	if _, err := model.LoadWithQoSOptionsError(defaultANEQoS, options); err == nil {
		return time.Since(start), nil
	} else {
		time.Sleep(100 * time.Millisecond)
		if _, retryErr := model.LoadWithQoSOptionsError(defaultANEQoS, options); retryErr != nil {
			return time.Since(start), fmt.Errorf("%s load after retry: %w", label, retryErr)
		}
		return time.Since(start), nil
	}
}

func evalLinearModel(ctx context.Context, model appleneuralengine.ANEInMemoryModel, x []float32, batch, inDim, outDim int) ([]float32, time.Duration, error) {
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}

	xANE, err := linearInputHostToANE(x, batch, inDim)
	if err != nil {
		return nil, 0, err
	}

	xSurf, err := newFloatSurface(len(xANE))
	if err != nil {
		return nil, 0, err
	}
	defer releaseIOSurface(xSurf)
	if err := writeFloat32IOSurface(xSurf, xANE); err != nil {
		return nil, 0, err
	}

	yCount := batch * outDim
	ySurf, err := newFloatSurface(yCount)
	if err != nil {
		return nil, 0, err
	}
	defer releaseIOSurface(ySurf)

	xObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(xSurf)
	yObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(ySurf)
	if xObj.GetID() == 0 || yObj.GetID() == 0 {
		return nil, 0, fmt.Errorf("appleneuralengine linear: create IOSurface object failed")
	}

	inputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{xObj}))
	inputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	outputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{yObj}))
	outputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	procedureIndex := foundation.NewNumberWithInt(0)

	requestObj := appleneuralengine.GetANERequestClass().RequestWithInputsInputIndicesOutputsOutputIndicesProcedureIndex(
		inputs,
		inputIndices,
		outputs,
		outputIndices,
		procedureIndex,
	)
	if requestObj.GetID() == 0 {
		return nil, 0, fmt.Errorf("appleneuralengine linear: create request failed")
	}
	request := appleneuralengine.ANERequestFromID(requestObj.GetID())
	if !request.Validate() {
		return nil, 0, fmt.Errorf("appleneuralengine linear: request validation failed")
	}

	unmap, err := mapModelRequestWithFallback(
		"appleneuralengine linear map IOSurfaces",
		model,
		requestObj,
		true,
	)
	if err != nil {
		return nil, 0, err
	}
	defer unmap()

	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}

	evalStart := time.Now()
	options := foundation.NewMutableDictionaryWithCapacity(0)
	if err := callModelEvaluateWithFallback(
		"appleneuralengine linear evaluate",
		model,
		defaultANEQoS,
		options,
		requestObj,
	); err != nil {
		return nil, time.Since(evalStart), err
	}
	evalDur := time.Since(evalStart)

	yANE, err := readFloat32IOSurface(ySurf, yCount)
	if err != nil {
		return nil, evalDur, err
	}
	y, err := linearOutputANEToHost(yANE, batch, outDim)
	if err != nil {
		return nil, evalDur, err
	}
	return y, evalDur, nil
}

// linearInputHostToANE maps host row-major [batch, inDim] to ANE tensor memory
// order for shape [1, inDim, 1, batch].
func linearInputHostToANE(x []float32, batch, inDim int) ([]float32, error) {
	if batch <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("appleneuralengine linear: invalid input layout dims batch=%d in=%d", batch, inDim)
	}
	if len(x) != batch*inDim {
		return nil, fmt.Errorf("appleneuralengine linear: input layout len=%d want=%d", len(x), batch*inDim)
	}
	if batch == 1 {
		out := make([]float32, len(x))
		copy(out, x)
		return out, nil
	}
	out := make([]float32, len(x))
	for b := 0; b < batch; b++ {
		for i := 0; i < inDim; i++ {
			out[i*batch+b] = x[b*inDim+i]
		}
	}
	return out, nil
}

// linearOutputANEToHost maps ANE tensor memory order for [1, outDim, 1, batch]
// back to host row-major [batch, outDim].
func linearOutputANEToHost(y []float32, batch, outDim int) ([]float32, error) {
	if batch <= 0 || outDim <= 0 {
		return nil, fmt.Errorf("appleneuralengine linear: invalid output layout dims batch=%d out=%d", batch, outDim)
	}
	if len(y) != batch*outDim {
		return nil, fmt.Errorf("appleneuralengine linear: output layout len=%d want=%d", len(y), batch*outDim)
	}
	if batch == 1 {
		out := make([]float32, len(y))
		copy(out, y)
		return out, nil
	}
	out := make([]float32, len(y))
	for b := 0; b < batch; b++ {
		for o := 0; o < outDim; o++ {
			out[b*outDim+o] = y[o*batch+b]
		}
	}
	return out, nil
}

func newFloatSurface(count int) (coregraphics.IOSurfaceRef, error) {
	if count <= 0 {
		return 0, fmt.Errorf("appleneuralengine linear: invalid surface count=%d", count)
	}
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
		return 0, fmt.Errorf("appleneuralengine linear: IOSurface allocation failed for %d float32 values", count)
	}
	return coregraphics.IOSurfaceRef(raw), nil
}

func releaseIOSurface(surface coregraphics.IOSurfaceRef) {
	if surface == 0 {
		return
	}
	corefoundation.CFRelease(corefoundation.CFTypeRef(surface))
}

func writeFloat32IOSurface(surface coregraphics.IOSurfaceRef, data []float32) error {
	if len(data) == 0 {
		return nil
	}
	s := iosurface.IOSurfaceRef(surface)
	if rc := iosurface.IOSurfaceLock(s, 0, nil); rc != 0 {
		return fmt.Errorf("appleneuralengine linear: IOSurface lock failed rc=%d", rc)
	}
	defer iosurface.IOSurfaceUnlock(s, 0, nil)

	base := iosurface.IOSurfaceGetBaseAddress(s)
	if base == nil {
		return fmt.Errorf("appleneuralengine linear: IOSurface base address is nil")
	}
	n := len(data) * 4
	if got := iosurface.IOSurfaceGetAllocSize(s); got < uintptr(n) {
		return fmt.Errorf("appleneuralengine linear: IOSurface size=%d want>=%d", got, n)
	}

	dst := unsafe.Slice((*byte)(base), n)
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), n)
	copy(dst, src)
	return nil
}

func readFloat32IOSurface(surface coregraphics.IOSurfaceRef, count int) ([]float32, error) {
	if count < 0 {
		return nil, fmt.Errorf("appleneuralengine linear: invalid output count=%d", count)
	}
	out := make([]float32, count)
	if count == 0 {
		return out, nil
	}

	s := iosurface.IOSurfaceRef(surface)
	if rc := iosurface.IOSurfaceLock(s, iosurface.KIOSurfaceLockReadOnly, nil); rc != 0 {
		return nil, fmt.Errorf("appleneuralengine linear: IOSurface read lock failed rc=%d", rc)
	}
	defer iosurface.IOSurfaceUnlock(s, iosurface.KIOSurfaceLockReadOnly, nil)

	base := iosurface.IOSurfaceGetBaseAddress(s)
	if base == nil {
		return nil, fmt.Errorf("appleneuralengine linear: output IOSurface base address is nil")
	}
	n := count * 4
	if got := iosurface.IOSurfaceGetAllocSize(s); got < uintptr(n) {
		return nil, fmt.Errorf("appleneuralengine linear: output IOSurface size=%d want>=%d", got, n)
	}

	src := unsafe.Slice((*byte)(base), n)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), n)
	copy(dst, src)
	return out, nil
}

func (e *applePrivateExecutor) linearCacheSize() int {
	if e == nil {
		return 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.linear)
}

func (e *applePrivateExecutor) lastLinearTelemetry() linearTelemetry {
	if e == nil {
		return linearTelemetry{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.last
}

// HasLinearModel reports whether the shape+weight key is already compiled.
func (e *applePrivateExecutor) HasLinearModel(batch, inDim, outDim int, weightHash uint64) bool {
	if e == nil {
		return false
	}
	key := linearModelKey{
		batch:      batch,
		inDim:      inDim,
		outDim:     outDim,
		weightHash: weightHash,
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.linear[key]
	return ok
}

// LinearCacheSize returns the count of compiled linear model entries.
func (e *applePrivateExecutor) LinearCacheSize() int {
	return e.linearCacheSize()
}

// LastLinearTelemetry returns the most recent linear telemetry snapshot.
func (e *applePrivateExecutor) LastLinearTelemetry() LinearTelemetry {
	m := e.lastLinearTelemetry()
	return LinearTelemetry{
		CacheHit: m.CacheHit,
		Build:    m.Build,
		Compile:  m.Compile,
		Load:     m.Load,
		Evaluate: m.Evaluate,
	}
}

func callObjCBoolWithNSError(label string, receiver objc.ID, selector string, args ...any) error {
	var errID objc.ID
	argsWithErr := append(args, unsafe.Pointer(&errID))
	ok := objc.Send[bool](receiver, objc.Sel(selector), argsWithErr...)
	if ok {
		return nil
	}
	if errID == 0 {
		return fmt.Errorf("%s failed (objc returned NO with nil NSError)", label)
	}
	return fmt.Errorf("%s: %s", label, objectivec.ObjectFromID(errID).Description())
}

func fallbackDebugf(format string, args ...any) {
	if os.Getenv("MLXGO_ANE_DEBUG_FALLBACK") == "" {
		return
	}
	fmt.Fprintf(os.Stderr, "mlxgoane fallback: "+format+"\n", args...)
}

func callModelEvaluateWithFallback(
	label string,
	model appleneuralengine.ANEInMemoryModel,
	qos uint32,
	options objectivec.IObject,
	request objectivec.IObject,
) error {
	shared := model.SharedConnection()
	base := model.Model()
	fallbackDebugf(
		"evaluate start label=%q shared=%#x base=%#x model=%#x",
		label,
		uintptr(iobjID(shared)),
		uintptr(iobjID(base)),
		uintptr(model.ID),
	)
	if iobjID(shared) != 0 && iobjID(base) != 0 &&
		objc.Send[bool](iobjID(shared), objc.Sel("respondsToSelector:"), objc.Sel("doEvaluateDirectWithModel:options:request:qos:error:")) {
		if _, err := shared.DoEvaluateDirectWithModelOptionsRequestQosError(base, options, request, qos); err == nil {
			fallbackDebugf("evaluate path=client_direct result=ok label=%q", label)
			return nil
		} else {
			fallbackDebugf("evaluate path=client_direct result=err label=%q err=%v", label, err)
		}
	} else {
		fallbackDebugf("evaluate path=client_direct result=skipped label=%q", label)
	}
	_, err := model.EvaluateWithQoSOptionsRequestError(qos, options, request)
	if err != nil {
		fallbackDebugf("evaluate path=inmemory result=err label=%q err=%v", label, err)
		return err
	}
	fallbackDebugf("evaluate path=inmemory result=ok label=%q", label)
	return nil
}

func mapModelRequestWithFallback(
	label string,
	model appleneuralengine.ANEInMemoryModel,
	request objectivec.IObject,
	cacheInference bool,
) (func(), error) {
	shared := model.SharedConnection()
	base := model.Model()
	fallbackDebugf(
		"map start label=%q shared=%#x base=%#x model=%#x cacheInference=%v",
		label,
		uintptr(iobjID(shared)),
		uintptr(iobjID(base)),
		uintptr(model.ID),
		cacheInference,
	)
	var clientErr error
	if iobjID(shared) != 0 && iobjID(base) != 0 &&
		objc.Send[bool](iobjID(shared), objc.Sel("respondsToSelector:"), objc.Sel("mapIOSurfacesWithModel:request:cacheInference:error:")) &&
		objc.Send[bool](iobjID(shared), objc.Sel("respondsToSelector:"), objc.Sel("unmapIOSurfacesWithModel:request:")) {
		if _, err := shared.MapIOSurfacesWithModelRequestCacheInferenceError(base, request, cacheInference); err == nil {
			fallbackDebugf("map path=client result=ok label=%q", label)
			return func() {
				shared.UnmapIOSurfacesWithModelRequest(base, request)
				fallbackDebugf("unmap path=client result=ok label=%q", label)
			}, nil
		} else {
			fallbackDebugf("map path=client result=err label=%q err=%v", label, err)
			clientErr = err
		}
	} else {
		fallbackDebugf("map path=client result=skipped label=%q", label)
		clientErr = fmt.Errorf(
			"daemon map unavailable (shared=%#x base=%#x mapSelector=%t unmapSelector=%t)",
			uintptr(iobjID(shared)),
			uintptr(iobjID(base)),
			iobjID(shared) != 0 && objc.Send[bool](iobjID(shared), objc.Sel("respondsToSelector:"), objc.Sel("mapIOSurfacesWithModel:request:cacheInference:error:")),
			iobjID(shared) != 0 && objc.Send[bool](iobjID(shared), objc.Sel("respondsToSelector:"), objc.Sel("unmapIOSurfacesWithModel:request:")),
		)
	}
	if os.Getenv("MLXGO_ANE_ALLOW_INMEMORY_MAP_FALLBACK") == "" {
		if isIOSurfaceMapFailure(clientErr) {
			return nil, fmt.Errorf(
				"%s: daemon IOSurface map failed: %w (local in-memory map fallback disabled; restart the process and reboot host if 0x1D persists)",
				label,
				clientErr,
			)
		}
		return nil, fmt.Errorf(
			"%s: daemon IOSurface map failed: %w (local in-memory map fallback disabled; set MLXGO_ANE_ALLOW_INMEMORY_MAP_FALLBACK=1 to override)",
			label,
			clientErr,
		)
	}
	fallbackDebugf("map path=inmemory fallback enabled label=%q", label)
	if _, err := model.MapIOSurfacesWithRequestCacheInferenceError(request, cacheInference); err != nil {
		fallbackDebugf("map path=inmemory result=err label=%q err=%v", label, err)
		if clientErr != nil {
			return nil, fmt.Errorf("%s: daemon IOSurface map failed: %v; in-memory fallback failed: %w", label, clientErr, err)
		}
		return nil, err
	}
	fallbackDebugf("map path=inmemory result=ok label=%q", label)
	return func() {
		model.UnmapIOSurfacesWithRequest(request)
		fallbackDebugf("unmap path=inmemory result=ok label=%q", label)
	}, nil
}

func isIOSurfaceMapFailure(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "Program IOSurfaces map failure") ||
		strings.Contains(msg, "mapIOSurfacesWithModel")
}

func iobjID(obj objectivec.IObject) objc.ID {
	if obj == nil {
		return 0
	}
	return obj.GetID()
}
