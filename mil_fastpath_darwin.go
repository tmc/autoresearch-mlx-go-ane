//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

const defaultMILFastPathKey = "s"

// ErrMILDirectoryUnsupported indicates that compileModel rejected a MIL-only
// directory and expected Espresso graph files.
var ErrMILDirectoryUnsupported = errors.New("mil-only directory unsupported by compileModel")

// ANEClientMILModel is a file-backed MIL model loaded through _ANEClient.
type ANEClientMILModel struct {
	client  appleneuralengine.ANEClient
	model   objectivec.IObject
	options foundation.NSMutableDictionary
	qos     uint32
	dir     string
	owned   bool

	closeOnce sync.Once
}

// CompileAndLoadMIL writes MIL + weight blob to disk and loads it via
// _ANEModel.modelAtURL:key: + _ANEClient compile/load selectors.
func CompileAndLoadMIL(
	client appleneuralengine.ANEClient,
	milText string,
	weightBlob []byte,
	key string,
	qos uint32,
) (*ANEClientMILModel, error) {
	return CompileAndLoadMILFiles(
		client,
		milText,
		[]ModelWeightFile{
			{Path: linearWeightBlobPathInMIL, Blob: weightBlob},
		},
		key,
		qos,
	)
}

// CompileAndLoadMILFiles writes MIL + weight files to disk and loads them via
// _ANEModel.modelAtURL:key: + _ANEClient compile/load selectors.
func CompileAndLoadMILFiles(
	client appleneuralengine.ANEClient,
	milText string,
	files []ModelWeightFile,
	key string,
	qos uint32,
) (*ANEClientMILModel, error) {
	if client.ID == 0 {
		return nil, fmt.Errorf("compile/load mil files: client is nil")
	}
	if strings.TrimSpace(milText) == "" {
		return nil, fmt.Errorf("compile/load mil files: mil text is empty")
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("compile/load mil files: weight files are empty")
	}
	if key == "" {
		key = defaultMILFastPathKey
	}
	if qos == 0 {
		qos = defaultANEQoS
	}

	dir, err := os.MkdirTemp("", "ane_model_*.mlmodelc")
	if err != nil {
		return nil, fmt.Errorf("compile/load mil files: create temp dir: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(dir) }

	if err := writeMILModelDirectory(dir, milText, files); err != nil {
		cleanup()
		return nil, fmt.Errorf("compile/load mil files: %w", err)
	}

	model, err := compileAndLoadModelDirectory(client, dir, key, qos)
	if err != nil {
		cleanup()
		if strings.Contains(err.Error(), "model.espresso.net") {
			return nil, fmt.Errorf("%w: %v", ErrMILDirectoryUnsupported, err)
		}
		return nil, err
	}
	model.dir = dir
	model.owned = true
	return model, nil
}

// CompileAndLoadEspresso loads an on-disk Espresso model directory through
// _ANEModel.modelAtURL:key: + _ANEClient compile/load selectors.
func CompileAndLoadEspresso(
	client appleneuralengine.ANEClient,
	dir string,
	key string,
	qos uint32,
) (*ANEClientMILModel, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, fmt.Errorf("compile/load espresso: model dir is empty")
	}
	if model, err := loadModelDirectoryWithoutCompile(client, dir, key, qos); err == nil {
		return model, nil
	}
	return compileAndLoadModelDirectory(client, dir, key, qos)
}

func compileAndLoadModelDirectory(
	client appleneuralengine.ANEClient,
	dir string,
	key string,
	qos uint32,
) (*ANEClientMILModel, error) {
	if client.ID == 0 {
		return nil, fmt.Errorf("compile/load model: client is nil")
	}
	if key == "" {
		key = defaultMILFastPathKey
	}
	if qos == 0 {
		qos = defaultANEQoS
	}

	modelURL := foundation.NewURLFileURLWithPath(dir)
	modelObj := appleneuralengine.GetANEModelClass().ModelAtURLKey(
		modelURL,
		foundation.NewStringWithString(key),
	)
	if modelObj.GetID() == 0 {
		return nil, fmt.Errorf("compile/load model: _ANEModel modelAtURL:key: returned nil")
	}
	applyBenchmarkPerfStatsMaskObject(modelObj)

	options := foundation.NewMutableDictionaryWithCapacity(0)
	if _, err := client.CompileModelOptionsQosError(modelObj, options, qos); err != nil {
		return nil, err
	}
	if _, err := client.LoadModelOptionsQosError(modelObj, options, qos); err != nil {
		return nil, err
	}
	return &ANEClientMILModel{
		client:  client,
		model:   modelObj,
		options: options,
		qos:     qos,
		dir:     dir,
	}, nil
}

func loadModelDirectoryWithoutCompile(
	client appleneuralengine.ANEClient,
	dir string,
	key string,
	qos uint32,
) (*ANEClientMILModel, error) {
	if client.ID == 0 {
		return nil, fmt.Errorf("load model: client is nil")
	}
	if key == "" {
		key = defaultMILFastPathKey
	}
	if qos == 0 {
		qos = defaultANEQoS
	}

	modelURL := foundation.NewURLFileURLWithPath(dir)
	modelObj := appleneuralengine.GetANEModelClass().ModelAtURLKey(
		modelURL,
		foundation.NewStringWithString(key),
	)
	if modelObj.GetID() == 0 {
		return nil, fmt.Errorf("load model: _ANEModel modelAtURL:key: returned nil")
	}
	applyBenchmarkPerfStatsMaskObject(modelObj)

	options := foundation.NewMutableDictionaryWithCapacity(0)
	if _, err := client.LoadModelOptionsQosError(modelObj, options, qos); err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return &ANEClientMILModel{
		client:  client,
		model:   modelObj,
		options: options,
		qos:     qos,
		dir:     dir,
	}, nil
}

func applyBenchmarkPerfStatsMaskObject(model objectivec.IObject) {
	mask := benchmarkPerfStatsMask()
	if mask == 0 || model.GetID() == 0 {
		return
	}
	defer func() { recover() }()
	appleneuralengine.ANEModelFromID(model.GetID()).SetPerfStatsMask(mask)
}

func writeMILModelDirectory(dir string, milText string, files []ModelWeightFile) error {
	if strings.TrimSpace(milText) == "" {
		return fmt.Errorf("write model dir: mil text is empty")
	}
	if len(files) == 0 {
		return fmt.Errorf("write model dir: weight files are empty")
	}
	if err := os.WriteFile(filepath.Join(dir, "model.mil"), []byte(milText), 0o644); err != nil {
		return fmt.Errorf("write model dir: write model.mil: %w", err)
	}
	for _, f := range files {
		relPath, err := relativeModelPath(f.Path)
		if err != nil {
			return fmt.Errorf("write model dir: invalid weight path %q: %w", f.Path, err)
		}
		dstPath := filepath.Join(dir, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
			return fmt.Errorf("write model dir: create weight dir: %w", err)
		}
		if err := os.WriteFile(dstPath, f.Blob, 0o644); err != nil {
			return fmt.Errorf("write model dir: write %q: %w", relPath, err)
		}
	}
	return nil
}

// EvalSingleIO evaluates one request against the loaded model.
//
// input is expected in ANE memory order matching the model input.
func (m *ANEClientMILModel) EvalSingleIO(
	ctx context.Context,
	input []float32,
	outputCount int,
	preferDirect bool,
) ([]float32, time.Duration, error) {
	if m == nil {
		return nil, 0, fmt.Errorf("mil fast path eval: model is nil")
	}
	if outputCount <= 0 {
		return nil, 0, fmt.Errorf("mil fast path eval: invalid outputCount=%d", outputCount)
	}
	if len(input) == 0 {
		return nil, 0, fmt.Errorf("mil fast path eval: input is empty")
	}
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}

	xSurf, err := newFloatSurface(len(input))
	if err != nil {
		return nil, 0, err
	}
	defer releaseIOSurface(xSurf)
	if err := writeFloat32IOSurface(xSurf, input); err != nil {
		return nil, 0, err
	}

	ySurf, err := newFloatSurface(outputCount)
	if err != nil {
		return nil, 0, err
	}
	defer releaseIOSurface(ySurf)

	xObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(xSurf)
	yObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(ySurf)
	if xObj.GetID() == 0 || yObj.GetID() == 0 {
		return nil, 0, fmt.Errorf("mil fast path eval: create IOSurface object failed")
	}
	inputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{xObj}))
	inputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	outputs := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{yObj}))
	outputIndices := objectivec.ObjectFromID(objectivec.IObjectSliceToNSArray([]objectivec.IObject{
		foundation.NewNumberWithInt(0),
	}))
	request := appleneuralengine.GetANERequestClass().RequestWithInputsInputIndicesOutputsOutputIndicesProcedureIndex(
		inputs,
		inputIndices,
		outputs,
		outputIndices,
		foundation.NewNumberWithInt(0),
	)
	if request.GetID() == 0 {
		return nil, 0, fmt.Errorf("mil fast path eval: request creation failed")
	}
	req := appleneuralengine.ANERequestFromID(request.GetID())
	if !req.Validate() {
		return nil, 0, fmt.Errorf("mil fast path eval: request validation failed")
	}

	if _, err := m.client.MapIOSurfacesWithModelRequestCacheInferenceError(m.model, request, true); err != nil {
		return nil, 0, err
	}
	defer m.client.UnmapIOSurfacesWithModelRequest(m.model, request)

	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}

	startEval := time.Now()
	if preferDirect &&
		objc.Send[bool](m.client.ID, objc.Sel("respondsToSelector:"), objc.Sel("doEvaluateDirectWithModel:options:request:qos:error:")) {
		if _, err := m.client.DoEvaluateDirectWithModelOptionsRequestQosError(m.model, m.options, request, m.qos); err != nil {
			return nil, time.Since(startEval), err
		}
	} else {
		if _, err := m.client.EvaluateWithModelOptionsRequestQosError(m.model, m.options, request, m.qos); err != nil {
			return nil, time.Since(startEval), err
		}
	}
	evalDur := time.Since(startEval)

	out, err := readFloat32IOSurface(ySurf, outputCount)
	if err != nil {
		return nil, evalDur, err
	}
	return out, evalDur, nil
}

// Close unloads the model and removes the temporary model directory.
func (m *ANEClientMILModel) Close() {
	if m == nil {
		return
	}
	m.closeOnce.Do(func() {
		if m.client.ID != 0 && m.model.GetID() != 0 {
			_, _ = m.client.UnloadModelOptionsQosError(m.model, m.options, m.qos)
		}
		if m.owned && m.dir != "" {
			_ = os.RemoveAll(m.dir)
		}
		m.model = objectivec.Object{}
		m.options = foundation.NSMutableDictionary{}
		m.client = appleneuralengine.ANEClient{}
		m.dir = ""
	})
}
