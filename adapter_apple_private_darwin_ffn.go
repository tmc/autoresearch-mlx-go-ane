//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
	anereify "github.com/tmc/mlx-go/modelir/target/mil/ane"
)

const modelPathPrefix = "@model_path/"

type ffnForwardModelKey struct {
	dim       int
	hiddenDim int
	seq       int
	rms2Hash  uint64
	w1Hash    uint64
	w3Hash    uint64
	w2Hash    uint64
}

type ffnBackwardModelKey struct {
	dim       int
	hiddenDim int
	seq       int
	w1Hash    uint64
	w3Hash    uint64
	w2Hash    uint64
}

type modelWeightFile = anereify.ModelWeightFile

// ModelWeightFile is one MIL weight payload.
//
// Path must use the @model_path/ prefix expected by ANE MIL descriptors.
type ModelWeightFile = modelWeightFile

var _ FFNExecutor = (*applePrivateExecutor)(nil)

var (
	modelMirrorRootMu sync.RWMutex
	modelMirrorRoot   string
)

// SetModelMirrorRoot overrides the directory used for mirrored MIL model files.
//
// When dir is empty, the package falls back to os.TempDir.
func SetModelMirrorRoot(dir string) {
	modelMirrorRootMu.Lock()
	defer modelMirrorRootMu.Unlock()
	modelMirrorRoot = strings.TrimSpace(dir)
}

func modelMirrorBaseDir() string {
	modelMirrorRootMu.RLock()
	defer modelMirrorRootMu.RUnlock()
	if modelMirrorRoot == "" {
		return os.TempDir()
	}
	return modelMirrorRoot
}

func descriptorMirrorBaseDir() string {
	return os.TempDir()
}

func (e *applePrivateExecutor) FFNForward(
	ctx context.Context,
	x, rms2, w1, w3, w2 []float32,
	dim, hiddenDim, seq int,
) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("appleneuralengine ffn forward: executor is nil")
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if dim <= 0 || hiddenDim <= 0 || seq <= 0 {
		return nil, fmt.Errorf(
			"appleneuralengine ffn forward: invalid dims dim=%d hiddenDim=%d seq=%d",
			dim, hiddenDim, seq,
		)
	}
	if len(x) != dim*seq {
		return nil, fmt.Errorf("appleneuralengine ffn forward: x len=%d want=%d", len(x), dim*seq)
	}
	mapSeq := effectiveFFNMapSeq(seq)
	mapInput := x
	if mapSeq != seq {
		padded, err := padANEChannelMajorSeq(x, dim, seq, mapSeq)
		if err != nil {
			return nil, fmt.Errorf("appleneuralengine ffn forward: pad input: %w", err)
		}
		mapInput = padded
	}

	key := ffnForwardModelKey{
		dim:       dim,
		hiddenDim: hiddenDim,
		seq:       mapSeq,
		rms2Hash:  hashFloat32Slice(rms2),
		w1Hash:    hashFloat32Slice(w1),
		w3Hash:    hashFloat32Slice(w3),
		w2Hash:    hashFloat32Slice(w2),
	}

	e.mu.Lock()
	model, ok := e.ffnForward[key]
	if !ok {
		built, err := buildFFNForwardModel(key, rms2, w1, w3, w2)
		if err != nil {
			e.mu.Unlock()
			return nil, err
		}
		model = built
		e.ffnForward[key] = model
	}
	e.mu.Unlock()

	outChannels := 2*dim + 3*hiddenDim
	packed, err := evalModelSingleIO(
		ctx, model, mapInput, outChannels*mapSeq, "appleneuralengine ffn forward",
	)
	if err != nil {
		return nil, err
	}
	if mapSeq == seq {
		return packed, nil
	}
	trimmed, err := trimANEChannelMajorSeq(packed, outChannels, mapSeq, seq)
	if err != nil {
		return nil, fmt.Errorf("appleneuralengine ffn forward: trim output: %w", err)
	}
	return trimmed, nil
}

func (e *applePrivateExecutor) FFNBackward(
	ctx context.Context,
	dffn, h1, h3, w1, w3, w2 []float32,
	dim, hiddenDim, seq int,
) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("appleneuralengine ffn backward: executor is nil")
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if dim <= 0 || hiddenDim <= 0 || seq <= 0 {
		return nil, fmt.Errorf(
			"appleneuralengine ffn backward: invalid dims dim=%d hiddenDim=%d seq=%d",
			dim, hiddenDim, seq,
		)
	}
	in, err := buildFFNBackwardInputANE(
		dffn,
		FFNForwardTaps{H1: h1, H3: h3},
		dim, hiddenDim, seq,
	)
	if err != nil {
		return nil, fmt.Errorf("appleneuralengine ffn backward: build input: %w", err)
	}
	mapSeq := effectiveFFNMapSeq(seq)
	mapIn := in
	if mapSeq != seq {
		padded, err := padANEChannelMajorSeq(in, dim+2*hiddenDim, seq, mapSeq)
		if err != nil {
			return nil, fmt.Errorf("appleneuralengine ffn backward: pad input: %w", err)
		}
		mapIn = padded
	}

	key := ffnBackwardModelKey{
		dim:       dim,
		hiddenDim: hiddenDim,
		seq:       mapSeq,
		w1Hash:    hashFloat32Slice(w1),
		w3Hash:    hashFloat32Slice(w3),
		w2Hash:    hashFloat32Slice(w2),
	}

	e.mu.Lock()
	model, ok := e.ffnBackward[key]
	if !ok {
		built, err := buildFFNBackwardModel(key, w1, w3, w2)
		if err != nil {
			e.mu.Unlock()
			return nil, err
		}
		model = built
		e.ffnBackward[key] = model
	}
	e.mu.Unlock()

	outChannels := dim + 2*hiddenDim
	packed, err := evalModelSingleIO(
		ctx, model, mapIn, outChannels*mapSeq, "appleneuralengine ffn backward",
	)
	if err != nil {
		return nil, err
	}
	if mapSeq == seq {
		return packed, nil
	}
	trimmed, err := trimANEChannelMajorSeq(packed, outChannels, mapSeq, seq)
	if err != nil {
		return nil, fmt.Errorf("appleneuralengine ffn backward: trim output: %w", err)
	}
	return trimmed, nil
}

func buildFFNForwardModel(
	key ffnForwardModelKey,
	rms2, w1, w3, w2 []float32,
) (appleneuralengine.ANEInMemoryModel, error) {
	blobs, err := buildFFNWeightBlobs(rms2, w1, w3, w2, key.dim, key.hiddenDim)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"appleneuralengine ffn forward model build: create weight blobs: %w", err,
		)
	}
	files := []modelWeightFile{
		{Path: ffnRMS2BlobPathInMIL, Blob: blobs.RMS2},
		{Path: ffnW1BlobPathInMIL, Blob: blobs.W1},
		{Path: ffnW3BlobPathInMIL, Blob: blobs.W3},
		{Path: ffnW2BlobPathInMIL, Blob: blobs.W2},
	}
	milText, err := ffnFwdTapsMILText(key.dim, key.hiddenDim, key.seq)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"appleneuralengine ffn forward model build: generate MIL text: %w", err,
		)
	}
	return buildModelFromMILText(
		"appleneuralengine ffn forward model build",
		milText,
		files,
	)
}

func buildFFNBackwardModel(
	key ffnBackwardModelKey,
	w1, w3, w2 []float32,
) (appleneuralengine.ANEInMemoryModel, error) {
	blobs, err := buildFFNBackwardWeightBlobs(w1, w3, w2, key.dim, key.hiddenDim)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"appleneuralengine ffn backward model build: create weight blobs: %w", err,
		)
	}
	files := []modelWeightFile{
		{Path: ffnW2TBlobPathInMIL, Blob: blobs.W2T},
		{Path: ffnW1TBlobPathInMIL, Blob: blobs.W1T},
		{Path: ffnW3TBlobPathInMIL, Blob: blobs.W3T},
	}
	milText, err := ffnBwdMILText(key.dim, key.hiddenDim, key.seq)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"appleneuralengine ffn backward model build: generate MIL text: %w", err,
		)
	}
	return buildModelFromMILText(
		"appleneuralengine ffn backward model build",
		milText,
		files,
	)
}

func buildModelFromMILText(
	label string,
	milText string,
	files []modelWeightFile,
) (appleneuralengine.ANEInMemoryModel, error) {
	weights := newWeightDictionary(files)
	descObj, usedMILInit, err := newMILTextDescriptor(milText, weights, objectivec.Object{})
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: %w", label, err)
	}
	modelDir, err := mirrorDescriptorFilesMulti(descObj, milText, files, label)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	defer os.RemoveAll(modelDir)
	modelObj := appleneuralengine.GetANEInMemoryModelClass().InMemoryModelWithDescriptor(descObj)
	if modelObj.GetID() == 0 {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: create in-memory model failed", label)
	}
	model := appleneuralengine.ANEInMemoryModelFromID(modelObj.GetID())
	if mask := benchmarkPerfStatsMask(); mask != 0 {
		model.SetPerfStatsMask(mask)
	}
	options := foundation.NewMutableDictionaryWithCapacity(0)
	compileLabel := label + " compile"
	if usedMILInit {
		compileLabel = label + " compile (isMILModel init)"
	}
	if err := compileWithFallbackProfiles(model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: %w", compileLabel, err)
	}
	if _, err := loadInMemoryModelWithRetry(label, model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	if err := applyModelStateSyncFromEnv(label, model); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	return model, nil
}

func buildModelFromMILTextRequireMILInit(
	label string,
	milText string,
	files []modelWeightFile,
) (appleneuralengine.ANEInMemoryModel, error) {
	weights := newWeightDictionary(files)
	descObj, usedMILInit, err := newMILTextDescriptor(milText, weights, objectivec.Object{})
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: %w", label, err)
	}
	if !usedMILInit {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: isMILModel initializer unavailable", label)
	}
	modelDir, err := mirrorDescriptorFilesMulti(descObj, milText, files, label)
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	defer os.RemoveAll(modelDir)

	modelObj := appleneuralengine.GetANEInMemoryModelClass().InMemoryModelWithDescriptor(descObj)
	if modelObj.GetID() == 0 {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s: create in-memory model failed", label)
	}
	model := appleneuralengine.ANEInMemoryModelFromID(modelObj.GetID())
	options := foundation.NewMutableDictionaryWithCapacity(0)
	if err := compileWithFallbackProfiles(model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf("%s compile (isMILModel init): %w", label, err)
	}
	if _, err := loadInMemoryModelWithRetry(label, model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	if err := applyModelStateSyncFromEnv(label, model); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, err
	}
	return model, nil
}

func buildModelFromMILTextWithDescriptorFallback(
	label string,
	milText string,
	files []modelWeightFile,
) (appleneuralengine.ANEInMemoryModel, error) {
	model, err := buildModelFromMILTextRequireMILInit(label, milText, files)
	if err == nil {
		return model, nil
	}
	strictErr := err

	weights := newWeightDictionary(files)
	descObj, usedMILInit, descErr := newMILTextDescriptorWithMode(
		milText,
		weights,
		objectivec.Object{},
		milDescriptorHelperOnly,
	)
	if descErr != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"%w (descriptor helper fallback: %v)",
			strictErr,
			descErr,
		)
	}
	if usedMILInit {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"%w (descriptor helper fallback unexpectedly used explicit MIL init)",
			strictErr,
		)
	}
	modelDir, err := mirrorDescriptorFilesMulti(descObj, milText, files, label+" helper descriptor fallback")
	if err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"%w (descriptor helper fallback: %v)",
			strictErr,
			err,
		)
	}
	defer os.RemoveAll(modelDir)

	modelObj := appleneuralengine.GetANEInMemoryModelClass().InMemoryModelWithDescriptor(descObj)
	if modelObj.GetID() == 0 {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"%w (descriptor helper fallback: create in-memory model failed)",
			strictErr,
		)
	}
	model = appleneuralengine.ANEInMemoryModelFromID(modelObj.GetID())
	options := foundation.NewMutableDictionaryWithCapacity(0)
	if err := compileWithFallbackProfiles(model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"%w (descriptor helper fallback compile: %v)",
			strictErr,
			err,
		)
	}
	if _, err := loadInMemoryModelWithRetry(label+" helper descriptor fallback", model, options); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"%w (descriptor helper fallback load: %v)",
			strictErr,
			err,
		)
	}
	if err := applyModelStateSyncFromEnv(label+" helper descriptor fallback", model); err != nil {
		return appleneuralengine.ANEInMemoryModel{}, fmt.Errorf(
			"%w (descriptor helper fallback sync: %v)",
			strictErr,
			err,
		)
	}
	return model, nil
}

func newWeightDictionary(files []modelWeightFile) foundation.NSMutableDictionary {
	weights := foundation.NewMutableDictionaryWithCapacity(uint(len(files)))
	for _, f := range files {
		info := foundation.NewMutableDictionaryWithCapacity(2)
		info.SetObjectForKey(
			foundation.NewNumberWithInt(0),
			foundation.NewStringWithString("offset"),
		)
		info.SetObjectForKey(
			foundation.NewDataWithBytesLength(f.Blob),
			foundation.NewStringWithString("data"),
		)
		weights.SetObjectForKey(
			info,
			foundation.NewStringWithString(f.Path),
		)
	}
	return weights
}

func mirrorDescriptorFilesMulti(
	descObj objectivec.IObject,
	milText string,
	files []modelWeightFile,
	label string,
) (string, error) {
	desc := appleneuralengine.ANEInMemoryModelDescriptorFromID(descObj.GetID())
	hexIDObj := desc.HexStringIdentifier()
	if hexIDObj.GetID() == 0 {
		return "", fmt.Errorf("%s: descriptor hex string identifier is nil", label)
	}
	hexID := foundation.NSStringFromID(hexIDObj.GetID()).String()
	if hexID == "" {
		return "", fmt.Errorf("%s: descriptor hex string identifier is empty", label)
	}

	root := descriptorMirrorBaseDir()
	if strings.TrimSpace(root) == "" {
		return "", fmt.Errorf("%s: descriptor mirror root is empty", label)
	}
	modelDir := filepath.Join(root, hexID)
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return "", fmt.Errorf("%s: create mirrored temp dir: %w", label, err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.mil"), []byte(milText), 0o644); err != nil {
		return "", fmt.Errorf("%s: write mirrored model.mil: %w", label, err)
	}

	for _, f := range files {
		relPath, err := relativeModelPath(f.Path)
		if err != nil {
			return "", fmt.Errorf("%s: invalid weight path %q: %w", label, f.Path, err)
		}
		dstPath := filepath.Join(modelDir, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
			return "", fmt.Errorf("%s: create mirrored weight dir: %w", label, err)
		}
		if err := os.WriteFile(dstPath, f.Blob, 0o644); err != nil {
			return "", fmt.Errorf("%s: write mirrored weight file %q: %w", label, relPath, err)
		}
	}
	return modelDir, nil
}

func relativeModelPath(path string) (string, error) {
	if !strings.HasPrefix(path, modelPathPrefix) {
		return "", fmt.Errorf("missing %q prefix", modelPathPrefix)
	}
	rel := strings.TrimPrefix(path, modelPathPrefix)
	if rel == "" {
		return "", fmt.Errorf("empty relative path")
	}
	return rel, nil
}

func evalModelSingleIO(
	ctx context.Context,
	model appleneuralengine.ANEInMemoryModel,
	input []float32,
	outputCount int,
	label string,
) ([]float32, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if outputCount <= 0 {
		return nil, fmt.Errorf("%s: invalid outputCount=%d", label, outputCount)
	}

	xSurf, err := newFloatSurface(len(input))
	if err != nil {
		return nil, err
	}
	defer releaseIOSurface(xSurf)
	if err := writeFloat32IOSurface(xSurf, input); err != nil {
		return nil, err
	}

	ySurf, err := newFloatSurface(outputCount)
	if err != nil {
		return nil, err
	}
	defer releaseIOSurface(ySurf)

	xObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(xSurf)
	yObj := appleneuralengine.GetANEIOSurfaceObjectClass().ObjectWithIOSurface(ySurf)
	if xObj.GetID() == 0 || yObj.GetID() == 0 {
		return nil, fmt.Errorf("%s: create IOSurface object failed", label)
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
		return nil, fmt.Errorf("%s: create request failed", label)
	}
	request := appleneuralengine.ANERequestFromID(requestObj.GetID())
	if !request.Validate() {
		return nil, fmt.Errorf("%s: request validation failed", label)
	}

	unmap, err := mapModelRequestWithFallback(
		label+" map IOSurfaces",
		model,
		requestObj,
		true,
	)
	if err != nil {
		return nil, err
	}
	defer unmap()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	options := foundation.NewMutableDictionaryWithCapacity(0)
	if err := callModelEvaluateWithFallback(
		label+" evaluate",
		model,
		defaultANEQoS,
		options,
		requestObj,
	); err != nil {
		return nil, err
	}
	return readFloat32IOSurface(ySurf, outputCount)
}
