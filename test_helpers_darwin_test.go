//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
	"github.com/tmc/mlx-go/modelir"
)

const ffnMILHeader = `program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
`

func testObjectDescription(o interface{ GetID() objc.ID }) string {
	if o.GetID() == 0 {
		return "<nil>"
	}
	descID := objc.Send[objc.ID](o.GetID(), objc.Sel("description"))
	if descID == 0 {
		return "<no description>"
	}
	return foundation.NSStringFromID(descID).String()
}

func testTransformerConfig(layers int) MILTransformerConfig {
	return MILTransformerConfig{
		NumLayers: layers,
		Dim:       8,
		NumHeads:  2,
		HeadDim:   4,
		HiddenDim: 16,
		MaxSeqLen: 8,
	}
}

func testTransformerWeights(cfg MILTransformerConfig) MILTransformerWeights {
	attnDim := cfg.NumHeads * cfg.HeadDim
	layers := make([]MILTransformerLayerWeights, cfg.NumLayers)
	for i := range layers {
		layers[i] = MILTransformerLayerWeights{
			QW: make([]float32, attnDim*cfg.Dim),
			QB: make([]float32, attnDim),
			KW: make([]float32, attnDim*cfg.Dim),
			KB: make([]float32, attnDim),
			VW: make([]float32, attnDim*cfg.Dim),
			VB: make([]float32, attnDim),
			OW: make([]float32, cfg.Dim*attnDim),
			OB: make([]float32, cfg.Dim),
			W1: make([]float32, cfg.HiddenDim*cfg.Dim),
			B1: make([]float32, cfg.HiddenDim),
			W3: make([]float32, cfg.HiddenDim*cfg.Dim),
			B3: make([]float32, cfg.HiddenDim),
			W2: make([]float32, cfg.Dim*cfg.HiddenDim),
			B2: make([]float32, cfg.Dim),

			InputNorm:         fillOnes(cfg.Dim),
			PostAttentionNorm: fillOnes(cfg.Dim),
			QNorm:             fillOnes(cfg.HeadDim),
			KNorm:             fillOnes(cfg.HeadDim),
		}
	}
	return MILTransformerWeights{
		Layers:    layers,
		FinalNorm: fillOnes(cfg.Dim),
		RopeCos:   fillOnes(cfg.MaxSeqLen * cfg.HeadDim),
		RopeSin:   make([]float32, cfg.MaxSeqLen*cfg.HeadDim),
	}
}

func testTransformerWeightsIdentity(cfg MILTransformerConfig) MILTransformerWeights {
	w := testTransformerWeights(cfg)
	attnDim := cfg.NumHeads * cfg.HeadDim
	for i := range w.Layers {
		layer := &w.Layers[i]
		fillLinearIdentity(layer.QW, attnDim, cfg.Dim, 1)
		fillLinearIdentity(layer.KW, attnDim, cfg.Dim, 1)
		fillLinearIdentity(layer.VW, attnDim, cfg.Dim, 1)
		fillLinearIdentity(layer.OW, cfg.Dim, attnDim, 1)
	}
	return w
}

func fillLinearIdentity(dst []float32, rows, cols int, scale float32) {
	clear(dst)
	n := rows
	if cols < n {
		n = cols
	}
	for i := 0; i < n; i++ {
		dst[i*cols+i] = scale
	}
}

func fillOnes(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func testModelIRTransformerWeights(cfg MILTransformerConfig, weights MILTransformerWeights) map[string]*modelir.Weight {
	out := map[string]*modelir.Weight{}
	add := func(path string, shape modelir.Shape, vals []float32) {
		out[path] = &modelir.Weight{
			DType: modelir.DTypeFP32,
			Shape: shape,
			Data:  float32sToBytes(vals),
		}
	}
	attnDim := cfg.NumHeads * cfg.HeadDim
	for i, layer := range weights.Layers {
		prefix := fmt.Sprintf("@model_path/weights/l%d_", i)
		add(prefix+"q_w.bin", modelir.Shape{int64(attnDim), int64(cfg.Dim)}, layer.QW)
		add(prefix+"q_b.bin", modelir.Shape{int64(attnDim)}, layer.QB)
		add(prefix+"k_w.bin", modelir.Shape{int64(attnDim), int64(cfg.Dim)}, layer.KW)
		add(prefix+"k_b.bin", modelir.Shape{int64(attnDim)}, layer.KB)
		add(prefix+"v_w.bin", modelir.Shape{int64(attnDim), int64(cfg.Dim)}, layer.VW)
		add(prefix+"v_b.bin", modelir.Shape{int64(attnDim)}, layer.VB)
		add(prefix+"o_w.bin", modelir.Shape{int64(cfg.Dim), int64(attnDim)}, layer.OW)
		add(prefix+"o_b.bin", modelir.Shape{int64(cfg.Dim)}, layer.OB)
		add(prefix+"w1.bin", modelir.Shape{int64(cfg.HiddenDim), int64(cfg.Dim)}, layer.W1)
		add(prefix+"b1.bin", modelir.Shape{int64(cfg.HiddenDim)}, layer.B1)
		add(prefix+"w3.bin", modelir.Shape{int64(cfg.HiddenDim), int64(cfg.Dim)}, layer.W3)
		add(prefix+"b3.bin", modelir.Shape{int64(cfg.HiddenDim)}, layer.B3)
		add(prefix+"w2.bin", modelir.Shape{int64(cfg.Dim), int64(cfg.HiddenDim)}, layer.W2)
		add(prefix+"b2.bin", modelir.Shape{int64(cfg.Dim)}, layer.B2)
		add(prefix+"input_norm.bin", modelir.Shape{int64(cfg.Dim)}, layer.InputNorm)
		add(prefix+"post_norm.bin", modelir.Shape{int64(cfg.Dim)}, layer.PostAttentionNorm)
		add(prefix+"q_norm.bin", modelir.Shape{int64(cfg.HeadDim)}, layer.QNorm)
		add(prefix+"k_norm.bin", modelir.Shape{int64(cfg.HeadDim)}, layer.KNorm)
	}
	add("@model_path/weights/final_norm.bin", modelir.Shape{int64(cfg.Dim)}, weights.FinalNorm)
	add("@model_path/weights/rope_cos.bin", modelir.Shape{int64(cfg.MaxSeqLen), int64(cfg.HeadDim)}, weights.RopeCos)
	add("@model_path/weights/rope_sin.bin", modelir.Shape{int64(cfg.MaxSeqLen), int64(cfg.HeadDim)}, weights.RopeSin)
	return out
}

func float32sToBytes(vals []float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(v))
	}
	return out
}

type compiledModelIRRuntimeTarget struct {
	clientModel   *ANEClientMILModel
	inMemoryModel appleneuralengine.ANEInMemoryModel
}

func (m compiledModelIRRuntimeTarget) Close() {
	switch {
	case m.clientModel != nil:
		m.clientModel.Close()
	case m.inMemoryModel.ID != 0:
		_ = callObjCBoolWithNSError(
			"modelir runtime target unload",
			m.inMemoryModel.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
	}
}

func compileModelIRRuntimeTarget(
	t *testing.T,
	textData []byte,
	opts ReifyOptions,
) (compiledModelIRRuntimeTarget, ReifiedMIL) {
	t.Helper()

	clientObj := appleneuralengine.GetANEClientClass().SharedConnection()
	if clientObj.GetID() == 0 {
		t.Fatal("_ANEClient sharedConnection returned nil")
	}
	client := appleneuralengine.ANEClientFromID(clientObj.GetID())

	model, reified, err := CompileAndLoadModelIRText(client, textData, opts, defaultMILFastPathKey, defaultANEQoS)
	if err == nil {
		return compiledModelIRRuntimeTarget{clientModel: model}, reified
	}
	if !errors.Is(err, ErrMILDirectoryUnsupported) {
		t.Fatalf("CompileAndLoadModelIRText: %v", err)
	}

	inMemoryModel, inMemoryReified, inMemoryErr := CompileModelIRTextInMemory(textData, opts)
	if inMemoryErr != nil {
		if strings.Contains(inMemoryErr.Error(), "ANECCompile() FAILED") {
			t.Skipf("in-memory MIL compile unavailable on this host: %v", inMemoryErr)
		}
		t.Fatalf("CompileModelIRTextInMemory fallback: %v", inMemoryErr)
	}
	return compiledModelIRRuntimeTarget{inMemoryModel: inMemoryModel}, inMemoryReified
}

func evalModelIRRuntimeSteps(
	ctx context.Context,
	target compiledModelIRRuntimeTarget,
	inputs [][]float32,
	outputCount int,
) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("eval modelir runtime steps: inputs are empty")
	}
	if outputCount <= 0 {
		return nil, fmt.Errorf("eval modelir runtime steps: invalid outputCount=%d", outputCount)
	}

	var (
		base objectivec.IObject
	)
	switch {
	case target.clientModel != nil:
		base = target.clientModel.model
	case target.inMemoryModel.ID != 0:
		base = target.inMemoryModel.Model()
	default:
		return nil, fmt.Errorf("eval modelir runtime steps: target model is nil")
	}
	surfs, err := newRuntimeSurfaces(base, len(inputs[0]), outputCount)
	if err != nil {
		return nil, fmt.Errorf("eval modelir runtime steps: allocate IO surfaces: %w", err)
	}
	defer surfs.Close()

	inputBindings := []SurfaceBinding{{Surface: surfs.Input, SymbolIndex: surfs.InputSymbolIndex}}
	inputBindings = append(inputBindings, surfs.StateBindings...)
	outputBindings := []SurfaceBinding{{Surface: surfs.Output, SymbolIndex: surfs.OutputSymbolIndex}}
	cfg := DefaultMultiSurfaceEvalPlanConfig()
	var plan *MultiSurfaceEvalPlan
	switch {
	case target.clientModel != nil:
		plan, err = NewMultiSurfaceEvalPlanWithClientModel(target.clientModel, inputBindings, outputBindings, cfg)
	case target.inMemoryModel.ID != 0:
		plan, err = NewMultiSurfaceEvalPlan(target.inMemoryModel, inputBindings, outputBindings, cfg)
	}
	if err != nil {
		return nil, fmt.Errorf("eval modelir runtime steps: new multi-surface plan: %w", err)
	}
	defer plan.Close()

	outs := make([][]float32, 0, len(inputs))
	inputCount := len(inputs[0])
	for i, input := range inputs {
		if len(input) != inputCount {
			return nil, fmt.Errorf("eval modelir runtime steps: input[%d] len=%d want=%d", i, len(input), inputCount)
		}
		if err := surfs.Input.Write(input); err != nil {
			return nil, fmt.Errorf("eval modelir runtime steps: write input[%d]: %w", i, err)
		}
		if err := plan.Eval(ctx); err != nil {
			return nil, fmt.Errorf("eval modelir runtime steps: eval[%d]: %w", i, err)
		}
		out, err := surfs.Output.Read()
		if err != nil {
			return nil, fmt.Errorf("eval modelir runtime steps: read output[%d]: %w", i, err)
		}
		outs = append(outs, out)
	}
	return outs, nil
}

type runtimeSurfaces struct {
	Input             *IOSurfaceFloat32
	Output            *IOSurfaceFloat32
	InputSymbolIndex  int
	OutputSymbolIndex int
	StateSurfaces     []*IOSurfaceFloat32
	StateBindings     []SurfaceBinding
}

func (r *runtimeSurfaces) Close() {
	if r == nil {
		return
	}
	if r.Output != nil {
		r.Output.Close()
	}
	for _, state := range r.StateSurfaces {
		if state != nil {
			state.Close()
		}
	}
	if r.Input != nil {
		r.Input.Close()
	}
}

func newRuntimeSurfaces(
	base objectivec.IObject,
	inputCount int,
	outputCount int,
) (*runtimeSurfaces, error) {
	if base == nil || base.GetID() == 0 {
		return nil, fmt.Errorf("compiled model is nil")
	}
	schema, err := parseCompiledModelSchema(base)
	if err == nil && len(schema.Inputs) > 0 && len(schema.Outputs) > 0 {
		inSurf, inErr := newIOSurfaceFloat32WithLayout(schema.Inputs[0])
		if inErr != nil {
			return nil, fmt.Errorf("input layout surface: %w", inErr)
		}
		outSurf, outErr := newIOSurfaceFloat32WithLayout(schema.Outputs[0])
		if outErr != nil {
			inSurf.Close()
			return nil, fmt.Errorf("output layout surface: %w", outErr)
		}
		stateSurfs := make([]*IOSurfaceFloat32, 0, len(schema.States))
		stateBindings := make([]SurfaceBinding, 0, len(schema.States))
		for _, layout := range schema.States {
			stateSurf, stateErr := newIOSurfaceFloat32WithLayout(layout)
			if stateErr != nil {
				outSurf.Close()
				inSurf.Close()
				for _, state := range stateSurfs {
					state.Close()
				}
				return nil, fmt.Errorf("state layout surface %q: %w", layout.Name, stateErr)
			}
			if stateErr := stateSurf.Write(make([]float32, stateSurf.Count())); stateErr != nil {
				stateSurf.Close()
				outSurf.Close()
				inSurf.Close()
				for _, state := range stateSurfs {
					state.Close()
				}
				return nil, fmt.Errorf("init state layout surface %q: %w", layout.Name, stateErr)
			}
			stateSurfs = append(stateSurfs, stateSurf)
			stateBindings = append(stateBindings, SurfaceBinding{Surface: stateSurf, SymbolIndex: layout.SymbolIndex})
		}
		return &runtimeSurfaces{
			Input:             inSurf,
			Output:            outSurf,
			InputSymbolIndex:  compiledLayoutSymbolIndex(schema.Inputs, 0),
			OutputSymbolIndex: compiledLayoutSymbolIndex(schema.Outputs, 0),
			StateSurfaces:     stateSurfs,
			StateBindings:     stateBindings,
		}, nil
	}
	inSurf, err := NewIOSurfaceFloat32(inputCount)
	if err != nil {
		return nil, err
	}
	outSurf, err := NewIOSurfaceFloat32(outputCount)
	if err != nil {
		inSurf.Close()
		return nil, err
	}
	return &runtimeSurfaces{
		Input:             inSurf,
		Output:            outSurf,
		InputSymbolIndex:  0,
		OutputSymbolIndex: 0,
	}, nil
}

func nearlyEqualFloat32s(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			return false
		}
	}
	return true
}

func envInt(key string, dflt int) int {
	v := os.Getenv(key)
	if v == "" {
		return dflt
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return dflt
	}
	return n
}

func envIntWithDefault(t *testing.T, key string, dflt int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return dflt
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("parse %s=%q: %v", key, v, err)
	}
	if n <= 0 {
		t.Fatalf("%s must be > 0 (got %d)", key, n)
	}
	return n
}

func makeDeterministicTensor(n int, scale float32, period int) []float32 {
	if n < 0 {
		return nil
	}
	if period <= 0 {
		period = 1
	}
	mid := period / 2
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%period)-mid) * scale
	}
	return out
}
