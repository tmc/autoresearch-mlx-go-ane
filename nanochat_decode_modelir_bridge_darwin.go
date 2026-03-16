//go:build ignore

package mlxgoane

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/mlx-go/internal/modelir"
	"github.com/tmc/mlx-go/internal/modelir/target/coremltools"
)

const (
	nanochatCoreMLToolsBridgeDir    = "coremltools"
	nanochatCoreMLToolsBridgeScript = "nanochat_decode.py"
)

const bridgeModelIRDType = modelir.DTypeFP32

var nanochatCoreMLToolsBridgeOptions = coremltools.ArtifactBuilderOptions{
	ConvertTo:               "neuralnetwork",
	MinimumDeploymentTarget: "ct.target.iOS14",
	OpsetVersion:            "ct.target.iOS14",
}

var errNanochatCoreMLToolsBridgeUnsupported = errors.New("nanochat coremltools bridge: unsupported decode config")

// BuildNanochatDecodeCoreMLToolsScript returns a Core ML Tools builder script
// for the compile-safe nanochat decode subset.
func BuildNanochatDecodeCoreMLToolsScript(
	cfg NanochatDecodeConfig,
	w NanochatDecodeWeights,
) (string, error) {
	prog, err := buildNanochatDecodeModelIR(cfg, w)
	if err != nil {
		return "", err
	}
	src, err := coremltools.EmitArtifactBuilderWithOptions(prog, nanochatCoreMLToolsBridgeOptions)
	if err != nil {
		return "", fmt.Errorf("build nanochat coremltools script: %w", err)
	}
	return src, nil
}

// WriteNanochatDecodeCoreMLToolsBridge writes a builder script plus external
// raw weight files under dir/coremltools.
func WriteNanochatDecodeCoreMLToolsBridge(
	dir string,
	cfg NanochatDecodeConfig,
	w NanochatDecodeWeights,
) error {
	if strings.TrimSpace(dir) == "" {
		return fmt.Errorf("write nanochat coremltools bridge: dir is empty")
	}
	prog, err := buildNanochatDecodeModelIR(cfg, w)
	if err != nil {
		return err
	}
	src, err := coremltools.EmitArtifactBuilderWithOptions(prog, nanochatCoreMLToolsBridgeOptions)
	if err != nil {
		return fmt.Errorf("write nanochat coremltools bridge: emit script: %w", err)
	}

	bridgeDir := filepath.Join(dir, nanochatCoreMLToolsBridgeDir)
	if err := os.MkdirAll(bridgeDir, 0o755); err != nil {
		return fmt.Errorf("write nanochat coremltools bridge: create bridge dir: %w", err)
	}
	scriptPath := filepath.Join(bridgeDir, nanochatCoreMLToolsBridgeScript)
	if err := os.WriteFile(scriptPath, []byte(src), 0o644); err != nil {
		return fmt.Errorf("write nanochat coremltools bridge: write script: %w", err)
	}
	for ref, weight := range prog.Weights {
		if weight == nil {
			continue
		}
		dst := filepath.Join(bridgeDir, filepath.FromSlash(ref))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return fmt.Errorf("write nanochat coremltools bridge: create weight dir for %q: %w", ref, err)
		}
		if err := os.WriteFile(dst, weight.Data, 0o644); err != nil {
			return fmt.Errorf("write nanochat coremltools bridge: write weight %q: %w", ref, err)
		}
	}
	return nil
}

func buildNanochatDecodeModelIR(
	cfg NanochatDecodeConfig,
	w NanochatDecodeWeights,
) (*modelir.Program, error) {
	cfg = normalizeNanochatDecodeConfig(cfg)
	if err := validateNanochatDecodeConfig(cfg); err != nil {
		return nil, fmt.Errorf("lower nanochat decode: %w", err)
	}
	if err := validateNanochatDecodeWeights(cfg, w); err != nil {
		return nil, fmt.Errorf("lower nanochat decode: %w", err)
	}
	if err := validateNanochatArtifactSupportedConfig(cfg); err != nil {
		return nil, fmt.Errorf("%w: %v", errNanochatCoreMLToolsBridgeUnsupported, err)
	}
	dim := int64(cfg.Dim)
	headCount := int64(cfg.NumHeads)
	headDim := int64(cfg.HeadDim)
	ffnDim := int64(cfg.HiddenDim)
	seqLen := int64(cfg.MaxSeqLen)

	inputs := []modelir.Value{
		modelirTensor("x", modelir.DTypeFP16, 1, 1, dim),
		modelirTensor("x0", modelir.DTypeFP16, 1, 1, dim),
		modelirTensor("pos_cos", modelir.DTypeFP16, 1, 1, dim),
		modelirTensor("pos_sin", modelir.DTypeFP16, 1, 1, dim),
	}
	consts := []modelir.Const{
		modelirConst("rope_rotate_w", modelir.DTypeFP16, modelir.Shape{dim, dim}, "weights/global/rope_rotate_w.bin"),
		modelirConst("rope_rotate_b", modelir.DTypeFP16, modelir.Shape{dim}, "weights/global/rope_rotate_b.bin"),
		modelirConst("ve_scale_two", modelir.DTypeFP16, modelir.Shape{1, 1, 1}, "weights/global/ve_scale_two.bin"),
		modelirConst("attn_scale", modelir.DTypeFP16, modelir.Shape{1, 1, 1, 1}, "weights/global/attn_scale.bin"),
	}
	if !cfg.IncludeFinalNorm {
		consts = append(consts, modelirConst("zero", modelir.DTypeFP16, modelir.Shape{1, 1, 1}, "weights/global/zero.bin"))
	}
	ops := make([]modelir.Op, 0, cfg.NumLayers*32)
	returns := []string{"out"}
	current := "x"

	prog := &modelir.Program{
		Version: "1",
		Target:  "coreml-ane-v1",
		Entry:   "decode",
		Functions: []modelir.Function{{
			Name:   "decode",
			Inputs: inputs,
			Consts: consts,
		}},
		HighLevel: map[string]string{
			"decode": "nanochat decode trunk with external KV cache and optional VE inputs",
		},
		Weights: map[string]*modelir.Weight{},
	}

	addWeight := func(ref string, shape modelir.Shape, vals []float32) {
		prog.Weights[ref] = &modelir.Weight{
			DType: bridgeModelIRDType,
			Shape: shape,
			Data:  rawFP32Bytes(vals),
		}
	}
	addConstWeight := func(name string, shape modelir.Shape, ref string, vals []float32) {
		prog.Functions[0].Consts = append(prog.Functions[0].Consts, modelirConst(name, modelir.DTypeFP16, shape, ref))
		addWeight(ref, shape, vals)
	}
	addWeight("weights/global/rope_rotate_w.bin", modelir.Shape{dim, dim}, buildRoPERotateMatrix(cfg.Dim, cfg.HeadDim))
	addWeight("weights/global/rope_rotate_b.bin", modelir.Shape{dim}, make([]float32, cfg.Dim))
	addWeight("weights/global/ve_scale_two.bin", modelir.Shape{1, 1, 1}, []float32{2})
	addWeight("weights/global/attn_scale.bin", modelir.Shape{1, 1, 1, 1}, []float32{float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))})
	if !cfg.IncludeFinalNorm {
		addWeight("weights/global/zero.bin", modelir.Shape{1, 1, 1}, []float32{0})
	}

	for i, layer := range w.Layers {
		pfx := fmt.Sprintf("l%03d_", i)
		layerRef := fmt.Sprintf("weights/layer_%03d", i)
		kCache := fmt.Sprintf("k_cache_%d", i)
		vCache := fmt.Sprintf("v_cache_%d", i)
		inputs = append(inputs,
			modelirTensor(kCache, modelir.DTypeFP16, 1, headCount, seqLen, headDim),
			modelirTensor(vCache, modelir.DTypeFP16, 1, headCount, seqLen, headDim),
		)
		if nanochatDecodeLayerUsesMask(cfg, i) {
			inputs = append(inputs, modelirTensor(fmt.Sprintf("mask_%d", i), modelir.DTypeFP16, 1, headCount, 1, seqLen+1))
		}
		if cfg.UseValueEmbeds[i] {
			inputs = append(inputs[:len(inputs)-2], append([]modelir.Value{
				modelirTensor(fmt.Sprintf("ve_%d_in", i), modelir.DTypeFP16, 1, 1, dim),
			}, inputs[len(inputs)-2:]...)...)
		}

		addConstWeight(pfx+"resid_lambda", modelir.Shape{1, 1, 1}, layerRef+"/resid_lambda.bin", []float32{layer.ResidLambda})
		addConstWeight(pfx+"x0_lambda", modelir.Shape{1, 1, 1}, layerRef+"/x0_lambda.bin", []float32{layer.X0Lambda})
		addConstWeight(pfx+"q_w", modelir.Shape{dim, dim}, layerRef+"/q_w.bin", layer.QW)
		addConstWeight(pfx+"q_b", modelir.Shape{dim}, layerRef+"/q_b.bin", layer.QB)
		addConstWeight(pfx+"k_w", modelir.Shape{dim, dim}, layerRef+"/k_w.bin", layer.KW)
		addConstWeight(pfx+"k_b", modelir.Shape{dim}, layerRef+"/k_b.bin", layer.KB)
		addConstWeight(pfx+"v_w", modelir.Shape{dim, dim}, layerRef+"/v_w.bin", layer.VW)
		addConstWeight(pfx+"v_b", modelir.Shape{dim}, layerRef+"/v_b.bin", layer.VB)
		addConstWeight(pfx+"o_w", modelir.Shape{dim, dim}, layerRef+"/o_w.bin", layer.OW)
		addConstWeight(pfx+"o_b", modelir.Shape{dim}, layerRef+"/o_b.bin", layer.OB)
		addConstWeight(pfx+"fc_w", modelir.Shape{ffnDim, dim}, layerRef+"/fc_w.bin", layer.FCW)
		addConstWeight(pfx+"fc_b", modelir.Shape{ffnDim}, layerRef+"/fc_b.bin", layer.FCB)
		addConstWeight(pfx+"proj_w", modelir.Shape{dim, ffnDim}, layerRef+"/proj_w.bin", layer.PW)
		addConstWeight(pfx+"proj_b", modelir.Shape{dim}, layerRef+"/proj_b.bin", layer.PB)
		if cfg.UseValueEmbeds[i] {
			addConstWeight(pfx+"ve_gate_w", modelir.Shape{int64(cfg.NumHeads), int64(cfg.ValueGateWidth)}, layerRef+"/ve_gate_w.bin", layer.VEGateW)
			addConstWeight(pfx+"ve_gate_b", modelir.Shape{headCount}, layerRef+"/ve_gate_b.bin", layer.VEGateB)
		}

		hMixedCur := pfx + "mixed_cur"
		hMixedX0 := pfx + "mixed_x0"
		hMixed := pfx + "mixed"
		h := pfx + "h"
		q := pfx + "q"
		k := pfx + "k"
		v := pfx + "v"
		qRot := pfx + "q_rot"
		qRotBias := pfx + "q_rot_bias"
		kRot := pfx + "k_rot"
		kRotBias := pfx + "k_rot_bias"
		qCos := pfx + "q_cos"
		qSin := pfx + "q_sin"
		qRope := pfx + "q_rope"
		kCos := pfx + "k_cos"
		kSin := pfx + "k_sin"
		kRope := pfx + "k_rope"
		qHeads := pfx + "q_heads"
		kCurr := pfx + "k_curr"
		vCurr := pfx + "v_curr"
		qNorm := pfx + "q_norm"
		kNorm := pfx + "k_norm"
		kAll := pfx + "k_all"
		vAll := pfx + "v_all"
		kAllT := pfx + "k_all_t"
		attnScores := pfx + "attn_scores"
		attnScoresScaled := pfx + "attn_scores_scaled"
		attnScoresMasked := pfx + "attn_scores_masked"
		attnWeights := pfx + "attn_weights"
		attn := pfx + "attn"
		attnFlat := pfx + "attn_flat"
		attnOut := pfx + "attn_out"
		afterAttn := pfx + "after_attn"
		h2 := pfx + "h2"
		fc := pfx + "fc"
		reluSqIn := pfx + "relu_sq_in"
		mlpAct := pfx + "mlp_act"
		mlpOut := pfx + "mlp_out"
		layerOut := pfx + "current"

		reshapeHeads := fmt.Sprintf("shape=[1, %d, 1, %d]", cfg.NumHeads, cfg.HeadDim)
		reshapeFlat := fmt.Sprintf("shape=[1, 1, %d]", cfg.Dim)
		ops = append(ops,
			modelirOp("mul", modelirOut(hMixedCur, modelir.DTypeFP16, 1, 1, dim), current, pfx+"resid_lambda"),
			modelirOp("mul", modelirOut(hMixedX0, modelir.DTypeFP16, 1, 1, dim), "x0", pfx+"x0_lambda"),
			modelirOp("add", modelirOut(hMixed, modelir.DTypeFP16, 1, 1, dim), hMixedCur, hMixedX0),
			modelirOpAttr("rmsnorm", "eps=1e-06", modelirOut(h, modelir.DTypeFP16, 1, 1, dim), hMixed),
		)
		ops = append(ops, modelirLinearOpsNamed(q, h, pfx+"q", dim)...)
		ops = append(ops, modelirLinearOpsNamed(k, h, pfx+"k", dim)...)
		ops = append(ops, modelirLinearOpsNamed(v, h, pfx+"v", dim)...)
		ops = append(ops,
			modelirOp("matmul", modelirOut(qRot, modelir.DTypeFP16, 1, 1, dim), q, "rope_rotate_w"),
			modelirOp("add", modelirOut(qRotBias, modelir.DTypeFP16, 1, 1, dim), qRot, "rope_rotate_b"),
			modelirOp("matmul", modelirOut(kRot, modelir.DTypeFP16, 1, 1, dim), k, "rope_rotate_w"),
			modelirOp("add", modelirOut(kRotBias, modelir.DTypeFP16, 1, 1, dim), kRot, "rope_rotate_b"),
			modelirOp("mul", modelirOut(qCos, modelir.DTypeFP16, 1, 1, dim), q, "pos_cos"),
			modelirOp("mul", modelirOut(qSin, modelir.DTypeFP16, 1, 1, dim), qRotBias, "pos_sin"),
			modelirOp("add", modelirOut(qRope, modelir.DTypeFP16, 1, 1, dim), qCos, qSin),
			modelirOp("mul", modelirOut(kCos, modelir.DTypeFP16, 1, 1, dim), k, "pos_cos"),
			modelirOp("mul", modelirOut(kSin, modelir.DTypeFP16, 1, 1, dim), kRotBias, "pos_sin"),
			modelirOp("add", modelirOut(kRope, modelir.DTypeFP16, 1, 1, dim), kCos, kSin),
			modelirOpAttr("reshape", reshapeHeads, modelirOut(qHeads, modelir.DTypeFP16, 1, headCount, 1, headDim), qRope),
			modelirOpAttr("reshape", reshapeHeads, modelirOut(kCurr, modelir.DTypeFP16, 1, headCount, 1, headDim), kRope),
			modelirOpAttr("reshape", reshapeHeads, modelirOut(vCurr, modelir.DTypeFP16, 1, headCount, 1, headDim), v),
			modelirOpAttr("rmsnorm", "eps=1e-06", modelirOut(qNorm, modelir.DTypeFP16, 1, headCount, 1, headDim), qHeads),
			modelirOpAttr("rmsnorm", "eps=1e-06", modelirOut(kNorm, modelir.DTypeFP16, 1, headCount, 1, headDim), kCurr),
		)
		vCacheInput := vCurr
		if cfg.UseValueEmbeds[i] {
			x32 := pfx + "x32"
			veGate := pfx + "ve_gate"
			veSig := pfx + "ve_sig"
			veGate2 := pfx + "ve_gate2"
			veGate4 := pfx + "ve_gate4"
			veCurr := pfx + "ve_curr"
			vWithVE := pfx + "v_with_ve"
			ops = append(ops,
				modelirOpAttr("slice_by_index", "begin=[0, 0, 0], end=[1, 1, 32]", modelirOut(x32, modelir.DTypeFP16, 1, 1, int64(cfg.ValueGateWidth)), h),
			)
			ops = append(ops, modelirLinearOpsNamed(veGate, x32, pfx+"ve_gate", int64(cfg.NumHeads))...)
			ops = append(ops,
				modelirOp("sigmoid", modelirOut(veSig, modelir.DTypeFP16, 1, 1, headCount), veGate),
				modelirOp("mul", modelirOut(veGate2, modelir.DTypeFP16, 1, 1, headCount), veSig, "ve_scale_two"),
				modelirOpAttr("reshape", fmt.Sprintf("shape=[1, %d, 1, 1]", cfg.NumHeads), modelirOut(veGate4, modelir.DTypeFP16, 1, headCount, 1, 1), veGate2),
				modelirOpAttr("reshape", reshapeHeads, modelirOut(veCurr, modelir.DTypeFP16, 1, headCount, 1, headDim), fmt.Sprintf("ve_%d_in", i)),
				modelirOp("mul", modelirOut(vWithVE+"_scaled", modelir.DTypeFP16, 1, headCount, 1, headDim), veGate4, veCurr),
				modelirOp("add", modelirOut(vWithVE, modelir.DTypeFP16, 1, headCount, 1, headDim), vCurr, vWithVE+"_scaled"),
			)
			vCacheInput = vWithVE
		}
		ops = append(ops,
			modelirOpAttr("concat", "axis=2", modelirOut(kAll, modelir.DTypeFP16, 1, headCount, seqLen+1, headDim), kCache, kNorm),
			modelirOpAttr("concat", "axis=2", modelirOut(vAll, modelir.DTypeFP16, 1, headCount, seqLen+1, headDim), vCache, vCacheInput),
			modelirOpAttr("transpose", "perm=[0,1,3,2]", modelirOut(kAllT, modelir.DTypeFP16, 1, headCount, headDim, seqLen+1), kAll),
			modelirOp("matmul", modelirOut(attnScores, modelir.DTypeFP16, 1, headCount, 1, seqLen+1), qNorm, kAllT),
			modelirOp("mul", modelirOut(attnScoresScaled, modelir.DTypeFP16, 1, headCount, 1, seqLen+1), attnScores, "attn_scale"),
		)
		attnSoftmaxIn := attnScoresScaled
		if nanochatDecodeLayerUsesMask(cfg, i) {
			ops = append(ops, modelirOp("add", modelirOut(attnScoresMasked, modelir.DTypeFP16, 1, headCount, 1, seqLen+1), attnScoresScaled, fmt.Sprintf("mask_%d", i)))
			attnSoftmaxIn = attnScoresMasked
		}
		ops = append(ops,
			modelirOpAttr("softmax", "axis=-1", modelirOut(attnWeights, modelir.DTypeFP16, 1, headCount, 1, seqLen+1), attnSoftmaxIn),
			modelirOp("matmul", modelirOut(attn, modelir.DTypeFP16, 1, headCount, 1, headDim), attnWeights, vAll),
			modelirOpAttr("reshape", reshapeFlat, modelirOut(attnFlat, modelir.DTypeFP16, 1, 1, dim), attn),
		)
		ops = append(ops, modelirLinearOpsNamed(attnOut, attnFlat, pfx+"o", dim)...)
		ops = append(ops,
			modelirOp("add", modelirOut(afterAttn, modelir.DTypeFP16, 1, 1, dim), hMixed, attnOut),
			modelirOpAttr("rmsnorm", "eps=1e-06", modelirOut(h2, modelir.DTypeFP16, 1, 1, dim), afterAttn),
		)
		ops = append(ops, modelirLinearOpsNamed(fc, h2, pfx+"fc", ffnDim)...)
		ops = append(ops,
			modelirOp("relu", modelirOut(reluSqIn, modelir.DTypeFP16, 1, 1, ffnDim), fc),
			modelirOp("mul", modelirOut(mlpAct, modelir.DTypeFP16, 1, 1, ffnDim), reluSqIn, reluSqIn),
		)
		ops = append(ops, modelirLinearOpsNamed(mlpOut, mlpAct, pfx+"proj", dim)...)
		ops = append(ops,
			modelirOp("add", modelirOut(layerOut, modelir.DTypeFP16, 1, 1, dim), afterAttn, mlpOut),
		)
		current = layerOut
		returns = append(returns, kNorm, vCacheInput)
	}

	if cfg.IncludeFinalNorm {
		ops = append(ops, modelirOpAttr("rmsnorm", "eps=1e-06", modelirOut("out", modelir.DTypeFP16, 1, 1, dim), current))
	} else {
		ops = append(ops, modelirOp("add", modelirOut("out", modelir.DTypeFP16, 1, 1, dim), current, "zero"))
	}
	prog.Functions[0].Inputs = inputs
	prog.Functions[0].Ops = ops
	prog.Functions[0].Returns = returns

	if err := modelir.Check(prog); err != nil {
		return nil, fmt.Errorf("lower nanochat decode: %w", err)
	}
	return prog, nil
}

func validateNanochatArtifactSupportedConfig(cfg NanochatDecodeConfig) error {
	cfg = normalizeNanochatDecodeConfig(cfg)
	if err := validateNanochatDecodeConfig(cfg); err != nil {
		return err
	}
	return nil
}

func modelirTensor(name string, dtype modelir.DType, shape ...int64) modelir.Value {
	_ = dtype
	return modelir.Value{
		Name: name,
		Type: modelir.TensorType{
			DType: bridgeModelIRDType,
			Shape: modelir.Shape(shape),
		},
	}
}

func modelirConst(name string, dtype modelir.DType, shape modelir.Shape, ref string) modelir.Const {
	_ = dtype
	return modelir.Const{
		Value: modelir.Value{
			Name: name,
			Type: modelir.TensorType{
				DType: bridgeModelIRDType,
				Shape: shape,
			},
		},
		WeightRef: ref,
	}
}

func modelirOut(name string, dtype modelir.DType, shape ...int64) modelir.Value {
	return modelirTensor(name, dtype, shape...)
}

func modelirOp(name string, output modelir.Value, inputs ...string) modelir.Op {
	return modelir.Op{
		Name:    name,
		Inputs:  append([]string(nil), inputs...),
		Outputs: []modelir.Value{output},
	}
}

func modelirOpAttr(name string, attrs string, output modelir.Value, inputs ...string) modelir.Op {
	return modelir.Op{
		Name:    name,
		Inputs:  append([]string(nil), inputs...),
		Outputs: []modelir.Value{output},
		Attrs:   attrs,
	}
}

func modelirLinearOps(prefix, input string, dim int64) []modelir.Op {
	return modelirLinearOpsNamed(prefix, input, prefix, dim)
}

func modelirLinearOpsNamed(outputPrefix, input, weightPrefix string, dim int64) []modelir.Op {
	return []modelir.Op{
		modelirOp("matmul", modelirOut(outputPrefix+"_proj", modelir.DTypeFP16, 1, 1, dim), input, weightPrefix+"_w"),
		modelirOp("add", modelirOut(outputPrefix, modelir.DTypeFP16, 1, 1, dim), outputPrefix+"_proj", weightPrefix+"_b"),
	}
}

func rawFP16Bytes(vals []float32) []byte {
	out := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(v))
	}
	return out
}

func rawFP32Bytes(vals []float32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
