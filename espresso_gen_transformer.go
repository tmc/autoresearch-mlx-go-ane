//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// TransformerConfig describes the generated transformer topology.
//
// Current generator emits a full decode-style block structure:
//   - embedding
//   - NumLayers * (attention + residual + FFN + residual)
//   - lm_head
//
// RMSNorm is intentionally left optional and disabled by default while
// acceptable reduce/pow layer combinations are still being probed.
type TransformerConfig struct {
	NumLayers int
	Dim       int
	NumHeads  int
	HeadDim   int
	HiddenDim int
	VocabSize int

	UseLookupEmbedding bool
	IncludeRMSNorm     bool
}

// TransformerLayerWeights contains one transformer block's projection tensors.
type TransformerLayerWeights struct {
	QProj []float32 // [dim, dim]
	KProj []float32 // [dim, dim]
	VProj []float32 // [dim, dim]
	OProj []float32 // [dim, dim]
	W1    []float32 // [hiddenDim, dim]
	W3    []float32 // [hiddenDim, dim]
	W2    []float32 // [dim, hiddenDim]

	AttnNorm []float32 // [dim], used when IncludeRMSNorm=true
	FFNNorm  []float32 // [dim], used when IncludeRMSNorm=true
}

// TransformerWeights contains all tensor data required by
// GenerateTransformerEspressoDir.
type TransformerWeights struct {
	Embedding []float32 // lookup: [dim, vocab], projection path: first [dim, dim] slice
	Layers    []TransformerLayerWeights
	FinalNorm []float32 // [dim], used when IncludeRMSNorm=true
	LMHead    []float32 // [vocab, dim]
}

type espressoWeightSlot struct {
	size uint64
	data []byte
}

type espressoWeightBuilder struct {
	slots []espressoWeightSlot
}

func (b *espressoWeightBuilder) addSlot(size uint64, data []byte) int {
	b.slots = append(b.slots, espressoWeightSlot{size: size, data: data})
	return len(b.slots) - 1
}

func (b *espressoWeightBuilder) addWeightOnly(weight []float32) int {
	descPos := b.addSlot(0, nil)
	weightBytes := float32SliceLEBytes(weight)
	b.addSlot(uint64(len(weightBytes)), weightBytes)
	return descPos
}

func (b *espressoWeightBuilder) addLinear(weight []float32, outDim int) (weightDescPos, biasDescPos int) {
	biasDescPos = b.addSlot(56, make([]byte, 56))
	b.addSlot(uint64(outDim*16), make([]byte, outDim*16))
	weightDescPos = b.addSlot(0, nil)
	weightBytes := float32SliceLEBytes(weight)
	b.addSlot(uint64(len(weightBytes)), weightBytes)
	return weightDescPos, biasDescPos
}

func (b *espressoWeightBuilder) count() int {
	return len(b.slots)
}

func (b *espressoWeightBuilder) blobID(pos int) int {
	n := b.count()
	if n == 0 {
		return 0
	}
	return (pos + 1) % n
}

func (b *espressoWeightBuilder) build() ([]byte, error) {
	if b.count() == 0 {
		return nil, fmt.Errorf("build transformer weights: empty slot table")
	}
	buf := &bytes.Buffer{}
	if err := binary.Write(buf, binary.LittleEndian, uint64(b.count())); err != nil {
		return nil, fmt.Errorf("build transformer weights: write blob count: %w", err)
	}
	if err := binary.Write(buf, binary.LittleEndian, uint64(0)); err != nil {
		return nil, fmt.Errorf("build transformer weights: write reserved field: %w", err)
	}
	for pos, slot := range b.slots {
		if err := binary.Write(buf, binary.LittleEndian, slot.size); err != nil {
			return nil, fmt.Errorf("build transformer weights: write size at slot=%d: %w", pos, err)
		}
		id := uint64(b.blobID(pos))
		if err := binary.Write(buf, binary.LittleEndian, id); err != nil {
			return nil, fmt.Errorf("build transformer weights: write id at slot=%d: %w", pos, err)
		}
	}
	payloadOffset := espressoWeightsPayloadOffsetForSlots(b.count())
	if buf.Len() > payloadOffset {
		return nil, fmt.Errorf(
			"build transformer weights: header length=%d exceeds payload offset=%d",
			buf.Len(),
			payloadOffset,
		)
	}
	if buf.Len() < payloadOffset {
		padding := make([]byte, payloadOffset-buf.Len())
		if _, err := buf.Write(padding); err != nil {
			return nil, fmt.Errorf("build transformer weights: write payload padding: %w", err)
		}
	}
	for pos, slot := range b.slots {
		if slot.size == 0 {
			continue
		}
		payload := slot.data
		if uint64(len(payload)) > slot.size {
			payload = payload[:slot.size]
		}
		if uint64(len(payload)) < slot.size {
			padded := make([]byte, slot.size)
			copy(padded, payload)
			payload = padded
		}
		if _, err := buf.Write(payload); err != nil {
			return nil, fmt.Errorf("build transformer weights: write payload at slot=%d: %w", pos, err)
		}
	}
	return buf.Bytes(), nil
}

func validateTransformerInputs(cfg TransformerConfig, weights TransformerWeights) error {
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("generate transformer espresso dir: invalid NumLayers=%d", cfg.NumLayers)
	}
	if cfg.Dim <= 0 || cfg.HiddenDim <= 0 || cfg.VocabSize <= 0 {
		return fmt.Errorf(
			"generate transformer espresso dir: invalid dims dim=%d hidden=%d vocab=%d",
			cfg.Dim,
			cfg.HiddenDim,
			cfg.VocabSize,
		)
	}
	if cfg.NumHeads <= 0 || cfg.HeadDim <= 0 {
		return fmt.Errorf("generate transformer espresso dir: invalid heads=%d headDim=%d", cfg.NumHeads, cfg.HeadDim)
	}
	if cfg.NumHeads*cfg.HeadDim != cfg.Dim {
		return fmt.Errorf(
			"generate transformer espresso dir: NumHeads*HeadDim=%d want Dim=%d",
			cfg.NumHeads*cfg.HeadDim,
			cfg.Dim,
		)
	}
	if cfg.IncludeRMSNorm {
		return fmt.Errorf("generate transformer espresso dir: IncludeRMSNorm is not implemented yet")
	}
	if len(weights.Layers) != cfg.NumLayers {
		return fmt.Errorf(
			"generate transformer espresso dir: layer weight count=%d want=%d",
			len(weights.Layers),
			cfg.NumLayers,
		)
	}
	if cfg.UseLookupEmbedding {
		if len(weights.Embedding) != cfg.Dim*cfg.VocabSize {
			return fmt.Errorf(
				"generate transformer espresso dir: embedding len=%d want=%d for lookup embedding",
				len(weights.Embedding),
				cfg.Dim*cfg.VocabSize,
			)
		}
	} else if len(weights.Embedding) < cfg.Dim*cfg.Dim {
		return fmt.Errorf(
			"generate transformer espresso dir: embedding len=%d want>=%d for projection embedding",
			len(weights.Embedding),
			cfg.Dim*cfg.Dim,
		)
	}
	if len(weights.LMHead) != cfg.VocabSize*cfg.Dim {
		return fmt.Errorf(
			"generate transformer espresso dir: lm_head len=%d want=%d",
			len(weights.LMHead),
			cfg.VocabSize*cfg.Dim,
		)
	}
	for i, layer := range weights.Layers {
		for name, tensor := range map[string][]float32{
			"q_proj": layer.QProj,
			"k_proj": layer.KProj,
			"v_proj": layer.VProj,
			"o_proj": layer.OProj,
		} {
			if len(tensor) != cfg.Dim*cfg.Dim {
				return fmt.Errorf(
					"generate transformer espresso dir: layer[%d] %s len=%d want=%d",
					i,
					name,
					len(tensor),
					cfg.Dim*cfg.Dim,
				)
			}
		}
		if len(layer.W1) != cfg.HiddenDim*cfg.Dim {
			return fmt.Errorf(
				"generate transformer espresso dir: layer[%d] w1 len=%d want=%d",
				i,
				len(layer.W1),
				cfg.HiddenDim*cfg.Dim,
			)
		}
		if len(layer.W3) != cfg.HiddenDim*cfg.Dim {
			return fmt.Errorf(
				"generate transformer espresso dir: layer[%d] w3 len=%d want=%d",
				i,
				len(layer.W3),
				cfg.HiddenDim*cfg.Dim,
			)
		}
		if len(layer.W2) != cfg.Dim*cfg.HiddenDim {
			return fmt.Errorf(
				"generate transformer espresso dir: layer[%d] w2 len=%d want=%d",
				i,
				len(layer.W2),
				cfg.Dim*cfg.HiddenDim,
			)
		}
	}
	return nil
}

func shapeRank3(width int) espressoTensorShape {
	return espressoTensorShape{
		K:    1,
		W:    width,
		N:    1,
		Rank: 3,
		H:    1,
	}
}

func shapeRank4(n, h, w, k int) espressoTensorShape {
	return espressoTensorShape{
		K:    k,
		W:    w,
		N:    n,
		Rank: 4,
		H:    h,
	}
}

// GenerateTransformerEspressoDir writes a transformer .mlmodelc directory as
// model.espresso.net + model.espresso.shape + model.espresso.weights.
func GenerateTransformerEspressoDir(dir string, cfg TransformerConfig, weights TransformerWeights) error {
	if err := validateTransformerInputs(cfg, weights); err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("generate transformer espresso dir: mkdir %q: %w", dir, err)
	}

	builder := &espressoWeightBuilder{}
	var (
		embeddingWeightPos int
		embeddingBiasPos   int
	)
	if cfg.UseLookupEmbedding {
		embeddingWeightPos = builder.addWeightOnly(weights.Embedding)
		embeddingBiasPos = -1
	} else {
		embedProj := weights.Embedding[:cfg.Dim*cfg.Dim]
		embeddingWeightPos, embeddingBiasPos = builder.addLinear(embedProj, cfg.Dim)
	}

	type layerBlobRef struct {
		qW, qB int
		kW, kB int
		vW, vB int
		oW, oB int
		w1, b1 int
		w3, b3 int
		w2, b2 int
	}
	refs := make([]layerBlobRef, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		qW, qB := builder.addLinear(weights.Layers[i].QProj, cfg.Dim)
		kW, kB := builder.addLinear(weights.Layers[i].KProj, cfg.Dim)
		vW, vB := builder.addLinear(weights.Layers[i].VProj, cfg.Dim)
		oW, oB := builder.addLinear(weights.Layers[i].OProj, cfg.Dim)
		w1, b1 := builder.addLinear(weights.Layers[i].W1, cfg.HiddenDim)
		w3, b3 := builder.addLinear(weights.Layers[i].W3, cfg.HiddenDim)
		w2, b2 := builder.addLinear(weights.Layers[i].W2, cfg.Dim)
		refs[i] = layerBlobRef{
			qW: qW, qB: qB,
			kW: kW, kB: kB,
			vW: vW, vB: vB,
			oW: oW, oB: oB,
			w1: w1, b1: b1,
			w3: w3, b3: b3,
			w2: w2, b2: b2,
		}
	}
	lmWPos, lmBPos := builder.addLinear(weights.LMHead, cfg.VocabSize)

	net := EspressoNet{
		FormatVersion:     200,
		Storage:           "model.espresso.weights",
		Analyses:          map[string]interface{}{},
		Properties:        map[string]interface{}{},
		MetadataInWeights: []interface{}{},
	}
	shape := espressoShapeFile{
		LayerShapes: map[string]espressoTensorShape{},
	}

	embedTop := "embed_0"
	if cfg.UseLookupEmbedding {
		shape.LayerShapes["input_ids"] = shapeRank3(1)
		net.Layers = append(net.Layers, EspressoEmbeddingLayer(
			"embedding",
			"input_ids",
			embedTop,
			cfg.VocabSize,
			cfg.Dim,
			builder.blobID(embeddingWeightPos),
			0,
		))
	} else {
		shape.LayerShapes["x"] = shapeRank3(cfg.Dim)
		net.Layers = append(net.Layers, EspressoLinearLayer(
			"embedding",
			"x",
			embedTop,
			cfg.Dim,
			cfg.Dim,
			builder.blobID(embeddingWeightPos),
			builder.blobID(embeddingBiasPos),
		))
	}
	shape.LayerShapes[embedTop] = shapeRank3(cfg.Dim)

	scale := 1.0 / math.Sqrt(float64(cfg.HeadDim))
	current := embedTop
	for i := 0; i < cfg.NumLayers; i++ {
		ref := refs[i]
		pfx := fmt.Sprintf("l%d_", i)

		qFlat := pfx + "q_flat"
		kFlat := pfx + "k_flat"
		vFlat := pfx + "v_flat"
		q4d := pfx + "q_4d"
		k4d := pfx + "k_4d"
		v4d := pfx + "v_4d"
		scores := pfx + "scores"
		scaled := pfx + "scores_scaled"
		weightsTop := pfx + "attn_weights"
		context4d := pfx + "context_4d"
		context := pfx + "context"
		attnProj := pfx + "attn_proj"
		attnResid := pfx + "attn_resid"

		net.Layers = append(net.Layers,
			EspressoLinearLayer(pfx+"q_proj", current, qFlat, cfg.Dim, cfg.Dim, builder.blobID(ref.qW), builder.blobID(ref.qB)),
			EspressoLinearLayer(pfx+"k_proj", current, kFlat, cfg.Dim, cfg.Dim, builder.blobID(ref.kW), builder.blobID(ref.kB)),
			EspressoLinearLayer(pfx+"v_proj", current, vFlat, cfg.Dim, cfg.Dim, builder.blobID(ref.vW), builder.blobID(ref.vB)),
			EspressoReshapeLayer(pfx+"q_reshape", qFlat, q4d, []int{1, cfg.NumHeads, 1, cfg.HeadDim}),
			EspressoReshapeLayer(pfx+"k_reshape", kFlat, k4d, []int{1, cfg.NumHeads, 1, cfg.HeadDim}),
			EspressoReshapeLayer(pfx+"v_reshape", vFlat, v4d, []int{1, cfg.NumHeads, 1, cfg.HeadDim}),
			EspressoBatchMatMulLayer(pfx+"qk", q4d, k4d, scores, false, true),
			EspressoLayer{
				Name:      pfx + "scale",
				Type:      "elementwise",
				Top:       scaled,
				Bottom:    scores,
				Operation: 2,
				Alpha:     scale,
				Beta:      0,
				FusedRelu: 0,
				Weights:   newEspressoLayerWeights(),
				DebugInfo: pfx + "scale",
			},
			EspressoSoftmaxLayer(pfx+"softmax", scaled, weightsTop, -1),
			EspressoBatchMatMulLayer(pfx+"wv", weightsTop, v4d, context4d, false, false),
			EspressoReshapeLayer(pfx+"context_reshape", context4d, context, []int{1, cfg.Dim, 1, 1}),
			EspressoLinearLayer(pfx+"o_proj", context, attnProj, cfg.Dim, cfg.Dim, builder.blobID(ref.oW), builder.blobID(ref.oB)),
			EspressoLayer{
				Name:      pfx + "attn_residual",
				Type:      "elementwise",
				Top:       attnResid,
				Bottom:    fmt.Sprintf("%s,%s", current, attnProj),
				Operation: 0,
				Alpha:     1,
				Beta:      0,
				FusedRelu: 0,
				Weights:   newEspressoLayerWeights(),
				DebugInfo: pfx + "attn_residual",
			},
		)

		shape.LayerShapes[qFlat] = shapeRank3(cfg.Dim)
		shape.LayerShapes[kFlat] = shapeRank3(cfg.Dim)
		shape.LayerShapes[vFlat] = shapeRank3(cfg.Dim)
		shape.LayerShapes[q4d] = shapeRank4(1, cfg.NumHeads, 1, cfg.HeadDim)
		shape.LayerShapes[k4d] = shapeRank4(1, cfg.NumHeads, 1, cfg.HeadDim)
		shape.LayerShapes[v4d] = shapeRank4(1, cfg.NumHeads, 1, cfg.HeadDim)
		shape.LayerShapes[scores] = shapeRank4(1, cfg.NumHeads, 1, 1)
		shape.LayerShapes[scaled] = shapeRank4(1, cfg.NumHeads, 1, 1)
		shape.LayerShapes[weightsTop] = shapeRank4(1, cfg.NumHeads, 1, 1)
		shape.LayerShapes[context4d] = shapeRank4(1, cfg.NumHeads, 1, cfg.HeadDim)
		shape.LayerShapes[context] = shapeRank3(cfg.Dim)
		shape.LayerShapes[attnProj] = shapeRank3(cfg.Dim)
		shape.LayerShapes[attnResid] = shapeRank3(cfg.Dim)

		l0 := pfx + "linear_0"
		lSig := pfx + "silu_sigmoid"
		lGate := pfx + "gate"
		l1 := pfx + "linear_1"
		lMix := pfx + "ffn_input"
		l2 := pfx + "linear_2"
		ffnResid := pfx + "ffn_resid"

		net.Layers = append(net.Layers,
			EspressoLinearLayer(pfx+"ffn_gate_proj", attnResid, l0, cfg.Dim, cfg.HiddenDim, builder.blobID(ref.w1), builder.blobID(ref.b1)),
			EspressoLayer{
				Name:      pfx + "ffn_sigmoid",
				Type:      "activation",
				Top:       lSig,
				Bottom:    l0,
				Mode:      3,
				Weights:   newEspressoLayerWeights(),
				DebugInfo: pfx + "ffn_sigmoid",
			},
			EspressoLayer{
				Name:      pfx + "ffn_silu",
				Type:      "elementwise",
				Top:       lGate,
				Bottom:    fmt.Sprintf("%s,%s", l0, lSig),
				Operation: 1,
				Alpha:     1,
				Beta:      0,
				FusedRelu: 0,
				Weights:   newEspressoLayerWeights(),
				DebugInfo: pfx + "ffn_silu",
			},
			EspressoLinearLayer(pfx+"ffn_up_proj", attnResid, l1, cfg.Dim, cfg.HiddenDim, builder.blobID(ref.w3), builder.blobID(ref.b3)),
			EspressoLayer{
				Name:      pfx + "ffn_mul",
				Type:      "elementwise",
				Top:       lMix,
				Bottom:    fmt.Sprintf("%s,%s", lGate, l1),
				Operation: 1,
				Alpha:     1,
				Beta:      0,
				FusedRelu: 0,
				Weights:   newEspressoLayerWeights(),
				DebugInfo: pfx + "ffn_mul",
			},
			EspressoLinearLayer(pfx+"ffn_down_proj", lMix, l2, cfg.HiddenDim, cfg.Dim, builder.blobID(ref.w2), builder.blobID(ref.b2)),
			EspressoLayer{
				Name:      pfx + "ffn_residual",
				Type:      "elementwise",
				Top:       ffnResid,
				Bottom:    fmt.Sprintf("%s,%s", attnResid, l2),
				Operation: 0,
				Alpha:     1,
				Beta:      0,
				FusedRelu: 0,
				Weights:   newEspressoLayerWeights(),
				DebugInfo: pfx + "ffn_residual",
			},
		)

		shape.LayerShapes[l0] = shapeRank3(cfg.HiddenDim)
		shape.LayerShapes[lSig] = shapeRank3(cfg.HiddenDim)
		shape.LayerShapes[lGate] = shapeRank3(cfg.HiddenDim)
		shape.LayerShapes[l1] = shapeRank3(cfg.HiddenDim)
		shape.LayerShapes[lMix] = shapeRank3(cfg.HiddenDim)
		shape.LayerShapes[l2] = shapeRank3(cfg.Dim)
		shape.LayerShapes[ffnResid] = shapeRank3(cfg.Dim)
		current = ffnResid
	}

	lm := EspressoLinearLayer(
		"lm_head",
		current,
		"logits",
		cfg.Dim,
		cfg.VocabSize,
		builder.blobID(lmWPos),
		builder.blobID(lmBPos),
	)
	lm.Attributes = map[string]int{"is_output": 1}
	net.Layers = append(net.Layers, lm)
	shape.LayerShapes["logits"] = shapeRank3(cfg.VocabSize)

	netBytes, err := json.MarshalIndent(net, "", "  ")
	if err != nil {
		return fmt.Errorf("generate transformer espresso dir: marshal net: %w", err)
	}
	netBytes = append(netBytes, '\n')
	if err := os.WriteFile(filepath.Join(dir, "model.espresso.net"), netBytes, 0o644); err != nil {
		return fmt.Errorf("generate transformer espresso dir: write model.espresso.net: %w", err)
	}

	shapeBytes, err := json.MarshalIndent(shape, "", "  ")
	if err != nil {
		return fmt.Errorf("generate transformer espresso dir: marshal shape: %w", err)
	}
	shapeBytes = append(shapeBytes, '\n')
	if err := os.WriteFile(filepath.Join(dir, "model.espresso.shape"), shapeBytes, 0o644); err != nil {
		return fmt.Errorf("generate transformer espresso dir: write model.espresso.shape: %w", err)
	}

	weightsBytes, err := builder.build()
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "model.espresso.weights"), weightsBytes, 0o644); err != nil {
		return fmt.Errorf("generate transformer espresso dir: write model.espresso.weights: %w", err)
	}
	return nil
}
