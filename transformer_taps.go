//go:build darwin && ane_appleneuralengine

package mlxgoane

import "fmt"

// TransformerForwardTapLayout describes packed concat output from transformer
// forward tap mode: [q_rope, k_rope, attn_scores, logits].
type TransformerForwardTapLayout struct {
	AttentionDim int
	LogitsDim    int
	TotalDim     int
}

// TransformerForwardTaps is the split view of packed transformer taps.
type TransformerForwardTaps struct {
	QRope      []float32
	KRope      []float32
	AttnScores []float32
	Logits     []float32
}

// TransformerTapLayoutForConfig derives tap layout from transformer config.
func TransformerTapLayoutForConfig(cfg MILTransformerConfig) (TransformerForwardTapLayout, error) {
	cfg = normalizeMILTransformerConfig(cfg)
	if err := validateMILTransformerConfig(cfg); err != nil {
		return TransformerForwardTapLayout{}, err
	}
	logitsDim := cfg.Dim
	if cfg.IncludeLMHead {
		logitsDim = cfg.VocabSize
	}
	return NewTransformerTapLayout(transformerAttentionDim(cfg), logitsDim)
}

// NewTransformerTapLayout validates and builds a packed tap layout.
func NewTransformerTapLayout(attnDim, logitsDim int) (TransformerForwardTapLayout, error) {
	if attnDim <= 0 || logitsDim <= 0 {
		return TransformerForwardTapLayout{}, fmt.Errorf(
			"transformer taps: invalid dims attention=%d logits=%d",
			attnDim,
			logitsDim,
		)
	}
	return TransformerForwardTapLayout{
		AttentionDim: attnDim,
		LogitsDim:    logitsDim,
		TotalDim:     3*attnDim + logitsDim,
	}, nil
}

// SplitTransformerForwardTaps splits packed transformer taps into named slices.
func SplitTransformerForwardTaps(out []float32, layout TransformerForwardTapLayout) (TransformerForwardTaps, error) {
	if layout.AttentionDim <= 0 || layout.LogitsDim <= 0 || layout.TotalDim <= 0 {
		return TransformerForwardTaps{}, fmt.Errorf(
			"transformer taps: invalid layout attention=%d logits=%d total=%d",
			layout.AttentionDim,
			layout.LogitsDim,
			layout.TotalDim,
		)
	}
	if len(out) != layout.TotalDim {
		return TransformerForwardTaps{}, fmt.Errorf(
			"transformer taps: output len=%d want=%d",
			len(out),
			layout.TotalDim,
		)
	}
	attn := layout.AttentionDim
	copyRange := func(start, end int) []float32 {
		segment := make([]float32, end-start)
		copy(segment, out[start:end])
		return segment
	}
	return TransformerForwardTaps{
		QRope:      copyRange(0, attn),
		KRope:      copyRange(attn, 2*attn),
		AttnScores: copyRange(2*attn, 3*attn),
		Logits:     copyRange(3*attn, 3*attn+layout.LogitsDim),
	}, nil
}
