package mlxgoane

// MLPKind describes the feed-forward block shape used by a decode model.
type MLPKind string

const (
	MLPKindUnknown     MLPKind = ""
	MLPKindSwiGLU      MLPKind = "swiglu"
	MLPKindReluSquared MLPKind = "relu_squared"
)

// IntegrationPath is the recommended first integration path for a decode model.
type IntegrationPath string

const (
	IntegrationPathSpeculativeDraft    IntegrationPath = "speculative_draft"
	IntegrationPathModelSpecificDecode IntegrationPath = "model_specific_decode"
)

// DecodeModelSpec captures the decode-time semantics that matter when deciding
// whether an existing ANE path can be reused.
type DecodeModelSpec struct {
	ModelDim            int
	NumHeads            int
	NumKVHeads          int
	HeadDim             int
	FFNDim              int
	MLPKind             MLPKind
	UsesResidualMix     bool
	UsesQKNorm          bool
	UsesValueEmbeddings bool
	UsesSlidingWindow   bool
}

// HasCustomDecodeSemantics reports whether the model deviates from the current
// stock draft/decode templates.
func (s DecodeModelSpec) HasCustomDecodeSemantics() bool {
	return s.UsesResidualMix ||
		s.UsesQKNorm ||
		s.UsesValueEmbeddings ||
		s.UsesSlidingWindow ||
		s.MLPKind != MLPKindSwiGLU
}

// HasWidenedAttention reports whether the attention geometry is wider than the
// model hidden size. Those models currently need extra care in draft routing.
func (s DecodeModelSpec) HasWidenedAttention() bool {
	if s.ModelDim <= 0 || s.NumHeads <= 0 || s.HeadDim <= 0 {
		return false
	}
	return s.NumHeads*s.HeadDim > s.ModelDim
}

// IntegrationAssessment summarizes what the current mlx-go-ane code can reuse
// for a model and which path should be tried first.
type IntegrationAssessment struct {
	RecommendedPath          IntegrationPath
	AvoidLinearHook          bool
	SupportsCurrentDraftMIL  bool
	NeedsModelSpecificDecode bool
	DraftLowConfidence       bool
	Reasons                  []string
}

// AssessDecodeModel returns a deterministic recommendation for integrating ANE
// into a decode model.
func AssessDecodeModel(spec DecodeModelSpec) IntegrationAssessment {
	a := IntegrationAssessment{
		AvoidLinearHook: true,
		Reasons:         []string{"prefer block-level decode integration over per-linear hooks"},
	}

	if spec.HasCustomDecodeSemantics() {
		a.RecommendedPath = IntegrationPathModelSpecificDecode
		a.NeedsModelSpecificDecode = true
		if spec.MLPKind != MLPKindSwiGLU {
			a.Reasons = append(a.Reasons, "current draft/decode templates assume SwiGLU FFN")
		}
		if spec.UsesResidualMix {
			a.Reasons = append(a.Reasons, "residual mixing must be preserved inside the decode backend")
		}
		if spec.UsesQKNorm {
			a.Reasons = append(a.Reasons, "QK norm changes the attention block contract")
		}
		if spec.UsesValueEmbeddings {
			a.Reasons = append(a.Reasons, "value embeddings require token-aware attention state")
		}
		if spec.UsesSlidingWindow {
			a.Reasons = append(a.Reasons, "sliding-window KV requires model-specific cache handling")
		}
	} else {
		a.RecommendedPath = IntegrationPathSpeculativeDraft
		a.SupportsCurrentDraftMIL = true
		a.Reasons = append(a.Reasons, "current MIL draft path matches the model's decode semantics")
	}

	if spec.HasWidenedAttention() {
		a.DraftLowConfidence = true
		a.Reasons = append(a.Reasons, "widened attention is currently lower-confidence for ANE draft quality")
	}
	return a
}
