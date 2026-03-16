package mlxgoane

import (
	"reflect"
	"testing"
)

func TestAssessDecodeModelQwenLike(t *testing.T) {
	got := AssessDecodeModel(DecodeModelSpec{
		ModelDim:   3072,
		NumHeads:   24,
		NumKVHeads: 8,
		HeadDim:    128,
		FFNDim:     8192,
		MLPKind:    MLPKindSwiGLU,
	})
	if got.RecommendedPath != IntegrationPathSpeculativeDraft {
		t.Fatalf("RecommendedPath = %q, want %q", got.RecommendedPath, IntegrationPathSpeculativeDraft)
	}
	if !got.AvoidLinearHook {
		t.Fatal("AvoidLinearHook = false, want true")
	}
	if !got.SupportsSurfaceDecodeFFN {
		t.Fatal("SupportsSurfaceDecodeFFN = false, want true")
	}
	if !got.SupportsCurrentDraftMIL {
		t.Fatal("SupportsCurrentDraftMIL = false, want true")
	}
	if got.NeedsModelSpecificDecode {
		t.Fatal("NeedsModelSpecificDecode = true, want false")
	}
	if got.DraftLowConfidence {
		t.Fatal("DraftLowConfidence = true, want false")
	}
}

func TestAssessDecodeModelNanochatLike(t *testing.T) {
	got := AssessDecodeModel(DecodeModelSpec{
		ModelDim:            768,
		NumHeads:            6,
		NumKVHeads:          6,
		HeadDim:             128,
		FFNDim:              3072,
		MLPKind:             MLPKindReluSquared,
		UsesResidualMix:     true,
		UsesQKNorm:          true,
		UsesValueEmbeddings: true,
		UsesSlidingWindow:   true,
	})
	if got.RecommendedPath != IntegrationPathModelSpecificDecode {
		t.Fatalf("RecommendedPath = %q, want %q", got.RecommendedPath, IntegrationPathModelSpecificDecode)
	}
	if !got.AvoidLinearHook {
		t.Fatal("AvoidLinearHook = false, want true")
	}
	if got.SupportsSurfaceDecodeFFN {
		t.Fatal("SupportsSurfaceDecodeFFN = true, want false")
	}
	if got.SupportsCurrentDraftMIL {
		t.Fatal("SupportsCurrentDraftMIL = true, want false")
	}
	if !got.NeedsModelSpecificDecode {
		t.Fatal("NeedsModelSpecificDecode = false, want true")
	}
	wantReasons := []string{
		"prefer block-level decode integration over per-linear hooks",
		"current draft/decode templates assume SwiGLU FFN",
		"residual mixing must be preserved inside the decode backend",
		"QK norm changes the attention block contract",
		"value embeddings require token-aware attention state",
		"sliding-window KV requires model-specific cache handling",
	}
	if !reflect.DeepEqual(got.Reasons, wantReasons) {
		t.Fatalf("Reasons = %#v, want %#v", got.Reasons, wantReasons)
	}
}

func TestAssessDecodeModelWidenedAttention(t *testing.T) {
	got := AssessDecodeModel(DecodeModelSpec{
		ModelDim:   768,
		NumHeads:   12,
		NumKVHeads: 12,
		HeadDim:    96,
		FFNDim:     3072,
		MLPKind:    MLPKindSwiGLU,
	})
	if !got.DraftLowConfidence {
		t.Fatal("DraftLowConfidence = false, want true")
	}
	if got.RecommendedPath != IntegrationPathSpeculativeDraft {
		t.Fatalf("RecommendedPath = %q, want %q", got.RecommendedPath, IntegrationPathSpeculativeDraft)
	}
}

func TestDecodeModelSpecHelpers(t *testing.T) {
	spec := DecodeModelSpec{
		ModelDim: 768,
		NumHeads: 12,
		HeadDim:  96,
		MLPKind:  MLPKindReluSquared,
	}
	if !spec.HasCustomDecodeSemantics() {
		t.Fatal("HasCustomDecodeSemantics = false, want true")
	}
	if !spec.HasWidenedAttention() {
		t.Fatal("HasWidenedAttention = false, want true")
	}
}
