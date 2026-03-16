//go:build darwin && ane_appleneuralengine

package anedraftimpl

import (
	"github.com/tmc/mlx-go-lm/mlxlm/decode"
	"github.com/tmc/mlx-go/mlx"
)

// aneSSDDrafterAdapter keeps ANE-specific SSD behavior isolated from decode.
//
// Today this is a thin forwarding layer over aneDraftDrafter. It provides a
// stable seam for SSD-specific ANE optimizations (for example, batched draft
// generation) without coupling decode.SpeculativeSSD to ANE internals.
type aneSSDDrafterAdapter struct {
	inner *aneDraftDrafter
}

func NewANESSDDrafterAdapter(d decode.Drafter) (decode.Drafter, error) {
	ane, ok := d.(*aneDraftDrafter)
	if !ok {
		// Reference-only fallback can intentionally return a non-ANE drafter.
		// Keep SSD path functional by passing it through unchanged.
		return d, nil
	}
	return &aneSSDDrafterAdapter{inner: ane}, nil
}

func newANESSDDrafterAdapter(d decode.Drafter) (decode.Drafter, error) {
	return NewANESSDDrafterAdapter(d)
}

func (a *aneSSDDrafterAdapter) Draft(lastToken *mlx.Array, n int) ([]int32, error) {
	return a.inner.Draft(lastToken, n)
}

func (a *aneSSDDrafterAdapter) Rewind(n int) {
	a.inner.Rewind(n)
}

func (a *aneSSDDrafterAdapter) Prefill(prompt *mlx.Array) error {
	return a.inner.Prefill(prompt)
}

func (a *aneSSDDrafterAdapter) Reset() {
	a.inner.Reset()
}

var _ decode.Drafter = (*aneSSDDrafterAdapter)(nil)

func (a *aneSSDDrafterAdapter) ANEDraftRuntimeStats() aneDraftRuntimeStats {
	if a == nil || a.inner == nil {
		return aneDraftRuntimeStats{}
	}
	return a.inner.ANEDraftRuntimeStats()
}
