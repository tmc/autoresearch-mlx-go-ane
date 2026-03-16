//go:build darwin && ane_appleneuralengine

package anedraftimpl

import "github.com/tmc/mlx-go-lm/mlxlm/decode"

func RuntimeActive(d decode.Drafter) bool {
	ad, ok := d.(*aneDraftDrafter)
	if !ok || ad == nil {
		return false
	}
	return !ad.referenceOnly
}

func aneDrafterRuntimeActive(d decode.Drafter) bool {
	return RuntimeActive(d)
}
