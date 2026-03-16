//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"strings"

	"github.com/tmc/mlx-go-ane/internal/milemit"
)

func writeLinearConstBlock(
	b *strings.Builder,
	wName, bName, wPath, bPath string,
	outDim, inDim int,
) {
	milemit.WriteLinearConstBlock(b, wName, bName, wPath, bPath, outDim, inDim)
}

func writeConvConstBlock(
	b *strings.Builder,
	wName, bName, wPath, bPath string,
	outDim, inDim int,
) {
	milemit.WriteConvConstBlock(b, wName, bName, wPath, bPath, outDim, inDim)
}

func writeRMSNorm3D(
	b *strings.Builder,
	prefix string,
	inVar string,
	dim int,
	eps float64,
	normPath string,
) string {
	return milemit.WriteRMSNorm3D(b, prefix, inVar, dim, eps, normPath)
}

func writeRMSNorm4D(
	b *strings.Builder,
	prefix string,
	inVar string,
	numHeads int,
	headDim int,
	eps float64,
	normPath string,
) string {
	return milemit.WriteRMSNorm4D(b, prefix, inVar, numHeads, headDim, eps, normPath)
}
