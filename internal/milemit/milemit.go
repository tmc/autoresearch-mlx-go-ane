//go:build darwin && ane_appleneuralengine

package milemit

import (
	"fmt"
	"strings"
)

const blobChunkOffset = 64

// WriteLinearConstBlock emits one MIL linear weight+bias const block.
func WriteLinearConstBlock(
	b *strings.Builder,
	wName, bName, wPath, bPath string,
	outDim, inDim int,
) {
	fmt.Fprintf(
		b,
		"        tensor<fp16, [%d, %d]> %s = const()[name=string(\"%s\"), val=tensor<fp16, [%d, %d]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
		outDim, inDim, wName, wName, outDim, inDim, wPath, blobChunkOffset,
	)
	fmt.Fprintf(
		b,
		"        tensor<fp16, [%d]> %s = const()[name=string(\"%s\"), val=tensor<fp16, [%d]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
		outDim, bName, bName, outDim, bPath, blobChunkOffset,
	)
}

// WriteConvConstBlock emits one MIL conv weight+bias const block.
func WriteConvConstBlock(
	b *strings.Builder,
	wName, bName, wPath, bPath string,
	outDim, inDim int,
) {
	fmt.Fprintf(
		b,
		"        tensor<fp16, [%d, %d, 1, 1]> %s = const()[name=string(\"%s\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
		outDim,
		inDim,
		wName,
		wName,
		outDim,
		inDim,
		wPath,
		blobChunkOffset,
	)
	fmt.Fprintf(
		b,
		"        tensor<fp16, [1, %d, 1, 1]> %s = const()[name=string(\"%s\"), val=tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
		outDim,
		bName,
		bName,
		outDim,
		bPath,
		blobChunkOffset,
	)
}

// WriteRMSNorm3D emits a numerically safer MIL RMSNorm sequence for [1,1,Dim].
func WriteRMSNorm3D(
	b *strings.Builder,
	prefix string,
	inVar string,
	dim int,
	eps float64,
	normPath string,
) string {
	absVar := prefix + "_abs"
	ax := prefix + "_ax"
	kd := prefix + "_kd"
	maxAbs := prefix + "_maxabs"
	epsVar := prefix + "_eps"
	safeMax := prefix + "_safemax"
	scaled := prefix + "_scaled"
	square := prefix + "_square"
	meanSq := prefix + "_meansq"
	meanEps := prefix + "_meaneps"
	rms := prefix + "_rms"
	scaledRMS := prefix + "_scaledrms"
	xn := prefix + "_norm"
	rw := prefix + "_rw"
	out := prefix + "_out"
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d]> %s = abs(x=%s)[name=string(\"%s\")];\n", dim, absVar, inVar, absVar)
	fmt.Fprintf(b, "        tensor<int32, [1]> %s = const()[name=string(\"%s\"), val=tensor<int32, [1]>([2])];\n", ax, ax)
	fmt.Fprintf(b, "        bool %s = const()[name=string(\"%s\"), val=bool(true)];\n", kd, kd)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1]> %s = reduce_max(x=%s, axes=%s, keep_dims=%s)[name=string(\"%s\")];\n", maxAbs, absVar, ax, kd, maxAbs)
	fmt.Fprintf(b, "        fp16 %s = const()[name=string(\"%s\"), val=fp16(%f)];\n", epsVar, epsVar, eps)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1]> %s = maximum(x=%s, y=%s)[name=string(\"%s\")];\n", safeMax, maxAbs, epsVar, safeMax)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d]> %s = real_div(x=%s, y=%s)[name=string(\"%s\")];\n", dim, scaled, inVar, safeMax, scaled)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d]> %s = square(x=%s)[name=string(\"%s\")];\n", dim, square, scaled, square)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1]> %s = reduce_mean(x=%s, axes=%s, keep_dims=%s)[name=string(\"%s\")];\n", meanSq, square, ax, kd, meanSq)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1]> %s = add(x=%s, y=%s)[name=string(\"%s\")];\n", meanEps, meanSq, epsVar, meanEps)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1]> %s = sqrt(x=%s)[name=string(\"%s\")];\n", rms, meanEps, rms)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1]> %s = mul(x=%s, y=%s)[name=string(\"%s\")];\n", scaledRMS, rms, safeMax, scaledRMS)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d]> %s = real_div(x=%s, y=%s)[name=string(\"%s\")];\n", dim, xn, inVar, scaledRMS, xn)
	fmt.Fprintf(
		b,
		"        tensor<fp16, [1,1,%d]> %s = const()[name=string(\"%s\"), val=tensor<fp16, [1,1,%d]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
		dim,
		rw,
		rw,
		dim,
		normPath,
		blobChunkOffset,
	)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d]> %s = mul(x=%s, y=%s)[name=string(\"%s\")];\n", dim, out, xn, rw, out)
	return out
}

// WriteRMSNorm4D emits a numerically safer MIL RMSNorm sequence for [1,H,1,D].
func WriteRMSNorm4D(
	b *strings.Builder,
	prefix string,
	inVar string,
	numHeads int,
	headDim int,
	eps float64,
	normPath string,
) string {
	absVar := prefix + "_abs"
	ax := prefix + "_ax"
	kd := prefix + "_kd"
	maxAbs := prefix + "_maxabs"
	epsVar := prefix + "_eps"
	safeMax := prefix + "_safemax"
	scaled := prefix + "_scaled"
	square := prefix + "_square"
	meanSq := prefix + "_meansq"
	meanEps := prefix + "_meaneps"
	rms := prefix + "_rms"
	scaledRMS := prefix + "_scaledrms"
	xn := prefix + "_norm"
	rw := prefix + "_rw"
	out := prefix + "_out"
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s = abs(x=%s)[name=string(\"%s\")];\n", numHeads, headDim, absVar, inVar, absVar)
	fmt.Fprintf(b, "        tensor<int32, [1]> %s = const()[name=string(\"%s\"), val=tensor<int32, [1]>([3])];\n", ax, ax)
	fmt.Fprintf(b, "        bool %s = const()[name=string(\"%s\"), val=bool(true)];\n", kd, kd)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,1]> %s = reduce_max(x=%s, axes=%s, keep_dims=%s)[name=string(\"%s\")];\n", numHeads, maxAbs, absVar, ax, kd, maxAbs)
	fmt.Fprintf(b, "        fp16 %s = const()[name=string(\"%s\"), val=fp16(%f)];\n", epsVar, epsVar, eps)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,1]> %s = maximum(x=%s, y=%s)[name=string(\"%s\")];\n", numHeads, safeMax, maxAbs, epsVar, safeMax)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s = real_div(x=%s, y=%s)[name=string(\"%s\")];\n", numHeads, headDim, scaled, inVar, safeMax, scaled)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s = square(x=%s)[name=string(\"%s\")];\n", numHeads, headDim, square, scaled, square)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,1]> %s = reduce_mean(x=%s, axes=%s, keep_dims=%s)[name=string(\"%s\")];\n", numHeads, meanSq, square, ax, kd, meanSq)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,1]> %s = add(x=%s, y=%s)[name=string(\"%s\")];\n", numHeads, meanEps, meanSq, epsVar, meanEps)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,1]> %s = sqrt(x=%s)[name=string(\"%s\")];\n", numHeads, rms, meanEps, rms)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,1]> %s = mul(x=%s, y=%s)[name=string(\"%s\")];\n", numHeads, scaledRMS, rms, safeMax, scaledRMS)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s = real_div(x=%s, y=%s)[name=string(\"%s\")];\n", numHeads, headDim, xn, inVar, scaledRMS, xn)
	fmt.Fprintf(
		b,
		"        tensor<fp16, [1,1,1,%d]> %s = const()[name=string(\"%s\"), val=tensor<fp16, [1,1,1,%d]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%d)))];\n",
		headDim,
		rw,
		rw,
		headDim,
		normPath,
		blobChunkOffset,
	)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s = mul(x=%s, y=%s)[name=string(\"%s\")];\n", numHeads, headDim, out, xn, rw, out)
	return out
}
