package mlxgoane

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	blobHeaderSize      = 64
	blobChunkHeaderSize = 64
	blobChunkOffset     = blobHeaderSize
	blobDataOffset      = blobHeaderSize + blobChunkHeaderSize

	linearWeightBlobPath       = "weights/weight.bin"
	linearWeightBlobOffset     = blobChunkOffset // chunk header offset used by MIL BLOBFILE
	linearWeightBlobDataOffset = blobDataOffset  // payload offset encoded in chunk header
)

// buildLinearWeightsBlob returns a weight blob for BLOBFILE-backed MIL constants.
//
// Input weights are in [outDim, inDim] row-major order and are stored as fp16
// payload in the same order for conv weight layout [outDim, inDim, 1, 1].
func buildLinearWeightsBlob(w []float32, outDim, inDim int) ([]byte, error) {
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("appleneuralengine linear: invalid weight dims out=%d in=%d", outDim, inDim)
	}
	if len(w) != outDim*inDim {
		return nil, fmt.Errorf("appleneuralengine linear: weight len=%d want=%d", len(w), outDim*inDim)
	}
	if len(w) == 0 {
		return buildBLOBFileFP16(nil), nil
	}
	payload := make([]float32, len(w))
	copy(payload, w)
	return buildBLOBFileFP16(payload), nil
}

// float32ToFloat16Bits converts float32 to IEEE-754 binary16 bits.
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff

	switch exp {
	case 0xff:
		if frac != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case 0:
		return sign
	}

	exp = exp - 127 + 15
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint32(14 - exp)
		rounded := frac + (1 << (shift - 1))
		return sign | uint16(rounded>>shift)
	}

	rounded := frac + 0x1000
	if rounded&0x800000 != 0 {
		exp++
		rounded = 0
		if exp >= 0x1f {
			return sign | 0x7c00
		}
	}
	return sign | uint16(exp<<10) | uint16(rounded>>13)
}

// buildBLOBFileFP16 wraps fp16 payload data with the BLOBFILE header/chunk format.
func buildBLOBFileFP16(payload []float32) []byte {
	wsize := len(payload) * 2
	blob := make([]byte, blobDataOffset+wsize)
	blob[0] = 0x01
	blob[4] = 0x02

	chunk := blob[blobChunkOffset:]
	chunk[0] = 0xEF
	chunk[1] = 0xBE
	chunk[2] = 0xAD
	chunk[3] = 0xDE
	chunk[4] = 0x01
	binary.LittleEndian.PutUint32(chunk[8:12], uint32(wsize))
	binary.LittleEndian.PutUint32(chunk[16:20], uint32(blobDataOffset))

	for i, v := range payload {
		binary.LittleEndian.PutUint16(
			blob[blobDataOffset+i*2:],
			float32ToFloat16Bits(v),
		)
	}
	return blob
}
