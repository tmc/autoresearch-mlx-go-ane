package mlxgoane

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestBuildLinearWeightsBlobLayout(t *testing.T) {
	w := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	blob, err := buildLinearWeightsBlob(w, 2, 3)
	if err != nil {
		t.Fatalf("buildLinearWeightsBlob: %v", err)
	}

	if got, want := len(blob), linearWeightBlobDataOffset+len(w)*2; got != want {
		t.Fatalf("blob len=%d want=%d", got, want)
	}
	if blob[0] != 0x01 || blob[4] != 0x02 {
		t.Fatalf("global header mismatch: b0=%#x b4=%#x", blob[0], blob[4])
	}
	chunk := blob[linearWeightBlobOffset:]
	if got, want := binary.LittleEndian.Uint32(chunk[0:4]), uint32(0xDEADBEEF); got != want {
		t.Fatalf("chunk magic=%#x want=%#x", got, want)
	}
	if got, want := chunk[4], byte(0x01); got != want {
		t.Fatalf("chunk version=%#x want=%#x", got, want)
	}
	if got, want := binary.LittleEndian.Uint32(chunk[8:12]), uint32(len(w)*2); got != want {
		t.Fatalf("chunk data_size=%d want=%d", got, want)
	}
	if got, want := binary.LittleEndian.Uint32(chunk[16:20]), uint32(linearWeightBlobDataOffset); got != want {
		t.Fatalf("chunk data_offset=%d want=%d", got, want)
	}

	// Payload should remain [out,in] row-major for conv weights.
	wantPayload := []float32{1, 2, 3, 4, 5, 6}
	for i, want := range wantPayload {
		bits := binary.LittleEndian.Uint16(blob[linearWeightBlobDataOffset+i*2:])
		got := float16BitsToFloat32(bits)
		if math.Abs(float64(got-want)) > 1e-3 {
			t.Fatalf("payload[%d]=%g want=%g", i, got, want)
		}
	}
}

func TestBuildLinearWeightsBlobRejectsInvalidDims(t *testing.T) {
	_, err := buildLinearWeightsBlob([]float32{1, 2, 3}, 0, 3)
	if err == nil {
		t.Fatal("buildLinearWeightsBlob returned nil error for outDim=0")
	}
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x03ff)

	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			// Normalize subnormal.
			e := int32(-14)
			f := frac
			for (f & 0x0400) == 0 {
				f <<= 1
				e--
			}
			f &= 0x03ff
			bits = (sign << 31) | (uint32(e+127) << 23) | (f << 13)
		}
	case 0x1f:
		bits = (sign << 31) | 0x7f800000 | (frac << 13)
	default:
		bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
	}
	return math.Float32frombits(bits)
}
