package mlxgoane

import "testing"

func TestHashFloat32SliceDeterministic(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	h1 := hashFloat32Slice(a)
	h2 := hashFloat32Slice(a)
	if h1 != h2 {
		t.Fatalf("hash not deterministic: %d != %d", h1, h2)
	}
}

func TestHashFloat32SliceDiffersForDifferentData(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{1, 2, 3, 5}
	if hashFloat32Slice(a) == hashFloat32Slice(b) {
		t.Fatal("different slices have identical hash")
	}
}

func TestHashFloat32SliceHandlesNilAndEmpty(t *testing.T) {
	nilHash := hashFloat32Slice(nil)
	emptyHash := hashFloat32Slice([]float32{})
	if nilHash != emptyHash {
		t.Fatalf("nil hash=%d empty hash=%d", nilHash, emptyHash)
	}
}
