package mlxgoane

import (
	"hash/fnv"
	"unsafe"
)

func hashFloat32Slice(data []float32) uint64 {
	h := fnv.New64a()
	if len(data) == 0 {
		return h.Sum64()
	}
	b := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
	_, _ = h.Write(b)
	return h.Sum64()
}
