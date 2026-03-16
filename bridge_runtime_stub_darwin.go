//go:build darwin && ane_appleneuralengine && !ane_bridge

package mlxgoane

import (
	"fmt"
	"unsafe"
)

type aneBridgeRuntime struct {
	open       func(string, string, uintptr, uintptr) uintptr
	close      func(uintptr)
	writeInput func(uintptr, unsafe.Pointer, int32)
	readOutput func(uintptr, unsafe.Pointer, int32)
	eval       func(uintptr) bool

	createSharedEvent func() uintptr
	sharedEventPort   func(uintptr) uint32
	releaseObjc       func(uintptr)

	evalWithSignalEvent func(
		uintptr,
		unsafe.Pointer, uint32,
		unsafe.Pointer, uint32,
		uint32, uint64,
	) int32
	evalBidirectional func(
		uintptr,
		unsafe.Pointer, uint32,
		unsafe.Pointer, uint32,
		uint32, uint64,
		uint32, uint64,
	) int32
	signalEventCPU func(uint32, uint64) int32
	waitEventCPU   func(uint32, uint64, uint32) int32
}

var (
	aneBridgeSignalEventCPU func(eventPort uint32, value uint64) int32
	aneBridgeWaitEventCPU   func(eventPort uint32, value uint64, timeoutMs uint32) int32
)

func loadANEBridgeRuntime() (*aneBridgeRuntime, error) {
	return nil, fmt.Errorf("ane bridge runtime requires ane_bridge build tag")
}
