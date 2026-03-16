//go:build darwin && ane_appleneuralengine && ane_bridge

package mlxgoane

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

type aneBridgeRuntime struct {
	lib uintptr

	init              func() int32
	open              func(string, string, uintptr, uintptr) uintptr
	close             func(uintptr)
	writeInput        func(uintptr, unsafe.Pointer, int32)
	readOutput        func(uintptr, unsafe.Pointer, int32)
	eval              func(uintptr) bool
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
	bridgeLoadOnce sync.Once
	bridgeRuntime  *aneBridgeRuntime
	bridgeLoadErr  error
	// Optional bridge CPU-side shared-event APIs.
	aneBridgeSignalEventCPU func(eventPort uint32, value uint64) int32
	aneBridgeWaitEventCPU   func(eventPort uint32, value uint64, timeoutMs uint32) int32
)

func loadANEBridgeRuntime() (*aneBridgeRuntime, error) {
	bridgeLoadOnce.Do(func() {
		home, err := os.UserHomeDir()
		if err != nil {
			bridgeLoadErr = fmt.Errorf("resolve home dir: %w", err)
			return
		}
		dylib := filepath.Join(home, "go", "src", "github.com", "maderix", "ANE", "bridge", "libane_bridge.dylib")
		if _, err := os.Stat(dylib); err != nil {
			bridgeLoadErr = fmt.Errorf("bridge dylib not found at %s: %w", dylib, err)
			return
		}
		lib, err := purego.Dlopen(dylib, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			bridgeLoadErr = fmt.Errorf("dlopen bridge dylib: %w", err)
			return
		}
		rt := &aneBridgeRuntime{lib: lib}
		purego.RegisterLibFunc(&rt.init, lib, "ane_bridge_init")
		purego.RegisterLibFunc(&rt.open, lib, "ane_bridge_client_open")
		purego.RegisterLibFunc(&rt.close, lib, "ane_bridge_client_close")
		purego.RegisterLibFunc(&rt.writeInput, lib, "ane_bridge_client_write_input")
		purego.RegisterLibFunc(&rt.readOutput, lib, "ane_bridge_client_read_output")
		purego.RegisterLibFunc(&rt.eval, lib, "ane_bridge_client_eval")
		if hasBridgeSymbol(lib, "ane_bridge_create_shared_event") {
			purego.RegisterLibFunc(&rt.createSharedEvent, lib, "ane_bridge_create_shared_event")
		}
		if hasBridgeSymbol(lib, "ane_bridge_shared_event_port") {
			purego.RegisterLibFunc(&rt.sharedEventPort, lib, "ane_bridge_shared_event_port")
		}
		if hasBridgeSymbol(lib, "ane_bridge_release_objc") {
			purego.RegisterLibFunc(&rt.releaseObjc, lib, "ane_bridge_release_objc")
		}

		if hasBridgeSymbol(lib, "ane_bridge_eval_with_signal_event") {
			purego.RegisterLibFunc(&rt.evalWithSignalEvent, lib, "ane_bridge_eval_with_signal_event")
		}
		if hasBridgeSymbol(lib, "ane_bridge_eval_bidirectional") {
			purego.RegisterLibFunc(&rt.evalBidirectional, lib, "ane_bridge_eval_bidirectional")
		}
		if hasBridgeSymbol(lib, "ane_bridge_signal_event_cpu") {
			purego.RegisterLibFunc(&rt.signalEventCPU, lib, "ane_bridge_signal_event_cpu")
			aneBridgeSignalEventCPU = rt.signalEventCPU
		}
		if hasBridgeSymbol(lib, "ane_bridge_wait_event_cpu") {
			purego.RegisterLibFunc(&rt.waitEventCPU, lib, "ane_bridge_wait_event_cpu")
			aneBridgeWaitEventCPU = rt.waitEventCPU
		}

		if rt.init() != 0 {
			bridgeLoadErr = fmt.Errorf("ane_bridge_init failed")
			return
		}
		bridgeRuntime = rt
	})
	if bridgeLoadErr != nil {
		return nil, bridgeLoadErr
	}
	return bridgeRuntime, nil
}

func hasBridgeSymbol(lib uintptr, name string) bool {
	_, err := purego.Dlsym(lib, name)
	return err == nil
}
