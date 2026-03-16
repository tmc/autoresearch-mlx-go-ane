//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework IOSurface -framework Metal -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <objc/runtime.h>
#include <objc/message.h>

typedef struct {
	uintptr_t shared_event;
	uintptr_t signal_event;
	uintptr_t signal_array;
	uintptr_t shared_wrapper;
	uintptr_t completion_handler;
	uint32_t event_port;
	char* error_msg;
} ane_shared_graph_result;

typedef struct {
	uintptr_t shared_event;
	uintptr_t wait_event;
	uintptr_t wait_array;
	uintptr_t shared_wrapper;
	uintptr_t completion_handler;
	uint32_t event_port;
	char* error_msg;
} ane_wait_graph_result;

typedef struct {
	uintptr_t wait_shared_event;
	uintptr_t wait_event;
	uintptr_t wait_array;
	uint32_t wait_event_port;
	uintptr_t signal_shared_event;
	uintptr_t signal_event;
	uintptr_t signal_array;
	uint32_t signal_event_port;
	uintptr_t shared_wrapper;
	uintptr_t completion_handler;
	char* error_msg;
} ane_wait_signal_graph_result;

static char* ane_copy_err(const char* msg) {
	if (msg == NULL) {
		return NULL;
	}
	size_t n = strlen(msg);
	char* out = (char*)malloc(n + 1);
	if (out == NULL) {
		return NULL;
	}
	memcpy(out, msg, n + 1);
	return out;
}

static char* ane_copy_nsstring(NSString* s) {
	if (s == nil) {
		return NULL;
	}
	const char* c = [s UTF8String];
	if (c == NULL) {
		return ane_copy_err("objc error with empty UTF8 string");
	}
	return ane_copy_err(c);
}

static uintptr_t ane_objc_attach_completion_handler(id request, char** error_out) {
	if (error_out != NULL) {
		*error_out = NULL;
	}
	if (request == nil) {
		if (error_out != NULL) {
			*error_out = ane_copy_err("nil request for completion handler");
		}
		return 0;
	}
	SEL setSel = sel_registerName("setCompletionHandler:");
	if (![request respondsToSelector:setSel]) {
		return 0;
	}
	void (^completionBlock)(BOOL, NSError *) = ^(BOOL success, NSError *error) {
		(void)success;
		(void)error;
	};
	id completionObj = [completionBlock copy];
	if (completionObj == nil) {
		if (error_out != NULL) {
			*error_out = ane_copy_err("completion block copy failed");
		}
		return 0;
	}
	((void (*)(id, SEL, id))objc_msgSend)(request, setSel, completionObj);
	return (uintptr_t)[completionObj retain];
}

static ane_shared_graph_result ane_objc_attach_shared_graph(uintptr_t request_id, uint32_t symbol_index, uint64_t signal_value) {
	ane_shared_graph_result out = {0};
	@autoreleasepool {
		id request = (id)request_id;
		if (request == nil) {
			out.error_msg = ane_copy_err("nil request");
			return out;
		}

		Class evClass = objc_getClass("IOSurfaceSharedEvent");
		Class sigClass = objc_getClass("_ANESharedSignalEvent");
		Class bundleClass = objc_getClass("_ANESharedEvents");
		if (evClass == Nil || sigClass == Nil || bundleClass == Nil) {
			out.error_msg = ane_copy_err("required class missing");
			return out;
		}

		id sharedEvent = ((id (*)(id, SEL))objc_msgSend)((id)evClass, sel_registerName("alloc"));
		if (sharedEvent == nil) {
			out.error_msg = ane_copy_err("IOSurfaceSharedEvent alloc failed");
			return out;
		}
		if ([sharedEvent respondsToSelector:sel_registerName("initWithOptions:")]) {
			sharedEvent = ((id (*)(id, SEL, uint64_t))objc_msgSend)(sharedEvent, sel_registerName("initWithOptions:"), (uint64_t)0);
		} else {
			sharedEvent = ((id (*)(id, SEL))objc_msgSend)(sharedEvent, sel_registerName("init"));
		}
		if (sharedEvent == nil) {
			out.error_msg = ane_copy_err("IOSurfaceSharedEvent init failed");
			return out;
		}
		if ([sharedEvent respondsToSelector:sel_registerName("eventPort")]) {
			uint32_t port = ((uint32_t (*)(id, SEL))objc_msgSend)(sharedEvent, sel_registerName("eventPort"));
			if (port == 0) {
				out.error_msg = ane_copy_err("IOSurfaceSharedEvent eventPort is zero");
				return out;
			}
			out.event_port = port;
		}

		id sig = ((id (*)(id, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
			(id)sigClass,
			sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:"),
			signal_value,
			symbol_index,
			(int64_t)5,
			sharedEvent
		);
		if (sig == nil) {
			out.error_msg = ane_copy_err("_ANESharedSignalEvent factory failed");
			return out;
		}

		id signalArray = [NSArray arrayWithObject:sig];
		id waitArray = [NSArray array];
		id wrapper = ((id (*)(id, SEL, id, id))objc_msgSend)(
			(id)bundleClass,
			sel_registerName("sharedEventsWithSignalEvents:waitEvents:"),
			signalArray,
			waitArray
		);
		if (wrapper == nil) {
			out.error_msg = ane_copy_err("_ANESharedEvents factory failed");
			return out;
		}

		((void (*)(id, SEL, id))objc_msgSend)(request, sel_registerName("setSharedEvents:"), wrapper);
		out.completion_handler = ane_objc_attach_completion_handler(request, &out.error_msg);
		if (out.error_msg != NULL) {
			return out;
		}

		out.shared_event = (uintptr_t)[sharedEvent retain];
		out.signal_event = (uintptr_t)[sig retain];
		out.signal_array = (uintptr_t)[signalArray retain];
		out.shared_wrapper = (uintptr_t)[wrapper retain];
	}
	return out;
}

static ane_wait_graph_result ane_objc_attach_wait_graph(uintptr_t request_id, uint64_t wait_value) {
	ane_wait_graph_result out = {0};
	@autoreleasepool {
		id request = (id)request_id;
		if (request == nil) {
			out.error_msg = ane_copy_err("nil request");
			return out;
		}

		Class evClass = objc_getClass("IOSurfaceSharedEvent");
		Class waitClass = objc_getClass("_ANESharedWaitEvent");
		Class bundleClass = objc_getClass("_ANESharedEvents");
		if (evClass == Nil || waitClass == Nil || bundleClass == Nil) {
			out.error_msg = ane_copy_err("required class missing");
			return out;
		}

		id sharedEvent = ((id (*)(id, SEL))objc_msgSend)((id)evClass, sel_registerName("alloc"));
		if (sharedEvent == nil) {
			out.error_msg = ane_copy_err("IOSurfaceSharedEvent alloc failed");
			return out;
		}
		if ([sharedEvent respondsToSelector:sel_registerName("initWithOptions:")]) {
			sharedEvent = ((id (*)(id, SEL, uint64_t))objc_msgSend)(sharedEvent, sel_registerName("initWithOptions:"), (uint64_t)0);
		} else {
			sharedEvent = ((id (*)(id, SEL))objc_msgSend)(sharedEvent, sel_registerName("init"));
		}
		if (sharedEvent == nil) {
			out.error_msg = ane_copy_err("IOSurfaceSharedEvent init failed");
			return out;
		}

		uint32_t port = 0;
		if ([sharedEvent respondsToSelector:sel_registerName("eventPort")]) {
			port = ((uint32_t (*)(id, SEL))objc_msgSend)(sharedEvent, sel_registerName("eventPort"));
		}
		if (port == 0) {
			out.error_msg = ane_copy_err("IOSurfaceSharedEvent eventPort is zero");
			return out;
		}

		id wait = ((id (*)(id, SEL, uint64_t, id))objc_msgSend)(
			(id)waitClass,
			sel_registerName("waitEventWithValue:sharedEvent:"),
			wait_value,
			sharedEvent
		);
		if (wait == nil) {
			out.error_msg = ane_copy_err("_ANESharedWaitEvent factory failed");
			return out;
		}

		id signalArray = [NSArray array];
		id waitArray = [NSArray arrayWithObject:wait];
		id wrapper = ((id (*)(id, SEL, id, id))objc_msgSend)(
			(id)bundleClass,
			sel_registerName("sharedEventsWithSignalEvents:waitEvents:"),
			signalArray,
			waitArray
		);
		if (wrapper == nil) {
			out.error_msg = ane_copy_err("_ANESharedEvents factory failed");
			return out;
		}

		((void (*)(id, SEL, id))objc_msgSend)(request, sel_registerName("setSharedEvents:"), wrapper);
		out.completion_handler = ane_objc_attach_completion_handler(request, &out.error_msg);
		if (out.error_msg != NULL) {
			return out;
		}

		out.shared_event = (uintptr_t)[sharedEvent retain];
		out.wait_event = (uintptr_t)[wait retain];
		out.wait_array = (uintptr_t)[waitArray retain];
		out.shared_wrapper = (uintptr_t)[wrapper retain];
		out.event_port = port;
	}
	return out;
}

static ane_wait_signal_graph_result ane_objc_attach_wait_signal_graph(
	uintptr_t request_id,
	uint64_t wait_value,
	uint32_t signal_symbol_index,
	uint64_t signal_value
) {
	ane_wait_signal_graph_result out = {0};
	@autoreleasepool {
		id request = (id)request_id;
		if (request == nil) {
			out.error_msg = ane_copy_err("nil request");
			return out;
		}

		Class evClass = objc_getClass("IOSurfaceSharedEvent");
		Class waitClass = objc_getClass("_ANESharedWaitEvent");
		Class sigClass = objc_getClass("_ANESharedSignalEvent");
		Class bundleClass = objc_getClass("_ANESharedEvents");
		if (evClass == Nil || waitClass == Nil || sigClass == Nil || bundleClass == Nil) {
			out.error_msg = ane_copy_err("required class missing");
			return out;
		}

		id waitShared = ((id (*)(id, SEL))objc_msgSend)((id)evClass, sel_registerName("alloc"));
		id signalShared = ((id (*)(id, SEL))objc_msgSend)((id)evClass, sel_registerName("alloc"));
		if (waitShared == nil || signalShared == nil) {
			out.error_msg = ane_copy_err("IOSurfaceSharedEvent alloc failed");
			return out;
		}
		SEL initWithOptionsSel = sel_registerName("initWithOptions:");
		SEL initSel = sel_registerName("init");
		if ([waitShared respondsToSelector:initWithOptionsSel]) {
			waitShared = ((id (*)(id, SEL, uint64_t))objc_msgSend)(waitShared, initWithOptionsSel, (uint64_t)0);
		} else {
			waitShared = ((id (*)(id, SEL))objc_msgSend)(waitShared, initSel);
		}
		if ([signalShared respondsToSelector:initWithOptionsSel]) {
			signalShared = ((id (*)(id, SEL, uint64_t))objc_msgSend)(signalShared, initWithOptionsSel, (uint64_t)0);
		} else {
			signalShared = ((id (*)(id, SEL))objc_msgSend)(signalShared, initSel);
		}
		if (waitShared == nil || signalShared == nil) {
			out.error_msg = ane_copy_err("IOSurfaceSharedEvent init failed");
			return out;
		}

		SEL eventPortSel = sel_registerName("eventPort");
		uint32_t waitPort = [waitShared respondsToSelector:eventPortSel]
			? ((uint32_t (*)(id, SEL))objc_msgSend)(waitShared, eventPortSel)
			: 0;
		uint32_t signalPort = [signalShared respondsToSelector:eventPortSel]
			? ((uint32_t (*)(id, SEL))objc_msgSend)(signalShared, eventPortSel)
			: 0;
		if (waitPort == 0 || signalPort == 0) {
			out.error_msg = ane_copy_err("shared event port is zero");
			return out;
		}

		id wait = ((id (*)(id, SEL, uint64_t, id))objc_msgSend)(
			(id)waitClass,
			sel_registerName("waitEventWithValue:sharedEvent:"),
			wait_value,
			waitShared
		);
		if (wait == nil) {
			out.error_msg = ane_copy_err("_ANESharedWaitEvent factory failed");
			return out;
		}
		id sig = ((id (*)(id, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
			(id)sigClass,
			sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:"),
			signal_value,
			signal_symbol_index,
			(int64_t)5,
			signalShared
		);
		if (sig == nil) {
			out.error_msg = ane_copy_err("_ANESharedSignalEvent factory failed");
			return out;
		}

		id signalArray = [NSArray arrayWithObject:sig];
		id waitArray = [NSArray arrayWithObject:wait];
		id wrapper = ((id (*)(id, SEL, id, id))objc_msgSend)(
			(id)bundleClass,
			sel_registerName("sharedEventsWithSignalEvents:waitEvents:"),
			signalArray,
			waitArray
		);
		if (wrapper == nil) {
			out.error_msg = ane_copy_err("_ANESharedEvents factory failed");
			return out;
		}

		((void (*)(id, SEL, id))objc_msgSend)(request, sel_registerName("setSharedEvents:"), wrapper);
		out.completion_handler = ane_objc_attach_completion_handler(request, &out.error_msg);
		if (out.error_msg != NULL) {
			return out;
		}

		out.wait_shared_event = (uintptr_t)[waitShared retain];
		out.wait_event = (uintptr_t)[wait retain];
		out.wait_array = (uintptr_t)[waitArray retain];
		out.wait_event_port = waitPort;
		out.signal_shared_event = (uintptr_t)[signalShared retain];
		out.signal_event = (uintptr_t)[sig retain];
		out.signal_array = (uintptr_t)[signalArray retain];
		out.signal_event_port = signalPort;
		out.shared_wrapper = (uintptr_t)[wrapper retain];
	}
	return out;
}

static int ane_objc_set_shared_event_value(uintptr_t shared_event_id, uint64_t value, char** error_out) {
	if (error_out != NULL) {
		*error_out = NULL;
	}
	@autoreleasepool {
		id sharedEvent = (id)shared_event_id;
		if (sharedEvent == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("nil shared event");
			}
			return 2;
		}
		SEL sel = sel_registerName("setSignaledValue:");
		if (![sharedEvent respondsToSelector:sel]) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("setSignaledValue: unavailable");
			}
			return 3;
		}
		((void (*)(id, SEL, uint64_t))objc_msgSend)(sharedEvent, sel, value);
		return 0;
	}
}

static int ane_objc_evaluate(
	uintptr_t client_id,
	uintptr_t model_id,
	uintptr_t options_id,
	uintptr_t request_id,
	uint32_t qos,
	int direct,
	char** error_out
) {
	if (error_out != NULL) {
		*error_out = NULL;
	}
	@autoreleasepool {
		id client = (id)client_id;
		id model = (id)model_id;
		id options = (id)options_id;
		id request = (id)request_id;
		if (client == nil || model == nil || request == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("nil evaluate argument");
			}
			return 2;
		}
		SEL sel = direct
			? sel_registerName("doEvaluateDirectWithModel:options:request:qos:error:")
			: sel_registerName("evaluateWithModel:options:request:qos:error:");
		if (![client respondsToSelector:sel]) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("evaluate selector unavailable");
			}
			return 3;
		}
		NSError* err = nil;
		BOOL ok = ((BOOL (*)(id, SEL, id, id, id, uint32_t, NSError**))objc_msgSend)(
			client, sel, model, options, request, qos, &err
		);
		if (ok) {
			return 0;
		}
		if (error_out != NULL) {
			if (err != nil) {
				*error_out = ane_copy_nsstring([err description]);
			} else {
				*error_out = ane_copy_err("objc returned NO with nil NSError");
			}
		}
		return 1;
	}
}

static int ane_objc_wait_for_metal_shared_event(
	uint32_t event_port,
	uint64_t value,
	uint64_t timeout_ms,
	char** error_out
) {
	if (error_out != NULL) {
		*error_out = NULL;
	}
	@autoreleasepool {
		if (event_port == 0) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("eventPort is zero");
			}
			return 2;
		}
		id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
		if (dev == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("MTLCreateSystemDefaultDevice returned nil");
			}
			return 3;
		}
		SEL newSharedSel = sel_registerName("newSharedEventWithMachPort:");
		if (![dev respondsToSelector:newSharedSel]) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("newSharedEventWithMachPort: unavailable");
			}
			return 4;
		}
		id sharedEvent = ((id (*)(id, SEL, uint32_t))objc_msgSend)(dev, newSharedSel, event_port);
		if (sharedEvent == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("newSharedEventWithMachPort returned nil");
			}
			return 5;
		}
		id<MTLCommandQueue> queue = [dev newCommandQueue];
		if (queue == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("newCommandQueue returned nil");
			}
			return 6;
		}
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		if (cb == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("commandBuffer returned nil");
			}
			return 7;
		}
		SEL waitSel = sel_registerName("encodeWaitForEvent:value:");
		if (![cb respondsToSelector:waitSel]) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("encodeWaitForEvent:value: unavailable");
			}
			return 8;
		}
		((void (*)(id, SEL, id, uint64_t))objc_msgSend)(cb, waitSel, sharedEvent, value);

		dispatch_semaphore_t sem = dispatch_semaphore_create(0);
		[cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull unused) {
			(void)unused;
			dispatch_semaphore_signal(sem);
		}];
		[cb commit];

		uint64_t ms = timeout_ms == 0 ? 5000 : timeout_ms;
		dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(ms * (uint64_t)NSEC_PER_MSEC));
		long waited = dispatch_semaphore_wait(sem, deadline);
		if (waited != 0) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("metal shared-event wait timed out");
			}
			return 9;
		}
		if (cb.status == MTLCommandBufferStatusError) {
			if (error_out != NULL) {
				NSError *err = cb.error;
				if (err != nil) {
					*error_out = ane_copy_nsstring([err description]);
				} else {
					*error_out = ane_copy_err("metal command buffer status=error");
				}
			}
			return 10;
		}
		return 0;
	}
}

static int ane_objc_signal_metal_shared_event(
	uint32_t event_port,
	uint64_t value,
	uint64_t timeout_ms,
	char** error_out
) {
	if (error_out != NULL) {
		*error_out = NULL;
	}
	@autoreleasepool {
		if (event_port == 0) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("eventPort is zero");
			}
			return 2;
		}
		id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
		if (dev == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("MTLCreateSystemDefaultDevice returned nil");
			}
			return 3;
		}
		SEL newSharedSel = sel_registerName("newSharedEventWithMachPort:");
		if (![dev respondsToSelector:newSharedSel]) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("newSharedEventWithMachPort: unavailable");
			}
			return 4;
		}
		id sharedEvent = ((id (*)(id, SEL, uint32_t))objc_msgSend)(dev, newSharedSel, event_port);
		if (sharedEvent == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("newSharedEventWithMachPort returned nil");
			}
			return 5;
		}
		id<MTLCommandQueue> queue = [dev newCommandQueue];
		if (queue == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("newCommandQueue returned nil");
			}
			return 6;
		}
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		if (cb == nil) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("commandBuffer returned nil");
			}
			return 7;
		}
		SEL signalSel = sel_registerName("encodeSignalEvent:value:");
		if (![cb respondsToSelector:signalSel]) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("encodeSignalEvent:value: unavailable");
			}
			return 8;
		}
		((void (*)(id, SEL, id, uint64_t))objc_msgSend)(cb, signalSel, sharedEvent, value);

		dispatch_semaphore_t sem = dispatch_semaphore_create(0);
		[cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull unused) {
			(void)unused;
			dispatch_semaphore_signal(sem);
		}];
		[cb commit];

		uint64_t ms = timeout_ms == 0 ? 5000 : timeout_ms;
		dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(ms * (uint64_t)NSEC_PER_MSEC));
		long waited = dispatch_semaphore_wait(sem, deadline);
		if (waited != 0) {
			if (error_out != NULL) {
				*error_out = ane_copy_err("metal shared-event signal timed out");
			}
			return 9;
		}
		if (cb.status == MTLCommandBufferStatusError) {
			if (error_out != NULL) {
				NSError *err = cb.error;
				if (err != nil) {
					*error_out = ane_copy_nsstring([err description]);
				} else {
					*error_out = ane_copy_err("metal command buffer status=error");
				}
			}
			return 10;
		}
		return 0;
	}
}

static void ane_objc_release(uintptr_t obj_id) {
	@autoreleasepool {
		id obj = (id)obj_id;
		if (obj != nil) {
			[obj release];
		}
	}
}

static void ane_objc_free_cstr(char* s) {
	if (s != NULL) {
		free(s);
	}
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/tmc/apple/objc"
)

type objcShimWaitGraph struct {
	Retained    []objc.ID
	EventPort   uint32
	SharedEvent objc.ID
}

type objcShimSignalGraph struct {
	Retained    []objc.ID
	EventPort   uint32
	SharedEvent objc.ID
}

type objcShimWaitSignalGraph struct {
	Retained          []objc.ID
	WaitEventPort     uint32
	WaitSharedEvent   objc.ID
	SignalEventPort   uint32
	SignalSharedEvent objc.ID
}

func objcShimAttachSharedEvents(request objc.ID, symbolIndex uint32) ([]objc.ID, error) {
	graph, err := objcShimAttachSignalEvents(request, symbolIndex, 1)
	if err != nil {
		return nil, err
	}
	return graph.Retained, nil
}

func objcShimAttachSignalEvents(request objc.ID, symbolIndex uint32, signalValue uint64) (objcShimSignalGraph, error) {
	res := C.ane_objc_attach_shared_graph(
		C.uintptr_t(request),
		C.uint32_t(symbolIndex),
		C.uint64_t(signalValue),
	)
	if res.error_msg != nil {
		msg := C.GoString(res.error_msg)
		C.ane_objc_free_cstr(res.error_msg)
		return objcShimSignalGraph{}, fmt.Errorf("objc shared shim: %s", msg)
	}
	ids := []objc.ID{
		objc.ID(res.shared_event),
		objc.ID(res.signal_event),
		objc.ID(res.signal_array),
		objc.ID(res.shared_wrapper),
		objc.ID(res.completion_handler),
	}
	out := ids[:0]
	for _, id := range ids {
		if id != 0 {
			out = append(out, id)
		}
	}
	if len(out) == 0 {
		return objcShimSignalGraph{}, fmt.Errorf("objc shared shim: no objects returned")
	}
	if res.event_port == 0 {
		objcShimRelease(out)
		return objcShimSignalGraph{}, fmt.Errorf("objc shared shim: eventPort is zero")
	}
	return objcShimSignalGraph{
		Retained:    out,
		EventPort:   uint32(res.event_port),
		SharedEvent: objc.ID(res.shared_event),
	}, nil
}

func objcShimAttachWaitEvents(request objc.ID, waitValue uint64) (objcShimWaitGraph, error) {
	res := C.ane_objc_attach_wait_graph(C.uintptr_t(request), C.uint64_t(waitValue))
	if res.error_msg != nil {
		msg := C.GoString(res.error_msg)
		C.ane_objc_free_cstr(res.error_msg)
		return objcShimWaitGraph{}, fmt.Errorf("objc wait shim: %s", msg)
	}
	ids := []objc.ID{
		objc.ID(res.shared_event),
		objc.ID(res.wait_event),
		objc.ID(res.wait_array),
		objc.ID(res.shared_wrapper),
		objc.ID(res.completion_handler),
	}
	out := ids[:0]
	for _, id := range ids {
		if id != 0 {
			out = append(out, id)
		}
	}
	if len(out) == 0 {
		return objcShimWaitGraph{}, fmt.Errorf("objc wait shim: no objects returned")
	}
	if res.event_port == 0 {
		objcShimRelease(out)
		return objcShimWaitGraph{}, fmt.Errorf("objc wait shim: eventPort is zero")
	}
	return objcShimWaitGraph{
		Retained:    out,
		EventPort:   uint32(res.event_port),
		SharedEvent: objc.ID(res.shared_event),
	}, nil
}

func objcShimAttachWaitSignalEvents(
	request objc.ID,
	waitValue uint64,
	signalSymbolIndex uint32,
	signalValue uint64,
) (objcShimWaitSignalGraph, error) {
	res := C.ane_objc_attach_wait_signal_graph(
		C.uintptr_t(request),
		C.uint64_t(waitValue),
		C.uint32_t(signalSymbolIndex),
		C.uint64_t(signalValue),
	)
	if res.error_msg != nil {
		msg := C.GoString(res.error_msg)
		C.ane_objc_free_cstr(res.error_msg)
		return objcShimWaitSignalGraph{}, fmt.Errorf("objc wait-signal shim: %s", msg)
	}
	ids := []objc.ID{
		objc.ID(res.wait_shared_event),
		objc.ID(res.wait_event),
		objc.ID(res.wait_array),
		objc.ID(res.signal_shared_event),
		objc.ID(res.signal_event),
		objc.ID(res.signal_array),
		objc.ID(res.shared_wrapper),
		objc.ID(res.completion_handler),
	}
	out := ids[:0]
	for _, id := range ids {
		if id != 0 {
			out = append(out, id)
		}
	}
	if len(out) == 0 {
		return objcShimWaitSignalGraph{}, fmt.Errorf("objc wait-signal shim: no objects returned")
	}
	if res.wait_event_port == 0 || res.signal_event_port == 0 {
		objcShimRelease(out)
		return objcShimWaitSignalGraph{}, fmt.Errorf(
			"objc wait-signal shim: invalid event ports wait=%d signal=%d",
			uint32(res.wait_event_port),
			uint32(res.signal_event_port),
		)
	}
	return objcShimWaitSignalGraph{
		Retained:          out,
		WaitEventPort:     uint32(res.wait_event_port),
		WaitSharedEvent:   objc.ID(res.wait_shared_event),
		SignalEventPort:   uint32(res.signal_event_port),
		SignalSharedEvent: objc.ID(res.signal_shared_event),
	}, nil
}

func objcShimEvaluate(client, model, options, request objc.ID, qos uint32, direct bool) error {
	var errMsg *C.char
	d := C.int(0)
	if direct {
		d = 1
	}
	rc := C.ane_objc_evaluate(
		C.uintptr_t(client),
		C.uintptr_t(model),
		C.uintptr_t(options),
		C.uintptr_t(request),
		C.uint32_t(qos),
		d,
		(**C.char)(unsafe.Pointer(&errMsg)),
	)
	if rc == 0 {
		return nil
	}
	msg := ""
	if errMsg != nil {
		msg = C.GoString(errMsg)
		C.ane_objc_free_cstr(errMsg)
	}
	if msg == "" {
		msg = fmt.Sprintf("rc=%d", int(rc))
	}
	return fmt.Errorf("objc evaluate shim failed: %s", msg)
}

func objcShimSetSharedEventValue(sharedEvent objc.ID, value uint64) error {
	var errMsg *C.char
	rc := C.ane_objc_set_shared_event_value(
		C.uintptr_t(sharedEvent),
		C.uint64_t(value),
		(**C.char)(unsafe.Pointer(&errMsg)),
	)
	if rc == 0 {
		return nil
	}
	msg := ""
	if errMsg != nil {
		msg = C.GoString(errMsg)
		C.ane_objc_free_cstr(errMsg)
	}
	if msg == "" {
		msg = fmt.Sprintf("rc=%d", int(rc))
	}
	return fmt.Errorf("objc shared event set value failed: %s", msg)
}

func objcShimMetalWaitSharedEvent(eventPort uint32, value uint64, timeoutMS uint64) error {
	var errMsg *C.char
	rc := C.ane_objc_wait_for_metal_shared_event(
		C.uint32_t(eventPort),
		C.uint64_t(value),
		C.uint64_t(timeoutMS),
		(**C.char)(unsafe.Pointer(&errMsg)),
	)
	if rc == 0 {
		return nil
	}
	msg := ""
	if errMsg != nil {
		msg = C.GoString(errMsg)
		C.ane_objc_free_cstr(errMsg)
	}
	if msg == "" {
		msg = fmt.Sprintf("rc=%d", int(rc))
	}
	return fmt.Errorf("objc metal wait failed: %s", msg)
}

func objcShimMetalSignalSharedEvent(eventPort uint32, value uint64, timeoutMS uint64) error {
	var errMsg *C.char
	rc := C.ane_objc_signal_metal_shared_event(
		C.uint32_t(eventPort),
		C.uint64_t(value),
		C.uint64_t(timeoutMS),
		(**C.char)(unsafe.Pointer(&errMsg)),
	)
	if rc == 0 {
		return nil
	}
	msg := ""
	if errMsg != nil {
		msg = C.GoString(errMsg)
		C.ane_objc_free_cstr(errMsg)
	}
	if msg == "" {
		msg = fmt.Sprintf("rc=%d", int(rc))
	}
	return fmt.Errorf("objc metal signal failed: %s", msg)
}

func objcShimRelease(ids []objc.ID) {
	for i := len(ids) - 1; i >= 0; i-- {
		if ids[i] == 0 {
			continue
		}
		C.ane_objc_release(C.uintptr_t(ids[i]))
	}
}
