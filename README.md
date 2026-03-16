# mlx-go-ane (experiment)

`mlxgoane` is an experimental package for ANE-oriented execution paths in
`mlx-go`.

Treat it as a standalone library with three primary seams:

- **Model assessment**: classify whether a model can reuse existing ANE decode
  paths or needs a model-specific backend.
- **Block runtimes**: build and run draft/decode stages without threading ANE
  details through command code.
- **IOSurface/shared-event data plane**: reuse plan-owned IOSurfaces,
  Metal-shared events, and reusable Metal buffer bindings for zero-copy handoff.

The model-assessment and routing helpers build everywhere. The block runtimes
and IOSurface/shared-event data plane are Darwin-only and currently require the
private-ANE build tags used in this repo.

This package is intentionally narrow and not API-stable.

Do not start a serious model integration at `InstallNNLinearHook` or
`Runtime.Linear`. Those APIs are useful for experiments and narrow kernels, but
the production path in this repo is now block-level decode or speculative
drafting. If you are integrating a new model family, start by classifying its
decode semantics and choose the integration point from there.

For implementation details and the phased integration roadmap, see
[`DESIGN.md`](./DESIGN.md).

## Preferred extension points

For downstream `mlx-go` code, the clean extension points are:

- `AssessDecodeModel`: decide whether to reuse the current stock draft/decode
  paths or build a model-specific backend.
- `ANEDraftModel`: reusable speculative-draft runtime.
- `SurfaceDecodeFFN`: reusable dense stage offload when a caller only needs a
  block-level FFN stage.
- `NanochatDecodeRuntime`: model-specific decode runtime for nanochat-style
  semantics.
- `SurfaceEvalPlan` / `MultiSurfaceEvalPlan`: reusable mapped request + surface
  ownership for callers that already own the model/runtime layer.
- `SurfaceSyncRuntime`: the narrow runtime interface for zero-copy output plus
  shared-event import (`ANEDraftModel`, `SurfaceDecodeFFN`, and
  `NanochatDecodeRuntime` all satisfy it).

In `mlx-go` itself, the intended ownership is:

- model packages own model-specific ANE backends
- `internal/specdec` owns shared speculative-draft construction and policy
- command packages only select policy, expose flags, and report metrics

When a caller only needs the synchronized output plane, depend on the narrow
runtime interface instead of a concrete type:

```go
func consumeStage(rt mlxgoane.SurfaceSyncRuntime) error {
	buf, err := rt.NewDefaultOutputMetalBufferBinding()
	if err != nil {
		return err
	}
	defer buf.Close()

	wait, err := rt.NewDefaultWaitMetalSharedEvent()
	if err != nil {
		return err
	}
	defer wait.Close()

	signal, err := rt.NewDefaultSignalMetalSharedEvent()
	if err != nil {
		return err
	}
	defer signal.Close()

	return nil
}
```

## Status

Implemented now:

- `AnalyzeArchive`, `AnalyzeTextFile`, and `LowerToMIL` coverage checks.
- `Runtime.Linear` with strict rank/dtype/shape validation.
- Fallback to native MLX matmul when ANE is unavailable or returns an error.
- Hook install/restore plumbing for `mlx/nn.Linear`.
- Darwin adapter prototype path for linear:
  - shape-specialized MIL template generation with baked `BLOBFILE` weights
  - in-memory model compile/load calls
  - descriptor temp-path mirroring (`<tmp>/<hexStringIdentifier>/...`) before compile
  - IOSurface request assembly and evaluate/map/unmap calls (runtime input: `x` only)
  - shape-keyed model cache in the executor
  - basic compile/load/evaluate timing telemetry (internal executor metrics)
- FFN training primitives:
  - `gen_ffn_bwd`-equivalent MIL generator (`ffnBwdMILText`)
  - backward transposed-weight blob packing (`W2t`, `W1t`, `W3t`)
  - tap split/pack helpers matching `stories_mil.h` channel offsets
  - `TrainFFNOneStepSGD` for one FFN-only ANE forward/backward + CPU SGD update
  - gradient-accumulated optimizer steps (`TrainFFNAccumSGD`,
    `TrainFFNAccumAdamW`)
- Optional linear route policy:
  - `LinearRouter` with deterministic MLX fallback guards for
    spatial/channel constraints and compile-budget misses
- Decode-model assessment helpers:
  - `DecodeModelSpec`
  - `AssessDecodeModel`
  - deterministic classification of current draft/decode path reuse
- Env-gated Darwin integration test for direct ANE linear execution and
  numerical validation.
- Standalone runtime/data-plane primitives:
  - `ANEDraftModel`
  - `SurfaceDecodeFFN`
  - `NanochatDecodeRuntime`
  - `SurfaceEvalPlan`
  - `MultiSurfaceEvalPlan`
  - `SurfaceSyncRuntime`
  - `SharedEvent`, `MetalSharedEvent`, and `MetalBufferBinding`

Not implemented yet:

- Host-validated robust end-to-end execution across OS/chip variants.
- Stable model template set beyond the initial linear template.
- Cache eviction and lifecycle policies for compiled model objects.
- General graph lowering/execution beyond linear.
- Generic reuse of current draft/decode templates for models with custom
  attention or MLP semantics. Those still need model-specific backends.
- A stable public package split between reusable runtime code and probe-only
  generators/tests. Today those still live together under `experiment/mlx-go-ane`.

## API surface

- `AnalyzeArchive(ar *mlxfntxt.Archive) (*Report, error)`
- `AnalyzeTextFile(path string) (*Report, error)`
- `LowerToMIL(ar *mlxfntxt.Archive) error`
- `NewRuntime(executor LinearExecutor) *Runtime`
- `NewRuntimeWithOptions(opts RuntimeOptions) *Runtime`
- `(*Runtime).Linear(ctx context.Context, x, w *mlx.Array) (*LinearResult, error)`
- `DefaultLinearRouteConfig() LinearRouteConfig`
- `LinearRouteConfigForProfile(profile LinearRouteProfile) LinearRouteConfig`
- `NewLinearRouter(cfg LinearRouteConfig) *LinearRouter`
- `(*LinearRouter).DecideLinear(in LinearRouteInput) RouteDecision`
- `InstallNNLinearHook(rt *Runtime) (restore func())`
- `NewApplePrivateExecutor() (LinearExecutor, error)`
- `TrainFFNOneStepSGD(...) (*FFNTrainStepResult, error)`
- `TrainFFNOneStepAdamW(...) (*FFNTrainStepResult, error)`
- `AssessDecodeModel(spec DecodeModelSpec) IntegrationAssessment`

Supporting types:

- `LinearExecutor` (ANE-compatible linear kernel interface)
- `RuntimeOptions` / `LinearRouteProfile`
- `FFNExecutor` (optional fused FFN forward/backward kernel interface)
- `DecodeModelSpec` / `IntegrationAssessment` / `IntegrationPath`
- `SurfaceOutputRuntime` / `SharedEventRuntime` / `SurfaceSyncRuntime`
- `IOSurfaceFloat32` / `IOSurfaceReadOnlyView`
- `SharedEvent`
- `MetalSharedEvent` / `MetalBuffer` / `MetalBufferBinding`

The runtime/data-plane types above are available only on the Darwin private-ANE
build (`-tags ane_appleneuralengine`).
- `LinearResult` (`Y`, `Backend`, `FallbackReason`)
- `FFNForwardTaps` / `FFNBackwardOutput` / `FFNTrainStepResult`
- `AdamWConfig` / `AdamWState` / `FFNAdamWState`
- `Report` / `Step` (lowering coverage report)

## Choosing an integration point

For end-to-end model integration, classify the decode contract first:

```go
spec := mlxgoane.DecodeModelSpec{
	ModelDim:   768,
	NumHeads:   6,
	NumKVHeads: 6,
	HeadDim:    128,
	FFNDim:     3072,
	MLPKind:    mlxgoane.MLPKindReluSquared,

	UsesResidualMix:     true,
	UsesQKNorm:          true,
	UsesValueEmbeddings: true,
	UsesSlidingWindow:   true,
}

assessment := mlxgoane.AssessDecodeModel(spec)
fmt.Println(assessment.RecommendedPath)
fmt.Println(assessment.Reasons)
```

Current recommendations:

- `speculative_draft`: use the current MIL draft/decode path when the model
  matches the stock transformer contract used in `mlx-go-ane`.
- `surface_decode_ffn`: narrow FFN-stage reuse for compatible SwiGLU models.
- `model_specific_decode`: build a model-specific decode backend when the model
  adds semantics like residual mixing, QK norm, value embeddings, sliding-window
  KV, or a non-SwiGLU MLP.

This package is meant to stay standalone at that boundary:

- classify first with `AssessDecodeModel`
- choose one runtime family
- depend on the narrow runtime/data-plane interfaces instead of command-local
  helpers
- keep command packages as policy and reporting layers only

For nanochat specifically, there is now a dedicated Espresso decode probe
generator in `nanochat_decode_espresso_darwin.go`. Current probe results are
negative: even a stripped single-layer token-only trunk fails in
`_ANEEspressoIRTranslator`, so raw Espresso JSON is not the recommended
shipping path for nanochat VE/depth today.

The modelir/CoreMLTools bridge is a better fit, but it has a different current
boundary: widened nanochat variants such as `2-layer no-VE full-context` and
`1-layer VE full-context` now compile successfully to compiled NeuralNetwork
artifacts, and the private `_ANEModel` loader can execute that supported subset
through the layout-aware client-model path. The remaining work there is runtime
overhead and broader semantic coverage, not basic loader acceptance.

For models that perform real decode work across multiple block semantics,
`Runtime.Linear` and `InstallNNLinearHook` are intentionally the wrong first
integration point.

## IOSurface + SharedEvent primitives

`mlxgoane` exposes the same IOSurface-backed request and shared-event wiring
used by the speculative decode experiments. The intended consumption pattern is:

```go
plan, err := mlxgoane.NewSurfaceEvalPlan(model, inSurf, outSurf, cfg)
if err != nil {
	return err
}
defer plan.Close()

if wait := plan.WaitEvent(); wait != nil {
	if err := wait.SetSignaledValue(wait.Value()); err != nil {
		return err
	}
}

if err := plan.Eval(ctx); err != nil {
	return err
}

view, err := plan.Output().ReadOnlyView()
if err != nil {
	return err
}
defer view.Close()

ptr := view.Pointer()
_ = ptr // pass to Metal newBufferWithBytesNoCopy or another zero-copy consumer
```

For steady-state decode loops, prefer a reusable Metal buffer binding:

```go
buf, err := stage.NewDefaultOutputMetalBufferBinding()
if err != nil {
	return err
}
defer buf.Close()

if err := stage.EvalPreparedSurface(ctx); err != nil {
	return err
}
if err := buf.LockReadOnly(); err != nil {
	return err
}
// Encode Metal reads against buf.Buffer(), then unlock after GPU completion.
if err := buf.UnlockReadOnly(); err != nil {
	return err
}
```

And for shared-event import, prefer the runtime-level helpers over threading raw
ports into Metal call sites:

```go
wait, err := stage.NewDefaultWaitMetalSharedEvent()
if err != nil {
	return err
}
defer wait.Close()

signal, err := stage.NewDefaultSignalMetalSharedEvent()
if err != nil {
	return err
}
defer signal.Close()
```

Notes:

- `WaitEvent()` and `SignalEvent()` expose plan-owned `IOSurfaceSharedEvent`
  wrappers. Use them instead of threading raw Mach ports through new code.
- `NewDefaultWaitMetalSharedEvent` / `NewDefaultSignalMetalSharedEvent` import
  those plan-owned events into Metal once so command buffers can reuse them.
- Explicit event ports in `SurfaceEvalPlanConfig` are still unsupported in the
  bridgeless path. The plan creates and owns the shared-event graph.
- `ReadOnlyView` is a scoped lock helper over `LockReadOnly`/`UnlockReadOnly`.
- `IOSurfaceFloat32.NewMetalBufferNoCopy` creates a scoped no-copy Metal buffer
  and holds the IOSurface read lock until `Close`.
- `IOSurfaceFloat32.NewMetalBufferBinding` creates a reusable no-copy Metal
  buffer and leaves read-lock synchronization to explicit
  `LockReadOnly`/`UnlockReadOnly` calls. Prefer this for steady-state decode
  loops; per-step `newBufferWithBytesNoCopy` allocation is expensive.
- `SharedEvent.NewMetalSharedEvent` imports the Mach port into Metal as an
  `MTLSharedEvent`.
- `MLXGO_ANE_METAL_SYNC=off|wait|signal|wait_signal` enables shared-event wiring
  for the multi-surface decode paths that opt into `applySharedEventModeFromEnv`.

## Quick start: runtime + nn hook

```go
exec, err := mlxgoane.NewApplePrivateExecutor()
if err != nil {
	// adapter disabled, no ANE host support, or private API unavailable
	exec = nil
}

rt := mlxgoane.NewRuntime(exec) // fallback enabled by default

restore := mlxgoane.InstallNNLinearHook(rt)
defer restore()

// Existing nn.Linear.Forward calls now route through rt.Linear.
```

Fallback behavior:

- If `Executor` is nil: uses MLX matmul.
- If `Executor` returns error and `AllowFallback` is true: uses MLX matmul and sets `FallbackReason`.
- If `AllowFallback` is false: returns ANE error.
- If `Router` denies ANE for a call: falls back to MLX with
  `FallbackReason` prefixed by `router:`.

Route profiles:

- `balanced` (default policy): measured-safe baseline.
- `conservative`: stricter spatial/channel/cache gates.
- `aggressive`: looser channel/cache gates, still enforces `batch>=16`.
- `disabled`: bypasses route gating and always attempts ANE when executor exists.

Example:

```go
rt := mlxgoane.NewRuntimeWithOptions(mlxgoane.RuntimeOptions{
	Executor:           exec,
	LinearRouteProfile: mlxgoane.LinearRouteProfileConservative,
})
```

## Quick start: coverage analysis

```go
rep, err := mlxgoane.AnalyzeTextFile("model.mlxfntxt")
if err != nil {
	return err
}

fmt.Printf("ops=%d direct=%d lowered=%d unsupported=%d unknown=%d\n",
	rep.TotalOps, rep.DirectOps, rep.LoweredOps, rep.UnsupportedOps, rep.UnknownOps)
if rep.UnsupportedOps > 0 || rep.UnknownOps > 0 {
	// archive is not currently lowerable end-to-end
}
```

`LowerToMIL` is currently a gate for parsed archives: it checks coverage and fails if unsupported or unknown ops remain.

## Runtime contract

`Runtime.Linear` enforces:

- `x` is non-nil `float32` rank-2, shape `[batch, inDim]`
- `w` is non-nil `float32` rank-2, shape `[outDim, inDim]`
- output shape is `[batch, outDim]`

The ANE executor path converts MLX arrays to `[]float32`, calls `LinearExecutor.Linear`, validates output length, then materializes the result back into an MLX array.

For the Apple private adapter path, fallback to MLX remains the safety
behavior. The file-backed Espresso surface backend is now also used by the
flag-gated Qwen3.5 decode plane in `examples/mlx-go-lm`, where it has been
validated on a real local checkpoint. Treat that Espresso path as the primary
working decode-plane backend behind the flag. It is still not the default: it
needs more parity/perf coverage, better init caching, and tighter copy/sync
behavior before it should replace the stock MLX route automatically.

## FFN Training Step Contract

`TrainFFNOneStepSGD` expects ANE channel-major buffers for `[1,C,1,S]`:

- `x`: `[dim, seq]`
- `dffn`: `[dim, seq]`
- forward taps are packed as `concat(y,h1,h3,gate,xn)` with channel offsets:
  - `y@0`
  - `h1@dim`
  - `h3@dim+hidden`
  - `gate@dim+2*hidden`
  - `xn@dim+3*hidden`

It computes:

- ANE forward (`gen_ffn_fwd_taps`-style)
- ANE backward (`gen_ffn_bwd`-style)
- CPU `dW2`, `dW1`, `dW3` outer products
- in-place SGD updates to `w2`, `w1`, `w3`

## Measured findings (Darwin private adapter)

These findings were reproduced in this repo on 2026-03-03 and are worth
preserving because they are not documented by Apple private API docs.

- `options` argument must be an empty dictionary (`@{}`), not `nil`, for
  ANE compile/load/evaluate selectors used here.
  - Affected selectors:
    - `compileWithQoS:options:error:`
    - `loadWithQoS:options:error:`
    - `evaluateWithQoS:options:request:error:`
  - Passing `nil` for `options` produced:
    `_ANEEspressoIRTranslator : error Cannot load network .../model.espresso.net`.
- The current linear conv template path is shape-sensitive on this host.
  - Observed behavior: requests with `batch` (MIL spatial `S`) below 16 hit
    `Program IOSurfaces map failure (0x1D)`.
  - Observed passing range in this setup: `batch >= 16`.
  - Integration tests use `batch=16, inDim=64, outDim=64` to stay in the
    validated range.

Treat both as measured implementation constraints, not universal ANE rules for
all chips/OS versions.

## Build tags

- Default build: Apple private adapter is disabled and returns an error.
- Private adapter build (darwin only):

```bash
go test -tags ane_appleneuralengine ./experiment/mlx-go-ane
```

- Optional legacy bridge experiment hooks (bench/test-only):

```bash
go test -tags "ane_appleneuralengine ane_bridge" ./experiment/mlx-go-ane
```

Without `ane_bridge`, bridge runtime loading is intentionally unavailable.

- Run hardware integration test (darwin host with ANE):

```bash
MLXGO_ANE_INTEGRATION=1 go test -tags ane_appleneuralengine ./experiment/mlx-go-ane -run TestApplePrivateExecutorLinearIntegration
```

- Run FFN hardware integration smoke test:

```bash
MLXGO_ANE_INTEGRATION_FFN=1 go test -tags ane_appleneuralengine ./experiment/mlx-go-ane -run TestApplePrivateExecutorFFNForwardBackwardIntegration
```

## Tests

- `runtime_test.go`: ANE path, fallback path, shape validation.
- `nnhook_test.go`: hook routing and bias handling.
- `plan_test.go`: coverage analysis example behavior.
- `adapter_apple_private_darwin_integration_test.go`: env-gated end-to-end ANE
  linear validation on real hardware.
- `adapter_apple_private_darwin_ffn_integration_test.go`: env-gated FFN
  forward/backward ANE smoke test on real hardware.
- `mlx/nn/linear_hook_test.go`: `nn.Linear` hook contract from the `mlx/nn` side.
- `routing_test.go` / `runtime_router_test.go`: route policy and runtime gating behavior.
- `runtime_options_test.go`: profile/options constructor behavior.

## Release check

Run the package release gate:

```bash
./experiment/mlx-go-ane/scripts/release_check.sh
```

To include hardware gates:

```bash
MLXGO_ANE_INTEGRATION=1 \
MLXGO_ANE_INTEGRATION_FFN=1 \
./experiment/mlx-go-ane/scripts/release_check.sh
```

### Chaining Spike Test (`_ANEClient + _ANEModel`)

This test is isolated and opt-in. It measures:

- `_ANEClient` compile/load times for a compiled model artifact
- Baseline `_ANEClient evaluateWithModel` latency
- 3-phase chaining calls (`prepare`, `buffersReady`, `enqueue`) and total latency
- Output parity between baseline and chaining paths

Required env vars:

- `ANE_CHAINING_SPIKE=1`
- If using `.mlmodelc` path:
  - `ANE_CHAINING_SPIKE_MODEL_PATH=/abs/path/to/model.mlmodelc`
  - `ANE_CHAINING_SPIKE_INPUT_COUNT=<int>`
  - `ANE_CHAINING_SPIKE_OUTPUT_COUNT=<int>`
- If `ANE_CHAINING_SPIKE_MODEL_PATH` is unset:
  - test uses in-memory linear model build path with:
    - `ANE_CHAINING_SPIKE_BATCH` (default `16`)
    - `ANE_CHAINING_SPIKE_IN_DIM` (default `64`)
    - `ANE_CHAINING_SPIKE_OUT_DIM` (default `64`)

Optional env vars:

- `ANE_CHAINING_SPIKE_MODEL_KEY` (default: `s`)
- `ANE_CHAINING_SPIKE_QOS` (default: `21`)
- `ANE_CHAINING_SPIKE_SKIP_COMPILE=1` (skip `compileModel` for already-compiled file-backed models)
- `ANE_CHAINING_SPIKE_PREPARE_ONLY=1` (skip baseline evaluate/parity check; run chaining phases only)
- `ANE_CHAINING_SPIKE_SKIP_MAP=1` (skip pre-map of regular `_ANERequest` in chaining probe)
- `ANE_CHAINING_SPIKE_PROCEDURE_INDEX=<int>` (default `0`; procedure index used for chaining objects)
- `ANE_CHAINING_SPIKE_LOAD_NEW_INSTANCE=1` (attempt `_ANEClient loadModelNewInstance...` diagnostics)
- `ANE_CHAINING_SPIKE_USE_DO_CALLS=1` (also try `doPrepare`/`doBuffersReady`/`doEnqueue`)
- `ANE_CHAINING_SPIKE_RAW_DEVICE=enqueue|ready|both` (call ANE device vtable methods directly for selector 3/4 and report raw return codes)
- `ANE_CHAINING_SPIKE_SET_ACTIVE_PROCEDURE=after-prepare,before-ready,before-enqueue` (issue selector-8 active-procedure call via raw device vtable)
- `ANE_CHAINING_SPIKE_SET_ACTIVE_PROCEDURE_STRICT=1` (fail immediately if selector-8 call is non-zero)
- `ANE_CHAINING_SPIKE_SET_ACTIVE_PROCEDURE_EXTRA_HEX=<32 hex chars>` (optional 16-byte tail payload for selector-8 params)
- `ANE_CHAINING_SPIKE_SIGNAL_TRANSITION=after-prepare,before-ready,after-ready,before-enqueue,after-enqueue` (bump shared-event `signaledValue` at selected stages)
- `ANE_CHAINING_SPIKE_SIGNAL_TRANSITION_DELTA=<uint64>` (default `1`; per-write delta for staged signal transitions)
- `ANE_CHAINING_SPIKE_SIGNAL_TRANSITION_COUNT=<int>` (default `1`; repeats the staged transition pass)
- `ANE_CHAINING_SPIKE_SIGNAL_TRANSITION_ALL=1` (include all signal event types; default is free events only, `eventType=4`)
- `ANE_CHAINING_SPIKE_REQUIRE_SPEEDUP=1` (fail unless `chaining_total < baseline_eval`)
- `ANE_CHAINING_SPIKE_MAX_RATIO=<float>` (fail unless `chaining_total/baseline_eval <= value`)

Run:

```bash
ANE_CHAINING_SPIKE=1 \
ANE_CHAINING_SPIKE_MODEL_PATH=/abs/path/to/model.mlmodelc \
ANE_CHAINING_SPIKE_INPUT_COUNT=1024 \
ANE_CHAINING_SPIKE_OUTPUT_COUNT=1024 \
go test -tags ane_appleneuralengine ./experiment/mlx-go-ane -run TestANEClientChainingSpikeSingleProcedure -v
```

File-backed prepare-only probe example:

```bash
ANE_CHAINING_SPIKE=1 \
ANE_CHAINING_SPIKE_MODEL_PATH=/System/iOSSupport/System/Library/PrivateFrameworks/CoreRecognition.framework/Versions/A/Resources/cc_excnnfs_model.mlmodelc.bundle \
ANE_CHAINING_SPIKE_MODEL_KEY=s \
ANE_CHAINING_SPIKE_INPUT_COUNT=1 \
ANE_CHAINING_SPIKE_OUTPUT_COUNT=1 \
ANE_CHAINING_SPIKE_SKIP_COMPILE=1 \
ANE_CHAINING_SPIKE_PREPARE_ONLY=1 \
ANE_CHAINING_SPIKE_SKIP_MAP=1 \
ANE_CHAINING_SPIKE_LOAD_NEW_INSTANCE=1 \
go test -tags ane_appleneuralengine ./experiment/mlx-go-ane -run TestANEClientChainingSpikeSingleProcedure -v
```

## Direct ANE execution roadmap

### Phase 1: single-shape linear kernel path

1. Build MIL text + weights bundle for one `(batch, inDim, outDim)` shape.
2. Construct `_ANEInMemoryModelDescriptor` and `_ANEInMemoryModel`.
3. Return deterministic errors with context on any private API failure.

Exit criteria:

- `applePrivateExecutor.Linear` can execute a fixed test shape and return `[]float32` of expected length.

### Phase 2: request + IOSurface data plane

1. Allocate input/output IOSurfaces.
2. Wrap with `_ANEIOSurfaceObject`.
3. Create `_ANERequest`, map/evaluate/unmap.
4. Copy host slices into input surface and output surface back to `[]float32`.

Exit criteria:

- Round-trip data path validated by tests against MLX reference for multiple inputs.

### Phase 3: caching and performance hygiene

1. Cache compiled model artifacts by `(batch, inDim, outDim, weightHash)`.
2. Reuse IOSurface/model/request objects where safe.
3. Add explicit cache invalidation behavior and bounded memory policy.

Exit criteria:

- Repeat calls with identical shapes/weights avoid recompilation and reduce latency.

### Phase 4: runtime integration hardening

1. Propagate context cancellation through ANE evaluate path.
2. Expand failure classification for `FallbackReason`.
3. Add stress tests for concurrency and repeated hook install/restore cycles.

Exit criteria:

- Stable fallback behavior under cancellation, transient ANE errors, and concurrent callers.

## Notes

- Uses Apple private APIs when the adapter tag is enabled.
- Expect host and OS sensitivity across macOS releases.
- Keep MLX fallback as the safety path during all phases.
