# mlx-go-ane Design

This document records architecture decisions for the ANE experiment in
`experiment/mlx-go-ane`.

## Current Standalone Boundary

`mlx-go-ane` is intended to stand on its own as a runtime/data-plane package.
The clean downstream integration points are:

- `AssessDecodeModel` for classification
- `ANEDraftModel`, `SurfaceDecodeFFN`, or `NanochatDecodeRuntime` for
  block-level execution
- `SurfaceEvalPlan`, `MultiSurfaceEvalPlan`, `SharedEvent`, and
  `MetalBufferBinding` for direct IOSurface/shared-event integration

Downstream `mlx-go` packages should keep ownership separated:

- model packages own model-specific ANE decode backends
- speculative-draft policy belongs in shared packages such as
  `examples/mlx-go-lm/internal/specdec`
- command packages should only choose policy and expose metrics

`Runtime.Linear` and `InstallNNLinearHook` remain experiment APIs. They are not
the preferred seam for real model integration.

## ADR-0001 Runtime Path Selection

Status: accepted  
Date: 2026-03-02

### Decision

Use **subgraph interception + parametric MIL templates** as the primary ANE
execution path.

Keep `mlxfn -> MIL` logic as an **offline coverage and readiness analyzer**
only. Do not make primitive-by-primitive lowering the default runtime execution
path.

Keep deterministic MLX fallback enabled for unsupported or failed ANE
execution.

### Context

Current experiment code already provides:

- `InstallNNLinearHook(rt)` interception for `nn.Linear`.
- `Runtime.Linear(ctx, x, w)` validation, ANE call site, and fallback logic.
- `AnalyzeArchive`, `AnalyzeTextFile`, and `LowerToMIL` coverage checks.

Current experiment code does not yet provide:

- A host-validated, production-stable ANE dispatch path.
- Broad template coverage beyond initial linear template wiring.
- Cache eviction and lifecycle policy for compiled model objects.

### Why This Decision

Primitive-by-primitive lowering is a poor first runtime target for this stack:

- It tends to map local ops, not high-value fused subgraphs.
- It is hard to guarantee shape/layout rewrites that align with ANE fast paths.
- It increases per-dispatch fragmentation risk.
- It does not naturally preserve model-specific intermediate "tap" outputs.

Template substitution on intercepted subgraphs is a better fit for initial
runtime performance and implementation speed:

- Easier to fuse known hot blocks.
- Easier to reason about and debug generated MIL.
- Clearer cache keys and failure boundaries.

### Explicit Non-Decision

This ADR does **not** claim:

- fixed performance constants are universal across macOS versions or chips.
- private APIs are stable.
- zero-copy buffer sharing is guaranteed with current MLX internals.

Those are tracked as validation gates below.

## Decision Matrix

| Option | Time to first working | Perf ceiling | Coverage scalability | Maintenance risk |
|---|---|---|---|---|
| `mlxfn -> MIL` as primary runtime | Slow | Medium | High (eventual) | High |
| Subgraph interception + MIL templates | Fast | High | Medium | Medium |
| CoreML-only offload path | Fast | Medium | High | Low |

Selected path: **Subgraph interception + MIL templates**, with MLX fallback.

## Runtime Model

Primary routing model:

1. Intercept eligible module/subgraph.
2. Attempt template match and instantiate MIL with runtime dimensions.
3. Compile/load or fetch cached compiled artifact.
4. Execute on ANE.
5. Fallback to MLX on any unsupported/failed path.

`mlxfn` role:

- Offline inspection, coverage reporting, and preflight gating.
- Not the main online execution compiler.

## Data Path (Linear Baseline)

1. Hook intercepts `nn.Linear.Forward`.
2. Runtime validates rank/dtype/shape.
3. Runtime computes cache key `(graphKind, shape tuple, dtype, weight hash)`.
4. Cache hit: execute request; cache miss: generate MIL, compile/load, then execute.
5. Return ANE result or MLX fallback result.

## Hypotheses Requiring Validation

Treat the following as hypotheses until reproduced in this repo:

- Dispatch overhead and small-graph behavior.
- Compile-resource leak behavior and compile-count limits.
- SRAM residency breakpoints and throughput cliffs.
- Usability of private shared-event/chaining APIs for synchronization.

## Go/No-Go Gates

### Gate 1: In-Memory Model Compile/Run

Question:

- Can we compile and run a minimal linear MIL program in-process through
  `AppleNeuralEngine.framework` bindings from Go?

Pass criteria:

- Tagged Darwin test executes ANE linear and returns numerically valid output.

Fail action:

- Keep adapter experimental-only and continue with MLX fallback; evaluate
  alternate public-path offload.

### Gate 2: Reuse and Cache Stability

Question:

- Can we reuse compiled artifacts safely across repeated calls with identical
  key material?

Pass criteria:

- Warm path avoids recompilation and improves p50/p95 latency.

Fail action:

- Disable aggressive reuse, keep correctness-first execution with bounded cache.

### Gate 3: Buffer Sharing Feasibility

Question:

- Can MLX arrays and ANE requests share IOSurface-backed memory safely?

Pass criteria:

- Correctness holds under repeated runs and concurrency tests with reduced copy
  count.

Fail action:

- Keep copy-in/copy-out path and focus on graph fusion wins first.

### Gate 4: Compile Churn Durability

Question:

- How does long-running compile churn behave on target hosts?

Pass criteria:

- Process remains stable under sustained compile stress or can be bounded by
  finite shape buckets.

Fail action:

- Enforce compile budgets and shape bucketing; use worker isolation only if
  necessary.

## Implementation Plan

### Phase 1 Linear End-to-End

Files:

- `adapter_apple_private_darwin.go`
- `runtime.go`

Deliverables:

1. Replace stubbed `applePrivateExecutor.Linear` with real execution path.
2. Add minimal MIL generator for linear.
3. Preserve existing fallback semantics and error prefixes.

Acceptance:

- ANE backend is observable on supported/tagged runs.
- Numerics match MLX reference within tolerance.

### Phase 2 Measurements and Caching

Files:

- `runtime.go`
- cache module in `experiment/mlx-go-ane`

Deliverables:

1. Add compile/runtime timing instrumentation.
2. Add bounded compiled-artifact cache.
3. Add benchmarks for cold vs warm paths.

Acceptance:

- Measured warm-path latency improvement with stable correctness.

### Phase 3 Macro-Level Subgraph Templates

Files:

- interception hooks (`nnhook.go` plus follow-on hook points)
- new template/matcher modules

Deliverables:

1. Add matcher for first fused block pattern.
2. Emit one monolithic MIL template per matched block.
3. Keep full fallback to MLX for unmatched patterns.

Acceptance:

- Fewer host dispatches than unfused path.
- Clean degradation on mismatch.

### Phase 4 Hybrid Routing

Files:

- `runtime.go`
- routing policy module

Deliverables:

1. Add policy hooks for throughput-oriented vs latency-oriented paths.
2. Route compatible prefill-style work to ANE; retain low-latency decode on
   MLX path when beneficial.

Acceptance:

- Policy is measurable and reversible by configuration.

### Phase 5 Optional Zero-Copy and Advanced Sync

Files:

- IOSurface bridge module(s)
- possible `mlxc` helper bindings

Deliverables:

1. Prototype shared-buffer path.
2. Validate lifetimes and synchronization invariants.

Acceptance:

- Copy count reduction without correctness regressions.

## Measurements and Tests

Track:

- Cold compile/load latency.
- Warm dispatch latency.
- End-to-end latency by representative shapes.
- Cache hit rate and memory footprint.
- Fallback rate and fallback reasons.

Required test classes:

- Unit tests for validation and fallback invariants.
- Tagged integration tests for ANE execution path.
- Regression tests for hook install/restore safety.

## Out of Scope (for now)

- Full training pipeline design.
- Full primitive-complete `mlxfn -> MIL` runtime compiler.
- Declaring ANE as a first-class MLX device backend.

## Current Evidence Inventory

### Reproduced in this repo

- End-to-end ANE linear dispatch path exists behind build tag
  `ane_appleneuralengine`.
- Tagged integration test path is wired and env-gated:
  `MLXGO_ANE_INTEGRATION=1 go test -tags ane_appleneuralengine ./experiment/mlx-go-ane`.
- Runtime fallback semantics (`BackendMLX` + reason) and hook routing are
  covered by unit tests.
- Private-API options contract matters for compiler/backend stability:
  - `compileWithQoS:options:error:`, `loadWithQoS:options:error:`, and
    `evaluateWithQoS:options:request:error:` must receive an empty options
    dictionary (`@{}`) in this adapter path.
  - Passing `nil` `options` reproduced
    `_ANEEspressoIRTranslator : error Cannot load network .../model.espresso.net`.
- Linear path shape floor (host-observed):
  - In this repo's current linear MIL + IOSurface path, `batch` (MIL spatial)
    below 16 reproduced `Program IOSurfaces map failure (0x1D)`.
  - Integration validation currently pins to `batch=16, inDim=64, outDim=64`.

### Reported externally, not yet reproduced here

- Runtime `weightsBuffer` ignored for post-compile dynamic weight mutation.
- Compile-churn instability around sustained model recompilation.
- Specific dispatch overhead constants and SRAM cliff thresholds.

Treat these as open validation items until measured in this codebase on target
hardware.

## Descriptor and Request Contracts

For current adapter work, enforce these contracts explicitly:

- MIL/Proto functions must use fixed rank and dtype declarations.
- `_ANERequest` input/output arrays must align with zero-based function symbol
  indices (`inputIndices`, `outputIndices`).
- If constants are expressed via `BLOBFILE(path="@model_path/weights/weight.bin", ...)`,
  descriptor `weights` must include matching blob bytes keyed by
  `weights/weight.bin`.
- Cache keys must include shape and weight identity when compile artifacts are
  weight-specialized.

## NotebookLM Question Pack

After uploading `MIL.proto` and related CoreML format protos, ask these:

1. "Given `MIL.proto`, what is the smallest valid `Program` for `y = matmul(x, wt)` including all required fields and map keys?"
2. "List required vs optional fields for `Program`, `Function`, `Block`, and `Operation`, and explain why each required field is required by the schema."
3. "For MIL text using `BLOBFILE`, what exact blob path/key contract is implied between MIL text and external weight mapping?"
4. "Compare two linear representations for ANE targets: `matmul` op vs 1x1 `conv` form. Which schema fields differ and which are identical?"
5. "What protobuf constraints force static shapes, and which parts can remain dynamic?"
6. "Generate a validation checklist for `_ANERequest` symbol index mapping against a function signature with multiple inputs."
7. "What schema-level features would make primitive-by-primitive lowering brittle versus macro-template substitution?"
8. "Given an `mlxfn` trace, what graph metadata is missing to safely decide ANE template substitution without semantic risk?"
9. "Identify protobuf fields that could encode multi-output tap patterns for backprop or KV-cache handoff."
10. "Design a diff procedure to compare two serialized MIL programs and detect only shape/weight binding changes relevant for cache keys."
