// Package mlxgoane provides experimental Apple Neural Engine execution
// primitives for mlx-go.
//
// The package is organized around three standalone seams:
//   - model assessment via AssessDecodeModel
//   - block-level runtimes such as ANEDraftModel, SurfaceDecodeFFN, and
//     NanochatDecodeRuntime
//   - IOSurface/shared-event data-plane helpers such as SurfaceEvalPlan,
//     MultiSurfaceEvalPlan, SharedEvent, MetalSharedEvent, and
//     MetalBufferBinding
//
// The model-assessment and routing helpers build everywhere. The block
// runtimes and IOSurface/shared-event data plane are Darwin-only and currently
// require the private-ANE build tags used by this repo.
//
// Downstream mlx-go code should integrate at those seams. In practice that
// means:
//   - speculative drafting belongs above mlxlm/decode, for example in
//     examples/mlx-go-lm/internal/specdec
//   - model-specific decode offload belongs in a backend owned by the model
//     package, not in command code
//   - commands should only choose policy and fallback behavior
//
// Runtime.Linear and InstallNNLinearHook remain useful for controlled
// experiments and narrow kernels, but they are not the primary path for
// end-to-end decode or training-time sampling.
//
// The main working decode-plane route today is file-backed, daemon-backed ANE
// execution. In-memory MIL remains useful for probe graphs and compiler
// experiments, but it is not the default route for real model weights.
//
// This package is intentionally narrow and non-stable.
package mlxgoane
