// Package mlxgoane provides experimental Apple Neural Engine execution
// primitives for mlx-go.
//
// The package is organized around three standalone seams:
//   - model assessment via AssessDecodeModel
//   - block-level runtimes such as ANEDraftModel
//   - IOSurface/shared-event data-plane helpers such as SurfaceEvalPlan,
//     MultiSurfaceEvalPlan, SharedEvent, MetalSharedEvent, and
//     MetalBufferBinding
//
// The model-assessment and routing helpers build everywhere. The block
// runtimes and IOSurface/shared-event data plane are Darwin-only and currently
// require the private-ANE build tags used by this repo.
//
// This package is intentionally narrow and non-stable.
package mlxgoane
