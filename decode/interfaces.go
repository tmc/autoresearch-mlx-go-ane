//go:build darwin && ane_appleneuralengine

package decode

import (
	"context"

	"github.com/tmc/mlx-go-lm/exp/anehooks"
	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go/mlx"
)

// ---------------------------------------------------------------------------
// Stage interfaces (decomposed from former 12-method DecodePlaneStage)
// ---------------------------------------------------------------------------

// stageEvaluator is the core eval loop interface for a stage.
type stageEvaluator interface {
	InputSurface() anehooks.InputSurface
	InputShape() []int
	OutputSurface() any // concrete type known to bridge, not leaked here
	ModelDim() int
	MapSeq() int
	EvalPreparedSurfaceAsync(context.Context) <-chan anehooks.AsyncResult
}

// synchronizer provides event-based sync for stages and direct blocks.
// Used at setup-time wiring (cross-stage signaling) and output
// materialization, not just eval. Stages and blocks both satisfy this.
type synchronizer interface {
	WaitEvent() anehooks.Event
	SignalEvent() anehooks.Event
	WaitValue() uint64
	SignalValue() uint64
}

// ---------------------------------------------------------------------------
// Direct block interfaces (decomposed from former 17-method DecodePlaneDirectBlock)
// ---------------------------------------------------------------------------

// blockEvaluator is the core eval interface for a direct block.
type blockEvaluator interface {
	InputSurface() anehooks.InputSurface
	PosCosSurface() anehooks.InputSurface
	PosSinSurface() anehooks.InputSurface
	OutputSurface() any // concrete type known to bridge
	EvalPreparedSurface(context.Context) error
}

// blockStepper manages per-step decode position tracking.
// Called every token during the decode loop.
type blockStepper interface {
	DecodePosition() int
	AdvanceDecodePosition() error
	CurrentRoPESlice() (cos, sin []float32, err error)
}

// blockResetter handles initialization and error recovery.
// Called once at init (SetRoPETables) and rarely during recovery.
type blockResetter interface {
	Reset() error
	SetRoPETables(cosTable, sinTable []float32, headDim, maxSeqLen int) error
	RestoreStatefulMILState(decodePos int, milState [][]float32) error
}

// ---------------------------------------------------------------------------
// Bridge interfaces (decomposed from former 5-method DecodePlaneBridge)
// ---------------------------------------------------------------------------

// bridgeAliaser provides zero-copy surface aliasing between GPU and ANE.
// The surface parameter is the opaque value from OutputSurface(); the
// concrete bridge implementation casts internally.
type bridgeAliaser interface {
	AliasWritableFloat32(surface any, shape []int) (*mlx.Array, func() error, error)
	AliasReadOnlyFloat32(surface any, shape []int) (*mlx.Array, func() error, error)
}

// bridgeTransfer moves data between GPU and ANE with optional signaling.
//
// CopyIntoSignalReady is the atomic (copy + signal) path — the signal fires
// within the same Metal command buffer as the copy, so the ANE cannot read
// stale data. Prefer this over separate CopyInto + SignalMLXReady calls.
//
// CopyInto and SignalMLXReady are exposed for cases where the caller needs
// fine-grained control (e.g., multiple copies before one signal). Callers
// must ensure both execute in the same command buffer submission.
type bridgeTransfer interface {
	CopyIntoSignalReady(dst, src *mlx.Array, stream *mlx.Stream, waitValue uint64) error
	CopyInto(dst, src *mlx.Array, stream *mlx.Stream) error
	SignalMLXReady(stream *mlx.Stream, value uint64) error
}

// bridgeSyncer coordinates ANE completion and stream finalization.
type bridgeSyncer interface {
	WaitForANE(stream *mlx.Stream, signalValue uint64) error
	FinalizeStream(stream *mlx.Stream) error
}

// bridgeFusedNorm provides optional fused norm+signal operations on the
// post-attention → ANE bridge path. The engine type-asserts the bridge to
// this interface; if unavailable, it falls back to separate norm + copy:
//
//	if fn, ok := bridge.(bridgeFusedNorm); ok {
//	    fn.AddRMSNormIntoSignalReady(dst, x, attnOut, weight, stream, eps, waitValue)
//	} else {
//	    h := mlx.Add(x, attnOut)
//	    postNormed := rmsNorm(h, weight, eps)
//	    bridge.CopyIntoSignalReady(dst, postNormed, stream, waitValue)
//	}
type bridgeFusedNorm interface {
	RMSNormIntoSignalReady(dst, src, weight *mlx.Array, stream *mlx.Stream, eps float32, waitValue uint64) error
	AddRMSNormIntoSignalReady(dst, x, y, weight *mlx.Array, stream *mlx.Stream, eps float32, waitValue uint64) error
	AddRMSNormInto(dst, x, y, weight *mlx.Array, stream *mlx.Stream, eps float32) error
}

// bridgeAdder provides optional fused element-wise addition.
// Type-asserted like bridgeFusedNorm; falls back to mlx.Add if unavailable.
type bridgeAdder interface {
	AddInto(dst, x, y *mlx.Array, stream *mlx.Stream) error
}

// ---------------------------------------------------------------------------
// Model extraction interfaces (consumer-defined, no shared types needed)
//
// The decode engine type-asserts models.LanguageModel to these interfaces.
// Models satisfy them via Go structural typing — no anehooks import needed
// on the model side. Each interface is asserted independently; a model can
// implement a subset and the engine degrades gracefully.
// ---------------------------------------------------------------------------

// layerAttentionForwarder runs per-layer attention on GPU.
// This is the critical interface for hybrid GPU-attention / ANE-FFN decode.
//
// LayerAttentionForward runs input norm → attention for layer i and returns
// the attention output BEFORE residual add. The engine handles residual
// connections, post-attention norm, and FFN dispatch to ANE.
//
// The mask parameter is any because different attention types accept
// different mask formats (standard attention mask vs SSM mask).
type layerAttentionForwarder interface {
	LayerAttentionForward(i int, x *mlx.Array, mask any, cache kvcache.Cache) (*mlx.Array, error)
}

// layerWeightProvider extracts per-layer norm and FFN weights.
// Norm weights are *mlx.Array (used in GPU operations).
// FFN weights are []float32 (dequantized + transposed, ready for ANE stage building).
//
// Weight references are live (not copies) and valid while the model is loaded.
type layerWeightProvider interface {
	LayerInputNormWeight(i int) *mlx.Array
	LayerPostNormWeight(i int) *mlx.Array
	LayerFFNWeights(i int) (gate, up, down []float32, err error)
}

// headProvider extracts final norm and output projection weights.
type headProvider interface {
	FinalNormWeight() *mlx.Array
	LMHeadWeight() *mlx.Array // nil if tied embeddings
}

// embedder performs token embedding lookup.
type embedder interface {
	EmbedTokens(ids *mlx.Array) (*mlx.Array, error)
}

// moeLayerProvider is type-asserted for models with Mixture-of-Experts layers.
// Dense-only models do not implement this; the engine uses layerWeightProvider
// for all layers. Per-expert access avoids allocating all expert weight slices
// at once (models may have 64-128 experts).
type moeLayerProvider interface {
	LayerIsMoE(i int) bool
	LayerNumExperts(i int) int
	LayerSharedExpertWeights(i int) (gate, up, down []float32, err error)
	LayerExpertWeights(i int, expert int) (gate, up, down []float32, err error)
	LayerRouterWeight(i int) *mlx.Array // router stays on GPU for forward routing
}

// linearLayerProvider is type-asserted for models with linear attention
// (GatedDeltaNet). When LayerIsLinear returns true, the engine passes
// an SSM-style mask instead of a standard attention mask.
type linearLayerProvider interface {
	LayerIsLinear(i int) bool
}
