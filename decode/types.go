//go:build darwin

package decode

import (
	"context"
	"sync"
	"time"

	"github.com/tmc/mlx-go/mlx"
	mlxcompile "github.com/tmc/mlx-go/mlx/compile"
)

// stageKey identifies a stage by layer index and expert index.
type stageKey struct {
	layer  int
	expert int
}

// stage holds the runtime state for a single ANE FFN stage.
type stage struct {
	name              string
	densePrepare      *densePrepare
	outputMode        outputMode
	waitMode          waitMode
	onHostFallback    func(reason string)
	onOutputZeroCopy  func()
	onOutputCopy      func()
	onOutputWait      func(time.Duration)
	onOutputPoolStall func()
	warn              func(format string, args ...any)
	hostFallbackOnce  sync.Once
	poolMu            sync.Mutex
	poolDepth         int
	slots             []*stageSlot
	slotBuilder       func() (*stageSlot, error)
	leaseSeq          uint64
	normWeightMu      sync.Mutex
	normWeight32      *mlx.Array
}

// directBlock holds runtime state for multi-layer fused ANE blocks.
type directBlock struct {
	name              string
	layers            []int
	hiddenDim         int
	maxSeqLen         int
	attnHeads         int
	kvHeads           int
	headDim           int
	outputMode        outputMode
	waitMode          waitMode
	onOutputZeroCopy  func()
	onOutputCopy      func()
	onOutputWait      func(time.Duration)
	onOutputPoolStall func()
	poolMu            sync.Mutex
	poolDepth         int
	slots             []*directSlot
	slotBuilder       func() (*directSlot, error)
	leaseSeq          uint64
}

// stageSlot is a pooled execution slot for a single ANE stage.
// The stage and bridge fields are any-typed; callers type-assert to
// consumer interfaces (stageEvaluator, synchronizer, bridgeAliaser, etc.)
// from interfaces.go.
type stageSlot struct {
	name             string
	stage            any // satisfies stageEvaluator + synchronizer
	bridge           any // satisfies bridgeAliaser + bridgeTransfer + bridgeSyncer
	mu               sync.Mutex
	directInputAlias *mlx.Array
	packedInputAlias *mlx.Array
	outputAlias      *mlx.Array
	outputF32        *mlx.Array
	outputNative     *mlx.Array
	nextNormF32      *mlx.Array
	inUse            bool
	leaseSeq         uint64
}

// directSlot is a pooled execution slot for a multi-layer direct block.
// The block and bridge fields are any-typed; callers type-assert to
// consumer interfaces (blockEvaluator, blockStepper, blockResetter,
// synchronizer, bridgeAliaser, etc.) from interfaces.go.
type directSlot struct {
	name         string
	block        any // satisfies blockEvaluator + blockStepper + blockResetter + synchronizer
	bridge       any // satisfies bridgeAliaser + bridgeTransfer + bridgeSyncer
	inputAlias   *mlx.Array
	outputAlias  *mlx.Array
	outputF32    *mlx.Array
	outputNative *mlx.Array
	zeroK        *mlx.Array
	zeroV        *mlx.Array
	mu           sync.Mutex
	inUse        bool
	leaseSeq     uint64
}

// preparedRun tracks a leased stage slot through its eval lifecycle.
type preparedRun struct {
	stage           *stage
	slot            *stageSlot
	leaseSeq        uint64
	inputCleanup    func() error
	synchronized    bool
	streamCommitted bool
}

// directSpan describes a contiguous range of layers handled by a direct block.
type directSpan struct {
	startLayer  int
	layerOffset int
	layers      []int
}

// directFallback records why a direct block fell back to per-stage dispatch.
type directFallback struct {
	err   error
	build bool
}

// outputView wraps a borrowed output array and its release callback.
type outputView struct {
	arr     *mlx.Array
	release func()
}

// decodeOutputDtype is the default output dtype for ANE decode stages.
const decodeOutputDtype = mlx.Float32

// stageKind distinguishes dense, shared-expert, and per-expert stages.
type stageKind uint8

const (
	stageDense stageKind = iota
	stageShared
	stageExpert
)

// densePrepare holds an optional compiled preparation function for stage inputs.
type densePrepare struct {
	compiled *mlxcompile.Compiled
}

// dispatchTiming records fine-grained latencies for a single ANE dispatch.
type dispatchTiming struct {
	Prepare  time.Duration
	Alias    time.Duration
	Eval     time.Duration
	Copy     time.Duration
	Finalize time.Duration
	ANE      time.Duration
	Wait     time.Duration
	Output   time.Duration
	Router   time.Duration
	Combine  time.Duration
	Total    time.Duration
}

// ---------------------------------------------------------------------------
// Bridge capability interfaces
//
// These are runtime variants that accept any-typed streams, unlike the
// consumer interfaces in interfaces.go which use *mlx.Stream. They match
// what concrete bridge implementations expose and are type-asserted at
// dispatch time.
// ---------------------------------------------------------------------------

// syncEvalStage is a stage that supports synchronous (blocking) evaluation.
type syncEvalStage interface {
	EvalPreparedSurface(ctx context.Context) error
}

// splitCopyBridge supports separated copy and signal operations.
type splitCopyBridge interface {
	CopyInto(dst, src *mlx.Array, stream any) error
	SignalMLXReady(stream any, value uint64) error
}

// waitBridge supports GPU-side waiting for ANE completion.
type waitBridge interface {
	WaitForANE(stream any, value uint64) error
}

// addBridge supports fused element-wise addition on the bridge.
type addBridge interface {
	AddInto(dst, x, y *mlx.Array, stream any) error
}

// rmsNormBridge supports fused RMS normalization with signaling.
type rmsNormBridge interface {
	RMSNormIntoSignalReady(dst, src, weight *mlx.Array, stream any, eps float32, value uint64) error
}

// addRMSNormPlainBridge supports fused add+RMS-norm without signaling.
type addRMSNormPlainBridge interface {
	AddRMSNormInto(dst, x, y, weight *mlx.Array, stream any, eps float32) error
}

// addRMSNormBridge supports fused add+RMS-norm with signaling.
type addRMSNormBridge interface {
	AddRMSNormIntoSignalReady(dst, x, y, weight *mlx.Array, stream any, eps float32, value uint64) error
}
