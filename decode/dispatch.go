//go:build darwin && ane_appleneuralengine

package decode

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/mlx-go-lm/exp/anehooks"
	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/fast"
)

// ---------------------------------------------------------------------------
// Local consumer interfaces
// ---------------------------------------------------------------------------

// cacheSlicer extracts per-layer caches from a top-level cache.
type cacheSlicer interface {
	GetAllCaches() []kvcache.Cache
}

// moeRouter runs the routing forward pass for Mixture-of-Experts layers.
// The engine type-asserts the model to this interface at dispatch time;
// if unavailable, MoE layers fall back to the GPU path.
type moeRouter interface {
	// MoERouteForward runs the router gate on normalized input and returns
	// selected expert indices (int32) and routing scores.
	MoERouteForward(layerIdx int, normalized *mlx.Array) (indices *mlx.Array, scores *mlx.Array, err error)
	// MoESharedExpertGateForward computes the shared-expert gating scalar.
	MoESharedExpertGateForward(layerIdx int, normalized *mlx.Array) (*mlx.Array, error)
}

// mlpForwarder provides GPU-side MLP evaluation as a fallback when ANE is unavailable.
type mlpForwarder interface {
	LayerMLPForward(layerIdx int, normalized *mlx.Array) (*mlx.Array, error)
}

// ---------------------------------------------------------------------------
// RMS norm helper
// ---------------------------------------------------------------------------

// rmsNorm applies RMS normalization on GPU using the fast kernel.
func rmsNorm(x, weight *mlx.Array, eps float64) (*mlx.Array, error) {
	return fast.RMSNorm(x, weight, float32(eps), nil)
}

// ---------------------------------------------------------------------------
// Output dtype
// ---------------------------------------------------------------------------

func (p *Plane) outputDtypeFor(arr *mlx.Array) mlx.Dtype {
	if p == nil || arr == nil || arr.IsNil() {
		return decodeOutputDtype
	}
	if p.exp.OutputMode != outputModeGPUNative {
		return decodeOutputDtype
	}
	switch arr.Dtype() {
	case mlx.Float16, mlx.Bfloat16, mlx.Float32:
		return arr.Dtype()
	default:
		return decodeOutputDtype
	}
}

// ---------------------------------------------------------------------------
// Metal consumer operand preparation
// ---------------------------------------------------------------------------

func prepareMetalConsumerOperand(arr *mlx.Array, dtype mlx.Dtype) (*mlx.Array, func(), error) {
	if arr == nil || arr.IsNil() {
		return nil, nil, fmt.Errorf("metal consumer operand is nil")
	}
	current := arr
	var cleanup []func()
	if dtype != 0 && current.Dtype() != dtype {
		cast, err := mlx.Astype(current, dtype, nil)
		if err != nil {
			for i := len(cleanup) - 1; i >= 0; i-- {
				cleanup[i]()
			}
			return nil, nil, fmt.Errorf("cast metal consumer operand: %w", err)
		}
		current = cast
		cleanup = append(cleanup, func() { cast.Free() })
	}
	rowContig, err := mlx.MLXArrayIsRowContiguous(current)
	if err != nil {
		for i := len(cleanup) - 1; i >= 0; i-- {
			cleanup[i]()
		}
		return nil, nil, fmt.Errorf("metal consumer operand contiguity: %w", err)
	}
	if !rowContig {
		contig, err := mlx.Copy(current, nil)
		if err != nil {
			for i := len(cleanup) - 1; i >= 0; i-- {
				cleanup[i]()
			}
			return nil, nil, fmt.Errorf("copy metal consumer operand: %w", err)
		}
		current = contig
		cleanup = append(cleanup, func() { contig.Free() })
	}
	return current, func() {
		for i := len(cleanup) - 1; i >= 0; i-- {
			cleanup[i]()
		}
	}, nil
}

// ---------------------------------------------------------------------------
// Metal consumer operations
// ---------------------------------------------------------------------------

// denseMetalResidualAdd performs fused residual addition on the Metal bridge.
func (p *Plane) denseMetalResidualAdd(s *stage, residual, mlpOut *mlx.Array) (*mlx.Array, bool, error) {
	if p == nil || s == nil || residual == nil || residual.IsNil() || mlpOut == nil || mlpOut.IsNil() {
		return nil, false, nil
	}
	if p.exp.ConsumerMode != consumerMetalResidualAdd {
		return nil, false, nil
	}
	primary := s.primarySlot()
	if primary == nil {
		return nil, false, fmt.Errorf("%s: primary slot is nil", s.name)
	}
	ab, ok := primary.bridge.(addBridge)
	if !ok {
		return nil, false, nil
	}
	dst, err := primary.nextNormBuffer(s.modelDim())
	if err != nil {
		return nil, false, err
	}
	x, releaseX, err := prepareMetalConsumerOperand(residual, mlpOut.Dtype())
	if err != nil {
		return nil, false, err
	}
	defer releaseX()
	y, releaseY, err := prepareMetalConsumerOperand(mlpOut, mlpOut.Dtype())
	if err != nil {
		return nil, false, err
	}
	defer releaseY()
	if err := ab.AddInto(dst, x, y, nil); err != nil {
		return nil, false, fmt.Errorf("%s: metal residual add: %w", s.name, err)
	}
	return dst, true, nil
}

// inputNormWeight32 lazily casts and caches the norm weight as float32.
func (p *Plane) inputNormWeight32(layerIdx int, weight *mlx.Array, dim int) (*mlx.Array, error) {
	if p == nil {
		return nil, fmt.Errorf("ane decode plane is nil")
	}
	if canUseDirectNormWeight(weight, dim) {
		return weight, nil
	}
	p.inputNormMu.Lock()
	defer p.inputNormMu.Unlock()
	if arr := p.inputNormF32[layerIdx]; arr != nil && !arr.IsNil() {
		return arr, nil
	}
	if weight == nil || weight.IsNil() {
		return nil, fmt.Errorf("layer %d input norm weight is nil", layerIdx)
	}
	cast, err := mlx.Astype(weight, mlx.Float32, nil)
	if err != nil {
		return nil, fmt.Errorf("layer %d input norm cast: %w", layerIdx, err)
	}
	arr := cast
	rowContig, err := mlx.MLXArrayIsRowContiguous(arr)
	if err != nil {
		arr.Free()
		return nil, fmt.Errorf("layer %d input norm contiguity: %w", layerIdx, err)
	}
	if !rowContig {
		copied, copyErr := mlx.Copy(arr, nil)
		arr.Free()
		if copyErr != nil {
			return nil, fmt.Errorf("layer %d input norm copy: %w", layerIdx, copyErr)
		}
		arr = copied
	}
	if !canUseDirectNormWeight(arr, dim) {
		arr.Free()
		return nil, fmt.Errorf("layer %d input norm shape=%v incompatible with dim=%d", layerIdx, weight.Shape(), dim)
	}
	p.inputNormF32[layerIdx] = arr
	return arr, nil
}

// denseMetalResidualNorm performs fused residual + RMSNorm for the next layer input.
func (p *Plane) denseMetalResidualNorm(layerIdx int, s *stage, residual, mlpOut *mlx.Array) (*mlx.Array, bool, error) {
	if p == nil || s == nil || residual == nil || residual.IsNil() || mlpOut == nil || mlpOut.IsNil() {
		return nil, false, nil
	}
	if p.exp.ConsumerMode != consumerMetalResidualNorm {
		return nil, false, nil
	}
	cfg := p.model.Config()
	nextIdx := layerIdx + 1
	if nextIdx >= cfg.NumLayers {
		return nil, false, nil
	}
	nextNormWeight := p.weightProv.LayerInputNormWeight(nextIdx)
	if nextNormWeight == nil || nextNormWeight.IsNil() {
		return nil, false, nil
	}
	primary := s.primarySlot()
	if primary == nil {
		return nil, false, fmt.Errorf("%s: primary slot is nil", s.name)
	}
	anb, ok := primary.bridge.(addRMSNormPlainBridge)
	if !ok {
		return nil, false, nil
	}
	dst, err := primary.nextNormBuffer(s.modelDim())
	if err != nil {
		return nil, false, err
	}
	normWeight, err := p.inputNormWeight32(nextIdx, nextNormWeight, s.modelDim())
	if err != nil {
		return nil, false, err
	}
	x, releaseX, err := prepareMetalConsumerOperand(residual, mlpOut.Dtype())
	if err != nil {
		return nil, false, err
	}
	defer releaseX()
	y, releaseY, err := prepareMetalConsumerOperand(mlpOut, mlpOut.Dtype())
	if err != nil {
		return nil, false, err
	}
	defer releaseY()
	eps := cfg.RMSNormEps
	if err := anb.AddRMSNormInto(dst, x, y, normWeight, nil, float32(eps)); err != nil {
		return nil, false, fmt.Errorf("%s: metal residual norm: %w", s.name, err)
	}
	return dst, true, nil
}

// moeStackOutputs stacks MoE expert outputs for weighted combination.
func moeStackOutputs(outputs []*mlx.Array) (*mlx.Array, error) {
	if len(outputs) == 0 {
		return nil, fmt.Errorf("moe outputs are empty")
	}
	expanded := make([]*mlx.Array, 0, len(outputs))
	for _, out := range outputs {
		if out == nil || out.IsNil() {
			for _, arr := range expanded {
				arr.Free()
			}
			return nil, fmt.Errorf("moe output is nil")
		}
		shape := out.Shape()
		if len(shape) != 3 {
			for _, arr := range expanded {
				arr.Free()
			}
			return nil, fmt.Errorf("moe output shape=%v want 3D", shape)
		}
		arr, err := mlx.Reshape(out, []int{shape[0], shape[1], 1, 1, shape[2]}, nil)
		if err != nil {
			for _, prev := range expanded {
				prev.Free()
			}
			return nil, fmt.Errorf("reshape moe output: %w", err)
		}
		expanded = append(expanded, arr)
	}
	if len(expanded) == 1 {
		return expanded[0], nil
	}
	stacked, err := mlx.ConcatenateAxis(expanded, 2, nil)
	for _, arr := range expanded {
		arr.Free()
	}
	if err != nil {
		return nil, fmt.Errorf("concatenate moe outputs: %w", err)
	}
	return stacked, nil
}

// moeCombineOutputs performs fused MoE combination on the Metal bridge.
func (p *Plane) moeCombineOutputs(s *stage, sharedOut *mlx.Array, switchViews []outputView, scores *mlx.Array) (*mlx.Array, bool, error) {
	if p == nil || s == nil || sharedOut == nil || sharedOut.IsNil() {
		return nil, false, nil
	}
	if p.exp.ConsumerMode != consumerMetalMoECombine {
		return nil, false, nil
	}
	if len(switchViews) == 0 || scores == nil || scores.IsNil() {
		return sharedOut, true, nil
	}
	if !mlx.HasExpertWeightedSum() {
		return nil, false, nil
	}
	outputs := make([]*mlx.Array, 0, len(switchViews))
	for _, view := range switchViews {
		if view.arr == nil || view.arr.IsNil() {
			return nil, false, fmt.Errorf("moe switch output is nil")
		}
		outputs = append(outputs, view.arr)
	}
	stacked, err := moeStackOutputs(outputs)
	if err != nil {
		return nil, false, err
	}
	defer stacked.Free()
	routed, err := mlx.ExpertWeightedSum(stacked, scores, nil)
	if err != nil {
		return nil, false, fmt.Errorf("fused moe weighted sum: %w", err)
	}
	primary := s.primarySlot()
	if primary == nil {
		return nil, false, fmt.Errorf("%s: primary slot is nil", s.name)
	}
	ab, ok := primary.bridge.(addBridge)
	if !ok {
		combined, addErr := mlx.Add(sharedOut, routed, nil)
		if addErr != nil {
			return nil, false, fmt.Errorf("combine shared and routed outputs: %w", addErr)
		}
		return combined, true, nil
	}
	dst, err := primary.nextNormBuffer(s.modelDim())
	if err != nil {
		return nil, false, err
	}
	x, releaseX, err := prepareMetalConsumerOperand(sharedOut, routed.Dtype())
	if err != nil {
		return nil, false, err
	}
	defer releaseX()
	y, releaseY, err := prepareMetalConsumerOperand(routed, routed.Dtype())
	if err != nil {
		return nil, false, err
	}
	defer releaseY()
	if err := ab.AddInto(dst, x, y, nil); err != nil {
		return nil, false, fmt.Errorf("%s: metal moe combine add: %w", s.name, err)
	}
	return dst, true, nil
}

// ---------------------------------------------------------------------------
// Input preparation
// ---------------------------------------------------------------------------

// prepareInput reshapes and optionally pads x for ANE stage input.
func prepareInput(x *mlx.Array, dim, mapSeq int) (*mlx.Array, error) {
	if x == nil || x.IsNil() {
		return nil, fmt.Errorf("decode plane input is nil")
	}
	shape := x.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != dim {
		return nil, fmt.Errorf("decode plane input shape=%v want=[1 1 %d]", shape, dim)
	}
	base := x
	var toFree []*mlx.Array
	if x.Dtype() != mlx.Float32 {
		cast, err := mlx.Astype(x, mlx.Float32, nil)
		if err != nil {
			return nil, fmt.Errorf("cast decode plane input: %w", err)
		}
		base = cast
		toFree = append(toFree, cast)
	}
	packed, err := mlx.Reshape(base, []int{dim, 1}, nil)
	if err != nil {
		for _, arr := range toFree {
			arr.Free()
		}
		return nil, fmt.Errorf("reshape decode plane input: %w", err)
	}
	if mapSeq == 1 {
		for _, arr := range toFree {
			arr.Free()
		}
		rowContig, err := mlx.MLXArrayIsRowContiguous(packed)
		if err != nil {
			packed.Free()
			return nil, fmt.Errorf("row contiguous decode plane input: %w", err)
		}
		if rowContig {
			return packed, nil
		}
		return makeContiguousCopy(packed)
	}
	if mapSeq > 1 {
		zeros, zerr := mlx.Zeros([]int{dim, mapSeq - 1}, mlx.Float32, nil)
		if zerr != nil {
			packed.Free()
			for _, arr := range toFree {
				arr.Free()
			}
			return nil, fmt.Errorf("pad decode plane input: %w", zerr)
		}
		padded, perr := mlx.ConcatenateAxis([]*mlx.Array{packed, zeros}, 1, nil)
		zeros.Free()
		packed.Free()
		for _, arr := range toFree {
			arr.Free()
		}
		if perr != nil {
			return nil, fmt.Errorf("concat decode plane input: %w", perr)
		}
		return makeContiguousCopy(padded)
	}
	for _, arr := range toFree {
		arr.Free()
	}
	return makeContiguousCopy(packed)
}

// prepareInputGraph applies norm + reshape into a contiguous ANE input.
func prepareInputGraph(x, weight *mlx.Array, eps float64, dim, mapSeq int) (*mlx.Array, error) {
	if x == nil || x.IsNil() {
		return nil, fmt.Errorf("decode plane graph input is nil")
	}
	normalized, err := rmsNorm(x, weight, eps)
	if err != nil {
		return nil, fmt.Errorf("decode plane graph rms norm: %w", err)
	}
	base := normalized
	if normalized.Dtype() != mlx.Float32 {
		base, err = mlx.Astype(normalized, mlx.Float32, nil)
		if err != nil {
			return nil, fmt.Errorf("decode plane graph cast: %w", err)
		}
	}
	packed, err := mlx.Reshape(base, []int{dim, 1}, nil)
	if err != nil {
		return nil, fmt.Errorf("decode plane graph reshape: %w", err)
	}
	if mapSeq > 1 {
		zeros, zerr := mlx.Zeros([]int{dim, mapSeq - 1}, mlx.Float32, nil)
		if zerr != nil {
			return nil, fmt.Errorf("decode plane graph pad zeros: %w", zerr)
		}
		padded, perr := mlx.ConcatenateAxis([]*mlx.Array{packed, zeros}, 1, nil)
		if perr != nil {
			return nil, fmt.Errorf("decode plane graph concat: %w", perr)
		}
		packed = padded
	}
	contiguous, err := mlx.Contiguous(packed, false, nil)
	if err != nil {
		return nil, fmt.Errorf("decode plane graph contiguous: %w", err)
	}
	return contiguous, nil
}

// makeContiguousCopy frees the source array and returns a contiguous copy.
func makeContiguousCopy(arr *mlx.Array) (*mlx.Array, error) {
	if arr == nil || arr.IsNil() {
		return nil, fmt.Errorf("decode plane input is nil")
	}
	contiguous, err := mlx.Contiguous(arr, false, nil)
	arr.Free()
	if err != nil {
		return nil, fmt.Errorf("contiguous decode plane input: %w", err)
	}
	return contiguous, nil
}

// prepareDirectBlockCopyInput ensures the input is float32 and row-contiguous.
func prepareDirectBlockCopyInput(arr *mlx.Array) (*mlx.Array, func(), error) {
	if arr == nil || arr.IsNil() {
		return nil, nil, fmt.Errorf("direct block input is nil")
	}
	copySrc := arr
	var owned []*mlx.Array
	if arr.Dtype() != mlx.Float32 {
		cast, err := mlx.Astype(arr, mlx.Float32, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("direct block astype float32: %w", err)
		}
		copySrc = cast
		owned = append(owned, cast)
	}
	rowContig, err := mlx.MLXArrayIsRowContiguous(copySrc)
	if err != nil {
		for _, ownedArr := range owned {
			ownedArr.Free()
		}
		return nil, nil, fmt.Errorf("direct block row contiguous: %w", err)
	}
	if !rowContig {
		contig, err := mlx.Contiguous(copySrc, false, nil)
		if err != nil {
			for _, ownedArr := range owned {
				ownedArr.Free()
			}
			return nil, nil, fmt.Errorf("direct block contiguous: %w", err)
		}
		copySrc = contig
		owned = append(owned, contig)
	}
	return copySrc, func() {
		for i := len(owned) - 1; i >= 0; i-- {
			owned[i].Free()
		}
	}, nil
}

// ---------------------------------------------------------------------------
// Core dispatch: fan-out/fan-in for ANE evaluation
// ---------------------------------------------------------------------------

// dispatchPreparedStages dispatches packed input to multiple stages in parallel,
// waits for ANE completion, and collects output views.
func dispatchPreparedStages(ctx context.Context, packed *mlx.Array, stages []*stage, dtype mlx.Dtype) ([]outputView, int, dispatchTiming, error) {
	if len(stages) == 0 {
		return nil, 0, dispatchTiming{}, nil
	}
	start := time.Now()
	var timing dispatchTiming
	materialized := false
	if len(stages) > 1 {
		evalStart := time.Now()
		if err := mlx.Eval(packed); err != nil {
			return nil, 0, dispatchTiming{}, fmt.Errorf("materialize shared decode-plane input: %w", err)
		}
		timing.Eval = time.Since(evalStart)
		materialized = true
	}
	runs := make([]*preparedRun, 0, len(stages))
	abortAll := func() {
		for i := len(runs) - 1; i >= 0; i-- {
			runs[i].abort()
		}
	}
	for _, s := range stages {
		run, stepTiming, err := s.beginPreparedInputTimed(packed, materialized, len(stages) == 1)
		if err != nil {
			abortAll()
			return nil, 0, dispatchTiming{}, err
		}
		timing.Alias += stepTiming.Alias
		timing.Copy += stepTiming.Copy
		timing.Prepare += stepTiming.Prepare
		runs = append(runs, run)
	}
	needFinalize := true
	for _, run := range runs {
		if run.streamCommitted {
			needFinalize = false
			break
		}
	}
	if needFinalize {
		bs, ok := runs[0].slot.bridge.(bridgeSyncer)
		if !ok {
			abortAll()
			return nil, 0, dispatchTiming{}, fmt.Errorf("bridge does not implement bridgeSyncer")
		}
		finalizeStart := time.Now()
		if err := bs.FinalizeStream(nil); err != nil {
			abortAll()
			return nil, 0, dispatchTiming{}, fmt.Errorf("finalize MLX stream: %w", err)
		}
		timing.Finalize += time.Since(finalizeStart)
		timing.Prepare += timing.Finalize
	}
	aneStart := time.Now()
	if len(runs) == 1 {
		if syncStg, ok := runs[0].slot.stage.(syncEvalStage); ok {
			if err := syncStg.EvalPreparedSurface(ctx); err != nil {
				runs[0].abort()
				return nil, 0, dispatchTiming{}, err
			}
		} else {
			res := <-runs[0].start(ctx)
			if res.Err != nil {
				runs[0].abort()
				return nil, 0, dispatchTiming{}, res.Err
			}
		}
	} else {
		evals := make([]<-chan anehooks.AsyncResult, len(runs))
		for i, run := range runs {
			evals[i] = run.start(ctx)
		}
		for i, ch := range evals {
			res := <-ch
			if res.Err != nil {
				for j := len(runs) - 1; j >= i; j-- {
					runs[j].abort()
				}
				return nil, 0, dispatchTiming{}, res.Err
			}
		}
	}
	timing.ANE = time.Since(aneStart)
	for _, run := range runs {
		run.finishInput()
	}
	outputStart := time.Now()
	views := make([]outputView, 0, len(runs))
	synchronizedStageCalls := 0
	for _, run := range runs {
		if run.synchronized {
			synchronizedStageCalls++
		}
		view, err := run.outputHiddenView(dtype)
		if err != nil {
			for i := len(views) - 1; i >= 0; i-- {
				if views[i].release != nil {
					views[i].release()
				}
			}
			return nil, 0, dispatchTiming{}, err
		}
		views = append(views, *view)
	}
	timing.Output = time.Since(outputStart)
	timing.Total = time.Since(start)
	return views, synchronizedStageCalls, timing, nil
}

// releaseOutputViews releases all output views in reverse order.
func releaseOutputViews(views []outputView) {
	for i := len(views) - 1; i >= 0; i-- {
		if views[i].release != nil {
			views[i].release()
		}
	}
}

// ---------------------------------------------------------------------------
// Dense dispatch variants
// ---------------------------------------------------------------------------

// useDecodePlane checks whether a forward call should use ANE dispatch.
func (p *Plane) useDecodePlane(embeddings *mlx.Array, cache kvcache.Cache) bool {
	if p == nil || p.model == nil || embeddings == nil || embeddings.IsNil() || cache == nil || p.isDisabled() {
		return false
	}
	shape := embeddings.Shape()
	return len(shape) == 3 && shape[0] == 1 && shape[1] == 1
}

// cacheSlice extracts per-layer caches from the top-level cache.
func (p *Plane) cacheSlice(cache kvcache.Cache) []kvcache.Cache {
	if cache == nil {
		return nil
	}
	if cs, ok := cache.(cacheSlicer); ok {
		return cs.GetAllCaches()
	}
	return nil
}

// runBaseFromEmbeddings delegates to the wrapped model's full forward.
func (p *Plane) runBaseFromEmbeddings(embeddings *mlx.Array, cache kvcache.Cache) (*mlx.Array, kvcache.Cache, error) {
	return p.model.Forward(embeddings, cache)
}

// denseOutput runs a single dense FFN layer through ANE.
func (p *Plane) denseOutput(layerIdx int, normalized *mlx.Array) (*mlx.Array, []outputView, error) {
	s, err := p.denseStage(layerIdx)
	if err != nil {
		return nil, nil, err
	}
	if canUseDirectInput(normalized, s.modelDim(), s.mapSeq()) {
		return p.denseOutputPrepared(s, normalized, p.outputDtypeFor(normalized), dispatchTiming{})
	}
	packed, err := prepareInput(normalized, s.modelDim(), s.mapSeq())
	if err != nil {
		return nil, nil, err
	}
	defer packed.Free()
	views, synchronizedStageCalls, timing, err := dispatchPreparedStages(context.Background(), packed, []*stage{s}, p.outputDtypeFor(normalized))
	if err != nil {
		return nil, nil, err
	}
	p.recordDispatch(stageDense, 1, synchronizedStageCalls, timing)
	if len(views) != 1 {
		releaseOutputViews(views)
		return nil, nil, fmt.Errorf("dense stage produced %d outputs", len(views))
	}
	return views[0].arr, views, nil
}

// denseOutputPrepared dispatches a pre-packed dense input to ANE.
func (p *Plane) denseOutputPrepared(s *stage, packed *mlx.Array, dtype mlx.Dtype, prepareTiming dispatchTiming) (*mlx.Array, []outputView, error) {
	if s == nil || s.primarySlot() == nil {
		return nil, nil, fmt.Errorf("dense stage is nil")
	}
	views, synchronizedStageCalls, timing, err := dispatchPreparedStages(context.Background(), packed, []*stage{s}, dtype)
	if err != nil {
		return nil, nil, err
	}
	timing.Prepare += prepareTiming.Prepare
	timing.Alias += prepareTiming.Alias
	timing.Eval += prepareTiming.Eval
	timing.Copy += prepareTiming.Copy
	timing.Finalize += prepareTiming.Finalize
	timing.Total += prepareTiming.Total
	p.recordDispatch(stageDense, 1, synchronizedStageCalls, timing)
	if len(views) != 1 {
		releaseOutputViews(views)
		return nil, nil, fmt.Errorf("dense stage produced %d outputs", len(views))
	}
	return views[0].arr, views, nil
}

// denseOutputFromResidual fuses RMSNorm into the input copy and dispatches to ANE.
func (p *Plane) denseOutputFromResidual(s *stage, residual, weight *mlx.Array, eps float64, dtype mlx.Dtype) (*mlx.Array, []outputView, bool, error) {
	if s == nil || s.primarySlot() == nil {
		return nil, nil, false, fmt.Errorf("dense stage is nil")
	}
	if _, ok := s.primarySlot().bridge.(rmsNormBridge); !ok {
		return nil, nil, false, nil
	}
	if !canUseDirectInput(residual, s.modelDim(), 1) {
		return nil, nil, false, nil
	}
	normWeight, err := s.directNormWeight(weight)
	if err != nil {
		return nil, nil, false, err
	}
	start := time.Now()
	run, timing, err := s.beginPreparedResidualNormTimed(residual, normWeight, float32(eps))
	if err != nil {
		return nil, nil, false, err
	}
	aneStart := time.Now()
	if syncStg, ok := run.slot.stage.(syncEvalStage); ok {
		if err := syncStg.EvalPreparedSurface(context.Background()); err != nil {
			run.abort()
			return nil, nil, true, err
		}
	} else {
		res := <-run.start(context.Background())
		if res.Err != nil {
			run.abort()
			return nil, nil, true, res.Err
		}
	}
	timing.ANE = time.Since(aneStart)
	run.finishInput()
	outputStart := time.Now()
	view, err := run.outputHiddenView(dtype)
	if err != nil {
		return nil, nil, true, err
	}
	timing.Output = time.Since(outputStart)
	timing.Total = time.Since(start)
	p.recordDispatch(stageDense, 1, 1, timing)
	return view.arr, []outputView{*view}, true, nil
}

// denseOutputFromInputs fuses residual add + RMSNorm into the input copy,
// computing the residual in parallel with ANE FFN dispatch.
func (p *Plane) denseOutputFromInputs(s *stage, x, attnOut, weight *mlx.Array, eps float64, dtype mlx.Dtype) (*mlx.Array, *mlx.Array, []outputView, bool, error) {
	if s == nil || s.primarySlot() == nil {
		return nil, nil, nil, false, fmt.Errorf("dense stage is nil")
	}
	if _, ok := s.primarySlot().bridge.(addRMSNormBridge); !ok {
		return nil, nil, nil, false, nil
	}
	if !canUseDirectResidualInputs(x, attnOut, s.modelDim()) {
		return nil, nil, nil, false, nil
	}
	normWeight, err := s.directNormWeight(weight)
	if err != nil {
		return nil, nil, nil, false, err
	}
	start := time.Now()
	run, timing, err := s.beginPreparedResidualAddNormTimed(x, attnOut, normWeight, float32(eps))
	if err != nil {
		return nil, nil, nil, true, err
	}
	aneCh := run.start(context.Background())
	residual, err := mlx.Add(x, attnOut, nil)
	if err != nil {
		run.abort()
		return nil, nil, nil, true, fmt.Errorf("attention residual: %w", err)
	}
	if err := mlx.Eval(residual); err != nil {
		residual.Free()
		run.abort()
		return nil, nil, nil, true, fmt.Errorf("eval attention residual: %w", err)
	}
	aneStart := time.Now()
	res := <-aneCh
	timing.ANE = time.Since(aneStart)
	if res.Err != nil {
		residual.Free()
		run.abort()
		return nil, nil, nil, true, res.Err
	}
	run.finishInput()
	outputStart := time.Now()
	view, err := run.outputHiddenView(dtype)
	if err != nil {
		residual.Free()
		return nil, nil, nil, true, err
	}
	timing.Output = time.Since(outputStart)
	timing.Total = time.Since(start)
	p.recordDispatch(stageDense, 1, 1, timing)
	return residual, view.arr, []outputView{*view}, true, nil
}

// ---------------------------------------------------------------------------
// Direct block dispatch
// ---------------------------------------------------------------------------

// expandRoPERow repeats a RoPE row across attention heads.
func expandRoPERow(row []float32, repeats int) []float32 {
	if repeats <= 1 {
		return append([]float32(nil), row...)
	}
	out := make([]float32, 0, len(row)*repeats)
	for i := 0; i < repeats; i++ {
		out = append(out, row...)
	}
	return out
}

// directBlockOutput evaluates a multi-layer direct block on ANE and returns
// the output, views, the last layer index handled, and whether the block was used.
func (p *Plane) directBlockOutput(layerIdx int, x *mlx.Array, caches []kvcache.Cache, dtype mlx.Dtype) (*mlx.Array, []outputView, int, bool, error) {
	if !p.exp.DirectBlock || x == nil || x.IsNil() {
		return nil, nil, 0, false, nil
	}
	span, ok := p.shouldUseDirectSpan(layerIdx)
	if !ok {
		return nil, nil, 0, false, nil
	}
	block, err := p.getDirectBlock(layerIdx)
	if err != nil {
		p.markDirectBlockFallback(span, true, err)
		return nil, nil, 0, false, nil
	}
	slot, lease, _, err := block.acquireSlot()
	if err != nil {
		p.markDirectBlockFallback(span, false, err)
		return nil, nil, 0, false, nil
	}
	slot.mu.Lock()
	released := false
	releaseSlot := func() {
		if released {
			return
		}
		released = true
		slot.mu.Unlock()
		block.releaseSlot(slot, lease)
	}
	ready, err := syncDirectBlockState(slot, block, caches)
	if err != nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, err)
		return nil, nil, 0, false, nil
	}
	if !ready {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: direct block state not ready", block.name))
		return nil, nil, 0, false, nil
	}
	stepper, ok := slot.block.(blockStepper)
	if !ok {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: block does not implement blockStepper", block.name))
		return nil, nil, 0, false, nil
	}
	cosRow, sinRow, err := stepper.CurrentRoPESlice()
	if err != nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: current rope slice: %w", block.name, err))
		return nil, nil, 0, false, nil
	}
	posCos := expandRoPERow(cosRow, block.attnHeads)
	posSin := expandRoPERow(sinRow, block.attnHeads)

	start := time.Now()
	var timing dispatchTiming
	eval, ok := slot.block.(blockEvaluator)
	if !ok {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: block does not implement blockEvaluator", block.name))
		return nil, nil, 0, false, nil
	}
	if surf := eval.PosCosSurface(); surf == nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: pos_cos surface is unavailable", block.name))
		return nil, nil, 0, false, nil
	} else if err := surf.Write(posCos); err != nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: write pos_cos: %w", block.name, err))
		return nil, nil, 0, false, nil
	}
	if surf := eval.PosSinSurface(); surf == nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: pos_sin surface is unavailable", block.name))
		return nil, nil, 0, false, nil
	} else if err := surf.Write(posSin); err != nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: write pos_sin: %w", block.name, err))
		return nil, nil, 0, false, nil
	}
	inputVals, err := arrayToFloat32Slice(x)
	if err != nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: prepare input: %w", block.name, err))
		return nil, nil, 0, false, nil
	}
	copyStart := time.Now()
	if surf := eval.InputSurface(); surf == nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: input surface is unavailable", block.name))
		return nil, nil, 0, false, nil
	} else if err := surf.Write(inputVals); err != nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: write input: %w", block.name, err))
		return nil, nil, 0, false, nil
	}
	timing.Copy = time.Since(copyStart)
	timing.Prepare = time.Since(start)

	aneStart := time.Now()
	if err := eval.EvalPreparedSurface(context.Background()); err != nil {
		releaseSlot()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: direct eval: %w", block.name, err))
		return nil, nil, 0, false, nil
	}
	timing.ANE = time.Since(aneStart)
	outputStart := time.Now()
	view, err := block.outputHiddenView(slot, lease, dtype)
	if err != nil {
		p.markDirectBlockFallback(span, false, err)
		return nil, nil, 0, false, nil
	}
	released = true
	timing.Output = time.Since(outputStart)
	zeroK, zeroV, err := slot.zeroRows(block.name, block.kvHeads, block.headDim, mlx.Float16)
	if err != nil {
		view.release()
		p.markDirectBlockFallback(span, false, err)
		return nil, nil, 0, false, nil
	}
	if err := advanceDirectBlockCaches(block, caches, zeroK, zeroV); err != nil {
		view.release()
		p.markDirectBlockFallback(span, false, err)
		return nil, nil, 0, false, nil
	}
	if err := stepper.AdvanceDecodePosition(); err != nil {
		view.release()
		p.markDirectBlockFallback(span, false, fmt.Errorf("%s: advance decode position: %w", block.name, err))
		return nil, nil, 0, false, nil
	}
	timing.Total = time.Since(start)
	p.recordDirectBlockDispatch(len(span.layers), timing)
	return view.arr, []outputView{*view}, span.layers[len(span.layers)-1], true, nil
}

// ---------------------------------------------------------------------------
// KV-cache sync for direct blocks
// ---------------------------------------------------------------------------

// syncDirectBlockState synchronizes the direct block's internal MIL state
// with the external KV caches.
func syncDirectBlockState(slot *directSlot, block *directBlock, caches []kvcache.Cache) (bool, error) {
	if block == nil || slot == nil || slot.block == nil {
		return false, fmt.Errorf("ane direct block is nil")
	}
	stepper, ok := slot.block.(blockStepper)
	if !ok {
		return false, fmt.Errorf("block does not implement blockStepper")
	}
	resetter, ok := slot.block.(blockResetter)
	if !ok {
		return false, fmt.Errorf("block does not implement blockResetter")
	}
	wantPos := 0
	for _, layerIdx := range block.layers {
		if layerIdx >= len(caches) || caches[layerIdx] == nil {
			continue
		}
		wantPos = caches[layerIdx].Offset()
		break
	}
	for _, layerIdx := range block.layers {
		if layerIdx >= len(caches) || caches[layerIdx] == nil {
			continue
		}
		if got := caches[layerIdx].Offset(); got != wantPos {
			return false, fmt.Errorf("direct block cache offset mismatch layer=%d got=%d want=%d", layerIdx, got, wantPos)
		}
	}
	if stepper.DecodePosition() == wantPos {
		return true, nil
	}
	if wantPos == 0 {
		return true, resetter.Reset()
	}
	if wantPos > block.maxSeqLen {
		return false, nil
	}
	state, err := directBlockStateFromCaches(caches, block.layers, block.attnHeads, block.kvHeads, block.headDim, block.maxSeqLen)
	if err != nil {
		return false, err
	}
	return true, resetter.RestoreStatefulMILState(wantPos, state)
}

// advanceDirectBlockCaches inserts zero rows into external caches to keep
// their offsets aligned with the direct block's internal position.
func advanceDirectBlockCaches(block *directBlock, caches []kvcache.Cache, zeroK, zeroV *mlx.Array) error {
	if block == nil {
		return fmt.Errorf("ane direct block is nil")
	}
	for _, layerIdx := range block.layers {
		if layerIdx >= len(caches) || caches[layerIdx] == nil {
			continue
		}
		if _, _, err := caches[layerIdx].UpdateAndFetch(zeroK, zeroV); err != nil {
			return fmt.Errorf("%s: advance external cache offset layer %d: %w", block.name, layerIdx, err)
		}
	}
	return nil
}

// directBlockStateFromCaches extracts and aligns KV state from external caches
// for all layers in a direct block.
func directBlockStateFromCaches(caches []kvcache.Cache, layers []int, stateHeads, cacheHeads, headDim, maxSeqLen int) ([][]float32, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("direct block state cache layers are empty")
	}
	state := make([][]float32, 0, len(layers)*2)
	for _, layerIdx := range layers {
		if layerIdx >= len(caches) || caches[layerIdx] == nil {
			return nil, fmt.Errorf("direct block state cache layer %d is nil", layerIdx)
		}
		keys, values := caches[layerIdx].GetValidPortion()
		if keys == nil || values == nil || keys.IsNil() || values.IsNil() {
			return nil, fmt.Errorf("direct block state cache layer %d is nil", layerIdx)
		}
		layerState, err := directBlockStateFromCache(keys, values, stateHeads, cacheHeads, headDim, maxSeqLen)
		if err != nil {
			return nil, fmt.Errorf("direct block state cache layer %d: %w", layerIdx, err)
		}
		state = append(state, layerState...)
	}
	return state, nil
}

// directBlockStateFromCache converts a single layer's KV tensors into
// right-aligned, head-expanded float32 state arrays.
func directBlockStateFromCache(keys, values *mlx.Array, stateHeads, cacheHeads, headDim, maxSeqLen int) ([][]float32, error) {
	if keys == nil || keys.IsNil() || values == nil || values.IsNil() {
		return nil, fmt.Errorf("direct block state cache is nil")
	}
	kShape := keys.Shape()
	vShape := values.Shape()
	if len(kShape) != 4 || len(vShape) != 4 {
		return nil, fmt.Errorf("direct block state cache shapes=%v/%v want 4D", kShape, vShape)
	}
	if kShape[0] != 1 || vShape[0] != 1 || kShape[1] != cacheHeads || vShape[1] != cacheHeads || kShape[3] != headDim || vShape[3] != headDim {
		return nil, fmt.Errorf("direct block state cache shapes=%v/%v incompatible with cache_heads=%d head_dim=%d", kShape, vShape, cacheHeads, headDim)
	}
	seqLen := kShape[2]
	if seqLen != vShape[2] {
		return nil, fmt.Errorf("direct block state seq mismatch k=%d v=%d", seqLen, vShape[2])
	}
	if seqLen > maxSeqLen {
		origKeys, origValues := keys, values
		var err error
		keys, err = mlx.Slice(keys, []int{0, 0, seqLen - maxSeqLen, 0}, []int{1, cacheHeads, seqLen, headDim}, []int{1, 1, 1, 1}, nil)
		if err != nil {
			return nil, fmt.Errorf("slice direct block K cache: %w", err)
		}
		values, err = mlx.Slice(values, []int{0, 0, seqLen - maxSeqLen, 0}, []int{1, cacheHeads, seqLen, headDim}, []int{1, 1, 1, 1}, nil)
		if err != nil {
			keys.Free()
			return nil, fmt.Errorf("slice direct block V cache: %w", err)
		}
		if origKeys != keys {
			defer keys.Free()
		}
		if origValues != values {
			defer values.Free()
		}
		seqLen = maxSeqLen
	}
	kFlat, err := arrayToFloat32Slice(keys)
	if err != nil {
		return nil, fmt.Errorf("materialize direct block K cache: %w", err)
	}
	vFlat, err := arrayToFloat32Slice(values)
	if err != nil {
		return nil, fmt.Errorf("materialize direct block V cache: %w", err)
	}
	return [][]float32{
		expandRightAlignDirectBlockState(kFlat, cacheHeads, stateHeads, seqLen, maxSeqLen, headDim),
		expandRightAlignDirectBlockState(vFlat, cacheHeads, stateHeads, seqLen, maxSeqLen, headDim),
	}, nil
}

// expandRightAlignDirectBlockState right-aligns and head-repeats KV state
// into the direct block's expected layout.
func expandRightAlignDirectBlockState(src []float32, cacheHeads, stateHeads, seqLen, maxSeqLen, headDim int) []float32 {
	dst := make([]float32, stateHeads*maxSeqLen*headDim)
	if seqLen <= 0 || cacheHeads <= 0 || stateHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return dst
	}
	repeats := 1
	if stateHeads != cacheHeads {
		if stateHeads%cacheHeads != 0 {
			return dst
		}
		repeats = stateHeads / cacheHeads
	}
	headSrcStride := seqLen * headDim
	headDstStride := maxSeqLen * headDim
	dstOff := (maxSeqLen - seqLen) * headDim
	for h := 0; h < cacheHeads; h++ {
		srcHead := src[h*headSrcStride : (h+1)*headSrcStride]
		for rep := 0; rep < repeats; rep++ {
			dstHead := dst[(h*repeats+rep)*headDstStride : (h*repeats+rep+1)*headDstStride]
			copy(dstHead[dstOff:], srcHead)
		}
	}
	return dst
}

// ---------------------------------------------------------------------------
// MoE dispatch
// ---------------------------------------------------------------------------

// moeOutput dispatches a MoE layer: routes tokens, evaluates shared + expert
// stages on ANE, and combines results.
//
// The model must implement the moeRouter interface for routing computation
// (gate forward, topK selection, shared expert gating). If unavailable, the
// caller should fall back to the GPU path.
func (p *Plane) moeOutput(layerIdx int, normalized *mlx.Array) (*mlx.Array, []outputView, error) {
	if p.moeProv == nil {
		return nil, nil, fmt.Errorf("moe output layer %d: model has no MoE provider", layerIdx)
	}
	router, ok := p.model.(moeRouter)
	if !ok {
		return nil, nil, fmt.Errorf("moe output layer %d: model does not implement moeRouter", layerIdx)
	}
	sharedStg, err := p.sharedStage(layerIdx)
	if err != nil {
		return nil, nil, err
	}
	routerStart := time.Now()
	inds, scores, err := router.MoERouteForward(layerIdx, normalized)
	if err != nil {
		return nil, nil, fmt.Errorf("router forward: %w", err)
	}
	routerDur := time.Since(routerStart)
	expertIDs, err := mlx.ToSlice[int32](inds)
	if err != nil {
		return nil, nil, fmt.Errorf("materialize expert indices: %w", err)
	}
	stages := make([]*stage, 0, 1+len(expertIDs))
	stages = append(stages, sharedStg)
	for _, expertID := range expertIDs {
		s, stageErr := p.expertStage(layerIdx, int(expertID))
		if stageErr != nil {
			return nil, nil, stageErr
		}
		stages = append(stages, s)
	}
	packed := normalized
	var releasePacked func()
	if !canUseDirectInput(normalized, sharedStg.modelDim(), sharedStg.mapSeq()) {
		var prepErr error
		packed, prepErr = prepareInput(normalized, sharedStg.modelDim(), sharedStg.mapSeq())
		if prepErr != nil {
			return nil, nil, prepErr
		}
		releasePacked = func() { packed.Free() }
	}
	if releasePacked != nil {
		defer releasePacked()
	}
	views, synchronizedStageCalls, timing, err := dispatchPreparedStages(context.Background(), packed, stages, p.outputDtypeFor(normalized))
	if err != nil {
		return nil, nil, err
	}
	timing.Router = routerDur
	if len(views) != len(stages) {
		releaseOutputViews(views)
		return nil, nil, fmt.Errorf("moe stage fan-out returned %d outputs want %d", len(views), len(stages))
	}
	combineStart := time.Now()
	sharedGate, err := router.MoESharedExpertGateForward(layerIdx, normalized)
	if err != nil {
		releaseOutputViews(views)
		return nil, nil, fmt.Errorf("shared expert gate: %w", err)
	}
	sharedGate, err = mlx.Sigmoid(sharedGate, nil)
	if err != nil {
		releaseOutputViews(views)
		return nil, nil, fmt.Errorf("sigmoid shared expert gate: %w", err)
	}
	sharedOut, err := mlx.Multiply(sharedGate, views[0].arr, nil)
	if err != nil {
		releaseOutputViews(views)
		return nil, nil, fmt.Errorf("gate shared expert output: %w", err)
	}
	combined, handled, combineErr := p.moeCombineOutputs(sharedStg, sharedOut, views[1:], scores)
	if combineErr != nil {
		releaseOutputViews(views)
		return nil, nil, combineErr
	}
	if !handled {
		combined = sharedOut
		for i := 0; i < len(expertIDs); i++ {
			score, sliceErr := mlx.Slice(scores, []int{0, 0, i}, []int{1, 1, i + 1}, []int{1, 1, 1}, nil)
			if sliceErr != nil {
				releaseOutputViews(views)
				return nil, nil, fmt.Errorf("slice expert score %d: %w", i, sliceErr)
			}
			weighted, weightErr := mlx.Multiply(views[i+1].arr, score, nil)
			if weightErr != nil {
				releaseOutputViews(views)
				return nil, nil, fmt.Errorf("weight expert output %d: %w", i, weightErr)
			}
			next, addErr := mlx.Add(combined, weighted, nil)
			if addErr != nil {
				releaseOutputViews(views)
				return nil, nil, fmt.Errorf("accumulate expert output %d: %w", i, addErr)
			}
			combined = next
		}
	}
	timing.Combine = time.Since(combineStart)
	timing.Total += timing.Router + timing.Combine
	p.recordDispatch(stageShared, len(stages), synchronizedStageCalls, timing)
	return combined, views, nil
}

// ---------------------------------------------------------------------------
// Per-layer forward
// ---------------------------------------------------------------------------

// fallbackMLP runs the MLP on GPU when ANE is disabled or encounters errors.
// Uses the mlpForwarder interface if the model implements it; otherwise returns
// an error indicating that GPU fallback is unavailable.
func (p *Plane) fallbackMLP(layerIdx int, normalized *mlx.Array) (*mlx.Array, error) {
	if fwd, ok := p.model.(mlpForwarder); ok {
		return fwd.LayerMLPForward(layerIdx, normalized)
	}
	return nil, fmt.Errorf("fallback mlp layer %d: model does not implement mlpForwarder", layerIdx)
}

// layerForward orchestrates a single transformer layer:
// attention on GPU, FFN on ANE, with fused residual/norm paths.
func (p *Plane) layerForward(layerIdx int, x, preparedNorm *mlx.Array, mask any, cache kvcache.Cache) (*mlx.Array, *mlx.Array, []outputView, error) {
	cfg := p.model.Config()
	eps := cfg.RMSNormEps

	// 1. Input norm → attention
	normalized := preparedNorm
	var err error
	if normalized == nil || normalized.IsNil() {
		normWeight := p.weightProv.LayerInputNormWeight(layerIdx)
		normalized, err = rmsNorm(x, normWeight, eps)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("input norm: %w", err)
		}
	}

	// Determine mask type based on linear attention.
	isLinear := p.linearProv != nil && p.linearProv.LayerIsLinear(layerIdx)
	_ = isLinear // mask is already selected by the caller

	attnOut, err := p.attnForwarder.LayerAttentionForward(layerIdx, normalized, mask, cache)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attention: %w", err)
	}

	// If ANE is disabled, fall back to GPU-only path.
	if p.isDisabled() {
		h, err := mlx.Add(x, attnOut, nil)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("attention residual: %w", err)
		}
		postNormWeight := p.weightProv.LayerPostNormWeight(layerIdx)
		postNormed, err := rmsNorm(h, postNormWeight, eps)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("post attention norm: %w", err)
		}
		mlpOut, fallbackErr := p.fallbackMLP(layerIdx, postNormed)
		if fallbackErr != nil {
			return nil, nil, nil, fallbackErr
		}
		out, addErr := mlx.Add(h, mlpOut, nil)
		if addErr != nil {
			return nil, nil, nil, addErr
		}
		return out, nil, nil, nil
	}

	var (
		mlpOut       *mlx.Array
		views        []outputView
		h            *mlx.Array
		postNorm     *mlx.Array
		nextPrepared *mlx.Array
		s            *stage
	)

	isMoE := p.moeProv != nil && p.moeProv.LayerIsMoE(layerIdx)

	if !isMoE {
		// Dense FFN path.
		var stageErr error
		s, stageErr = p.denseStage(layerIdx)
		if stageErr != nil {
			err = stageErr
		} else {
			// Try compiled prepare path.
			if s.densePrepare != nil && s.densePrepare.compiled != nil {
				prepStart := time.Now()
				postNormWeight := p.weightProv.LayerPostNormWeight(layerIdx)
				outs, prepErr := s.densePrepare.compiled.Compute(context.Background(), x, attnOut, postNormWeight)
				prepDur := time.Since(prepStart)
				if prepErr == nil && len(outs) >= 2 {
					preparedH := outs[0]
					packed := outs[1]
					h = preparedH
					prepareTiming := dispatchTiming{Prepare: prepDur, Eval: prepDur, Total: prepDur}
					mlpOut, views, err = p.denseOutputPrepared(s, packed, p.outputDtypeFor(preparedH), prepareTiming)
					packed.Free()
					if err == nil {
						goto postFFN
					}
					prepErr = err
				}
				if s.warn != nil && prepErr != nil {
					s.warn("%s: compiled dense prepare fallback: %v", s.name, prepErr)
				}
			}
			// Try fused add+norm path.
			postNormWeight := p.weightProv.LayerPostNormWeight(layerIdx)
			var handled bool
			var fastErr error
			h, mlpOut, views, handled, fastErr = p.denseOutputFromInputs(s, x, attnOut, postNormWeight, eps, p.outputDtypeFor(x))
			if fastErr != nil {
				err = fastErr
			} else if handled {
				goto postFFN
			} else {
				// Try norm-only path.
				h, err = mlx.Add(x, attnOut, nil)
				if err != nil {
					return nil, nil, nil, fmt.Errorf("attention residual: %w", err)
				}
				mlpOut, views, handled, fastErr = p.denseOutputFromResidual(s, h, postNormWeight, eps, p.outputDtypeFor(h))
				if fastErr != nil {
					err = fastErr
				} else if handled {
					goto postFFN
				} else {
					// Standard path: explicit norm + dispatch.
					postNorm, err = rmsNorm(h, postNormWeight, eps)
					if err != nil {
						return nil, nil, nil, fmt.Errorf("post attention norm: %w", err)
					}
					mlpOut, views, err = p.denseOutput(layerIdx, postNorm)
				}
			}
		}
	} else {
		// MoE path.
		h, err = mlx.Add(x, attnOut, nil)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("attention residual: %w", err)
		}
		postNormWeight := p.weightProv.LayerPostNormWeight(layerIdx)
		postNorm, err = rmsNorm(h, postNormWeight, eps)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("post attention norm: %w", err)
		}
		mlpOut, views, err = p.moeOutput(layerIdx, postNorm)
	}

	// Error fallback: disable ANE and run MLP on GPU.
	if err != nil {
		if h == nil || h.IsNil() {
			h, err = mlx.Add(x, attnOut, nil)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("attention residual: %w", err)
			}
		}
		if postNorm == nil || postNorm.IsNil() {
			postNormWeight := p.weightProv.LayerPostNormWeight(layerIdx)
			postNorm, err = rmsNorm(h, postNormWeight, eps)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("post attention norm: %w", err)
			}
		}
		p.disable(err)
		mlpOut, fallbackErr := p.fallbackMLP(layerIdx, postNorm)
		if fallbackErr != nil {
			return nil, nil, nil, fallbackErr
		}
		out, addErr := mlx.Add(h, mlpOut, nil)
		if addErr != nil {
			return nil, nil, nil, addErr
		}
		return out, nil, nil, nil
	}

postFFN:
	// Ensure residual is computed.
	if h == nil || h.IsNil() {
		h, err = mlx.Add(x, attnOut, nil)
		if err != nil {
			releaseOutputViews(views)
			return nil, nil, nil, fmt.Errorf("attention residual: %w", err)
		}
	}
	// Try fused residual norm for next layer's input.
	if prepared, handled, normErr := p.denseMetalResidualNorm(layerIdx, s, h, mlpOut); normErr != nil {
		releaseOutputViews(views)
		return nil, nil, nil, normErr
	} else if handled {
		nextPrepared = prepared
	}
	// Try fused residual add.
	if out, handled, combineErr := p.denseMetalResidualAdd(s, h, mlpOut); combineErr != nil {
		releaseOutputViews(views)
		return nil, nil, nil, combineErr
	} else if handled {
		return out, nextPrepared, views, nil
	}
	// Plain GPU residual add.
	if h.Dtype() != mlpOut.Dtype() {
		cast, castErr := mlx.Astype(h, mlpOut.Dtype(), nil)
		if castErr != nil {
			releaseOutputViews(views)
			return nil, nil, nil, fmt.Errorf("cast residual: %w", castErr)
		}
		h.Free()
		h = cast
	}
	out, err := mlx.Add(h, mlpOut, nil)
	if err != nil {
		releaseOutputViews(views)
		return nil, nil, nil, err
	}
	return out, nextPrepared, views, nil
}

// ---------------------------------------------------------------------------
// Main eval loop
// ---------------------------------------------------------------------------

// forwardFromEmbeddingsWithHook is the outer loop: embed → layerForward for
// each layer → final norm → lm_head.
func (p *Plane) forwardFromEmbeddingsWithHook(embeddings *mlx.Array, cache kvcache.Cache, hook func(layer int, h *mlx.Array)) (*mlx.Array, kvcache.Cache, error) {
	if !p.useDecodePlane(embeddings, cache) {
		return p.runBaseFromEmbeddings(embeddings, cache)
	}
	caches := p.cacheSlice(cache)
	if caches == nil {
		return p.runBaseFromEmbeddings(embeddings, cache)
	}
	cfg := p.model.Config()
	numLayers := cfg.NumLayers

	// Build masks.
	faMask := any("causal")
	var ssmMask *mlx.Array
	// Check for SSM-style mask from the first linear layer's cache.
	for i := 0; i < numLayers; i++ {
		if p.linearProv == nil || !p.linearProv.LayerIsLinear(i) {
			continue
		}
		if i < len(caches) && caches[i] != nil {
			if ac, ok := caches[i].(*kvcache.Arrays); ok {
				shape := embeddings.Shape()
				ssmMask = ac.MakeMask(shape[1])
			}
		}
		break
	}

	var releases []outputView
	defer releaseOutputViews(releases)
	h := embeddings
	var preparedNorm *mlx.Array

	for i := 0; i < numLayers; i++ {
		// Try direct block dispatch first (skips multiple layers).
		if hook == nil {
			if out, layerReleases, lastLayer, handled, err := p.directBlockOutput(i, h, caches, p.outputDtypeFor(h)); handled {
				if err != nil {
					return nil, nil, fmt.Errorf("layer %d: %w", i, err)
				}
				h = out
				preparedNorm = nil
				if len(layerReleases) != 0 {
					releases = append(releases, layerReleases...)
				}
				i = lastLayer
				continue
			}
		}
		var layerCache kvcache.Cache
		if i < len(caches) {
			layerCache = caches[i]
		}
		isLinear := p.linearProv != nil && p.linearProv.LayerIsLinear(i)
		var mask any
		if isLinear {
			mask = ssmMask
		} else {
			mask = faMask
		}
		next, nextPrepared, layerReleases, err := p.layerForward(i, h, preparedNorm, mask, layerCache)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d: %w", i, err)
		}
		h = next
		preparedNorm = nextPrepared
		if len(layerReleases) != 0 {
			releases = append(releases, layerReleases...)
		}
		if hook != nil {
			hook(i, h)
		}
	}

	// Final norm.
	finalNormWeight := p.headProv.FinalNormWeight()
	out, err := rmsNorm(h, finalNormWeight, cfg.RMSNormEps)
	if err != nil {
		return nil, nil, err
	}

	// LM head projection.
	lmHeadWeight := p.headProv.LMHeadWeight()
	if lmHeadWeight == nil || lmHeadWeight.IsNil() {
		// Tied embeddings: use embed weight as lm_head.
		embedWeight, err := p.embed.EmbedTokens(nil)
		if err != nil || embedWeight == nil || embedWeight.IsNil() {
			return nil, nil, fmt.Errorf("lm head: no weight available")
		}
		lmHeadWeight = embedWeight
	}
	// out [1, 1, dim] @ lmHeadWeight.T [dim, vocab] → [1, 1, vocab]
	lmT, err := mlx.Transpose(lmHeadWeight, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("lm head transpose: %w", err)
	}
	logits, err := mlx.Matmul(out, lmT, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("lm head matmul: %w", err)
	}

	if err := mlx.Eval(logits); err != nil {
		return nil, nil, err
	}
	if err := mlx.Synchronize(nil); err != nil {
		return nil, nil, err
	}
	return logits, cache, nil
}

// ---------------------------------------------------------------------------
// LanguageModel forward methods (ANE-intercepted)
// ---------------------------------------------------------------------------

// ForwardDecode intercepts single-token forward calls and routes FFN through
// ANE when applicable, falling back to the wrapped model otherwise.
func (p *Plane) ForwardDecode(inputs *mlx.Array, cache kvcache.Cache) (*mlx.Array, kvcache.Cache, error) {
	if inputs == nil {
		return nil, nil, fmt.Errorf("input is nil")
	}
	// Normalize 1D inputs to 2D [1, seq_len].
	normalized := inputs
	if inputs.Ndim() == 1 {
		shape := inputs.Shape()
		var err error
		normalized, err = mlx.Reshape(inputs, []int{1, shape[0]}, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("reshape 1d input: %w", err)
		}
	}
	embeddings, err := p.embed.EmbedTokens(normalized)
	if err != nil {
		return nil, nil, fmt.Errorf("embed lookup: %w", err)
	}
	return p.forwardFromEmbeddingsWithHook(embeddings, cache, nil)
}

// ForwardDecodeWithHook is like ForwardDecode but invokes hook(layer, hidden)
// after each layer. When hook is non-nil, direct block dispatch is skipped
// to allow per-layer introspection.
func (p *Plane) ForwardDecodeWithHook(inputs *mlx.Array, cache kvcache.Cache, hook func(layer int, h *mlx.Array)) (*mlx.Array, kvcache.Cache, error) {
	if inputs == nil {
		return nil, nil, fmt.Errorf("input is nil")
	}
	normalized := inputs
	if inputs.Ndim() == 1 {
		shape := inputs.Shape()
		var err error
		normalized, err = mlx.Reshape(inputs, []int{1, shape[0]}, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("reshape 1d input: %w", err)
		}
	}
	embeddings, err := p.embed.EmbedTokens(normalized)
	if err != nil {
		return nil, nil, fmt.Errorf("embed lookup: %w", err)
	}
	return p.forwardFromEmbeddingsWithHook(embeddings, cache, hook)
}

// ForwardFromEmbeddings takes pre-computed embeddings and runs the decode loop.
func (p *Plane) ForwardFromEmbeddings(embeddings *mlx.Array, cache kvcache.Cache) (*mlx.Array, kvcache.Cache, error) {
	return p.forwardFromEmbeddingsWithHook(embeddings, cache, nil)
}
