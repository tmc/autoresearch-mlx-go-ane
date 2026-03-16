//go:build darwin

package decode

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/mlx-go-lm/exp/anehooks"
	"github.com/tmc/mlx-go/mlx"
)

// ---------------------------------------------------------------------------
// Stage pool management
// ---------------------------------------------------------------------------

func (s *stage) primarySlot() *stageSlot {
	if s == nil || len(s.slots) == 0 {
		return nil
	}
	return s.slots[0]
}

func (s *stage) modelDim() int {
	slot := s.primarySlot()
	if slot == nil || slot.stage == nil {
		return 0
	}
	if eval, ok := slot.stage.(stageEvaluator); ok {
		return eval.ModelDim()
	}
	return 0
}

func (s *stage) mapSeq() int {
	slot := s.primarySlot()
	if slot == nil || slot.stage == nil {
		return 0
	}
	if eval, ok := slot.stage.(stageEvaluator); ok {
		return eval.MapSeq()
	}
	return 0
}

func (s *stage) acquireSlot() (*stageSlot, uint64, bool, error) {
	if s == nil {
		return nil, 0, false, fmt.Errorf("ane stage is nil")
	}
	for {
		s.poolMu.Lock()
		for _, slot := range s.slots {
			if !slot.inUse {
				s.leaseSeq++
				slot.inUse = true
				slot.leaseSeq = s.leaseSeq
				lease := slot.leaseSeq
				s.poolMu.Unlock()
				return slot, lease, false, nil
			}
		}
		if len(s.slots) < s.poolDepth {
			slot, err := s.slotBuilder()
			if err != nil {
				s.poolMu.Unlock()
				return nil, 0, false, err
			}
			s.leaseSeq++
			slot.inUse = true
			slot.leaseSeq = s.leaseSeq
			s.slots = append(s.slots, slot)
			lease := slot.leaseSeq
			s.poolMu.Unlock()
			return slot, lease, false, nil
		}
		s.poolMu.Unlock()
		if err := mlx.Synchronize(nil); err != nil {
			return nil, 0, true, fmt.Errorf("%s: synchronize output pool: %w", s.name, err)
		}
		if s.onOutputPoolStall != nil {
			s.onOutputPoolStall()
		}
		s.poolMu.Lock()
		var oldest *stageSlot
		for _, slot := range s.slots {
			if !slot.inUse {
				s.leaseSeq++
				slot.inUse = true
				slot.leaseSeq = s.leaseSeq
				lease := slot.leaseSeq
				s.poolMu.Unlock()
				return slot, lease, true, nil
			}
			if oldest == nil || slot.leaseSeq < oldest.leaseSeq {
				oldest = slot
			}
		}
		if oldest == nil {
			s.poolMu.Unlock()
			return nil, 0, true, fmt.Errorf("%s: output pool is empty", s.name)
		}
		s.leaseSeq++
		oldest.inUse = true
		oldest.leaseSeq = s.leaseSeq
		lease := oldest.leaseSeq
		s.poolMu.Unlock()
		return oldest, lease, true, nil
	}
}

func (s *stage) releaseSlot(slot *stageSlot, lease uint64) {
	if s == nil || slot == nil {
		return
	}
	s.poolMu.Lock()
	defer s.poolMu.Unlock()
	if slot.leaseSeq != lease {
		return
	}
	slot.inUse = false
}

func (s *stage) recordOutputWait(d time.Duration) {
	if s == nil || s.onOutputWait == nil || d <= 0 {
		return
	}
	s.onOutputWait(d)
}

func (s *stage) waitForOutputReady(slot *stageSlot) (time.Duration, error) {
	if s == nil || slot == nil || slot.stage == nil {
		return 0, fmt.Errorf("ane stage is nil")
	}
	sync, ok := slot.stage.(synchronizer)
	if !ok {
		return 0, fmt.Errorf("%s: stage does not implement synchronizer", s.name)
	}
	target := sync.SignalValue()
	switch s.waitMode {
	case waitModeGPUWait:
		if wb, ok := slot.bridge.(waitBridge); ok {
			start := time.Now()
			if err := wb.WaitForANE(nil, target); err != nil {
				return 0, fmt.Errorf("%s: gpu wait for output: %w", s.name, err)
			}
			return time.Since(start), nil
		}
	}
	ev := sync.SignalEvent()
	if ev == nil {
		return 0, fmt.Errorf("%s: signal event is unavailable", s.name)
	}
	start := time.Now()
	if err := ev.WaitCPU(target, 2*time.Second); err != nil {
		return 0, fmt.Errorf("%s: cpu wait for output: %w", s.name, err)
	}
	return time.Since(start), nil
}

func (s *stage) directNormWeight(weight *mlx.Array) (*mlx.Array, error) {
	if s == nil {
		return nil, fmt.Errorf("ane stage is nil")
	}
	if canUseDirectNormWeight(weight, s.modelDim()) {
		return weight, nil
	}
	s.normWeightMu.Lock()
	defer s.normWeightMu.Unlock()
	if s.normWeight32 != nil && !s.normWeight32.IsNil() {
		return s.normWeight32, nil
	}
	if weight == nil || weight.IsNil() {
		return nil, fmt.Errorf("%s: norm weight is nil", s.name)
	}
	cast, err := mlx.Astype(weight, mlx.Float32, nil)
	if err != nil {
		return nil, fmt.Errorf("%s: cast norm weight: %w", s.name, err)
	}
	arr := cast
	rowContig, err := mlx.MLXArrayIsRowContiguous(arr)
	if err != nil {
		arr.Free()
		return nil, fmt.Errorf("%s: norm weight contiguity: %w", s.name, err)
	}
	if !rowContig {
		copied, copyErr := mlx.Copy(arr, nil)
		arr.Free()
		if copyErr != nil {
			return nil, fmt.Errorf("%s: copy norm weight: %w", s.name, copyErr)
		}
		arr = copied
	}
	if !canUseDirectNormWeight(arr, s.modelDim()) {
		arr.Free()
		return nil, fmt.Errorf("%s: norm weight shape=%v incompatible with dim=%d", s.name, weight.Shape(), s.modelDim())
	}
	s.normWeight32 = arr
	return s.normWeight32, nil
}

// ---------------------------------------------------------------------------
// Stage slot buffer management
// ---------------------------------------------------------------------------

func (slot *stageSlot) inputAlias(shape []int) (*mlx.Array, error) {
	if slot == nil || slot.stage == nil || slot.bridge == nil {
		return nil, fmt.Errorf("ane stage slot is nil")
	}
	ba, ok := slot.bridge.(bridgeAliaser)
	if !ok {
		return nil, fmt.Errorf("bridge does not implement bridgeAliaser")
	}
	eval, ok := slot.stage.(stageEvaluator)
	if !ok {
		return nil, fmt.Errorf("stage does not implement stageEvaluator")
	}
	if len(shape) == 3 {
		if slot.directInputAlias != nil && !slot.directInputAlias.IsNil() {
			return slot.directInputAlias, nil
		}
		arr, _, err := ba.AliasWritableFloat32(eval.InputSurface(), shape)
		if err != nil {
			return nil, err
		}
		slot.directInputAlias = arr
		return arr, nil
	}
	if slot.packedInputAlias != nil && !slot.packedInputAlias.IsNil() {
		return slot.packedInputAlias, nil
	}
	arr, _, err := ba.AliasWritableFloat32(eval.InputSurface(), shape)
	if err != nil {
		return nil, err
	}
	slot.packedInputAlias = arr
	return arr, nil
}

func (slot *stageSlot) outputHiddenAlias(dim int) (*mlx.Array, error) {
	if slot == nil || slot.stage == nil || slot.bridge == nil {
		return nil, fmt.Errorf("ane stage slot is nil")
	}
	if slot.outputAlias != nil && !slot.outputAlias.IsNil() {
		return slot.outputAlias, nil
	}
	ba, ok := slot.bridge.(bridgeAliaser)
	if !ok {
		return nil, fmt.Errorf("bridge does not implement bridgeAliaser")
	}
	eval, ok := slot.stage.(stageEvaluator)
	if !ok {
		return nil, fmt.Errorf("stage does not implement stageEvaluator")
	}
	arr, _, err := ba.AliasReadOnlyFloat32(eval.OutputSurface(), []int{1, 1, dim})
	if err != nil {
		return nil, err
	}
	slot.outputAlias = arr
	return arr, nil
}

func (slot *stageSlot) outputBuffer(dim int, dtype mlx.Dtype) (*mlx.Array, error) {
	if slot == nil {
		return nil, fmt.Errorf("ane stage slot is nil")
	}
	shape := []int{1, 1, dim}
	if dtype == mlx.Float32 {
		if slot.outputF32 != nil && !slot.outputF32.IsNil() {
			return slot.outputF32, nil
		}
		arr, err := mlx.Zeros(shape, mlx.Float32, nil)
		if err != nil {
			return nil, fmt.Errorf("%s: allocate float32 output buffer: %w", slot.name, err)
		}
		slot.outputF32 = arr
		return arr, nil
	}
	if slot.outputNative != nil && !slot.outputNative.IsNil() && slot.outputNative.Dtype() == dtype {
		return slot.outputNative, nil
	}
	if slot.outputNative != nil && !slot.outputNative.IsNil() {
		slot.outputNative.Free()
		slot.outputNative = nil
	}
	arr, err := mlx.Zeros(shape, dtype, nil)
	if err != nil {
		return nil, fmt.Errorf("%s: allocate native output buffer: %w", slot.name, err)
	}
	slot.outputNative = arr
	return arr, nil
}

func (slot *stageSlot) nextNormBuffer(dim int) (*mlx.Array, error) {
	if slot == nil {
		return nil, fmt.Errorf("ane stage slot is nil")
	}
	if slot.nextNormF32 != nil && !slot.nextNormF32.IsNil() {
		return slot.nextNormF32, nil
	}
	arr, err := mlx.Zeros([]int{1, 1, dim}, mlx.Float32, nil)
	if err != nil {
		return nil, fmt.Errorf("%s: allocate next-norm buffer: %w", slot.name, err)
	}
	slot.nextNormF32 = arr
	return arr, nil
}

// ---------------------------------------------------------------------------
// Prepared run lifecycle
// ---------------------------------------------------------------------------

func (s *stage) beginPreparedInput(packed *mlx.Array) (*preparedRun, error) {
	run, _, err := s.beginPreparedInputTimed(packed, false, false)
	return run, err
}

func (s *stage) beginPreparedInputTimed(packed *mlx.Array, materialized bool, singleStage bool) (*preparedRun, dispatchTiming, error) {
	if s == nil {
		return nil, dispatchTiming{}, fmt.Errorf("ane stage is nil")
	}
	overallStart := time.Now()
	slot, lease, _, err := s.acquireSlot()
	if err != nil {
		return nil, dispatchTiming{}, err
	}
	slot.mu.Lock()
	ok := false
	defer func() {
		if !ok {
			slot.mu.Unlock()
			s.releaseSlot(slot, lease)
		}
	}()
	eval, evalOK := slot.stage.(stageEvaluator)
	sync, syncOK := slot.stage.(synchronizer)
	if !evalOK || !syncOK {
		return nil, dispatchTiming{}, fmt.Errorf("%s: stage does not implement required interfaces", s.name)
	}
	bt, btOK := slot.bridge.(bridgeTransfer)
	if !btOK {
		return nil, dispatchTiming{}, fmt.Errorf("%s: bridge does not implement bridgeTransfer", s.name)
	}
	wait := sync.WaitEvent()
	hasEvents := wait != nil
	if hasEvents {
		if err := wait.SetSignaledValue(0); err != nil {
			return nil, dispatchTiming{}, fmt.Errorf("%s: reset wait event: %w", s.name, err)
		}
	}
	var timing dispatchTiming
	reason := "other"
	inputShape := eval.InputShape()
	if canUseDirectInput(packed, eval.ModelDim(), eval.MapSeq()) {
		inputShape = packed.Shape()
	}
	aliasStart := time.Now()
	dst, aliasErr := slot.inputAlias(inputShape)
	timing.Alias = time.Since(aliasStart)
	if aliasErr == nil && hasEvents {
		if !materialized {
			reason = "eval"
		}
		if singleStage {
			if scb, scbOK := slot.bridge.(splitCopyBridge); scbOK {
				reason = "copy"
				copyStart := time.Now()
				copyErr := scb.CopyInto(dst, packed, nil)
				timing.Copy = time.Since(copyStart)
				if copyErr == nil {
					finalizeStart := time.Now()
					copyErr = scb.SignalMLXReady(nil, sync.WaitValue())
					timing.Finalize = time.Since(finalizeStart)
				}
				if copyErr == nil {
					ok = true
					timing.Prepare = time.Since(overallStart)
					return &preparedRun{
						stage:           s,
						slot:            slot,
						leaseSeq:        lease,
						synchronized:    true,
						streamCommitted: true,
					}, timing, nil
				}
			}
		}
		reason = "copy"
		copyStart := time.Now()
		copyErr := bt.CopyIntoSignalReady(dst, packed, nil, sync.WaitValue())
		timing.Copy = time.Since(copyStart)
		if copyErr == nil {
			ok = true
			timing.Prepare = time.Since(overallStart)
			return &preparedRun{
				stage:        s,
				slot:         slot,
				leaseSeq:     lease,
				synchronized: true,
			}, timing, nil
		}
	} else if aliasErr != nil {
		reason = "alias"
	}
	s.hostFallbackOnce.Do(func() {
		if s.onHostFallback != nil {
			s.onHostFallback(reason)
		}
		if s.warn != nil {
			s.warn("%s: falling back to host input materialization (%s)", s.name, reason)
		}
	})
	host, err := arrayToFloat32Slice(packed)
	if err != nil {
		return nil, timing, fmt.Errorf("%s: materialize input: %w", s.name, err)
	}
	if err := eval.InputSurface().Write(host); err != nil {
		return nil, timing, fmt.Errorf("%s: write input IOSurface: %w", s.name, err)
	}
	if hasEvents {
		if err := wait.SetSignaledValue(sync.WaitValue()); err != nil {
			return nil, timing, fmt.Errorf("%s: signal wait event after input write: %w", s.name, err)
		}
	}
	ok = true
	timing.Prepare = time.Since(overallStart)
	return &preparedRun{stage: s, slot: slot, leaseSeq: lease}, timing, nil
}

func (s *stage) beginPreparedResidualNormTimed(residual, weight *mlx.Array, eps float32) (*preparedRun, dispatchTiming, error) {
	if s == nil {
		return nil, dispatchTiming{}, fmt.Errorf("ane stage is nil")
	}
	primary := s.primarySlot()
	if primary == nil {
		return nil, dispatchTiming{}, fmt.Errorf("ane stage is nil")
	}
	nb, ok := primary.bridge.(rmsNormBridge)
	if !ok {
		return nil, dispatchTiming{}, fmt.Errorf("%s: rmsnorm bridge is unavailable", s.name)
	}
	overallStart := time.Now()
	slot, lease, _, err := s.acquireSlot()
	if err != nil {
		return nil, dispatchTiming{}, err
	}
	slot.mu.Lock()
	locked := true
	defer func() {
		if locked {
			slot.mu.Unlock()
			s.releaseSlot(slot, lease)
		}
	}()
	sync, syncOK := slot.stage.(synchronizer)
	if !syncOK {
		return nil, dispatchTiming{}, fmt.Errorf("%s: stage does not implement synchronizer", s.name)
	}
	wait := sync.WaitEvent()
	if wait == nil {
		return nil, dispatchTiming{}, fmt.Errorf("%s: wait event is unavailable", s.name)
	}
	if err := wait.SetSignaledValue(0); err != nil {
		return nil, dispatchTiming{}, fmt.Errorf("%s: reset wait event: %w", s.name, err)
	}
	aliasStart := time.Now()
	dst, aliasErr := slot.inputAlias(residual.Shape())
	timing := dispatchTiming{Alias: time.Since(aliasStart)}
	if aliasErr != nil {
		return nil, timing, fmt.Errorf("%s: alias input surface: %w", s.name, aliasErr)
	}
	copyStart := time.Now()
	if err := nb.RMSNormIntoSignalReady(dst, residual, weight, nil, eps, sync.WaitValue()); err != nil {
		return nil, timing, fmt.Errorf("%s: rmsnorm into input IOSurface: %w", s.name, err)
	}
	timing.Copy = time.Since(copyStart)
	timing.Prepare = time.Since(overallStart)
	locked = false
	return &preparedRun{
		stage:           s,
		slot:            slot,
		leaseSeq:        lease,
		synchronized:    true,
		streamCommitted: true,
	}, timing, nil
}

func (s *stage) beginPreparedResidualAddNormTimed(x, y, weight *mlx.Array, eps float32) (*preparedRun, dispatchTiming, error) {
	if s == nil {
		return nil, dispatchTiming{}, fmt.Errorf("ane stage is nil")
	}
	primary := s.primarySlot()
	if primary == nil {
		return nil, dispatchTiming{}, fmt.Errorf("ane stage is nil")
	}
	anb, ok := primary.bridge.(addRMSNormBridge)
	if !ok {
		return nil, dispatchTiming{}, fmt.Errorf("%s: add+rmsnorm bridge is unavailable", s.name)
	}
	overallStart := time.Now()
	slot, lease, _, err := s.acquireSlot()
	if err != nil {
		return nil, dispatchTiming{}, err
	}
	slot.mu.Lock()
	locked := true
	defer func() {
		if locked {
			slot.mu.Unlock()
			s.releaseSlot(slot, lease)
		}
	}()
	sync, syncOK := slot.stage.(synchronizer)
	if !syncOK {
		return nil, dispatchTiming{}, fmt.Errorf("%s: stage does not implement synchronizer", s.name)
	}
	wait := sync.WaitEvent()
	if wait == nil {
		return nil, dispatchTiming{}, fmt.Errorf("%s: wait event is unavailable", s.name)
	}
	if err := wait.SetSignaledValue(0); err != nil {
		return nil, dispatchTiming{}, fmt.Errorf("%s: reset wait event: %w", s.name, err)
	}
	aliasStart := time.Now()
	dst, aliasErr := slot.inputAlias(x.Shape())
	timing := dispatchTiming{Alias: time.Since(aliasStart)}
	if aliasErr != nil {
		return nil, timing, fmt.Errorf("%s: alias input surface: %w", s.name, aliasErr)
	}
	copyStart := time.Now()
	if err := anb.AddRMSNormIntoSignalReady(dst, x, y, weight, nil, eps, sync.WaitValue()); err != nil {
		return nil, timing, fmt.Errorf("%s: add+rmsnorm into input IOSurface: %w", s.name, err)
	}
	timing.Copy = time.Since(copyStart)
	timing.Prepare = time.Since(overallStart)
	locked = false
	return &preparedRun{
		stage:           s,
		slot:            slot,
		leaseSeq:        lease,
		synchronized:    true,
		streamCommitted: true,
	}, timing, nil
}

func (r *preparedRun) abort() {
	if r == nil || r.stage == nil || r.slot == nil {
		return
	}
	if r.inputCleanup != nil {
		_ = r.inputCleanup()
		r.inputCleanup = nil
	}
	r.slot.mu.Unlock()
	r.stage.releaseSlot(r.slot, r.leaseSeq)
}

func (r *preparedRun) finishInput() {
	if r == nil || r.inputCleanup == nil {
		return
	}
	_ = r.inputCleanup()
	r.inputCleanup = nil
}

func (r *preparedRun) start(ctx context.Context) <-chan anehooks.AsyncResult {
	eval := r.slot.stage.(stageEvaluator)
	return eval.EvalPreparedSurfaceAsync(ctx)
}

func (r *preparedRun) outputHiddenView(dtype mlx.Dtype) (*outputView, error) {
	if r == nil || r.stage == nil || r.slot == nil {
		return nil, fmt.Errorf("ane prepared run is nil")
	}
	src, err := r.slot.outputHiddenAlias(r.stage.modelDim())
	if err != nil {
		r.slot.mu.Unlock()
		r.stage.releaseSlot(r.slot, r.leaseSeq)
		return nil, fmt.Errorf("%s: alias output surface: %w", r.stage.name, err)
	}
	switch r.stage.outputMode {
	case outputModeIOSurfaceAlias:
		if dtype != mlx.Float32 {
			break
		}
		if r.stage.onOutputZeroCopy != nil {
			r.stage.onOutputZeroCopy()
		}
		r.slot.mu.Unlock()
		return &outputView{
			arr: src,
			release: func() {
				r.stage.releaseSlot(r.slot, r.leaseSeq)
			},
		}, nil
	case outputModeGPUMaterialize:
		waitDur, err := r.stage.waitForOutputReady(r.slot)
		if err != nil {
			r.slot.mu.Unlock()
			r.stage.releaseSlot(r.slot, r.leaseSeq)
			return nil, err
		}
		r.stage.recordOutputWait(waitDur)
		scb, ok := r.slot.bridge.(splitCopyBridge)
		if !ok {
			break
		}
		dst, err := r.slot.outputBuffer(r.stage.modelDim(), mlx.Float32)
		if err != nil {
			r.slot.mu.Unlock()
			r.stage.releaseSlot(r.slot, r.leaseSeq)
			return nil, err
		}
		if err := scb.CopyInto(dst, src, nil); err != nil {
			r.slot.mu.Unlock()
			r.stage.releaseSlot(r.slot, r.leaseSeq)
			return nil, fmt.Errorf("%s: gpu materialize output: %w", r.stage.name, err)
		}
		r.slot.mu.Unlock()
		return &outputView{
			arr: dst,
			release: func() {
				r.stage.releaseSlot(r.slot, r.leaseSeq)
			},
		}, nil
	case outputModeGPUNative:
		waitDur, err := r.stage.waitForOutputReady(r.slot)
		if err != nil {
			r.slot.mu.Unlock()
			r.stage.releaseSlot(r.slot, r.leaseSeq)
			return nil, err
		}
		r.stage.recordOutputWait(waitDur)
		if dtype == mlx.Float32 {
			dst, bufErr := r.slot.outputBuffer(r.stage.modelDim(), mlx.Float32)
			if bufErr != nil {
				r.slot.mu.Unlock()
				r.stage.releaseSlot(r.slot, r.leaseSeq)
				return nil, bufErr
			}
			if scb, ok := r.slot.bridge.(splitCopyBridge); ok {
				if err := scb.CopyInto(dst, src, nil); err == nil {
					r.slot.mu.Unlock()
					return &outputView{
						arr: dst,
						release: func() {
							r.stage.releaseSlot(r.slot, r.leaseSeq)
						},
					}, nil
				}
			}
		} else {
			out, castErr := mlx.Astype(src, dtype, nil)
			if castErr == nil {
				r.slot.mu.Unlock()
				return &outputView{
					arr: out,
					release: func() {
						out.Free()
						r.stage.releaseSlot(r.slot, r.leaseSeq)
					},
				}, nil
			}
		}
	}
	if r.stage.onOutputCopy != nil {
		r.stage.onOutputCopy()
	}
	out := src
	toFree := []*mlx.Array(nil)
	if dtype != mlx.Float32 {
		out, err = mlx.Astype(out, dtype, nil)
		if err != nil {
			r.slot.mu.Unlock()
			r.stage.releaseSlot(r.slot, r.leaseSeq)
			return nil, fmt.Errorf("%s: cast output: %w", r.stage.name, err)
		}
		toFree = append(toFree, out)
	}
	owned, err := mlx.Copy(out, nil)
	for i := len(toFree) - 1; i >= 0; i-- {
		toFree[i].Free()
	}
	r.slot.mu.Unlock()
	r.stage.releaseSlot(r.slot, r.leaseSeq)
	if err != nil {
		return nil, fmt.Errorf("%s: copy output: %w", r.stage.name, err)
	}
	return &outputView{
		arr: owned,
		release: func() {
			owned.Free()
		},
	}, nil
}

// ---------------------------------------------------------------------------
// Direct block pool management
// ---------------------------------------------------------------------------

func (b *directBlock) primarySlot() *directSlot {
	if b == nil || len(b.slots) == 0 {
		return nil
	}
	return b.slots[0]
}

func (b *directBlock) acquireSlot() (*directSlot, uint64, bool, error) {
	if b == nil {
		return nil, 0, false, fmt.Errorf("ane direct block is nil")
	}
	for {
		b.poolMu.Lock()
		for _, slot := range b.slots {
			if !slot.inUse {
				b.leaseSeq++
				slot.inUse = true
				slot.leaseSeq = b.leaseSeq
				lease := slot.leaseSeq
				b.poolMu.Unlock()
				return slot, lease, false, nil
			}
		}
		if len(b.slots) < b.poolDepth {
			slot, err := b.slotBuilder()
			if err != nil {
				b.poolMu.Unlock()
				return nil, 0, false, err
			}
			b.leaseSeq++
			slot.inUse = true
			slot.leaseSeq = b.leaseSeq
			b.slots = append(b.slots, slot)
			lease := slot.leaseSeq
			b.poolMu.Unlock()
			return slot, lease, false, nil
		}
		b.poolMu.Unlock()
		if err := mlx.Synchronize(nil); err != nil {
			return nil, 0, true, fmt.Errorf("%s: synchronize output pool: %w", b.name, err)
		}
		if b.onOutputPoolStall != nil {
			b.onOutputPoolStall()
		}
		b.poolMu.Lock()
		var oldest *directSlot
		for _, slot := range b.slots {
			if !slot.inUse {
				b.leaseSeq++
				slot.inUse = true
				slot.leaseSeq = b.leaseSeq
				lease := slot.leaseSeq
				b.poolMu.Unlock()
				return slot, lease, true, nil
			}
			if oldest == nil || slot.leaseSeq < oldest.leaseSeq {
				oldest = slot
			}
		}
		if oldest == nil {
			b.poolMu.Unlock()
			return nil, 0, true, fmt.Errorf("%s: output pool is empty", b.name)
		}
		b.leaseSeq++
		oldest.inUse = true
		oldest.leaseSeq = b.leaseSeq
		lease := oldest.leaseSeq
		b.poolMu.Unlock()
		return oldest, lease, true, nil
	}
}

func (b *directBlock) releaseSlot(slot *directSlot, lease uint64) {
	if b == nil || slot == nil {
		return
	}
	b.poolMu.Lock()
	defer b.poolMu.Unlock()
	if slot.leaseSeq != lease {
		return
	}
	slot.inUse = false
}

func (b *directBlock) recordOutputWait(d time.Duration) {
	if b == nil || b.onOutputWait == nil || d <= 0 {
		return
	}
	b.onOutputWait(d)
}

func (b *directBlock) waitForOutputReady(slot *directSlot) (time.Duration, error) {
	if b == nil || slot == nil || slot.block == nil {
		return 0, fmt.Errorf("ane direct block is nil")
	}
	sync, ok := slot.block.(synchronizer)
	if !ok {
		return 0, fmt.Errorf("%s: block does not implement synchronizer", b.name)
	}
	target := sync.SignalValue()
	switch b.waitMode {
	case waitModeGPUWait:
		if wb, ok := slot.bridge.(waitBridge); ok {
			start := time.Now()
			if err := wb.WaitForANE(nil, target); err != nil {
				return 0, fmt.Errorf("%s: gpu wait for output: %w", b.name, err)
			}
			return time.Since(start), nil
		}
	}
	ev := sync.SignalEvent()
	if ev == nil {
		return 0, fmt.Errorf("%s: signal event is unavailable", b.name)
	}
	start := time.Now()
	if err := ev.WaitCPU(target, 2*time.Second); err != nil {
		return 0, fmt.Errorf("%s: cpu wait for output: %w", b.name, err)
	}
	return time.Since(start), nil
}

// ---------------------------------------------------------------------------
// Direct slot buffer management
// ---------------------------------------------------------------------------

func (slot *directSlot) inputHiddenAlias(shape []int) (*mlx.Array, error) {
	if slot == nil || slot.block == nil || slot.bridge == nil {
		return nil, fmt.Errorf("ane direct block slot is nil")
	}
	if slot.inputAlias != nil && !slot.inputAlias.IsNil() {
		return slot.inputAlias, nil
	}
	ba, ok := slot.bridge.(bridgeAliaser)
	if !ok {
		return nil, fmt.Errorf("bridge does not implement bridgeAliaser")
	}
	eval, ok := slot.block.(blockEvaluator)
	if !ok {
		return nil, fmt.Errorf("block does not implement blockEvaluator")
	}
	arr, _, err := ba.AliasWritableFloat32(eval.InputSurface(), shape)
	if err != nil {
		return nil, err
	}
	slot.inputAlias = arr
	return arr, nil
}

func (slot *directSlot) outputHiddenAlias(hiddenDim int) (*mlx.Array, error) {
	if slot == nil || slot.block == nil || slot.bridge == nil {
		return nil, fmt.Errorf("ane direct block slot is nil")
	}
	if slot.outputAlias != nil && !slot.outputAlias.IsNil() {
		return slot.outputAlias, nil
	}
	ba, ok := slot.bridge.(bridgeAliaser)
	if !ok {
		return nil, fmt.Errorf("bridge does not implement bridgeAliaser")
	}
	eval, ok := slot.block.(blockEvaluator)
	if !ok {
		return nil, fmt.Errorf("block does not implement blockEvaluator")
	}
	arr, _, err := ba.AliasReadOnlyFloat32(eval.OutputSurface(), []int{1, 1, hiddenDim})
	if err != nil {
		return nil, err
	}
	slot.outputAlias = arr
	return arr, nil
}

func (slot *directSlot) outputBuffer(hiddenDim int, dtype mlx.Dtype) (*mlx.Array, error) {
	if slot == nil {
		return nil, fmt.Errorf("ane direct block slot is nil")
	}
	shape := []int{1, 1, hiddenDim}
	if dtype == mlx.Float32 {
		if slot.outputF32 != nil && !slot.outputF32.IsNil() {
			return slot.outputF32, nil
		}
		arr, err := mlx.Zeros(shape, mlx.Float32, nil)
		if err != nil {
			return nil, fmt.Errorf("%s: allocate float32 output buffer: %w", slot.name, err)
		}
		slot.outputF32 = arr
		return arr, nil
	}
	if slot.outputNative != nil && !slot.outputNative.IsNil() && slot.outputNative.Dtype() == dtype {
		return slot.outputNative, nil
	}
	if slot.outputNative != nil && !slot.outputNative.IsNil() {
		slot.outputNative.Free()
		slot.outputNative = nil
	}
	arr, err := mlx.Zeros(shape, dtype, nil)
	if err != nil {
		return nil, fmt.Errorf("%s: allocate native output buffer: %w", slot.name, err)
	}
	slot.outputNative = arr
	return arr, nil
}

func (slot *directSlot) zeroRows(name string, kvHeads, headDim int, dtype mlx.Dtype) (*mlx.Array, *mlx.Array, error) {
	if slot == nil {
		return nil, nil, fmt.Errorf("ane direct block slot is nil")
	}
	if slot.zeroK != nil && !slot.zeroK.IsNil() && slot.zeroV != nil && !slot.zeroV.IsNil() {
		return slot.zeroK, slot.zeroV, nil
	}
	if kvHeads <= 0 || headDim <= 0 {
		return nil, nil, fmt.Errorf("%s: invalid kv dims heads=%d head_dim=%d", name, kvHeads, headDim)
	}
	if dtype == 0 {
		dtype = mlx.Float16
	}
	shape := []int{1, kvHeads, 1, headDim}
	zeroK, err := mlx.Zeros(shape, dtype, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: zero K row: %w", name, err)
	}
	zeroV, err := mlx.Zeros(shape, dtype, nil)
	if err != nil {
		zeroK.Free()
		return nil, nil, fmt.Errorf("%s: zero V row: %w", name, err)
	}
	slot.zeroK = zeroK
	slot.zeroV = zeroV
	return slot.zeroK, slot.zeroV, nil
}

func (b *directBlock) outputHiddenView(slot *directSlot, lease uint64, dtype mlx.Dtype) (*outputView, error) {
	if b == nil || slot == nil {
		return nil, fmt.Errorf("ane direct block is nil")
	}
	src, err := slot.outputHiddenAlias(b.hiddenDim)
	if err != nil {
		slot.mu.Unlock()
		b.releaseSlot(slot, lease)
		return nil, err
	}
	switch b.outputMode {
	case outputModeIOSurfaceAlias:
		if dtype != mlx.Float32 {
			break
		}
		if b.onOutputZeroCopy != nil {
			b.onOutputZeroCopy()
		}
		slot.mu.Unlock()
		return &outputView{
			arr: src,
			release: func() {
				b.releaseSlot(slot, lease)
			},
		}, nil
	case outputModeGPUMaterialize:
		waitDur, err := b.waitForOutputReady(slot)
		if err != nil {
			slot.mu.Unlock()
			b.releaseSlot(slot, lease)
			return nil, err
		}
		b.recordOutputWait(waitDur)
		scb, ok := slot.bridge.(splitCopyBridge)
		if !ok {
			break
		}
		dst, err := slot.outputBuffer(b.hiddenDim, mlx.Float32)
		if err != nil {
			slot.mu.Unlock()
			b.releaseSlot(slot, lease)
			return nil, err
		}
		if err := scb.CopyInto(dst, src, nil); err != nil {
			slot.mu.Unlock()
			b.releaseSlot(slot, lease)
			return nil, fmt.Errorf("%s: gpu materialize output: %w", b.name, err)
		}
		slot.mu.Unlock()
		return &outputView{
			arr: dst,
			release: func() {
				b.releaseSlot(slot, lease)
			},
		}, nil
	case outputModeGPUNative:
		waitDur, err := b.waitForOutputReady(slot)
		if err != nil {
			slot.mu.Unlock()
			b.releaseSlot(slot, lease)
			return nil, err
		}
		b.recordOutputWait(waitDur)
		if dtype == mlx.Float32 {
			dst, bufErr := slot.outputBuffer(b.hiddenDim, mlx.Float32)
			if bufErr != nil {
				slot.mu.Unlock()
				b.releaseSlot(slot, lease)
				return nil, bufErr
			}
			if scb, ok := slot.bridge.(splitCopyBridge); ok {
				if err := scb.CopyInto(dst, src, nil); err == nil {
					slot.mu.Unlock()
					return &outputView{
						arr: dst,
						release: func() {
							b.releaseSlot(slot, lease)
						},
					}, nil
				}
			}
		} else {
			out, castErr := mlx.Astype(src, dtype, nil)
			if castErr == nil {
				slot.mu.Unlock()
				return &outputView{
					arr: out,
					release: func() {
						out.Free()
						b.releaseSlot(slot, lease)
					},
				}, nil
			}
		}
	}
	if b.onOutputCopy != nil {
		b.onOutputCopy()
	}
	out := src
	toFree := []*mlx.Array(nil)
	if dtype != mlx.Float32 {
		out, err = mlx.Astype(out, dtype, nil)
		if err != nil {
			slot.mu.Unlock()
			b.releaseSlot(slot, lease)
			return nil, fmt.Errorf("%s: cast output: %w", b.name, err)
		}
		toFree = append(toFree, out)
	}
	owned, err := mlx.Copy(out, nil)
	for i := len(toFree) - 1; i >= 0; i-- {
		toFree[i].Free()
	}
	slot.mu.Unlock()
	b.releaseSlot(slot, lease)
	if err != nil {
		return nil, fmt.Errorf("%s: copy output: %w", b.name, err)
	}
	return &outputView{
		arr: owned,
		release: func() {
			owned.Free()
		},
	}, nil
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

func arrayToFloat32Slice(arr *mlx.Array) ([]float32, error) {
	if arr == nil || arr.IsNil() {
		return nil, fmt.Errorf("array is nil")
	}
	arr32 := arr
	var cleanup func()
	if arr.Dtype() != mlx.Float32 {
		cast, err := mlx.Astype(arr, mlx.Float32, nil)
		if err != nil {
			return nil, fmt.Errorf("cast to float32: %w", err)
		}
		arr32 = cast
		cleanup = func() { cast.Free() }
	}
	if cleanup != nil {
		defer cleanup()
	}
	vals, err := mlx.ToSlice[float32](arr32)
	if err != nil {
		return nil, err
	}
	return append([]float32(nil), vals...), nil
}

func canUseDirectInput(x *mlx.Array, dim, mapSeq int) bool {
	if x == nil || x.IsNil() || mapSeq != 1 || x.Dtype() != mlx.Float32 {
		return false
	}
	shape := x.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != dim {
		return false
	}
	ok, err := mlx.MLXArrayIsRowContiguous(x)
	return err == nil && ok
}

func canUseDirectNormWeight(weight *mlx.Array, dim int) bool {
	if weight == nil || weight.IsNil() || weight.Dtype() != mlx.Float32 {
		return false
	}
	shape := weight.Shape()
	if len(shape) == 0 {
		return false
	}
	total := 1
	for _, n := range shape {
		if n <= 0 {
			return false
		}
		total *= n
	}
	if total != dim {
		return false
	}
	ok, err := mlx.MLXArrayIsRowContiguous(weight)
	return err == nil && ok
}

func canUseDirectResidualInputs(x, y *mlx.Array, dim int) bool {
	if x == nil || x.IsNil() || y == nil || y.IsNil() {
		return false
	}
	if x.Dtype() != y.Dtype() {
		return false
	}
	switch x.Dtype() {
	case mlx.Float16, mlx.Bfloat16, mlx.Float32:
	default:
		return false
	}
	xShape := x.Shape()
	yShape := y.Shape()
	if len(xShape) != 3 || len(yShape) != 3 || xShape[0] != 1 || xShape[1] != 1 || xShape[2] != dim {
		return false
	}
	if yShape[0] != 1 || yShape[1] != 1 || yShape[2] != dim {
		return false
	}
	xOK, xErr := mlx.MLXArrayIsRowContiguous(x)
	if xErr != nil || !xOK {
		return false
	}
	yOK, yErr := mlx.MLXArrayIsRowContiguous(y)
	return yErr == nil && yOK
}
