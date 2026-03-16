//go:build darwin

package decode

import (
	"fmt"
	"math"
	"time"

	"github.com/tmc/mlx-go-lm/exp/anehooks"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
)

// ---------------------------------------------------------------------------
// Runtime interface
// ---------------------------------------------------------------------------

// runtimeStageBuilder is the local consumer interface for the registered
// decode-plane runtime. The runtime is stored as any in anehooks; callers
// type-assert to this interface at the point of use.
type runtimeStageBuilder interface {
	SetModelMirrorRoot(cacheDir string)
	NewQwen35Stage(dim, hidden int, cacheDir string, w1, w3, w2 []float32) (any, any, error)
	NewQwen35DirectBlock(prog any, cfg anehooks.DirectBlockConfig) (any, any, error)
}

// runtimeDirectBlock is the local consumer interface for a direct block
// returned by the runtime. Callers type-assert to blockEvaluator,
// blockStepper, blockResetter, and synchronizer as needed.
type runtimeDirectBlock interface {
	Close()
	SetRoPETables(cosTable, sinTable []float32, headDim, maxSeqLen int) error
}

func decodeRuntime() (runtimeStageBuilder, error) {
	rt := anehooks.DecodePlaneRuntime()
	if rt == nil {
		return nil, fmt.Errorf("ane decode plane runtime unavailable")
	}
	sb, ok := rt.(runtimeStageBuilder)
	if !ok {
		return nil, fmt.Errorf("ane decode plane: runtime %T does not implement stage builder", rt)
	}
	return sb, nil
}

// ---------------------------------------------------------------------------
// Stage acquisition (once-and-cache with pending channel pattern)
// ---------------------------------------------------------------------------

func (p *Plane) denseStage(layerIdx int) (*stage, error) {
	for {
		p.mu.Lock()
		if s := p.denseStages[layerIdx]; s != nil {
			p.mu.Unlock()
			return s, nil
		}
		if err := p.denseErrs[layerIdx]; err != nil {
			p.mu.Unlock()
			return nil, err
		}
		if pending := p.densePending[layerIdx]; pending != nil {
			p.mu.Unlock()
			<-pending
			continue
		}
		pending := make(chan struct{})
		p.densePending[layerIdx] = pending
		p.mu.Unlock()

		start := time.Now()
		s, err := p.buildDenseStage(layerIdx)
		p.recordStageBuild(stageDense, s, time.Since(start), err)

		p.mu.Lock()
		delete(p.densePending, layerIdx)
		if err != nil {
			p.denseErrs[layerIdx] = err
		} else {
			p.denseStages[layerIdx] = s
		}
		close(pending)
		p.mu.Unlock()
		return s, err
	}
}

func (p *Plane) getDirectBlock(layerIdx int) (*directBlock, error) {
	_, ok := p.directSpans[layerIdx]
	if !ok {
		return nil, fmt.Errorf("direct block layer %d is not supported", layerIdx)
	}
	for {
		p.mu.Lock()
		if block := p.directBlocks[layerIdx]; block != nil {
			p.mu.Unlock()
			return block, nil
		}
		if err := p.directErrs[layerIdx]; err != nil {
			p.mu.Unlock()
			return nil, err
		}
		if pending := p.directPending[layerIdx]; pending != nil {
			p.mu.Unlock()
			<-pending
			continue
		}
		pending := make(chan struct{})
		p.directPending[layerIdx] = pending
		p.mu.Unlock()

		span := p.directSpans[layerIdx]
		start := time.Now()
		block, err := p.buildDirectBlock(span)
		p.recordDirectBlockBuild(block, time.Since(start), err)

		p.mu.Lock()
		delete(p.directPending, layerIdx)
		if err != nil {
			p.directErrs[layerIdx] = err
		} else {
			p.directBlocks[layerIdx] = block
		}
		close(pending)
		p.mu.Unlock()
		return block, err
	}
}

func (p *Plane) sharedStage(layerIdx int) (*stage, error) {
	for {
		p.mu.Lock()
		if s := p.sharedStages[layerIdx]; s != nil {
			p.mu.Unlock()
			return s, nil
		}
		if err := p.sharedErrs[layerIdx]; err != nil {
			p.mu.Unlock()
			return nil, err
		}
		if pending := p.sharedPending[layerIdx]; pending != nil {
			p.mu.Unlock()
			<-pending
			continue
		}
		pending := make(chan struct{})
		p.sharedPending[layerIdx] = pending
		p.mu.Unlock()

		start := time.Now()
		s, err := p.buildSharedExpertStage(layerIdx)
		p.recordStageBuild(stageShared, s, time.Since(start), err)

		p.mu.Lock()
		delete(p.sharedPending, layerIdx)
		if err != nil {
			p.sharedErrs[layerIdx] = err
		} else {
			p.sharedStages[layerIdx] = s
		}
		close(pending)
		p.mu.Unlock()
		return s, err
	}
}

func (p *Plane) expertStage(layerIdx, expert int) (*stage, error) {
	key := stageKey{layer: layerIdx, expert: expert}
	for {
		p.mu.Lock()
		if s := p.expertStages[key]; s != nil {
			p.mu.Unlock()
			return s, nil
		}
		if err := p.expertErrs[key]; err != nil {
			p.mu.Unlock()
			return nil, err
		}
		if pending := p.expertPending[key]; pending != nil {
			p.mu.Unlock()
			<-pending
			continue
		}
		pending := make(chan struct{})
		p.expertPending[key] = pending
		p.mu.Unlock()

		s, err := p.buildExpertStage(layerIdx, expert)

		p.mu.Lock()
		delete(p.expertPending, key)
		if err != nil {
			p.expertErrs[key] = err
		} else {
			p.expertStages[key] = s
		}
		close(pending)
		p.mu.Unlock()
		return s, err
	}
}

// ---------------------------------------------------------------------------
// Stage building (uses consumer interfaces, not model internals)
// ---------------------------------------------------------------------------

func (p *Plane) buildDenseStage(layerIdx int) (*stage, error) {
	gate, up, down, err := p.weightProv.LayerFFNWeights(layerIdx)
	if err != nil {
		return nil, fmt.Errorf("dense layer %d weights: %w", layerIdx, err)
	}
	cfg := p.model.Config()
	return newStage(
		fmt.Sprintf("dense layer %d", layerIdx),
		p.stageCacheDir,
		cfg.HiddenSize,
		cfg.IntermediateSize,
		p.exp.PoolDepth,
		gate, up, down,
		p.exp.OutputMode,
		p.exp.WaitMode,
		p.recordHostInputFallback,
		p.recordOutputZeroCopy,
		p.recordOutputCopyFallback,
		p.recordOutputWait,
		p.recordOutputPoolStall,
		p.warn,
	)
}

func (p *Plane) buildDirectBlock(span directSpan) (*directBlock, error) {
	type irLowerable interface {
		LowerDecodeModelIR(opts interface{}) (interface{}, error)
	}
	lowerable, ok := p.model.(irLowerable)
	if !ok {
		return nil, fmt.Errorf("direct block: model %T does not implement LowerDecodeModelIR", p.model)
	}
	runtime, err := decodeRuntime()
	if err != nil {
		return nil, err
	}
	cfg := p.model.Config()
	maxSeqLen := directBlockMaxSeqLen(cfg)

	headDim := cfg.HeadDim
	if headDim <= 0 && cfg.NumHeads > 0 && cfg.HiddenSize%cfg.NumHeads == 0 {
		headDim = cfg.HiddenSize / cfg.NumHeads
	}
	numAttnHeads := cfg.NumHeads
	numKVHeads := 0
	if cfg.NumKeyValueHeads != nil {
		numKVHeads = *cfg.NumKeyValueHeads
	}
	if numKVHeads <= 0 {
		numKVHeads = numAttnHeads
	}
	if numAttnHeads <= 0 {
		numAttnHeads = numKVHeads
	}
	ropeTheta := cfg.RopeTheta
	if ropeTheta <= 0 {
		ropeTheta = 10000
	}
	cosTable, sinTable := buildRoPETables(maxSeqLen, headDim, ropeTheta)

	prog, err := lowerable.LowerDecodeModelIR(struct {
		LayerOffset   int
		MaxLayers     int
		MaxSeqLen     int
		IncludeLMHead bool
		SkipFinalNorm bool
		StatefulKV    bool
		AttentionMask bool
	}{
		LayerOffset:   span.layerOffset,
		MaxLayers:     len(span.layers),
		MaxSeqLen:     maxSeqLen,
		IncludeLMHead: false,
		SkipFinalNorm: true,
		StatefulKV:    true,
		AttentionMask: false,
	})
	if err != nil {
		return nil, fmt.Errorf("lower direct block layer %d: %w", span.startLayer, err)
	}

	slotBuilder := func() (*directSlot, error) {
		blockAny, bridgeAny, err := runtime.NewQwen35DirectBlock(prog, anehooks.DirectBlockConfig{
			HiddenDim:      cfg.HiddenSize,
			VocabSize:      cfg.VocabSize,
			OutputDim:      cfg.HiddenSize,
			MaxSeqLen:      maxSeqLen,
			SelectedLayers: len(span.layers),
		})
		if err != nil {
			return nil, err
		}
		rdb, ok := blockAny.(runtimeDirectBlock)
		if !ok {
			return nil, fmt.Errorf("direct block runtime returned %T, want runtimeDirectBlock", blockAny)
		}
		if err := rdb.SetRoPETables(cosTable, sinTable, headDim, maxSeqLen); err != nil {
			rdb.Close()
			return nil, fmt.Errorf("configure direct block rope tables layer %d: %w", span.startLayer, err)
		}
		return &directSlot{
			name:   fmt.Sprintf("direct block layer %d", span.startLayer),
			block:  blockAny,
			bridge: bridgeAny,
		}, nil
	}
	first, err := slotBuilder()
	if err != nil {
		return nil, err
	}
	return &directBlock{
		name:              fmt.Sprintf("direct block layer %d", span.startLayer),
		layers:            append([]int(nil), span.layers...),
		hiddenDim:         cfg.HiddenSize,
		maxSeqLen:         maxSeqLen,
		attnHeads:         numAttnHeads,
		kvHeads:           numKVHeads,
		headDim:           headDim,
		outputMode:        p.exp.OutputMode,
		waitMode:          p.exp.WaitMode,
		onOutputZeroCopy:  p.recordOutputZeroCopy,
		onOutputCopy:      p.recordOutputCopyFallback,
		onOutputWait:      p.recordOutputWait,
		onOutputPoolStall: p.recordOutputPoolStall,
		poolDepth:         maxInt(1, p.exp.PoolDepth),
		slots:             []*directSlot{first},
		slotBuilder:       slotBuilder,
	}, nil
}

func (p *Plane) buildSharedExpertStage(layerIdx int) (*stage, error) {
	if p.moeProv == nil {
		return nil, fmt.Errorf("shared expert stage layer %d: model has no MoE provider", layerIdx)
	}
	gate, up, down, err := p.moeProv.LayerSharedExpertWeights(layerIdx)
	if err != nil {
		return nil, fmt.Errorf("shared expert layer %d weights: %w", layerIdx, err)
	}
	cfg := p.model.Config()
	// Shared expert intermediate size may differ; use the weights length to
	// infer the intermediate dimension (gate weight is [intermediate, hidden]).
	intermediateSize := len(gate) / cfg.HiddenSize
	if intermediateSize <= 0 {
		intermediateSize = cfg.IntermediateSize
	}
	return newStage(
		fmt.Sprintf("shared expert layer %d", layerIdx),
		p.stageCacheDir,
		cfg.HiddenSize,
		intermediateSize,
		p.exp.PoolDepth,
		gate, up, down,
		p.exp.OutputMode,
		p.exp.WaitMode,
		p.recordHostInputFallback,
		p.recordOutputZeroCopy,
		p.recordOutputCopyFallback,
		p.recordOutputWait,
		p.recordOutputPoolStall,
		p.warn,
	)
}

func (p *Plane) buildExpertStage(layerIdx, expert int) (*stage, error) {
	if p.moeProv == nil {
		return nil, fmt.Errorf("expert stage layer %d: model has no MoE provider", layerIdx)
	}
	gate, up, down, err := p.moeProv.LayerExpertWeights(layerIdx, expert)
	if err != nil {
		return nil, fmt.Errorf("expert %d layer %d weights: %w", expert, layerIdx, err)
	}
	cfg := p.model.Config()
	intermediateSize := len(gate) / cfg.HiddenSize
	if intermediateSize <= 0 {
		intermediateSize = cfg.IntermediateSize
	}
	start := time.Now()
	s, err := newStage(
		fmt.Sprintf("expert layer %d expert %d", layerIdx, expert),
		p.stageCacheDir,
		cfg.HiddenSize,
		intermediateSize,
		p.exp.PoolDepth,
		gate, up, down,
		p.exp.OutputMode,
		p.exp.WaitMode,
		p.recordHostInputFallback,
		p.recordOutputZeroCopy,
		p.recordOutputCopyFallback,
		p.recordOutputWait,
		p.recordOutputPoolStall,
		p.warn,
	)
	p.recordStageBuild(stageExpert, s, time.Since(start), err)
	return s, err
}

func newStage(
	name, cacheDir string,
	dim, hidden int,
	poolDepth int,
	w1, w3, w2 []float32,
	om outputMode,
	wm waitMode,
	onHostFallback func(reason string),
	onOutputZeroCopy func(),
	onOutputCopy func(),
	onOutputWait func(time.Duration),
	onOutputPoolStall func(),
	warn func(format string, args ...any),
) (*stage, error) {
	runtime, err := decodeRuntime()
	if err != nil {
		return nil, err
	}
	slotBuilder := func() (*stageSlot, error) {
		aneStage, bridge, err := runtime.NewQwen35Stage(dim, hidden, cacheDir, w1, w3, w2)
		if err != nil {
			return nil, err
		}
		return &stageSlot{
			name:   name,
			stage:  aneStage,
			bridge: bridge,
		}, nil
	}
	slot, err := slotBuilder()
	if err != nil {
		return nil, err
	}
	return &stage{
		name:              name,
		onHostFallback:    onHostFallback,
		outputMode:        om,
		waitMode:          wm,
		onOutputZeroCopy:  onOutputZeroCopy,
		onOutputCopy:      onOutputCopy,
		onOutputWait:      onOutputWait,
		onOutputPoolStall: onOutputPoolStall,
		warn:              warn,
		poolDepth:         maxInt(1, poolDepth),
		slots:             []*stageSlot{slot},
		slotBuilder:       slotBuilder,
	}, nil
}

// ---------------------------------------------------------------------------
// RoPE table construction
// ---------------------------------------------------------------------------

func directBlockMaxSeqLen(cfg *models.ModelConfig) int {
	if cfg == nil {
		return 256
	}
	maxSeqLen := cfg.MaxPositionEmbeddings
	if maxSeqLen <= 0 || maxSeqLen > 256 {
		maxSeqLen = 256
	}
	if maxSeqLen < 2 {
		maxSeqLen = 2
	}
	return maxSeqLen
}

func buildRoPETables(maxSeqLen, headDim int, theta float64) ([]float32, []float32) {
	cosTable := make([]float32, maxSeqLen*headDim)
	sinTable := make([]float32, maxSeqLen*headDim)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < headDim; i++ {
			pair := i / 2
			freq := math.Pow(theta, -2*float64(pair)/float64(headDim))
			angle := float64(pos) * freq
			cosTable[pos*headDim+i] = float32(math.Cos(angle))
			sinTable[pos*headDim+i] = float32(math.Sin(angle))
		}
	}
	return cosTable, sinTable
}

// ExpandRoPERow repeats a single RoPE position row for multi-head use.
// Each attention head receives the same positional encoding, so the row is
// concatenated repeats times. If repeats <= 1, a copy of the input is returned.
func ExpandRoPERow(row []float32, repeats int) []float32 {
	if repeats <= 1 {
		return append([]float32(nil), row...)
	}
	out := make([]float32, 0, len(row)*repeats)
	for i := 0; i < repeats; i++ {
		out = append(out, row...)
	}
	return out
}
