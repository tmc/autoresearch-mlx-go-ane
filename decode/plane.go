//go:build darwin

package decode

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmc/mlx-go-lm/exp/anehooks"
	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
)

// Plane wraps a models.LanguageModel and intercepts decode steps, routing
// FFN/MoE computation through ANE while keeping attention on GPU.
type Plane struct {
	model         models.LanguageModel
	modelPath     string
	cacheRoot     string
	stageCacheDir string
	mirrorRoot    string
	warn          func(format string, args ...any)
	exp           experimentConfig

	// Cached type-asserted interfaces from the wrapped model.
	attnForwarder layerAttentionForwarder
	weightProv    layerWeightProvider
	headProv      headProvider
	embed         embedder
	moeProv       moeLayerProvider   // nil if model has no MoE layers
	linearProv    linearLayerProvider // nil if model has no linear attention

	mu         sync.RWMutex
	disabled   bool
	disableErr error
	warnOnce   sync.Once

	statsMu sync.Mutex
	stats   Stats

	inputNormMu  sync.Mutex
	inputNormF32 map[int]*mlx.Array

	denseStages     map[int]*stage
	denseErrs       map[int]error
	densePending    map[int]chan struct{}
	directBlocks    map[int]*directBlock
	directErrs      map[int]error
	directPending   map[int]chan struct{}
	directFallbacks map[int]directFallback
	sharedStages    map[int]*stage
	sharedErrs      map[int]error
	sharedPending   map[int]chan struct{}
	expertStages    map[stageKey]*stage
	expertErrs      map[stageKey]error
	expertPending   map[stageKey]chan struct{}
	directSpans     map[int]directSpan
	prewarmOnce     sync.Once
	prewarmErr      error
}

// Stats reports coarse runtime counters for the decode plane.
type Stats struct {
	Enabled       bool
	Disabled      bool
	DisableReason string

	StageBuilds                 int
	StageBuildFailures          int
	DenseStageBuilds            int
	SharedStageBuilds           int
	ExpertStageBuilds           int
	DirectBlockConfigured       int
	DirectBlockConfiguredLayers int
	DirectBlockBuilds           int
	DirectBlockBuildFailures    int
	DirectBlockFallbacks        int
	DirectBlockBuildFallbacks   int
	DirectBlockRuntimeFallbacks int
	DirectBlockDisabledSpans    int
	ArtifactCacheHits           int
	ArtifactCacheMisses         int
	StageBuildDuration          time.Duration
	DenseBuildDuration          time.Duration
	SharedBuildDuration         time.Duration
	ExpertBuildDuration         time.Duration
	DirectBlockBuildDuration    time.Duration
	ArtifactReady               time.Duration
	CompileLoad                 time.Duration
	MapDuration                 time.Duration

	DecodeSteps                int
	SynchronizedDecodeSteps    int
	DenseSteps                 int
	MoESteps                   int
	DirectBlockSteps           int
	DirectBlockLayerExecutions int
	StageCalls                 int
	SynchronizedStageCalls     int
	PrepareDuration            time.Duration
	ANEDuration                time.Duration
	OutputDuration             time.Duration
	TotalStepDuration          time.Duration
	InputAliasDuration         time.Duration
	InputEvalDuration          time.Duration
	InputCopyDuration          time.Duration
	StreamFinalizeDuration     time.Duration
	DensePrepareDuration       time.Duration
	DenseANEDuration           time.Duration
	DenseOutputDuration        time.Duration
	MoEPrepareDuration         time.Duration
	MoEANEDuration             time.Duration
	MoEOutputDuration          time.Duration
	MoERouterDuration          time.Duration
	MoECombineDuration         time.Duration
	DirectBlockPrepareDuration time.Duration
	DirectBlockANEDuration     time.Duration
	DirectBlockOutputDuration  time.Duration
	OutputWaitDuration         time.Duration
	OutputZeroCopySteps        int
	OutputCopyFallbacks        int
	OutputPoolDepth            int
	OutputPoolStalls           int
	HostInputFallbacks         int
	HostFallbackAlias          int
	HostFallbackEval           int
	HostFallbackCopy           int
	HostFallbackOther          int
}

// EffectiveHostInputFallbacks reports the best available count of stage calls
// that fell back to host-side input materialization.
func (s Stats) EffectiveHostInputFallbacks() int {
	if s.StageCalls >= s.SynchronizedStageCalls && s.StageCalls > 0 {
		return s.StageCalls - s.SynchronizedStageCalls
	}
	if n := s.HostFallbackAlias + s.HostFallbackEval + s.HostFallbackCopy + s.HostFallbackOther; n > 0 {
		return n
	}
	return s.HostInputFallbacks
}

var _ models.LanguageModel = (*Plane)(nil)

// Wrap creates a decode Plane around the given model, offloading FFN/MoE
// to the Apple Neural Engine during single-token decode steps.
//
// The model must implement layerAttentionForwarder, layerWeightProvider,
// headProvider, and embedder. MoE and linear attention interfaces are optional.
func Wrap(model models.LanguageModel, opts Options) (*Plane, error) {
	if model == nil {
		return nil, fmt.Errorf("ane decode plane: model is nil")
	}
	cfg := model.Config()
	if cfg == nil {
		return nil, fmt.Errorf("ane decode plane: model config is nil")
	}

	attnFwd, ok := model.(layerAttentionForwarder)
	if !ok {
		return nil, fmt.Errorf("ane decode plane: model %T does not implement layerAttentionForwarder", model)
	}
	wp, ok := model.(layerWeightProvider)
	if !ok {
		return nil, fmt.Errorf("ane decode plane: model %T does not implement layerWeightProvider", model)
	}
	hp, ok := model.(headProvider)
	if !ok {
		return nil, fmt.Errorf("ane decode plane: model %T does not implement headProvider", model)
	}
	emb, ok := model.(embedder)
	if !ok {
		return nil, fmt.Errorf("ane decode plane: model %T does not implement embedder", model)
	}

	moeProv, _ := model.(moeLayerProvider)
	linearProv, _ := model.(linearLayerProvider)

	cacheDir, err := resolveANEDecodeCacheDir(opts.CacheDir)
	if err != nil {
		return nil, err
	}
	stageCacheDir := filepath.Join(cacheDir, "stages")
	mirrorRoot := filepath.Join(cacheDir, "mirror")
	if err := os.MkdirAll(stageCacheDir, 0o755); err != nil {
		return nil, fmt.Errorf("ane decode plane: create cache dir: %w", err)
	}
	if err := os.MkdirAll(mirrorRoot, 0o755); err != nil {
		return nil, fmt.Errorf("ane decode plane: create mirror root: %w", err)
	}
	runtime, err := decodeRuntime()
	if err != nil {
		return nil, err
	}
	runtime.SetModelMirrorRoot(mirrorRoot)

	directSpans := selectDirectBlockSpans(
		cfg.NumLayers,
		fullAttentionDenseOffsets(cfg.NumLayers, moeProv, linearProv),
		os.Getenv("MLXGO_ANE_DIRECT_BLOCK_OFFSETS"),
	)

	plane := &Plane{
		model:         model,
		modelPath:     opts.ModelPath,
		cacheRoot:     cacheDir,
		stageCacheDir: stageCacheDir,
		mirrorRoot:    mirrorRoot,
		warn:          opts.Warn,
		exp:           experimentConfigFromEnv(),
		attnForwarder: attnFwd,
		weightProv:    wp,
		headProv:      hp,
		embed:         emb,
		moeProv:       moeProv,
		linearProv:    linearProv,
		stats:         Stats{Enabled: true},
		inputNormF32:  make(map[int]*mlx.Array),
		denseStages:   make(map[int]*stage),
		denseErrs:     make(map[int]error),
		densePending:  make(map[int]chan struct{}),
		directBlocks:  make(map[int]*directBlock),
		directErrs:    make(map[int]error),
		directPending: make(map[int]chan struct{}),
		directFallbacks: make(map[int]directFallback),
		sharedStages:    make(map[int]*stage),
		sharedErrs:      make(map[int]error),
		sharedPending:   make(map[int]chan struct{}),
		expertStages:    make(map[stageKey]*stage),
		expertErrs:      make(map[stageKey]error),
		expertPending:   make(map[stageKey]chan struct{}),
		directSpans:     directSpans,
	}
	plane.stats.OutputPoolDepth = plane.exp.PoolDepth
	plane.stats.DirectBlockConfigured = len(directSpans)
	plane.stats.DirectBlockConfiguredLayers = directSpanLayerCount(directSpans)
	if err := plane.prepareEagerStages(); err != nil {
		return nil, err
	}
	return plane, nil
}

func resolveANEDecodeCacheDir(dir string) (string, error) {
	dir = strings.TrimSpace(dir)
	if dir != "" {
		return dir, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("ane decode plane: resolve home dir: %w", err)
	}
	return filepath.Join(home, "Library", "Caches", "mlx-go", "ane-decode"), nil
}

// fullAttentionDenseOffsets builds a map from layer index to dense-stage offset
// for layers that use standard (non-linear) attention and dense (non-MoE) FFN.
func fullAttentionDenseOffsets(numLayers int, moeProv moeLayerProvider, linearProv linearLayerProvider) map[int]int {
	offsets := make(map[int]int)
	offset := 0
	for i := 0; i < numLayers; i++ {
		if linearProv != nil && linearProv.LayerIsLinear(i) {
			continue
		}
		if moeProv != nil && moeProv.LayerIsMoE(i) {
			continue
		}
		offsets[i] = offset
		offset++
	}
	return offsets
}

// selectDirectBlockSpans selects which direct-block spans to enable based on
// the MLXGO_ANE_DIRECT_BLOCK_OFFSETS environment variable.
func selectDirectBlockSpans(numLayers int, offsets map[int]int, spec string) map[int]directSpan {
	if len(offsets) == 0 {
		return nil
	}
	spans := buildDirectSpans(numLayers, offsets)
	if len(spans) == 0 {
		return nil
	}
	spec = strings.ToLower(strings.TrimSpace(spec))
	switch spec {
	case "", "default", "auto":
		selected := make(map[int]directSpan)
		for layerIdx, span := range spans {
			if len(span.layers) > 1 {
				selected[layerIdx] = span
			}
		}
		if len(selected) != 0 {
			return selected
		}
		selected = make(map[int]directSpan, len(spans))
		for layerIdx, span := range spans {
			selected[layerIdx] = span
		}
		return selected
	case "multi":
		selected := make(map[int]directSpan)
		for layerIdx, span := range spans {
			if len(span.layers) > 1 {
				selected[layerIdx] = span
			}
		}
		return selected
	case "first":
		selected := make(map[int]directSpan, 1)
		for layerIdx, span := range spans {
			if span.layerOffset == 0 {
				selected[layerIdx] = span
				return selected
			}
		}
		return nil
	case "all":
		selected := make(map[int]directSpan, len(spans))
		for layerIdx, span := range spans {
			selected[layerIdx] = span
		}
		return selected
	}
	allowed := make(map[int]struct{})
	for _, field := range strings.Split(spec, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		off, err := strconv.Atoi(field)
		if err != nil || off < 0 {
			return nil
		}
		allowed[off] = struct{}{}
	}
	if len(allowed) == 0 {
		return nil
	}
	selected := make(map[int]directSpan, len(allowed))
	for layerIdx, span := range spans {
		if _, ok := allowed[span.layerOffset]; ok {
			selected[layerIdx] = span
		}
	}
	return selected
}

// buildDirectSpans groups contiguous dense-attention layers into spans.
func buildDirectSpans(numLayers int, offsets map[int]int) map[int]directSpan {
	if len(offsets) == 0 {
		return nil
	}
	spans := make(map[int]directSpan)
	for i := 0; i < numLayers; {
		offset, ok := offsets[i]
		if !ok {
			i++
			continue
		}
		span := directSpan{
			startLayer:  i,
			layerOffset: offset,
			layers:      []int{i},
		}
		j := i + 1
		nextOffset := offset + 1
		for j < numLayers {
			offsetJ, ok := offsets[j]
			if !ok || offsetJ != nextOffset {
				break
			}
			span.layers = append(span.layers, j)
			j++
			nextOffset++
		}
		spans[span.startLayer] = span
		i = j
	}
	return spans
}

func directSpanLayerCount(spans map[int]directSpan) int {
	total := 0
	for _, span := range spans {
		total += len(span.layers)
	}
	return total
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

func (p *Plane) warnf(format string, args ...any) {
	if p == nil || p.warn == nil {
		return
	}
	p.warn(format, args...)
}

func (p *Plane) disable(err error) {
	if p == nil || err == nil {
		return
	}
	p.mu.Lock()
	alreadyDisabled := p.disabled
	if !p.disabled {
		p.disabled = true
		p.disableErr = err
	}
	p.mu.Unlock()
	p.statsMu.Lock()
	p.stats.Disabled = true
	if p.stats.DisableReason == "" {
		p.stats.DisableReason = err.Error()
	}
	p.statsMu.Unlock()
	if alreadyDisabled {
		return
	}
	p.warnOnce.Do(func() {
		p.warnf("disabling ANE decode plane: %v", err)
	})
}

func (p *Plane) isDisabled() bool {
	if p == nil {
		return true
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.disabled
}

// Stats returns a snapshot of the decode plane runtime counters.
func (p *Plane) Stats() Stats {
	if p == nil {
		return Stats{}
	}
	p.statsMu.Lock()
	defer p.statsMu.Unlock()
	return p.stats
}

// ---------------------------------------------------------------------------
// Stats recording
// ---------------------------------------------------------------------------

func (p *Plane) recordHostInputFallback(reason string) {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	p.stats.HostInputFallbacks++
	switch reason {
	case "alias":
		p.stats.HostFallbackAlias++
	case "eval":
		p.stats.HostFallbackEval++
	case "copy":
		p.stats.HostFallbackCopy++
	default:
		p.stats.HostFallbackOther++
	}
	p.statsMu.Unlock()
}

func (p *Plane) recordOutputZeroCopy() {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	p.stats.OutputZeroCopySteps++
	p.statsMu.Unlock()
}

func (p *Plane) recordOutputCopyFallback() {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	p.stats.OutputCopyFallbacks++
	p.statsMu.Unlock()
}

func (p *Plane) recordOutputPoolStall() {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	p.stats.OutputPoolStalls++
	p.statsMu.Unlock()
}

func (p *Plane) recordOutputWait(d time.Duration) {
	if p == nil || d <= 0 {
		return
	}
	p.statsMu.Lock()
	p.stats.OutputWaitDuration += d
	p.statsMu.Unlock()
}

func (p *Plane) recordInitStats(init anehooks.InitStats) {
	if init.ArtifactCacheHit {
		p.stats.ArtifactCacheHits++
	} else {
		p.stats.ArtifactCacheMisses++
	}
	p.stats.ArtifactReady += init.ArtifactReady
	p.stats.CompileLoad += init.CompileLoad
	p.stats.MapDuration += init.Map
}

func (p *Plane) recordStageBuild(kind stageKind, s *stage, dur time.Duration, err error) {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	defer p.statsMu.Unlock()
	p.stats.StageBuilds++
	p.stats.StageBuildDuration += dur
	if err != nil {
		p.stats.StageBuildFailures++
	}
	switch kind {
	case stageDense:
		p.stats.DenseStageBuilds++
		p.stats.DenseBuildDuration += dur
	case stageShared:
		p.stats.SharedStageBuilds++
		p.stats.SharedBuildDuration += dur
	case stageExpert:
		p.stats.ExpertStageBuilds++
		p.stats.ExpertBuildDuration += dur
	}
	if s != nil {
		if primary := s.primarySlot(); primary != nil && primary.stage != nil {
			if st, ok := primary.stage.(interface{ InitStats() anehooks.InitStats }); ok {
				p.recordInitStats(st.InitStats())
			}
		}
	}
}

func (p *Plane) recordDispatch(kind stageKind, stageCalls, synchronizedStageCalls int, timing dispatchTiming) {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	defer p.statsMu.Unlock()
	p.stats.DecodeSteps++
	if synchronizedStageCalls > 0 {
		p.stats.SynchronizedDecodeSteps++
	}
	p.stats.StageCalls += stageCalls
	p.stats.SynchronizedStageCalls += synchronizedStageCalls
	p.stats.PrepareDuration += timing.Prepare
	p.stats.InputAliasDuration += timing.Alias
	p.stats.InputEvalDuration += timing.Eval
	p.stats.InputCopyDuration += timing.Copy
	p.stats.StreamFinalizeDuration += timing.Finalize
	p.stats.ANEDuration += timing.ANE
	p.stats.OutputWaitDuration += timing.Wait
	p.stats.OutputDuration += timing.Output
	p.stats.TotalStepDuration += timing.Total
	switch kind {
	case stageDense:
		p.stats.DenseSteps++
		p.stats.DensePrepareDuration += timing.Prepare
		p.stats.DenseANEDuration += timing.ANE
		p.stats.DenseOutputDuration += timing.Output
	case stageShared, stageExpert:
		p.stats.MoESteps++
		p.stats.MoEPrepareDuration += timing.Prepare
		p.stats.MoEANEDuration += timing.ANE
		p.stats.MoEOutputDuration += timing.Output
		p.stats.MoERouterDuration += timing.Router
		p.stats.MoECombineDuration += timing.Combine
	}
}

func (p *Plane) recordDirectBlockBuild(block *directBlock, dur time.Duration, err error) {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	defer p.statsMu.Unlock()
	p.stats.StageBuilds++
	p.stats.StageBuildDuration += dur
	p.stats.DirectBlockBuilds++
	p.stats.DirectBlockBuildDuration += dur
	if err != nil {
		p.stats.StageBuildFailures++
		p.stats.DirectBlockBuildFailures++
	}
	if block != nil {
		if primary := block.primarySlot(); primary != nil && primary.block != nil {
			if st, ok := primary.block.(interface{ InitStats() anehooks.InitStats }); ok {
				p.recordInitStats(st.InitStats())
			}
		}
	}
}

func (p *Plane) recordDirectBlockDispatch(numLayers int, timing dispatchTiming) {
	if p == nil {
		return
	}
	p.statsMu.Lock()
	defer p.statsMu.Unlock()
	p.stats.DecodeSteps++
	p.stats.SynchronizedDecodeSteps++
	p.stats.StageCalls++
	p.stats.SynchronizedStageCalls++
	p.stats.PrepareDuration += timing.Prepare
	p.stats.InputCopyDuration += timing.Copy
	p.stats.StreamFinalizeDuration += timing.Finalize
	p.stats.ANEDuration += timing.ANE
	p.stats.OutputWaitDuration += timing.Wait
	p.stats.OutputDuration += timing.Output
	p.stats.TotalStepDuration += timing.Total
	p.stats.DirectBlockSteps++
	p.stats.DirectBlockLayerExecutions += numLayers
	p.stats.DirectBlockPrepareDuration += timing.Prepare
	p.stats.DirectBlockANEDuration += timing.ANE
	p.stats.DirectBlockOutputDuration += timing.Output
}

func (p *Plane) markDirectBlockFallback(span directSpan, build bool, err error) {
	if p == nil || err == nil {
		return
	}
	disabledNew := false
	p.mu.Lock()
	if _, ok := p.directFallbacks[span.startLayer]; !ok {
		p.directFallbacks[span.startLayer] = directFallback{err: err, build: build}
		disabledNew = true
	}
	p.mu.Unlock()
	if !disabledNew {
		return
	}
	p.statsMu.Lock()
	p.stats.DirectBlockFallbacks++
	p.stats.DirectBlockDisabledSpans++
	if build {
		p.stats.DirectBlockBuildFallbacks++
	} else {
		p.stats.DirectBlockRuntimeFallbacks++
	}
	p.statsMu.Unlock()
	p.warnf("falling back from ANE direct block at layer %d: %v", span.startLayer, err)
}

func (p *Plane) shouldUseDirectSpan(layerIdx int) (directSpan, bool) {
	span, ok := p.directSpans[layerIdx]
	if !ok {
		return directSpan{}, false
	}
	p.mu.RLock()
	_, disabled := p.directFallbacks[layerIdx]
	p.mu.RUnlock()
	return span, !disabled
}

// ---------------------------------------------------------------------------
// Prewarm
// ---------------------------------------------------------------------------

func (p *Plane) prepareEagerStages() error {
	return nil
}

// Prewarm eagerly builds ANE stages for all layers outside the timed
// generation path.
func (p *Plane) Prewarm() error {
	if p == nil {
		return nil
	}
	p.prewarmOnce.Do(func() {
		p.prewarmErr = p.prewarm()
	})
	return p.prewarmErr
}

func (p *Plane) prewarm() error {
	if p == nil || p.model == nil {
		return nil
	}
	cfg := p.model.Config()
	for i := 0; i < cfg.NumLayers; i++ {
		if p.isDisabled() {
			return p.disableErr
		}
		isMoE := p.moeProv != nil && p.moeProv.LayerIsMoE(i)
		isLinear := p.linearProv != nil && p.linearProv.LayerIsLinear(i)

		if !isMoE {
			// Dense FFN layer.
			if p.exp.DirectBlock && !isLinear {
				if span, ok := p.shouldUseDirectSpan(i); ok {
					if _, err := p.getDirectBlock(i); err == nil {
						continue
					} else {
						p.markDirectBlockFallback(span, true, fmt.Errorf("prewarm direct block layer %d: %w", i, err))
					}
				}
			}
			if _, err := p.denseStage(i); err != nil {
				return fmt.Errorf("prewarm dense layer %d: %w", i, err)
			}
		} else {
			// MoE layer: prewarm shared expert stage.
			if _, err := p.sharedStage(i); err != nil {
				return fmt.Errorf("prewarm shared expert layer %d: %w", i, err)
			}
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// LanguageModel delegation
// ---------------------------------------------------------------------------

func (p *Plane) Forward(inputs *mlx.Array, cache kvcache.Cache) (*mlx.Array, kvcache.Cache, error) {
	return p.model.Forward(inputs, cache)
}

func (p *Plane) Config() *models.ModelConfig {
	return p.model.Config()
}

func (p *Plane) Sanitize(weights map[string]*mlx.Array) map[string]*mlx.Array {
	return p.model.Sanitize(weights)
}

func (p *Plane) LoadWeights(weightFiles ...string) error {
	return p.model.LoadWeights(weightFiles...)
}
