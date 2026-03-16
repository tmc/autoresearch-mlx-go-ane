//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
)

const surfaceEspressoFFNArtifactVersion = "surface_espresso_ffn_v1"

var surfaceEspressoFFNArtifactLocks sync.Map

var (
	surfaceEspressoModelCacheMu sync.Mutex
	surfaceEspressoModelCache   = map[string]*surfaceEspressoModelCacheEntry{}
)

type surfaceEspressoModelCacheEntry struct {
	model *ANEClientMILModel
	refs  int
}

type surfaceEspressoFFNFingerprint struct {
	Version   string `json:"version"`
	Dim       int    `json:"dim"`
	HiddenDim int    `json:"hidden_dim"`
	Seq       int    `json:"seq"`
	W1        uint64 `json:"w1"`
	W3        uint64 `json:"w3"`
	W2        uint64 `json:"w2"`
}

// SurfaceEspressoInitStats reports where stage initialization time was spent.
type SurfaceEspressoInitStats struct {
	ArtifactDir      string
	ArtifactCacheHit bool
	ArtifactReady    time.Duration
	CompileLoad      time.Duration
	Map              time.Duration
	Total            time.Duration
}

// SurfaceEspressoFFN is a reusable decode-stage FFN body offload path.
//
// The stage owns a file-backed Espresso model, persistent IOSurfaces, and a
// mapped multi-surface request. It consumes normalized hidden state [1,1,Dim]
// and writes only the FFN body output [1,1,Dim].
type SurfaceEspressoFFN struct {
	mu        sync.Mutex
	client    *ANEClientMILModel
	release   func()
	input     *IOSurfaceFloat32
	output    *IOSurfaceFloat32
	plan      *MultiSurfaceEvalPlan
	initStats SurfaceEspressoInitStats
	dim       int
	hiddenDim int
	seq       int
	closed    bool
}

// NewSurfaceEspressoFFN constructs a surface-backed FFN body stage using the
// in-repo Espresso generator.
//
// The current generator supports singleton decode only, so seq must be 1.
func NewSurfaceEspressoFFN(
	dim, hiddenDim, seq int,
	cacheRoot string,
	w1, w3, w2 []float32,
	cfg MultiSurfaceEvalPlanConfig,
) (*SurfaceEspressoFFN, error) {
	if dim <= 0 || hiddenDim <= 0 {
		return nil, fmt.Errorf("new surface espresso ffn: invalid dims dim=%d hiddenDim=%d", dim, hiddenDim)
	}
	if seq != 1 {
		return nil, fmt.Errorf("new surface espresso ffn: seq=%d want=1", seq)
	}
	if len(w1) != hiddenDim*dim {
		return nil, fmt.Errorf("new surface espresso ffn: w1 len=%d want=%d", len(w1), hiddenDim*dim)
	}
	if len(w3) != hiddenDim*dim {
		return nil, fmt.Errorf("new surface espresso ffn: w3 len=%d want=%d", len(w3), hiddenDim*dim)
	}
	if len(w2) != dim*hiddenDim {
		return nil, fmt.Errorf("new surface espresso ffn: w2 len=%d want=%d", len(w2), dim*hiddenDim)
	}

	totalStart := time.Now()
	key, err := surfaceEspressoFFNCacheKey(dim, hiddenDim, seq, w1, w3, w2)
	if err != nil {
		return nil, fmt.Errorf("new surface espresso ffn: compute cache key: %w", err)
	}
	artifactStart := time.Now()
	dir, cacheHit, err := ensureSurfaceEspressoFFNDir(cacheRoot, key, dim, hiddenDim, seq, w1, w3, w2)
	if err != nil {
		return nil, fmt.Errorf("new surface espresso ffn: ensure artifact dir: %w", err)
	}
	artifactReady := time.Since(artifactStart)
	compileStart := time.Now()
	model, release, err := acquireSurfaceEspressoClientModel(key, dir)
	if err != nil {
		return nil, fmt.Errorf("new surface espresso ffn: compile/load espresso: %w", err)
	}
	compileLoad := time.Since(compileStart)

	inSurf, err := NewIOSurfaceFloat32(dim * seq)
	if err != nil {
		release()
		return nil, fmt.Errorf("new surface espresso ffn: allocate input surface: %w", err)
	}
	outSurf, err := NewIOSurfaceFloat32(dim * seq)
	if err != nil {
		inSurf.Close()
		release()
		return nil, fmt.Errorf("new surface espresso ffn: allocate output surface: %w", err)
	}
	if err := inSurf.Write(make([]float32, dim*seq)); err != nil {
		outSurf.Close()
		inSurf.Close()
		release()
		return nil, fmt.Errorf("new surface espresso ffn: initialize input surface: %w", err)
	}
	if err := outSurf.Write(make([]float32, dim*seq)); err != nil {
		outSurf.Close()
		inSurf.Close()
		release()
		return nil, fmt.Errorf("new surface espresso ffn: initialize output surface: %w", err)
	}

	inSym, err := firstProcedureSymbolIndex(model.model, true)
	if err != nil {
		outSurf.Close()
		inSurf.Close()
		release()
		return nil, fmt.Errorf("new surface espresso ffn: resolve input symbol index: %w", err)
	}
	outSym, err := firstProcedureSymbolIndex(model.model, false)
	if err != nil {
		outSurf.Close()
		inSurf.Close()
		release()
		return nil, fmt.Errorf("new surface espresso ffn: resolve output symbol index: %w", err)
	}

	dflt := DefaultMultiSurfaceEvalPlanConfig()
	if cfg == (MultiSurfaceEvalPlanConfig{}) {
		cfg = dflt
	} else {
		if cfg.QoS == 0 {
			cfg.QoS = dflt.QoS
		}
		if cfg.WaitValue == 0 {
			cfg.WaitValue = dflt.WaitValue
		}
		if cfg.SignalValue == 0 {
			cfg.SignalValue = dflt.SignalValue
		}
	}

	mapStart := time.Now()
	plan, err := NewMultiSurfaceEvalPlanWithClientModel(
		model,
		[]SurfaceBinding{{Surface: inSurf, SymbolIndex: inSym}},
		[]SurfaceBinding{{Surface: outSurf, SymbolIndex: outSym}},
		cfg,
	)
	if err != nil && !cfg.DisableCacheMapping {
		retry := cfg
		retry.DisableCacheMapping = true
		plan, err = NewMultiSurfaceEvalPlanWithClientModel(
			model,
			[]SurfaceBinding{{Surface: inSurf, SymbolIndex: inSym}},
			[]SurfaceBinding{{Surface: outSurf, SymbolIndex: outSym}},
			retry,
		)
	}
	if err != nil {
		outSurf.Close()
		inSurf.Close()
		release()
		return nil, fmt.Errorf("new surface espresso ffn: map plan: %w", err)
	}
	mapTime := time.Since(mapStart)

	return &SurfaceEspressoFFN{
		client:  model,
		release: release,
		input:   inSurf,
		output:  outSurf,
		plan:    plan,
		initStats: SurfaceEspressoInitStats{
			ArtifactDir:      dir,
			ArtifactCacheHit: cacheHit,
			ArtifactReady:    artifactReady,
			CompileLoad:      compileLoad,
			Map:              mapTime,
			Total:            time.Since(totalStart),
		},
		dim:       dim,
		hiddenDim: hiddenDim,
		seq:       seq,
	}, nil
}

func ensureSurfaceEspressoFFNDir(
	cacheRoot string,
	key string,
	dim, hiddenDim, seq int,
	w1, w3, w2 []float32,
) (string, bool, error) {
	if cacheRoot == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			base = os.TempDir()
		}
		cacheRoot = filepath.Join(base, "mlx-go", "ane-decode")
	}
	dir := filepath.Join(cacheRoot, "espresso-ffn", key)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", false, fmt.Errorf("create cache dir %q: %w", dir, err)
	}
	lock, _ := surfaceEspressoFFNArtifactLocks.LoadOrStore(key, &sync.Mutex{})
	mu := lock.(*sync.Mutex)
	mu.Lock()
	defer mu.Unlock()
	netPath := filepath.Join(dir, "model.espresso.net")
	shapePath := filepath.Join(dir, "model.espresso.shape")
	weightsPath := filepath.Join(dir, "model.espresso.weights")
	if fileExists(netPath) && fileExists(shapePath) && fileExists(weightsPath) {
		return dir, true, nil
	}
	if err := GenerateFFNEspressoDir(dir, dim, hiddenDim, w1, w3, w2); err != nil {
		return "", false, err
	}
	return dir, false, nil
}

func acquireSurfaceEspressoClientModel(key, dir string) (*ANEClientMILModel, func(), error) {
	surfaceEspressoModelCacheMu.Lock()
	if entry := surfaceEspressoModelCache[key]; entry != nil && entry.model != nil {
		entry.refs++
		model := entry.model
		surfaceEspressoModelCacheMu.Unlock()
		return model, func() { releaseSurfaceEspressoClientModel(key) }, nil
	}
	surfaceEspressoModelCacheMu.Unlock()

	client, err := openPreferredANEClient()
	if err != nil {
		return nil, nil, fmt.Errorf("open client: %w", err)
	}
	model, err := CompileAndLoadEspresso(client, dir, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		return nil, nil, err
	}

	surfaceEspressoModelCacheMu.Lock()
	defer surfaceEspressoModelCacheMu.Unlock()
	if entry := surfaceEspressoModelCache[key]; entry != nil && entry.model != nil {
		entry.refs++
		// Another goroutine populated the cache while we compiled this model.
		model.Close()
		return entry.model, func() { releaseSurfaceEspressoClientModel(key) }, nil
	}
	surfaceEspressoModelCache[key] = &surfaceEspressoModelCacheEntry{model: model, refs: 1}
	return model, func() { releaseSurfaceEspressoClientModel(key) }, nil
}

func releaseSurfaceEspressoClientModel(key string) {
	surfaceEspressoModelCacheMu.Lock()
	entry := surfaceEspressoModelCache[key]
	if entry == nil {
		surfaceEspressoModelCacheMu.Unlock()
		return
	}
	entry.refs--
	if entry.refs > 0 {
		surfaceEspressoModelCacheMu.Unlock()
		return
	}
	delete(surfaceEspressoModelCache, key)
	model := entry.model
	surfaceEspressoModelCacheMu.Unlock()
	if model != nil {
		model.Close()
	}
}

func surfaceEspressoFFNCacheKey(dim, hiddenDim, seq int, w1, w3, w2 []float32) (string, error) {
	fp := surfaceEspressoFFNFingerprint{
		Version:   surfaceEspressoFFNArtifactVersion,
		Dim:       dim,
		HiddenDim: hiddenDim,
		Seq:       seq,
		W1:        hashFloat32Slice(w1),
		W3:        hashFloat32Slice(w3),
		W2:        hashFloat32Slice(w2),
	}
	data, err := json.Marshal(fp)
	if err != nil {
		return "", fmt.Errorf("marshal cache key: %w", err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func firstProcedureSymbolIndex(base objectivec.IObject, input bool) (int, error) {
	if base == nil || base.GetID() == 0 {
		return 0, fmt.Errorf("model is nil")
	}
	sel := "outputSymbolIndicesForProcedureIndex:"
	if input {
		sel = "inputSymbolIndicesForProcedureIndex:"
	}
	containerID := objc.Send[objc.ID](base.GetID(), objc.Sel(sel), uint32(0))
	if containerID == 0 {
		return 0, fmt.Errorf("%s returned nil", sel)
	}
	if idx, ok := firstSymbolIndex(objectivec.ObjectFromID(containerID)); ok {
		return idx, nil
	}
	return 0, fmt.Errorf("%s returned no indices", sel)
}

// ModelDim returns the FFN model dimension.
func (s *SurfaceEspressoFFN) ModelDim() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dim
}

// Seq returns the logical decode sequence length accepted by the stage.
func (s *SurfaceEspressoFFN) Seq() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seq
}

// MapSeq returns the mapped sequence length. Espresso FFN currently runs only
// singleton decode, so this is always 1.
func (s *SurfaceEspressoFFN) MapSeq() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seq
}

// InputShape returns the mapped input surface shape as [channels, seq].
func (s *SurfaceEspressoFFN) InputShape() []int {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return []int{s.dim, s.seq}
}

// OutputShape returns the mapped output surface shape as [channels, seq].
func (s *SurfaceEspressoFFN) OutputShape() []int {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return []int{s.dim, s.seq}
}

// InitStats returns one-time initialization timing for the stage.
func (s *SurfaceEspressoFFN) InitStats() SurfaceEspressoInitStats {
	if s == nil {
		return SurfaceEspressoInitStats{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.initStats
}

// InputSurface returns the input IOSurface.
func (s *SurfaceEspressoFFN) InputSurface() *IOSurfaceFloat32 {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.input
}

// OutputSurface returns the output IOSurface.
func (s *SurfaceEspressoFFN) OutputSurface() *IOSurfaceFloat32 {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.output
}

// WaitEvent returns the attached Metal->ANE wait event when configured.
func (s *SurfaceEspressoFFN) WaitEvent() *SharedEvent {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return nil
	}
	return s.plan.WaitEvent()
}

// WaitValue returns the configured Metal->ANE wait target value.
func (s *SurfaceEspressoFFN) WaitValue() uint64 {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return 0
	}
	return s.plan.WaitValue()
}

// SignalEvent returns the attached ANE->Metal signal event when configured.
func (s *SurfaceEspressoFFN) SignalEvent() *SharedEvent {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return nil
	}
	return s.plan.SignalEvent()
}

// SignalPort returns the ANE->Metal signal-event port when configured.
func (s *SurfaceEspressoFFN) SignalPort() uint32 {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return 0
	}
	return s.plan.SignalEventPort()
}

// SignalValue returns the configured ANE->Metal signal target value.
func (s *SurfaceEspressoFFN) SignalValue() uint64 {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return 0
	}
	return s.plan.SignalValue()
}

// NewOutputMetalBufferBinding creates a reusable no-copy Metal buffer binding
// over the output IOSurface.
func (s *SurfaceEspressoFFN) NewOutputMetalBufferBinding(device metal.MTLDevice) (*MetalBufferBinding, error) {
	if s == nil {
		return nil, fmt.Errorf("surface espresso ffn metal buffer binding: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("surface espresso ffn metal buffer binding: stage is closed")
	}
	if s.output == nil {
		return nil, fmt.Errorf("surface espresso ffn metal buffer binding: output IOSurface unavailable")
	}
	return s.output.NewMetalBufferBinding(device)
}

// NewDefaultOutputMetalBufferBinding creates a reusable no-copy Metal buffer
// binding over the output IOSurface using the default Metal device.
func (s *SurfaceEspressoFFN) NewDefaultOutputMetalBufferBinding() (*MetalBufferBinding, error) {
	if s == nil {
		return nil, fmt.Errorf("surface espresso ffn metal buffer binding: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("surface espresso ffn metal buffer binding: stage is closed")
	}
	if s.output == nil {
		return nil, fmt.Errorf("surface espresso ffn metal buffer binding: output IOSurface unavailable")
	}
	return s.output.NewDefaultMetalBufferBinding()
}

// EvalPreparedSurface runs one FFN body step assuming the mapped input
// IOSurface has already been filled by the caller.
func (s *SurfaceEspressoFFN) EvalPreparedSurface(ctx context.Context) (DecodeFFNTiming, error) {
	if s == nil {
		return DecodeFFNTiming{}, fmt.Errorf("surface espresso ffn eval prepared surface: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return DecodeFFNTiming{}, fmt.Errorf("surface espresso ffn eval prepared surface: stage is closed")
	}
	if s.plan == nil || s.input == nil || s.output == nil {
		return DecodeFFNTiming{}, fmt.Errorf("surface espresso ffn eval prepared surface: plan is unavailable")
	}

	totalStart := time.Now()
	aneStart := time.Now()
	if err := s.plan.Eval(ctx); err != nil {
		return DecodeFFNTiming{}, fmt.Errorf("surface espresso ffn eval prepared surface: ANE: %w", err)
	}
	ane := time.Since(aneStart)

	return DecodeFFNTiming{
		ANE:   ane,
		Total: time.Since(totalStart),
	}, nil
}

// EvalPreparedSurfaceAsync runs EvalPreparedSurface on a background goroutine.
func (s *SurfaceEspressoFFN) EvalPreparedSurfaceAsync(ctx context.Context) <-chan DecodeFFNSurfaceAsyncResult {
	ch := make(chan DecodeFFNSurfaceAsyncResult, 1)
	go func() {
		timing, err := s.EvalPreparedSurface(ctx)
		ch <- DecodeFFNSurfaceAsyncResult{Timing: timing, Err: err}
		close(ch)
	}()
	return ch
}

// Close releases mapped resources and unloads the file-backed model.
func (s *SurfaceEspressoFFN) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	plan := s.plan
	in := s.input
	out := s.output
	client := s.client
	release := s.release
	s.plan = nil
	s.input = nil
	s.output = nil
	s.client = nil
	s.release = nil
	s.mu.Unlock()

	if plan != nil {
		plan.Close()
	}
	if out != nil {
		out.Close()
	}
	if in != nil {
		in.Close()
	}
	if release != nil {
		release()
		return
	}
	if client != nil {
		client.Close()
	}
}
