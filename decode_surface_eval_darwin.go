//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

// DecodeFFNTiming reports where time is spent in one FFN offload step.
type DecodeFFNTiming struct {
	CopyIn  time.Duration
	ANE     time.Duration
	CopyOut time.Duration
	Total   time.Duration
}

// DecodeFFNResult is one FFN decode step result.
type DecodeFFNResult struct {
	Y      []float32
	Timing DecodeFFNTiming
}

// DecodeFFNAsyncResult is the asynchronous form of DecodeFFNResult.
type DecodeFFNAsyncResult struct {
	Result DecodeFFNResult
	Err    error
}

// DecodeFFNSurfaceAsyncResult is the asynchronous form of EvalPreparedSurface.
type DecodeFFNSurfaceAsyncResult struct {
	Timing DecodeFFNTiming
	Err    error
}

// SurfaceDecodeFFN is a reusable decode-stage FFN offload path.
//
// The stage compiles an FFN MIL model once, creates a mapped SurfaceEvalPlan,
// and reuses shared IOSurfaces across decode steps.
//
// SurfaceDecodeFFN satisfies SurfaceSyncRuntime.
type SurfaceDecodeFFN struct {
	mu          sync.Mutex
	model       appleneuralengine.ANEInMemoryModel
	clientModel *ANEClientMILModel
	input       *IOSurfaceFloat32
	output      *IOSurfaceFloat32
	plan        *SurfaceEvalPlan
	planErr     error
	fallback    FFNExecutor
	rms2        []float32
	w1          []float32
	w3          []float32
	w2          []float32
	dim         int
	hiddenDim   int
	seq         int
	mapSeq      int
	closed      bool
}

const envFFNEspressoFallback = "MLXGO_ANE_FFN_ALLOW_ESPRESSO_FALLBACK"

func allowFFNEspressoFallback() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(envFFNEspressoFallback))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// NewSurfaceDecodeFFNFromProfile constructs a decode FFN stage using dimensions
// from a parsed model profile.
func NewSurfaceDecodeFFNFromProfile(
	profile FFNShapeProfile,
	seq int,
	rms2, w1, w3, w2 []float32,
	cfg SurfaceEvalPlanConfig,
) (*SurfaceDecodeFFN, error) {
	if profile.Dim <= 0 || profile.HiddenDim <= 0 {
		return nil, fmt.Errorf(
			"new surface decode ffn from profile: invalid dims dim=%d hiddenDim=%d",
			profile.Dim, profile.HiddenDim,
		)
	}
	return NewSurfaceDecodeFFN(profile.Dim, profile.HiddenDim, seq, rms2, w1, w3, w2, cfg)
}

// NewSurfaceDecodeFFNFromConfig constructs a decode FFN stage using dimensions
// loaded from a model config.
func NewSurfaceDecodeFFNFromConfig(
	configPath string,
	seq int,
	rms2, w1, w3, w2 []float32,
	cfg SurfaceEvalPlanConfig,
) (*SurfaceDecodeFFN, error) {
	profile, err := LoadFFNShapeProfileFromConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("new surface decode ffn from config: %w", err)
	}
	return NewSurfaceDecodeFFNFromProfile(profile, seq, rms2, w1, w3, w2, cfg)
}

// NewSurfaceDecodeFFNQwen35Small constructs a decode FFN stage using local
// Qwen3.5 small-profile dimensions.
func NewSurfaceDecodeFFNQwen35Small(
	seq int,
	rms2, w1, w3, w2 []float32,
	cfg SurfaceEvalPlanConfig,
) (*SurfaceDecodeFFN, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("new surface decode ffn qwen3.5: resolve home dir: %w", err)
	}
	configPath := DefaultQwen35SmallConfigPath(home)
	return NewSurfaceDecodeFFNFromConfig(configPath, seq, rms2, w1, w3, w2, cfg)
}

// NewSurfaceDecodeFFN constructs a layer-ready FFN offload stage.
//
// attentionOut fed to Eval must have len=dim*seq.
func NewSurfaceDecodeFFN(
	dim, hiddenDim, seq int,
	rms2, w1, w3, w2 []float32,
	cfg SurfaceEvalPlanConfig,
) (*SurfaceDecodeFFN, error) {
	if dim <= 0 || hiddenDim <= 0 || seq <= 0 {
		return nil, fmt.Errorf(
			"new surface decode ffn: invalid dims dim=%d hiddenDim=%d seq=%d",
			dim, hiddenDim, seq,
		)
	}
	mapSeq := effectiveFFNMapSeq(seq)
	dflt := DefaultSurfaceEvalPlanConfig()
	if cfg.BridgeModelKey == "" {
		cfg.BridgeModelKey = dflt.BridgeModelKey
	}
	applySurfaceSharedEventModeFromEnv(&cfg)
	inCount := dim * mapSeq
	outCount := (2*dim + 3*hiddenDim) * mapSeq

	key := ffnForwardModelKey{
		dim:       dim,
		hiddenDim: hiddenDim,
		seq:       mapSeq,
		rms2Hash:  hashFloat32Slice(rms2),
		w1Hash:    hashFloat32Slice(w1),
		w3Hash:    hashFloat32Slice(w3),
		w2Hash:    hashFloat32Slice(w2),
	}
	model, err := buildFFNForwardModel(key, rms2, w1, w3, w2)
	var clientModel *ANEClientMILModel
	if err != nil {
		if allowFFNEspressoFallback() {
			clientModel, err = buildFFNForwardClientModel(key, w1, w3, w2)
		}
		if err != nil || clientModel == nil {
			return nil, fmt.Errorf("new surface decode ffn: build model: %w", err)
		}
	} else {
		base := model.Model()
		if base.GetID() != 0 {
			m := appleneuralengine.ANEModelFromID(base.GetID())
			if idx, ok := firstSymbolIndex(m.InputSymbolIndicesForProcedureIndex(uint32(cfg.ProcedureIndex))); ok {
				cfg.InputSymbolIndex = idx
			}
			if idx, ok := firstSymbolIndex(m.OutputSymbolIndicesForProcedureIndex(uint32(cfg.ProcedureIndex))); ok {
				cfg.OutputSymbolIndex = idx
			}
		}
	}

	inSurf, err := NewIOSurfaceFloat32(inCount)
	if err != nil {
		closeSurfaceDecodeFFNModel(model, clientModel)
		return nil, fmt.Errorf("new surface decode ffn: allocate input surface: %w", err)
	}
	outSurf, err := NewIOSurfaceFloat32(outCount)
	if err != nil {
		inSurf.Close()
		closeSurfaceDecodeFFNModel(model, clientModel)
		return nil, fmt.Errorf("new surface decode ffn: allocate output surface: %w", err)
	}
	// Match the proven eval lifecycle: initialize IOSurfaces before mapping.
	if err := inSurf.Write(make([]float32, inCount)); err != nil {
		outSurf.Close()
		inSurf.Close()
		closeSurfaceDecodeFFNModel(model, clientModel)
		return nil, fmt.Errorf("new surface decode ffn: initialize input surface: %w", err)
	}
	if err := outSurf.Write(make([]float32, outCount)); err != nil {
		outSurf.Close()
		inSurf.Close()
		closeSurfaceDecodeFFNModel(model, clientModel)
		return nil, fmt.Errorf("new surface decode ffn: initialize output surface: %w", err)
	}

	var plan *SurfaceEvalPlan
	if clientModel != nil {
		plan, err = NewSurfaceEvalPlanWithClientModel(clientModel, inSurf, outSurf, cfg)
	} else {
		plan, err = NewSurfaceEvalPlan(model, inSurf, outSurf, cfg)
	}
	if err != nil && !cfg.DisableCacheMapping {
		retryCfg := cfg
		retryCfg.DisableCacheMapping = true
		if clientModel != nil {
			plan, err = NewSurfaceEvalPlanWithClientModel(clientModel, inSurf, outSurf, retryCfg)
		} else {
			plan, err = NewSurfaceEvalPlan(model, inSurf, outSurf, retryCfg)
		}
	}
	planErr := err
	var fallbackExec FFNExecutor
	if err != nil {
		// Keep model loaded and fall back to per-call mapped evals.
		outSurf.Close()
		inSurf.Close()
		outSurf = nil
		inSurf = nil
		closeSurfaceDecodeFFNModel(model, clientModel)
		clientModel = nil
		model = appleneuralengine.ANEInMemoryModel{}
		rebuilt, rebuildErr := buildFFNForwardModel(key, rms2, w1, w3, w2)
		if rebuildErr != nil {
			return nil, fmt.Errorf(
				"new surface decode ffn: create plan failed (%v), rebuild fallback model failed: %w",
				err, rebuildErr,
			)
		}
		model = rebuilt
		exec, execErr := NewApplePrivateExecutor()
		if execErr != nil {
			return nil, fmt.Errorf(
				"new surface decode ffn: create plan failed (%v), fallback executor unavailable: %w",
				err, execErr,
			)
		}
		ffnExec, ok := exec.(FFNExecutor)
		if !ok {
			return nil, fmt.Errorf(
				"new surface decode ffn: create plan failed (%v), fallback executor missing FFN support",
				err,
			)
		}
		fallbackExec = ffnExec
	}
	return &SurfaceDecodeFFN{
		model:       model,
		clientModel: clientModel,
		input:       inSurf,
		output:      outSurf,
		plan:        plan,
		planErr:     planErr,
		fallback:    fallbackExec,
		rms2:        append([]float32(nil), rms2...),
		w1:          append([]float32(nil), w1...),
		w3:          append([]float32(nil), w3...),
		w2:          append([]float32(nil), w2...),
		dim:         dim,
		hiddenDim:   hiddenDim,
		seq:         seq,
		mapSeq:      mapSeq,
	}, nil
}

func buildFFNForwardClientModel(
	key ffnForwardModelKey,
	w1, w3, w2 []float32,
) (*ANEClientMILModel, error) {
	client, err := openPreferredANEClient()
	if err != nil {
		return nil, fmt.Errorf("open ANE client: %w", err)
	}
	dir, err := os.MkdirTemp(modelMirrorBaseDir(), "ane_ffn_espresso_*.mlmodelc")
	if err != nil {
		return nil, fmt.Errorf("create espresso temp dir: %w", err)
	}
	if err := GenerateFFNEspressoDir(dir, key.dim, key.hiddenDim, w1, w3, w2); err != nil {
		_ = os.RemoveAll(dir)
		return nil, fmt.Errorf("generate espresso dir: %w", err)
	}
	model, err := CompileAndLoadEspresso(client, dir, defaultMILFastPathKey, defaultANEQoS)
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, fmt.Errorf("compile/load espresso: %w", err)
	}
	model.dir = filepath.Clean(dir)
	model.owned = true
	return model, nil
}

func closeSurfaceDecodeFFNModel(model appleneuralengine.ANEInMemoryModel, clientModel *ANEClientMILModel) {
	if clientModel != nil {
		clientModel.Close()
		return
	}
	if model.GetID() != 0 {
		_ = callObjCBoolWithNSError(
			"appleneuralengine decode ffn unload",
			model.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
	}
}

func firstSymbolIndex(obj objectivec.IObject) (int, bool) {
	if obj == nil || obj.GetID() == 0 {
		return 0, false
	}
	if objc.Send[bool](obj.GetID(), objc.Sel("respondsToSelector:"), objc.Sel("firstIndex")) {
		v := objc.Send[uint](obj.GetID(), objc.Sel("firstIndex"))
		// NSIndexSetNotFound is NSUIntegerMax.
		if v == ^uint(0) {
			return 0, false
		}
		return int(v), true
	}
	count := objc.Send[uint](obj.GetID(), objc.Sel("count"))
	if count == 0 {
		return 0, false
	}
	first := objc.Send[objc.ID](obj.GetID(), objc.Sel("objectAtIndex:"), 0)
	if first == 0 {
		return 0, false
	}
	v := objc.Send[uint32](first, objc.Sel("unsignedIntValue"))
	return int(v), true
}

// Eval runs one decode FFN step: copy-in, ANE eval, copy-out.
func (s *SurfaceDecodeFFN) Eval(ctx context.Context, attentionOut []float32) (DecodeFFNResult, error) {
	if s == nil {
		return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: stage is closed")
	}
	want := s.dim * s.seq
	if len(attentionOut) != want {
		return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: attentionOut len=%d want=%d", len(attentionOut), want)
	}

	totalStart := time.Now()

	copyInStart := time.Now()
	mapInput := attentionOut
	if s.mapSeq != s.seq {
		padded, err := padANEChannelMajorSeq(attentionOut, s.dim, s.seq, s.mapSeq)
		if err != nil {
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: pad input: %w", err)
		}
		mapInput = padded
	}
	if s.plan != nil && s.input != nil {
		if err := s.input.Write(mapInput); err != nil {
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: copy in: %w", err)
		}
	} else {
		// The fallback path performs its own IOSurface writes internally.
		_ = append([]float32(nil), mapInput...)
	}
	copyIn := time.Since(copyInStart)

	var (
		ane     time.Duration
		copyOut time.Duration
		y       []float32
	)
	if s.plan != nil && s.output != nil {
		aneStart := time.Now()
		if err := s.plan.Eval(ctx); err != nil {
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: ANE: %w", err)
		}
		ane = time.Since(aneStart)

		copyOutStart := time.Now()
		packed, err := s.output.Read()
		if err != nil {
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: read output: %w", err)
		}
		taps, err := splitFFNForwardTapsANE(packed, s.dim, s.hiddenDim, s.mapSeq)
		if err != nil {
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: split taps: %w", err)
		}
		if s.mapSeq == s.seq {
			y = append([]float32(nil), taps.Y...)
		} else {
			trimmed, trimErr := trimANEChannelMajorSeq(taps.Y, s.dim, s.mapSeq, s.seq)
			if trimErr != nil {
				return DecodeFFNResult{}, fmt.Errorf("decode ffn eval: trim output: %w", trimErr)
			}
			y = trimmed
		}
		copyOut = time.Since(copyOutStart)
	} else {
		aneStart := time.Now()
		if s.fallback == nil {
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval fallback: executor is nil (plan err: %v)", s.planErr)
		}
		packed, err := s.fallback.FFNForward(
			ctx,
			append([]float32(nil), attentionOut...),
			s.rms2,
			s.w1,
			s.w3,
			s.w2,
			s.dim,
			s.hiddenDim,
			s.seq,
		)
		if err != nil {
			if s.planErr != nil {
				return DecodeFFNResult{}, fmt.Errorf("decode ffn eval fallback (plan unavailable: %v): %w", s.planErr, err)
			}
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval fallback: %w", err)
		}
		ane = time.Since(aneStart)
		copyOutStart := time.Now()
		taps, err := splitFFNForwardTapsANE(packed, s.dim, s.hiddenDim, s.seq)
		if err != nil {
			return DecodeFFNResult{}, fmt.Errorf("decode ffn eval fallback split taps: %w", err)
		}
		y = append([]float32(nil), taps.Y...)
		copyOut = time.Since(copyOutStart)
	}

	return DecodeFFNResult{
		Y: y,
		Timing: DecodeFFNTiming{
			CopyIn:  copyIn,
			ANE:     ane,
			CopyOut: copyOut,
			Total:   time.Since(totalStart),
		},
	}, nil
}

// EvalWithTelemetry runs one decode FFN step and returns a sampled telemetry
// snapshot for the underlying mapped plan.
//
// This is intended for benchmarks and diagnostics. The timed benchmark path
// should continue to use Eval and collect telemetry from a coarse sample
// outside the timed loop.
func (s *SurfaceDecodeFFN) EvalWithTelemetry(
	ctx context.Context,
	attentionOut []float32,
	opts SurfaceEvalTelemetryOptions,
) (DecodeFFNResult, SurfaceEvalTelemetry, error) {
	if s == nil {
		return DecodeFFNResult{}, SurfaceEvalTelemetry{}, fmt.Errorf("decode ffn eval telemetry: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return DecodeFFNResult{}, SurfaceEvalTelemetry{}, fmt.Errorf("decode ffn eval telemetry: stage is closed")
	}
	want := s.dim * s.seq
	if len(attentionOut) != want {
		return DecodeFFNResult{}, SurfaceEvalTelemetry{}, fmt.Errorf(
			"decode ffn eval telemetry: attentionOut len=%d want=%d",
			len(attentionOut),
			want,
		)
	}

	totalStart := time.Now()

	copyInStart := time.Now()
	mapInput := attentionOut
	if s.mapSeq != s.seq {
		padded, err := padANEChannelMajorSeq(attentionOut, s.dim, s.seq, s.mapSeq)
		if err != nil {
			return DecodeFFNResult{}, SurfaceEvalTelemetry{}, fmt.Errorf("decode ffn eval telemetry: pad input: %w", err)
		}
		mapInput = padded
	}
	if s.plan != nil && s.input != nil {
		if err := s.input.Write(mapInput); err != nil {
			return DecodeFFNResult{}, SurfaceEvalTelemetry{}, fmt.Errorf("decode ffn eval telemetry: copy in: %w", err)
		}
	}
	copyIn := time.Since(copyInStart)

	var (
		ane       time.Duration
		copyOut   time.Duration
		y         []float32
		telemetry SurfaceEvalTelemetry
	)
	if s.plan != nil && s.output != nil {
		aneStart := time.Now()
		sampled, err := s.plan.EvalWithTelemetry(ctx, opts)
		if err != nil {
			return DecodeFFNResult{}, sampled, fmt.Errorf("decode ffn eval telemetry: ANE: %w", err)
		}
		ane = time.Since(aneStart)
		telemetry = sampled

		copyOutStart := time.Now()
		packed, err := s.output.Read()
		if err != nil {
			return DecodeFFNResult{}, telemetry, fmt.Errorf("decode ffn eval telemetry: read output: %w", err)
		}
		taps, err := splitFFNForwardTapsANE(packed, s.dim, s.hiddenDim, s.mapSeq)
		if err != nil {
			return DecodeFFNResult{}, telemetry, fmt.Errorf("decode ffn eval telemetry: split taps: %w", err)
		}
		if s.mapSeq == s.seq {
			y = append([]float32(nil), taps.Y...)
		} else {
			trimmed, trimErr := trimANEChannelMajorSeq(taps.Y, s.dim, s.mapSeq, s.seq)
			if trimErr != nil {
				return DecodeFFNResult{}, telemetry, fmt.Errorf("decode ffn eval telemetry: trim output: %w", trimErr)
			}
			y = trimmed
		}
		copyOut = time.Since(copyOutStart)
	} else {
		aneStart := time.Now()
		if s.fallback == nil {
			return DecodeFFNResult{}, telemetry, fmt.Errorf(
				"decode ffn eval telemetry fallback: executor is nil (plan err: %v)",
				s.planErr,
			)
		}
		packed, err := s.fallback.FFNForward(
			ctx,
			append([]float32(nil), attentionOut...),
			s.rms2,
			s.w1,
			s.w3,
			s.w2,
			s.dim,
			s.hiddenDim,
			s.seq,
		)
		if err != nil {
			if s.planErr != nil {
				return DecodeFFNResult{}, telemetry, fmt.Errorf(
					"decode ffn eval telemetry fallback (plan unavailable: %v): %w",
					s.planErr,
					err,
				)
			}
			return DecodeFFNResult{}, telemetry, fmt.Errorf("decode ffn eval telemetry fallback: %w", err)
		}
		ane = time.Since(aneStart)
		copyOutStart := time.Now()
		taps, err := splitFFNForwardTapsANE(packed, s.dim, s.hiddenDim, s.seq)
		if err != nil {
			return DecodeFFNResult{}, telemetry, fmt.Errorf("decode ffn eval telemetry fallback split taps: %w", err)
		}
		y = append([]float32(nil), taps.Y...)
		copyOut = time.Since(copyOutStart)
	}

	return DecodeFFNResult{
		Y: y,
		Timing: DecodeFFNTiming{
			CopyIn:  copyIn,
			ANE:     ane,
			CopyOut: copyOut,
			Total:   time.Since(totalStart),
		},
	}, telemetry, nil
}

// EvalToSurface runs one decode FFN step but leaves the packed output resident
// in the stage's IOSurface.
//
// This is the zero-copy path for downstream consumers that can read directly
// from the output IOSurface. It requires an active surface plan.
func (s *SurfaceDecodeFFN) EvalToSurface(ctx context.Context, attentionOut []float32) (DecodeFFNTiming, error) {
	if s == nil {
		return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval to surface: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval to surface: stage is closed")
	}
	want := s.dim * s.seq
	if len(attentionOut) != want {
		return DecodeFFNTiming{}, fmt.Errorf(
			"decode ffn eval to surface: attentionOut len=%d want=%d",
			len(attentionOut),
			want,
		)
	}
	if s.plan == nil || s.input == nil || s.output == nil {
		return DecodeFFNTiming{}, fmt.Errorf(
			"decode ffn eval to surface: surface plan unavailable (plan err: %v)",
			s.planErr,
		)
	}

	totalStart := time.Now()
	copyInStart := time.Now()
	mapInput := attentionOut
	if s.mapSeq != s.seq {
		padded, err := padANEChannelMajorSeq(attentionOut, s.dim, s.seq, s.mapSeq)
		if err != nil {
			return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval to surface: pad input: %w", err)
		}
		mapInput = padded
	}
	if err := s.input.Write(mapInput); err != nil {
		return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval to surface: copy in: %w", err)
	}
	copyIn := time.Since(copyInStart)

	aneStart := time.Now()
	if err := s.plan.Eval(ctx); err != nil {
		return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval to surface: ANE: %w", err)
	}
	ane := time.Since(aneStart)

	return DecodeFFNTiming{
		CopyIn: copyIn,
		ANE:    ane,
		Total:  time.Since(totalStart),
	}, nil
}

// EvalPreparedSurface runs one decode FFN step assuming the mapped input
// IOSurface has already been filled by the caller.
//
// This is the MLX->ANE handoff path used by the decode plane. It requires an
// active surface plan and leaves the packed output resident in the output
// IOSurface for zero-copy consumers.
func (s *SurfaceDecodeFFN) EvalPreparedSurface(ctx context.Context) (DecodeFFNTiming, error) {
	if s == nil {
		return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval prepared surface: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval prepared surface: stage is closed")
	}
	if s.plan == nil || s.input == nil || s.output == nil {
		return DecodeFFNTiming{}, fmt.Errorf(
			"decode ffn eval prepared surface: surface plan unavailable (plan err: %v)",
			s.planErr,
		)
	}

	totalStart := time.Now()
	aneStart := time.Now()
	if err := s.plan.Eval(ctx); err != nil {
		return DecodeFFNTiming{}, fmt.Errorf("decode ffn eval prepared surface: ANE: %w", err)
	}
	ane := time.Since(aneStart)

	return DecodeFFNTiming{
		ANE:   ane,
		Total: time.Since(totalStart),
	}, nil
}

// UsingSurfacePlan reports whether persistent request mapping is active.
func (s *SurfaceDecodeFFN) UsingSurfacePlan() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.plan != nil
}

// PlanError returns the mapping error that disabled persistent plan mode, if any.
func (s *SurfaceDecodeFFN) PlanError() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.planErr
}

// ModelDim returns the FFN model dimension.
func (s *SurfaceDecodeFFN) ModelDim() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dim
}

// Seq returns the logical decode sequence length accepted by the stage.
func (s *SurfaceDecodeFFN) Seq() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seq
}

// MapSeq returns the padded ANE map sequence length used by the stage.
func (s *SurfaceDecodeFFN) MapSeq() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.mapSeq
}

// InputShape returns the mapped input surface shape as [channels, seq].
func (s *SurfaceDecodeFFN) InputShape() []int {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return []int{s.dim, s.mapSeq}
}

// OutputShape returns the mapped packed output surface shape as [channels, seq].
func (s *SurfaceDecodeFFN) OutputShape() []int {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return []int{2*s.dim + 3*s.hiddenDim, s.mapSeq}
}

// InputSurface returns the input IOSurface when persistent plan mode is active.
func (s *SurfaceDecodeFFN) InputSurface() *IOSurfaceFloat32 {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.input
}

// OutputSurface returns the output IOSurface when persistent plan mode is
// active. Callers can use ReadOnlyView for zero-copy handoff.
func (s *SurfaceDecodeFFN) OutputSurface() *IOSurfaceFloat32 {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.output
}

// NewOutputMetalBufferBinding creates a reusable no-copy Metal buffer binding
// over the active output IOSurface.
func (s *SurfaceDecodeFFN) NewOutputMetalBufferBinding(device metal.MTLDevice) (*MetalBufferBinding, error) {
	if s == nil {
		return nil, fmt.Errorf("decode ffn metal buffer binding: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("decode ffn metal buffer binding: stage is closed")
	}
	if s.output == nil {
		return nil, fmt.Errorf("decode ffn metal buffer binding: output IOSurface unavailable")
	}
	return s.output.NewMetalBufferBinding(device)
}

// NewDefaultOutputMetalBufferBinding creates a reusable no-copy Metal buffer
// binding over the active output IOSurface using the default Metal device.
func (s *SurfaceDecodeFFN) NewDefaultOutputMetalBufferBinding() (*MetalBufferBinding, error) {
	if s == nil {
		return nil, fmt.Errorf("decode ffn metal buffer binding: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("decode ffn metal buffer binding: stage is closed")
	}
	if s.output == nil {
		return nil, fmt.Errorf("decode ffn metal buffer binding: output IOSurface unavailable")
	}
	return s.output.NewDefaultMetalBufferBinding()
}

// WaitEvent returns the attached Metal->ANE wait event when configured.
func (s *SurfaceDecodeFFN) WaitEvent() *SharedEvent {
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

// NewWaitMetalSharedEvent imports the attached wait event into Metal.
func (s *SurfaceDecodeFFN) NewWaitMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if s == nil {
		return nil, fmt.Errorf("decode ffn wait metal event: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return nil, fmt.Errorf("decode ffn wait metal event: plan is nil")
	}
	return s.plan.NewWaitMetalSharedEvent(device)
}

// NewDefaultWaitMetalSharedEvent imports the attached wait event into Metal
// using the default Metal device.
func (s *SurfaceDecodeFFN) NewDefaultWaitMetalSharedEvent() (*MetalSharedEvent, error) {
	if s == nil {
		return nil, fmt.Errorf("decode ffn wait metal event: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return nil, fmt.Errorf("decode ffn wait metal event: plan is nil")
	}
	return s.plan.NewDefaultWaitMetalSharedEvent()
}

// WaitEventPort returns the Metal->ANE wait-event port when configured.
func (s *SurfaceDecodeFFN) WaitEventPort() uint32 {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return 0
	}
	return s.plan.WaitEventPort()
}

// WaitPort is a shorthand alias for WaitEventPort.
func (s *SurfaceDecodeFFN) WaitPort() uint32 {
	return s.WaitEventPort()
}

// WaitValue returns the configured Metal->ANE wait target value.
func (s *SurfaceDecodeFFN) WaitValue() uint64 {
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

// SignalEventPort returns the ANE->Metal signal-event port when configured.
func (s *SurfaceDecodeFFN) SignalEventPort() uint32 {
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

// SignalEvent returns the attached ANE->Metal signal event when configured.
func (s *SurfaceDecodeFFN) SignalEvent() *SharedEvent {
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

// NewSignalMetalSharedEvent imports the attached signal event into Metal.
func (s *SurfaceDecodeFFN) NewSignalMetalSharedEvent(device metal.MTLDevice) (*MetalSharedEvent, error) {
	if s == nil {
		return nil, fmt.Errorf("decode ffn signal metal event: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return nil, fmt.Errorf("decode ffn signal metal event: plan is nil")
	}
	return s.plan.NewSignalMetalSharedEvent(device)
}

// NewDefaultSignalMetalSharedEvent imports the attached signal event into
// Metal using the default Metal device.
func (s *SurfaceDecodeFFN) NewDefaultSignalMetalSharedEvent() (*MetalSharedEvent, error) {
	if s == nil {
		return nil, fmt.Errorf("decode ffn signal metal event: stage is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return nil, fmt.Errorf("decode ffn signal metal event: plan is nil")
	}
	return s.plan.NewDefaultSignalMetalSharedEvent()
}

// SignalPort is a shorthand alias for SignalEventPort.
func (s *SurfaceDecodeFFN) SignalPort() uint32 {
	return s.SignalEventPort()
}

// SignalValue returns the configured ANE->Metal signal target value.
func (s *SurfaceDecodeFFN) SignalValue() uint64 {
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

// EvalAsync runs Eval on a background goroutine.
func (s *SurfaceDecodeFFN) EvalAsync(ctx context.Context, attentionOut []float32) <-chan DecodeFFNAsyncResult {
	ch := make(chan DecodeFFNAsyncResult, 1)
	in := append([]float32(nil), attentionOut...)
	go func() {
		res, err := s.Eval(ctx, in)
		ch <- DecodeFFNAsyncResult{Result: res, Err: err}
		close(ch)
	}()
	return ch
}

// EvalPreparedSurfaceAsync runs EvalPreparedSurface on a background goroutine.
func (s *SurfaceDecodeFFN) EvalPreparedSurfaceAsync(ctx context.Context) <-chan DecodeFFNSurfaceAsyncResult {
	ch := make(chan DecodeFFNSurfaceAsyncResult, 1)
	go func() {
		timing, err := s.EvalPreparedSurface(ctx)
		ch <- DecodeFFNSurfaceAsyncResult{Timing: timing, Err: err}
		close(ch)
	}()
	return ch
}

// Close releases mapped resources and unloads the in-memory model.
func (s *SurfaceDecodeFFN) Close() {
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
	model := s.model
	clientModel := s.clientModel
	s.plan = nil
	s.input = nil
	s.output = nil
	s.model = appleneuralengine.ANEInMemoryModel{}
	s.clientModel = nil
	s.fallback = nil
	s.rms2 = nil
	s.w1 = nil
	s.w3 = nil
	s.w2 = nil
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
	closeSurfaceDecodeFFNModel(model, clientModel)
}
