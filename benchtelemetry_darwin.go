//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/private/appleneuralengine"
	xane "github.com/tmc/apple/x/ane"
	xanetelemetry "github.com/tmc/apple/x/ane/telemetry"
)

type metricReporter interface {
	ReportMetric(float64, string)
}

type suffixMetricReporter struct {
	dst    metricReporter
	suffix string
}

func (m suffixMetricReporter) ReportMetric(v float64, unit string) {
	if m.dst == nil || unit == "" {
		return
	}
	m.dst.ReportMetric(v, unit+m.suffix)
}

// BenchmarkHeader describes the ANE environment used by a benchmark run.
type BenchmarkHeader struct {
	Snapshot xanetelemetry.ClientSnapshot
	ProbeErr error
}

var (
	benchmarkHeaderOnce sync.Once
	benchmarkHeaderSnap BenchmarkHeader
)

// ProbeBenchmarkHeader returns a cached ANE environment snapshot for benchmarks.
func ProbeBenchmarkHeader() BenchmarkHeader {
	benchmarkHeaderOnce.Do(func() {
		rt, err := xane.Open()
		if err != nil {
			benchmarkHeaderSnap.ProbeErr = err
			return
		}
		defer rt.Close()
		benchmarkHeaderSnap.Snapshot = xanetelemetry.Snapshot(rt)
	})
	return benchmarkHeaderSnap
}

// ReportMetrics emits numeric benchmark header fields.
func (h BenchmarkHeader) ReportMetrics(m metricReporter) {
	if m == nil {
		return
	}
	reportBool := func(v bool, unit string) {
		if v {
			m.ReportMetric(1, unit)
			return
		}
		m.ReportMetric(0, unit)
	}
	reportBool(h.ProbeErr == nil, "ane_probe_ok")
	reportBool(h.Snapshot.Device.HasANE, "ane_has_device")
	reportBool(h.Snapshot.Available(), "ane_snapshot_available")
	h.Snapshot.ReportMetrics(m)
}

// Logf logs string-valued benchmark header fields.
func (h BenchmarkHeader) Logf(logf func(string, ...any)) {
	if logf == nil {
		return
	}
	if h.ProbeErr != nil {
		logf("ane benchmark header probe error: %v", h.ProbeErr)
		return
	}
	logf(
		"ane benchmark header device=%s product=%s build=%s subtype=%s board=%d cache_dir=%q client_available=%t capability_mask_known=%t",
		h.Snapshot.Device.Architecture,
		h.Snapshot.Device.Product,
		h.Snapshot.Device.BuildVersion,
		h.Snapshot.Device.SubType,
		h.Snapshot.Device.BoardType,
		h.Snapshot.Cache.CacheDir,
		h.Snapshot.Client.Available(),
		h.Snapshot.Client.CapabilityMaskKnown,
	)
	if h.Snapshot.Client.Available() {
		logf(
			"ane benchmark client num_anes=%d known=%t num_cores=%d known=%t arch=%q known=%t board=%d known=%t",
			h.Snapshot.Client.NumANEs,
			h.Snapshot.Client.NumANEsKnown,
			h.Snapshot.Client.NumCores,
			h.Snapshot.Client.NumCoresKnown,
			h.Snapshot.Client.ArchitectureStr,
			h.Snapshot.Client.ArchitectureStrKnown,
			h.Snapshot.Client.BoardType,
			h.Snapshot.Client.BoardTypeKnown,
		)
	}
}

// SurfaceEvalTelemetry captures one sampled eval snapshot for a mapped plan.
type SurfaceEvalTelemetry struct {
	Stats             xanetelemetry.EvalStats
	DiagnosticsBefore xanetelemetry.Diagnostics
	DiagnosticsAfter  xanetelemetry.Diagnostics
	EventTiming       xanetelemetry.EventTiming
	SignalWait        time.Duration
	WallDuration      time.Duration
}

// ReportMetrics emits numeric telemetry fields for one sampled eval.
func (t SurfaceEvalTelemetry) ReportMetrics(m metricReporter) {
	if m == nil {
		return
	}
	m.ReportMetric(float64(t.WallDuration.Nanoseconds()), "eval-wall-ns/op")
	t.Stats.ReportMetrics(m)
	t.DiagnosticsBefore.ReportMetrics(suffixMetricReporter{dst: m, suffix: "-before"})
	t.DiagnosticsAfter.ReportMetrics(suffixMetricReporter{dst: m, suffix: "-after"})
	if t.DiagnosticsBefore.AsyncRequestsInFlightKnown && t.DiagnosticsAfter.AsyncRequestsInFlightKnown {
		m.ReportMetric(
			float64(t.DiagnosticsAfter.AsyncRequestsInFlight-t.DiagnosticsBefore.AsyncRequestsInFlight),
			"async-in-flight-delta",
		)
	}
	t.EventTiming.ReportMetrics(m)
	if t.SignalWait > 0 {
		m.ReportMetric(float64(t.SignalWait.Nanoseconds()), "wait-ns/op")
	}
}

// Logf logs string-valued telemetry fields for one sampled eval.
func (t SurfaceEvalTelemetry) Logf(logf func(string, ...any)) {
	if logf == nil || len(t.Stats.PerfCounters) == 0 {
		return
	}
	names := make([]string, 0, len(t.Stats.PerfCounters))
	for _, pc := range t.Stats.PerfCounters {
		if pc.Name == "" {
			continue
		}
		names = append(names, pc.Name)
	}
	if len(names) == 0 {
		return
	}
	logf("ane perf counter labels: %s", strings.Join(names, ", "))
}

// SurfaceEvalTelemetryOptions controls optional sampled telemetry collection.
type SurfaceEvalTelemetryOptions struct {
	CollectPerfStats  bool
	SampleDiagnostics bool
	WaitForSignal     bool
	SignalWaitTimeout time.Duration
	SignalEventValue  uint64
}

// EvalWithTelemetry runs one eval and returns a sampled telemetry snapshot.
func (p *SurfaceEvalPlan) EvalWithTelemetry(ctx context.Context, opts SurfaceEvalTelemetryOptions) (SurfaceEvalTelemetry, error) {
	if p == nil {
		return SurfaceEvalTelemetry{}, fmt.Errorf("surface eval telemetry: plan is nil")
	}
	var telemetry SurfaceEvalTelemetry
	if opts.SampleDiagnostics {
		telemetry.DiagnosticsBefore = p.Diagnostics()
	}

	var perfStats *appleneuralengine.ANEPerformanceStats
	if opts.CollectPerfStats {
		ps := appleneuralengine.NewANEPerformanceStats()
		perfStats = &ps
		request := appleneuralengine.ANERequestFromID(p.request.GetID())
		request.SetPerfStats(perfStats)
	}

	start := time.Now()
	err := p.Eval(ctx)
	telemetry.WallDuration = time.Since(start)
	telemetry.EventTiming.EnqueueNS = telemetry.WallDuration.Nanoseconds()
	telemetry.EventTiming.TotalNS = telemetry.WallDuration.Nanoseconds()
	if perfStats != nil {
		telemetry.Stats = surfaceEvalStatsFromPerf(*perfStats)
	}
	if opts.SampleDiagnostics {
		telemetry.DiagnosticsAfter = p.Diagnostics()
	}
	if err != nil {
		return telemetry, err
	}
	if opts.WaitForSignal {
		ev := p.SignalEvent()
		if ev != nil {
			target := opts.SignalEventValue
			if target == 0 {
				target = p.SignalValue()
			}
			timeout := opts.SignalWaitTimeout
			if timeout <= 0 {
				timeout = 5 * time.Second
			}
			waitStart := time.Now()
			if waitErr := ev.WaitCPU(target, timeout); waitErr != nil {
				telemetry.SignalWait = time.Since(waitStart)
				return telemetry, waitErr
			}
			telemetry.SignalWait = time.Since(waitStart)
		}
	}
	return telemetry, nil
}

// Diagnostics returns best-effort queue and in-flight state for the mapped plan.
func (p *SurfaceEvalPlan) Diagnostics() xanetelemetry.Diagnostics {
	if p == nil {
		return xanetelemetry.Diagnostics{}
	}
	var d xanetelemetry.Diagnostics
	switch {
	case p.clientModel != nil:
		d.ProgramClass = "ANEModel"
		d.ProgramClassKnown = true
		if depth, ok := surfaceProbeQueueDepth(p.clientModel.model.GetID()); ok {
			d.ModelQueueDepth = depth
			d.ModelQueueDepthKnown = true
		}
		if prog := surfaceProbeProgramForEval(p.clientModel.model.GetID()); prog != 0 {
			surfaceProbeProgramDiags(prog, &d)
		}
		if handle, ok := surfaceProbeProgramHandle(p.clientModel.model.GetID()); ok {
			d.ProgramHandle = handle
			d.ProgramHandleKnown = true
		}
		if state, ok := surfaceProbeModelState(p.clientModel.model.GetID()); ok {
			d.ModelState = state
			d.ModelStateKnown = true
		}
	case p.model.GetID() != 0:
		d.ProgramClass = "ANEInMemoryModel"
		d.ProgramClassKnown = true
		if depth, ok := surfaceProbeQueueDepth(p.model.ID); ok {
			d.ModelQueueDepth = depth
			d.ModelQueueDepthKnown = true
		}
		if prog := surfaceProbeProgramForEval(p.model.ID); prog != 0 {
			surfaceProbeProgramDiags(prog, &d)
		}
		if handle, ok := surfaceProbeProgramHandle(p.model.ID); ok {
			d.ProgramHandle = handle
			d.ProgramHandleKnown = true
		}
		if state, ok := surfaceProbeModelState(p.model.ID); ok {
			d.ModelState = state
			d.ModelStateKnown = true
		}
	}
	return d
}

func surfaceEvalStatsFromPerf(perfStats appleneuralengine.ANEPerformanceStats) xanetelemetry.EvalStats {
	var stats xanetelemetry.EvalStats
	func() {
		defer func() { recover() }()
		stats.HWExecutionNS = perfStats.HwExecutionTime()
	}()
	func() {
		defer func() { recover() }()
		if d := perfStats.PerfCounterData(); d != nil {
			stats.PerfCounterData = foundation.NSDataFromID(d.GetID()).GoBytes()
		}
	}()
	func() {
		defer func() { recover() }()
		if d := perfStats.PStatsRawData(); d != nil {
			stats.RawStatsData = foundation.NSDataFromID(d.GetID()).GoBytes()
		}
	}()
	func() {
		defer func() { recover() }()
		const maxPerfCounters = 256
		for i := range maxPerfCounters {
			obj := perfStats.StringForPerfCounter(i)
			if obj == nil || obj.GetID() == 0 {
				break
			}
			name := benchDescriptionString(obj.GetID())
			if name == "" {
				break
			}
			stats.PerfCounters = append(stats.PerfCounters, xanetelemetry.PerfCounter{Index: i, Name: name})
		}
		if len(stats.PerfCounters) == maxPerfCounters {
			obj := perfStats.StringForPerfCounter(maxPerfCounters)
			if obj != nil && obj.GetID() != 0 {
				stats.PerfCountersTruncated = true
			}
		}
	}()
	func() {
		defer func() { recover() }()
		countersObj := perfStats.PerformanceCounters()
		if countersObj == nil || countersObj.GetID() == 0 {
			return
		}
		dict := foundation.NSDictionaryFromID(countersObj.GetID())
		keys := dict.AllKeys()
		for _, key := range keys {
			idx := foundation.NSNumberFromID(key.GetID()).IntValue()
			val := foundation.NSNumberFromID(dict.ObjectForKey(key).GetID()).UnsignedLongLongValue()
			found := false
			for j := range stats.PerfCounters {
				if stats.PerfCounters[j].Index == idx {
					stats.PerfCounters[j].Value = val
					found = true
					break
				}
			}
			if !found {
				stats.PerfCounters = append(stats.PerfCounters, xanetelemetry.PerfCounter{Index: idx, Value: val})
			}
		}
	}()
	return stats
}

func surfaceProbeQueueDepth(id objc.ID) (depth int, ok bool) {
	defer func() { recover() }()
	rv := objc.Send[int8](id, objc.Sel("queueDepth"))
	return int(rv), true
}

func surfaceProbeProgramForEval(id objc.ID) objc.ID {
	defer func() { recover() }()
	return objc.Send[objc.ID](id, objc.Sel("program"))
}

func surfaceProbeProgramDiags(prog objc.ID, d *xanetelemetry.Diagnostics) {
	func() {
		defer func() { recover() }()
		rv := objc.Send[int64](prog, objc.Sel("currentAsyncRequestsInFlight"))
		d.AsyncRequestsInFlight = rv
		d.AsyncRequestsInFlightKnown = true
	}()
	func() {
		defer func() { recover() }()
		rv := objc.Send[int8](prog, objc.Sel("queueDepth"))
		d.ProgramQueueDepth = int(rv)
		d.ProgramQueueDepthKnown = true
	}()
	func() {
		defer func() { recover() }()
		rv := objc.Send[uint64](prog, objc.Sel("programHandle"))
		if !d.ProgramHandleKnown {
			d.ProgramHandle = rv
			d.ProgramHandleKnown = true
		}
	}()
}

func surfaceProbeProgramHandle(id objc.ID) (uint64, bool) {
	defer func() { recover() }()
	rv := objc.Send[uint64](id, objc.Sel("programHandle"))
	return rv, true
}

func surfaceProbeModelState(id objc.ID) (uint64, bool) {
	defer func() { recover() }()
	rv := objc.Send[uint64](id, objc.Sel("state"))
	return rv, true
}

func benchDescriptionString(id objc.ID) string {
	if id == 0 {
		return ""
	}
	rv := objc.Send[objc.ID](id, objc.Sel("description"))
	if rv == 0 {
		return ""
	}
	return objc.Send[string](rv, objc.Sel("UTF8String"))
}

var (
	benchmarkPerfMaskOnce sync.Once
	benchmarkPerfMask     uint32
)

func benchmarkPerfStatsMask() uint32 {
	benchmarkPerfMaskOnce.Do(func() {
		raw := strings.TrimSpace(os.Getenv("MLXGO_ANE_BENCH_PERF_STATS_MASK"))
		if raw == "" {
			return
		}
		v, err := strconv.ParseUint(raw, 0, 32)
		if err == nil {
			benchmarkPerfMask = uint32(v)
		}
	})
	return benchmarkPerfMask
}
