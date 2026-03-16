package mlxgoane

import (
	"sort"
	"strings"
	"sync"
	"time"
)

// LinearHookStats accumulates backend-selection stats for InstallNNLinearHook.
type LinearHookStats struct {
	mu sync.Mutex

	totalCalls      uint64
	aneCalls        uint64
	mlxCalls        uint64
	routerFallbacks uint64
	errorFallbacks  uint64
	cacheKnownCalls uint64
	cacheHits       uint64
	cacheMisses     uint64
	buildTotal      time.Duration
	compileTotal    time.Duration
	loadTotal       time.Duration
	evaluateTotal   time.Duration
	fallbackReasons map[string]uint64
}

// LinearHookStatsSnapshot is an immutable summary of hook routing activity.
type LinearHookStatsSnapshot struct {
	TotalCalls      uint64
	ANECalls        uint64
	MLXCalls        uint64
	RouterFallbacks uint64
	ErrorFallbacks  uint64
	CacheKnownCalls uint64
	CacheHits       uint64
	CacheMisses     uint64
	BuildTotal      time.Duration
	CompileTotal    time.Duration
	LoadTotal       time.Duration
	EvaluateTotal   time.Duration
	FallbackReasons map[string]uint64
}

// NewLinearHookStats returns an empty stats collector.
func NewLinearHookStats() *LinearHookStats {
	return &LinearHookStats{fallbackReasons: make(map[string]uint64)}
}

// Reset clears the accumulated stats.
func (s *LinearHookStats) Reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.totalCalls = 0
	s.aneCalls = 0
	s.mlxCalls = 0
	s.routerFallbacks = 0
	s.errorFallbacks = 0
	s.cacheKnownCalls = 0
	s.cacheHits = 0
	s.cacheMisses = 0
	s.buildTotal = 0
	s.compileTotal = 0
	s.loadTotal = 0
	s.evaluateTotal = 0
	clear(s.fallbackReasons)
}

// Snapshot returns a copy of the accumulated stats.
func (s *LinearHookStats) Snapshot() LinearHookStatsSnapshot {
	if s == nil {
		return LinearHookStatsSnapshot{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	reasons := make(map[string]uint64, len(s.fallbackReasons))
	for k, v := range s.fallbackReasons {
		reasons[k] = v
	}
	return LinearHookStatsSnapshot{
		TotalCalls:      s.totalCalls,
		ANECalls:        s.aneCalls,
		MLXCalls:        s.mlxCalls,
		RouterFallbacks: s.routerFallbacks,
		ErrorFallbacks:  s.errorFallbacks,
		CacheKnownCalls: s.cacheKnownCalls,
		CacheHits:       s.cacheHits,
		CacheMisses:     s.cacheMisses,
		BuildTotal:      s.buildTotal,
		CompileTotal:    s.compileTotal,
		LoadTotal:       s.loadTotal,
		EvaluateTotal:   s.evaluateTotal,
		FallbackReasons: reasons,
	}
}

// ANEFraction returns the fraction of calls served by ANE.
func (s LinearHookStatsSnapshot) ANEFraction() float64 {
	if s.TotalCalls == 0 {
		return 0
	}
	return float64(s.ANECalls) / float64(s.TotalCalls)
}

// ZeroBenefit reports whether the hook observed linear calls but none reached ANE.
func (s LinearHookStatsSnapshot) ZeroBenefit() bool {
	return s.TotalCalls > 0 && s.ANECalls == 0
}

// FormatFallbackReasons returns a stable compact summary of fallback reasons.
func (s LinearHookStatsSnapshot) FormatFallbackReasons() string {
	if len(s.FallbackReasons) == 0 {
		return ""
	}
	keys := make([]string, 0, len(s.FallbackReasons))
	for k := range s.FallbackReasons {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+itoa64(s.FallbackReasons[k]))
	}
	return strings.Join(parts, ",")
}

func (s *LinearHookStats) record(rt *Runtime, res *LinearResult) {
	if s == nil || res == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	s.totalCalls++
	switch res.Backend {
	case BackendANE:
		s.aneCalls++
		if provider, ok := linearTelemetryProvider(rt); ok {
			t := provider.LastLinearTelemetry()
			s.cacheKnownCalls++
			if t.CacheHit {
				s.cacheHits++
			} else {
				s.cacheMisses++
			}
			s.buildTotal += t.Build
			s.compileTotal += t.Compile
			s.loadTotal += t.Load
			s.evaluateTotal += t.Evaluate
		}
	case BackendMLX:
		s.mlxCalls++
		reason := strings.TrimSpace(res.FallbackReason)
		if strings.HasPrefix(reason, "router: ") {
			s.routerFallbacks++
			reason = strings.TrimPrefix(reason, "router: ")
		} else if reason != "" {
			s.errorFallbacks++
		}
		if reason != "" {
			s.fallbackReasons[reason]++
		}
	}
}

func linearTelemetryProvider(rt *Runtime) (LinearTelemetryProvider, bool) {
	if rt == nil || rt.Executor == nil {
		return nil, false
	}
	p, ok := rt.Executor.(LinearTelemetryProvider)
	return p, ok
}

func itoa64(v uint64) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
