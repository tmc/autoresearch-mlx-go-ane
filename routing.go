package mlxgoane

import "math"

const (
	routeReasonEligible          = "eligible"
	routeReasonCached            = "cache_hit"
	routeReasonInvalidDims       = "invalid_dims"
	routeReasonSmallSpatial      = "spatial_below_min"
	routeReasonUnalignedChannels = "unaligned_channels"
	routeReasonCompileBudget     = "compile_budget_exhausted"
	routeReasonColdTooSmall      = "cold_work_below_min"
	routeReasonWarmTooSmall      = "warm_work_below_min"
	routeReasonWorkOverflow      = "work_overflow"
)

// LinearRouteProfile selects a predefined linear routing policy.
type LinearRouteProfile string

const (
	LinearRouteProfileBalanced     LinearRouteProfile = "balanced"
	LinearRouteProfileConservative LinearRouteProfile = "conservative"
	LinearRouteProfileAggressive   LinearRouteProfile = "aggressive"
	LinearRouteProfileDisabled     LinearRouteProfile = "disabled"
)

// RouteDecision captures whether a call should use ANE or MLX fallback.
type RouteDecision struct {
	UseANE bool
	Reason string
}

// LinearRouteConfig controls the ANE routing policy for linear calls.
type LinearRouteConfig struct {
	// MinSpatial requires batch>=MinSpatial before ANE is considered.
	// Value 0 uses the default. Set <0 to disable this check.
	MinSpatial int

	// ChannelMultiple requires in/out dimensions to be divisible by this value.
	// Value 0 uses the default. Set <0 to disable this check.
	ChannelMultiple int

	// MaxCompileCacheSize prevents new model compiles when cache size reaches
	// this value. Value 0 uses the default. Set <0 to disable this check.
	MaxCompileCacheSize int

	// ColdMinMACs is the minimum MAC count required when route info indicates
	// the model key is not cached (or cache status is unknown).
	ColdMinMACs uint64

	// WarmMinMACs is the minimum MAC count required when route info indicates
	// the model key is cached.
	WarmMinMACs uint64
}

// DefaultLinearRouteConfig returns a conservative policy based on measured
// adapter constraints and compile-budget limits.
func DefaultLinearRouteConfig() LinearRouteConfig {
	return LinearRouteConfigForProfile(LinearRouteProfileBalanced)
}

// LinearRouteConfigForProfile returns a preset policy.
func LinearRouteConfigForProfile(profile LinearRouteProfile) LinearRouteConfig {
	switch profile {
	case LinearRouteProfileConservative:
		return LinearRouteConfig{
			MinSpatial:          32,
			ChannelMultiple:     16,
			MaxCompileCacheSize: 80,
		}
	case LinearRouteProfileAggressive:
		return LinearRouteConfig{
			MinSpatial:          16,
			ChannelMultiple:     -1,
			MaxCompileCacheSize: 110,
		}
	case LinearRouteProfileDisabled:
		return LinearRouteConfig{
			MinSpatial:          -1,
			ChannelMultiple:     -1,
			MaxCompileCacheSize: -1,
		}
	default:
		return LinearRouteConfig{
			MinSpatial:          16,
			ChannelMultiple:     8,
			MaxCompileCacheSize: 100,
		}
	}
}

// LinearRouteInput provides route-time facts for one linear call.
type LinearRouteInput struct {
	Batch  int
	InDim  int
	OutDim int

	CacheKnown bool
	CacheHit   bool
	CacheSize  int
}

// LinearRouter applies a deterministic policy for linear ANE routing.
type LinearRouter struct {
	cfg LinearRouteConfig
}

// NewLinearRouter builds a router from cfg.
func NewLinearRouter(cfg LinearRouteConfig) *LinearRouter {
	return &LinearRouter{cfg: normalizeLinearRouteConfig(cfg)}
}

// Config returns the normalized router config.
func (r *LinearRouter) Config() LinearRouteConfig {
	if r == nil {
		return normalizeLinearRouteConfig(LinearRouteConfig{})
	}
	return r.cfg
}

// DecideLinear returns whether the request should be routed to ANE.
func (r *LinearRouter) DecideLinear(in LinearRouteInput) RouteDecision {
	cfg := r.Config()
	if in.Batch <= 0 || in.InDim <= 0 || in.OutDim <= 0 {
		return RouteDecision{UseANE: false, Reason: routeReasonInvalidDims}
	}
	if cfg.MinSpatial > 0 && in.Batch < cfg.MinSpatial {
		return RouteDecision{UseANE: false, Reason: routeReasonSmallSpatial}
	}
	if cfg.ChannelMultiple > 0 {
		if in.InDim%cfg.ChannelMultiple != 0 || in.OutDim%cfg.ChannelMultiple != 0 {
			return RouteDecision{UseANE: false, Reason: routeReasonUnalignedChannels}
		}
	}
	if cfg.MaxCompileCacheSize > 0 && in.CacheKnown && !in.CacheHit && in.CacheSize >= cfg.MaxCompileCacheSize {
		return RouteDecision{UseANE: false, Reason: routeReasonCompileBudget}
	}

	macs, ok := linearMACs(in.Batch, in.InDim, in.OutDim)
	if !ok {
		return RouteDecision{UseANE: false, Reason: routeReasonWorkOverflow}
	}

	if in.CacheKnown && in.CacheHit {
		if cfg.WarmMinMACs > 0 && macs < cfg.WarmMinMACs {
			return RouteDecision{UseANE: false, Reason: routeReasonWarmTooSmall}
		}
		return RouteDecision{UseANE: true, Reason: routeReasonCached}
	}
	if cfg.ColdMinMACs > 0 && macs < cfg.ColdMinMACs {
		return RouteDecision{UseANE: false, Reason: routeReasonColdTooSmall}
	}
	return RouteDecision{UseANE: true, Reason: routeReasonEligible}
}

func normalizeLinearRouteConfig(cfg LinearRouteConfig) LinearRouteConfig {
	if cfg == (LinearRouteConfig{}) {
		return DefaultLinearRouteConfig()
	}
	return cfg
}

func linearMACs(batch, inDim, outDim int) (uint64, bool) {
	if batch < 0 || inDim < 0 || outDim < 0 {
		return 0, false
	}
	b := uint64(batch)
	in := uint64(inDim)
	out := uint64(outDim)
	if b > 0 && in > math.MaxUint64/b {
		return 0, false
	}
	bi := b * in
	if bi > 0 && out > math.MaxUint64/bi {
		return 0, false
	}
	return bi * out, true
}
