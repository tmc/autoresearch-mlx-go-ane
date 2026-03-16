package mlxgoane

import "testing"

func TestLinearRouterDecide(t *testing.T) {
	r := NewLinearRouter(DefaultLinearRouteConfig())

	tests := []struct {
		name string
		in   LinearRouteInput
		want RouteDecision
	}{
		{
			name: "reject_invalid_dims",
			in: LinearRouteInput{
				Batch: 0, InDim: 64, OutDim: 64,
			},
			want: RouteDecision{UseANE: false, Reason: routeReasonInvalidDims},
		},
		{
			name: "reject_small_spatial",
			in: LinearRouteInput{
				Batch: 8, InDim: 64, OutDim: 64,
			},
			want: RouteDecision{UseANE: false, Reason: routeReasonSmallSpatial},
		},
		{
			name: "reject_unaligned_channels",
			in: LinearRouteInput{
				Batch: 16, InDim: 62, OutDim: 64,
			},
			want: RouteDecision{UseANE: false, Reason: routeReasonUnalignedChannels},
		},
		{
			name: "reject_compile_budget_miss",
			in: LinearRouteInput{
				Batch:      16,
				InDim:      64,
				OutDim:     64,
				CacheKnown: true,
				CacheHit:   false,
				CacheSize:  100,
			},
			want: RouteDecision{UseANE: false, Reason: routeReasonCompileBudget},
		},
		{
			name: "allow_cached",
			in: LinearRouteInput{
				Batch:      16,
				InDim:      64,
				OutDim:     64,
				CacheKnown: true,
				CacheHit:   true,
				CacheSize:  100,
			},
			want: RouteDecision{UseANE: true, Reason: routeReasonCached},
		},
		{
			name: "allow_cold_eligible",
			in: LinearRouteInput{
				Batch:      16,
				InDim:      64,
				OutDim:     64,
				CacheKnown: true,
				CacheHit:   false,
				CacheSize:  12,
			},
			want: RouteDecision{UseANE: true, Reason: routeReasonEligible},
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := r.DecideLinear(tc.in)
			if got != tc.want {
				t.Fatalf("DecideLinear(%+v)=%+v want=%+v", tc.in, got, tc.want)
			}
		})
	}
}

func TestLinearRouterThresholds(t *testing.T) {
	r := NewLinearRouter(LinearRouteConfig{
		MinSpatial:          16,
		ChannelMultiple:     8,
		MaxCompileCacheSize: 100,
		ColdMinMACs:         2000,
		WarmMinMACs:         1000,
	})

	coldSmall := r.DecideLinear(LinearRouteInput{
		Batch: 16, InDim: 8, OutDim: 8, // 1024 MACs
	})
	if coldSmall.UseANE || coldSmall.Reason != routeReasonColdTooSmall {
		t.Fatalf("coldSmall=%+v want reason=%q", coldSmall, routeReasonColdTooSmall)
	}

	warmSmall := r.DecideLinear(LinearRouteInput{
		Batch:      16,
		InDim:      8,
		OutDim:     8, // 1024 MACs
		CacheKnown: true,
		CacheHit:   true,
	})
	if !warmSmall.UseANE {
		t.Fatalf("warmSmall=%+v want ANE route", warmSmall)
	}
}

func TestLinearRouterConfigDefaultsAndDisable(t *testing.T) {
	def := NewLinearRouter(LinearRouteConfig{}).Config()
	if def.MinSpatial != 16 || def.ChannelMultiple != 8 || def.MaxCompileCacheSize != 100 {
		t.Fatalf("default config=%+v", def)
	}

	disabled := NewLinearRouter(LinearRouteConfig{
		MinSpatial:          -1,
		ChannelMultiple:     -1,
		MaxCompileCacheSize: -1,
	}).DecideLinear(LinearRouteInput{
		Batch:      1,
		InDim:      3,
		OutDim:     5,
		CacheKnown: true,
		CacheHit:   false,
		CacheSize:  1_000,
	})
	if !disabled.UseANE {
		t.Fatalf("disabled checks unexpectedly rejected route: %+v", disabled)
	}
}

func TestLinearRouteConfigForProfile(t *testing.T) {
	tests := []struct {
		name    string
		profile LinearRouteProfile
		want    LinearRouteConfig
	}{
		{
			name:    "balanced",
			profile: LinearRouteProfileBalanced,
			want: LinearRouteConfig{
				MinSpatial:          16,
				ChannelMultiple:     8,
				MaxCompileCacheSize: 100,
			},
		},
		{
			name:    "conservative",
			profile: LinearRouteProfileConservative,
			want: LinearRouteConfig{
				MinSpatial:          32,
				ChannelMultiple:     16,
				MaxCompileCacheSize: 80,
			},
		},
		{
			name:    "aggressive",
			profile: LinearRouteProfileAggressive,
			want: LinearRouteConfig{
				MinSpatial:          16,
				ChannelMultiple:     -1,
				MaxCompileCacheSize: 110,
			},
		},
		{
			name:    "disabled",
			profile: LinearRouteProfileDisabled,
			want: LinearRouteConfig{
				MinSpatial:          -1,
				ChannelMultiple:     -1,
				MaxCompileCacheSize: -1,
			},
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := LinearRouteConfigForProfile(tc.profile)
			if got != tc.want {
				t.Fatalf("LinearRouteConfigForProfile(%q)=%+v want=%+v", tc.profile, got, tc.want)
			}
		})
	}
}

func TestLinearMACsOverflow(t *testing.T) {
	_, ok := linearMACs(1<<30, 1<<30, 1<<30)
	if ok {
		t.Fatal("linearMACs overflow unexpectedly reported ok")
	}
}
