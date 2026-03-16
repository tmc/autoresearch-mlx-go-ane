package mlxgoane

import "testing"

func TestNewRuntimeWithOptionsDefaults(t *testing.T) {
	rt := NewRuntimeWithOptions(RuntimeOptions{})
	if rt == nil {
		t.Fatal("runtime is nil")
	}
	if !rt.AllowFallback {
		t.Fatal("AllowFallback=false want true")
	}
	if rt.Router != nil {
		t.Fatal("Router is set without profile/options")
	}
}

func TestNewRuntimeWithOptionsProfile(t *testing.T) {
	rt := NewRuntimeWithOptions(RuntimeOptions{
		LinearRouteProfile: LinearRouteProfileConservative,
	})
	if rt.Router == nil {
		t.Fatal("Router is nil for conservative profile")
	}
	cfg := rt.Router.Config()
	if cfg.MinSpatial != 32 || cfg.ChannelMultiple != 16 || cfg.MaxCompileCacheSize != 80 {
		t.Fatalf("conservative config=%+v", cfg)
	}
}

func TestNewRuntimeWithOptionsRouterOverridesProfile(t *testing.T) {
	custom := NewLinearRouter(LinearRouteConfig{
		MinSpatial:          48,
		ChannelMultiple:     32,
		MaxCompileCacheSize: 64,
	})
	rt := NewRuntimeWithOptions(RuntimeOptions{
		Router:             custom,
		LinearRouteProfile: LinearRouteProfileAggressive,
	})
	if rt.Router != custom {
		t.Fatal("custom router was not preserved")
	}
}

func TestNewRuntimeWithOptionsAllowFallback(t *testing.T) {
	no := false
	rt := NewRuntimeWithOptions(RuntimeOptions{
		AllowFallback: &no,
	})
	if rt.AllowFallback {
		t.Fatal("AllowFallback=true want false")
	}
}
