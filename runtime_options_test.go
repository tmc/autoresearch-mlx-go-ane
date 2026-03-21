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
		LinearRouteProfile:         LinearRouteProfileConservative,
		TrainingLinearRouteProfile: LinearRouteProfileBalanced,
	})
	if rt.Router == nil {
		t.Fatal("Router is nil for conservative profile")
	}
	cfg := rt.Router.Config()
	if cfg.MinSpatial != 32 || cfg.ChannelMultiple != 16 || cfg.MaxCompileCacheSize != 80 {
		t.Fatalf("conservative config=%+v", cfg)
	}
	if rt.TrainingRouter == nil {
		t.Fatal("TrainingRouter is nil for balanced training profile")
	}
	trainingCfg := rt.TrainingRouter.Config()
	if trainingCfg.MinSpatial != 8 || trainingCfg.ChannelMultiple != 8 || trainingCfg.MaxCompileCacheSize != 128 {
		t.Fatalf("balanced training config=%+v", trainingCfg)
	}
}

func TestNewRuntimeWithOptionsRouterOverridesProfile(t *testing.T) {
	custom := NewLinearRouter(LinearRouteConfig{
		MinSpatial:          48,
		ChannelMultiple:     32,
		MaxCompileCacheSize: 64,
	})
	trainingCustom := NewLinearRouter(LinearRouteConfig{
		MinSpatial:          4,
		ChannelMultiple:     -1,
		MaxCompileCacheSize: 256,
	})
	rt := NewRuntimeWithOptions(RuntimeOptions{
		Router:                     custom,
		TrainingRouter:             trainingCustom,
		LinearRouteProfile:         LinearRouteProfileAggressive,
		TrainingLinearRouteProfile: LinearRouteProfileConservative,
	})
	if rt.Router != custom {
		t.Fatal("custom router was not preserved")
	}
	if rt.TrainingRouter != trainingCustom {
		t.Fatal("custom training router was not preserved")
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
