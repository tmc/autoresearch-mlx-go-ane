//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import "testing"

func TestApplySurfaceSharedEventModeFromEnv(t *testing.T) {
	tests := []struct {
		name       string
		env        string
		wantWait   bool
		wantSignal bool
	}{
		{name: "off default", env: "", wantWait: false, wantSignal: false},
		{name: "wait", env: "wait", wantWait: true, wantSignal: false},
		{name: "signal", env: "signal", wantWait: false, wantSignal: true},
		{name: "wait signal", env: "wait_signal", wantWait: true, wantSignal: true},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(sharedEventModeEnv, tc.env)
			cfg := SurfaceEvalPlanConfig{}
			applySurfaceSharedEventModeFromEnv(&cfg)
			if cfg.EnableMetalWait != tc.wantWait {
				t.Fatalf("EnableMetalWait=%v want=%v", cfg.EnableMetalWait, tc.wantWait)
			}
			if cfg.EnableMetalSignal != tc.wantSignal {
				t.Fatalf("EnableMetalSignal=%v want=%v", cfg.EnableMetalSignal, tc.wantSignal)
			}
		})
	}
}

func TestApplySharedEventModeFromEnv(t *testing.T) {
	tests := []struct {
		name       string
		env        string
		wantWait   bool
		wantSignal bool
	}{
		{name: "off default", env: "", wantWait: false, wantSignal: false},
		{name: "wait", env: "wait", wantWait: true, wantSignal: false},
		{name: "signal", env: "signal", wantWait: false, wantSignal: true},
		{name: "wait signal", env: "wait_signal", wantWait: true, wantSignal: true},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(sharedEventModeEnv, tc.env)
			cfg := MultiSurfaceEvalPlanConfig{}
			applySharedEventModeFromEnv(&cfg)
			if cfg.EnableMetalWait != tc.wantWait {
				t.Fatalf("EnableMetalWait=%v want=%v", cfg.EnableMetalWait, tc.wantWait)
			}
			if cfg.EnableMetalSignal != tc.wantSignal {
				t.Fatalf("EnableMetalSignal=%v want=%v", cfg.EnableMetalSignal, tc.wantSignal)
			}
		})
	}
}
