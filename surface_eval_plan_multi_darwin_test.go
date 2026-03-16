//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import "testing"

func TestDefaultMultiSurfaceEvalPlanConfig(t *testing.T) {
	cfg := DefaultMultiSurfaceEvalPlanConfig()
	if cfg.QoS != defaultANEQoS {
		t.Fatalf("QoS=%d want=%d", cfg.QoS, defaultANEQoS)
	}
	if !cfg.PreferDirect {
		t.Fatal("PreferDirect=false want=true")
	}
	if cfg.EnableMetalWait {
		t.Fatal("EnableMetalWait=true want=false")
	}
	if cfg.EnableMetalSignal {
		t.Fatal("EnableMetalSignal=true want=false")
	}
	if cfg.WaitValue == 0 {
		t.Fatal("WaitValue=0 want non-zero")
	}
	if cfg.SignalValue == 0 {
		t.Fatal("SignalValue=0 want non-zero")
	}
	if cfg.EnableFWToFWSignal {
		t.Fatal("EnableFWToFWSignal=true want=false")
	}
	if cfg.SignalOutputBinding != 0 {
		t.Fatalf("SignalOutputBinding=%d want=0", cfg.SignalOutputBinding)
	}
}

func TestMultiSurfaceEvalPlanNilSetters(t *testing.T) {
	var plan *MultiSurfaceEvalPlan
	if err := plan.SetWaitEventSignaledValue(1); err == nil {
		t.Fatal("SetWaitEventSignaledValue error=nil")
	}
	if err := plan.SetSignalEventSignaledValue(1); err == nil {
		t.Fatal("SetSignalEventSignaledValue error=nil")
	}
}
