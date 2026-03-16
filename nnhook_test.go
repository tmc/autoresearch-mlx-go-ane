package mlxgoane

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/nn"
)

func TestInstallNNLinearHookRoutesForward(t *testing.T) {
	rt := NewRuntime(fakeExecutor{y: []float32{10, 20, 30, 40}})
	restore := InstallNNLinearHook(rt)
	t.Cleanup(restore)

	weight, err := mlx.FromSlice([]float32{1, 0, 0, 1}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice weight: %v", err)
	}
	defer weight.Free()

	layer := nn.NewLinearWithArrays(weight, nil)

	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()

	y := layer.Forward(x)
	defer y.Free()

	got := mlx.MustToSlice[float32](y)
	want := []float32{10, 20, 30, 40}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("y[%d]=%g want=%g", i, got[i], want[i])
		}
	}
}

func TestInstallNNLinearHookAppliesBias(t *testing.T) {
	rt := NewRuntime(fakeExecutor{y: []float32{1, 2, 3, 4}})
	restore := InstallNNLinearHook(rt)
	t.Cleanup(restore)

	weight, err := mlx.FromSlice([]float32{1, 0, 0, 1}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice weight: %v", err)
	}
	defer weight.Free()
	bias, err := mlx.FromSlice([]float32{10, 20}, []int{2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice bias: %v", err)
	}
	defer bias.Free()

	layer := nn.NewLinearWithArrays(weight, bias)

	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()

	y := layer.Forward(x)
	defer y.Free()

	got := mlx.MustToSlice[float32](y)
	want := []float32{11, 22, 13, 24}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("y[%d]=%g want=%g", i, got[i], want[i])
		}
	}
}

type fakeTelemetryExecutor struct {
	fakeExecutor
	telemetry LinearTelemetry
}

func (f fakeTelemetryExecutor) LastLinearTelemetry() LinearTelemetry { return f.telemetry }
func (f fakeTelemetryExecutor) LinearCacheSize() int                 { return 1 }

func TestInstallNNLinearHookWithStatsRecordsANEAndFallback(t *testing.T) {
	weight, err := mlx.FromSlice([]float32{1, 0, 0, 1}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice weight: %v", err)
	}
	defer weight.Free()

	x, err := mlx.FromSlice([]float32{1, 2, 3, 4}, []int{2, 2}, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice x: %v", err)
	}
	defer x.Free()

	statsANE := NewLinearHookStats()
	rtANE := NewRuntime(fakeTelemetryExecutor{
		fakeExecutor: fakeExecutor{y: []float32{10, 20, 30, 40}},
		telemetry: LinearTelemetry{
			CacheHit: true,
			Build:    2 * time.Millisecond,
			Compile:  3 * time.Millisecond,
			Load:     4 * time.Millisecond,
			Evaluate: 5 * time.Millisecond,
		},
	})
	restore := InstallNNLinearHookWithStats(rtANE, statsANE)
	layer := nn.NewLinearWithArrays(weight, nil)
	y := layer.Forward(x)
	y.Free()
	restore()

	gotANE := statsANE.Snapshot()
	if gotANE.TotalCalls != 1 || gotANE.ANECalls != 1 || gotANE.MLXCalls != 0 {
		t.Fatalf("ANE snapshot = %+v", gotANE)
	}
	if gotANE.CacheHits != 1 || gotANE.CacheMisses != 0 {
		t.Fatalf("ANE cache stats = %+v", gotANE)
	}
	if gotANE.BuildTotal == 0 || gotANE.CompileTotal == 0 || gotANE.LoadTotal == 0 || gotANE.EvaluateTotal == 0 {
		t.Fatalf("ANE telemetry totals missing: %+v", gotANE)
	}

	statsFallback := NewLinearHookStats()
	rtFallback := NewRuntime(fakeExecutor{err: context.DeadlineExceeded})
	restore = InstallNNLinearHookWithStats(rtFallback, statsFallback)
	y = layer.Forward(x)
	y.Free()
	restore()

	gotFallback := statsFallback.Snapshot()
	if gotFallback.TotalCalls != 1 || gotFallback.MLXCalls != 1 || gotFallback.ErrorFallbacks != 1 {
		t.Fatalf("fallback snapshot = %+v", gotFallback)
	}
	if gotFallback.FallbackReasons[context.DeadlineExceeded.Error()] != 1 {
		t.Fatalf("fallback reasons = %+v", gotFallback.FallbackReasons)
	}
}

func TestLinearHookStatsResetAndFraction(t *testing.T) {
	stats := NewLinearHookStats()
	stats.record(nil, &LinearResult{Backend: BackendANE})
	stats.record(nil, &LinearResult{Backend: BackendMLX, FallbackReason: "router: small"})

	got := stats.Snapshot()
	if got.TotalCalls != 2 || got.ANECalls != 1 || got.MLXCalls != 1 {
		t.Fatalf("snapshot = %+v", got)
	}
	if got.ANEFraction() != 0.5 {
		t.Fatalf("ANEFraction=%v want 0.5", got.ANEFraction())
	}
	if got.ZeroBenefit() {
		t.Fatal("ZeroBenefit=true want false")
	}

	stats.Reset()
	reset := stats.Snapshot()
	if reset.TotalCalls != 0 || reset.ANECalls != 0 || reset.MLXCalls != 0 {
		t.Fatalf("reset snapshot = %+v", reset)
	}
	if len(reset.FallbackReasons) != 0 {
		t.Fatalf("reset fallback reasons = %+v", reset.FallbackReasons)
	}
}
