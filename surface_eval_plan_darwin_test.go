//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/apple/private/appleneuralengine"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/nn"
)

func TestIOSurfaceFloat32RoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		count int
	}{
		{name: "small", count: 1},
		{name: "vector16", count: 16},
		{name: "vector1024", count: 1024},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			surf, err := NewIOSurfaceFloat32(tc.count)
			if err != nil {
				t.Fatalf("NewIOSurfaceFloat32: %v", err)
			}
			defer surf.Close()
			if got := surf.ID(); got == 0 {
				t.Fatalf("IOSurface ID=0")
			}
			if got, want := surf.ByteLen(), tc.count*4; got != want {
				t.Fatalf("ByteLen=%d want=%d", got, want)
			}
			in := makeDeterministicTensor(tc.count, 0.25, 19)
			if err := surf.Write(in); err != nil {
				t.Fatalf("Write: %v", err)
			}
			out, err := surf.Read()
			if err != nil {
				t.Fatalf("Read: %v", err)
			}
			requireClose(t, tc.name, out, in, 0)
		})
	}
}

func TestIOSurfaceFloat32StridedFP16RoundTrip(t *testing.T) {
	layout := compiledTensorLayout{
		Channels:    2,
		Height:      3,
		Width:       4,
		ElemSize:    2,
		RowStride:   64,
		PlaneStride: 3 * 64,
		Name:        "test",
		Symbol:      "test",
	}
	surf, err := newIOSurfaceFloat32WithLayout(layout)
	if err != nil {
		t.Fatalf("newIOSurfaceFloat32WithLayout: %v", err)
	}
	defer surf.Close()
	if got, want := surf.ByteLen(), layout.allocSize(); got != want {
		t.Fatalf("ByteLen=%d want=%d", got, want)
	}

	rowSize := layout.Channels * layout.Width
	row1 := make([]float32, rowSize)
	for i := range row1 {
		row1[i] = 0.25 + float32(i)/16
	}
	if err := surf.WriteAt(rowSize, row1); err != nil {
		t.Fatalf("WriteAt: %v", err)
	}

	full, err := surf.Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	want := make([]float32, layout.logicalCount())
	copy(want[rowSize:2*rowSize], row1)
	requireClose(t, "strided_fp16_full", full, want, 2e-2)

	gotRow, err := surf.ReadAt(rowSize, rowSize)
	if err != nil {
		t.Fatalf("ReadAt: %v", err)
	}
	requireClose(t, "strided_fp16_row", gotRow, row1, 2e-2)
}

func TestIOSurfaceFloat32LockReadOnly(t *testing.T) {
	surf, err := NewIOSurfaceFloat32(8)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()

	in := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if err := surf.Write(in); err != nil {
		t.Fatalf("Write: %v", err)
	}

	base, n, err := surf.LockReadOnly()
	if err != nil {
		t.Fatalf("LockReadOnly: %v", err)
	}
	if n != len(in)*4 {
		t.Fatalf("LockReadOnly bytes=%d want=%d", n, len(in)*4)
	}
	raw := unsafe.Slice((*float32)(base), len(in))
	requireClose(t, "lock_read", raw, in, 0)
	if err := surf.UnlockReadOnly(); err != nil {
		t.Fatalf("UnlockReadOnly: %v", err)
	}
}

func TestIOSurfaceFloat32ReadOnlyView(t *testing.T) {
	surf, err := NewIOSurfaceFloat32(4)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32: %v", err)
	}
	defer surf.Close()

	in := []float32{9, 8, 7, 6}
	if err := surf.Write(in); err != nil {
		t.Fatalf("Write: %v", err)
	}

	view, err := surf.ReadOnlyView()
	if err != nil {
		t.Fatalf("ReadOnlyView: %v", err)
	}
	if view.Pointer() == nil {
		t.Fatalf("ReadOnlyView pointer=nil")
	}
	if got, want := view.ByteLen(), len(in)*4; got != want {
		t.Fatalf("ReadOnlyView bytes=%d want=%d", got, want)
	}
	requireClose(t, "readonly_view", view.Float32s(), in, 0)
	if err := view.Close(); err != nil {
		t.Fatalf("ReadOnlyView.Close: %v", err)
	}
}

func TestWrapSharedEventPort(t *testing.T) {
	ev, err := WrapSharedEventPort(17, 9)
	if err != nil {
		t.Fatalf("WrapSharedEventPort: %v", err)
	}
	if got, want := ev.Port(), uint32(17); got != want {
		t.Fatalf("Port=%d want=%d", got, want)
	}
	if got, want := ev.Value(), uint64(9); got != want {
		t.Fatalf("Value=%d want=%d", got, want)
	}
	if err := ev.SetSignaledValue(10); err == nil {
		t.Fatalf("SetSignaledValue succeeded without retained shared object")
	}
}

func TestSurfaceDecodeFFNOutputMetalBufferBindingNil(t *testing.T) {
	var stage *SurfaceDecodeFFN
	if _, err := stage.NewDefaultOutputMetalBufferBinding(); err == nil {
		t.Fatal("nil stage NewDefaultOutputMetalBufferBinding error=nil")
	}
	if _, err := stage.NewDefaultWaitMetalSharedEvent(); err == nil {
		t.Fatal("nil stage NewDefaultWaitMetalSharedEvent error=nil")
	}
	if _, err := stage.NewDefaultSignalMetalSharedEvent(); err == nil {
		t.Fatal("nil stage NewDefaultSignalMetalSharedEvent error=nil")
	}

	stage = &SurfaceDecodeFFN{}
	if _, err := stage.NewDefaultOutputMetalBufferBinding(); err == nil {
		t.Fatal("empty stage NewDefaultOutputMetalBufferBinding error=nil")
	}
	if _, err := stage.NewDefaultWaitMetalSharedEvent(); err == nil {
		t.Fatal("empty stage NewDefaultWaitMetalSharedEvent error=nil")
	}
	if _, err := stage.NewDefaultSignalMetalSharedEvent(); err == nil {
		t.Fatal("empty stage NewDefaultSignalMetalSharedEvent error=nil")
	}
}

func TestDefaultSurfaceEvalPlanConfig(t *testing.T) {
	cfg := DefaultSurfaceEvalPlanConfig()
	if cfg.QoS != defaultANEQoS {
		t.Fatalf("QoS=%d want=%d", cfg.QoS, defaultANEQoS)
	}
	if !cfg.PreferDirect {
		t.Fatalf("PreferDirect=false want=true")
	}
	if cfg.EnableMetalWait {
		t.Fatalf("EnableMetalWait=true want=false")
	}
	if cfg.EnableMetalSignal {
		t.Fatalf("EnableMetalSignal=true want=false")
	}
	if cfg.WaitEventPort != 0 {
		t.Fatalf("WaitEventPort=%d want=0", cfg.WaitEventPort)
	}
	if cfg.SignalEventPort != 0 {
		t.Fatalf("SignalEventPort=%d want=0", cfg.SignalEventPort)
	}
	if cfg.WaitValue == 0 {
		t.Fatalf("WaitValue=0 want non-zero")
	}
	if cfg.SignalValue == 0 {
		t.Fatalf("SignalValue=0 want non-zero")
	}
	if cfg.BridgeClientHandle != 0 {
		t.Fatalf("BridgeClientHandle=%d want=0", cfg.BridgeClientHandle)
	}
	if cfg.BridgeModelPath != "" {
		t.Fatalf("BridgeModelPath=%q want empty", cfg.BridgeModelPath)
	}
	if cfg.BridgeModelKey != "s" {
		t.Fatalf("BridgeModelKey=%q want=s", cfg.BridgeModelKey)
	}
}

func TestSurfaceEvalPlanManualSharedIOSurfaceIntegration(t *testing.T) {
	if os.Getenv("MLXGO_ANE_INTEGRATION_CHAIN") == "" {
		t.Skip("set MLXGO_ANE_INTEGRATION_CHAIN=1 to run manual IOSurface sharing integration test")
	}
	if testing.Short() {
		t.Skip("skipping ANE integration test in short mode")
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		batch  = 16
		inDim  = 64
		midDim = 64
		outDim = 64
	)
	xHost := makeDeterministicTensor(batch*inDim, 0.02, 29)
	wA := makeDeterministicTensor(midDim*inDim, 0.015, 31)
	wB := makeDeterministicTensor(outDim*midDim, 0.01, 23)

	modelA, _, err := buildLinearModel(linearModelKey{
		batch:      batch,
		inDim:      inDim,
		outDim:     midDim,
		weightHash: hashFloat32Slice(wA),
	}, wA)
	if err != nil {
		t.Fatalf("buildLinearModel A: %v", err)
	}
	defer unloadLinearModelForTest(t, modelA)

	modelB, _, err := buildLinearModel(linearModelKey{
		batch:      batch,
		inDim:      midDim,
		outDim:     outDim,
		weightHash: hashFloat32Slice(wB),
	}, wB)
	if err != nil {
		t.Fatalf("buildLinearModel B: %v", err)
	}
	defer unloadLinearModelForTest(t, modelB)

	xANE, err := linearInputHostToANE(xHost, batch, inDim)
	if err != nil {
		t.Fatalf("linearInputHostToANE: %v", err)
	}

	inputSurf, err := NewIOSurfaceFloat32(len(xANE))
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32 input: %v", err)
	}
	defer inputSurf.Close()
	sharedSurf, err := NewIOSurfaceFloat32(batch * midDim)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32 shared: %v", err)
	}
	defer sharedSurf.Close()
	outputSurf, err := NewIOSurfaceFloat32(batch * outDim)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32 output: %v", err)
	}
	defer outputSurf.Close()

	if err := inputSurf.Write(xANE); err != nil {
		t.Fatalf("inputSurf.Write: %v", err)
	}

	cfg := DefaultSurfaceEvalPlanConfig()
	cfg.PreferDirect = true
	planA, err := NewSurfaceEvalPlan(modelA, inputSurf, sharedSurf, cfg)
	if err != nil {
		t.Fatalf("NewSurfaceEvalPlan A: %v", err)
	}
	defer planA.Close()

	planB, err := NewSurfaceEvalPlan(modelB, sharedSurf, outputSurf, cfg)
	if err != nil {
		t.Fatalf("NewSurfaceEvalPlan B: %v", err)
	}
	defer planB.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := planA.Eval(ctx); err != nil {
		t.Fatalf("planA.Eval: %v", err)
	}
	if err := planB.Eval(ctx); err != nil {
		t.Fatalf("planB.Eval: %v", err)
	}

	yANE, err := outputSurf.Read()
	if err != nil {
		t.Fatalf("outputSurf.Read: %v", err)
	}
	yHost, err := linearOutputANEToHost(yANE, batch, outDim)
	if err != nil {
		t.Fatalf("linearOutputANEToHost: %v", err)
	}

	wantMid := linearReference(xHost, wA, batch, inDim, midDim)
	want := linearReference(wantMid, wB, batch, midDim, outDim)
	requireClose(t, "manual_chain", yHost, want, 2e-2)
}

func TestSurfaceEvalPlanMetalWaitIntegration(t *testing.T) {
	if os.Getenv("MLXGO_ANE_INTEGRATION_SHARED_WAIT") == "" {
		t.Skip("set MLXGO_ANE_INTEGRATION_SHARED_WAIT=1 to run shared wait-event integration test")
	}
	if os.Getenv("MLXGO_ANE_WAIT_CHILD") == "" {
		cmd := exec.Command(os.Args[0], "-test.run", "^TestSurfaceEvalPlanMetalWaitIntegration$")
		cmd.Env = append(os.Environ(),
			"MLXGO_ANE_INTEGRATION_SHARED_WAIT=1",
			"MLXGO_ANE_WAIT_CHILD=1",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			if os.Getenv("MLXGO_ANE_STRICT_SHARED_WAIT") != "" {
				t.Fatalf("wait-event child failed: %v\n%s", err, out)
			}
			t.Skipf("wait-event child failed (non-strict): %v\n%s", err, out)
		}
		return
	}

	if testing.Short() {
		t.Skip("skipping ANE integration test in short mode")
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		batch  = 16
		inDim  = 64
		outDim = 64
	)
	xHost := makeDeterministicTensor(batch*inDim, 0.03, 17)
	w := makeDeterministicTensor(outDim*inDim, 0.02, 13)

	model, _, err := buildLinearModel(linearModelKey{
		batch:      batch,
		inDim:      inDim,
		outDim:     outDim,
		weightHash: hashFloat32Slice(w),
	}, w)
	if err != nil {
		t.Fatalf("buildLinearModel: %v", err)
	}
	defer unloadLinearModelForTest(t, model)

	xANE, err := linearInputHostToANE(xHost, batch, inDim)
	if err != nil {
		t.Fatalf("linearInputHostToANE: %v", err)
	}
	inputSurf, err := NewIOSurfaceFloat32(len(xANE))
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32 input: %v", err)
	}
	defer inputSurf.Close()
	outputSurf, err := NewIOSurfaceFloat32(batch * outDim)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32 output: %v", err)
	}
	defer outputSurf.Close()
	if err := inputSurf.Write(xANE); err != nil {
		t.Fatalf("inputSurf.Write: %v", err)
	}

	cfg := DefaultSurfaceEvalPlanConfig()
	cfg.EnableMetalWait = true
	cfg.WaitValue = 1
	cfg.EnableFWToFWSignal = true
	plan, err := NewSurfaceEvalPlan(model, inputSurf, outputSurf, cfg)
	if err != nil {
		t.Fatalf("NewSurfaceEvalPlan: %v", err)
	}
	defer plan.Close()
	waitEvent := plan.WaitEvent()
	if waitEvent == nil {
		t.Fatalf("WaitEvent=nil")
	}
	if waitEvent.Port() == 0 {
		t.Fatalf("wait-event port is zero")
	}
	if got, want := waitEvent.Value(), plan.WaitValue(); got != want {
		t.Fatalf("wait-event value=%d want=%d", got, want)
	}
	if err := waitEvent.SetSignaledValue(waitEvent.Value()); err != nil {
		t.Fatalf("WaitEvent.SetSignaledValue: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := plan.Eval(ctx); err != nil {
		t.Fatalf("plan.Eval: %v", err)
	}

	gotANE, err := outputSurf.Read()
	if err != nil {
		t.Fatalf("outputSurf.Read: %v", err)
	}
	got, err := linearOutputANEToHost(gotANE, batch, outDim)
	if err != nil {
		t.Fatalf("linearOutputANEToHost: %v", err)
	}
	want := linearReference(xHost, w, batch, inDim, outDim)
	requireClose(t, "wait_eval", got, want, 2e-2)
}

func TestSurfaceEvalPlanMetalSignalIntegration(t *testing.T) {
	if os.Getenv("MLXGO_ANE_INTEGRATION_SHARED_SIGNAL") == "" {
		t.Skip("set MLXGO_ANE_INTEGRATION_SHARED_SIGNAL=1 to run shared signal-event integration test")
	}
	if os.Getenv("MLXGO_ANE_SIGNAL_CHILD") == "" {
		cmd := exec.Command(os.Args[0], "-test.run", "^TestSurfaceEvalPlanMetalSignalIntegration$")
		cmd.Env = append(os.Environ(),
			"MLXGO_ANE_INTEGRATION_SHARED_SIGNAL=1",
			"MLXGO_ANE_SIGNAL_CHILD=1",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			if os.Getenv("MLXGO_ANE_STRICT_SHARED_SIGNAL") != "" {
				t.Fatalf("signal-event child failed: %v\n%s", err, out)
			}
			t.Skipf("signal-event child failed (non-strict): %v\n%s", err, out)
		}
		return
	}

	if testing.Short() {
		t.Skip("skipping ANE integration test in short mode")
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		t.Skip("ANE unavailable on this host")
	}

	const (
		batch  = 16
		inDim  = 64
		outDim = 64
	)
	xHost := makeDeterministicTensor(batch*inDim, 0.03, 17)
	w := makeDeterministicTensor(outDim*inDim, 0.02, 13)

	model, _, err := buildLinearModel(linearModelKey{
		batch:      batch,
		inDim:      inDim,
		outDim:     outDim,
		weightHash: hashFloat32Slice(w),
	}, w)
	if err != nil {
		t.Fatalf("buildLinearModel: %v", err)
	}
	defer unloadLinearModelForTest(t, model)

	xANE, err := linearInputHostToANE(xHost, batch, inDim)
	if err != nil {
		t.Fatalf("linearInputHostToANE: %v", err)
	}
	inputSurf, err := NewIOSurfaceFloat32(len(xANE))
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32 input: %v", err)
	}
	defer inputSurf.Close()
	outputSurf, err := NewIOSurfaceFloat32(batch * outDim)
	if err != nil {
		t.Fatalf("NewIOSurfaceFloat32 output: %v", err)
	}
	defer outputSurf.Close()
	if err := inputSurf.Write(xANE); err != nil {
		t.Fatalf("inputSurf.Write: %v", err)
	}

	cfg := DefaultSurfaceEvalPlanConfig()
	cfg.EnableMetalSignal = true
	cfg.SignalValue = 1
	cfg.EnableFWToFWSignal = false
	plan, err := NewSurfaceEvalPlan(model, inputSurf, outputSurf, cfg)
	if err != nil {
		t.Fatalf("NewSurfaceEvalPlan: %v", err)
	}
	defer plan.Close()
	signalEvent := plan.SignalEvent()
	if signalEvent == nil {
		t.Fatalf("SignalEvent=nil")
	}
	if signalEvent.Port() == 0 {
		t.Fatalf("signal-event port is zero")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := plan.Eval(ctx); err != nil {
		t.Fatalf("plan.Eval: %v", err)
	}

	gotANE, err := outputSurf.Read()
	if err != nil {
		t.Fatalf("outputSurf.Read: %v", err)
	}
	got, err := linearOutputANEToHost(gotANE, batch, outDim)
	if err != nil {
		t.Fatalf("linearOutputANEToHost: %v", err)
	}
	want := linearReference(xHost, w, batch, inDim, outDim)
	requireClose(t, "signal_eval", got, want, 2e-2)
	if err := signalEvent.WaitCPU(signalEvent.Value(), 2*time.Second); err != nil {
		t.Fatalf("SignalEvent.WaitCPU: %v", err)
	}
}

func unloadLinearModelForTest(t *testing.T, model appleneuralengine.ANEInMemoryModel) {
	t.Helper()
	_ = callObjCBoolWithNSError(
		"appleneuralengine test unload",
		model.ID,
		"unloadWithQoS:error:",
		defaultANEQoS,
	)
}

type decodeBenchFixture struct {
	dim       int
	hiddenDim int
	seq       int
	steps     int
	profile   string
	tokens    [][]float32
	stage     *SurfaceDecodeFFN
	attnW     *mlx.Array
	rms2      *mlx.Array
	w1        *mlx.Array
	w3        *mlx.Array
	w2        *mlx.Array
}

func newDecodeBenchFixture(b *testing.B) *decodeBenchFixture {
	b.Helper()
	dim := envInt("MLXGO_ANE_BENCH_DIM", 0)
	hiddenDim := envInt("MLXGO_ANE_BENCH_HIDDEN_DIM", 0)
	var shapeProfile FFNShapeProfile
	useProfileCtor := false
	profile := "env"
	if dim == 0 || hiddenDim == 0 {
		shapeProfile, profile = decodeBenchQwen35Profile(b)
		dim = shapeProfile.Dim
		hiddenDim = shapeProfile.HiddenDim
		useProfileCtor = true
	}
	if dim <= 0 {
		dim = 1024
	}
	if hiddenDim <= 0 {
		hiddenDim = 3584
	}
	seq := envInt("MLXGO_ANE_BENCH_SEQ", 1)
	steps := envInt("MLXGO_ANE_BENCH_STEPS", 16)
	if dim <= 0 || hiddenDim <= 0 || seq <= 0 || steps <= 1 {
		b.Fatalf(
			"invalid decode bench dimensions: dim=%d hiddenDim=%d seq=%d steps=%d",
			dim, hiddenDim, seq, steps,
		)
	}

	attnWeight := makeDeterministicTensor(dim*dim, 0.002, 67)
	rms2 := makeDeterministicTensor(dim, 0.005, 17)
	w1 := makeDeterministicTensor(hiddenDim*dim, 0.003, 23)
	w3 := makeDeterministicTensor(hiddenDim*dim, 0.004, 19)
	w2 := makeDeterministicTensor(dim*hiddenDim, 0.0025, 29)

	cfg := DefaultSurfaceEvalPlanConfig()
	cfg.PreferDirect = true
	if !useProfileCtor {
		shapeProfile = FFNShapeProfile{
			Dim:       dim,
			HiddenDim: hiddenDim,
			ModelType: "env",
			Config:    "",
		}
	}
	stage, err := NewSurfaceDecodeFFNFromProfile(shapeProfile, seq, rms2, w1, w3, w2, cfg)
	if err != nil {
		b.Fatalf("NewSurfaceDecodeFFN: %v", err)
	}

	attnW, err := weightOutInToInOut(attnWeight, dim, dim)
	if err != nil {
		stage.Close()
		b.Fatalf("attention weight setup: %v", err)
	}
	rms2Arr, err := mlx.FromSlice(rms2, []int{1, 1, dim}, mlx.Float32)
	if err != nil {
		attnW.Free()
		stage.Close()
		b.Fatalf("rms2 setup: %v", err)
	}
	w1Arr, err := weightOutInToInOut(w1, hiddenDim, dim)
	if err != nil {
		rms2Arr.Free()
		attnW.Free()
		stage.Close()
		b.Fatalf("w1 setup: %v", err)
	}
	w3Arr, err := weightOutInToInOut(w3, hiddenDim, dim)
	if err != nil {
		w1Arr.Free()
		rms2Arr.Free()
		attnW.Free()
		stage.Close()
		b.Fatalf("w3 setup: %v", err)
	}
	w2Arr, err := weightOutInToInOut(w2, dim, hiddenDim)
	if err != nil {
		w3Arr.Free()
		w1Arr.Free()
		rms2Arr.Free()
		attnW.Free()
		stage.Close()
		b.Fatalf("w2 setup: %v", err)
	}

	tokens := make([][]float32, steps)
	for i := range tokens {
		tokens[i] = makeDeterministicTensor(dim*seq, 0.02+0.0001*float32(i), 31+i)
	}
	return &decodeBenchFixture{
		dim:       dim,
		hiddenDim: hiddenDim,
		seq:       seq,
		steps:     steps,
		profile:   profile,
		tokens:    tokens,
		stage:     stage,
		attnW:     attnW,
		rms2:      rms2Arr,
		w1:        w1Arr,
		w3:        w3Arr,
		w2:        w2Arr,
	}
}

func decodeBenchQwen35Profile(b *testing.B) (FFNShapeProfile, string) {
	b.Helper()

	home, err := os.UserHomeDir()
	if err != nil {
		return FFNShapeProfile{
			Dim:       1024,
			HiddenDim: 3584,
			ModelType: "qwen3_5_text",
			Config:    "",
		}, "qwen3.5-fallback-home"
	}
	configPath := os.Getenv("MLXGO_ANE_BENCH_CONFIG")
	if configPath == "" {
		configPath = DefaultQwen35SmallConfigPath(home)
	}
	p, err := LoadFFNShapeProfileFromConfig(configPath)
	if err != nil {
		b.Logf("decode bench qwen3.5 config parse failed path=%s err=%v (using fallback dims)", configPath, err)
		return FFNShapeProfile{
			Dim:       1024,
			HiddenDim: 3584,
			ModelType: "qwen3_5_text",
			Config:    "",
		}, "qwen3.5-fallback-parse"
	}
	return p, fmt.Sprintf("%s (%s)", p.ModelType, p.Config)
}

func (f *decodeBenchFixture) Close() {
	if f == nil {
		return
	}
	if f.stage != nil {
		f.stage.Close()
	}
	if f.w2 != nil {
		f.w2.Free()
	}
	if f.w3 != nil {
		f.w3.Free()
	}
	if f.w1 != nil {
		f.w1.Free()
	}
	if f.rms2 != nil {
		f.rms2.Free()
	}
	if f.attnW != nil {
		f.attnW.Free()
	}
}

func BenchmarkDecodeStep(b *testing.B) {
	if os.Getenv("MLXGO_ANE_BENCH_DECODE") == "" {
		b.Skip("set MLXGO_ANE_BENCH_DECODE=1 to run decode benchmark")
	}
	if !appleneuralengine.GetANEDeviceInfoClass().HasANE() {
		b.Skip("ANE unavailable on this host")
	}
	fx := newDecodeBenchFixture(b)
	defer fx.Close()
	strictSurfacePlan := os.Getenv("MLXGO_ANE_BENCH_REQUIRE_SURFACE_PLAN") != ""
	b.Logf(
		"decode fixture profile=%s dim=%d hidden=%d seq=%d steps=%d surface_plan=%v strict_surface_plan=%v",
		fx.profile, fx.dim, fx.hiddenDim, fx.seq, fx.steps, fx.stage.UsingSurfacePlan(), strictSurfacePlan,
	)
	if strictSurfacePlan && !fx.stage.UsingSurfacePlan() {
		b.Fatalf("surface plan required but unavailable: %v", fx.stage.PlanError())
	}

	b.ReportAllocs()

	b.Run("baseline_metal_only", func(b *testing.B) {
		ctx := context.Background()
		var total time.Duration
		for i := 0; i < b.N; i++ {
			start := time.Now()
			for step := 0; step < fx.steps; step++ {
				att, err := fx.metalAttention(fx.tokens[step])
				if err != nil {
					b.Fatalf("metal attention step=%d: %v", step, err)
				}
				y, err := fx.metalFFN(att)
				if err != nil {
					b.Fatalf("metal ffn step=%d: %v", step, err)
				}
				arr, err := mlx.FromSlice(y, []int{1, fx.seq, fx.dim}, mlx.Float32)
				if err != nil {
					b.Fatalf("materialize metal output step=%d: %v", step, err)
				}
				arr.Free()
				select {
				case <-ctx.Done():
					b.Fatalf("context canceled in baseline: %v", ctx.Err())
				default:
				}
			}
			total += time.Since(start)
		}
		perStep := float64(total.Microseconds()) / float64(b.N*fx.steps)
		b.ReportMetric(perStep, "total_us/step")
	})

	b.Run("ane_ffn_sequential", func(b *testing.B) {
		ctx := context.Background()
		var copyIn, ane, copyOut, total time.Duration
		for i := 0; i < b.N; i++ {
			for step := 0; step < fx.steps; step++ {
				att, err := fx.metalAttention(fx.tokens[step])
				if err != nil {
					b.Fatalf("metal attention step=%d: %v", step, err)
				}
				res, err := fx.stage.Eval(ctx, att)
				if err != nil {
					if strictSurfacePlan {
						b.Fatalf("ANE sequential path failed at step=%d: %v", step, err)
					}
					b.Skipf("ANE sequential path unavailable at step=%d: %v", step, err)
					return
				}
				outArrStart := time.Now()
				outArr, err := mlx.FromSlice(res.Y, []int{1, fx.seq, fx.dim}, mlx.Float32)
				if err != nil {
					b.Fatalf("materialize ANE output step=%d: %v", step, err)
				}
				outArr.Free()
				outArrDur := time.Since(outArrStart)

				copyIn += res.Timing.CopyIn
				ane += res.Timing.ANE
				copyOut += res.Timing.CopyOut + outArrDur
				total += res.Timing.Total + outArrDur
			}
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(copyOut.Microseconds())/steps, "copy_out_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	b.Run("ane_ffn_async_overlap", func(b *testing.B) {
		ctx := context.Background()
		var copyIn, ane, copyOut, total time.Duration
		for i := 0; i < b.N; i++ {
			att0, err := fx.metalAttention(fx.tokens[0])
			if err != nil {
				b.Fatalf("metal attention step=0: %v", err)
			}
			copyInStart := time.Now()
			pending := fx.stage.EvalAsync(ctx, att0)
			copyIn += time.Since(copyInStart)

			for step := 1; step < fx.steps; step++ {
				// Overlap: compute current attention while previous ANE eval runs.
				att, err := fx.metalAttention(fx.tokens[step])
				if err != nil {
					b.Fatalf("metal attention step=%d: %v", step, err)
				}

				prev := <-pending
				if prev.Err != nil {
					if strictSurfacePlan {
						b.Fatalf("ANE async path failed at step=%d: %v", step-1, prev.Err)
					}
					b.Skipf("ANE async path unavailable at step=%d: %v", step-1, prev.Err)
					return
				}
				outArrStart := time.Now()
				outArr, err := mlx.FromSlice(prev.Result.Y, []int{1, fx.seq, fx.dim}, mlx.Float32)
				if err != nil {
					b.Fatalf("materialize ANE output step=%d: %v", step-1, err)
				}
				outArr.Free()
				outArrDur := time.Since(outArrStart)

				copyInStart := time.Now()
				pending = fx.stage.EvalAsync(ctx, att)
				copyIn += time.Since(copyInStart)

				ane += prev.Result.Timing.ANE
				copyOut += prev.Result.Timing.CopyOut + outArrDur
				total += prev.Result.Timing.Total + outArrDur
			}

			last := <-pending
			if last.Err != nil {
				if strictSurfacePlan {
					b.Fatalf("ANE async path failed at final step: %v", last.Err)
				}
				b.Skipf("ANE async path unavailable at final step: %v", last.Err)
				return
			}
			outArrStart := time.Now()
			outArr, err := mlx.FromSlice(last.Result.Y, []int{1, fx.seq, fx.dim}, mlx.Float32)
			if err != nil {
				b.Fatalf("materialize ANE output final step: %v", err)
			}
			outArr.Free()
			outArrDur := time.Since(outArrStart)

			ane += last.Result.Timing.ANE
			copyOut += last.Result.Timing.CopyOut + outArrDur
			total += last.Result.Timing.Total + outArrDur
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(copyOut.Microseconds())/steps, "copy_out_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	b.Run("ane_surface_plan_eval_only", func(b *testing.B) {
		if !fx.stage.UsingSurfacePlan() {
			b.Skipf("surface plan unavailable: %v", fx.stage.PlanError())
		}
		ctx := context.Background()
		var copyIn, ane, total time.Duration
		for i := 0; i < b.N; i++ {
			for step := 0; step < fx.steps; step++ {
				att, err := fx.metalAttention(fx.tokens[step])
				if err != nil {
					b.Fatalf("metal attention step=%d: %v", step, err)
				}
				timing, err := fx.stage.EvalToSurface(ctx, att)
				if err != nil {
					b.Fatalf("surface-plan eval-only step=%d: %v", step, err)
				}
				copyIn += timing.CopyIn
				ane += timing.ANE
				total += timing.Total
			}
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	b.Run("ane_surface_plan_zero_copy_view", func(b *testing.B) {
		if !fx.stage.UsingSurfacePlan() {
			b.Skipf("surface plan unavailable: %v", fx.stage.PlanError())
		}
		outSurf := fx.stage.OutputSurface()
		if outSurf == nil {
			b.Skip("surface plan output IOSurface unavailable")
		}
		ctx := context.Background()
		var sink float32
		var copyIn, ane, zeroCopy, total time.Duration
		for i := 0; i < b.N; i++ {
			for step := 0; step < fx.steps; step++ {
				att, err := fx.metalAttention(fx.tokens[step])
				if err != nil {
					b.Fatalf("metal attention step=%d: %v", step, err)
				}
				timing, err := fx.stage.EvalToSurface(ctx, att)
				if err != nil {
					b.Fatalf("surface-plan zero-copy step=%d: %v", step, err)
				}
				viewStart := time.Now()
				view, err := outSurf.ReadOnlyView()
				if err != nil {
					b.Fatalf("surface-plan zero-copy ReadOnlyView step=%d: %v", step, err)
				}
				vals := view.Float32s()
				if len(vals) > 0 {
					sink += vals[0]
				}
				if err := view.Close(); err != nil {
					b.Fatalf("surface-plan zero-copy Close step=%d: %v", step, err)
				}
				viewDur := time.Since(viewStart)

				copyIn += timing.CopyIn
				ane += timing.ANE
				zeroCopy += viewDur
				total += timing.Total + viewDur
			}
		}
		decodeBenchmarkSink = sink
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(zeroCopy.Microseconds())/steps, "zero_copy_view_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	b.Run("ane_surface_plan_metal_no_copy_buffer", func(b *testing.B) {
		if !fx.stage.UsingSurfacePlan() {
			b.Skipf("surface plan unavailable: %v", fx.stage.PlanError())
		}
		outSurf := fx.stage.OutputSurface()
		if outSurf == nil {
			b.Skip("surface plan output IOSurface unavailable")
		}
		ctx := context.Background()
		var copyIn, ane, metalBuf, total time.Duration
		for i := 0; i < b.N; i++ {
			for step := 0; step < fx.steps; step++ {
				att, err := fx.metalAttention(fx.tokens[step])
				if err != nil {
					b.Fatalf("metal attention step=%d: %v", step, err)
				}
				timing, err := fx.stage.EvalToSurface(ctx, att)
				if err != nil {
					b.Fatalf("surface-plan metal buffer step=%d: %v", step, err)
				}
				bufStart := time.Now()
				buf, err := outSurf.NewDefaultMetalBufferNoCopy()
				if err != nil {
					b.Fatalf("surface-plan metal buffer step=%d: %v", step, err)
				}
				if buf.Buffer() == nil || buf.Buffer().GetID() == 0 {
					b.Fatalf("surface-plan metal buffer step=%d: zero metal buffer id", step)
				}
				if err := buf.Close(); err != nil {
					b.Fatalf("surface-plan metal buffer close step=%d: %v", step, err)
				}
				bufDur := time.Since(bufStart)

				copyIn += timing.CopyIn
				ane += timing.ANE
				metalBuf += bufDur
				total += timing.Total + bufDur
			}
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(metalBuf.Microseconds())/steps, "metal_buffer_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	b.Run("ane_surface_plan_metal_buffer_reuse", func(b *testing.B) {
		if !fx.stage.UsingSurfacePlan() {
			b.Skipf("surface plan unavailable: %v", fx.stage.PlanError())
		}
		binding, err := fx.stage.NewDefaultOutputMetalBufferBinding()
		if err != nil {
			b.Skipf("surface plan reusable metal buffer unavailable: %v", err)
		}
		defer binding.Close()

		ctx := context.Background()
		var copyIn, ane, metalSync, total time.Duration
		for i := 0; i < b.N; i++ {
			for step := 0; step < fx.steps; step++ {
				att, err := fx.metalAttention(fx.tokens[step])
				if err != nil {
					b.Fatalf("metal attention step=%d: %v", step, err)
				}
				timing, err := fx.stage.EvalToSurface(ctx, att)
				if err != nil {
					b.Fatalf("surface-plan metal buffer reuse step=%d: %v", step, err)
				}
				syncStart := time.Now()
				if err := binding.LockReadOnly(); err != nil {
					b.Fatalf("surface-plan metal buffer reuse lock step=%d: %v", step, err)
				}
				if binding.Buffer() == nil || binding.Buffer().GetID() == 0 {
					b.Fatalf("surface-plan metal buffer reuse step=%d: zero metal buffer id", step)
				}
				if err := binding.UnlockReadOnly(); err != nil {
					b.Fatalf("surface-plan metal buffer reuse unlock step=%d: %v", step, err)
				}
				syncDur := time.Since(syncStart)

				copyIn += timing.CopyIn
				ane += timing.ANE
				metalSync += syncDur
				total += timing.Total + syncDur
			}
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(metalSync.Microseconds())/steps, "metal_sync_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	if os.Getenv("MLXGO_ANE_BENCH_BRIDGE") == "" {
		b.Run("bridge_sequential", func(b *testing.B) {
			b.Skip("set MLXGO_ANE_BENCH_BRIDGE=1 to run bridge-backed benchmarks")
		})
		b.Run("bridge_async_overlap", func(b *testing.B) {
			b.Skip("set MLXGO_ANE_BENCH_BRIDGE=1 to run bridge-backed benchmarks")
		})
		b.Run("bridge_bidirectional_roundtrip", func(b *testing.B) {
			b.Skip("set MLXGO_ANE_BENCH_BRIDGE=1 to run bridge-backed benchmarks")
		})
		b.Run("bridge_copy_scaling", func(b *testing.B) {
			b.Skip("set MLXGO_ANE_BENCH_BRIDGE=1 to run bridge-backed benchmarks")
		})
		return
	}

	api, err := loadBridgeAPI()
	if err != nil {
		b.Run("bridge_sequential", func(b *testing.B) {
			b.Skipf("bridge unavailable: %v", err)
		})
		b.Run("bridge_async_overlap", func(b *testing.B) {
			b.Skipf("bridge unavailable: %v", err)
		})
		b.Run("bridge_bidirectional_roundtrip", func(b *testing.B) {
			b.Skipf("bridge unavailable: %v", err)
		})
		b.Run("bridge_copy_scaling", func(b *testing.B) {
			b.Skipf("bridge unavailable: %v", err)
		})
		return
	}

	modelPath, err := filepath.Abs("testdata/chaining/simple_add_nn.mlmodelc")
	if err != nil {
		b.Fatalf("bridge benchmark model path: %v", err)
	}

	b.Run("bridge_sequential", func(b *testing.B) {
		const ioBytes = 4096
		handle := api.open(modelPath, "s", uintptr(ioBytes), uintptr(ioBytes))
		if handle == 0 {
			b.Skip("bridge client open failed")
		}
		defer api.close(handle)

		count := ioBytes / 4
		in := make([]float32, count)
		out := make([]float32, count)

		var copyIn, ane, copyOut, total time.Duration
		for i := 0; i < b.N; i++ {
			for step := 0; step < fx.steps; step++ {
				fillBridgeInput(in, fx.tokens[step])
				t0 := time.Now()
				api.writeInput(handle, unsafe.Pointer(&in[0]), int32(len(in)))
				t1 := time.Now()
				if !api.eval(handle) {
					b.Skipf("bridge eval failed at step=%d", step)
					return
				}
				t2 := time.Now()
				api.readOutput(handle, unsafe.Pointer(&out[0]), int32(len(out)))
				t3 := time.Now()

				copyIn += t1.Sub(t0)
				ane += t2.Sub(t1)
				copyOut += t3.Sub(t2)
				total += t3.Sub(t0)
			}
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(copyOut.Microseconds())/steps, "copy_out_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	b.Run("bridge_async_overlap", func(b *testing.B) {
		const ioBytes = 4096
		handle := api.open(modelPath, "s", uintptr(ioBytes), uintptr(ioBytes))
		if handle == 0 {
			b.Skip("bridge client open failed")
		}
		defer api.close(handle)

		count := ioBytes / 4
		var copyIn, ane, copyOut, total time.Duration
		for i := 0; i < b.N; i++ {
			first := make([]float32, count)
			fillBridgeInput(first, fx.tokens[0])
			pending := runBridgeEvalAsync(api, handle, first, count)

			for step := 1; step < fx.steps; step++ {
				if _, err := fx.metalAttention(fx.tokens[step]); err != nil {
					b.Fatalf("metal attention overlap step=%d: %v", step, err)
				}
				prev := <-pending
				if prev.err != nil {
					b.Skipf("bridge async eval failed at step=%d: %v", step-1, prev.err)
					return
				}

				copyIn += prev.copyIn
				ane += prev.ane
				copyOut += prev.copyOut
				total += prev.total

				next := make([]float32, count)
				fillBridgeInput(next, fx.tokens[step])
				pending = runBridgeEvalAsync(api, handle, next, count)
			}

			last := <-pending
			if last.err != nil {
				b.Skipf("bridge async eval failed at final step: %v", last.err)
				return
			}
			copyIn += last.copyIn
			ane += last.ane
			copyOut += last.copyOut
			total += last.total
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(copyIn.Microseconds())/steps, "copy_in_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(copyOut.Microseconds())/steps, "copy_out_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "total_us/step")
	})

	b.Run("bridge_bidirectional_roundtrip", func(b *testing.B) {
		if api.evalBidirectional == nil {
			b.Skip("bridge symbol ane_bridge_eval_bidirectional unavailable")
		}
		if api.signalEventCPU == nil || api.waitEventCPU == nil {
			b.Skip("bridge CPU event APIs unavailable")
		}
		if api.createSharedEvent == nil || api.sharedEventPort == nil || api.releaseObjc == nil {
			b.Skip("bridge shared-event helpers unavailable")
		}

		const ioBytes = 4096
		handle := api.open(modelPath, "s", uintptr(ioBytes), uintptr(ioBytes))
		if handle == 0 {
			b.Skip("bridge client open failed")
		}
		defer api.close(handle)

		waitObj := api.createSharedEvent()
		if waitObj == 0 {
			b.Skip("bridge create wait shared event failed")
		}
		defer api.releaseObjc(waitObj)
		signalObj := api.createSharedEvent()
		if signalObj == 0 {
			b.Skip("bridge create signal shared event failed")
		}
		defer api.releaseObjc(signalObj)
		waitPort := api.sharedEventPort(waitObj)
		signalPort := api.sharedEventPort(signalObj)
		if waitPort == 0 || signalPort == 0 {
			b.Skipf("invalid event ports wait=%d signal=%d", waitPort, signalPort)
		}

		count := ioBytes / 4
		in := make([]float32, count)
		out := make([]float32, count)

		var cpuSignal, ane, cpuWait, total time.Duration
		for i := 0; i < b.N; i++ {
			for step := 0; step < fx.steps; step++ {
				fillBridgeInput(in, fx.tokens[step])
				v := uint64(i*fx.steps + step + 1)

				t0 := time.Now()
				if rc := api.signalEventCPU(waitPort, v); rc != 0 {
					b.Skipf("CPU signal wait-event failed rc=%d", rc)
					return
				}
				t1 := time.Now()

				rc := api.evalBidirectional(
					handle,
					unsafe.Pointer(&in[0]), uint32(len(in)),
					unsafe.Pointer(&out[0]), uint32(len(out)),
					waitPort, v,
					signalPort, v,
				)
				t2 := time.Now()
				if rc != 0 {
					b.Skipf("bridge bidirectional eval failed rc=%d step=%d", rc, step)
					return
				}
				if rc := api.waitEventCPU(signalPort, v, 5000); rc != 0 {
					b.Skipf("CPU wait on signal event failed rc=%d", rc)
					return
				}
				t3 := time.Now()

				cpuSignal += t1.Sub(t0)
				ane += t2.Sub(t1)
				cpuWait += t3.Sub(t2)
				total += t3.Sub(t0)
			}
		}
		steps := float64(b.N * fx.steps)
		b.ReportMetric(float64(cpuSignal.Microseconds())/steps, "cpu_signal_us/step")
		b.ReportMetric(float64(cpuSignal.Microseconds())/steps, "metal_signal_us/step")
		b.ReportMetric(float64(ane.Microseconds())/steps, "ane_eval_us/step")
		b.ReportMetric(float64(cpuWait.Microseconds())/steps, "cpu_wait_us/step")
		b.ReportMetric(float64(cpuWait.Microseconds())/steps, "metal_wait_us/step")
		b.ReportMetric(float64(total.Microseconds())/steps, "roundtrip_us/step")
	})

	b.Run("bridge_copy_scaling", func(b *testing.B) {
		cases := []struct {
			name  string
			bytes int
		}{
			{name: "4KB", bytes: 4 * 1024},
			{name: "64KB", bytes: 64 * 1024},
			{name: "256KB", bytes: 256 * 1024},
			{name: "1MB", bytes: 1024 * 1024},
		}
		for _, tc := range cases {
			tc := tc
			b.Run(tc.name, func(b *testing.B) {
				handle := api.open(modelPath, "s", uintptr(tc.bytes), uintptr(tc.bytes))
				if handle == 0 {
					b.Skipf("bridge client open failed for size=%d", tc.bytes)
				}
				defer api.close(handle)

				count := tc.bytes / 4
				in := make([]float32, count)
				out := make([]float32, count)
				for i := range in {
					in[i] = float32(i%31) * 0.01
				}

				var copyIn, ane, copyOut, total time.Duration
				for i := 0; i < b.N; i++ {
					t0 := time.Now()
					api.writeInput(handle, unsafe.Pointer(&in[0]), int32(len(in)))
					t1 := time.Now()
					if !api.eval(handle) {
						b.Skipf("bridge eval failed for size=%d at i=%d", tc.bytes, i)
						return
					}
					t2 := time.Now()
					api.readOutput(handle, unsafe.Pointer(&out[0]), int32(len(out)))
					t3 := time.Now()

					copyIn += t1.Sub(t0)
					ane += t2.Sub(t1)
					copyOut += t3.Sub(t2)
					total += t3.Sub(t0)
				}
				n := float64(b.N)
				b.ReportMetric(float64(copyIn.Microseconds())/n, "copy_in_us/op")
				b.ReportMetric(float64(ane.Microseconds())/n, "ane_eval_us/op")
				b.ReportMetric(float64(copyOut.Microseconds())/n, "copy_out_us/op")
				b.ReportMetric(float64(total.Microseconds())/n, "total_us/op")
			})
		}
	})
}

var decodeBenchmarkSink float32

type bridgeAPI struct {
	open       func(string, string, uintptr, uintptr) uintptr
	close      func(uintptr)
	writeInput func(uintptr, unsafe.Pointer, int32)
	readOutput func(uintptr, unsafe.Pointer, int32)
	eval       func(uintptr) bool

	evalWithSignalEvent func(
		uintptr,
		unsafe.Pointer, uint32,
		unsafe.Pointer, uint32,
		uint32, uint64,
	) int32
	evalBidirectional func(
		uintptr,
		unsafe.Pointer, uint32,
		unsafe.Pointer, uint32,
		uint32, uint64,
		uint32, uint64,
	) int32
	signalEventCPU    func(uint32, uint64) int32
	waitEventCPU      func(uint32, uint64, uint32) int32
	createSharedEvent func() uintptr
	sharedEventPort   func(uintptr) uint32
	releaseObjc       func(uintptr)
}

type bridgeEvalResult struct {
	copyIn  time.Duration
	ane     time.Duration
	copyOut time.Duration
	total   time.Duration
	err     error
}

func loadBridgeAPI() (*bridgeAPI, error) {
	rt, err := loadANEBridgeRuntime()
	if err != nil {
		return nil, err
	}
	return &bridgeAPI{
		open:                rt.open,
		close:               rt.close,
		writeInput:          rt.writeInput,
		readOutput:          rt.readOutput,
		eval:                rt.eval,
		evalWithSignalEvent: rt.evalWithSignalEvent,
		evalBidirectional:   rt.evalBidirectional,
		signalEventCPU:      rt.signalEventCPU,
		waitEventCPU:        rt.waitEventCPU,
		createSharedEvent:   rt.createSharedEvent,
		sharedEventPort:     rt.sharedEventPort,
		releaseObjc:         rt.releaseObjc,
	}, nil
}

func fillBridgeInput(dst []float32, src []float32) {
	if len(dst) == 0 {
		return
	}
	if len(src) == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	for i := range dst {
		dst[i] = src[i%len(src)]
	}
}

func runBridgeEvalAsync(api *bridgeAPI, handle uintptr, input []float32, count int) <-chan bridgeEvalResult {
	ch := make(chan bridgeEvalResult, 1)
	go func() {
		if len(input) == 0 || count <= 0 {
			ch <- bridgeEvalResult{err: fmt.Errorf("bridge async eval: empty input")}
			close(ch)
			return
		}
		out := make([]float32, count)
		t0 := time.Now()
		api.writeInput(handle, unsafe.Pointer(&input[0]), int32(len(input)))
		t1 := time.Now()
		if !api.eval(handle) {
			ch <- bridgeEvalResult{err: fmt.Errorf("bridge eval returned false")}
			close(ch)
			return
		}
		t2 := time.Now()
		api.readOutput(handle, unsafe.Pointer(&out[0]), int32(len(out)))
		t3 := time.Now()
		ch <- bridgeEvalResult{
			copyIn:  t1.Sub(t0),
			ane:     t2.Sub(t1),
			copyOut: t3.Sub(t2),
			total:   t3.Sub(t0),
		}
		close(ch)
	}()
	return ch
}

func envInt(key string, dflt int) int {
	v := os.Getenv(key)
	if v == "" {
		return dflt
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return dflt
	}
	return n
}

func weightOutInToInOut(weight []float32, outDim, inDim int) (*mlx.Array, error) {
	w, err := mlx.FromSlice(weight, []int{outDim, inDim}, mlx.Float32)
	if err != nil {
		return nil, fmt.Errorf("weight setup from slice: %w", err)
	}
	wt, err := mlx.Transpose(w, nil)
	w.Free()
	if err != nil {
		return nil, fmt.Errorf("weight setup transpose: %w", err)
	}
	return wt, nil
}

func (f *decodeBenchFixture) metalAttention(token []float32) ([]float32, error) {
	x, err := mlx.FromSlice(token, []int{1, f.seq, f.dim}, mlx.Float32)
	if err != nil {
		return nil, fmt.Errorf("attention input from slice: %w", err)
	}
	defer x.Free()
	out, err := mlx.Matmul(x, f.attnW, nil)
	if err != nil {
		return nil, fmt.Errorf("attention matmul: %w", err)
	}
	defer out.Free()
	v, err := mlx.ToSlice[float32](out)
	if err != nil {
		return nil, fmt.Errorf("attention to slice: %w", err)
	}
	return v, nil
}

func (f *decodeBenchFixture) metalFFN(att []float32) ([]float32, error) {
	x, err := mlx.FromSlice(att, []int{1, f.seq, f.dim}, mlx.Float32)
	if err != nil {
		return nil, fmt.Errorf("ffn input from slice: %w", err)
	}
	defer x.Free()
	xn, err := mlx.Multiply(x, f.rms2, nil)
	if err != nil {
		return nil, fmt.Errorf("ffn rms2 multiply: %w", err)
	}
	defer xn.Free()

	gate, err := mlx.Matmul(xn, f.w1, nil)
	if err != nil {
		return nil, fmt.Errorf("ffn gate matmul: %w", err)
	}
	defer gate.Free()
	up, err := mlx.Matmul(xn, f.w3, nil)
	if err != nil {
		return nil, fmt.Errorf("ffn up matmul: %w", err)
	}
	defer up.Free()

	act := nn.SiLU(gate)
	defer act.Free()
	inter, err := mlx.Multiply(act, up, nil)
	if err != nil {
		return nil, fmt.Errorf("ffn swiglu multiply: %w", err)
	}
	defer inter.Free()
	y, err := mlx.Matmul(inter, f.w2, nil)
	if err != nil {
		return nil, fmt.Errorf("ffn down matmul: %w", err)
	}
	defer y.Free()

	v, err := mlx.ToSlice[float32](y)
	if err != nil {
		return nil, fmt.Errorf("ffn output to slice: %w", err)
	}
	return v, nil
}
