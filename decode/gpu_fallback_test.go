//go:build darwin

package decode

import (
	"testing"

	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
)

type gpuFallbackFakeModel struct {
	cfg *models.ModelConfig

	baseForwardCalls int
	embedCalls       int
	attnCalls        int
	mlpCalls         int

	inputNorm *mlx.Array
	postNorm  *mlx.Array
	finalNorm *mlx.Array
	lmHead    *mlx.Array
	embedding *mlx.Array
}

func newGPUFallbackFakeModel(t *testing.T) *gpuFallbackFakeModel {
	t.Helper()
	model := &gpuFallbackFakeModel{
		cfg: &models.ModelConfig{
			ModelType:         "fake",
			VocabSize:         4,
			HiddenSize:        2,
			NumLayers:         1,
			RMSNormEps:        1e-5,
			TieWordEmbeddings: false,
		},
	}
	model.inputNorm = mustFloat32Array(t, []float32{1, 1}, []int{2})
	model.postNorm = mustFloat32Array(t, []float32{1, 1}, []int{2})
	model.finalNorm = mustFloat32Array(t, []float32{1, 1}, []int{2})
	model.lmHead = mustFloat32Array(t, []float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 1,
	}, []int{4, 2})
	model.embedding = mustFloat32Array(t, []float32{2, 1}, []int{1, 1, 2})
	return model
}

func (m *gpuFallbackFakeModel) Forward(inputs *mlx.Array, cache models.Cache) (*mlx.Array, models.Cache, error) {
	m.baseForwardCalls++
	shape := inputs.Shape()
	seqLen := 1
	if len(shape) >= 2 {
		seqLen = shape[1]
	}
	logits := make([]float32, seqLen*m.cfg.VocabSize)
	for i := range logits {
		logits[i] = 9
	}
	out, err := mlx.FromSlice(logits, []int{1, seqLen, m.cfg.VocabSize}, mlx.Float32)
	if err != nil {
		return nil, cache, err
	}
	return out, cache, nil
}

func (m *gpuFallbackFakeModel) Config() *models.ModelConfig {
	return m.cfg
}

func (m *gpuFallbackFakeModel) Sanitize(weights map[string]*mlx.Array) map[string]*mlx.Array {
	return weights
}

func (m *gpuFallbackFakeModel) LoadWeights(weightFiles ...string) error {
	return nil
}

func (m *gpuFallbackFakeModel) LayerAttentionForward(i int, x *mlx.Array, mask any, cache kvcache.Cache) (*mlx.Array, error) {
	m.attnCalls++
	return mlx.Zeros(x.Shape(), x.Dtype(), nil)
}

func (m *gpuFallbackFakeModel) LayerInputNormWeight(i int) *mlx.Array {
	return m.inputNorm
}

func (m *gpuFallbackFakeModel) LayerPostNormWeight(i int) *mlx.Array {
	return m.postNorm
}

func (m *gpuFallbackFakeModel) LayerFFNWeights(i int) (gate, up, down []float32, err error) {
	return nil, nil, nil, nil
}

func (m *gpuFallbackFakeModel) FinalNormWeight() *mlx.Array {
	return m.finalNorm
}

func (m *gpuFallbackFakeModel) LMHeadWeight() *mlx.Array {
	return m.lmHead
}

func (m *gpuFallbackFakeModel) EmbedTokens(ids *mlx.Array) (*mlx.Array, error) {
	m.embedCalls++
	return m.embedding, nil
}

func (m *gpuFallbackFakeModel) LayerMLPForward(layerIdx int, normalized *mlx.Array) (*mlx.Array, error) {
	m.mlpCalls++
	return mlx.Zeros(normalized.Shape(), normalized.Dtype(), nil)
}

func TestWrapGPUFallbackInterceptsSingleTokenDecode(t *testing.T) {
	model := newGPUFallbackFakeModel(t)
	plane, err := Wrap(model, Options{Mode: "gpu_fallback"})
	if err != nil {
		t.Fatalf("Wrap: %v", err)
	}

	input := mustInt32Array(t, []int32{1}, []int{1, 1})
	cache := newTestCache(t, model.cfg.NumLayers)

	logits, _, err := plane.Forward(input, cache)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if err := mlx.Eval(logits); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if got, want := logits.Shape(), []int{1, 1, model.cfg.VocabSize}; !equalInts(got, want) {
		t.Fatalf("logits shape = %v, want %v", got, want)
	}
	if model.baseForwardCalls != 0 {
		t.Fatalf("baseForwardCalls = %d, want 0", model.baseForwardCalls)
	}
	if model.embedCalls != 1 || model.attnCalls != 1 || model.mlpCalls != 1 {
		t.Fatalf("embed=%d attn=%d mlp=%d, want 1/1/1", model.embedCalls, model.attnCalls, model.mlpCalls)
	}
}

func TestWrapGPUFallbackLeavesPrefillOnBaseModel(t *testing.T) {
	model := newGPUFallbackFakeModel(t)
	plane, err := Wrap(model, Options{Mode: "gpu_fallback"})
	if err != nil {
		t.Fatalf("Wrap: %v", err)
	}

	input := mustInt32Array(t, []int32{1, 2}, []int{1, 2})
	cache := newTestCache(t, model.cfg.NumLayers)

	logits, _, err := plane.Forward(input, cache)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if err := mlx.Eval(logits); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if model.baseForwardCalls != 1 {
		t.Fatalf("baseForwardCalls = %d, want 1", model.baseForwardCalls)
	}
	if model.embedCalls != 0 || model.attnCalls != 0 || model.mlpCalls != 0 {
		t.Fatalf("embed=%d attn=%d mlp=%d, want 0/0/0", model.embedCalls, model.attnCalls, model.mlpCalls)
	}
}

func mustFloat32Array(t *testing.T, data []float32, shape []int) *mlx.Array {
	t.Helper()
	arr, err := mlx.FromSlice(data, shape, mlx.Float32)
	if err != nil {
		t.Fatalf("FromSlice(%v, %v): %v", data, shape, err)
	}
	return arr
}

func mustInt32Array(t *testing.T, data []int32, shape []int) *mlx.Array {
	t.Helper()
	arr, err := mlx.FromSlice(data, shape, mlx.Int32)
	if err != nil {
		t.Fatalf("FromSlice(%v, %v): %v", data, shape, err)
	}
	return arr
}

func newTestCache(t *testing.T, numLayers int) models.Cache {
	t.Helper()
	caches := make([]kvcache.Cache, numLayers)
	for i := range caches {
		c, err := kvcache.New(kvcache.DefaultConfig())
		if err != nil {
			t.Fatalf("kvcache.New: %v", err)
		}
		caches[i] = c
	}
	return models.NewMultiLayerCacheFromList(caches)
}

func equalInts(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
