package mlxgoane

import (
	"context"
	"fmt"

	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
)

func modelForward(ctx context.Context, model models.LanguageModel, input *mlx.Array, cache models.Cache) (*mlx.Array, models.Cache, error) {
	if model == nil {
		return nil, cache, fmt.Errorf("model is nil")
	}
	if fwd, ok := model.(interface {
		Forward(inputs *mlx.Array, cache models.Cache) (*mlx.Array, models.Cache, error)
	}); ok {
		return fwd.Forward(input, cache)
	}
	if fwd, ok := model.(interface {
		Forward(ctx context.Context, inputs *mlx.Array, cache models.Cache) (*mlx.Array, models.Cache)
	}); ok {
		out, next := fwd.Forward(ctx, input, cache)
		return out, next, nil
	}
	return nil, cache, fmt.Errorf("model %T does not implement a supported forward interface", model)
}
