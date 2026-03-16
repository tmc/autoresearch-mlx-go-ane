//go:build darwin && ane_appleneuralengine

package anedraftimpl

import (
	anedraft "github.com/tmc/mlx-go-lm/anedraft"
	"github.com/tmc/mlx-go-lm/mlxlm/decode"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
)

func init() {
	anedraft.RegisterBackend(backend{})
}

type backend struct{}

func (backend) NewDrafter(modelcPath string, draftModel models.LanguageModel, options anedraft.Options) (decode.Drafter, func(), error) {
	return NewANEDrafter(modelcPath, draftModel, ANEDraftOptions(options))
}

func (backend) WrapSSDDrafter(d decode.Drafter) (decode.Drafter, error) {
	return NewANESSDDrafterAdapter(d)
}

func (backend) RuntimeActive(d decode.Drafter) bool {
	return RuntimeActive(d)
}
