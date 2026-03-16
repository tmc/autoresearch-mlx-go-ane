//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/tmc/apple/private/appleneuralengine"
)

type modelStateSnapshot struct {
	InMemoryState              uint64
	ModelState                 uint64
	InMemoryProgramHandle      uint64
	ModelProgramHandle         uint64
	InMemoryIntermediateHandle uint64
	ModelIntermediateHandle    uint64
	InMemoryQueueDepth         int8
	ModelQueueDepth            int8
}

func snapshotModelState(model appleneuralengine.ANEInMemoryModel) modelStateSnapshot {
	snap := modelStateSnapshot{}
	if model.ID == 0 {
		return snap
	}
	snap.InMemoryState = model.State()
	snap.InMemoryProgramHandle = model.ProgramHandle()
	snap.InMemoryIntermediateHandle = model.IntermediateBufferHandle()
	snap.InMemoryQueueDepth = model.QueueDepth()
	base := model.Model()
	if base.GetID() == 0 {
		return snap
	}
	aneModel := appleneuralengine.ANEModelFromID(base.GetID())
	snap.ModelState = aneModel.State()
	snap.ModelProgramHandle = aneModel.ProgramHandle()
	snap.ModelIntermediateHandle = aneModel.IntermediateBufferHandle()
	snap.ModelQueueDepth = aneModel.QueueDepth()
	return snap
}

func applyModelStateSyncFromEnv(label string, model appleneuralengine.ANEInMemoryModel) error {
	mode := strings.TrimSpace(os.Getenv("MLXGO_ANE_STATE_HANDLE_SYNC"))
	if mode == "" {
		return nil
	}
	before := snapshotModelState(model)
	if err := applyModelStateSync(mode, model); err != nil {
		return fmt.Errorf("%s: %w", label, err)
	}
	if envTruthy("MLXGO_ANE_DEBUG_MODEL_STATE") {
		after := snapshotModelState(model)
		slog.Info(
			"ANE model state sync",
			"label", label,
			"mode", mode,
			"before", before,
			"after", after,
		)
	}
	return nil
}

func applyModelStateSync(mode string, model appleneuralengine.ANEInMemoryModel) error {
	if model.ID == 0 {
		return fmt.Errorf("model state sync: in-memory model is nil")
	}
	base := model.Model()
	if base.GetID() == 0 {
		return fmt.Errorf("model state sync: underlying ane model is nil")
	}
	aneModel := appleneuralengine.ANEModelFromID(base.GetID())
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "copy_model_to_inmem":
		state := aneModel.State()
		if state == 0 {
			return fmt.Errorf("model state sync: underlying ane model state is zero")
		}
		model.SetState(state)
		return nil
	case "copy_inmem_to_model":
		state := model.State()
		if state == 0 {
			return fmt.Errorf("model state sync: in-memory model state is zero")
		}
		aneModel.SetState(state)
		return nil
	case "refresh_model_attrs":
		state := chooseNonZero(model.State(), aneModel.State())
		if state == 0 {
			return fmt.Errorf("model state sync: no state handle available for attribute refresh")
		}
		attrs := aneModel.ModelAttributes()
		if attrs.GetID() == 0 {
			return fmt.Errorf("model state sync: model attributes are nil")
		}
		aneModel.UpdateModelAttributesStateProgramHandleIntermediateBufferHandleQueueDepth(
			attrs,
			state,
			chooseNonZero(aneModel.ProgramHandle(), model.ProgramHandle()),
			chooseNonZero(aneModel.IntermediateBufferHandle(), model.IntermediateBufferHandle()),
			chooseNonZeroInt8(aneModel.QueueDepth(), model.QueueDepth()),
		)
		return nil
	case "copy_and_refresh":
		if err := applyModelStateSync("copy_model_to_inmem", model); err != nil {
			return err
		}
		return applyModelStateSync("refresh_model_attrs", model)
	default:
		return fmt.Errorf("model state sync: unsupported mode %q", mode)
	}
}

func chooseNonZero(a, b uint64) uint64 {
	if a != 0 {
		return a
	}
	return b
}

func chooseNonZeroInt8(a, b int8) int8 {
	if a != 0 {
		return a
	}
	return b
}
