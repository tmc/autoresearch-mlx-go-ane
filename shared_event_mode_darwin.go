//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"os"
	"strings"
)

const sharedEventModeEnv = "MLXGO_ANE_METAL_SYNC"

func sharedEventModeFromEnv() string {
	return strings.ToLower(strings.TrimSpace(os.Getenv(sharedEventModeEnv)))
}

func applySharedEventModeFromEnv(cfg *MultiSurfaceEvalPlanConfig) {
	if cfg == nil {
		return
	}
	switch sharedEventModeFromEnv() {
	case "", "off", "0", "false", "none":
		return
	case "wait", "metal_to_ane":
		cfg.EnableMetalWait = true
	case "signal", "ane_to_metal":
		cfg.EnableMetalSignal = true
	case "wait_signal", "signal_wait", "both", "full":
		cfg.EnableMetalWait = true
		cfg.EnableMetalSignal = true
	}
}

func applySurfaceSharedEventModeFromEnv(cfg *SurfaceEvalPlanConfig) {
	if cfg == nil {
		return
	}
	switch sharedEventModeFromEnv() {
	case "", "off", "0", "false", "none":
		return
	case "wait", "metal_to_ane":
		cfg.EnableMetalWait = true
	case "signal", "ane_to_metal":
		cfg.EnableMetalSignal = true
	case "wait_signal", "signal_wait", "both", "full":
		cfg.EnableMetalWait = true
		cfg.EnableMetalSignal = true
	}
}
