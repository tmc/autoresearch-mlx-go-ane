//go:build darwin

package decode

import (
	"os"
	"strconv"
	"strings"
)

// outputMode controls how ANE stage outputs are materialized for GPU consumption.
type outputMode string

const (
	outputModeCopy           outputMode = "copy"
	outputModeIOSurfaceAlias outputMode = "iosurface_alias"
	outputModeGPUMaterialize outputMode = "gpu_materialize_f32"
	outputModeGPUNative      outputMode = "gpu_materialize_native"
)

// consumerMode selects how GPU-side kernels consume ANE outputs.
type consumerMode string

const (
	consumerMLX               consumerMode = "mlx"
	consumerMetalResidualAdd  consumerMode = "metal_residual_add"
	consumerMetalResidualNorm consumerMode = "metal_residual_norm"
	consumerMetalMoECombine   consumerMode = "metal_moe_combine"
	consumerMetalBlockBound   consumerMode = "metal_block_boundary"
)

// waitMode controls synchronization between ANE completion and GPU readiness.
type waitMode string

const (
	waitModeCPUSync waitMode = "cpu_sync"
	waitModeGPUWait waitMode = "gpu_wait"
)

// experimentConfig holds tunable knobs populated from environment variables.
type experimentConfig struct {
	OutputMode      outputMode
	ConsumerMode    consumerMode
	PoolDepth       int
	WaitMode        waitMode
	CompiledPrepare bool
	DirectBlock     bool
}

const defaultOutputPoolDepth = 2

func compiledPrepareEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MLXGO_ANE_QWEN35_COMPILED_PREPARE"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func directBlockEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MLXGO_ANE_QWEN35_DIRECT_BLOCK"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func experimentOutputMode() outputMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MLXGO_ANE_QWEN35_OUTPUT_MODE"))) {
	case "", string(outputModeCopy):
		return outputModeCopy
	case string(outputModeIOSurfaceAlias):
		return outputModeIOSurfaceAlias
	case string(outputModeGPUMaterialize):
		return outputModeGPUMaterialize
	case string(outputModeGPUNative):
		return outputModeGPUNative
	default:
		return outputModeCopy
	}
}

func experimentConsumerMode() consumerMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MLXGO_ANE_QWEN35_CONSUMER_MODE"))) {
	case "", string(consumerMLX):
		return consumerMLX
	case string(consumerMetalResidualAdd):
		return consumerMetalResidualAdd
	case string(consumerMetalResidualNorm):
		return consumerMetalResidualNorm
	case string(consumerMetalMoECombine):
		return consumerMetalMoECombine
	case string(consumerMetalBlockBound):
		return consumerMetalBlockBound
	default:
		return consumerMLX
	}
}

func experimentWaitMode() waitMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MLXGO_ANE_QWEN35_WAIT_MODE"))) {
	case "", string(waitModeCPUSync):
		return waitModeCPUSync
	case string(waitModeGPUWait):
		return waitModeGPUWait
	default:
		return waitModeCPUSync
	}
}

func experimentPoolDepth() int {
	raw := strings.TrimSpace(os.Getenv("MLXGO_ANE_QWEN35_OUTPUT_POOL_DEPTH"))
	if raw == "" {
		return defaultOutputPoolDepth
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return defaultOutputPoolDepth
	}
	return n
}

func experimentConfigFromEnv() experimentConfig {
	return experimentConfig{
		OutputMode:      experimentOutputMode(),
		ConsumerMode:    experimentConsumerMode(),
		PoolDepth:       experimentPoolDepth(),
		WaitMode:        experimentWaitMode(),
		CompiledPrepare: compiledPrepareEnabled(),
		DirectBlock:     directBlockEnabled(),
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Options configures the decode-plane engine.
type Options struct {
	Mode      string
	ModelPath string
	CacheDir  string
	Warn      func(format string, args ...any)
}
