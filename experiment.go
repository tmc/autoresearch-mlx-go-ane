// experiment.go — THE MODIFIABLE FILE
//
// The autonomous agent edits this file to explore inference configurations,
// sampling parameters, and generation strategies. All other harness files
// are read-only.

package mlxgoane

const (
	// DefaultModel is the HuggingFace model ID or local path to use.
	DefaultModel = "mlx-community/Qwen3.5-4B-4bit"

	// DefaultPrompt is the prompt used for benchmarking.
	DefaultPrompt = "Explain the theory of relativity in simple terms."

	// GenerateTokens is the number of tokens to generate per benchmark iteration.
	GenerateTokens = 100

	// Sampling parameters.
	Temperature = 0.0
	TopP        = 1.0
	MinP        = 0.0
	TopK        = 0

	// WarmupEnabled controls whether a warmup pass runs before benchmarking.
	WarmupEnabled = true

	// CacheType selects the KV cache strategy: "default", "inplace", "rotating", "prealloc".
	CacheType = "default"

	// ANEDecodePlaneMode selects the ANE decode plane mode: "qwen35", "off".
	ANEDecodePlaneMode = "qwen35"

	// UseChatTemplate controls whether prompts are wrapped in chat template.
	UseChatTemplate = true

	// Seed for random number generation (0 = no explicit seed).
	Seed = 0
)
