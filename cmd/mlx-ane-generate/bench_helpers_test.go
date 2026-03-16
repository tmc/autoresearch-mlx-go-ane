package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
)

type speculativeStatsJSON struct {
	Mode                   string  `json:"mode"`
	DraftBackend           string  `json:"draft_backend"`
	DraftModelKind         string  `json:"draft_model_kind"`
	MaxLayers              int     `json:"max_layers"`
	DirectModelIR          bool    `json:"direct_modelir"`
	DirectCompileFallback  bool    `json:"direct_compile_fallback"`
	DirectCompileProfile   string  `json:"direct_compile_profile"`
	DirectCompileTier      string  `json:"direct_compile_tier"`
	DirectCompileNote      string  `json:"direct_compile_note"`
	DirectQwenOnANE        string  `json:"direct_qwen_on_ane"`
	DirectQwenOnMLX        string  `json:"direct_qwen_on_mlx"`
	PrefillMS              float64 `json:"ane_prefill_ms"`
	DraftCalls             int     `json:"ane_draft_calls"`
	DraftMS                float64 `json:"ane_draft_ms"`
	EnsureTokenCalls       int     `json:"ane_ensure_token_calls"`
	EnsureTokenMS          float64 `json:"ane_ensure_token_ms"`
	FeedTokenCalls         int     `json:"ane_feed_token_calls"`
	FeedTokenMS            float64 `json:"ane_feed_token_ms"`
	EmbeddingLookups       int     `json:"ane_embedding_lookups"`
	EmbeddingMS            float64 `json:"ane_embedding_ms"`
	ModelEvalCalls         int     `json:"ane_model_eval_calls"`
	ModelEvalMS            float64 `json:"ane_model_eval_ms"`
	NextTokenCalls         int     `json:"ane_next_token_calls"`
	NextTokenMS            float64 `json:"ane_next_token_ms"`
	AdvanceDecodeCalls     int     `json:"ane_advance_decode_calls"`
	AdvanceDecodeMS        float64 `json:"ane_advance_decode_ms"`
	RewindCalls            int     `json:"ane_rewind_calls"`
	RewindMS               float64 `json:"ane_rewind_ms"`
	RebuildCount           int     `json:"ane_rebuild_count"`
	RebuildMS              float64 `json:"ane_rebuild_ms"`
	TotalTokens            int     `json:"total_tokens"`
	DraftAccepted          int     `json:"draft_accepted"`
	AcceptRate             float64 `json:"accept_rate"`
	AcceptedTokensPerSec   float64 `json:"accepted_tokens_per_sec"`
	GenerationTokensPerSec float64 `json:"generation_tokens_per_sec"`
	PrefillTokensPerSec    float64 `json:"prefill_tokens_per_sec"`
	InitMS                 float64 `json:"init_ms"`
	ReferenceOnly          bool    `json:"reference_only"`
	GuardTriggered         bool    `json:"guard_triggered"`
	FFNFallbackUsed        bool    `json:"ffn_fallback_used"`
}

func parseGoTPS(out string) float64 {
	re := regexp.MustCompile(`Generation: \d+ tokens, ([\d\.]+) tokens-per-sec`)
	matches := re.FindStringSubmatch(out)
	if len(matches) > 1 {
		val, _ := strconv.ParseFloat(matches[1], 64)
		return val
	}
	return 0
}

func parseGoPrefillTPS(out string) float64 {
	re := regexp.MustCompile(`Prefill: \d+ tokens, ([\d\.]+) tokens-per-sec`)
	matches := re.FindStringSubmatch(out)
	if len(matches) > 1 {
		val, _ := strconv.ParseFloat(matches[1], 64)
		return val
	}
	return 0
}

type qwenFamilyBenchTarget struct {
	Name       string
	ModelPath  string
	RequireMoE bool
}

func qwenFamilyDenseTargets() []qwenFamilyBenchTarget {
	return append([]qwenFamilyBenchTarget(nil), qwenFamilyTargets(false)...)
}

func qwenFamilyAllTargets() []qwenFamilyBenchTarget {
	return append([]qwenFamilyBenchTarget(nil), qwenFamilyTargets(true)...)
}

func qwenFamilyDraftModelPath(target qwenFamilyBenchTarget) string {
	if p := envOrSnapshot("QWEN35_DRAFT_MODEL_PATH", ""); p != "" {
		return p
	}
	if p := os.Getenv("MLX_TEST_DRAFT_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return target.ModelPath
}

func qwenFamilyTargets(includeMoE bool) []qwenFamilyBenchTarget {
	targets := []qwenFamilyBenchTarget{
		{Name: "qwen35_08b", ModelPath: envOrSnapshot("QWEN35_08B_MODEL_PATH", "models--mlx-community--Qwen3.5-0.8B-4bit"), RequireMoE: false},
		{Name: "qwen35_4b", ModelPath: envOrSnapshot("QWEN35_4B_MODEL_PATH", "models--mlx-community--Qwen3.5-4B-4bit"), RequireMoE: false},
	}
	if includeMoE {
		targets = append(targets, qwenFamilyBenchTarget{
			Name:       "qwen35_35b_a3b",
			ModelPath:  envOrSnapshot("QWEN35_MOE_MODEL_PATH", "models--mlx-community--Qwen3.5-35B-A3B-4bit"),
			RequireMoE: true,
		})
	}
	filtered := targets[:0]
	for _, target := range targets {
		if target.ModelPath == "" {
			continue
		}
		filtered = append(filtered, target)
	}
	return filtered
}

func envOrSnapshot(envKey, snapshotDir string) string {
	if p := os.Getenv(envKey); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		return ""
	}
	if snapshotDir == "" {
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	base := filepath.Join(home, ".cache", "huggingface", "hub", snapshotDir, "snapshots")
	entries, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		return filepath.Join(base, entry.Name())
	}
	return ""
}
