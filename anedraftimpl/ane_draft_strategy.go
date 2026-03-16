
package anedraftimpl

import (
	"fmt"
	"os"
	"strings"

	"github.com/tmc/mlx-go-lm/mlxlm/models"
)

type DraftStrategy int

const (
	DraftStrategyFullMIL DraftStrategy = iota
	DraftStrategyPartial4
	DraftStrategyPartial2
	DraftStrategyPartial1
	DraftStrategyReferenceOnly
	DraftStrategyFFNFallback
)

func (s DraftStrategy) String() string {
	switch s {
	case DraftStrategyFullMIL:
		return "full"
	case DraftStrategyPartial4:
		return "partial4"
	case DraftStrategyPartial2:
		return "partial2"
	case DraftStrategyPartial1:
		return "partial1"
	case DraftStrategyReferenceOnly:
		return "reference-only"
	case DraftStrategyFFNFallback:
		return "ffn"
	default:
		return "unknown"
	}
}

type DraftOutputMode int

const (
	DraftOutputModeAuto DraftOutputMode = iota
	DraftOutputModeLMHead
	DraftOutputModeFinalHidden
	DraftOutputModeLastFFNResidual
	DraftOutputModeLastAttnResidual
)

func (m DraftOutputMode) String() string {
	switch m {
	case DraftOutputModeAuto:
		return "auto"
	case DraftOutputModeLMHead:
		return "lm_head"
	case DraftOutputModeFinalHidden:
		return "final_hidden"
	case DraftOutputModeLastFFNResidual:
		return "last_ffn_residual"
	case DraftOutputModeLastAttnResidual:
		return "last_attn_residual"
	default:
		return "unknown"
	}
}

type DraftModelKind string

const (
	DraftModelKindUnknown          DraftModelKind = "unknown"
	DraftModelKindExternalModelC   DraftModelKind = "external_modelc"
	DraftModelKindLMHead           DraftModelKind = "lm_head"
	DraftModelKindFinalHidden      DraftModelKind = "final_hidden"
	DraftModelKindLastFFNResidual  DraftModelKind = "last_ffn_residual"
	DraftModelKindLastAttnResidual DraftModelKind = "last_attn_residual"
	DraftModelKindReferenceOnly    DraftModelKind = "reference_only"
	DraftModelKindFFNFallback      DraftModelKind = "ffn_fallback"
)

type ANEDraftPolicy struct {
	Mode              string
	OutputMode        DraftOutputMode
	Strategies        []DraftStrategy
	WidenedAttention  bool
	TotalLayers       int
	ExplicitReference bool
}

func ResolveANEDraftPolicy(cfg *models.ModelConfig) (ANEDraftPolicy, error) {
	policy := ANEDraftPolicy{
		Mode:       "auto",
		OutputMode: DraftOutputModeAuto,
	}
	if cfg != nil {
		policy.TotalLayers = cfg.NumLayers
		policy.WidenedAttention = IsWidenedAttentionDraft(cfg)
	}
	strategyMode := strings.ToLower(strings.TrimSpace(os.Getenv("MLXGO_ANE_DRAFT_STRATEGY")))
	if strategyMode == "" {
		strategyMode = "auto"
	}
	policy.Mode = strategyMode
	strategy, autoStrategy, err := ParseDraftStrategyMode(strategyMode)
	if err != nil {
		return ANEDraftPolicy{}, err
	}
	outputMode, err := ParseDraftOutputMode(strings.TrimSpace(os.Getenv("MLXGO_ANE_DRAFT_OUTPUT_MODE")))
	if err != nil {
		return ANEDraftPolicy{}, err
	}
	policy.OutputMode = outputMode
	if autoStrategy {
		policy.Strategies = AutoDraftStrategies(
			policy.WidenedAttention,
			!draftEnvTruthy("MLXGO_ANE_DRAFT_DISABLE_AUTO_REF_GUARD"),
		)
	} else {
		policy.Strategies = []DraftStrategy{strategy}
	}
	policy.ExplicitReference = len(policy.Strategies) == 1 && policy.Strategies[0] == DraftStrategyReferenceOnly
	return policy, nil
}

func ParseDraftStrategyMode(raw string) (DraftStrategy, bool, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "auto":
		return DraftStrategyFullMIL, true, nil
	case "full":
		return DraftStrategyFullMIL, false, nil
	case "partial4":
		return DraftStrategyPartial4, false, nil
	case "partial2":
		return DraftStrategyPartial2, false, nil
	case "partial1":
		return DraftStrategyPartial1, false, nil
	case "reference-only":
		return DraftStrategyReferenceOnly, false, nil
	case "ffn":
		return DraftStrategyFFNFallback, false, nil
	default:
		return DraftStrategyFullMIL, false, fmt.Errorf(
			"parse MLXGO_ANE_DRAFT_STRATEGY=%q: want auto, full, partial4, partial2, partial1, reference-only, or ffn",
			raw,
		)
	}
}

func ParseDraftOutputMode(raw string) (DraftOutputMode, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "auto":
		return DraftOutputModeAuto, nil
	case "lm_head":
		return DraftOutputModeLMHead, nil
	case "final_hidden":
		return DraftOutputModeFinalHidden, nil
	case "last_ffn_residual":
		return DraftOutputModeLastFFNResidual, nil
	case "last_attn_residual":
		return DraftOutputModeLastAttnResidual, nil
	default:
		return DraftOutputModeAuto, fmt.Errorf(
			"parse MLXGO_ANE_DRAFT_OUTPUT_MODE=%q: want auto, lm_head, final_hidden, last_ffn_residual, or last_attn_residual",
			raw,
		)
	}
}

func AutoDraftStrategies(widened, preferReferenceOnly bool) []DraftStrategy {
	if widened {
		if preferReferenceOnly {
			return []DraftStrategy{
				DraftStrategyReferenceOnly,
				DraftStrategyPartial1,
				DraftStrategyFFNFallback,
			}
		}
		return []DraftStrategy{
			DraftStrategyPartial1,
			DraftStrategyReferenceOnly,
			DraftStrategyFFNFallback,
		}
	}
	return []DraftStrategy{
		DraftStrategyFullMIL,
		DraftStrategyPartial4,
		DraftStrategyPartial2,
		DraftStrategyPartial1,
		DraftStrategyFFNFallback,
	}
}

func DraftOutputModeCandidates(strategy DraftStrategy, selected DraftOutputMode, widened bool) []DraftOutputMode {
	if selected != DraftOutputModeAuto {
		return []DraftOutputMode{selected}
	}
	if strategy == DraftStrategyPartial1 && widened {
		return []DraftOutputMode{
			DraftOutputModeLastAttnResidual,
			DraftOutputModeFinalHidden,
		}
	}
	switch strategy {
	case DraftStrategyReferenceOnly, DraftStrategyFFNFallback:
		return nil
	default:
		return []DraftOutputMode{
			DraftOutputModeLMHead,
			DraftOutputModeFinalHidden,
			DraftOutputModeLastFFNResidual,
		}
	}
}

func DraftStrategyMaxLayers(strategy DraftStrategy, totalLayers int) int {
	switch strategy {
	case DraftStrategyPartial4:
		return minPositive(totalLayers, 4)
	case DraftStrategyPartial2:
		return minPositive(totalLayers, 2)
	case DraftStrategyPartial1:
		return minPositive(totalLayers, 1)
	case DraftStrategyFullMIL:
		return totalLayers
	default:
		return 0
	}
}

func minPositive(a, b int) int {
	switch {
	case a <= 0:
		return b
	case b <= 0:
		return a
	case a < b:
		return a
	default:
		return b
	}
}

func IsWidenedAttentionDraft(cfg *models.ModelConfig) bool {
	if cfg == nil || cfg.NumHeads <= 0 || cfg.HeadDim <= 0 || cfg.HiddenSize <= 0 {
		return false
	}
	return cfg.HeadDim*cfg.NumHeads > cfg.HiddenSize
}

func DraftSelectionLowConfidence(strategy DraftStrategy, kind DraftModelKind, widened bool) bool {
	if strategy == DraftStrategyFFNFallback {
		return true
	}
	if strategy == DraftStrategyPartial1 && widened {
		return true
	}
	return kind != DraftModelKindLMHead && kind != DraftModelKindReferenceOnly && kind != DraftModelKindExternalModelC
}

func draftEnvTruthy(name string) bool {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	switch raw {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

type draftStrategy = DraftStrategy

const (
	draftStrategyFullMIL       = DraftStrategyFullMIL
	draftStrategyPartial4      = DraftStrategyPartial4
	draftStrategyPartial2      = DraftStrategyPartial2
	draftStrategyPartial1      = DraftStrategyPartial1
	draftStrategyReferenceOnly = DraftStrategyReferenceOnly
	draftStrategyFFNFallback   = DraftStrategyFFNFallback
)

type draftOutputMode = DraftOutputMode

const (
	draftOutputModeAuto             = DraftOutputModeAuto
	draftOutputModeLMHead           = DraftOutputModeLMHead
	draftOutputModeFinalHidden      = DraftOutputModeFinalHidden
	draftOutputModeLastFFNResidual  = DraftOutputModeLastFFNResidual
	draftOutputModeLastAttnResidual = DraftOutputModeLastAttnResidual
)

type draftModelKind = DraftModelKind

const (
	draftModelKindUnknown          = DraftModelKindUnknown
	draftModelKindExternalModelC   = DraftModelKindExternalModelC
	draftModelKindLMHead           = DraftModelKindLMHead
	draftModelKindFinalHidden      = DraftModelKindFinalHidden
	draftModelKindLastFFNResidual  = DraftModelKindLastFFNResidual
	draftModelKindLastAttnResidual = DraftModelKindLastAttnResidual
	draftModelKindReferenceOnly    = DraftModelKindReferenceOnly
	draftModelKindFFNFallback      = DraftModelKindFFNFallback
)

type aneDraftPolicy = ANEDraftPolicy

func resolveANEDraftPolicy(cfg *models.ModelConfig) (aneDraftPolicy, error) {
	return ResolveANEDraftPolicy(cfg)
}

func parseDraftStrategyMode(raw string) (draftStrategy, bool, error) {
	return ParseDraftStrategyMode(raw)
}

func parseDraftOutputMode(raw string) (draftOutputMode, error) {
	return ParseDraftOutputMode(raw)
}

func autoDraftStrategies(widened, preferReferenceOnly bool) []draftStrategy {
	return AutoDraftStrategies(widened, preferReferenceOnly)
}

func draftOutputModeCandidates(strategy draftStrategy, selected draftOutputMode, widened bool) []draftOutputMode {
	return DraftOutputModeCandidates(strategy, selected, widened)
}

func draftStrategyMaxLayers(strategy draftStrategy, totalLayers int) int {
	return DraftStrategyMaxLayers(strategy, totalLayers)
}

func isWidenedAttentionDraft(cfg *models.ModelConfig) bool {
	return IsWidenedAttentionDraft(cfg)
}

func draftSelectionLowConfidence(strategy draftStrategy, kind draftModelKind, widened bool) bool {
	return DraftSelectionLowConfidence(strategy, kind, widened)
}
