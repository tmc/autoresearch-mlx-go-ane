//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"
	"strings"
)

type modelIRCompileFallbackProfile struct {
	Label string
	Key   string
	Opts  ReifyOptions
}

// ModelIRCompileCoverage summarizes which Qwen decode features remain on ANE
// for a given direct ModelIR compile profile.
type ModelIRCompileCoverage struct {
	Attention           bool
	StatefulKV          bool
	DynamicRoPE         bool
	InputNorm           bool
	QKNorm              bool
	FinalNorm           bool
	FFN                 bool
	AttentionOutputGate bool
	FusedLinearFFN      bool
}

// ModelIRCompileProfileTier classifies the effective direct ModelIR compile profile.
//
// On the current host/toolchain, the defended stateful-Qwen tier is currently
// "fused_ffn_no_gate_disable_qk_norms". Targeted compile probes show
// that:
//   - dynamic or runtime-sourced RoPE can coexist with the attention trunk and
//     the first live FFN affine;
//   - constant-backed RoPE and parameterized transforms become compile-hostile
//     before that first live FFN affine;
//   - disabling the learned attention-output gate is required before the
//     stronger fused-linear FFN tier can compile on this host; and
//   - qk norm also stays on MLX in the defended host tier.
//
// These tiers are therefore explicit measured host fallbacks, not arbitrary
// degradations.
func ModelIRCompileProfileTier(profile string) string {
	switch profile {
	case "":
		return "full"
	case "skip_ffn=false attention_output_gate=true fused_linear_ffn=true disable_norm=false disable_input_norms=true disable_qk_norms=true disable_final_norm=false":
		return "fused_ffn_disable_input_qk_norms"
	case "skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=true disable_final_norm=false":
		return "fused_ffn_no_gate_disable_qk_norms"
	case "skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=true disable_qk_norms=true disable_final_norm=false":
		return "fused_ffn_no_gate_disable_input_qk_norms"
	case "skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false":
		return "fused_ffn_no_gate"
	case "skip_ffn=false attention_output_gate=true fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false":
		return "fused_ffn"
	case "skip_ffn=true attention_output_gate=false fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false":
		return "skip_ffn_no_gate"
	case "skip_ffn=true attention_output_gate=true fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false":
		return "skip_ffn"
	case "skip_ffn=true attention_output_gate=true fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=true":
		return "skip_ffn_disable_final_norm"
	case "skip_ffn=true attention_output_gate=false fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=true":
		return "skip_ffn_disable_final_norm_no_gate"
	case "skip_ffn=true attention_output_gate=true fused_linear_ffn=false disable_norm=false disable_input_norms=true disable_qk_norms=false disable_final_norm=false":
		return "skip_ffn_disable_input_norms"
	case "skip_ffn=true attention_output_gate=true fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=true disable_final_norm=false":
		return "skip_ffn_disable_qk_norms"
	case "skip_ffn=true attention_output_gate=true fused_linear_ffn=false disable_norm=true disable_input_norms=false disable_qk_norms=false disable_final_norm=false":
		return "skip_ffn_disable_norm"
	default:
		return "custom"
	}
}

// ModelIRCompileProfileNote explains the measured host behavior for a compile
// tier when we have a concrete explanation worth surfacing to callers.
//
// The current defended host tier keeps the learned output-gate branch and qk
// norm on MLX while leaving the stateful attention trunk, dynamic RoPE path,
// input norm, fused first-stage FFN, and final norm on ANE.
func ModelIRCompileProfileNote(profile string) string {
	switch ModelIRCompileProfileTier(profile) {
	case "fused_ffn_disable_input_qk_norms":
		return "constant-backed or parameterized rope-side transforms before the first live FFN affine are compile-hostile on this host; fused first-stage FFN and the learned output-gate branch survive when input norm and qk norms stay on MLX"
	case "fused_ffn_no_gate_disable_qk_norms":
		return "constant-backed or parameterized rope-side transforms before the first live FFN affine are compile-hostile on this host; fused first-stage FFN survives when the learned output-gate branch and qk norm stay on MLX"
	case "fused_ffn_no_gate_disable_input_qk_norms":
		return "constant-backed or parameterized rope-side transforms before the first live FFN affine are compile-hostile on this host; fused first-stage FFN survives when the learned output-gate branch, input norm, and qk norms stay on MLX"
	case "fused_ffn_no_gate":
		return "constant-backed or parameterized rope-side transforms before the first live FFN affine are compile-hostile on this host; fused first-stage FFN survives when the learned output-gate branch stays on MLX"
	case "skip_ffn_no_gate":
		return "constant-backed or parameterized rope-side transforms before the first live FFN affine are compile-hostile on this host; FFN and the learned output-gate branch stay on MLX"
	default:
		return ""
	}
}

// ModelIRCompileProfileCoverage returns the Qwen decode feature coverage implied
// by a direct ModelIR compile profile on the current host.
func ModelIRCompileProfileCoverage(profile string) ModelIRCompileCoverage {
	cov := ModelIRCompileCoverage{
		Attention:           true,
		StatefulKV:          true,
		DynamicRoPE:         true,
		InputNorm:           true,
		QKNorm:              true,
		FinalNorm:           true,
		FFN:                 true,
		AttentionOutputGate: true,
	}
	flags := parseModelIRCompileProfile(profile)
	if flags["skip_ffn"] {
		cov.FFN = false
	}
	if flags["disable_final_norm"] {
		cov.FinalNorm = false
	}
	if flags["disable_input_norms"] {
		cov.InputNorm = false
	}
	if flags["disable_qk_norms"] {
		cov.QKNorm = false
	}
	if v, ok := flags["attention_output_gate"]; ok {
		cov.AttentionOutputGate = v
	}
	if v, ok := flags["fused_linear_ffn"]; ok {
		cov.FusedLinearFFN = v
	}
	return cov
}

// ModelIRCompileProfileQwenSupport returns short human-facing lists of the Qwen
// decode features active on ANE versus those still staying on MLX.
func ModelIRCompileProfileQwenSupport(profile string) (onANE string, onMLX string) {
	cov := ModelIRCompileProfileCoverage(profile)
	var ane []string
	var mlx []string
	add := func(ok bool, name string) {
		if ok {
			ane = append(ane, name)
			return
		}
		mlx = append(mlx, name)
	}
	add(cov.Attention, "attention")
	add(cov.StatefulKV, "stateful_kv")
	add(cov.DynamicRoPE, "dynamic_rope")
	add(cov.InputNorm, "input_norm")
	add(cov.QKNorm, "qk_norm")
	add(cov.FinalNorm, "final_norm")
	add(cov.AttentionOutputGate, "output_gate")
	add(cov.FFN, "ffn")
	if cov.FusedLinearFFN {
		ane = append(ane, "fused_linear_ffn")
	}
	if len(ane) == 0 {
		onANE = "none"
	} else {
		onANE = joinProfileParts(ane)
	}
	if len(mlx) == 0 {
		onMLX = "none"
	} else {
		onMLX = joinProfileParts(mlx)
	}
	return onANE, onMLX
}

func parseModelIRCompileProfile(profile string) map[string]bool {
	out := make(map[string]bool)
	for _, part := range splitProfileParts(profile) {
		key, value, ok := strings.Cut(part, "=")
		if !ok {
			continue
		}
		out[key] = value == "true"
	}
	return out
}

func splitProfileParts(profile string) []string {
	if profile == "" {
		return nil
	}
	var parts []string
	for _, part := range strings.Fields(profile) {
		if part != "" {
			parts = append(parts, part)
		}
	}
	return parts
}

func joinProfileParts(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, ",")
}

func modelIRCompileProfileKey(cfg MILTransformerConfig) string {
	attentionOutputGate := cfg.AttentionOutputGate && !cfg.DisableAttentionOutputGate
	return fmt.Sprintf(
		"skip_ffn=%t attention_output_gate=%t fused_linear_ffn=%t disable_norm=%t disable_input_norms=%t disable_qk_norms=%t disable_final_norm=%t",
		cfg.SkipFFN,
		attentionOutputGate,
		cfg.FusedLinearFFN,
		cfg.DisableNormOps,
		cfg.DisableInputNormOps,
		cfg.DisableQKNormOps,
		cfg.DisableFinalNormOp,
	)
}

// modelIRCompileFallbackProfiles orders fallback tiers from least to most
// degraded according to measured host compiler behavior.
//
// The most important host-specific rule today is that stateful Qwen trunks with
// rope-side constant-backed or parameterized transforms become compile-hostile
// once they enter the region before the first live FFN affine stage. We
// therefore try removing FFN and the learned attention-output gate before
// dropping final norm or broader norm support.
func modelIRCompileFallbackProfiles(base ReifyOptions, effective MILTransformerConfig) []modelIRCompileFallbackProfile {
	if effective.NumLayers == 0 && effective.Dim == 0 && effective.AttentionDim == 0 &&
		effective.NumHeads == 0 && effective.HeadDim == 0 && effective.HiddenDim == 0 &&
		effective.VocabSize == 0 && !effective.SkipFFN && !effective.AttentionOutputGate &&
		!effective.DisableAttentionOutputGate && !effective.FusedLinearFFN && !effective.DisableNormOps &&
		!effective.DisableInputNormOps && !effective.DisableQKNormOps &&
		!effective.DisableFinalNormOp && !effective.DynamicRoPEInputs {
		effective = base.TransformerConfig
	}
	baseKey := modelIRCompileProfileKey(effective)
	seen := map[string]bool{baseKey: true}
	var out []modelIRCompileFallbackProfile

	add := func(label string, apply func(*ReifyOptions)) {
		try := base
		try.TransformerConfig = effective
		apply(&try)
		key := modelIRCompileProfileKey(try.TransformerConfig)
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, modelIRCompileFallbackProfile{
			Label: label,
			Key:   key,
			Opts:  try,
		})
	}

	if effective.AttentionOutputGate && !effective.DisableAttentionOutputGate {
		add("fused_ffn_disable_input_qk_norms", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = false
			o.TransformerConfig.FusedLinearFFN = true
			o.TransformerConfig.LinearFFN = true
			o.TransformerConfig.UseConvFFN = false
			o.TransformerConfig.DisableInputNormOps = true
			o.TransformerConfig.DisableQKNormOps = true
		})
		add("fused_ffn_no_gate_disable_qk_norms", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = false
			o.TransformerConfig.DisableAttentionOutputGate = true
			o.TransformerConfig.FusedLinearFFN = true
			o.TransformerConfig.LinearFFN = true
			o.TransformerConfig.UseConvFFN = false
			o.TransformerConfig.DisableQKNormOps = true
		})
		add("fused_ffn_no_gate_disable_input_qk_norms", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = false
			o.TransformerConfig.DisableAttentionOutputGate = true
			o.TransformerConfig.FusedLinearFFN = true
			o.TransformerConfig.LinearFFN = true
			o.TransformerConfig.UseConvFFN = false
			o.TransformerConfig.DisableInputNormOps = true
			o.TransformerConfig.DisableQKNormOps = true
		})
		add("fused_ffn_no_gate", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = false
			o.TransformerConfig.DisableAttentionOutputGate = true
			o.TransformerConfig.FusedLinearFFN = true
			o.TransformerConfig.LinearFFN = true
			o.TransformerConfig.UseConvFFN = false
		})
	}
	add("fused_ffn", func(o *ReifyOptions) {
		o.TransformerConfig.SkipFFN = false
		o.TransformerConfig.FusedLinearFFN = true
		o.TransformerConfig.LinearFFN = true
		o.TransformerConfig.UseConvFFN = false
	})
	if effective.AttentionOutputGate && !effective.DisableAttentionOutputGate {
		add("skip_ffn_no_gate", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = true
			o.TransformerConfig.FusedLinearFFN = false
			o.TransformerConfig.DisableAttentionOutputGate = true
		})
	}
	add("skip_ffn", func(o *ReifyOptions) {
		o.TransformerConfig.SkipFFN = true
		o.TransformerConfig.FusedLinearFFN = false
	})
	add("skip_ffn_disable_final_norm", func(o *ReifyOptions) {
		o.TransformerConfig.SkipFFN = true
		o.TransformerConfig.FusedLinearFFN = false
		o.TransformerConfig.DisableFinalNormOp = true
	})
	if effective.AttentionOutputGate && !effective.DisableAttentionOutputGate {
		add("skip_ffn_disable_final_norm_no_gate", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = true
			o.TransformerConfig.FusedLinearFFN = false
			o.TransformerConfig.DisableFinalNormOp = true
			o.TransformerConfig.DisableAttentionOutputGate = true
		})
	}
	if enableSelectiveNormCompileFallback() {
		add("skip_ffn_disable_input_norms", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = true
			o.TransformerConfig.FusedLinearFFN = false
			o.TransformerConfig.DisableInputNormOps = true
		})
		add("skip_ffn_disable_qk_norms", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = true
			o.TransformerConfig.FusedLinearFFN = false
			o.TransformerConfig.DisableQKNormOps = true
		})
	}
	if enableDisableNormCompileFallback() {
		add("skip_ffn_disable_norm", func(o *ReifyOptions) {
			o.TransformerConfig.SkipFFN = true
			o.TransformerConfig.FusedLinearFFN = false
			o.TransformerConfig.DisableNormOps = true
		})
	}
	return out
}
