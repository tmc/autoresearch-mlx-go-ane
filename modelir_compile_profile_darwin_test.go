//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"slices"
	"strings"
	"testing"
)

const expectedHostDefendedProfile = "skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=true disable_final_norm=false"

func TestModelIRCompileFallbackProfilesWithGate(t *testing.T) {
	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			DynamicRoPEInputs: true,
		},
	}
	effective := opts.TransformerConfig
	effective.AttentionOutputGate = true
	var got []string
	for _, p := range modelIRCompileFallbackProfiles(opts, effective) {
		got = append(got, p.Label)
	}
	want := []string{
		"fused_ffn_disable_input_qk_norms",
		"fused_ffn_no_gate_disable_qk_norms",
		"fused_ffn_no_gate_disable_input_qk_norms",
		"fused_ffn_no_gate",
		"fused_ffn",
		"skip_ffn_no_gate",
		"skip_ffn",
		"skip_ffn_disable_final_norm",
		"skip_ffn_disable_final_norm_no_gate",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("labels=%v want %v", got, want)
	}
}

func TestModelIRCompileFallbackProfilesWithGatePrefersDefendedHostTier(t *testing.T) {
	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			DynamicRoPEInputs:          true,
			AttentionOutputGate:        true,
			DisableFinalNormOp:         false,
			DisableInputNormOps:        false,
			DisableQKNormOps:           false,
			DisableNormOps:             false,
			DisableAttentionOutputGate: false,
		},
	}
	profiles := modelIRCompileFallbackProfiles(opts, opts.TransformerConfig)
	if len(profiles) == 0 {
		t.Fatal("no fallback profiles generated")
	}
	if got := profiles[0].Label; got != "fused_ffn_disable_input_qk_norms" {
		t.Fatalf("profiles[0].Label=%q want fused_ffn_disable_input_qk_norms", got)
	}
	want := "skip_ffn=false attention_output_gate=true fused_linear_ffn=true disable_norm=false disable_input_norms=true disable_qk_norms=true disable_final_norm=false"
	if got := profiles[0].Key; got != want {
		t.Fatalf("profiles[0].Key=%q want %q", got, want)
	}
}

func TestModelIRCompileFallbackProfilesWithoutGate(t *testing.T) {
	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			AttentionOutputGate: false,
		},
	}
	var got []string
	for _, p := range modelIRCompileFallbackProfiles(opts, opts.TransformerConfig) {
		got = append(got, p.Label)
	}
	want := []string{
		"fused_ffn",
		"skip_ffn",
		"skip_ffn_disable_final_norm",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("labels=%v want %v", got, want)
	}
}

func TestModelIRCompileFallbackProfilesEnvFallbacks(t *testing.T) {
	t.Setenv("MLXGO_ANE_MODELIR_ENABLE_SELECTIVE_NORM_FALLBACK", "1")
	t.Setenv("MLXGO_ANE_MODELIR_ENABLE_DISABLE_NORM_FALLBACK", "1")
	opts := ReifyOptions{
		TransformerConfig: MILTransformerConfig{
			AttentionOutputGate: true,
		},
	}
	var got []string
	for _, p := range modelIRCompileFallbackProfiles(opts, opts.TransformerConfig) {
		got = append(got, p.Label)
	}
	want := []string{
		"fused_ffn_disable_input_qk_norms",
		"fused_ffn_no_gate_disable_qk_norms",
		"fused_ffn_no_gate_disable_input_qk_norms",
		"fused_ffn_no_gate",
		"fused_ffn",
		"skip_ffn_no_gate",
		"skip_ffn",
		"skip_ffn_disable_final_norm",
		"skip_ffn_disable_final_norm_no_gate",
		"skip_ffn_disable_input_norms",
		"skip_ffn_disable_qk_norms",
		"skip_ffn_disable_norm",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("labels=%v want %v", got, want)
	}
}

func TestModelIRCompileProfileKeyIncludesGate(t *testing.T) {
	key := modelIRCompileProfileKey(MILTransformerConfig{
		SkipFFN:             true,
		AttentionOutputGate: false,
		DisableFinalNormOp:  true,
	})
	want := "skip_ffn=true attention_output_gate=false fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=true"
	if key != want {
		t.Fatalf("key=%q want %q", key, want)
	}
}

func TestModelIRCompileProfileKeyDisableGateWins(t *testing.T) {
	key := modelIRCompileProfileKey(MILTransformerConfig{
		SkipFFN:                    true,
		AttentionOutputGate:        true,
		DisableAttentionOutputGate: true,
	})
	want := "skip_ffn=true attention_output_gate=false fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false"
	if key != want {
		t.Fatalf("key=%q want %q", key, want)
	}
}

func TestModelIRCompileProfileTier(t *testing.T) {
	cases := []struct {
		name    string
		profile string
		want    string
	}{
		{name: "full", profile: "", want: "full"},
		{name: "fused_ffn_disable_input_qk_norms", profile: "skip_ffn=false attention_output_gate=true fused_linear_ffn=true disable_norm=false disable_input_norms=true disable_qk_norms=true disable_final_norm=false", want: "fused_ffn_disable_input_qk_norms"},
		{name: "fused_ffn_disable_input_qk_norms", profile: "skip_ffn=false attention_output_gate=true fused_linear_ffn=true disable_norm=false disable_input_norms=true disable_qk_norms=true disable_final_norm=false", want: "fused_ffn_disable_input_qk_norms"},
		{name: "fused_ffn_no_gate_disable_qk_norms", profile: "skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=true disable_final_norm=false", want: "fused_ffn_no_gate_disable_qk_norms"},
		{name: "fused_ffn_no_gate_disable_input_qk_norms", profile: "skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=true disable_qk_norms=true disable_final_norm=false", want: "fused_ffn_no_gate_disable_input_qk_norms"},
		{name: "fused_ffn_no_gate", profile: "skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false", want: "fused_ffn_no_gate"},
		{name: "fused_ffn", profile: "skip_ffn=false attention_output_gate=true fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false", want: "fused_ffn"},
		{name: "fused_ffn_no_gate_disable_qk_norms_alias", profile: expectedHostDefendedProfile, want: "fused_ffn_no_gate_disable_qk_norms"},
		{name: "skip_ffn_no_gate", profile: "skip_ffn=true attention_output_gate=false fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false", want: "skip_ffn_no_gate"},
		{name: "skip_ffn", profile: "skip_ffn=true attention_output_gate=true fused_linear_ffn=false disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false", want: "skip_ffn"},
		{name: "custom", profile: "skip_ffn=true attention_output_gate=false fused_linear_ffn=false disable_norm=true disable_input_norms=false disable_qk_norms=false disable_final_norm=false", want: "custom"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := ModelIRCompileProfileTier(tc.profile); got != tc.want {
				t.Fatalf("ModelIRCompileProfileTier(%q)=%q want %q", tc.profile, got, tc.want)
			}
		})
	}
}

func TestModelIRCompileProfileNote(t *testing.T) {
	got := ModelIRCompileProfileNote(expectedHostDefendedProfile)
	if got == "" {
		t.Fatal("ModelIRCompileProfileNote(defended host profile) returned empty note")
	}
	if !strings.Contains(got, "before the first live FFN affine") {
		t.Fatalf("ModelIRCompileProfileNote(defended host profile)=%q missing refined hazard note", got)
	}
	if !strings.Contains(got, "constant-backed") {
		t.Fatalf("ModelIRCompileProfileNote(defended host profile)=%q missing rope-source hazard note", got)
	}
	if got := ModelIRCompileProfileNote(""); got != "" {
		t.Fatalf("ModelIRCompileProfileNote(full)=%q want empty", got)
	}
	fused := ModelIRCompileProfileNote("skip_ffn=false attention_output_gate=false fused_linear_ffn=true disable_norm=false disable_input_norms=false disable_qk_norms=false disable_final_norm=false")
	if !strings.Contains(fused, "fused first-stage FFN") {
		t.Fatalf("ModelIRCompileProfileNote(fused tier)=%q missing fused note", fused)
	}
	fusedNorms := ModelIRCompileProfileNote("skip_ffn=false attention_output_gate=true fused_linear_ffn=true disable_norm=false disable_input_norms=true disable_qk_norms=true disable_final_norm=false")
	if !strings.Contains(fusedNorms, "input norm") || !strings.Contains(fusedNorms, "qk norms") || !strings.Contains(fusedNorms, "output-gate") {
		t.Fatalf("ModelIRCompileProfileNote(fused selective tier)=%q missing selective norm note", fusedNorms)
	}
	fusedNoGate := ModelIRCompileProfileNote(expectedHostDefendedProfile)
	if !strings.Contains(fusedNoGate, "qk norm") || !strings.Contains(fusedNoGate, "output-gate") {
		t.Fatalf("ModelIRCompileProfileNote(defended fused no-gate tier)=%q missing qk/output-gate note", fusedNoGate)
	}
	if strings.Contains(fusedNoGate, "input norm") {
		t.Fatalf("ModelIRCompileProfileNote(defended fused no-gate tier)=%q unexpectedly mentions input norm on MLX", fusedNoGate)
	}
}

func TestModelIRCompileProfileQwenSupport(t *testing.T) {
	onANE, onMLX := ModelIRCompileProfileQwenSupport(expectedHostDefendedProfile)
	if onANE != "attention,stateful_kv,dynamic_rope,input_norm,final_norm,ffn,fused_linear_ffn" {
		t.Fatalf("onANE=%q", onANE)
	}
	if onMLX != "qk_norm,output_gate" {
		t.Fatalf("onMLX=%q", onMLX)
	}

	fullANE, fullMLX := ModelIRCompileProfileQwenSupport("")
	if fullANE != "attention,stateful_kv,dynamic_rope,input_norm,qk_norm,final_norm,output_gate,ffn" {
		t.Fatalf("full onANE=%q", fullANE)
	}
	if fullMLX != "none" {
		t.Fatalf("full onMLX=%q", fullMLX)
	}
}
