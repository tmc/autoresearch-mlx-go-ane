package main

import "testing"

func TestLoadANEEnvGPUFallbackCompat(t *testing.T) {
	t.Setenv("MLXGO_ANE_GPU_FALLBACK", "1")
	t.Setenv("MLXGO_ANE_DECODE_CACHE", "/tmp/cache")

	loadANEEnv()

	if got, want := *aneDecodePlane, "gpu_fallback"; got != want {
		t.Fatalf("aneDecodePlane = %q, want %q", got, want)
	}
	if got, want := *aneDecodeCache, "/tmp/cache"; got != want {
		t.Fatalf("aneDecodeCache = %q, want %q", got, want)
	}
}

func TestLoadANEEnvExplicitModeWins(t *testing.T) {
	t.Setenv("MLXGO_ANE_DECODE_PLANE", "off")
	t.Setenv("MLXGO_ANE_GPU_FALLBACK", "1")

	loadANEEnv()

	if got, want := *aneDecodePlane, "off"; got != want {
		t.Fatalf("aneDecodePlane = %q, want %q", got, want)
	}
}

func TestNormalizeANEDecodePlaneMode(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "off", in: "off", want: "off"},
		{name: "qwen35 alias", in: "qwen3.5", want: "qwen35"},
		{name: "gpu fallback underscore", in: "gpu_fallback", want: "gpu_fallback"},
		{name: "gpu fallback hyphen", in: "gpu-fallback", want: "gpu_fallback"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeANEDecodePlaneMode(tt.in)
			if err != nil {
				t.Fatalf("normalizeANEDecodePlaneMode(%q): %v", tt.in, err)
			}
			if got != tt.want {
				t.Fatalf("normalizeANEDecodePlaneMode(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
