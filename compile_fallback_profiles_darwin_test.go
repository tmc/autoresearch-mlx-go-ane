//go:build darwin && ane_appleneuralengine

package mlxgoane

import "testing"

func TestCompileFallbackProfilesFromEnvPrecedence(t *testing.T) {
	t.Setenv(aneCompileFallbackProfilesEnv, "")
	t.Setenv(mlxgoCompileFallbackProfilesEnv, "kANEFModelMIL:<empty>")
	profiles, err := compileFallbackProfilesFromEnv()
	if err != nil {
		t.Fatalf("compileFallbackProfilesFromEnv mlxgo env: %v", err)
	}
	if len(profiles) != 1 || profiles[0].ModelType != "kANEFModelMIL" || profiles[0].NetPlist != "" {
		t.Fatalf("mlxgo env profiles=%v want [{kANEFModelMIL \"\"}]", profiles)
	}

	t.Setenv(aneCompileFallbackProfilesEnv, "<empty>:fallback.plist")
	profiles, err = compileFallbackProfilesFromEnv()
	if err != nil {
		t.Fatalf("compileFallbackProfilesFromEnv ane env: %v", err)
	}
	if len(profiles) != 1 || profiles[0].ModelType != "" || profiles[0].NetPlist != "fallback.plist" {
		t.Fatalf("ane env profiles=%v want [{\"\" fallback.plist}]", profiles)
	}
}
