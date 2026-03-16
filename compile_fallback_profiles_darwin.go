//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"
	"os"
	"strings"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

func compileWithFallbackProfiles(
	model appleneuralengine.ANEInMemoryModel,
	options foundation.NSMutableDictionary,
) error {
	if _, err := model.CompileWithQoSOptionsError(defaultANEQoS, options); err == nil {
		return nil
	} else {
		profiles, parseErr := compileFallbackProfilesFromEnv()
		if parseErr != nil {
			return fmt.Errorf("%w (parse %s: %v)", err, activeCompileFallbackProfilesEnv(), parseErr)
		}
		if len(profiles) == 0 {
			return err
		}
		attempts := make([]string, 0, len(profiles))
		for _, profile := range profiles {
			retryOptions := foundation.NewMutableDictionaryWithCapacity(2)
			applyCompileFallbackProfile(retryOptions, profile)
			if _, retryErr := model.CompileWithQoSOptionsError(defaultANEQoS, retryOptions); retryErr == nil {
				return nil
			} else {
				attempts = append(attempts, fmt.Sprintf("%s => %v", profile.String(), retryErr))
			}
		}
		return fmt.Errorf(
			"%w (fallback profiles exhausted: %s)",
			err,
			strings.Join(attempts, "; "),
		)
	}
}

func compileFallbackProfilesFromEnv() ([]aneCompileFallbackProfile, error) {
	return parseANECompileFallbackProfiles(os.Getenv(activeCompileFallbackProfilesEnv()))
}

func activeCompileFallbackProfilesEnv() string {
	if strings.TrimSpace(os.Getenv(aneCompileFallbackProfilesEnv)) != "" {
		return aneCompileFallbackProfilesEnv
	}
	return mlxgoCompileFallbackProfilesEnv
}

func applyCompileFallbackProfile(
	options foundation.NSMutableDictionary,
	profile aneCompileFallbackProfile,
) {
	if options.GetID() == 0 {
		return
	}
	if profile.ModelType != "" {
		setANEOptionObject(
			options,
			appleneuralengine.KANEFModelTypeKey,
			"kANEFModelTypeKey",
			resolveCompileProfileModelType(profile.ModelType),
		)
	}
	if profile.NetPlist != "" {
		setANEOptionObject(
			options,
			appleneuralengine.KANEFNetPlistFilenameKey,
			"kANEFNetPlistFilenameKey",
			foundation.NewStringWithString(profile.NetPlist),
		)
	}
}

func setANEOptionObject(
	options foundation.NSMutableDictionary,
	key objectivec.IObject,
	fallbackKey string,
	value objectivec.IObject,
) {
	if value == nil || value.GetID() == 0 {
		return
	}
	if key != nil && key.GetID() != 0 {
		options.SetObjectForKey(value, key)
		return
	}
	options.SetObjectForKey(value, foundation.NewStringWithString(fallbackKey))
}

func resolveCompileProfileModelType(token string) objectivec.IObject {
	switch strings.ToLower(strings.TrimSpace(token)) {
	case "mil", "kanefmodelmil", "kanefmodelmilvalue":
		if appleneuralengine.KANEFModelMILValue.GetID() != 0 {
			return appleneuralengine.KANEFModelMILValue
		}
	case "precompiled", "kanefmodelprecompiled", "kanefmodelprecompiledvalue":
		if appleneuralengine.KANEFModelPreCompiledValue.GetID() != 0 {
			return appleneuralengine.KANEFModelPreCompiledValue
		}
	case "anecir", "kanefmodelanecir", "kanefmodelanecirvalue":
		if appleneuralengine.KANEFModelANECIRValue.GetID() != 0 {
			return appleneuralengine.KANEFModelANECIRValue
		}
	case "mlir", "kanefmodelmlir", "kanefmodelmlirvalue":
		if appleneuralengine.KANEFModelMLIRValue.GetID() != 0 {
			return appleneuralengine.KANEFModelMLIRValue
		}
	case "coreml", "kanefmodelcoreml", "kanefmodelcoremlvalue":
		if appleneuralengine.KANEFModelCoreMLValue.GetID() != 0 {
			return appleneuralengine.KANEFModelCoreMLValue
		}
	}
	return foundation.NewStringWithString(strings.TrimSpace(token))
}
