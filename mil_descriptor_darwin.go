//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"
	"strings"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

type milDescriptorMode int

const (
	milDescriptorPreferExplicit milDescriptorMode = iota
	milDescriptorRequireExplicit
	milDescriptorHelperOnly
)

// newMILTextDescriptor builds a descriptor for MIL text and prefers the
// explicit isMILModel initializer path. It returns whether that preferred path
// was used.
func newMILTextDescriptor(
	milText string,
	weights objectivec.IObject,
	plist objectivec.IObject,
) (objectivec.IObject, bool, error) {
	return newMILTextDescriptorWithMode(milText, weights, plist, milDescriptorPreferExplicit)
}

func newMILTextDescriptorWithMode(
	milText string,
	weights objectivec.IObject,
	plist objectivec.IObject,
	mode milDescriptorMode,
) (objectivec.IObject, bool, error) {
	if strings.TrimSpace(milText) == "" {
		return objectivec.Object{}, false, fmt.Errorf("mil descriptor: mil text is empty")
	}
	milData := foundation.NewDataWithBytesLength([]byte(milText))

	if mode != milDescriptorHelperOnly {
		// Preferred path: explicitly mark descriptor as MIL-backed.
		desc := appleneuralengine.NewANEInMemoryModelDescriptorWithNetworkTextWeightsOptionsPlistIsMILModel(
			milData,
			weights,
			plist,
			true,
		)
		if desc.GetID() != 0 {
			return objectivec.ObjectFromID(desc.GetID()), true, nil
		}
		if mode == milDescriptorRequireExplicit {
			return objectivec.Object{}, false, fmt.Errorf("mil descriptor: explicit isMILModel initializer returned nil")
		}
	}

	// Fallback path: class helper seen in earlier experiments.
	descObj := appleneuralengine.GetANEInMemoryModelDescriptorClass().ModelWithMILTextWeightsOptionsPlist(
		milData,
		weights,
		plist,
	)
	if descObj.GetID() == 0 {
		return objectivec.Object{}, false, fmt.Errorf("mil descriptor: descriptor creation returned nil")
	}
	return descObj, false, nil
}
