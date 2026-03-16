package anedraftimpl

import (
	"os"
	"strings"
)

const (
	SpeculativePathSSD = "ssd"
	SSDDraftBackendMLX = "mlx"
	SSDDraftBackendANE = "ane"
)

func ShouldUseANEDraft(specPath, ssdBackend, explicitModelC string) bool {
	if explicitModelC != "" {
		return true
	}
	if strings.TrimSpace(os.Getenv("MLXGO_ANE_DRAFT_STRATEGY")) != "" {
		return true
	}
	if strings.TrimSpace(os.Getenv("MLXGO_ANE_DRAFT_OUTPUT_MODE")) != "" {
		return true
	}
	return specPath == SpeculativePathSSD && ssdBackend != SSDDraftBackendMLX
}

func ShouldRequireANEDraft(specPath, ssdBackend string) bool {
	return specPath == SpeculativePathSSD && ssdBackend == SSDDraftBackendANE
}

func shouldUseANEDraft(specPath, ssdBackend, explicitModelC string) bool {
	return ShouldUseANEDraft(specPath, ssdBackend, explicitModelC)
}

func shouldRequireANEDraft(specPath, ssdBackend string) bool {
	return ShouldRequireANEDraft(specPath, ssdBackend)
}
