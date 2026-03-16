package mlxgoane

import (
	"fmt"
	"strings"
)

const (
	aneCompileFallbackProfilesEnv      = "ANE_COMPILE_FALLBACK_PROFILES"
	mlxgoCompileFallbackProfilesEnv    = "MLXGO_ANE_COMPILE_FALLBACK_PROFILES"
	compileFallbackEmptyFieldSentinel  = "<empty>"
	compileFallbackDisabledFieldSymbol = "-"
)

type aneCompileFallbackProfile struct {
	ModelType string
	NetPlist  string
}

func (p aneCompileFallbackProfile) String() string {
	modelType := p.ModelType
	if strings.TrimSpace(modelType) == "" {
		modelType = compileFallbackEmptyFieldSentinel
	}
	netPlist := p.NetPlist
	if strings.TrimSpace(netPlist) == "" {
		netPlist = compileFallbackEmptyFieldSentinel
	}
	return modelType + ":" + netPlist
}

func parseANECompileFallbackProfiles(raw string) ([]aneCompileFallbackProfile, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	out := make([]aneCompileFallbackProfile, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		pair := strings.SplitN(part, ":", 2)
		if len(pair) != 2 {
			return nil, fmt.Errorf("parse compile fallback profiles: invalid entry %q (want modelType:netPlist)", part)
		}
		modelType := normalizeCompileProfileField(pair[0])
		netPlist := normalizeCompileProfileField(pair[1])
		profile := aneCompileFallbackProfile{
			ModelType: modelType,
			NetPlist:  netPlist,
		}
		key := profile.ModelType + "\x00" + profile.NetPlist
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, profile)
	}
	return out, nil
}

func normalizeCompileProfileField(v string) string {
	s := strings.TrimSpace(v)
	switch strings.ToLower(s) {
	case "", compileFallbackEmptyFieldSentinel, compileFallbackDisabledFieldSymbol:
		return ""
	default:
		return s
	}
}

func isInvalidMILProgramCompileError(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(strings.ToLower(err.Error()), "invalidmilprogram")
}
