//go:build darwin

package mlxgoane

import (
	"os"
	"testing"
)

const aneIntegrationTestsEnv = "MLXGO_RUN_ANE_INTEGRATION_TESTS"

func requireANEIntegrationTests(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping ANE integration test in short mode")
	}
	switch os.Getenv(aneIntegrationTestsEnv) {
	case "1", "true", "TRUE", "yes", "YES", "on", "ON":
		return
	}
	t.Skipf("set %s=1 to run ANE integration tests", aneIntegrationTestsEnv)
}
