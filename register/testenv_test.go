//go:build darwin

package register

import (
	"os"
	"testing"
)

func requireANEFFNTests(t *testing.T) {
	t.Helper()
	if os.Getenv("MLXGO_RUN_ANE_FFN_TESTS") != "1" {
		t.Skip("set MLXGO_RUN_ANE_FFN_TESTS=1 to run hardware FFN integration tests")
	}
}
