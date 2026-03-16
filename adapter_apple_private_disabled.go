//go:build !darwin || !ane_appleneuralengine

package mlxgoane

import "fmt"

// NewApplePrivateExecutor returns an executor backed by private ANE bindings.
//
// Build with `-tags ane_appleneuralengine` on darwin to enable this adapter.
func NewApplePrivateExecutor() (LinearExecutor, error) {
	return nil, fmt.Errorf("apple private ANE adapter is disabled (requires darwin and -tags ane_appleneuralengine)")
}

// NewApplePrivateDynamicLinearExecutor returns a training-oriented executor
// backed by private ANE bindings that treats weights as runtime inputs.
func NewApplePrivateDynamicLinearExecutor() (LinearExecutor, error) {
	return nil, fmt.Errorf("apple private ANE adapter is disabled (requires darwin and -tags ane_appleneuralengine)")
}
