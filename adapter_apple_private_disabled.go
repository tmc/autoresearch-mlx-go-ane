//go:build !darwin

package mlxgoane

import "fmt"

// NewApplePrivateExecutor returns an executor backed by private ANE bindings.
//
// This stub is compiled on non-darwin platforms.
func NewApplePrivateExecutor() (LinearExecutor, error) {
	return nil, fmt.Errorf("apple private ANE adapter requires darwin")
}

// NewApplePrivateDynamicLinearExecutor returns a training-oriented executor
// backed by private ANE bindings that treats weights as runtime inputs.
func NewApplePrivateDynamicLinearExecutor() (LinearExecutor, error) {
	return nil, fmt.Errorf("apple private ANE adapter requires darwin")
}
