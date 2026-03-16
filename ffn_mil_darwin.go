//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"fmt"

	"github.com/tmc/apple/private/appleneuralengine"
)

// FFNMILModel is a compiled MIL-based SwiGLU FFN model with layout-aware
// IOSurfaces ready for evaluation via SurfaceEvalPlan.
type FFNMILModel struct {
	Model  appleneuralengine.ANEInMemoryModel
	Input  *IOSurfaceFloat32
	Output *IOSurfaceFloat32
	Dim    int
}

// Close releases the model and IOSurfaces.
func (m *FFNMILModel) Close() {
	if m == nil {
		return
	}
	if m.Input != nil {
		m.Input.Close()
	}
	if m.Output != nil {
		m.Output.Close()
	}
	if m.Model.ID != 0 {
		_ = callObjCBoolWithNSError(
			"ffn mil model unload",
			m.Model.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
	}
}

// CompileFFNMIL builds a MIL-based SwiGLU FFN and returns the compiled model
// with layout-aware IOSurfaces.
//
// Weights are expected in [out, in] row-major order:
//   - gate (w1): [hidden, dim]
//   - up   (w3): [hidden, dim]
//   - down (w2): [dim, hidden]
func CompileFFNMIL(dim, hidden int, gate, up, down []float32) (*FFNMILModel, error) {
	milText, files, err := BuildFFNMILArtifacts(dim, hidden, gate, up, down)
	if err != nil {
		return nil, err
	}

	model, err := buildModelFromMILTextWithDescriptorFallback(
		fmt.Sprintf("ffn mil %dx%d", dim, hidden),
		milText,
		files,
	)
	if err != nil {
		return nil, fmt.Errorf("compile ffn MIL: %w", err)
	}

	// Parse compiled model schema for layout-aware IOSurfaces.
	schema, err := parseCompiledModelSchema(model.Model())
	if err != nil {
		return nil, fmt.Errorf("compile ffn MIL: parse schema: %w", err)
	}
	if len(schema.Inputs) == 0 || len(schema.Outputs) == 0 {
		return nil, fmt.Errorf("compile ffn MIL: schema has %d inputs, %d outputs", len(schema.Inputs), len(schema.Outputs))
	}

	input, err := newIOSurfaceFloat32WithLayout(schema.Inputs[0])
	if err != nil {
		return nil, fmt.Errorf("compile ffn MIL: input surface: %w", err)
	}
	output, err := newIOSurfaceFloat32WithLayout(schema.Outputs[0])
	if err != nil {
		input.Close()
		return nil, fmt.Errorf("compile ffn MIL: output surface: %w", err)
	}

	return &FFNMILModel{
		Model:  model,
		Input:  input,
		Output: output,
		Dim:    dim,
	}, nil
}
