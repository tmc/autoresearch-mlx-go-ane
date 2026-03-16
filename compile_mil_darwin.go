//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"fmt"
	"strings"

	"github.com/tmc/apple/private/appleneuralengine"
)

// CompiledMILModel is a compiled in-memory MIL model with layout-aware
// IOSurfaces ready for evaluation via SurfaceEvalPlan.
type CompiledMILModel struct {
	Model   appleneuralengine.ANEInMemoryModel
	Inputs  []*IOSurfaceFloat32
	Outputs []*IOSurfaceFloat32
	States  []*IOSurfaceFloat32
	Schema  compiledModelSchema
}

// Close releases the model and all IOSurfaces.
func (m *CompiledMILModel) Close() {
	if m == nil {
		return
	}
	for _, s := range m.Inputs {
		if s != nil {
			s.Close()
		}
	}
	for _, s := range m.Outputs {
		if s != nil {
			s.Close()
		}
	}
	for _, s := range m.States {
		if s != nil {
			s.Close()
		}
	}
	if m.Model.ID != 0 {
		_ = callObjCBoolWithNSError(
			"compiled mil model unload",
			m.Model.ID,
			"unloadWithQoS:error:",
			defaultANEQoS,
		)
	}
}

// CompileFromReified compiles a ReifiedMIL artifact bundle to an in-memory
// ANE model with layout-aware IOSurfaces.
//
// This is the integration point between the new modelir/target/mil pipeline
// (EmitFunction + transforms) and the ANE runtime. ReifiedMIL is the universal
// artifact format produced by both ReifyToANEMIL and the new ReifyFunction.
func CompileFromReified(reified ReifiedMIL) (*CompiledMILModel, error) {
	return CompileMILText(reified.MILText, cloneModelWeightFiles(reified.WeightFiles))
}

// CompileMILText compiles raw MIL text and weight files to an in-memory ANE
// model with layout-aware IOSurfaces.
//
// Use this when you have MIL text from EmitFunction directly (without going
// through the full ReifiedMIL pipeline).
func CompileMILText(milText string, files []ModelWeightFile) (*CompiledMILModel, error) {
	if strings.TrimSpace(milText) == "" {
		return nil, fmt.Errorf("compile mil: mil text is empty")
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("compile mil: weight files are empty")
	}

	model, err := buildModelFromMILTextWithDescriptorFallback("compile mil", milText, files)
	if err != nil {
		return nil, fmt.Errorf("compile mil: %w", err)
	}

	schema, err := parseCompiledModelSchema(model.Model())
	if err != nil {
		return nil, fmt.Errorf("compile mil: parse schema: %w", err)
	}

	inputs, err := allocSurfaces(schema.Inputs)
	if err != nil {
		return nil, fmt.Errorf("compile mil: input surfaces: %w", err)
	}
	outputs, err := allocSurfaces(schema.Outputs)
	if err != nil {
		closeSurfaces(inputs)
		return nil, fmt.Errorf("compile mil: output surfaces: %w", err)
	}
	states, err := allocSurfaces(schema.States)
	if err != nil {
		closeSurfaces(inputs)
		closeSurfaces(outputs)
		return nil, fmt.Errorf("compile mil: state surfaces: %w", err)
	}

	return &CompiledMILModel{
		Model:   model,
		Inputs:  inputs,
		Outputs: outputs,
		States:  states,
		Schema:  schema,
	}, nil
}

func allocSurfaces(layouts []compiledTensorLayout) ([]*IOSurfaceFloat32, error) {
	surfaces := make([]*IOSurfaceFloat32, len(layouts))
	for i, layout := range layouts {
		s, err := newIOSurfaceFloat32WithLayout(layout)
		if err != nil {
			closeSurfaces(surfaces[:i])
			return nil, fmt.Errorf("surface[%d] %q: %w", i, layout.Name, err)
		}
		surfaces[i] = s
	}
	return surfaces, nil
}

func closeSurfaces(surfaces []*IOSurfaceFloat32) {
	for _, s := range surfaces {
		if s != nil {
			s.Close()
		}
	}
}
