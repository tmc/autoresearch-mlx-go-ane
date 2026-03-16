//go:build darwin && ane_appleneuralengine

package mlxgoane

import (
	"fmt"

	"github.com/tmc/mlx-go-ane/internal/milspec"
	"google.golang.org/protobuf/proto"
)

// linearNetworkDescription returns serialized MIL Program protobuf bytes for:
//
//	y = matmul(x, wt)
//
// Inputs:
//   - x shape [batch, inDim]
//   - wt shape [inDim, outDim]
func linearNetworkDescription(batch, inDim, outDim int) ([]byte, error) {
	if batch <= 0 {
		return nil, fmt.Errorf("linear network description: invalid batch=%d", batch)
	}
	if inDim <= 0 {
		return nil, fmt.Errorf("linear network description: invalid inDim=%d", inDim)
	}
	if outDim <= 0 {
		return nil, fmt.Errorf("linear network description: invalid outDim=%d", outDim)
	}

	xType := tensorTypeFP32(batch, inDim)
	wtType := tensorTypeFP32(inDim, outDim)
	yType := tensorTypeFP32(batch, outDim)

	op := &milspec.Operation{
		Type: "matmul",
		Inputs: map[string]*milspec.Argument{
			"x": argName("x"),
			"y": argName("wt"),
		},
		Outputs: []*milspec.NamedValueType{
			{Name: "y", Type: yType},
		},
	}

	block := &milspec.Block{
		Operations: []*milspec.Operation{op},
		Outputs:    []string{"y"},
	}

	fn := &milspec.Function{
		Inputs: []*milspec.NamedValueType{
			{Name: "x", Type: xType},
			{Name: "wt", Type: wtType},
		},
		Opset: "CoreML8",
		BlockSpecializations: map[string]*milspec.Block{
			"CoreML8": block,
		},
	}

	prog := &milspec.Program{
		Version: 1,
		Functions: map[string]*milspec.Function{
			"main": fn,
		},
	}
	buf, err := proto.Marshal(prog)
	if err != nil {
		return nil, fmt.Errorf("linear network description: marshal MIL program: %w", err)
	}
	return buf, nil
}

func tensorTypeFP32(dims ...int) *milspec.ValueType {
	d := make([]*milspec.Dimension, 0, len(dims))
	for _, dim := range dims {
		d = append(d, constDim(dim))
	}
	return &milspec.ValueType{
		Type: &milspec.ValueType_TensorType{
			TensorType: &milspec.TensorType{
				DataType:   milspec.DataType_FLOAT32,
				Rank:       int64(len(dims)),
				Dimensions: d,
			},
		},
	}
}

func constDim(size int) *milspec.Dimension {
	return &milspec.Dimension{
		Dimension: &milspec.Dimension_Constant{
			Constant: &milspec.Dimension_ConstantDimension{
				Size: uint64(size),
			},
		},
	}
}

func argName(name string) *milspec.Argument {
	return &milspec.Argument{
		Arguments: []*milspec.Argument_Binding{
			{
				Binding: &milspec.Argument_Binding_Name{Name: name},
			},
		},
	}
}
