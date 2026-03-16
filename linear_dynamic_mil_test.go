package mlxgoane

import (
	"strings"
	"testing"
)

func TestDynamicLinearMILTextRejectsInvalidDims(t *testing.T) {
	tests := []struct {
		name        string
		batch       int
		inDim       int
		outDim      int
		wantErrPart string
	}{
		{name: "batch", batch: 0, inDim: 4, outDim: 8, wantErrPart: "batch"},
		{name: "in_dim", batch: 1, inDim: 0, outDim: 8, wantErrPart: "inDim"},
		{name: "out_dim", batch: 1, inDim: 4, outDim: 0, wantErrPart: "outDim"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dynamicLinearMILText(tc.batch, tc.inDim, tc.outDim)
			if err == nil {
				t.Fatalf("dynamicLinearMILText(%d,%d,%d) error=nil", tc.batch, tc.inDim, tc.outDim)
			}
			if !strings.Contains(err.Error(), tc.wantErrPart) {
				t.Fatalf("error=%q want substring %q", err, tc.wantErrPart)
			}
		})
	}
}

func TestDynamicLinearMILTextIncludesInputsAndMatmul(t *testing.T) {
	got, err := dynamicLinearMILText(2, 3, 4)
	if err != nil {
		t.Fatalf("dynamicLinearMILText: %v", err)
	}
	parts := []string{
		"program(1.3)",
		"func main<ios18>(tensor<fp32, [1, 1, 2, 3]> x_in, tensor<fp32, [1, 1, 3, 4]> w_in)",
		"cast_x",
		"cast_w",
		"matmul(",
		"transpose_x=tx",
		"transpose_y=ty",
		"tensor<fp32, [1,1,2,4]> y = cast",
	}
	for _, part := range parts {
		if !strings.Contains(got, part) {
			t.Fatalf("MIL missing %q:\n%s", part, got)
		}
	}
	if strings.Contains(got, "BLOBFILE") {
		t.Fatalf("dynamic MIL unexpectedly bakes weights:\n%s", got)
	}
}

func TestDynamicLinearWeightsHostToANE(t *testing.T) {
	got, err := dynamicLinearWeightsHostToANE([]float32{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)
	if err != nil {
		t.Fatalf("dynamicLinearWeightsHostToANE: %v", err)
	}
	want := []float32{
		1, 4,
		2, 5,
		3, 6,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want=%g (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestDynamicLinearPackedMILTextIncludesSliceAndReshape(t *testing.T) {
	got, err := dynamicLinearPackedMILText(2, 3, 4)
	if err != nil {
		t.Fatalf("dynamicLinearPackedMILText: %v", err)
	}
	parts := []string{
		"func main<ios18>(tensor<fp32, [1, 1, 6, 3]> packed_in)",
		"packed16",
		"slice_by_size(x=packed16, begin=bx, size=sx)",
		"slice_by_size(x=packed16, begin=bw, size=sw)",
		"tensor<int32, [4]> sx",
		"tensor<int32, [4]> sw",
		"matmul(",
		"transpose_x=tx",
		"transpose_y=ty",
		"tensor<fp32, [1,1,2,4]> y = cast",
	}
	for _, part := range parts {
		if !strings.Contains(got, part) {
			t.Fatalf("packed MIL missing %q:\n%s", part, got)
		}
	}
}

func TestDynamicLinearPackedHostToANE(t *testing.T) {
	got, err := dynamicLinearPackedHostToANE(
		[]float32{10, 11, 12, 13, 14, 15},
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
		2,
		3,
		2,
	)
	if err != nil {
		t.Fatalf("dynamicLinearPackedHostToANE: %v", err)
	}
	want := []float32{
		10, 11, 12, 13, 14, 15,
		1, 2, 3, 4, 5, 6,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want=%g (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestDynamicLinearPaddedDims(t *testing.T) {
	inDim, outDim := dynamicLinearPaddedDims(32, 3)
	if inDim != 32 || outDim != 16 {
		t.Fatalf("dynamicLinearPaddedDims(32,3)=(%d,%d) want (32,16)", inDim, outDim)
	}
}

func TestDynamicLinearPackedHostToANEWithPadding(t *testing.T) {
	got, err := dynamicLinearPackedHostToANEWithPadding(
		[]float32{10, 11, 12, 13, 14, 15},
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
		2,
		3,
		2,
		4,
		4,
	)
	if err != nil {
		t.Fatalf("dynamicLinearPackedHostToANEWithPadding: %v", err)
	}
	want := []float32{
		10, 11, 12, 0,
		13, 14, 15, 0,
		1, 2, 3, 0,
		4, 5, 6, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want=%g (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestDynamicLinearPackedHostToANEWithPaddingInto(t *testing.T) {
	got := make([]float32, 24)
	for i := range got {
		got[i] = -1
	}
	err := dynamicLinearPackedHostToANEWithPaddingInto(
		got,
		[]float32{10, 11, 12, 13, 14, 15},
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
		2,
		3,
		2,
		4,
		4,
	)
	if err != nil {
		t.Fatalf("dynamicLinearPackedHostToANEWithPaddingInto: %v", err)
	}
	want := []float32{
		10, 11, 12, 0,
		13, 14, 15, 0,
		1, 2, 3, 0,
		4, 5, 6, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want=%g (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestDynamicLinearPackedHostToANEWithPaddingIntoAligned(t *testing.T) {
	got := make([]float32, 12)
	for i := range got {
		got[i] = -1
	}
	err := dynamicLinearPackedHostToANEWithPaddingInto(
		got,
		[]float32{10, 11, 12, 13, 14, 15},
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
		2,
		3,
		2,
		3,
		2,
	)
	if err != nil {
		t.Fatalf("dynamicLinearPackedHostToANEWithPaddingInto aligned: %v", err)
	}
	want := []float32{
		10, 11, 12, 13, 14, 15,
		1, 2, 3, 4, 5, 6,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want=%g (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestDynamicLinearOutputANEToHostWithPadding(t *testing.T) {
	got, err := dynamicLinearOutputANEToHostWithPadding(
		[]float32{
			1, 2, 3, 0,
			4, 5, 6, 0,
		},
		2,
		3,
		4,
	)
	if err != nil {
		t.Fatalf("dynamicLinearOutputANEToHostWithPadding: %v", err)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want=%g (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestDynamicLinearOutputANEToHostWithPaddingInto(t *testing.T) {
	got := make([]float32, 6)
	err := dynamicLinearOutputANEToHostWithPaddingInto(
		got,
		[]float32{
			1, 2, 3, 0,
			4, 5, 6, 0,
		},
		2,
		3,
		4,
	)
	if err != nil {
		t.Fatalf("dynamicLinearOutputANEToHostWithPaddingInto: %v", err)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want=%g (full=%v)", i, got[i], want[i], got)
		}
	}
}
