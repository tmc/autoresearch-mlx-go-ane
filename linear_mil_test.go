package mlxgoane

import (
	"strings"
	"testing"
)

func TestLinearMILTextRejectsInvalidDims(t *testing.T) {
	tests := []struct {
		name        string
		batch       int
		inDim       int
		outDim      int
		wantErrPart string
	}{
		{name: "batch", batch: 0, inDim: 4, outDim: 8, wantErrPart: "batch"},
		{name: "inDim", batch: 1, inDim: 0, outDim: 8, wantErrPart: "inDim"},
		{name: "outDim", batch: 1, inDim: 4, outDim: 0, wantErrPart: "outDim"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := linearMILText(tc.batch, tc.inDim, tc.outDim)
			if err == nil {
				t.Fatalf("linearMILText(%d,%d,%d) error=nil", tc.batch, tc.inDim, tc.outDim)
			}
			if !strings.Contains(err.Error(), tc.wantErrPart) {
				t.Fatalf("error=%q want substring %q", err, tc.wantErrPart)
			}
		})
	}
}

func TestLinearMILTextIncludesShapeAndOps(t *testing.T) {
	got, err := linearMILText(2, 3, 4)
	if err != nil {
		t.Fatalf("linearMILText: %v", err)
	}
	parts := []string{
		"program(1.3)",
		"func main<ios18>(tensor<fp32, [1, 3, 1, 2]> x)",
		"tensor<fp16, [4,3,1,1]>",
		"conv(",
		"BLOBFILE",
		"@model_path/weights/weight.bin",
		"offset=uint64(64)",
	}
	for _, part := range parts {
		if !strings.Contains(got, part) {
			t.Fatalf("MIL missing %q:\n%s", part, got)
		}
	}
	if strings.Contains(got, "%wt:") {
		t.Fatalf("MIL unexpectedly declares runtime wt input:\n%s", got)
	}
}
