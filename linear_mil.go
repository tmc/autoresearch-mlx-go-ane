package mlxgoane

import "fmt"

const (
	linearWeightBlobPathInMIL = "@model_path/" + linearWeightBlobPath
)

const linearMILHeader = `program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
`

const linearMILConvConst = `        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
`

// linearMILText returns a MIL source template for y = x @ w^T with baked weights.
//
// Inputs:
//   - x shape [batch, inDim]
//
// Output:
//   - y shape [batch, outDim]
func linearMILText(batch, inDim, outDim int) (string, error) {
	if batch <= 0 {
		return "", fmt.Errorf("linear MIL: invalid batch=%d", batch)
	}
	if inDim <= 0 {
		return "", fmt.Errorf("linear MIL: invalid inDim=%d", inDim)
	}
	if outDim <= 0 {
		return "", fmt.Errorf("linear MIL: invalid outDim=%d", outDim)
	}

	return fmt.Sprintf(
		`%s    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {
%s        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to_fp16,x=x)[name=string("cast_in")];
        tensor<fp16, [%d,%d,1,1]> W = const()[name=string("W"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string("%s"), offset=uint64(%d)))];
        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)[name=string("conv")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to_fp32,x=y16)[name=string("cast_out")];
    } -> (y);
}
`,
		linearMILHeader,
		inDim, batch,
		linearMILConvConst,
		inDim, batch,
		outDim, inDim, outDim, inDim, linearWeightBlobPathInMIL, linearWeightBlobOffset,
		outDim, batch,
		outDim, batch,
	), nil
}
