package mlxgoane

import "fmt"

// dynamicLinearMILText returns a MIL source template for y = x @ w where both
// activations and weights are runtime inputs.
//
// Inputs:
//   - x shape [batch, inDim], materialized as tensor<fp32, [1, 1, batch, inDim]>
//   - w shape [outDim, inDim], materialized as tensor<fp32, [1, 1, inDim, outDim]>
//
// Output:
//   - y shape [batch, outDim], materialized as tensor<fp32, [1, 1, batch, outDim]>
func dynamicLinearMILText(batch, inDim, outDim int) (string, error) {
	if batch <= 0 {
		return "", fmt.Errorf("dynamic linear MIL: invalid batch=%d", batch)
	}
	if inDim <= 0 {
		return "", fmt.Errorf("dynamic linear MIL: invalid inDim=%d", inDim)
	}
	if outDim <= 0 {
		return "", fmt.Errorf("dynamic linear MIL: invalid outDim=%d", outDim)
	}

	return fmt.Sprintf(
		`%s    func main<ios18>(tensor<fp32, [1, 1, %d, %d]> x_in, tensor<fp32, [1, 1, %d, %d]> w_in) {
        string to_fp16 = const()[name=string("to_fp16"), val=string("fp16")];
        tensor<fp16, [1,1,%d,%d]> x = cast(dtype=to_fp16, x=x_in)[name=string("cast_x")];
        tensor<fp16, [1,1,%d,%d]> w = cast(dtype=to_fp16, x=w_in)[name=string("cast_w")];
        bool tx = const()[name=string("tx"), val=bool(false)];
        bool ty = const()[name=string("ty"), val=bool(false)];
        tensor<fp16, [1,1,%d,%d]> y16 = matmul(transpose_x=tx, transpose_y=ty, x=x, y=w)[name=string("mm")];
        string to_fp32 = const()[name=string("to_fp32"), val=string("fp32")];
        tensor<fp32, [1,1,%d,%d]> y = cast(dtype=to_fp32, x=y16)[name=string("cast_out")];
    } -> (y);
}
`,
		linearMILHeader,
		batch, inDim,
		inDim, outDim,
		batch, inDim,
		inDim, outDim,
		batch, outDim,
		batch, outDim,
	), nil
}

func dynamicLinearInputHostToANE(x []float32, batch, inDim int) ([]float32, error) {
	if batch <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("dynamic linear input layout: invalid dims batch=%d in=%d", batch, inDim)
	}
	if len(x) != batch*inDim {
		return nil, fmt.Errorf("dynamic linear input layout: len=%d want=%d", len(x), batch*inDim)
	}
	out := make([]float32, len(x))
	copy(out, x)
	return out, nil
}

func dynamicLinearWeightsHostToANE(w []float32, outDim, inDim int) ([]float32, error) {
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("dynamic linear weight layout: invalid dims out=%d in=%d", outDim, inDim)
	}
	if len(w) != outDim*inDim {
		return nil, fmt.Errorf("dynamic linear weight layout: len=%d want=%d", len(w), outDim*inDim)
	}

	out := make([]float32, len(w))
	for o := 0; o < outDim; o++ {
		src := w[o*inDim : (o+1)*inDim]
		for i, v := range src {
			out[i*outDim+o] = v
		}
	}
	return out, nil
}

func dynamicLinearOutputANEToHost(y []float32, batch, outDim int) ([]float32, error) {
	if batch <= 0 || outDim <= 0 {
		return nil, fmt.Errorf("dynamic linear output layout: invalid dims batch=%d out=%d", batch, outDim)
	}
	if len(y) != batch*outDim {
		return nil, fmt.Errorf("dynamic linear output layout: len=%d want=%d", len(y), batch*outDim)
	}
	out := make([]float32, len(y))
	copy(out, y)
	return out, nil
}

// dynamicLinearPackedMILText returns a single-input MIL source template for
// y = x @ w where x and w are packed into one runtime tensor.
//
// Input:
//   - packed_in shape [1, 1, batch+outDim, inDim]
//   - packed layout: x rows first, then w rows in host [outDim, inDim] order
//
// Output:
//   - y shape [batch, outDim], materialized as tensor<fp32, [1, 1, batch, outDim]>
func dynamicLinearPackedMILText(batch, inDim, outDim int) (string, error) {
	if batch <= 0 {
		return "", fmt.Errorf("dynamic packed linear MIL: invalid batch=%d", batch)
	}
	if inDim <= 0 {
		return "", fmt.Errorf("dynamic packed linear MIL: invalid inDim=%d", inDim)
	}
	if outDim <= 0 {
		return "", fmt.Errorf("dynamic packed linear MIL: invalid outDim=%d", outDim)
	}

	return fmt.Sprintf(
		`%s    func main<ios18>(tensor<fp32, [1, 1, %d, %d]> packed_in) {
        string to_fp16 = const()[name=string("to_fp16"), val=string("fp16")];
        tensor<fp16, [1,1,%d,%d]> packed16 = cast(dtype=to_fp16, x=packed_in)[name=string("packed16")];
        tensor<int32, [4]> bx = const()[name=string("bx"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [4]> sx = const()[name=string("sx"), val=tensor<int32, [4]>([1,1,%d,%d])];
        tensor<fp16, [1,1,%d,%d]> x = slice_by_size(x=packed16, begin=bx, size=sx)[name=string("x_slice")];
        tensor<int32, [4]> bw = const()[name=string("bw"), val=tensor<int32, [4]>([0,0,%d,0])];
        tensor<int32, [4]> sw = const()[name=string("sw"), val=tensor<int32, [4]>([1,1,%d,%d])];
        tensor<fp16, [1,1,%d,%d]> w = slice_by_size(x=packed16, begin=bw, size=sw)[name=string("w_slice")];
        bool tx = const()[name=string("tx"), val=bool(false)];
        bool ty = const()[name=string("ty"), val=bool(true)];
        tensor<fp16, [1,1,%d,%d]> y16 = matmul(transpose_x=tx, transpose_y=ty, x=x, y=w)[name=string("mm")];
        string to_fp32 = const()[name=string("to_fp32"), val=string("fp32")];
        tensor<fp32, [1,1,%d,%d]> y = cast(dtype=to_fp32, x=y16)[name=string("cast_out")];
    } -> (y);
}
`,
		linearMILHeader,
		batch+outDim, inDim,
		batch+outDim, inDim,
		batch, inDim,
		batch, inDim,
		batch,
		outDim, inDim,
		outDim, inDim,
		batch, outDim,
		batch, outDim,
	), nil
}

func dynamicLinearPackedHostToANE(x, w []float32, batch, inDim, outDim int) ([]float32, error) {
	if batch <= 0 || inDim <= 0 || outDim <= 0 {
		return nil, fmt.Errorf(
			"dynamic packed linear layout: invalid dims batch=%d in=%d out=%d",
			batch,
			inDim,
			outDim,
		)
	}
	if len(x) != batch*inDim {
		return nil, fmt.Errorf("dynamic packed linear layout: x len=%d want=%d", len(x), batch*inDim)
	}
	if len(w) != outDim*inDim {
		return nil, fmt.Errorf("dynamic packed linear layout: w len=%d want=%d", len(w), outDim*inDim)
	}

	out := make([]float32, len(x)+len(w))
	copy(out, x)
	copy(out[len(x):], w)
	return out, nil
}

func dynamicLinearPaddedDims(inDim, outDim int) (paddedInDim, paddedOutDim int) {
	return alignDimUp(inDim, 16), alignDimUp(outDim, 16)
}

func dynamicLinearPackedHostToANEWithPadding(
	x, w []float32,
	batch, inDim, outDim, paddedInDim, paddedOutDim int,
) ([]float32, error) {
	out := make([]float32, batch*paddedInDim+paddedOutDim*paddedInDim)
	if err := dynamicLinearPackedHostToANEWithPaddingInto(
		out,
		x,
		w,
		batch,
		inDim,
		outDim,
		paddedInDim,
		paddedOutDim,
	); err != nil {
		return nil, err
	}
	return out, nil
}

func dynamicLinearPackedHostToANEWithPaddingInto(
	out, x, w []float32,
	batch, inDim, outDim, paddedInDim, paddedOutDim int,
) error {
	if batch <= 0 || inDim <= 0 || outDim <= 0 {
		return fmt.Errorf(
			"dynamic padded linear layout: invalid dims batch=%d in=%d out=%d",
			batch,
			inDim,
			outDim,
		)
	}
	if paddedInDim < inDim || paddedOutDim < outDim {
		return fmt.Errorf(
			"dynamic padded linear layout: padded dims in=%d out=%d smaller than actual in=%d out=%d",
			paddedInDim,
			paddedOutDim,
			inDim,
			outDim,
		)
	}
	if len(x) != batch*inDim {
		return fmt.Errorf("dynamic padded linear layout: x len=%d want=%d", len(x), batch*inDim)
	}
	if len(w) != outDim*inDim {
		return fmt.Errorf("dynamic padded linear layout: w len=%d want=%d", len(w), outDim*inDim)
	}
	want := batch*paddedInDim + paddedOutDim*paddedInDim
	if len(out) != want {
		return fmt.Errorf("dynamic padded linear layout: dst len=%d want=%d", len(out), want)
	}
	if paddedInDim == inDim && paddedOutDim == outDim {
		copy(out, x)
		copy(out[len(x):], w)
		return nil
	}

	clear(out)
	for b := 0; b < batch; b++ {
		copy(out[b*paddedInDim:], x[b*inDim:(b+1)*inDim])
	}
	base := batch * paddedInDim
	for o := 0; o < outDim; o++ {
		rowBase := base + o*paddedInDim
		copy(out[rowBase:rowBase+inDim], w[o*inDim:(o+1)*inDim])
	}
	return nil
}

func dynamicLinearOutputANEToHostWithPadding(y []float32, batch, outDim, paddedOutDim int) ([]float32, error) {
	out := make([]float32, batch*outDim)
	if err := dynamicLinearOutputANEToHostWithPaddingInto(out, y, batch, outDim, paddedOutDim); err != nil {
		return nil, err
	}
	return out, nil
}

func dynamicLinearOutputANEToHostWithPaddingInto(dst, y []float32, batch, outDim, paddedOutDim int) error {
	if batch <= 0 || outDim <= 0 || paddedOutDim <= 0 {
		return fmt.Errorf(
			"dynamic padded linear output layout: invalid dims batch=%d out=%d padded_out=%d",
			batch,
			outDim,
			paddedOutDim,
		)
	}
	if paddedOutDim < outDim {
		return fmt.Errorf(
			"dynamic padded linear output layout: padded_out=%d smaller than out=%d",
			paddedOutDim,
			outDim,
		)
	}
	if len(y) != batch*paddedOutDim {
		return fmt.Errorf(
			"dynamic padded linear output layout: len=%d want=%d",
			len(y),
			batch*paddedOutDim,
		)
	}
	if len(dst) != batch*outDim {
		return fmt.Errorf("dynamic padded linear output layout: dst len=%d want=%d", len(dst), batch*outDim)
	}
	if paddedOutDim == outDim {
		copy(dst, y)
		return nil
	}
	for b := 0; b < batch; b++ {
		copy(dst[b*outDim:(b+1)*outDim], y[b*paddedOutDim:b*paddedOutDim+outDim])
	}
	return nil
}

func alignDimUp(v, multiple int) int {
	if multiple <= 0 || v <= 0 {
		return v
	}
	r := v % multiple
	if r == 0 {
		return v
	}
	return v + multiple - r
}
