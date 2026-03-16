package mlxgoane

import "fmt"

// ffnSwiGLUConvConst holds shared conv parameter declarations for the FFN MIL.
const ffnSwiGLUConvConst = `        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
`

const (
	ffnGateWeightPath = "@model_path/weights/gate.bin"
	ffnUpWeightPath   = "@model_path/weights/up.bin"
	ffnDownWeightPath = "@model_path/weights/down.bin"
)

// ffnMILText generates MIL source for a SwiGLU FFN block:
//
//	output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
//
// All projections use conv with fp16 weights to avoid float16 accumulation
// overflow that occurs with Espresso inner_product at large dimensions.
//
// Input shape:  [1, dim, 1, 1]
// Output shape: [1, dim, 1, 1]
func ffnMILText(dim, hidden int) (string, error) {
	if dim <= 0 {
		return "", fmt.Errorf("ffn MIL: invalid dim=%d", dim)
	}
	if hidden <= 0 {
		return "", fmt.Errorf("ffn MIL: invalid hidden=%d", hidden)
	}

	return fmt.Sprintf(
		`%s    func main<ios18>(tensor<fp32, [1, %d, 1, 1]> x) {
%s        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp16, [1,%d,1,1]> x16 = cast(dtype=to_fp16,x=x)[name=string("cast_in")];

        tensor<fp16, [%d,%d,1,1]> Wgate = const()[name=string("Wgate"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string("%s"), offset=uint64(%d)))];
        tensor<fp16, [1,%d,1,1]> gate = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wgate,x=x16)[name=string("gate_proj")];

        tensor<fp16, [%d,%d,1,1]> Wup = const()[name=string("Wup"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string("%s"), offset=uint64(%d)))];
        tensor<fp16, [1,%d,1,1]> up = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wup,x=x16)[name=string("up_proj")];

        tensor<fp16, [1,%d,1,1]> gate_sig = sigmoid(x=gate)[name=string("gate_sigmoid")];
        tensor<fp16, [1,%d,1,1]> gate_act = mul(x=gate,y=gate_sig)[name=string("gate_silu")];
        tensor<fp16, [1,%d,1,1]> mix = mul(x=gate_act,y=up)[name=string("ffn_mix")];

        tensor<fp16, [%d,%d,1,1]> Wdown = const()[name=string("Wdown"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string("%s"), offset=uint64(%d)))];
        tensor<fp16, [1,%d,1,1]> down16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wdown,x=mix)[name=string("down_proj")];

        tensor<fp32, [1,%d,1,1]> y = cast(dtype=to_fp32,x=down16)[name=string("cast_out")];
    } -> (y);
}
`,
		linearMILHeader,
		dim,
		ffnSwiGLUConvConst,
		// cast_in
		dim,
		// Wgate: [hidden, dim, 1, 1]
		hidden, dim, hidden, dim, ffnGateWeightPath, linearWeightBlobOffset,
		// gate conv output: [1, hidden, 1, 1]
		hidden,
		// Wup: [hidden, dim, 1, 1]
		hidden, dim, hidden, dim, ffnUpWeightPath, linearWeightBlobOffset,
		// up conv output: [1, hidden, 1, 1]
		hidden,
		// SiLU: sigmoid + mul
		hidden,
		hidden,
		// mix: gate_act * up
		hidden,
		// Wdown: [dim, hidden, 1, 1]
		dim, hidden, dim, hidden, ffnDownWeightPath, linearWeightBlobOffset,
		// down conv output: [1, dim, 1, 1]
		dim,
		// cast_out
		dim,
	), nil
}

// BuildFFNMILArtifacts generates MIL text and weight files for a SwiGLU FFN.
//
// Weights are expected in [out, in] row-major order:
//   - gate (w1): [hidden, dim]
//   - up   (w3): [hidden, dim]
//   - down (w2): [dim, hidden]
func BuildFFNMILArtifacts(dim, hidden int, gate, up, down []float32) (milText string, files []ModelWeightFile, err error) {
	if len(gate) != hidden*dim {
		return "", nil, fmt.Errorf("ffn MIL: gate weight len=%d want=%d", len(gate), hidden*dim)
	}
	if len(up) != hidden*dim {
		return "", nil, fmt.Errorf("ffn MIL: up weight len=%d want=%d", len(up), hidden*dim)
	}
	if len(down) != dim*hidden {
		return "", nil, fmt.Errorf("ffn MIL: down weight len=%d want=%d", len(down), dim*hidden)
	}

	milText, err = ffnMILText(dim, hidden)
	if err != nil {
		return "", nil, err
	}

	gateBlob, err := buildLinearWeightsBlob(gate, hidden, dim)
	if err != nil {
		return "", nil, fmt.Errorf("ffn MIL: gate weights: %w", err)
	}
	upBlob, err := buildLinearWeightsBlob(up, hidden, dim)
	if err != nil {
		return "", nil, fmt.Errorf("ffn MIL: up weights: %w", err)
	}
	downBlob, err := buildLinearWeightsBlob(down, dim, hidden)
	if err != nil {
		return "", nil, fmt.Errorf("ffn MIL: down weights: %w", err)
	}

	files = []ModelWeightFile{
		{Path: ffnGateWeightPath, Blob: gateBlob},
		{Path: ffnUpWeightPath, Blob: upBlob},
		{Path: ffnDownWeightPath, Blob: downBlob},
	}
	return milText, files, nil
}
