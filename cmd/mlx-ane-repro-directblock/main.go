//go:build darwin && ane_appleneuralengine

package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/tmc/mlx-go-lm/mlxlm/models"
	mlxgoane "github.com/tmc/mlx-go-ane"
	"github.com/tmc/mlx-go/modelir"
)

func main() {
	var (
		modelPath     string
		mirrorRoot    string
		maxLayers     int
		maxSeqLen     int
		skipFinalNorm bool
	)
	flag.StringVar(&modelPath, "model", "", "Path to a local Qwen3.5 checkpoint")
	flag.StringVar(&mirrorRoot, "mirror-root", "", "Optional mirror root to apply before direct-block compile")
	flag.IntVar(&maxLayers, "max-layers", 1, "Number of lowered full-attention layers to include")
	flag.IntVar(&maxSeqLen, "max-seq-len", 256, "Maximum decode sequence length")
	flag.BoolVar(&skipFinalNorm, "skip-final-norm", true, "Lower raw hidden output instead of final norm output")
	flag.Parse()

	if modelPath == "" {
		fmt.Fprintln(os.Stderr, "missing -model")
		os.Exit(2)
	}
	if mirrorRoot != "" {
		mlxgoane.SetModelMirrorRoot(mirrorRoot)
		defer mlxgoane.SetModelMirrorRoot("")
	}

	model, _, err := models.LoadModel(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	weightFiles, err := models.DiscoverWeightFiles(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "discover weights: %v\n", err)
		os.Exit(1)
	}
	if err := model.LoadWeights(weightFiles...); err != nil {
		fmt.Fprintf(os.Stderr, "load weights: %v\n", err)
		os.Exit(1)
	}

	lowerable, ok := model.(models.DecodeModelIRLowerable)
	if !ok {
		fmt.Fprintf(os.Stderr, "model %T does not implement DecodeModelIRLowerable\n", model)
		os.Exit(1)
	}
	cfg := model.Config()
	if cfg == nil {
		fmt.Fprintln(os.Stderr, "model.Config() is nil")
		os.Exit(1)
	}
	if maxSeqLen <= 0 || maxSeqLen > 256 {
		maxSeqLen = 256
	}
	prog, err := lowerable.LowerDecodeModelIR(models.DecodeModelIROptions{
		MaxLayers:     maxLayers,
		MaxSeqLen:     maxSeqLen,
		IncludeLMHead: false,
		SkipFinalNorm: skipFinalNorm,
		StatefulKV:    true,
		AttentionMask: false,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "lower decode modelir: %v\n", err)
		os.Exit(1)
	}
	selectedLayers := decodeProgramSelectedLayers(prog)
	if selectedLayers <= 0 {
		if maxLayers > 0 {
			selectedLayers = maxLayers
		} else {
			selectedLayers = cfg.NumLayers
		}
	}

	reifyOpts := mlxgoane.ReifyOptions{
		TransformerConfig: mlxgoane.MILTransformerConfig{
			NumLayers:          selectedLayers,
			MaxSeqLen:          maxSeqLen,
			KVCacheState:       true,
			KVCacheMaxLen:      maxSeqLen,
			AttentionMaskInput: false,
		},
		RequestedLayers: selectedLayers,
		SelectedLayers:  selectedLayers,
	}
	draft, reified, err := mlxgoane.NewANEDraftModelFromModelIRProgram(
		prog,
		reifyOpts,
		cfg.HiddenSize,
		cfg.VocabSize,
		cfg.HiddenSize,
		nil,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "compile direct block: %v\n", err)
		os.Exit(1)
	}
	defer draft.Close()

	profile := ""
	for _, d := range reified.Diagnostics {
		if d.Code == "compile_fallback" {
			profile = d.Message
			break
		}
	}
	fmt.Printf("compile succeeded\n")
	fmt.Printf("mirror_root=%q\n", mirrorRoot)
	fmt.Printf("selected_layers=%d\n", reified.SelectedLayers)
	fmt.Printf("diagnostic=%q\n", profile)
}

func decodeProgramSelectedLayers(prog *modelir.Program) int {
	if prog == nil {
		return 0
	}
	return decodeSelectedLayers(prog.HighLevel, prog.Entry)
}

func decodeSelectedLayers(highLevel map[string]string, entry string) int {
	summary := highLevel[entry]
	const key = "full_attention_layers="
	idx := strings.Index(summary, key)
	if idx < 0 {
		return 0
	}
	start := idx + len(key)
	end := start
	for end < len(summary) && summary[end] >= '0' && summary[end] <= '9' {
		end++
	}
	if end == start {
		return 0
	}
	n, err := strconv.Atoi(summary[start:end])
	if err != nil {
		return 0
	}
	return n
}
