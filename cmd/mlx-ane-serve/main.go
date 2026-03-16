package main

import (
	"os"

	"github.com/tmc/mlx-go-lm/lmserve"
	"github.com/tmc/mlx-go-ane/internal/cmdwrap"
	_ "github.com/tmc/mlx-go-ane/register"
)

var aneServeFlags = []cmdwrap.FlagSpec{
	{Name: "ane-speculative", Env: "MLXGO_ANE_SPECULATIVE", Usage: "Route speculative linear ops to ANE: off, draft-prefill, target-prefill, both-prefill, draft-all, target-all, both-all", Kind: cmdwrap.StringFlag},
	{Name: "ane-speculative-min-seq", Env: "MLXGO_ANE_SPECULATIVE_MIN_SEQ", Usage: "Minimum sequence length required before speculative calls are routed to ANE", Kind: cmdwrap.IntFlag},
	{Name: "ane-forward", Env: "MLXGO_ANE_FORWARD", Usage: "Route standard target forward linear ops to ANE: off, prefill, all", Kind: cmdwrap.StringFlag},
	{Name: "ane-forward-min-seq", Env: "MLXGO_ANE_FORWARD_MIN_SEQ", Usage: "Minimum sequence length required before standard target forward calls are routed to ANE", Kind: cmdwrap.IntFlag},
	{Name: "ane-decode-plane", Env: "MLXGO_ANE_DECODE_PLANE", Usage: "Decode-plane backend: off or qwen35", Kind: cmdwrap.StringFlag},
	{Name: "ane-decode-cache", Env: "MLXGO_ANE_DECODE_CACHE", Usage: "Directory for ANE decode-plane artifacts", Kind: cmdwrap.StringFlag},
	{Name: "ane-runtime-policy", Env: "MLXGO_ANE_RUNTIME_POLICY", Usage: "ANE runtime policy: auto, prefer-bridge, prefer-inmemory", Kind: cmdwrap.StringFlag},
	{Name: "ane-routing-cache", Env: "MLXGO_ANE_ROUTING_CACHE", Usage: "Enable ANE route cache: on or off", Kind: cmdwrap.StringFlag},
}

func main() {
	prepared, err := cmdwrap.Prepare(os.Args[1:], aneServeFlags)
	if err != nil {
		_, _ = os.Stderr.WriteString("Error: " + err.Error() + "\n")
		os.Exit(2)
	}
	if prepared.Help {
		cmdwrap.PrintHelp(aneServeFlags)
	}
	if err := cmdwrap.ApplyEnv(prepared.Env); err != nil {
		_, _ = os.Stderr.WriteString("Error: " + err.Error() + "\n")
		os.Exit(2)
	}
	os.Args = append([]string{"mlx-ane-serve"}, prepared.Args...)
	lmserve.Main()
}
