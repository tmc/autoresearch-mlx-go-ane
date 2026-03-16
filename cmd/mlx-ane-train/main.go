package main

import (
	"os"

	"github.com/tmc/mlx-go-lm/lmtrain"
	"github.com/tmc/mlx-go-ane/internal/cmdwrap"
	_ "github.com/tmc/mlx-go-ane/register"
)

var aneTrainFlags = []cmdwrap.FlagSpec{
	{Name: "ane-forward", Env: "MLXGO_ANE_FORWARD", Usage: "Route training forward linear ops to ANE: off or all", Kind: cmdwrap.StringFlag},
	{Name: "ane-route-profile", Env: "MLXGO_ANE_ROUTE_PROFILE", Usage: "ANE linear routing profile: balanced, conservative, or aggressive", Kind: cmdwrap.StringFlag},
	{Name: "ane-allow-fallback", Env: "MLXGO_ANE_ALLOW_FALLBACK", Usage: "Allow MLX fallback when ANE training forward routing declines or fails", Kind: cmdwrap.BoolFlag},
}

func main() {
	prepared, err := cmdwrap.Prepare(os.Args[1:], aneTrainFlags)
	if err != nil {
		_, _ = os.Stderr.WriteString("Error: " + err.Error() + "\n")
		os.Exit(2)
	}
	if prepared.Help {
		cmdwrap.PrintHelp(aneTrainFlags)
	}
	if err := cmdwrap.ApplyEnv(prepared.Env); err != nil {
		_, _ = os.Stderr.WriteString("Error: " + err.Error() + "\n")
		os.Exit(2)
	}
	os.Args = append([]string{"mlx-ane-train"}, prepared.Args...)
	lmtrain.Main()
}
