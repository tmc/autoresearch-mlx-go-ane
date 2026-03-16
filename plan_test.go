
package mlxgoane

import (
	"fmt"

	"github.com/tmc/mlx-go/exp/mlxfntxt"
)

func ExampleAnalyzeArchive() {
	ar := &mlxfntxt.Archive{
		Functions: []mlxfntxt.Function{
			{
				Name: "fn0",
				Ops: []mlxfntxt.Op{
					{Name: "Matmul"},
					{Name: "RMSNorm"},
				},
			},
		},
	}
	rep, err := AnalyzeArchive(ar)
	if err != nil {
		panic(err)
	}
	fmt.Printf("ops=%d direct=%d lowered=%d unsupported=%d unknown=%d\n",
		rep.TotalOps, rep.DirectOps, rep.LoweredOps, rep.UnsupportedOps, rep.UnknownOps)
	// Output:
	// ops=2 direct=1 lowered=1 unsupported=0 unknown=0
}
