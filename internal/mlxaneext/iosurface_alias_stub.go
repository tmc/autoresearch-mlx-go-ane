//go:build !darwin || !cgo

package mlxaneext

import (
	"fmt"

	"github.com/tmc/mlx-go/mlxc"
)

func ImportIOSurfaceFloat32(uint64, []int32) (mlxc.Array, error) {
	return mlxc.Array{}, fmt.Errorf("import iosurface float32: unavailable on this platform")
}

func ImportIOSurfaceFloat32ReadOnly(uint64, []int32) (mlxc.Array, error) {
	return mlxc.Array{}, fmt.Errorf("import iosurface float32 read-only: unavailable on this platform")
}
