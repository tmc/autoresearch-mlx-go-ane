
package mlxgoane

import (
	"fmt"

	"github.com/tmc/mlx-go/exp/mlxfntxt"
)

// Status values for an op lowering step.
const (
	StatusDirect      = "direct"
	StatusLowered     = "lowered"
	StatusUnsupported = "unsupported"
	StatusUnknown     = "unknown"
)

// Step describes one op in a lowering plan.
type Step struct {
	Function  string
	OpIndex   int
	Primitive string
	Status    string
	MILOpcs   []string
	Note      string
}

// Report summarizes mapping coverage and the generated plan.
type Report struct {
	TotalOps       int
	DirectOps      int
	LoweredOps     int
	UnsupportedOps int
	UnknownOps     int
	Steps          []Step
}

// AnalyzeArchive builds a lowering report from an MLX archive.
func AnalyzeArchive(ar *mlxfntxt.Archive) (*Report, error) {
	if ar == nil {
		return nil, fmt.Errorf("archive is nil")
	}

	report := &Report{}
	for _, fn := range ar.Functions {
		for i, op := range fn.Ops {
			report.TotalOps++

			mapping, ok := mlxfntxt.PrimitiveMILMapping(op.Name)
			step := Step{
				Function:  fn.Name,
				OpIndex:   i,
				Primitive: op.Name,
				Status:    StatusUnknown,
			}
			if ok {
				step.Status = mapping.Status
				step.MILOpcs = append([]string(nil), mapping.MILOpcs...)
				step.Note = mapping.Note
			}

			switch step.Status {
			case StatusDirect:
				report.DirectOps++
			case StatusLowered:
				report.LoweredOps++
			case StatusUnsupported:
				report.UnsupportedOps++
			default:
				report.UnknownOps++
			}

			report.Steps = append(report.Steps, step)
		}
	}
	return report, nil
}

// AnalyzeTextFile parses a .mlxfntxt archive and analyzes lowering coverage.
func AnalyzeTextFile(path string) (*Report, error) {
	ar, err := mlxfntxt.ParseFile(path)
	if err != nil {
		return nil, fmt.Errorf("parse mlxfntxt: %w", err)
	}
	return AnalyzeArchive(ar)
}

// LowerToMIL is the future lowering entrypoint. It currently validates
// coverage and returns an error if any op is unknown/unsupported.
func LowerToMIL(ar *mlxfntxt.Archive) error {
	rep, err := AnalyzeArchive(ar)
	if err != nil {
		return err
	}
	if rep.UnsupportedOps > 0 || rep.UnknownOps > 0 {
		return fmt.Errorf("cannot lower: unsupported=%d unknown=%d", rep.UnsupportedOps, rep.UnknownOps)
	}
	return nil
}
