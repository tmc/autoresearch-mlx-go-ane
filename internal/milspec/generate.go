package milspec

//go:generate protoc --go_out=. --go_opt=paths=source_relative --go_opt=MMIL.proto=github.com/tmc/mlx-go-ane/internal/milspec MIL.proto
