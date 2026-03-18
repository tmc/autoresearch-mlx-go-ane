module github.com/tmc/mlx-go-ane

go 1.25.6

require (
	github.com/ebitengine/purego v0.10.0
	github.com/tmc/apple v0.3.6
	github.com/tmc/mlx-go v0.0.0
	github.com/tmc/mlx-go-lm v0.0.0
	google.golang.org/protobuf v1.36.11
)

require (
	golang.org/x/image v0.36.0 // indirect
	golang.org/x/tools v0.43.0 // indirect
)

replace github.com/tmc/mlx-go => ../mlx-go

replace github.com/tmc/mlx-go-lm => ../mlx-go/examples/mlx-go-lm
