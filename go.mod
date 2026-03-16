module github.com/tmc/mlx-go-ane

go 1.25.6

require (
	github.com/ebitengine/purego v0.10.0
	github.com/tmc/apple v0.0.0
	github.com/tmc/mlx-go v0.0.0
	github.com/tmc/mlx-go-lm v0.0.0
	golang.org/x/tools v0.43.0
	google.golang.org/protobuf v1.36.11
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/prometheus/client_golang v1.23.2 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.66.1 // indirect
	github.com/prometheus/procfs v0.16.1 // indirect
	go.yaml.in/yaml/v2 v2.4.2 // indirect
	golang.org/x/image v0.36.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
)

replace github.com/tmc/mlx-go => ../mlx-go

replace github.com/tmc/mlx-go-lm => ../mlx-go/examples/mlx-go-lm

replace github.com/ebitengine/purego => github.com/tmc/purego v0.10.0-alpha.2.0.20260207193206-ff6e796b10b1

replace github.com/tmc/apple => /Users/tmc/go/src/github.com/tmc/apple
