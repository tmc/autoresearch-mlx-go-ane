package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"sync"

	_ "github.com/tmc/mlx-go-ane/register"
	"github.com/tmc/mlx-go-lm/lmtrain"
	"github.com/tmc/mlx-go-lm/mlxlm/offload"
)

var (
	aneForward       = flag.String("ane-forward", os.Getenv("MLXGO_ANE_FORWARD"), "Route training forward linear ops to ANE: off or all")
	aneRouteProfile  = flag.String("ane-route-profile", os.Getenv("MLXGO_ANE_ROUTE_PROFILE"), "ANE linear routing profile: balanced, conservative, or aggressive")
	aneAllowFallback = flag.Bool("ane-allow-fallback", os.Getenv("MLXGO_ANE_ALLOW_FALLBACK") == "true" || os.Getenv("MLXGO_ANE_ALLOW_FALLBACK") == "1", "Allow MLX fallback when ANE training forward routing declines or fails")
	anePerf          = flag.Bool("aneperf", false, "Start aneperf agent to report ANE utilization")
)

type anePerfSample struct {
	AneUtilizationPct *float64         `json:"ane_utilization_pct"`
	Channels          []anePerfChannel `json:"channels"`
}

type anePerfChannel struct {
	Channel string                `json:"channel"`
	States  []anePerfChannelState `json:"states"`
}

type anePerfChannelState struct {
	Name        string `json:"name"`
	ResidencyNS int64  `json:"residency_ns"`
}

type anePerfStats struct {
	mu      sync.Mutex
	lastPct float64
	maxPct  float64
	samples int
}

func (s *anePerfStats) update(pct float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastPct = pct
	if pct > s.maxPct {
		s.maxPct = pct
	}
	s.samples++
}

func (s *anePerfStats) status() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.samples == 0 {
		return ""
	}
	return fmt.Sprintf(" ANE:%.1f%%", s.lastPct)
}

func (s *anePerfStats) summary() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.samples == 0 {
		return ""
	}
	return fmt.Sprintf("ANE utilization: last=%.1f%% max=%.1f%% samples=%d", s.lastPct, s.maxPct, s.samples)
}

func parseANEUtilization(out []byte) (float64, error) {
	var sample anePerfSample
	if err := json.Unmarshal(out, &sample); err != nil {
		return 0, err
	}
	if sample.AneUtilizationPct != nil {
		return *sample.AneUtilizationPct, nil
	}
	for _, ch := range sample.Channels {
		if ch.Channel != "PACC1_ANE" {
			continue
		}
		var act, inact float64
		for _, state := range ch.States {
			switch strings.TrimSpace(state.Name) {
			case "ACT":
				act = float64(state.ResidencyNS)
			case "INACT":
				inact = float64(state.ResidencyNS)
			}
		}
		total := act + inact
		if total == 0 {
			return 0, nil
		}
		return math.Float64frombits(math.Float64bits((act / total) * 100)), nil
	}
	return 0, fmt.Errorf("aneperf output missing utilization fields")
}

func main() {
	cfg := lmtrain.RegisterFlags(flag.CommandLine)
	flag.Parse()

	if cfg.ModelPath == "" && !cfg.Train && !cfg.Test {
		// Just a sanity check for basic usage message
		flag.Usage()
		os.Exit(1)
	}

	var (
		perfStats  *anePerfStats
		perfCancel context.CancelFunc
		perfWG     sync.WaitGroup
	)
	if *anePerf {
		perfStats = new(anePerfStats)
		lmtrain.IterStatusHook = perfStats.status
		perfCtx, cancel := context.WithCancel(context.Background())
		perfCancel = cancel
		perfWG.Add(1)
		go func() {
			defer perfWG.Done()
			reportedError := false
			for {
				out, err := exec.CommandContext(perfCtx, "aneperf", "-interval", "100ms", "-json").Output()
				if err != nil {
					if perfCtx.Err() != nil {
						return
					}
					if !reportedError {
						fmt.Printf("aneperf error: %v\n", err)
						reportedError = true
					}
					continue
				}
				pct, err := parseANEUtilization(out)
				if err != nil {
					if !reportedError {
						fmt.Printf("aneperf parse error: %v\n", err)
						reportedError = true
					}
					continue
				}
				reportedError = false
				perfStats.update(pct)
			}
		}()
	}

	routingMode := *aneForward
	if routingMode == "" {
		routingMode = "off" // align with register's default
	}

	profile := *aneRouteProfile
	if profile == "" {
		profile = "balanced"
	}

	if routingMode == "all" {
		if os.Getenv("MLX_TRAIN_PAD_TO") == "" && cfg.BatchSize <= 1 {
			_ = os.Setenv("MLX_TRAIN_PAD_TO", "256")
		}
		ctx := context.Background()
		r, err := offload.SetupTrainingRouting(ctx, routingMode, profile, *aneAllowFallback)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to setup training routing: %v\n", err)
			os.Exit(1)
		}
		defer func() {
			r.Report()
			r.Close()
		}()
		fmt.Printf("ANE training routing enabled (profile: %s, fallback: %v)\n", profile, *aneAllowFallback)
	}

	ctx := context.Background()
	err := lmtrain.Run(ctx, cfg)
	if perfCancel != nil {
		perfCancel()
		perfWG.Wait()
		if summary := perfStats.summary(); summary != "" {
			fmt.Println(summary)
		}
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
