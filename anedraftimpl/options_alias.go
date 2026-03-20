//go:build darwin

package anedraftimpl

import anedraft "github.com/tmc/mlx-go-lm/draftbackend"

type ANEDraftOptions = anedraft.Options
type ANEDraftRuntimeStats = anedraft.RuntimeStats
type ANEDraftStatsReporter = anedraft.StatsReporter

type aneDraftOptions = ANEDraftOptions
type aneDraftRuntimeStats = ANEDraftRuntimeStats
type aneDraftStatsReporter = ANEDraftStatsReporter
