package mlxgoane

import "time"

// LinearTelemetry captures per-call ANE linear execution timings.
//
// Compile/load durations are expected to be zero on cache-hit paths.
type LinearTelemetry struct {
	CacheHit bool
	Build    time.Duration
	Compile  time.Duration
	Load     time.Duration
	Evaluate time.Duration
}

// LinearTelemetryProvider exposes runtime telemetry for ANE-backed executors.
type LinearTelemetryProvider interface {
	LastLinearTelemetry() LinearTelemetry
	LinearCacheSize() int
}
