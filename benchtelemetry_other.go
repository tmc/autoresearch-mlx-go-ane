//go:build !darwin || !ane_appleneuralengine

package mlxgoane

type metricReporter interface {
	ReportMetric(float64, string)
}

// BenchmarkHeader describes the ANE environment used by a benchmark run.
type BenchmarkHeader struct{}

// ProbeBenchmarkHeader returns an empty header when ANE telemetry is unavailable.
func ProbeBenchmarkHeader() BenchmarkHeader { return BenchmarkHeader{} }

// ReportMetrics is a no-op when ANE telemetry is unavailable.
func (BenchmarkHeader) ReportMetrics(metricReporter) {}

// Logf is a no-op when ANE telemetry is unavailable.
func (BenchmarkHeader) Logf(func(string, ...any)) {}
