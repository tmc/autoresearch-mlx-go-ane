package mlxgoane

// RuntimeOptions configures runtime construction.
type RuntimeOptions struct {
	Executor LinearExecutor

	// AllowFallback defaults to true when nil.
	AllowFallback *bool

	// Router overrides profile-based router selection when set.
	Router *LinearRouter

	// TrainingRouter overrides TrainingLinearRouteProfile when set.
	TrainingRouter *LinearRouter

	// LinearRouteProfile is used when Router is nil.
	LinearRouteProfile LinearRouteProfile

	// TrainingLinearRouteProfile is used when TrainingRouter is nil.
	TrainingLinearRouteProfile LinearRouteProfile
}

// NewRuntimeWithOptions returns a runtime configured from opts.
func NewRuntimeWithOptions(opts RuntimeOptions) *Runtime {
	rt := &Runtime{
		Executor:      opts.Executor,
		AllowFallback: true,
	}
	if opts.AllowFallback != nil {
		rt.AllowFallback = *opts.AllowFallback
	}
	if opts.Router != nil {
		rt.Router = opts.Router
	} else if opts.LinearRouteProfile != "" {
		rt.Router = NewLinearRouter(LinearRouteConfigForProfile(opts.LinearRouteProfile))
	}

	if opts.TrainingRouter != nil {
		rt.TrainingRouter = opts.TrainingRouter
	} else if opts.TrainingLinearRouteProfile != "" {
		rt.TrainingRouter = NewLinearRouter(TrainingLinearRouteConfigForProfile(opts.TrainingLinearRouteProfile))
	}
	return rt
}
