package mlxgoane

// RuntimeOptions configures runtime construction.
type RuntimeOptions struct {
	Executor LinearExecutor

	// AllowFallback defaults to true when nil.
	AllowFallback *bool

	// Router overrides profile-based router selection when set.
	Router *LinearRouter

	// LinearRouteProfile is used when Router is nil.
	LinearRouteProfile LinearRouteProfile
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
		return rt
	}
	if opts.LinearRouteProfile != "" {
		rt.Router = NewLinearRouter(LinearRouteConfigForProfile(opts.LinearRouteProfile))
	}
	return rt
}
