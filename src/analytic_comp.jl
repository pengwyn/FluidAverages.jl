import PercusYevickSSF

"""
    CompareWithAnalytic(a, ϕ ; kwds...)

A comparison with my other analytical hard-sphere PY result. Only works if that module is available. `a` is the cross section and `phi` the filling fraction. All other `kwds` are passed to `PYHardSphereStepped`.
"""
function CompareWithAnalytic(a, ϕ ; kwds...)
    radius,ρ = PercusYevickSSF.HardsphereRadiusDensity(a, ϕ, factor_4pi=false)

    R,g = PYHardSphereStepped(a, ϕ ; ignore_kb_int=false, kwds...)
    # A finer grid
    # R,g = PYHardSphere(a, ϕ ; init=g, N=1000, α=0.01, potval=30.0, kwds...)

    g_anly = PercusYevickSSF.PairCorrelator(R, radius, ρ)

    return R,g,g_anly
end
