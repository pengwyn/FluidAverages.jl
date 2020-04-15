
import PercusYevickSSF
"""A comparison with my other analytical hard-sphere PY result. Only works if that module is available."""
function CompareWithAnalytic(a, ϕ, β=1. ; potval=1e0, kwds...)
    # Note: β only enters with the potential, which is being taken to infinity anyway.

    r,ρ = PercusYevickSSF.HardsphereRadiusDensity(a, ϕ, factor_4pi=false)

    @show r ρ

    pot = R -> (R < 2*r ? potval : 0.)

    R = LinRange(0,5,1001)[2:end]
    g_anly = PercusYevickSSF.PairCorrelator(R, r, ρ)

    R_num,g_num,y_num = PercusYevick(201, 5.0, pot, β, ρ ; kwds...)

    return R,g_anly,R_num,g_num,y_num
end
