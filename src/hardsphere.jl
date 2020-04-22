

"""
    HardsphereRadiusDensity(cross_section, phi)

This is a convenience function to work out the radius and density of a system of hard-spheres.
"""
function HardsphereRadiusDensity(cross_section, phi)
    @assert phi != 0
    
    r = sqrt(cross_section / pi)
    n0 = phi / (4/3 * pi * r^3)

    r,n0
end

"""
    PYHardSphere(a, ϕ ; N=201, b=5.0, potval=1.0, α=0.01, kwds...)
    
Determine the hard-sphere Percus-Yevick approximation. The cross section is `a` and the filing fraction `phi`. The parameters `N`, `b` and `α` are passed to `PercusYevick`.

This is actually solved with a "soft" sphere of potential height `potval`. Ideally this should be raised until convergence is reached, but this requires an appropriate initial guess to be passed through to `PercusYevick`.
    
Note that temperature is redundant in the hard-sphere problem, as only the combination β*V is used in the calculations. Hence, it is assumed that β=1.
"""
function PYHardSphere(a, ϕ ; N=201, b=5.0, potval=1.0, α=0.01, kwds...)
    radius,ρ = HardsphereRadiusDensity(a, ϕ)

    pot = R -> (R < 2*radius ? potval : 0.)

    # Note: β only enters with the potential, which is being taken to infinity anyway.
    # Set it to 1.0 so that we have an idea what is going into exp(...)
    β = 1.0
    R,g,y = PercusYevick(N, b, pot, β, ρ ; α=α, kwds...)

    return R,g
end

"""
    PYHardSphereStepped(a, ϕ ; potval_steps=[5,10,15], kwds...)
    
Same as `PYHardSphere` but increases the potential through steps given in `potval_steps`, reusing the previous step as an initial guess.
"""
function PYHardSphereStepped(a, ϕ ; potval_steps=[5,10,15,30], kwds...)
    steps = copy(potval_steps)
    init_potval = popfirst!(steps)

    R,g = PYHardSphere(a,ϕ ; potval=init_potval, ignore_kb_int=false, kwds...)
    for step in steps
        R,g = PYHardSphere(a,ϕ ; potval=step, init=g, ignore_kb_int=false, kwds...)
    end

    return R,g
end
