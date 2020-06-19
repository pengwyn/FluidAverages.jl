
LJpot(R) = 4*(1/R^12 - 1/R^6)

"""
    LJCalc(β,ρ ; maxpot_val=Inf, maxR=Inf, kwds...)

Do a `PercusYevick` iteration with the Lennard-Jones potential. `β` and `ρ` are to
be in LJ units. The potential is capped to the maximum value `maxpot_val` and is
truncated (but not shifted) after `maxR`.

Other kwds are passed to `PercusYevick`.
"""
function LJCalc(β,ρ, ;
                N=101,
                b=10,
                α=0.1,
                maxpot_val=Inf,
                maxR = Inf,
                kwds...)

    maxpot = R -> R > maxR ? 0. : min(LJpot(R), maxpot_val / β)

    R,g,y = PercusYevick(N, b, maxpot, β, ρ ; α=α, kwds...)

    return R,g,y
end

"""
    ApproachLJ(β_in, ρ, N, Nfinal=N)

Calculate a `PercusYevick` iteration solution to the Lennard-Jones problem at
`β_in`=1/kT and `ρ`. If there are convergence issues, the function will move to a
higher temperature to find a solution which does converge, and then uses that
solution as an initial guess for the desired temperature.
    
`N` is the number of grid points passed to `PercusYevick`. The size of the grid
can be controlled via an optional keyword `b` as described in `PercusYevick`.
`b` defaults to 10 as set in `LJCalc`.

As an additional step, if `Nfinal`≠`N` then after convergence, the code will
perform one extra iteration to increase the grid size from `N` to `Nfinal`.
"""
function ApproachLJ(β_in, ρ, N, Nfinal=N ;
                    safe_β=1/1.5,
                    β_step=β_in - safe_β,
                    β_step_adjust=true,
                    init=:auto,
                    kwds...)
    y = init
    R = nothing
    local g
    # for β in [(safe_β:β_step:β_in) ; β_in]
    last_β = min(safe_β, β_in)
    while last_β < β_in
        β = last_β + β_step
        β = min(β, β_in)

        @show β
        try
            R,g,y = LJCalc(β,ρ, N=N ; ignore_kb_int=false, init=y, kwds...)
        catch exc
            (exc isa ErrorException && occursin("NaN", exc.msg)) || rethrow()
            β_step /= 2.5
            if β_step < 0.001
                @error "β_step got too small. Maybe this is in the coexistence region?" β_step last_β β 1/β ρ
                error("β_step got too small. Maybe this is in the coexistence region?")
            end
            @info "Adjusting β_step to be smaller" β_step
            continue
        end

        last_β = β
    end

    if Nfinal != N || R == nothing
        if Nfinal isa Int
            Nfinal = [Nfinal]
        end
        
        for N in Nfinal
            R,g,y = LJCalc(β_in,ρ, N=N ; ignore_kb_int=false, init=y, kwds...)
        end
    end

    return R,g,y
end


####################################################
# * Thermdynamic quantities
#--------------------------------------------------

using NumericalIntegration: integrate

using ForwardDiff: derivative
LJ_deriv(R) = derivative(LJpot, R)

function LJ_Pressure(r,g,ρ,T)
    val = 4π * integrate(R, @. g * LJ_deriv(R) * R^3)
    P = ρ*T - 1/6*ρ^2*val
end
function LJ_EnergyDens(r,g,ρ,T)
    val = 4π * integrate(R, @. g * LJpot(R) * R^2)
    UonN = 3/2*ρ*T - 1/2*ρ^2*val
end
function LJ_Compressibility(r,g,ρ,T)
    # This is really S(K=0)
    val = 4π * integrate(R, @. (g-1) * R^2)
    kTdρdP = 1 + ρ*val
    χT = kTdρdP / ρ / T
end
