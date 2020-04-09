module FluidAverages

using Dierckx, QuadGK
using ProgressMeter

using Distributed
using NumericalIntegration: integrate, cumul_integrate

"""
    ConvergeAlphaDash(alpha_R, alpha, gs, g, dens ; imax=10, calc_R=alpha_R, tol=1e-6, kwds...)

Solves for the screened r-dependent polarisability α′(r) iteratively. The input
single-atom polarisability is given by vectors `alpha_R` and `alpha`, and the
pair correlation function g(r) is given by `gs` and `g`. The equations are
solved for a density `dens`.

The output is given on the mesh `calc_R` which defaults to `alpha_R`.

The iteration stops after `imax` iterations of if a difference between two
iterations is less than `tol`.

`kwds` are passed through to the `CalcAlphaDash` function, called at each
iteration. It is highly recommended to set `s_max` for this to a reasonably
large value.
k"""
function ConvergeAlphaDash(alpha_R, alpha, gs, g, dens ; imax=10, calc_R=alpha_R, tol=1e-6, kwds...)
    fL = 1 / (1 + (8/3)*pi*dens*alpha[end])

    alpha_interp = Spline1D(alpha_R, alpha, k=1, bc="nearest")

    cur = alpha_interp(calc_R)*fL

    for i in 1:imax
        new = CalcAlphaDash(alpha_R, alpha, gs, g, dens, calc_R=calc_R, alpha_prev=cur ; kwds...)
        diff = NumericalIntegration.integrate(calc_R, abs.(new - cur))
        @show diff

        abs(diff) < tol && break

        cur = new
    end

    return cur
end

function CalcAlphaDash(alpha_R, alpha, gs, g, dens ; s_max=gs[end], calc_R=alpha_R, alpha_prev=:auto, show_figures=TypeFalse(), method=:inv_quadgk)
    @assert method ∈ [:quadgk, :inv_quadgk, :inv_trapz]
    
    if gs[1] == 0.
        popfirst!(gs)
        popfirst!(g)
    end

    if show_figures isa Real
        @assert show_figures ∈ calc_R
    end

    alpha_interp = Spline1D(alpha_R, alpha, k=1, bc="nearest")
    if alpha_prev == :auto
        alpha_prev = alpha_interp.(calc_R)
    end
    alpha_prev_interp = Spline1D(calc_R, alpha_prev, k=1, bc="nearest")
    g_interp = Spline1D(gs, g, k=1, bc="nearest")

    alpha_dash = similar(alpha)

    function Theta(Rsqr,ssqr,t)
        tsqr = t^2

        out = 3/2 / ssqr * (ssqr + tsqr - Rsqr) * (ssqr + Rsqr - tsqr) + (tsqr + Rsqr - ssqr)
        return out
    end

    function sintegral(s, R)
        Rsqr = R^2
        ssqr = s^2

        tmin = 0.
        tintmin = max(1e-8, tmin, abs(R - s))
        tintmax = R + s

        invt_intmin = 1/tintmin
        invt_intmax = 1/tintmax

        if method == :quadgk
            func = t -> 1/t^2 * Theta(Rsqr, ssqr, t) * alpha_prev_interp(t)
            out,err = quadgk(func, tintmin, tintmax)
        elseif method == :inv_quadgk
            func = invt -> -Theta(Rsqr, ssqr, 1/invt) * alpha_prev_interp(1/invt)
            out,err = quadgk(func, invt_intmin, invt_intmax, atol=1e-10, rtol=1e-6)
        elseif method == :inv_trapz
            dt = 0.01
            t = tintmin:dt:tintmax
            if t[end] != tintmax
                t = [t ; tintmax]
            end

            func = t -> Theta(Rsqr, ssqr, t) * alpha_prev_interp(t)
            temp = -NumericalIntegration.integrate(1 ./ t, func.(t))

            return temp
        end

        out
    end

    function Rintegral(R)
        sint_min = gs[1]
        sint_max = s_max

        if method == :quadgk
            func = s -> 1/s^2 * g_interp(s) * alpha_interp(s) * sintegral(s, R)
            out,err = quadgk(func, sint_min, sint_max)
            return out
        elseif method == :inv_quadgk
            func = invs -> -g_interp(1/invs) * alpha_interp(1/invs) * sintegral(1/invs, R)

            invs_intmin = 1/gs[1]
            invs_intmax = 1/s_max
            out,err = quadgk(func, invs_intmin, invs_intmax, atol=1e-10, rtol=1e-6)
            return out
        elseif method == :inv_trapz
            smesh = range(gs[1], s_max, length=1001)
            svals = sintegral.(smesh, R)
            svals .*= g_interp.(smesh) .* alpha_interp.(smesh)

            return -NumericalIntegration.integrate(1 ./ smesh, svals)
        end
    end

    alpha_dash = @showprogress pmap(Rintegral, calc_R)
    alpha_dash *= -pi*dens
    alpha_dash += alpha_interp.(calc_R)

    alpha_dash
end


function ReadFRAndGSFromBobInputs(f_filename, g_filename)
    gmat = readdlm(g_filename, skipstart=1)
    gs,g = cols(gmat)
    gheader = readline(g_filename)
    _,gmin,dens = parse.(split(gheader)[1:3])

    g[gs.<gmin] = 0.
    @assert all(g.>=0)

    fmat = readcsv(f_filename)
    fR,f = cols(fmat)

    return fR,f,gs,g,dens
end

function LogMesh(N::Int, Z::Int, rho::Float64=-8.0, h::Float64=0.0625)
    logmesh = exp.(rho + (0:(N-1)) * h) / Z
end

function ReadAlphaFromBobInput(filename)
    local mesh
    local vals
    
    open(filename) do file

        line = readline(file)
        N,h,Z,rho = split(line)[1:4] .|> parse

        line = readline(file)
        NPOT = split(line)[1] |> parse

        # Create the log mesh
        mesh = LogMesh(N, Int(Z), rho, h)
        
        count = 0
        while count < 2
            line = readline(file)
            if startswith(line, " ATOM")
                count += 1
            end
        end

        vals = zeros(N,NPOT)

        for pot_ind = 1:NPOT-2
            line = readline(file)
            i = 1
            while !isalpha(strip(line)[1])
                thisvals = split(line) .|> parse
                n = length(thisvals)
                #append!(vals, thisvals)
                @show i,n,pot_ind
                vals[i:i+n-1,pot_ind] = thisvals
                i += n

                line = readline(file)
            end
            @assert i == N+1
        end
    end

    mesh,vals
end

using Polynomials
function SurroundingAverage(gs,g,r,func,dens ; asymp_pot=false)

    func_interp = Spline1D(r,func)
    
    # Manually doing cumtrapz
    integrand = gs .* g

    terms = RunningAvg(integrand) .* diff(gs)
    terms = [0 ; terms]
    sgs_cumint = cumsum(terms)

    sgscum_spline = Spline1D(gs, sgs_cumint,k=1)
    sgscum_extrap = s -> sgs_cumint[end] + 1/2*(s^2 - gs[end]^2)
    sgscum_comb = s -> s < gs[end] ? sgscum_spline(s) : sgscum_extrap(s)
    
    function TIntegrand(r,t)
        smin = abs(r - t)
        smax = r + t

        out = t * func_interp(t) * (sgscum_comb(smax) - sgscum_comb(smin))
    end

    tmax = r[end]
    func2 = map(r) do r
        # if @printagain()
        #     println("Up to r=$r")
        # end
        2*pi*dens / r * quadgk(t -> TIntegrand(r,t), 0, tmax)[1]
    end


    if asymp_pot
        # Add on the long-range form of the polarisation potential
        coeffs = polyfit(log.(r[end-10:end]), log.(-func[end-10:end]), 1)
        @assert isapprox(coeffs[1], -4., atol=0.1)

        A = exp(coeffs[0])
        asympcorr = -4*pi*dens*A / tmax
        @show asympcorr
        @show A
        func2 .+= asympcorr
    end
    
    func2
end




# Trying PY approx

# function PercusYevick(N, pot, β, ρ ; tol=1e-6, imax=10^6, α=1.0)
function PercusYevick(N, b, pot, β, ρ ;
                      tol=1e-6,
                      imax=10^6,
                      α=1.0,
                      suppress_r=0.1,
                      renormalise=false,
                      init=:auto,
                      init_N=N,
                      ignore_kb_int=true
                      )
    local R = nothing

    if init_N == N || init == :auto
        y = init
    else
        R_old, = PYR(init_N, b)
        R,= PYR(N,b)
        y = Spline1D(R_old, init).(R)
    end

    E(Rin) = (Rin == 0 ? 0. : exp(-β*pot(Rin)))
    gfunc(y) = @. E(R) * y

    prog = ProgressThresh(tol)

    try
        for i in 1:imax
            R,new = PercusYevickWorker(N, b, y, pot, β, ρ)

            any(isnan,new) && error("Values went NaN after $i iterations!")

            if y != :auto
                # diff = NumericalIntegration.integrate(R, abs.(new - y))
                # Comparing with g's since the y can have troubles converging at small R.
                new[R .< suppress_r] .= 1.
                if renormalise
                    # Conserve the number of particles
                    # extra = NumericalIntegration.integrate(R, y .- 1)

                    # Actually the above wouldn't work very well. So instead,
                    # force the outer part to go to one.
                    new ./= new[end]
                end
                old_g = gfunc(y)
                new_g = gfunc(new)
                diff = NumericalIntegration.integrate(R, abs.(old_g - new_g))


                update!(prog, diff)
                # @show diff


                abs(diff) < tol && break

                y = y + α*(new-y)
            else
                y = new
            end
        end
    catch exc
        (exc isa InterruptException && ignore_kb_int) || rethrow()
        @warn "Ignoring interrupt"
    end

    g = gfunc(y)
    return R,g,y
end


using Plots
# function PercusYevickWorker(Rlist, y, pot, β, ρ)
#     max_s = Rlist[end]
    
#     yspl = Spline1D(Rlist,y,k=1)
#     yfunc(R) = (minimum(Rlist) <= R <= maximum(Rlist) ? yspl(R) : 1.)

#     g(R) = exp(-β*pot(R)) * yspl(R)
#     h(R) = g(R) - 1
#     f(R) = exp(-β*pot(R)) - 1
#     # fy(R) = (1 - exp(β*pot(R)))*y(R)

#     # gspl = Spline1D(R,g,k=1)

#     # y(R) = exp(β*pot(R)) * gspl(R)
#     # h(R) = gspl(R) - 1
#     # f(R) = 1 - exp(β*pot(R))
#     # fy(R) = (1 - exp(β*pot(R)))*y(R)

#     # cspl = Spline1D(R,c,k=1)

#     # y(R) = exp(β*pot(R)) * gspl(R)
#     # h(R) = gspl(R) - 1
#     # f(R) = 1 - exp(β*pot(R))
#     # fy(R) = (1 - exp(β*pot(R)))*y(R)


#     function sintegral(s, R)
#         tmin = 1e-8
#         tint_min = max(tmin, abs(R - s))
#         tint_max = R + s
#         if tint_min > tint_max
#             return 0.
#         end
          
#         # tint_points = [range(tint_min, tint_max, step=1.) ; tint_max]
#         tint_points = (tint_min, tint_max)

#         # invt_intmin = 1/tintmin
#         # invt_intmax = 1/tintmax

#         func = t -> t * h(t)
#         if true
#             # func = t -> t * h(t)
#             out,err = quadgk(func, tint_points..., atol=1e-6, rtol=1e-6)
#             return out
#         elseif false

#             func = invt -> -Theta(Rsqr, ssqr, 1/invt) * alpha_prev_interp(1/invt)
#             out,err = quadgk(func, invt_intmin, invt_intmax, atol=1e-10, rtol=1e-6)
#             # out,err = quadgk(func, invt_intmin, invt_intmax, rtol=1e-6)
#             return out
#         else
#             dt = 0.01
#             t = tint_min:dt:tint_max
#             if t[end] != tint_max
#                 t = [t ; tint_max]
#             end

#             temp = NumericalIntegration.integrate(t, func.(t))
#             return temp
#         end
#     end

#     function Rintegral(R)
#         sint_min = 1e-8
#         sint_max = max_s*2

#         # sint_points = range(sint_min, sint_max, step=1.)
#         sint_points = (sint_min, sint_max)

#         func = s -> s * f(s) * yspl(s) * sintegral(s, R)
#         if true
#             # temps = LinRange(sint_min, sint_max, 1001)
#             # p = plot(temps, func.(temps))
#             # gui()
#             # print("Gui-ed:")
#             # readline(stdin)
#             out,err = quadgk(func, sint_points..., rtol=1e-6, atol=1e-6)

#             return out
#         elseif false
#             func = invs -> -g_interp(1/invs) * alpha_interp(1/invs) * sintegral(1/invs, R)

#             invs_intmin = 1/gs[1]
#             invs_intmax = 1/s_max
#             out,err = quadgk(func, invs_intmin, invs_intmax, atol=1e-10, rtol=1e-6)
#             (show_figures isa Real && R == show_figures) && plot!(func, invs_intmax, invs_intmin)
#             return out
#         else
#             smesh = range(sint_min, sint_max, length=1001)
#             # svals = sintegral.(smesh, R)
#             # @. svals *= smesh * f(smesh)

#             return NumericalIntegration.integrate(smesh, func.(smesh))
#         end
#     end

#     y_dash = @showprogress pmap(Rintegral, Rlist)
#     # y_dash = map(Rintegral, R)
#     y_dash .*= 2π*ρ./Rlist
#     y_dash .+= 1

#     return y_dash
# end

# function PercusYevickWorker(Rlist, y, pot, β, ρ)
#     max_s = Rlist[end]*2
    
#     yspl = Spline1D(Rlist,y,k=1)
#     xfunc(R) = (minimum(Rlist) <= R <= maximum(Rlist) ? yspl(R) : 1.) * R

#     # g(R) = exp(-β*pot(R)) * yspl(R)
#     # h(R) = g(R) - 1
#     # f(R) = exp(-β*pot(R)) - 1
#     E(R) = exp(-β*pot(R))
#     # fy(R) = (1 - exp(β*pot(R)))*y(R)

#     function sintegral(R)
#         sint_min = 1e-8
#         sint_max = max_s

#         # sint_points = range(sint_min, sint_max, step=1.)
#         sint_points = (sint_min, sint_max)

#         function func(s)
#             out = (1 - E(s)) * xfunc(s)
#             tmin = abs(s-R)
#             tmax = s+R
#             out *= (xfunc(tmax)*E(tmax) + sign(s-R)*xfunc(tmin)*E(tmin) - 2*s)
#         end

#         if true
#             out,err = quadgk(func, sint_points..., rtol=1e-6, atol=1e-6)
#         end

#         out = 1 - 2π*ρ*out
#     end

#     if true
#         dx = @showprogress pmap(sintegral, Rlist)
#         x = NumericalIntegration.cumul_integrate(Rlist, dx)
#     else
#         dxlist = @showprogress pmap(2:length(Rlist)) do ind
#             dx = quadgk(sintegral, Rlist[ind-1], Rlist[ind], rtol=1e-6, atol=1e-6)[1]
#         end
#         dxlist = [0 ; dxlist]
#         x = cumsum(dxlist)
#     end

#     y = x./Rlist
# end

function PYR(N,b)
    extRlist = LinRange(0,2*b,2N+1)
    Rlist = filter(<=(b), extRlist)

    return Rlist,extRlist
end
function PercusYevickWorker(N, b, y, pot, β, ρ)
    Rlist,extRlist = PYR(N,b)

    if y == :auto
        y = ones(length(Rlist))
    end

    yext = [y ; ones(length(extRlist) - length(Rlist))]
    
    x = yext .* extRlist

    E = @. exp(-β*pot(extRlist))
    E[1] = 0.

    function sintegral(Rind)
        integrand = map(eachindex(Rlist)) do sind
            out = (1 - E[sind]) * x[sind]
            tmin_ind = abs(sind-Rind) + 1
            tmax_ind = (sind-1+Rind-1) + 1
            out *= (x[tmax_ind]*E[tmax_ind] + sign(sind-Rind)*x[tmin_ind]*E[tmin_ind] - 2*Rlist[sind])
        end

        out = NumericalIntegration.integrate(Rlist, integrand)
        out = 1 - 2π*ρ*out
    end

    # dx = @showprogress pmap(sintegral, eachindex(Rlist))
    dx = sintegral.(eachindex(Rlist))
    x = NumericalIntegration.cumul_integrate(Rlist, dx)

    y = x./Rlist
    y[1] = 1.

    Rlist,y
end
# function PercusYevickWorker(N, b, y, pot, β, ρ)
#     # This version uses the Baxter 1967 paper result
#     Rlist,extRlist = PYR(N,b)

#     if y == :auto
#         y = ones(length(Rlist))
#     end

#     yext = [y ; ones(length(extRlist) - length(Rlist))]
    
#     x = yext .* extRlist

#     E = @. exp(-β*pot(extRlist))
#     E[1] = 0.

#     idiff(x,y) = (x - y) + 1

#     integrate = NumericalIntegration.integrate

#     function tintegral(Rind)
#         lessthanR = Rlist[1:Rind]

#         term1 = 2π*ρ * integrate(Rlist[1:Rind], [H[tind]*H[idiff(Rind,tind)] for tind in 1:Rind])

#         term2 = -4π*ρ * (integrate(Rlist[1:Rind], [C[sind]*H[idiff(Rind,sind)] for sind in 1:Rind])
#                          - integrate(Rlist[Rind:end], [C[sind]*H[idiff(sind,Rind)] for sind in Rind:length(Rlist)]))

#         W = map(eachindex(Rlist)) do sind
#             integrate(Rlist[1:sind], H .*

#         term3 = 4π^2*ρ^2 * integrate(Rlist, C .* W)

#         integrand = map(eachindex(Rlist)) do sind
#             out = (1 - E[sind]) * x[sind]
#             tmin_ind = abs(sind-Rind) + 1
#             tmax_ind = (sind-1+Rind-1) + 1
#             out *= (x[tmax_ind]*E[tmax_ind] + sign(sind-Rind)*x[tmin_ind]*E[tmin_ind] - 2*Rlist[sind])
#         end

#         out = NumericalIntegration.integrate(Rlist, integrand)
#         out = 1 - 2π*ρ*out
#     end

#     dH = tintegral.(eachindex(Rlist))
#     H = NumericalIntegration.cumul_integrate(Rlist, dH)

#     H += C

#     y = x./Rlist
#     y[1] = 1.

#     Rlist,y
# end

function LJTest(y=:auto; β=1., ρ)
    R = LinRange(0,10,1001)[2:end]
    y = (y == :auto ? ones(length(R)) : y)

    pot = R -> (1/R^12 - 1/R^6)
    # pot = R -> (R < 1. ? 10. : 0.)

    ydash = PercusYevick(R, y, pot, β, ρ)

    return R,ydash
end

import PercusYevickSSF
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


using Constants
function LJCompare(T,ρ, σ_LJ, ϵ_LJ ; N=101, b=10, kwds...)
    β = ϵ_LJ/(kB*T)
    ρ = ρ * σ_LJ^3

    @show β 1/β ρ

    maxpot = R -> min(pot(R), 10. / β)

    R,g, = PercusYevick(N, b,  pot, β, ρ ; kwds...)

    R *= σ_LJ

    return R,g
end

function LJCalc(β,ρ, ;
                N=101,
                b=10,
                α=0.1,
                maxpot_val=Inf,
                maxR = Inf,
                kwds...)
    pot = R -> 4*(1/R^12 - 1/R^6)

    maxpot = R -> R > maxR ? 0. : min(pot(R), maxpot_val / β)

    R,g,y = PercusYevick(N, b, maxpot, β, ρ ; α=α, kwds...)

    return R,g,y
end

function ApproachLJ(β_in, ρ, N, Nfinal=N ;
                    safe_β=1/1.5,
                    β_step=β_in - safe_β,
                    β_step_adjust=true,
                    kwds...)
    y = :auto
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
                error("Break")
            end
            @info "Adjusting β_step to be smaller" β_step
            continue
        end

        last_β = β
    end

    if Nfinal != N || R == nothing
        R,g,y = LJCalc(β_in,ρ, N=Nfinal ; ignore_kb_int=false, init=y, init_N=N, kwds...)
    end
        

    return R,g,y
end

end

end # module
