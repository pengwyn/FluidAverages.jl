using FluidAverages

using Test

@testset "PercusYevick" begin
    ##############################
    # * LJ tests
    #----------------------------

    @testset "Lennard-Jones" begin
        @test FluidAverages.ApproachLJ(1/0.8, 0.8, 101) != nothing

        # The tests below rely on hitting the imax which was 10_000 at the time of writing this.

        # Failing in the coexistence region
        @test_throws ErrorException("Couldn't converge") FluidAverages.ApproachLJ(1/0.8, 0.3, 101)
        # Test for too low a temperature
        @test_throws ErrorException("Couldn't converge") FluidAverages.ApproachLJ(1/0.3, 0.8, 101)
    end



    ######################################
    # * Hardsphere tests
    #------------------------------------

    @testset "Hardsphere" begin

        dir = joinpath(dirname(pathof(FluidAverages)), "../test")
        anly_data = readdlm(joinpath(dir, "analytical_PY.dat"), skipstart=1)

        R,g = FluidAverages.PYHardSphereStepped(1.0, 0.4, ignore_kb_int=false)

        @test R == anly_data[:,1]

        g_anly = anly_data[:,2]

        using NumericalIntegration

        # This is only testing to within 6% because that's the amount of error around the sharp turn on.
        # The differences could be due to the Wertheim correction factor... I should test this.
        atol = 1e-3
        rtol = 6e-2
        diff = abs.(g - g_anly)
        @test all(@. diff < atol + rtol*g_anly)

    end
end
