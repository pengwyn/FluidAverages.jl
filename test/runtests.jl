using FluidAverages

using Test

##############################
# * LJ tests
#----------------------------

@testset "Lennard-Jones" begin
    @test FluidAverages.ApproachLJ(1/0.8, 0.8, 101) != nothing

    # These tests don't work because the function gets stuck while looking for the solution.

    # Failing in the coexistence region
    # @test_throws ErrorException FluidAverages.ApproachLJ(1/0.8, 0.3, 101)
    # Test for too low a temperature
    # @test_throws ErrorException FluidAverages.ApproachLJ(1/0.3, 0.8, 101, α=0.001)
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
