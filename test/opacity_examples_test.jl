using Test
using ForwardDiff

include(joinpath(@__DIR__, "..", "examples", "opacity.jl"))

function write_synthetic_opacity_table(path::String, X::Float64, Z::Float64, logk::Float64)
    open(path, "w") do io
        write(io, "Synthetic opacity table\n")
        write(io, " 1 0 $(X) $(Z) 2 0 0 2\n")
        write(io, " -8.0 -7.0\n")
        write(io, " 3.0 $(logk) $(logk)\n")
        write(io, " 4.0 $(logk) $(logk)\n")
    end
end

@testset "opacity example parser and interpolation" begin
    mktempdir() do tmpdir
        single_file = joinpath(tmpdir, "single_parser_case.data")
        write_synthetic_opacity_table(single_file, 0.70, 0.02, 0.3)

        table = RT_table_opacity(single_file)
        @test table.X ≈ 0.70
        @test table.Z ≈ 0.02
        @test table.logTs == [3.0, 4.0]
        @test table.logRs == [-8.0, -7.0]
        @test length(table.kap_data) == 4

        write_synthetic_opacity_table(joinpath(tmpdir, "a_lowT_mix.data"), 0.60, 0.01, 0.0)
        write_synthetic_opacity_table(joinpath(tmpdir, "b_lowT_mix.data"), 0.70, 0.01, 0.0)
        write_synthetic_opacity_table(joinpath(tmpdir, "c_lowT_mix.data"), 0.60, 0.02, 0.0)
        write_synthetic_opacity_table(joinpath(tmpdir, "d_lowT_mix.data"), 0.70, 0.02, 0.0)

        col = Opacity_table_collector(tmpdir, "lowT_mix")
        xa = Float64[0.70, 0.28, 0.02]
        species = [:H1, :He4, :C12]

        κ1 = get_opacity_table_collection(col, log(10.0^3.0), log(1e-3), xa, species)
        κ2 = get_opacity_table_collection(col, log(10.0^8.0), log(1e5), xa, species)
        @test κ1 > 0
        @test κ2 > 0

        # Incomplete grid should fail at collector construction
        missing_dir = joinpath(tmpdir, "missing")
        mkpath(missing_dir)
        write_synthetic_opacity_table(joinpath(missing_dir, "a_lowT_mix.data"), 0.60, 0.01, 0.0)
        write_synthetic_opacity_table(joinpath(missing_dir, "b_lowT_mix.data"), 0.70, 0.01, 0.0)
        write_synthetic_opacity_table(joinpath(missing_dir, "c_lowT_mix.data"), 0.60, 0.02, 0.0)
        @test_throws ErrorException Opacity_table_collector(missing_dir, "lowT_mix")
    end
end

@testset "opacity composite transition continuity" begin
    mktempdir() do tmpdir
        low_dir = joinpath(tmpdir, "low")
        high_dir = joinpath(tmpdir, "high")
        mkpath(low_dir)
        mkpath(high_dir)

        for X in (0.60, 0.70), Z in (0.01, 0.02)
            write_synthetic_opacity_table(joinpath(low_dir, "$(X)_$(Z)_low_mix.data"), X, Z, 0.0)
            write_synthetic_opacity_table(joinpath(high_dir, "$(X)_$(Z)_high_mix.data"), X, Z, 1.0)
        end

        low_col = Opacity_table_collector(low_dir, "low_mix")
        high_col = Opacity_table_collector(high_dir, "high_mix")
        composite = CompositeOpacity(low_col, high_col, 3.8, 4.2)

        xa = Float64[0.70, 0.28, 0.02]
        species = [:H1, :He4, :C12]

        κ_low = Jems.Opacity.get_opacity_resultsTρ(composite, log(10.0^3.8), log(1e-3), xa, species)
        κ_mid = Jems.Opacity.get_opacity_resultsTρ(composite, log(10.0^4.0), log(1e-3), xa, species)
        κ_high = Jems.Opacity.get_opacity_resultsTρ(composite, log(10.0^4.2), log(1e-3), xa, species)

        @test κ_low ≈ 1.0 atol = 1e-10
        @test κ_high ≈ 10.0 atol = 1e-10
        @test κ_low < κ_mid < κ_high
    end
end
