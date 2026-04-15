using Jems
using ForwardDiff

const INV_LN10 = 0.4342944819032518
const _OPACITY_OFFSETS = (-1, 0, 1, 2)
# Species indices are cached by species vector identity to avoid repeated symbol scans in hot paths.
const _SPECIES_INDEX_CACHE_LOCK = ReentrantLock()
const _SPECIES_INDEX_CACHE = Dict{UInt64,NTuple{2,Int}}()

"""
Structure to store data from a single OPLIB data file and store it as a 1D array of R,T alon with some metadata
"""
struct RT_table_opacity <: Jems.Opacity.AbstractOpacity
    X::Float64
    Z::Float64
    logTs::Vector{Float64}
    logRs::Vector{Float64}
    kap_data::Vector{Float64}
    inv_dlogT::Vector{Float64}
    inv_dlogR::Vector{Float64}
end

"""
Structure to store a matrix with X,Z being the rows and columns, and each element being a RT opacity table
"""
struct Opacity_table_collector <: Jems.Opacity.AbstractOpacity
    Xs::Vector{Float64}
    Zs::Vector{Float64}
    tables::Matrix{RT_table_opacity}
    inv_dX::Vector{Float64}
    inv_dZ::Vector{Float64}
end

"""
Structure to store opacaity tables of X,Z,R,T from different data set, also the meta data required to smooth interopolate between them
using a cubic spline
"""
struct CompositeOpacity <: Jems.Opacity.AbstractOpacity
    low_T_collector::Opacity_table_collector # Holds the matrix of X,Z with each element being RT_table (for low T)
    high_T_collector::Opacity_table_collector #Holds the matrix of X,Z with each element being RT_table (for high T)

    # Holds the transition points
    trans_logT_min::Float64
    trans_logT_max::Float64
end

mutable struct _OpacityLookupCache
    ix::Int
    iz::Int
    iT::Int
    iR::Int
end

_OpacityLookupCache() = _OpacityLookupCache(1, 1, 1, 1)

struct _OpacityEvalState{T<:Real}
    logT::T
    logR::T
    val_logT::Float64
    val_logR::Float64
    X::T
    Z::T
    val_X::Float64
    val_Z::Float64
end

@inline function _inverse_spacings(grid::Vector{Float64}, grid_name::String)
    n = length(grid)
    n >= 2 || error("$grid_name grid must contain at least 2 points, got $n")
    inv_d = Vector{Float64}(undef, n - 1)
    @inbounds for i in 1:(n - 1)
        δ = grid[i + 1] - grid[i]
        δ > 0 || error("$grid_name grid must be strictly increasing at index $i")
        inv_d[i] = inv(δ)
    end
    return inv_d
end

function RT_table_opacity(X::Float64, Z::Float64, logTs::Vector{Float64}, logRs::Vector{Float64}, kap_data::Vector{Float64})
    inv_dlogT = _inverse_spacings(logTs, "logT")
    inv_dlogR = _inverse_spacings(logRs, "logR")
    return RT_table_opacity(X, Z, logTs, logRs, kap_data, inv_dlogT, inv_dlogR)
end

@inline function _parse_numeric_row!(storage::Vector{Float64}, line::AbstractString, expected_cols::Int)
    parts = split(line)
    length(parts) == expected_cols || return false
    @inbounds for i in 1:expected_cols
        parsed = tryparse(Float64, parts[i])
        parsed === nothing && return false
        storage[i] = parsed
    end
    return true
end

function RT_table_opacity(filepath::String)
    meta_found = false
    grid_found = false
    data_started = false

    X_val = 0.0
    Z_val = 0.0
    num_Rs = 0
    num_Ts = 0

    logRs = Float64[]
    logTs = Float64[]
    data = Float64[]

    grid_buffer = Float64[]
    data_row_buffer = Float64[]
    row_counter = 0

    open(filepath, "r") do io
        for (line_idx, line) in enumerate(eachline(io))
            if !meta_found
                if occursin(r"^\s*1\s+\d+", line)
                    parts = split(line)
                    length(parts) >= 8 || error("Invalid metadata line at $filepath:$line_idx")

                    X_val = parse(Float64, parts[3])
                    Z_val = parse(Float64, parts[4])
                    num_Rs = parse(Int, parts[5])
                    num_Ts = parse(Int, parts[8])

                    num_Rs > 1 || error("Invalid R-grid size in $filepath:$line_idx")
                    num_Ts > 1 || error("Invalid T-grid size in $filepath:$line_idx")

                    logRs = Vector{Float64}(undef, num_Rs)
                    logTs = Vector{Float64}(undef, num_Ts)
                    data = Vector{Float64}(undef, num_Rs * num_Ts)

                    grid_buffer = Vector{Float64}(undef, num_Rs)
                    data_row_buffer = Vector{Float64}(undef, num_Rs + 1)
                    meta_found = true
                end
                continue
            end

            if !grid_found
                if _parse_numeric_row!(grid_buffer, line, num_Rs)
                    copyto!(logRs, grid_buffer)
                    grid_found = true
                end
                continue
            end

            if _parse_numeric_row!(data_row_buffer, line, num_Rs + 1)
                data_started = true
                row_counter += 1
                row_counter <= num_Ts || error("Too many data rows in $filepath:$line_idx")

                @inbounds begin
                    logTs[row_counter] = data_row_buffer[1]
                    base = (row_counter - 1) * num_Rs
                    for j in 1:num_Rs
                        data[base + j] = data_row_buffer[j + 1]
                    end
                end
            elseif data_started && !isempty(strip(line))
                error("Malformed data row in $filepath:$line_idx; expected $(num_Rs + 1) numeric columns")
            end
        end
    end

    meta_found || error("Could not find metadata line in $filepath")
    grid_found || error("Could not find logR grid row with $num_Rs entries in $filepath")
    row_counter == num_Ts || error("Unexpected data row count in $filepath: expected $num_Ts, found $row_counter")

    return RT_table_opacity(X_val, Z_val, logTs, logRs, data)
end

function _build_validated_grid(temp_tables::Vector{RT_table_opacity})
    isempty(temp_tables) && error("No opacity tables were loaded")

    unique_Xs = sort(unique(t.X for t in temp_tables))
    unique_Zs = sort(unique(t.Z for t in temp_tables))

    length(unique_Xs) >= 2 || error("Opacity X grid needs at least 2 points")
    length(unique_Zs) >= 2 || error("Opacity Z grid needs at least 2 points")

    x_to_idx = Dict{Float64,Int}(x => i for (i, x) in enumerate(unique_Xs))
    z_to_idx = Dict{Float64,Int}(z => i for (i, z) in enumerate(unique_Zs))

    maybe_grid = Matrix{Union{Nothing,RT_table_opacity}}(nothing, length(unique_Xs), length(unique_Zs))

    for table in temp_tables
        ix = x_to_idx[table.X]
        iz = z_to_idx[table.Z]
        isnothing(maybe_grid[ix, iz]) || error("Duplicate opacity table for X=$(table.X), Z=$(table.Z)")
        maybe_grid[ix, iz] = table
    end

    missing_coords = Tuple{Float64,Float64}[]
    for ix in eachindex(unique_Xs)
        for iz in eachindex(unique_Zs)
            if isnothing(maybe_grid[ix, iz])
                push!(missing_coords, (unique_Xs[ix], unique_Zs[iz]))
            end
        end
    end

    isempty(missing_coords) || error("Incomplete opacity grid; missing tables for coordinates: $missing_coords")

    grid = Matrix{RT_table_opacity}(undef, length(unique_Xs), length(unique_Zs))
    for ix in eachindex(unique_Xs)
        for iz in eachindex(unique_Zs)
            grid[ix, iz] = maybe_grid[ix, iz]::RT_table_opacity
        end
    end

    return unique_Xs, unique_Zs, grid
end

function Opacity_table_collector(directory::String, mixture_key::String)
    files = readdir(directory; join = true)
    opacity_files = filter(f -> endswith(f, ".data") && occursin(mixture_key, basename(f)), files)

    isempty(opacity_files) && error("No .data files with mixture key '$mixture_key' found in $directory")

    temp_tables = RT_table_opacity[]
    for file in opacity_files
        push!(temp_tables, RT_table_opacity(file))
    end

    unique_Xs, unique_Zs, grid = _build_validated_grid(temp_tables)
    inv_dX = _inverse_spacings(unique_Xs, "X")
    inv_dZ = _inverse_spacings(unique_Zs, "Z")

    println("Opacity tables loaded")
    return Opacity_table_collector(unique_Xs, unique_Zs, grid, inv_dX, inv_dZ)
end

@inline function damp_slope(x::T) where T
    if x > 1.0
        # Start at edge (1.0) + tiny fraction of the excess
        return 1.0 + 0.5 * (x - 1.0)
    elseif x < 0.0
        # Start at edge (0.0) + tiny fraction of the deficit
        return 0.0 + 0.5 * x
    else
        return x
    end
end

# The following function implements Bicubic interpolation using Catmull-Rom Cubic Basis Functions (for unit square)
@inline function cubic_weights(t::T) where T
    t2 = t * t
    t3 = t2 * t

    # Standard Catmull-Rom weights: [1 t t^2 t^3] M_SH (M_SH is the  Catmull-Rom Cubic Basis matrix for 1D)
    # https://en.wikipedia.org/wiki/Catmull–Rom_spline
    w_m1 = -0.5 * t3 + 1.0 * t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_2 = 0.5 * t3 - 0.5 * t2

    return (w_m1, w_0, w_1, w_2)
end

@inline function _walk_cell_index(grid::Vector{Float64}, value::Float64, previous_idx::Int)
    n = length(grid)
    n >= 2 || return 1

    i = clamp(previous_idx, 1, n - 1)
    @inbounds begin
        if value < grid[i]
            while i > 1 && value < grid[i]
                i -= 1
            end
        elseif value >= grid[i + 1]
            while i < (n - 1) && value >= grid[i + 1]
                i += 1
            end
        end
    end
    return i
end

@inline function _resolve_species_indices(species::Vector{Symbol})
    key = objectid(species)
    lock(_SPECIES_INDEX_CACHE_LOCK)
    try
        if haskey(_SPECIES_INDEX_CACHE, key)
            return _SPECIES_INDEX_CACHE[key]
        end

        iH1 = findfirst(==(:H1), species)
        iHe4 = findfirst(==(:He4), species)
        iH1 === nothing && error("Opacity requires species :H1")
        iHe4 === nothing && error("Opacity requires species :He4")

        idx = (iH1::Int, iHe4::Int)
        _SPECIES_INDEX_CACHE[key] = idx
        return idx
    finally
        unlock(_SPECIES_INDEX_CACHE_LOCK)
    end
end

@inline function _build_eval_state(lnT::TT, lnρ::TT, xa::AbstractVector{<:TT}, species::Vector{Symbol}) where {TT<:Real}
    logT = lnT * INV_LN10
    logρ = lnρ * INV_LN10
    logR = logρ - 3 * logT + 18

    iH1, iHe4 = _resolve_species_indices(species)
    X = xa[iH1]
    Y = xa[iHe4]
    Z = one(TT) - X - Y

    return _OpacityEvalState(logT, logR, ForwardDiff.value(logT), ForwardDiff.value(logR), X, Z,
                             ForwardDiff.value(X), ForwardDiff.value(Z))
end

@inline function _get_log_kappa_per_table(table::RT_table_opacity, eval_state::_OpacityEvalState{T}, cache::_OpacityLookupCache) where {T<:Real}
    min_T, max_T = table.logTs[1], table.logTs[end]
    min_R, max_R = table.logRs[1], table.logRs[end]

    eff_logT = clamp(eval_state.logT, min_T, max_T)
    eff_logR = clamp(eval_state.logR, min_R, max_R)
    eff_val_logT = clamp(eval_state.val_logT, min_T, max_T)
    eff_val_logR = clamp(eval_state.val_logR, min_R, max_R)

    i_T = _walk_cell_index(table.logTs, eff_val_logT, cache.iT)
    i_R = _walk_cell_index(table.logRs, eff_val_logR, cache.iR)
    cache.iT = i_T
    cache.iR = i_R

    @inbounds begin
        u = (eff_logT - table.logTs[i_T]) * table.inv_dlogT[i_T]
        v = (eff_logR - table.logRs[i_R]) * table.inv_dlogR[i_R]

        wt = cubic_weights(u)
        wr = cubic_weights(v)

        nR = length(table.logRs)
        nT = length(table.logTs)
        log_kappa = zero(T)

        for it in 1:4
            real_iT = clamp(i_T + _OPACITY_OFFSETS[it], 1, nT)
            base = (real_iT - 1) * nR
            wt_i = wt[it]
            for ir in 1:4
                real_iR = clamp(i_R + _OPACITY_OFFSETS[ir], 1, nR)
                log_kappa += table.kap_data[base + real_iR] * wt_i * wr[ir]
            end
        end

        return log_kappa
    end
end

function get_log_kappa_per_table(table::RT_table_opacity, val_logT::Float64, val_logR::Float64, logT::T, logR::T) where {T<:Real}
    eval_state = _OpacityEvalState(logT, logR, val_logT, val_logR, zero(T), zero(T), 0.0, 0.0)
    return _get_log_kappa_per_table(table, eval_state, _OpacityLookupCache())
end

@inline function _get_log_kappa_collection(collection::Opacity_table_collector, eval_state::_OpacityEvalState{TT}, cache::_OpacityLookupCache) where {TT<:Real}
    min_X, max_X = collection.Xs[1], collection.Xs[end]
    min_Z, max_Z = collection.Zs[1], collection.Zs[end]

    eff_X = clamp(eval_state.X, min_X, max_X)
    eff_Z = clamp(eval_state.Z, min_Z, max_Z)
    eff_val_X = clamp(eval_state.val_X, min_X, max_X)
    eff_val_Z = clamp(eval_state.val_Z, min_Z, max_Z)

    ix = _walk_cell_index(collection.Xs, eff_val_X, cache.ix)
    iz = _walk_cell_index(collection.Zs, eff_val_Z, cache.iz)
    cache.ix = ix
    cache.iz = iz

    @inbounds begin
        u_X = (eff_X - collection.Xs[ix]) * collection.inv_dX[ix]
        u_Z = (eff_Z - collection.Zs[iz]) * collection.inv_dZ[iz]

        wX = cubic_weights(u_X)
        wZ = cubic_weights(u_Z)

        Nx = length(collection.Xs)
        Nz = length(collection.Zs)
        log_kappa_final = zero(TT)

        for itx in 1:4
            real_ix = clamp(ix + _OPACITY_OFFSETS[itx], 1, Nx)
            wx = wX[itx]
            for itz in 1:4
                real_iz = clamp(iz + _OPACITY_OFFSETS[itz], 1, Nz)
                table = collection.tables[real_ix, real_iz]
                table_log_kappa = _get_log_kappa_per_table(table, eval_state, cache)
                log_kappa_final += table_log_kappa * wx * wZ[itz]
            end
        end

        return log_kappa_final
    end
end

function get_opacity_table_collection(collection::Opacity_table_collector, lnT::TT, lnρ::TT,
                                      xa::AbstractVector{<:TT}, species::Vector{Symbol})::TT where {TT<:Real}
    eval_state = _build_eval_state(lnT, lnρ, xa, species)
    log_kappa = _get_log_kappa_collection(collection, eval_state, _OpacityLookupCache())
    return 10^log_kappa
end

# Smoothing function to interpolate smoothly between the two datasets
# Using the function f(t)=3t^2 - 2t^3
@inline function smooth_step_func(x::T, floor::Float64, ceil::Float64) where T
    if x <= floor
        return zero(T)
    elseif x >= ceil
        return one(T)
    else
        t = (x - floor) / (ceil - floor)

        return t * t * (3.0 - 2.0 * t)
    end
end

function Jems.Opacity.get_opacity_resultsTρ(composite::CompositeOpacity, lnT::TT, lnρ::TT,
                                            xa::AbstractVector{<:TT}, species::Vector{Symbol})::TT where {TT<:Real}
    eval_state = _build_eval_state(lnT, lnρ, xa, species)
    val_logT = eval_state.val_logT

    if val_logT >= composite.trans_logT_max
        log_κ = _get_log_kappa_collection(composite.high_T_collector, eval_state, _OpacityLookupCache())
        return 10^log_κ
    elseif val_logT <= composite.trans_logT_min
        log_κ = _get_log_kappa_collection(composite.low_T_collector, eval_state, _OpacityLookupCache())
        return 10^log_κ
    else
        low_cache = _OpacityLookupCache()
        high_cache = _OpacityLookupCache()

        log_κ_low = _get_log_kappa_collection(composite.low_T_collector, eval_state, low_cache)
        log_κ_high = _get_log_kappa_collection(composite.high_T_collector, eval_state, high_cache)

        w = smooth_step_func(eval_state.logT, composite.trans_logT_min, composite.trans_logT_max)
        smooth_log_κ = (one(TT) - w) * log_κ_low + w * log_κ_high

        return 10^smooth_log_κ
    end
end
