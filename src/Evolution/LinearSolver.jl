using LinearAlgebra

##
# Bicyclic (cyclic reduction) algorithm following
# BCYCLIC: A parallel block tridiagonal matrix cyclic solver (Hirshman et al. 2010)
function ensure_cyclic_level_storage!(solver_data, level, n_even, matrix_sample, vector_sample)
    while length(solver_data.cr_even_L) < level
        push!(solver_data.cr_even_L, Vector{typeof(matrix_sample)}())
        push!(solver_data.cr_even_U, Vector{typeof(matrix_sample)}())
        push!(solver_data.cr_even_b, Vector{typeof(vector_sample)}())
    end

    level_L = solver_data.cr_even_L[level]
    level_U = solver_data.cr_even_U[level]
    level_b = solver_data.cr_even_b[level]

    resize!(level_L, n_even)
    resize!(level_U, n_even)
    resize!(level_b, n_even)

    for i in 1:n_even
        if !isassigned(level_L, i)
            level_L[i] = similar(matrix_sample)
        end
        if !isassigned(level_U, i)
            level_U[i] = similar(matrix_sample)
        end
        if !isassigned(level_b, i)
            level_b[i] = similar(vector_sample)
        end
    end
    return level_L, level_U, level_b
end

function block_tridiagonal_solver!(sm, ::StellarModels.ThomasSolverData)
    eqs_numbers = sm.solver_data.eqs_numbers
    jacobian_D = sm.solver_data.jacobian_D
    jacobian_L = sm.solver_data.jacobian_L
    jacobian_U = sm.solver_data.jacobian_U
    solver_tmp1 = sm.solver_data.solver_tmp1
    solver_x = sm.solver_data.solver_x
    solver_β = sm.solver_data.solver_β
    solver_corr = sm.solver_data.solver_corr

    # Simple row preconditioning, divide each row by its maximum value
    # this is turned off by default as it seems to not change much. If we end up
    # using at some point it needs to be optimized
    if sm.opt.solver.use_preconditioning
        for i in 1:sm.props.nz
            for j in 1:sm.nvars
                maxval = 0
                for k in 1:sm.nvars
                    if i != 1
                        if abs(jacobian_L[i][j,k]) > maxval
                            maxval = abs(jacobian_L[i][j,k])
                        end
                    end
                    if abs(jacobian_D[i][j,k]) > maxval
                        maxval = abs(jacobian_D[i][j,k])
                    end
                    if i != sm.props.nz
                        if abs(jacobian_U[i][j,k]) > maxval
                            maxval = abs(jacobian_U[i][j,k])
                        end
                    end
                end
                for k in 1:sm.nvars
                    jacobian_L[i][j,k] = jacobian_L[i][j,k]/maxval
                    jacobian_D[i][j,k] = jacobian_D[i][j,k]/maxval
                    jacobian_U[i][j,k] = jacobian_U[i][j,k]/maxval
                end
                eqs_numbers[(i-1)*sm.nvars + j] = eqs_numbers[(i-1)*sm.nvars + j]/maxval
            end
        end
        # Simple column-preconditioning. Divide all columns by their maximum value. This means
        # we need to rescale the correction afterwards
        for i in 1:sm.props.nz
            for j in 1:sm.nvars
                maxval = 0
                for k in 1:sm.nvars
                    if i != sm.props.nz
                        if abs(jacobian_L[i+1][k,j]) > maxval
                            maxval = abs(jacobian_L[i+1][k,j])
                        end
                    end
                    if abs(jacobian_D[i][k,j]) > maxval
                        maxval = abs(jacobian_D[i][k,j])
                    end
                    if i != 1
                        if abs(jacobian_U[i-1][k,j]) > maxval
                            maxval = abs(jacobian_U[i-1][k,j])
                        end
                    end
                end
                for k in 1:sm.nvars
                    if i != sm.props.nz
                        jacobian_L[i+1][k,j] = jacobian_L[i+1][k,j]/maxval
                    end
                    jacobian_D[i][k,j] = jacobian_D[i][k,j]/maxval
                    if i != 1
                        jacobian_U[i-1][k,j] = jacobian_U[i-1][k,j]/maxval
                    end
                end
                sm.solver_data.preconditioning_factor[(i-1)*sm.nvars + j] = 1/maxval
            end
        end
    else
        sm.solver_data.preconditioning_factor .= 1.0
    end

    n = sm.props.nz
    nvars = sm.nvars
    n_original = n

    for i in 1:n
        for j in 1:nvars
            solver_β[i][j] = -eqs_numbers[(i-1)*nvars + j]
        end
    end

    empty!(sm.solver_data.cr_levels)

    tmp_mat = solver_tmp1[1]
    tmp_vec = similar(solver_β[1])

    level = 0
    while n > 1
        level += 1
        push!(sm.solver_data.cr_levels, n)
        n_even = n ÷ 2
        n_odd = n - n_even

        level_L, level_U, level_b = ensure_cyclic_level_storage!(
            sm.solver_data, level, n_even, jacobian_D[1], solver_β[1]
        )

        for k in 1:n_even
            i = 2 * k
            # safe to factor in place: even-row diagonals are not reused after this stage and
            # the Jacobian is rebuilt each Newton step, so destructive updates are acceptable
            LU = lu!(jacobian_D[i])
            ldiv!(level_L[k], LU, jacobian_L[i])
            ldiv!(level_U[k], LU, jacobian_U[i])
            ldiv!(level_b[k], LU, solver_β[i])
        end

        for k in 1:n_odd
            i = 2 * k - 1
            D_next = jacobian_D[k]
            L_next = jacobian_L[k]
            U_next = jacobian_U[k]
            b_next = solver_β[k]

            copyto!(D_next, jacobian_D[i])
            copyto!(b_next, solver_β[i])

            if i > 1
                L_i = jacobian_L[i]
                prev_even_idx = k - 1
                mul!(tmp_mat, L_i, level_U[prev_even_idx])
                D_next .-= tmp_mat
                mul!(tmp_mat, L_i, level_L[prev_even_idx])
                L_next .= -tmp_mat
                mul!(tmp_vec, L_i, level_b[prev_even_idx])
                b_next .-= tmp_vec
            else
                fill!(L_next, 0)
            end

            if i < n
                U_i = jacobian_U[i]
                next_even_idx = k
                mul!(tmp_mat, U_i, level_L[next_even_idx])
                D_next .-= tmp_mat
                mul!(tmp_mat, U_i, level_U[next_even_idx])
                U_next .= -tmp_mat
                mul!(tmp_vec, U_i, level_b[next_even_idx])
                b_next .-= tmp_vec
            else
                fill!(U_next, 0)
            end
        end

        n = n_odd
    end

    LU = lu!(jacobian_D[1])
    ldiv!(solver_x[1], LU, solver_β[1])

    x_odd = solver_x
    # reuse RHS buffer for reconstructed full solution at each level (RHS no longer needed after reduction)
    x_full_buffer = solver_β
    # track which buffer currently holds the odd-indexed solution as we swap buffers each level
    x_odd_is_solver_x = true

    for level_index in length(sm.solver_data.cr_levels):-1:1
        n_level = sm.solver_data.cr_levels[level_index]
        n_even = n_level ÷ 2
        n_odd = n_level - n_even

        level_L = sm.solver_data.cr_even_L[level_index]
        level_U = sm.solver_data.cr_even_U[level_index]
        level_b = sm.solver_data.cr_even_b[level_index]

        for k in 1:n_odd
            x_full_buffer[2 * k - 1] .= x_odd[k]
        end

        for k in 1:n_even
            i = 2 * k
            x_even = x_full_buffer[i]
            x_even .= level_b[k]
            mul!(tmp_vec, level_L[k], x_odd[k])
            x_even .-= tmp_vec
            if k + 1 <= n_odd
                mul!(tmp_vec, level_U[k], x_odd[k+1])
                x_even .-= tmp_vec
            end
        end

        # after each level, the reconstructed full solution becomes the odd-indexed solution for the next level
        x_odd, x_full_buffer = x_full_buffer, x_odd
        x_odd_is_solver_x = !x_odd_is_solver_x
    end

    if !x_odd_is_solver_x # buffers swapped; copy back into solver_x for downstream consumers
        for i in 1:n_original
            solver_x[i] .= x_odd[i]
        end
    end

    # unload result into solver_corr
    for i=1:n_original
        for j=1:nvars
            solver_corr[(i-1)*nvars+j] = solver_x[i][j] *
                sm.solver_data.preconditioning_factor[(i-1)*nvars + j]
        end
    end
    return
end
