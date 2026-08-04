# Current TensorKit control for TeNeT's typed operation matrix.
# Usage:
#   julia --project=benchmarks/tensorkit_benchmark \
#       benchmarks/tensorkit_microbench.jl [degeneracy] [min_ms]

using LinearAlgebra
using AppleAccelerate
using TensorKit

BLAS.set_num_threads(1)

const DEGENERACY = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
const MIN_MS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300

function sample(f, symmetry, operation, form, first_phase, min_ms)
    first = @timed f()
    println("$symmetry,$operation,$form,$first_phase,1,$(first.time * 1e6),$(first.bytes)")

    f()
    f()
    iterations = 0
    start = time_ns()
    min_ns = min_ms * 1_000_000
    while iterations < 2 || time_ns() - start < min_ns
        f()
        iterations += 1
    end
    elapsed = time_ns() - start
    bytes = @allocated f()
    println("$symmetry,$operation,$form,warm,$iterations,$(elapsed / 1_000 / iterations),$bytes")
    return first.value
end

function contract_input_swap(A, B)
    @tensor C[a b; g h] := A[a b; c d] * B[d c; g h]
    return C
end

function contract_identity(A, B)
    @tensor C[a b; g h] := A[a b; c d] * B[c d; g h]
    return C
end

function contract_identity!(C, A, B)
    @tensor C[a b; g h] = A[a b; c d] * B[c d; g h]
    return C
end

function contract_input_swap!(C, A, B)
    @tensor C[a b; g h] = A[a b; c d] * B[d c; g h]
    return C
end

function contract_input_output_swap(A, B)
    @tensor C[b a; g h] := A[a b; c d] * B[d c; g h]
    return C
end

function contract_input_output_swap!(C, A, B)
    @tensor C[b a; g h] = A[a b; c d] * B[d c; g h]
    return C
end

function run_case(symmetry, V)
    A = randn(Float64, V ⊗ V ← V ⊗ V)
    B = randn(Float64, V ⊗ V ← V ⊗ V)
    P = randn(Float64, V ⊗ V ← V)

    permutation = ((2,), (3, 1))
    expected_permuted = sample(
        () -> permute(P, permutation),
        symmetry,
        "permute",
        "owned",
        "process_first_for_row",
        MIN_MS,
    )
    @assert isfinite(norm(expected_permuted))
    permuted_destination = similar(expected_permuted)
    actual_permuted = sample(
        () -> permute!(permuted_destination, P, permutation),
        symmetry,
        "permute",
        "destination",
        "first_after_setup",
        MIN_MS,
    )
    permutation_error =
        norm(actual_permuted - expected_permuted) / max(norm(expected_permuted), eps(Float64))
    @assert permutation_error <= 256eps(Float64)

    expected_transposed = sample(
        () -> transpose(P),
        symmetry,
        "transpose",
        "owned",
        "process_first_for_row",
        MIN_MS,
    )
    @assert isfinite(norm(expected_transposed))
    transposed_destination = similar(expected_transposed)
    actual_transposed = sample(
        () -> transpose!(transposed_destination, P),
        symmetry,
        "transpose",
        "destination",
        "first_after_setup",
        MIN_MS,
    )
    transpose_error =
        norm(actual_transposed - expected_transposed) / max(norm(expected_transposed), eps(Float64))
    @assert transpose_error <= 256eps(Float64)

    composed = sample(
        () -> A * B,
        symmetry,
        "compose",
        "owned",
        "process_first_for_row",
        MIN_MS,
    )
    @assert isfinite(norm(composed))

    for (operation, owned, destination) in (
        (
            "contract_identity",
            () -> contract_identity(A, B),
            C -> contract_identity!(C, A, B),
        ),
        (
            "contract_input_swap",
            () -> contract_input_swap(A, B),
            C -> contract_input_swap!(C, A, B),
        ),
        (
            "contract_input_output_swap",
            () -> contract_input_output_swap(A, B),
            C -> contract_input_output_swap!(C, A, B),
        ),
    )
        expected = sample(
            owned,
            symmetry,
            operation,
            "owned",
            "process_first_for_row",
            MIN_MS,
        )
        @assert isfinite(norm(expected))
        if operation == "contract_identity"
            compose_error = norm(expected - composed) / max(norm(composed), eps(Float64))
            @assert compose_error <= 256eps(Float64)
        end

        C = similar(expected)
        actual = sample(
            () -> destination(C),
            symmetry,
            operation,
            "destination",
            "first_after_setup",
            MIN_MS,
        )
        error = norm(actual - expected) / max(norm(expected), eps(Float64))
        @assert error <= 256eps(Float64)
    end
end

println("# tensorkit_authority=f87ca7fe557abbc79561d23298028664ed5dbcd6")
println("# tensorKit_version=$(pkgversion(TensorKit)) julia_version=$(VERSION)")
println("# apple_accelerate_version=$(pkgversion(AppleAccelerate))")
println("# host_kernel=$(Sys.KERNEL) machine=$(Sys.MACHINE) cpu_reported_by_julia=$(Sys.CPU_NAME)")
println("# blas_threads=$(BLAS.get_num_threads()) blas_config=$(BLAS.get_config())")
println("# cold_scope=first call for this row; Julia compilation and process-global TensorKit caches may already be warm")
println("symmetry,operation,form,phase,iterations,us_per_iter,allocated_bytes")

d = DEGENERACY
run_case("U1", Vect[Irrep[U₁]](-1 => d, 0 => d, 1 => d))
run_case("SU2", Vect[Irrep[SU₂]](0 => d, 1 // 2 => d, 1 => d))
