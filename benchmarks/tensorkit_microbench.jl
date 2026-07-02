# Warm-cache contraction microbenchmark across symmetries.
#
# Mirrors `tenet-operations/examples/microbench_fusion.rs`: rank-4 tensors
# `A, B` in `V ⊗ V ← V ⊗ V` with a uniform degeneracy per sector, three
# workloads:
#
# - compose:  C[a b; g h] = A[a b; c d] * B[c d; g h]
# - swap:     C[a b; g h] = A[a b; c d] * B[d c; g h]
# - swap+out: C[b a; g h] = A[a b; c d] * B[d c; g h]
#
# Usage: julia --project=@. tensorkit_microbench.jl [degeneracy] [min_ms]

using TensorKit
using LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)

const DEGENERACY = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
const MIN_MS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300

function time_loop(f!, min_ms)
    f!(); f!(); f!()  # warm caches
    iters = 0
    start = time_ns()
    min_ns = min_ms * 1_000_000
    while time_ns() - start < min_ns
        f!()
        iters += 1
    end
    elapsed_ns = time_ns() - start
    return elapsed_ns / 1_000 / iters, iters
end

function run_case(name, V)
    A = randn(Float64, V ⊗ V ← V ⊗ V)
    B = randn(Float64, V ⊗ V ← V ⊗ V)
    C = similar(A)

    us, iters = time_loop(() -> mul!(C, A, B), MIN_MS)
    println("$name\tcompose(mul!)\t$(round(us, digits=2))\t$iters")

    us, iters = time_loop(MIN_MS) do
        @tensor C[a b; g h] = A[a b; c d] * B[c d; g h]
    end
    println("$name\tcompose\t$(round(us, digits=2))\t$iters")

    us, iters = time_loop(MIN_MS) do
        @tensor C[a b; g h] = A[a b; c d] * B[d c; g h]
    end
    println("$name\tswap\t$(round(us, digits=2))\t$iters")

    us, iters = time_loop(MIN_MS) do
        @tensor C[b a; g h] = A[a b; c d] * B[d c; g h]
    end
    println("$name\tswap+out\t$(round(us, digits=2))\t$iters")
end

d = DEGENERACY
println("# TensorKit $(pkgversion(TensorKit)) microbench: degeneracy=$d min_ms=$MIN_MS threads(BLAS)=1")
println("# symmetry\tworkload\tus_per_iter\titers")

run_case("U1", Vect[Irrep[U₁]](0 => d, 1 => d, -1 => d))
run_case("fZ2", Vect[FermionParity](0 => d, 1 => d))
run_case("SU2", Vect[Irrep[SU₂]](0 => d, 1 // 2 => d, 1 => d))
run_case("U1xfZ2", Vect[Irrep[U₁] ⊠ FermionParity]((0, 0) => d, (1, 1) => d, (-1, 1) => d))
