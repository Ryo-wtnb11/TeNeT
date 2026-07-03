# Prints per-sector singular values for deterministic fusion tensors so they
# can be compared against TeNeT (`tenet-operations/examples/tsvd_crosscheck.rs`).
#
# Both sides fill every fusion-tree pair block with the same integer-hash
# function of the sector labels and one-based degeneracy indices.

using TensorKit
using LinearAlgebra: diag

const DEGENERACY = 2

fill_value(l1, l2, m1, m2, lc, i1, i2, j1, j2) =
    Float64(mod(3 + 7l1 + 11l2 + 13m1 + 17m2 + 19lc + 23i1 + 29i2 + 31j1 + 37j2, 41) - 20)

label(c::U1Irrep) = Int(c.charge)
label(c::SU2Irrep) = Int(2 * c.j)

function run_case(name, V)
    t = zeros(Float64, V ⊗ V ← V ⊗ V)
    for (f1, f2) in fusiontrees(t)
        block = t[f1, f2]
        l1, l2 = label.(f1.uncoupled)
        m1, m2 = label.(f2.uncoupled)
        lc = label(f1.coupled)
        for j2 in axes(block, 4), j1 in axes(block, 3), i2 in axes(block, 2), i1 in axes(block, 1)
            block[i1, i2, j1, j2] = fill_value(l1, l2, m1, m2, lc, i1, i2, j1, j2)
        end
    end
    _, S, _ = svd_compact(t)
    entries = sort([(label(c), sort(diag(block(S, c)), rev=true)) for c in blocksectors(S)])
    for (lab, values) in entries
        println(name, "\t", lab, "\t", join([string(round(v, digits=10)) for v in values], ","))
    end
end

run_case("U1", Vect[Irrep[U₁]](0 => DEGENERACY, 1 => DEGENERACY, -1 => DEGENERACY))
run_case("SU2", Vect[Irrep[SU₂]](0 => DEGENERACY, 1 // 2 => DEGENERACY))
