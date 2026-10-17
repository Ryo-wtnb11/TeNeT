# Offline generator for tensorkit-0.17.1-table-reductions.txt. Nothing in the
# shipped crates runs this; it is a manual step.
#
# Reproduce in a scratch project (NOT this directory):
#
#   julia --project=/tmp/fixture -e 'using Pkg; Pkg.add(name="TensorKit", version="0.17.1"); Pkg.add(name="TensorKitSectors", version="0.3.9")'
#   julia --project=/tmp/fixture path/to/tensorkit_table_reductions.jl > path/to/tensorkit-0.17.1-table-reductions.txt
#
# TensorKit's own `tr`, `norm`, `inner`, `scalar` and `dim` on Fibonacci and
# Ising tensors whose every entry is a function of its fusion-tree pair and
# degeneracy index alone (`entry` below; `table_reductions.rs` computes the
# same function). Sectors are written with AnyonWiki's labels, which
# `TableRule` uses: Fibonacci 1 = I, 2 = τ; Ising 1 = I, 2 = ψ, 3 = σ.
#
# AnyonWiki's Ising gauge differs from TensorKit's by the vertex sign
# u^{σσ}_ψ = -1 (#2277). Every printed quantity is invariant under it:
# `norm` and `inner` multiply an entry by the conjugate of an entry with the
# same trees, `tr` reads entries whose codomain and domain trees agree, and
# the scalar's only coupled sector is the unit, whose trees have no ψ vertex.
# Values use Julia's shortest round-trip `repr`.
using Pkg
using TensorKit

dep(uuid) = Pkg.dependencies()[Base.UUID(uuid)]
for (name, uuid) in (("TensorKit", "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"),
                     ("TensorKitSectors", "13a9c161-d5da-41f0-bcbd-e1a08ae0647f"))
    d = dep(uuid)
    println("# ", name, " v", d.version, " tree ", d.tree_hash, " (", basename(d.source), ")")
end
println("# rows: category quantity re im")

function fnv1a(s::AbstractString)
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    return h
end
unit_interval(h::UInt64) = Float64(h >> 11) / 2.0^53 - 0.5

join_labels(label, xs) = join((label(x) for x in xs), ",")

function entry(label, tag, f₁, f₂, index)
    key = string(tag, "|", label(f₁.coupled), "|",
                 join_labels(label, f₁.uncoupled), "|", join_labels(label, f₁.innerlines), "|",
                 join_labels(label, f₂.uncoupled), "|", join_labels(label, f₂.innerlines), "|",
                 join((i - 1 for i in Tuple(index)), ","))
    return complex(unit_interval(fnv1a(key * "|re")), unit_interval(fnv1a(key * "|im")))
end

function filled(label, tag, space)
    t = zeros(ComplexF64, space)
    for (f₁, f₂) in fusiontrees(t)
        block = t[f₁, f₂]
        for index in CartesianIndices(block)
            block[index] = entry(label, tag, f₁, f₂, index)
        end
    end
    return t
end

emit(category, quantity, z) =
    println(category, " ", quantity, " ", repr(real(z)), " ", repr(imag(z)))

function report(category, label, V)
    t = filled(label, "t", V ⊗ V ← V ⊗ V)
    s = filled(label, "s", V ⊗ V ← V ⊗ V)
    x = filled(label, "x", one(V) ← V)
    y = filled(label, "y", V ← one(V))
    emit(category, "tr", tr(t))
    emit(category, "norm2", norm(t, 2))
    emit(category, "norm1", norm(t, 1))
    emit(category, "norminf", norm(t, Inf))
    emit(category, "inner", inner(s, t))
    emit(category, "scalar", scalar(x * y))
    emit(category, "dim", dim(V))
end

const fib_labels = Dict(FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2)
report("fibonacci", a -> fib_labels[a],
       Vect[FibonacciAnyon](FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2))

const ising_labels = Dict(IsingAnyon(:I) => 1, IsingAnyon(:ψ) => 2, IsingAnyon(:σ) => 3)
report("ising", a -> ising_labels[a],
       Vect[IsingAnyon](IsingAnyon(:I) => 1, IsingAnyon(:ψ) => 1, IsingAnyon(:σ) => 2))
