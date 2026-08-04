# TensorKit semantic oracle for the TeNeT verification suite (issue #9).
#
# Section 1 dumps F/R/twist/frobenius_schur_phase symbol tables; a
# representative subset is hardcoded in `tenet-core/tests/semantic_axioms.rs`.
# Section 2 runs TensorKit's own pentagon/hexagon checkers over the same
# sector windows the Rust axiom sweep uses (assurance that both sides check
# the same equations, TensorKitSectors/src/sectors.jl `pentagon_equation` /
# `hexagon_equation`).
# Section 3 produces the basis-independent invariant stream (norm, tr,
# singular values) for the seeded operation sequence mirrored by
# `cross_library_invariant_stream_*` in `tenet/tests/semantic_suite.rs`.
# Section 4 records planar repartition fixtures consumed by the typed-facade
# regression tests in `tenet/tests/typed_facade.rs`.
# Section 6 records the disjoint U(1) zero-map null-space completion consumed
# by `tenet/tests/tk_disjoint_null.rs`.
#
# Run (Julia 1.11.6, pinned TensorKit main `f87ca7f`):
#   julia --project=benchmarks/tensorkit_oracle \
#     benchmarks/tensorkit_semantic_oracle.jl
#
# The committed reference output and SHA-256 manifest live next to this script.

using TensorKit
using TensorKit: pentagon_equation, hexagon_equation
using LinearAlgebra: diag, tr
using Printf

const fZ2 = FermionParity
const I3 = FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂]

su2(twoj) = SU2Irrep(twoj // 2)
triple(p, q, twoj) = fZ2(p) ⊠ U1Irrep(q) ⊠ su2(twoj)

fmt(x) = @sprintf("%.15e", x)

# ---------------------------------------------------------------------------
# Section 1: symbol tables
# ---------------------------------------------------------------------------
println("== section 1: symbol tables ==")

println("-- fZ2 --")
for p in 0:1
    a = fZ2(p)
    println("twist fZ2 $p = ", fmt(twist(a)), "  fs = ", fmt(frobenius_schur_phase(a)))
end
for pa in 0:1, pb in 0:1
    a, b = fZ2(pa), fZ2(pb)
    c = first(a ⊗ b)
    println("R fZ2 $pa $pb = ", fmt(Rsymbol(a, b, c)))
end

println("-- SU2 (labels are 2j) --")
for twoj in 0:6
    a = su2(twoj)
    println(
        "dim SU2 $twoj = ", fmt(dim(a)), "  twist = ", fmt(twist(a)),
        "  fs = ", fmt(frobenius_schur_phase(a)),
    )
end
for ja in 0:4, jb in 0:4, jc in 0:4
    a, b = su2(ja), su2(jb)
    c = su2(jc)
    if c in a ⊗ b
        println("R SU2 $ja $jb $jc = ", fmt(Rsymbol(a, b, c)))
    end
end
# All admissible F sextuples with every label 2j <= 3.
for ja in 0:3, jb in 0:3, jc in 0:3, jd in 0:3, je in 0:3, jf in 0:3
    a, b, c, d = su2(ja), su2(jb), su2(jc), su2(jd)
    e, f = su2(je), su2(jf)
    (e in a ⊗ b && d in e ⊗ c && f in b ⊗ c && d in a ⊗ f) || continue
    println("F SU2 $ja $jb $jc $jd $je $jf = ", fmt(Fsymbol(a, b, c, d, e, f)))
end

println("-- fZ2 x U1 x SU2 (labels: parity charge 2j) --")
for (pa, qa, ja, pb, qb, jb) in (
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 0, 2, 0),
        (1, -1, 1, 1, 1, 1),
        (0, 0, 2, 0, 0, 2),
    )
    a, b = triple(pa, qa, ja), triple(pb, qb, jb)
    for c in a ⊗ b
        lc = (Int(c.sectors[1].isodd), Int(c.sectors[2].charge), Int(2 * c.sectors[3].j))
        println(
            "R I3 ($pa,$qa,$ja) ($pb,$qb,$jb) -> $lc = ",
            fmt(Rsymbol(a, b, c)),
        )
    end
end
for (p, q, twoj) in ((0, 0, 0), (1, 1, 1), (0, 2, 0), (1, -1, 3))
    a = triple(p, q, twoj)
    println(
        "twist I3 ($p,$q,$twoj) = ", fmt(twist(a)),
        "  fs = ", fmt(frobenius_schur_phase(a)),
        "  dim = ", fmt(dim(a)),
    )
end

# ---------------------------------------------------------------------------
# Section 2: TensorKit's own pentagon/hexagon over the Rust sweep windows
# ---------------------------------------------------------------------------
println("== section 2: TensorKit pentagon/hexagon sanity ==")

function sweep(name, sectors)
    for a in sectors, b in sectors, c in sectors
        @assert hexagon_equation(a, b, c; atol = 1.0e-12) "hexagon failed: $name $a $b $c"
        for d in sectors
            @assert pentagon_equation(a, b, c, d; atol = 1.0e-12) "pentagon failed: $name $a $b $c $d"
        end
    end
    println("$name: pentagon+hexagon OK over $(length(sectors)) sectors")
end

sweep("Z2", [Z2Irrep(0), Z2Irrep(1)])
sweep("fZ2", [fZ2(0), fZ2(1)])
sweep("U1", [U1Irrep(q) for q in -4:4])
sweep("SU2", [su2(twoj) for twoj in 0:6])
sweep("U1xfZ2", [U1Irrep(q) ⊠ fZ2(p) for q in -2:2 for p in 0:1])
sweep("fZ2xU1xSU2", [triple(p, q, twoj) for p in 0:1 for q in -1:1 for twoj in 0:1])

# ---------------------------------------------------------------------------
# Section 3: cross-library invariant stream
# ---------------------------------------------------------------------------
# Rank-4 tensors in V ⊗ V ← V ⊗ V, every fusion-tree pair block filled with
# the same integer function of the sector labels and one-based degeneracy
# indices (same convention as tensorkit_tsvd_crosscheck.jl, which already
# validates block/tree alignment between the two libraries).
println("== section 3: invariant stream ==")

fill_value(c0, l1, l2, m1, m2, lc, i1, i2, j1, j2) = Float64(
    mod(c0 + 7l1 + 11l2 + 13m1 + 17m2 + 19lc + 23i1 + 29i2 + 31j1 + 37j2, 41) - 20,
)

label(c::U1Irrep) = Int(c.charge)
label(c::SU2Irrep) = Int(2 * c.j)
label(c::I3) = 100 * Int(c.sectors[1].isodd) + 10 * Int(c.sectors[2].charge) +
    Int(2 * c.sectors[3].j)

function filled(V, c0)
    t = zeros(Float64, V ⊗ V ← V ⊗ V)
    for (f1, f2) in fusiontrees(t)
        block = t[f1, f2]
        l1, l2 = label.(f1.uncoupled)
        m1, m2 = label.(f2.uncoupled)
        lc = label(f1.coupled)
        for j2 in axes(block, 4), j1 in axes(block, 3),
                i2 in axes(block, 2), i1 in axes(block, 1)
            block[i1, i2, j1, j2] = fill_value(c0, l1, l2, m1, m2, lc, i1, i2, j1, j2)
        end
    end
    return t
end

function stream(name, V)
    println("-- $name --")
    inv(step, t) = println("$step norm = ", fmt(norm(t)), "  tr = ", fmt(real(tr(t))))
    A = filled(V, 3)
    B = filled(V, 5)
    inv("s1a", A)
    inv("s1b", B)
    C = A * B
    inv("s2", C)
    D = permute(A, ((2, 1), (4, 3)))
    inv("s3", D)
    E = D * C
    inv("s4", E)
    G = A' * A
    inv("s5", G)
    _, S, _ = svd_compact(E)
    values = sort(vcat([diag(b) for (_, b) in blocks(S)]...); rev = true)
    println("s6 svdcount = ", length(values))
    for (k, v) in enumerate(values)
        println("s6 svd[$k] = ", fmt(v))
    end
    H = E + 0.5 * A
    inv("s7", H)
    println("s8 tr(H*H) = ", fmt(real(tr(H * H))))
    return
end

stream("U1", U1Space(-1 => 2, 0 => 3, 1 => 2))
stream("SU2", SU2Space(0 => 2, 1 // 2 => 2, 1 => 1))
stream("fZ2xU1xSU2", Vect[I3]((0, 0, 0) => 2, (1, 1, 1 // 2) => 2, (0, 2, 0) => 1))

function fz2_map(V, even, odd)
    t = zeros(Float64, V ← V)
    for (c, b) in blocks(t)
        b[1, 1] = c.isodd ? odd : even
    end
    return t
end

println("-- fZ2 closed loops --")
Vf = Vect[fZ2](0 => 1, 1 => 1)
A, B, C = fz2_map(Vf, 1, 4), fz2_map(Vf, 2, 1.5), fz2_map(Vf, 0.5, 2.5)
D, T = fz2_map(Vf, 2, 3), fz2_map(Vf, 3, 2)
println("ordinary tr(D) = ", fmt(tr(D)))
println("twisted tr(D) = ", fmt(tr(twist(D, 1))))
@tensor loop_ab = A[i; j] * B[j; i]
@tensor loop_abc = A[i; j] * B[j; k] * C[k; i]
_, S, _ = svd_compact(T)
@tensor loop_asb = A[i; j] * S[j; k] * B[k; i]
@tensor loop_sab = S[i; j] * A[j; k] * B[k; i]
println("loop A*B = ", fmt(loop_ab))
println("loop A*B*C = ", fmt(loop_abc))
println("loop A*S*B = ", fmt(loop_asb))
println("loop S*A*B = ", fmt(loop_sab))

# ---------------------------------------------------------------------------
# Section 4: planar repartition fixtures
# ---------------------------------------------------------------------------
println("== section 4: planar repartition fixtures ==")

function sequential_tensor(A, B, C, D)
    tensor = zeros(Float64, A ⊗ B ← C ⊗ D)
    tensor.data .= eachindex(tensor.data)
    return tensor
end

function repartition_fixture(name, tensor, numout)
    output = repartition(tensor, numout)
    println("-- $name --")
    println("ranks = $(numout),$(numind(output) - numout)")
    println("data = [", join(fmt.(output.data), ", "), "]")
    return
end

u1_spaces = (
    U1Space(0 => 1), U1Space(0 => 2), U1Space(0 => 3), U1Space(0 => 4),
)
fz2_spaces = (
    Vect[fZ2](0 => 1, 1 => 1), Vect[fZ2](0 => 2, 1 => 1),
    Vect[fZ2](0 => 1, 1 => 2), Vect[fZ2](0 => 2, 1 => 2),
)
su2_spaces = (
    SU2Space(0 => 1, 1 // 2 => 1), SU2Space(0 => 1, 1 // 2 => 1),
    SU2Space(0 => 1, 1 // 2 => 1), SU2Space(0 => 1, 1 // 2 => 2),
)
i3_spaces = (
    Vect[I3]((0, 0, 0) => 1, (1, 0, 1 // 2) => 1),
    Vect[I3]((0, 0, 0) => 1, (1, 0, 1 // 2) => 1),
    Vect[I3]((0, 0, 0) => 1, (1, 0, 1 // 2) => 1),
    Vect[I3]((0, 0, 0) => 1, (1, 0, 1 // 2) => 2),
)

repartition_fixture("U1 2|2 -> 3|1", sequential_tensor(u1_spaces...), 3)
repartition_fixture("fZ2 2|2 -> 3|1", sequential_tensor(fz2_spaces...), 3)
repartition_fixture("SU2 2|2 -> 3|1", sequential_tensor(su2_spaces...), 3)
repartition_fixture("fZ2xU1xSU2 2|2 -> 3|1", sequential_tensor(i3_spaces...), 3)
repartition_fixture("fZ2 2|2 -> 1|3", sequential_tensor(fz2_spaces...), 1)
repartition_fixture("fZ2xU1xSU2 2|2 -> 1|3", sequential_tensor(i3_spaces...), 1)
repartition_fixture("fZ2xU1xSU2 2|2 -> 0|4", sequential_tensor(i3_spaces...), 0)
repartition_fixture("fZ2xU1xSU2 2|2 -> 4|0", sequential_tensor(i3_spaces...), 4)

# ---------------------------------------------------------------------------
# Section 5: scalar reductions and blockwise matrix exponential
# ---------------------------------------------------------------------------
println("== section 5: scalar reductions and exp ==")

function exp_fixture(::Type{T}, scale) where {T}
    V = U1Space(0 => 3, 1 => 2)
    t = zeros(T, V ← V)
    for (c, b) in blocks(t), j in axes(b, 2), i in axes(b, 1)
        re = 0.5 + 0.25 * (i - 1) - 0.75 * (j - 1) + 0.125 * Int(c.charge)
        value = T <: Complex ?
            complex(re, 0.125 * (i - 1) + 0.375 * (j - 1) - 0.25) : re
        b[i, j] = scale * value
    end
    return norm(t), norm(exp(t))
end

println("-- exp --")
for (T, scale) in ((Float64, 1.0), (Float64, 4.0), (ComplexF64, 1.0))
    input_norm, output_norm = exp_fixture(T, scale)
    @printf("%s %.1f %.17g %.17g\n", T, scale, input_norm, output_norm)
end

norm_real_fill(i, j) = 1.0 + 0.5 * (i - 1) - 0.25 * (j - 1)
norm_complex_fill(i, j) =
    norm_real_fill(i, j) + im * (0.5 + 0.125 * (i - 1) + 0.375 * (j - 1))

function norm_p_fixture(::Type{T}, V, W) where {T}
    t = zeros(T, V ← W)
    for (_, b) in blocks(t), j in axes(b, 2), i in axes(b, 1)
        b[i, j] = T <: Complex ? norm_complex_fill(i, j) : norm_real_fill(i, j)
    end
    return [norm(t, p) for p in (1.0, 2.0, 3.0, Inf)]
end

println("-- norm_p --")
norm_spaces = (
    ("U1", U1Space(-1 => 2, 0 => 3, 1 => 4), U1Space(-1 => 5, 0 => 1, 1 => 2)),
    (
        "SU2", SU2Space(0 => 2, 1 // 2 => 3, 1 => 4),
        SU2Space(0 => 5, 1 // 2 => 1, 1 => 2),
    ),
)
for (name, V, W) in norm_spaces, T in (Float64, ComplexF64)
    values = norm_p_fixture(T, V, W)
    @printf("%s %s", name, T)
    for value in values
        @printf(" %.17g", value)
    end
    println()
end

# ---------------------------------------------------------------------------
# Section 6: disjoint zero-map null-space completion
# ---------------------------------------------------------------------------
println("== section 6: disjoint U1 null spaces ==")

Vout = U1Space(0 => 2)
Vin = U1Space(1 => 3)
A0 = zeros(Float64, Vout ← Vin)
for (name, tensor) in (("direct", A0), ("adjoint", A0'))
    left = left_null(tensor)
    right = right_null(tensor)
    println("-- $name --")
    stored = tensor isa TensorMap ? length(tensor.data) : length(parent(tensor).data)
    @printf("source stored = %d\n", stored)
    @printf("left stored = %d norm = %.17g\n", length(left.data), norm(left))
    @printf("right stored = %d norm = %.17g\n", length(right.data), norm(right))
end
