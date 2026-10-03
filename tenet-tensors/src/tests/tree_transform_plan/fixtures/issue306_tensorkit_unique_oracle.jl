using TensorKit
using Printf

const TK = TensorKit
const TKS = TensorKit.TensorKitSectors
const Z4 = Z4Element{2}

fmt(z) = @sprintf("%.17g%+.17gim", real(z), imag(z))
sectorlabels(values) = Tuple(x.n for x in values)
labels(tree) = sectorlabels(tree.uncoupled)

println("TensorKit_commit=cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91")
println("TensorKit_version=", pkgversion(TensorKit))
println("TensorKitSectors_version=", pkgversion(TKS))

source3 = FusionTree{Z4}(
    (Z4(1), Z4(2), Z4(3)),
    Z4(2),
    (false, false, false),
    (Z4(3),),
    (1, 1),
)
for inv in (false, true)
    output, coefficient = TK.artin_braid(source3, 2; inv)
    println(inv ? "inverse3.uncoupled=" : "forward3.uncoupled=", labels(output))
    println(inv ? "inverse3.inner=" : "forward3.inner=", sectorlabels(output.innerlines))
    println(inv ? "inverse3.coeff=" : "forward3.coeff=", fmt(coefficient))
end

source3_domain = FusionTree{Z4}((Z4(2),), Z4(2), (false,), (), ())
for levels in (((1, 2, 3), (4,)), ((1, 3, 2), (4,)))
    output, coefficient = TK.braid(
        (source3, source3_domain),
        ((1, 3, 2), (4,)),
        levels,
    )
    println("pair3.levels=", levels)
    println("pair3.codomain.uncoupled=", labels(output[1]))
    println("pair3.codomain.inner=", sectorlabels(output[1].innerlines))
    println("pair3.domain.uncoupled=", labels(output[2]))
    println("pair3.coeff=", fmt(coefficient))
end

z4_out = FusionTree{Z4}(
    (Z4(1), Z4(2)),
    Z4(3),
    (false, true),
    (),
    (1,),
)
z4_in = FusionTree{Z4}((Z4(3),), Z4(3), (true,), (), ())
z4_transposed, z4_coefficient = transpose((z4_out, z4_in), ((2, 3), (1,)))
println("transpose.codomain.uncoupled=", labels(z4_transposed[1]))
println("transpose.codomain.isdual=", z4_transposed[1].isdual)
println("transpose.domain.uncoupled=", labels(z4_transposed[2]))
println("transpose.domain.isdual=", z4_transposed[2].isdual)
println("transpose.coeff=", fmt(z4_coefficient))
println("sector1.fs=", fmt(frobenius_schur_phase(Z4(1))))
println(
    "sector1.A11=",
    fmt(TKS.Asymbol(Z4(1), Z4(1), only(Z4(1) ⊗ Z4(1)))),
)

const F = FermionParity
odd = F(1)
fz2_out = FusionTree{F}((odd,), odd, (false,), (), ())
fz2_in = FusionTree{F}((odd,), odd, (true,), (), ())
fz2_pair, fz2_coefficient = TK.braid(
    (fz2_out, fz2_in),
    ((2,), (1,)),
    ((1,), (2,)),
)
println("fz2.codomain.isdual=", fz2_pair[1].isdual)
println("fz2.domain.isdual=", fz2_pair[2].isdual)
println("fz2.domain_crossing.coeff=", fmt(fz2_coefficient))
