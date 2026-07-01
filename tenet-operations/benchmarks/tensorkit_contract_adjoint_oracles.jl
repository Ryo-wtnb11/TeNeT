using TensorKit

const LHS_DATA = ComplexF64[
    1 + 1im,
    3 + 2im,
    2 - 1im,
    4 - 1im,
    -1 + 2im,
    2 - 2im,
    0 + 3im,
    -3 + 1im,
]

const RHS_DATA = ComplexF64[
    5 - 2im,
    7 - 4im,
    6 + 1im,
    8 + 2im,
    4 + 1im,
    1 - 3im,
    -2 + 2im,
    5 - 1im,
]

const EXPECTED = Dict(
    "lhs_conj" => ComplexF64[
        16 - 33im,
        44 - 8im,
        35 - 15im,
        41 + 24im,
        6 - 13im,
        -3 - 4im,
        18 + 10im,
        -10 + 4im,
    ],
    "rhs_conj" => ComplexF64[
        14 - 1im,
        34 + 6im,
        17 - 1im,
        43 + 10im,
        4 + 3im,
        14 - 6im,
        -10 + 14im,
        -8 + 6im,
    ],
    "both_conj" => ComplexF64[
        23 - 18im,
        33 + 11im,
        31 - 25im,
        44 + 15im,
        -6 - 15im,
        1 - 4im,
        13 + 7im,
        -5 - 11im,
    ],
)

const CONTRACT_CASES = (
    ("lhs_conj", ((2,), (1,)), true, ((1,), (2,)), false),
    ("rhs_conj", ((1,), (2,)), false, ((2,), (1,)), true),
    ("both_conj", ((2,), (1,)), true, ((2,), (1,)), true),
)

function run_case(::Type{I}, label, pA, conjA, pB, conjB) where {I}
    V = Vect[I](0 => 2, 1 => 2)
    A = zeros(ComplexF64, V ← V)
    B = zeros(ComplexF64, V ← V)
    C = zeros(ComplexF64, V ← V)
    A.data .= LHS_DATA
    B.data .= RHS_DATA

    TensorKit.TO.tensorcontract!(
        C,
        A,
        pA,
        conjA,
        B,
        pB,
        conjB,
        ((1,), (2,)),
        one(ComplexF64),
        zero(ComplexF64),
    )
    C.data == EXPECTED[label] || error("$I $label oracle mismatch: $(C.data)")
    return C.data
end

function print_oracle(::Type{I}, sector_label) where {I}
    for (label, pA, conjA, pB, conjB) in CONTRACT_CASES
        data = run_case(I, label, pA, conjA, pB, conjB)
        println("$sector_label,$label,$(collect(data))")
    end
end

print_oracle(Z2Irrep, "Z2Irrep")
print_oracle(FermionParity, "FermionParity")

function assert_close(label, actual, expected; atol = 1.0e-12)
    length(actual) == length(expected) || error("$label length mismatch")
    for (index, (a, e)) in enumerate(zip(actual, expected))
        abs(a - e) <= atol || error("$label[$index] mismatch: $a != $e")
    end
end

function print_su2_noncanonical_both_adjoint_oracle()
    V = SU2Space(1//2 => 2)
    A = zeros(ComplexF64, V ⊗ V ⊗ V ← V)
    B = zeros(ComplexF64, V ← V ⊗ V ⊗ V)
    A.data .= [
        complex(1 + 0.125 * (k - 1), -0.5 + 0.0625 * (k - 1)) for
            k in eachindex(A.data)
    ]
    B.data .= [
        complex(-3 + 0.25 * (k - 1), 0.75 - 0.03125 * (k - 1)) for
            k in eachindex(B.data)
    ]
    C = zeros(ComplexF64, V ← V)
    C.data .= ComplexF64[2 - 1im, -1 + 0.5im, 4 + 2im, -3 - 0.25im]
    TensorKit.TO.tensorcontract!(
        C,
        A,
        ((4,), (1, 2, 3)),
        true,
        B,
        ((2, 3, 4), (1,)),
        true,
        ((1,), (2,)),
        complex(-1.5, 0.25),
        complex(0.25, -0.125),
    )
    expected = ComplexF64[
        -64.5 + 35.4609375im,
        -89.0625 + 72.5859375im,
        -75.5 + 36.25390625im,
        -113.53125 + 79.56640625im,
    ]
    assert_close("SU2 noncanonical both_conj", C.data, expected)
    println("SU2Irrep,noncanonical_both_conj,$(collect(C.data))")
end

function print_product_fz2_u1_su2_contract_oracle()
    I = FermionParity ⊠ U1Irrep ⊠ SU2Irrep
    a = I((1, 1, 1//2))
    b = I((1, -1, 1//2))
    c0 = I((0, 0, 0))
    c1 = I((0, 0, 1))
    Va = Vect[I](a => 1)
    Vb = Vect[I](b => 1)
    Vc = Vect[I](c0 => 1, c1 => 1)
    A = zeros(ComplexF64, (Va ⊗ Vb) ← Vc)
    B = zeros(ComplexF64, (Vc ⊗ Va ⊗ Vb) ← one(Va))
    A.data .= ComplexF64[1 + 2im, 3 - 1im]
    B.data .= ComplexF64[-2 + 0.5im, 4 + 3im]
    C = TensorKit.TO.tensoralloc_contract(
        ComplexF64,
        A,
        ((1, 2), (3,)),
        false,
        B,
        ((1,), (2, 3)),
        false,
        ((1, 3, 2, 4), ()),
        Val(false),
    )
    C.data .= ComplexF64[5 + 1im, -2 + 4im]
    TensorKit.TO.tensorcontract!(
        C,
        A,
        ((1, 2), (3,)),
        false,
        B,
        ((1,), (2, 3)),
        false,
        ((1, 3, 2, 4), ()),
        2 - 0.25im,
        -1 + 0.5im,
    )
    expected = ComplexF64[
        -29.12579386826373 - 0.7876587736527441im,
        21.57892465101803 + 3.5376587736527494im,
    ]
    assert_close("Product(FermionParity,U1,SU2) contract", C.data, expected)
    println("FermionParityxU1xSU2,component_contract,$(collect(C.data))")
end

print_su2_noncanonical_both_adjoint_oracle()
print_product_fz2_u1_su2_contract_oracle()
