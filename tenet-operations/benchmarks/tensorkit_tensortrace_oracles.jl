using TensorKit

function assert_close(label, actual, expected; atol = 1.0e-12)
    length(actual) == length(expected) || error("$label length mismatch")
    for (index, (a, e)) in enumerate(zip(actual, expected))
        abs(a - e) <= atol || error("$label[$index] mismatch: $a != $e")
    end
end

function print_fermion_open_output_trace_oracle()
    V = Vect[FermionParity](0 => 1, 1 => 1)
    A = zeros(Float64, (V ⊗ V) ← (V ⊗ V))
    A.data .= collect(1.0:length(A.data))
    C = zeros(Float64, V ← V)
    C.data .= [10.0, 20.0]
    TensorKit.TO.tensortrace!(
        C,
        A,
        ((1,), (3,)),
        ((2,), (4,)),
        false,
        2.0,
        3.0,
    )
    expected = [16.0, 62.0]
    assert_close("FermionParity open-output tensortrace", C.data, expected)
    println("FermionParity,open_output_trace,$(collect(C.data))")
end

function print_fermion_two_pair_trace_oracle()
    V = Vect[FermionParity](0 => 1, 1 => 1)
    A = zeros(Float64, (V ⊗ V) ← (V ⊗ V))
    A.data .= collect(1.0:length(A.data))
    C = zeros(Float64, one(V) ← one(V))
    C.data .= [5.0]
    TensorKit.TO.tensortrace!(
        C,
        A,
        ((), ()),
        ((1, 2), (3, 4)),
        false,
        2.0,
        3.0,
    )
    expected = [-1.0]
    assert_close("FermionParity two-pair tensortrace", C.data, expected)
    println("FermionParity,two_pair_trace,$(collect(C.data))")
end

print_fermion_open_output_trace_oracle()
print_fermion_two_pair_trace_oracle()
