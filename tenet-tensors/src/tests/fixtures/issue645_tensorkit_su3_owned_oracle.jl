using TensorKit
using SUNRepresentations

fmt(values) = join(values, ",")

println("TensorKit_version=", pkgversion(TensorKit))
println("SUNRepresentations_version=", pkgversion(SUNRepresentations))

eight = SUNIrrep{3}((1, 1))
source = zeros(Float64, Vect[SU3Irrep](eight => 2) ⊗ Vect[SU3Irrep](eight => 3) ← Vect[SU3Irrep](eight => 5))
source.data .= 0:(length(source.data) - 1)

for (name, destination) in (
    ("r_only", ((2, 1), (3,))),
    ("inner_f_r", ((1, 3), (2,))),
)
    output = permute(source, destination)
    println(name, ".flat=", fmt(output.data))
end
