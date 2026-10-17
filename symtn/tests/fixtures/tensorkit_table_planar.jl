# Offline generator for tensorkit-0.17.1-table-planar.txt. Nothing in the
# shipped crates runs this; it is a manual step.
#
# Reproduce in a scratch project (NOT this directory):
#
#   julia --project=/tmp/fixture -e 'using Pkg; Pkg.add(name="TensorKit", version="0.17.1"); Pkg.add(name="TensorKitSectors", version="0.3.9")'
#   julia --project=/tmp/fixture path/to/tensorkit_table_planar.jl > path/to/tensorkit-0.17.1-table-planar.txt
#
# It runs, with TensorKit's own operations on `IsingAnyon`, the planar workflow
# of `table_planar_workflow.rs` on `σ ⊗ σ ⊗ σ ← σ`, in TensorKit's basis
# `[(σσ) → I, (σσ) → ψ]` of that space: the Artin generators σ₁, σ₂ (source
# levels increasing; reversed for the inverse) and the matrix read off the
# crossing-free `transpose` then `compose` with the dual basis. Values are
# printed with Julia's shortest round-trip `repr`.
#
# The `semion` rows run the same transpose and braids on the one state of
# `s ⊗ s ⊗ s ← s` for `Z2Element{1}` (`src/groupelements.jl`: `cocycle`,
# `Rsymbol`), the semion with `F^{sss}_s = −1` and `R^{ss}_𝟙 = i`: the
# AnyonWiki entry `[2,1,0,1,1,1,1]` has exactly these symbols.
#
# The `rep_s3_f` rows are the 3x3 transpose-then-compose matrix on
# `x ⊗ x ⊗ x ← x` for the two-dimensional irrep `x` of `D3Irrep` (Rep(S₃),
# `src/irreps/dnirrep.jl`; no braiding is used), in the channel order
# `(σσ) → A₁, A₂, E` that is also AnyonWiki's object order.
using Pkg
using TensorKit

version(uuid) = Pkg.dependencies()[Base.UUID(uuid)].version
println("# TensorKit v", version("07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"),
        ", TensorKitSectors v", version("13a9c161-d5da-41f0-bcbd-e1a08ae0647f"))
println("# Ising basis [(σσ)->I, (σσ)->ψ]; rows: tag row column re im (zero-based); semion rows: tag re im")

const σ = IsingAnyon(:σ)
const channels = (IsingAnyon(:I), IsingAnyon(:ψ))
const V = Vect[IsingAnyon](σ => 1)

function basis_state(space, c)
    t = zeros(ComplexF64, space)
    for (f₁, f₂) in fusiontrees(t)
        t[f₁, f₂] .= f₁.innerlines[1] == c ? 1 : 0
    end
    return t
end

function coordinates(t)
    values = zeros(ComplexF64, 2)
    for (f₁, f₂) in fusiontrees(t)
        values[findfirst(==(f₁.innerlines[1]), channels)] = t[f₁, f₂][1, 1, 1, 1]
    end
    return values
end

function sigma(t, generator, inverse)
    codomain = generator == 1 ? (2, 1, 3) : (1, 3, 2)
    levels = inverse ? (4, 3, 2, 1) : (1, 2, 3, 4)
    return braid(t, (codomain, (4,)), levels)
end

basis = [basis_state(V ⊗ V ⊗ V ← V, c) for c in channels]
dual_basis = [basis_state(V' ⊗ V' ⊗ V' ← V', c) for c in channels]

emit(tag, m) = for r in 1:2, c in 1:2
    println(tag, " ", r - 1, " ", c - 1, " ", repr(real(m[r, c])), " ", repr(imag(m[r, c])))
end

for generator in (1, 2), inverse in (false, true)
    m = hcat((coordinates(sigma(basis[c], generator, inverse)) for c in 1:2)...)
    emit((inverse ? "sigma_inv" : "sigma") * string(generator), m)
end

f = [only(block(transpose(basis[c], ((4,), (3, 2, 1))) * dual_basis[r], σ)) for r in 1:2, c in 1:2]
emit("f", f)

const s = Z2Element{1}(1)
const S = Vect[Z2Element{1}](s => 1)
semion = ones(ComplexF64, S ⊗ S ⊗ S ← S)
semion_dual = ones(ComplexF64, S' ⊗ S' ⊗ S' ← S')
only_value(t) = only(only(blocks(t))[2])
semion_value(tag, z) = println(tag, " ", repr(real(z)), " ", repr(imag(z)))
semion_value("semion_f", only(block(transpose(semion, ((4,), (3, 2, 1))) * semion_dual, s)))
for generator in (1, 2), inverse in (false, true)
    semion_value((inverse ? "semion_sigma_inv" : "semion_sigma") * string(generator),
                 only_value(sigma(semion, generator, inverse)))
end

const D3 = TensorKit.TensorKitSectors.DNIrrep{3}
const x = D3(1)
const X = Vect[D3](x => 1)
const rep_channels = (D3(0, false), D3(0, true), x)
function rep_basis(space, c)
    t = zeros(ComplexF64, space)
    for (f₁, f₂) in fusiontrees(t)
        t[f₁, f₂] .= f₁.innerlines[1] == c ? 1 : 0
    end
    return t
end
for r in 1:3, c in 1:3
    t = transpose(rep_basis(X ⊗ X ⊗ X ← X, rep_channels[c]), ((4,), (3, 2, 1)))
    z = only(block(t * rep_basis(X' ⊗ X' ⊗ X' ← X', rep_channels[r]), x))
    println("rep_s3_f ", r - 1, " ", c - 1, " ", repr(real(z)), " ", repr(imag(z)))
end
