# Explicit payload precision conversions — H2

Authority: TeNeT `fb5f9a91` (`origin/main`), issue
[#1323](https://github.com/Ryo-wtnb11/TeNeT/issues/1323), plan
[#1065](https://github.com/Ryo-wtnb11/TeNeT/issues/1065), base admission
`issue-1315-single-precision-base.md` (revision-pinned, not rewritten here).

This artifact is revision-pinned evidence. Current authority is
`tenet/src/typed.rs` (`TensorMap::{to_f64, to_c32, to_c64, narrow_to_f32,
narrow_to_c32}`) and `tenet/tests/precision_conversions.rs`.

References:

* TensorKit.jl `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91`, identical in these
  files to the registered 0.17.1 (`~/.julia/packages/TensorKit/DQFb5`).
* QSpace v4: **no corresponding path**; its payload is `double` /
  `wbcomplex<double>` only.

## API

| Source | Method | Result | Exact |
| --- | --- | --- | --- |
| `f32` | `to_f64` | `f64` | yes |
| `f32` | `to_c32` | `Complex32`, imaginary `+0` | yes |
| `f32` | `to_c64` | `Complex64`, imaginary `+0` | yes |
| `f64` | `to_c64` | `Complex64`, imaginary `+0` | yes (existed; now keeps a lazy adjoint lazy) |
| `Complex32` | `to_c64` | `Complex64` | yes |
| `f64` | `narrow_to_f32` | `f32` | no: `as f32` |
| `Complex64` | `narrow_to_c32` | `Complex32` | no: `as f32` per component |

Not provided: `f64 -> Complex32` (lossy; compose `narrow_to_f32().to_c32()`),
complex -> real (`re`/`im` already exist and are projections, not precision
changes), `From`/`Into` between payload dtypes, and any implicit conversion
inside an operation (mixed dtypes stay a compile error; `compile_fail`
doctests on `to_f64`).

## TensorKit correspondence

| TensorKit | TeNeT |
| --- | --- |
| `abstracttensor.jl:Base.complex(t)` = `copy!(similar(t, complex(T)), t)` | `to_c32` / `to_c64` |
| `abstracttensor.jl:similar(t, ::Type{T})` + `copy!` | the `to_*` / `narrow_to_*` family |
| `tensor.jl:Base.convert(::Type{TensorMap{T,S,N₁,N₂,A}}, t)` (`TT(undef, space(t))` + `copy!`) | same, named by direction |
| `diagonal.jl:Base.convert(D::Type{<:DiagonalTensorMap}, d)` / `similar_diagonal(d, T)` — stays diagonal | compact diagonal stays compact |
| `tensor.jl:Base.promote_rule` — mixed scalar types promote implicitly | no counterpart: mixing is a compile error |
| `Base.complex(t::AdjointTensorMap)` via `similar` — materializes | lazy adjoint stays lazy over a converted parent |

Why TeNeT differs in name: TensorKit's `convert`/`promote_rule` make a
`Float64 -> Float32` conversion silent (Julia `convert` rounds). TeNeT forbids
implicit lossy narrowing, so the direction is in the name: `to_*` is exact and
`narrow_to_*` is the only lossy entry point. `complex(t)` is not reused as a
name because TeNeT needs the target precision (`to_c32` vs `to_c64`) explicit.

## Storage forms

* **Dense**: one `Vec` of the target dtype, `collect` over an exact-size
  iterator: one pass, one payload-sized allocation.
* **Compact diagonal**: stays compact (`map_spectrum_dtype`); allocates the
  per-sector spectra its representation holds, `O(Σ_c k_c)` values.
* **Lazy adjoint of a dense parent**, decided by the provider mode's
  `TypedTensorModeDispatch::LAZY_ADJOINT_OPERANDS`:
  * multiplicity free (every operation, factorizations included, reads lazy
    adjoints): the parent is converted and the view rebuilt over it (cold
    `materialized` cache) — one allocation, no permutation. The real -> complex
    embedding does not commute with conjugation at the sign of zero
    (`conj(x + 0i) = x - 0i`), so a real *parent* is embedded as `x - 0i`,
    which makes the logical entries exactly `x + 0i`. Rounding and widening
    commute with conjugation, so the parent uses the same conversion;
  * checked Generic (its factorizations reject lazy adjoints): materialized in
    the source dtype, then converted, as `f64::to_c64` did before this leaf —
    an operation-local source-sized buffer plus the output. Keeping it lazy
    would make `s.adjoint()?.to_c64().qr_compact()` fail (review of #1446).
* **Lazy adjoint of a compact diagonal** (checked Generic only; the
  multiplicity-free `adjoint` already returns an owned diagonal): an owned
  compact diagonal of `convert(conj(value))` on the logical space, which is
  what the multiplicity-free `adjoint` emits — a bond space is its own
  adjoint. Its unstored zeros are `+0`. (The complex checked-Generic lazy view
  of a diagonal itself materializes those zeros as `0 - 0i`; that is
  pre-existing and only the sign of an unstored zero differs.)
* **Device** (`CudaStorage`): not provided. The methods exist only on the host
  `Vec<D>` storage, so a device tensor is a compile error (`compile_fail`
  doctest), not a runtime fallback. A device path would be download, convert,
  upload; a fused device kernel is a follow-up only if a workload needs it.

Structure, spaces and gauge are copied (`BoundDynamicFusionMapSpace` clone:
`Arc` handles), and the runtime handle is shared.

## Narrowing semantics

`value as f32`: IEEE 754 round to nearest, ties to even; a value whose
rounding exceeds `f32::MAX` becomes `±inf` (`f32::MAX + 2^102` rounds back to
`f32::MAX`, `f32::MAX + 2^103` is a tie with an odd significand and becomes
`inf`); subnormals are produced, not flushed; `±0`/`±inf` keep their sign;
NaN stays NaN. Rust does not promise NaN payload bits from `as`, so the tests
compare NaN bits against a runtime `as` oracle and otherwise assert NaN-ness.

## Evidence

`tenet/tests/precision_conversions.rs`, bitwise throughout (data movement per
`docs/testing_numerics.md`):

* every conversion against `f64::from` / `as f32` per element, and both round
  trips (`narrow_to_f32 ∘ to_f64`, `narrow_to_c32 ∘ to_c64`, real and genuinely
  complex payloads) returning the original bits — on a fixture holding `±0`,
  the smallest and largest subnormal, `f32::MAX`, `±inf` and a payload NaN;
* rounding, overflow and underflow edges against hand values independent of
  `as`;
* codomain, domain, block count and per-block fusion trees preserved;
* U(1), fZ2 × U(1) × SU(2) and, under `racah-generated`, Checked-Generic SU(3)
  with an outer-multiplicity vertex; dense, compact diagonal (stays compact for
  every conversion) and lazy adjoint (real and complex parents), each lazy
  conversion compared bitwise with the element-wise conversion of the
  materialized source view, including ±inf and NaN parents;
* checked Generic: the converted dense and diagonal adjoints are owned and
  `qr_compact` succeeds on them; the provider `Arc` is shared (`ptr::eq`);
* allocations: between two payload sizes of one structure the call count is
  equal, the byte difference is exactly the payload difference, and exactly
  one allocation is payload-sized — for every conversion and for a lazy
  adjoint, whose converted view reads its parent back with zero allocations.
