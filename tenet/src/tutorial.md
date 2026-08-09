# Tutorial

Everyday TeNeT code uses the **user layer**: one `use tenet::prelude::*;`
import gives [`prelude::Runtime`], [`prelude::GradedSpace`],
[`prelude::TensorMap`], built-in providers and [`prelude::Truncation`]. The
provider, scalar and storage are type parameters; rank and sector content remain
runtime values. The `tensor!` contraction frontend is provided by the
`tenet-network` crate. Expert layers ([`core`], [`operations`], [`dense`],
[`matrixalgebra`]) stay available underneath — see the appendix at the end.

Every code block in this tutorial runs as a doctest, so it is guaranteed to
compile and pass against the current API.

## 1. Quick Start

A [`prelude::Runtime`] is built once and then carried implicitly by every
tensor created from it (it owns the contraction/tree-transform caches and
the dense backend). A [`prelude::GradedSpace`] is a graded vector space for one
tensor leg: `(sector, degeneracy)` pairs plus a dual flag. A
[`prelude::TensorMap`] is a block-sparse symmetric tensor `codomain <- domain`
with dynamic rank.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;

// U(1): (charge, degeneracy) pairs. dim = 2 + 3 + 2.
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [
        (U1Irrep::new(-1), 2),
        (U1Irrep::new(0), 3),
        (U1Irrep::new(1), 2),
    ],
    false,
)?;
assert_eq!(v.dim()?, 7.0);
assert_eq!(v.try_dual()?.try_dual()?, v);

// SU(2): (twice_spin, degeneracy) pairs; dim is quantum-dimension
// weighted: 2 * 1 (spin 0) + 2 * 2 (spin 1/2).
let s = GradedSpace::try_new_owned(
    SU2FusionRule,
    [
        (SU2Irrep::from_twice_spin(0), 2),
        (SU2Irrep::from_twice_spin(1), 2),
    ],
    false,
)?;
assert_eq!(s.dim()?, 6.0);

// Tensors on codomain <- domain leg lists.
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;
assert_eq!((a.codomain_rank(), a.domain_rank(), a.rank()), (2, 2, 4));
let z = TensorMap::<U1FusionRule, f64>::zeros(&rt, [&v], [&v])?;
assert_eq!(z.norm()?, 0.0);
# Ok::<(), Error>(())
```

[`prelude::TensorMap::from_block_fn`] fills every symmetry-allowed block
element from a closure over the block key and block-local degeneracy
indices:

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new_owned(
    Z2FusionRule,
    [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    false,
)?;

// A diagonal Z2 matrix: 2 on the even block, 3 on the odd block.
let a: TensorMap<Z2FusionRule, f64> =
    TensorMap::from_block_fn(&rt, [&v], [&v], |trees, _| {
        if *trees.coupled() == Z2Irrep::EVEN { 2.0 } else { 3.0 }
    })?;
let b: TensorMap<Z2FusionRule, f64> =
    TensorMap::from_block_fn(&rt, [&v], [&v], |trees, _| {
        if *trees.coupled() == Z2Irrep::EVEN { 5.0 } else { 7.0 }
    })?;
assert_eq!(a.compose(&b)?.data(), &[10.0, 21.0]);
# Ok::<(), Error>(())
```

### Scalar dtype

The scalar is the second type parameter: `TensorMap<R, f64>` or
`TensorMap<R, Complex64>`.
[`prelude::TensorMap::from_block_fn`] can infer it from the closure. Mixed
scalar operations are rejected at compile time; widen explicitly with
[`prelude::TensorMap::to_c64`]. Scalar-returning methods return `D` directly.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [
        (U1Irrep::new(-1), 1),
        (U1Irrep::new(0), 2),
        (U1Irrep::new(1), 1),
    ],
    false,
)?;

let re = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
let cx: TensorMap<U1FusionRule, Complex64> = TensorMap::from_block_fn(&rt, [&v], [&v], |_, indices| {
    Complex64::new(indices[0] as f64, -(indices[1] as f64))
})?;

let inner = re.inner(&re)?;
assert!((inner - re.norm()?.powi(2)).abs() <= 1e-10 * (1.0 + inner));

let cc = cx.inner(&cx)?;
assert!(cc.im.abs() <= 1e-12 * (1.0 + cc.re));

assert!(re.to_c64().compose(&cx).is_ok());
# Ok::<(), Error>(())
```

### Which legs may contract?

Contraction compatibility uses oriented dual pairing:
the two selected legs must represent dual vector spaces in the current
codomain/domain orientation. A codomain leg built from `v` contracts a
domain leg built from the same `v`. To contract two same-side legs
(e.g. domain against domain), build exactly one of them from `v.dual()`.

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>obj</mi><mi>A</mi></msub><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo>
  <mo>≅</mo>
  <msup>
    <mrow><msub><mi>obj</mi><mi>B</mi></msub><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow>
    <mo>*</mo>
  </msup>
</math>
</div>

See [`mathematics`] for the full tensor-map convention, dual, same-side
contraction, and categorical `flip` conventions.

Provider, scalar, and storage mismatches are compile-time type errors. Separate
provider allocations of the same type are admitted only when their semantic
rule identities agree. Runtime and space mismatches return [`prelude::Error`].
`tensor!` label mistakes (dangling or repeated labels) are compile-time errors.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
// Works for any charge set, including ones that are not symmetric under
// negation (a hardcore boson).
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    false,
)?;

// Codomain-vs-domain legs of the same GradedSpace contract directly...
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
let _ = a.compose(&a)?;

// ...domain-vs-domain legs need one side built from the dual space.
let dual = v.try_dual()?;
let b = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&dual])?;
let _ = a.contract(&b, &[1], &[1], &[0, 1])?;

// So is mixing runtimes.
let rt2 = Runtime::builder().build()?;
let c = TensorMap::<U1FusionRule, f64>::rand(&rt2, [&v], [&v])?;
assert!(matches!(a.compose(&c), Err(Error::RuntimeMismatch)));
# Ok::<(), Error>(())
```

## 2. Contraction

### `tensor!` — the way to contract

The `tensor!` macro (crate `tenet-network`) is @tensor-style index
notation over homogeneous [`prelude::TensorMap`] operands. The output signature
comes first: `[codomain; domain]`; the `;`
is optional (`[a, b]` = all-codomain output) and `[]` is a rank-0 (scalar)
output, read out with [`prelude::TensorMap::scalar`]. `conj(x)` marks an
adjoint operand. A label appearing on two operands is contracted; a label
appearing once must be listed in the output — violations are compile
errors. With three or more operands the pairwise order is chosen
automatically by a greedy planner. There are no einsum strings anywhere.

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd>
        <msub><mi>C</mi><mrow><mi>i</mi><mi>j</mi><mo>;</mo><mi>m</mi><mi>n</mi></mrow></msub>
        <mo>=</mo>
        <munder><mo>∑</mo><mrow><mi>k</mi><mo>,</mo><mi>l</mi></mrow></munder>
        <msub><mi>A</mi><mrow><mi>i</mi><mi>j</mi><mo>;</mo><mi>k</mi><mi>l</mi></mrow></msub>
        <mspace width="0.35em"/>
        <msub><mi>B</mi><mrow><mi>k</mi><mi>l</mi><mo>;</mo><mi>m</mi><mi>n</mi></mrow></msub>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd>
        <mi>E</mi>
        <mo>=</mo>
        <munder><mo>∑</mo><mrow><mi>p</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>l</mi><mo>,</mo><mi>r</mi></mrow></munder>
        <mover>
          <msub><mi>ψ</mi><mrow><mi>p</mi><mo>;</mo><mi>l</mi><mi>r</mi></mrow></msub>
          <mo>¯</mo>
        </mover>
        <mspace width="0.35em"/>
        <msub><mi>H</mi><mrow><mi>p</mi><mo>;</mo><mi>q</mi></mrow></msub>
        <mspace width="0.35em"/>
        <msub><mi>ψ</mi><mrow><mi>q</mi><mo>;</mo><mi>l</mi><mi>r</mi></mrow></msub>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

```rust
use tenet::prelude::*;
use tenet_network::tensor;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;
let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 1)?;
let b = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 2)?;

// Pairwise contraction with an explicit output signature.
let c = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n])?;
assert_eq!((c.codomain_rank(), c.domain_rank()), (2, 2));

// conj() + rank-0 output computes the weighted self inner product.
let n2 = tensor!([] = conj(a)[i, j; k, l] * a[i, j; k, l])?.scalar()?;
let norm = a.norm()?;
assert!((n2 - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

// A single operand is a permute.
let p = tensor!([j, i; m, n] = c[i, j; m, n])?;
assert_eq!(p.rank(), 4);

// N-ary: an energy contraction; greedy planning picks the order.
let psi = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v], [&v, &v], 3)?;
let h = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v], [&v], 4)?;
let e = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])?.scalar()?;
assert!(e.is_finite());
# Ok::<(), Error>(())
```

Label errors do not survive to runtime — this does not compile because `k`
and `j` each appear once without being output labels:

```rust,compile_fail
use tenet::prelude::{Error, TensorMap, U1FusionRule};
use tenet_network::tensor;

fn wrong(
    a: &TensorMap<U1FusionRule, f64>,
    b: &TensorMap<U1FusionRule, f64>,
) -> Result<TensorMap<U1FusionRule, f64>, Error> {
    tensor!([i; m] = a[i; k] * b[j; m])
}
```

A written `;` split that contradicts the tensor's actual codomain rank is
checked at plan time (runtime `InvalidArgument`), since the macro cannot
see the tensor's shape.

### The method API underneath

`tensor!` lowers to pairwise steps over the typed explicit method API, which is
available directly when you want to spell the axes:

- [`prelude::TensorMap::compose`] — categorical composition (TensorKit
  `A * B` / `mul!`), also spelled `&a * &b`. **No** fermionic supertrace
  twist on dual composed legs.
- [`prelude::TensorMap::contract`] — contract arbitrary axis pairs with an
  explicit output order (TensorKit `tensorcontract!` and its `pAB`). Like
  `tensor!`, this **twists** dual
  contracted legs on fermionic rules — bosonic results are identical to
  `compose`, fermionic ones can differ by signs.
- [`prelude::TensorMap::contract_ordered`] — documented alias of `contract`.
- [`prelude::TensorMap::permute`] / [`prelude::TensorMap::braid`] /
  [`prelude::TensorMap::transpose`] — TensorKit's leg re-arrangements
  (symmetric braiding / explicit braid levels / planar transpose).
- [`prelude::TensorMap::adjoint`] — dagger: swaps codomain and domain.

Axes are zero-based and flat: codomain axes first, then domain axes.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;
let b = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;

let c1 = a.compose(&b)?;
let c2 = a.contract(&b, &[2, 3], &[0, 1], &[0, 1, 2, 3])?;
assert_eq!(c1.data(), c2.data());
let _c3 = a.contract(&b, &[2, 3], &[0, 1], &[1, 0, 2, 3])?;

let p = c1.permute(&[0, 2], &[1, 3])?;
assert!((p.norm()? - c1.norm()?).abs() <= 1e-10 * (1.0 + c1.norm()?));
let t = c1.transpose()?;
assert_eq!(t.transpose()?.data().len(), c1.data().len());
let h = c1.adjoint()?;
assert_eq!((h.codomain_rank(), h.domain_rank()), (2, 2));
# Ok::<(), Error>(())
```

## 3. Tensor algebra: vector interface, index ops, and sectors

### Vector interface

The VectorInterface / LinearAlgebra surface mirrors TensorKit:
[`prelude::TensorMap::norm`], [`prelude::TensorMap::normalize`],
[`prelude::TensorMap::inner`] / [`prelude::TensorMap::dot`],
[`prelude::TensorMap::scale`], [`prelude::TensorMap::add`] (the `α·self + β·other`
combination, covering TensorKit's `axpy!`/`axpby!`),
[`prelude::TensorMap::tr`], and [`prelude::TensorMap::zeros_like`] (TensorKit
`zerovector`). Structural predicates match TensorKit's
`ishermitian`/`isantihermitian`/`isisometric`/`isunitary`/`isposdef`, with the
`(t ± t†)/2` projectors [`prelude::TensorMap::project_hermitian`] /
[`prelude::TensorMap::project_antihermitian`].

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
let b = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;

// α·a + β·b (TensorKit axpby), scaling, and unit normalization.
let _diff = a.add(&b, 1.0, -1.0)?;    // a - b
let _scaled = a.scale(2.0);
let unit = a.normalize()?;
assert!((unit.norm()? - 1.0).abs() <= 1e-12);

// inner / dot agree, and norm² == <a, a>.
let ip = a.inner(&a)?;
assert!((ip - a.norm()?.powi(2)).abs() <= 1e-10 * (1.0 + ip));

// A same-shape zero (zerovector) and the trace of an endomorphism.
let zero = a.zeros_like();
assert_eq!(zero.norm()?, 0.0);
let _trace = a.tr()?;

// Structural predicates: the identity is Hermitian, unitary, positive definite.
let id = TensorMap::<U1FusionRule, f64>::id(&rt, [&v])?;
assert!(id.is_hermitian(1e-12)? && id.is_unitary(1e-12)? && id.is_posdef(1e-12)?);
# Ok::<(), Error>(())
```

### Index operations

Leg rearrangements follow TensorKit's names. Axis lists are flat and
zero-based (codomain axes first). [`prelude::TensorMap::permute`] chooses new
codomain/domain axis lists; [`prelude::TensorMap::repartition`] re-splits the
legs at a codomain count while keeping their order (TensorKit `repartition`);
[`prelude::TensorMap::transpose`] is the planar transpose,
[`prelude::TensorMap::adjoint`] the dagger, and
[`prelude::TensorMap::twist`] / [`prelude::TensorMap::flip`] act on chosen legs.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;

// permute: new codomain axes | new domain axes.
let p = a.permute(&[0, 2], &[1, 3])?;
assert_eq!((p.codomain_rank(), p.domain_rank()), (2, 2));

// repartition: move the codomain/domain split, order preserved; invertible.
let r = a.repartition(1)?;
assert_eq!((r.codomain_rank(), r.domain_rank()), (1, 3));
assert_eq!(r.repartition(2)?.data(), a.data());

// transpose (planar), adjoint (dagger), twist and flip on chosen legs.
let _t = a.transpose()?;
let h = a.adjoint()?;
assert_eq!((h.codomain_rank(), h.domain_rank()), (2, 2));
let _twisted = a.twist(&[0])?;
let _flipped = a.flip(&[0])?;
# Ok::<(), Error>(())
```

### Sectors and space algebra

A [`prelude::GradedSpace`] carries provider-labelled `(sector, degeneracy)`
content queried through [`prelude::GradedSpace::sectors`],
[`prelude::GradedSpace::degeneracy`] (TensorKit `dim(V, c)`),
[`prelude::GradedSpace::has_sector`] (TensorKit `hassector`),
[`prelude::GradedSpace::fuse`] (`⊗`), and [`prelude::GradedSpace::oplus`] (`⊕`).

```rust
use tenet::prelude::*;

let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 2), (0, 3), (1, 2)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;

// Enumerate sectors and query membership / degeneracy.
assert_eq!(v.sectors()?.len(), 3);
assert_eq!(v.degeneracy(&U1Irrep::new(0))?, 3);
assert!(v.has_sector(&U1Irrep::new(1))?);
assert!(!v.has_sector(&U1Irrep::new(9))?);

// fuse (⊗) collapses two legs; oplus (⊕) sums per-sector degeneracies.
let w = GradedSpace::try_new_owned(
    U1FusionRule,
    [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    false,
)?;
assert_eq!(v.fuse(&w)?.dim()?, v.dim()? * w.dim()?);
assert_eq!(v.oplus(&w)?.degeneracy(&U1Irrep::new(0))?, 3 + 1);

// SU(2) dims are quantum-dimension weighted.
let s = GradedSpace::try_new_owned(
    SU2FusionRule,
    [
        (SU2Irrep::from_twice_spin(0), 1),
        (SU2Irrep::from_twice_spin(1), 1),
    ],
    false,
)?;
assert_eq!(s.dim()?, 3.0);
# Ok::<(), Error>(())
```

### Combining symmetries

A symmetry is a value, so combining two is a method call rather than a new type
in the library. `left.product(right)` builds a provider for `left ⊠ right`; its
labels are [`prelude::ProductSector`], built with [`prelude::product_sector`],
and everything else — spaces, tensors, contraction, decompositions — is
unchanged.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;

// fZ2 ⊠ U(1): fermion parity together with particle number.
let rule = FermionParityFusionRule.product(U1FusionRule);
let v = GradedSpace::try_new_owned(
    rule,
    [
        (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
        (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
    ],
    false,
)?;

let t = TensorMap::<_, f64>::zeros(&rt, [&v], [&v])?;
assert_eq!(t.block_count(), 2);

// Labels come back as the pair, with each component in its own type.
let sectors = v.sectors()?;
assert_eq!(sectors[0].left(), &Z2Irrep::EVEN);
assert_eq!(sectors[0].right(), &U1Irrep::new(0));
# Ok::<(), Error>(())
```

Products nest, so `a.product(b).product(c)` is a three-component provider. Two
consequences worth knowing before you pick an order:

- **Order and association are part of the type.** `U(1) ⊠ fZ2` and
  `fZ2 ⊠ U(1)` are both legal and are different providers; a tensor built with
  one does not compose with a tensor built with the other. Fix the order once,
  at the top of your model.
- **The coefficient scalar is promoted, not fixed.** A product of two
  real-coefficient providers stays real; a component with complex F/R data,
  such as `FibonacciFusionRule`, widens the product to `Complex64`. Tensors
  over such a product need a complex payload.

To use a symmetry that is not built in, implement the provider traits in your
own crate — nothing in the engine enumerates symmetries, so an external
provider reaches the same `GradedSpace` / `TensorMap` API. The obligations, the
laws the engine assumes without checking, and the current restrictions are in
`docs/provider_interface.md`.

## 4. Decompositions

TeNeT applies decompositions independently per coupled sector across the
codomain | domain split. Method names use the established
TensorKit/MatrixAlgebraKit vocabulary where the operation agrees:

- [`prelude::TensorMap::svd_trunc`] — truncated SVD; see below.
- [`prelude::TensorMap::svd_compact`] / [`prelude::TensorMap::svd_full`] /
  [`prelude::TensorMap::svd_vals`].

- [`prelude::TensorMap::qr_compact`] / [`prelude::TensorMap::qr_full`],
  [`prelude::TensorMap::lq_compact`] / [`prelude::TensorMap::lq_full`].
- [`prelude::TensorMap::left_orth`] / [`prelude::TensorMap::right_orth`] —
  TensorKit's default kinds (QR / LQ), including the positive-diagonal
  gauge (`positive = true`, MatrixAlgebraKit's default).
- [`prelude::TensorMap::left_null`] / [`prelude::TensorMap::right_null`],
  [`prelude::TensorMap::left_polar`] / [`prelude::TensorMap::right_polar`].
- [`prelude::TensorMap::eigh_full`] / [`prelude::TensorMap::eigh_trunc`] /
  [`prelude::TensorMap::eigh_vals`] — Hermitian eigendecomposition.
- [`prelude::TensorMap::eig_full`] / [`prelude::TensorMap::eig_trunc`] /
  [`prelude::TensorMap::eig_vals`] — general eigendecomposition; outputs are
  `c64` even for real input.
- [`prelude::TensorMap::exp`] / [`prelude::TensorMap::inv`] /
  [`prelude::TensorMap::pinv`] — matrix functions of endomorphisms.

Hermitian `eigh_*` keeps the input dtype and reports real eigenvalues.
General `eig_*` is complex-valued by construction, so the returned
diagonal/eigenvector tensors are always `c64`.

Truncation is controlled by [`prelude::Truncation`]: `Full`,
`Rank(n)` (`Truncation::rank(n)`), checked tolerance constructors
(`absolute_cutoff` / `relative_cutoff` / `relative_inf_cutoff`),
`relative_error`, and `and` (intersection of rules). All bounds and reported
errors are
**quantum-dimension weighted**: `Rank(n)` bounds the weighted kept bond
dimension, and the `error` field of [`typed::SvdTrunc`] /
[`typed::EighTrunc`] is the weighted 2-norm of everything discarded, so
the reconstruction distance equals the reported error in the weighted
Frobenius norm.

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.45em">
    <mtr>
      <mtd>
        <mi>T</mi>
        <mo>≈</mo>
        <mi>U</mi><mspace width="0.2em"/><mi>S</mi><mspace width="0.2em"/><msup><mi>V</mi><mi>†</mi></msup>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msub>
          <mrow><mo>∥</mo><mi>T</mi><mo>−</mo><mi>U</mi><mspace width="0.2em"/><mi>S</mi><mspace width="0.2em"/><msup><mi>V</mi><mi>†</mi></msup><mo>∥</mo></mrow>
          <mrow><mi>F</mi><mo>,</mo><mi>w</mi></mrow>
        </msub>
        <mo>=</mo><mi>ε</mi>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

A worked mini-example — split a rank-4 tensor across the current
codomain/domain boundary, truncate the bond, and check the reported error
against the actual reconstruction distance:

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;
let t = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;

// Truncated SVD across the codomain | domain split.
let svd = t.svd_trunc(&Truncation::rank(6))?;
assert_eq!((svd.u.codomain_rank(), svd.u.domain_rank()), (2, 1));
assert_eq!((svd.vh.codomain_rank(), svd.vh.domain_rank()), (1, 2));

// The kept spectra are reported per coupled sector...
assert!(!svd.singular_values.is_empty());

// ...and the reported error is exactly the reconstruction distance.
let recon = svd.u.compose(&svd.s)?.compose(&svd.vh)?;
let diff = recon.add(&t, 1.0, -1.0)?.norm()?;
assert!((diff - svd.error).abs() <= 1e-8 * (1.0 + svd.error));

// Orthogonality: q from QR is an isometry, so q r reconstructs t.
let (q, r) = t.qr_compact()?;
let qr = q.compose(&r)?;
let diff = qr.add(&t, 1.0, -1.0)?.norm()?;
assert!(diff <= 1e-10 * (1.0 + t.norm()?));

// General eigendecomposition is complex-valued even for real input.
let (d, w) = t.eig_full()?;
let _: &TensorMap<U1FusionRule, Complex64> = &d;
let _: &TensorMap<U1FusionRule, Complex64> = &w;

// Hermitian eigendecomposition keeps the real scalar type.
let h = t.add(&t.adjoint()?, 0.5, 0.5)?;
let (evals, vecs) = h.eigh_full()?;
let _: &TensorMap<U1FusionRule, f64> = &evals;
let _: &TensorMap<U1FusionRule, f64> = &vecs;
# Ok::<(), Error>(())
```

To split a tensor along a different bipartition than its current
codomain | domain split, `permute` first —
that is exactly what the next section does.

For the QR path, the compact factor obeys the usual isometry relation:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup><mi>Q</mi><mi>†</mi></msup><mi>Q</mi><mo>=</mo><mi>I</mi>
  <mo>,</mo>
  <mi>T</mi><mo>=</mo><mi>Q</mi><mi>R</mi>
</math>
</div>

## 5. Worked Example: a U(1) Two-Site Imaginary-Time Step

The simple-update kernel: apply the two-site imaginary-time gate to a
two-site wavefunction, regroup the legs around the bond, and truncate the
bond back with `svd_trunc`.

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>G</mi><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi mathvariant="normal">exp</mi><mo stretchy="false">(</mo><mo>−</mo><mi>τ</mi><mspace width="0.2em"/><mi>H</mi><mo stretchy="false">)</mo>
</math>
</div>

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd>
        <msub><mi>θ</mi><mrow><mi>a</mi><mi>l</mi><mo>;</mo><mi>b</mi><mi>r</mi></mrow></msub>
        <mo>=</mo>
        <munder><mo>∑</mo><mrow><mi>p</mi><mo>,</mo><mi>q</mi></mrow></munder>
        <msub><mi>G</mi><mrow><mi>a</mi><mi>b</mi><mo>;</mo><mi>p</mi><mi>q</mi></mrow></msub>
        <mspace width="0.35em"/>
        <msub><mi>ψ</mi><mrow><mi>p</mi><mi>q</mi><mo>;</mo><mi>l</mi><mi>r</mi></mrow></msub>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd>
        <msub><mi>θ</mi><mrow><mi>a</mi><mi>l</mi><mo>;</mo><mi>b</mi><mi>r</mi></mrow></msub>
        <mo>≈</mo>
        <munder><mo>∑</mo><mi>α</mi></munder>
        <msub><mi>U</mi><mrow><mi>a</mi><mi>l</mi><mo>;</mo><mi>α</mi></mrow></msub>
        <mspace width="0.35em"/>
        <msub><mi>S</mi><mrow><mi>α</mi><mo>;</mo><mi>α</mi></mrow></msub>
        <mspace width="0.35em"/>
        <msub><msup><mi>V</mi><mi>†</mi></msup><mrow><mi>α</mi><mo>;</mo><mi>b</mi><mi>r</mi></mrow></msub>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

```rust
use tenet::prelude::*;
use tenet_network::tensor;

let rt = Runtime::builder().build()?;

// Physical leg: spin-1/2 with U(1) Sz charges +-1. Virtual bond legs.
let p = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 1), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    false,
)?;

// Two-site wavefunction with two physical and two virtual legs.
let psi = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&p, &p], [&v, &v], 10)?;

// Hermitian two-site Hamiltonian and the imaginary-time gate.
let h0 = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&p, &p], [&p, &p], 11)?;
let h = h0.add(&h0.adjoint()?, 0.5, 0.5)?;
let tau = 0.05;
let gate = h.scale(-tau).exp()?;

// Apply the gate and regroup (site 1 + left bond | site 2 + right bond).
let theta = tensor!([a, l; b, r] = gate[a, b; p, q] * psi[p, q; l, r])?;

// Truncate the bond back: new site tensors plus the bond weights.
let svd = theta.svd_trunc(&Truncation::rank(4))?;
let left = svd.u;    // [p, l] <- [bond]   new site 1
let right = svd.vh;  // [bond] <- [p, r]   new site 2
let weights = svd.s; // [bond] <- [bond]   kept for the inverse-weight trick
assert_eq!((left.codomain_rank(), left.domain_rank()), (2, 1));
assert_eq!((right.codomain_rank(), right.domain_rank()), (1, 2));

// The truncated factors recompose to the rank-limited theta, and the
// reported error is exactly the reconstruction distance.
let recon = left.compose(&weights)?.compose(&right)?;
let diff = recon.add(&theta, 1.0, -1.0)?.norm()?;
assert!((diff - svd.error).abs() <= 1e-8 * (1.0 + svd.error));
println!("truncation error: {:.3e}", svd.error);
# Ok::<(), Error>(())
```

In a real simple-update loop this step runs once per bond per sweep, with
the stored bond weights absorbed and re-extracted around each gate.

## 6. Under the Hood: the Expert Layers

The provider-typed user layer delegates storage/layout work to four expert
modules:

- [`core`] — structural data layer: sectors and fusion rules
  ([`core::SectorLeg`], `U1FusionRule`, ...), fusion-tree spaces
  ([`core::FusionProductSpace`], [`core::FusionTreeHomSpace`],
  [`core::FusionTensorMapSpace`]), block layout
  ([`core::BlockStructure`]), and the typed tensor
  ([`core::TensorMap`]).
- [`operations`] — execution: contraction
  ([`operations::tensorcontract_fusion_into`] with
  [`operations::TensorContractSpec`]), tree transforms
  ([`operations::permute_into`], [`operations::braid_into`],
  [`operations::transpose_into`]), tensoradd/trace, and the
  context/cache types the [`prelude::Runtime`] wraps
  ([`operations::TensorContractFusionExecutionContext`]).
- [`dense`] — the dense block execution boundary (GEMM etc.).
- [`matrixalgebra`] — the curated typed `svd_compact` workflow and its bound
  input/result types. Broader unstable factorization and dynamic workflows
  require a direct `tenet-matrixalgebra` dependency.
- `tenet-network` (separate crate) — the `tensor!` macro, the label
  planner (`NetworkIR`, greedy and optional `opt-einsum-path`
  optimizers, slicing types), and the pairwise executor over homogeneous
  [`prelude::TensorMap`] operands.

Storage is column-major inside each dense block; symmetric tensors use the
TeNeT's canonical **coupled-sector matrix layout** ([`prelude::TensorMap::data`]
exposes the flat storage). Axis numbers are zero-based, codomain axes
first.

### Two expert APIs

**Typed const-generic API** (`core::TensorMap<T, NOUT, NIN>` plus the curated
[`operations`] functions `tensorcontract_into`, `tensoradd_into`,
`tensortrace_into`, `permute_into`, `braid_into`, and `transpose_into`): rank
is in the type and outputs are preallocated. For broader unstable `_into`
families, depend directly on `tenet-tensors`. Example — a plain dense matrix
product:

```rust
use tenet::core::{TensorMap, TensorMapSpace};
use tenet::operations::{tensorcontract_into, TensorContractSpec};

let space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
// Column-major storage for [[1, 2], [3, 4]] and [[5, 6], [7, 8]].
let a = TensorMap::<f64, 1, 1>::from_vec(vec![1.0, 3.0, 2.0, 4.0], space.clone()).unwrap();
let b = TensorMap::<f64, 1, 1>::from_vec(vec![5.0, 7.0, 6.0, 8.0], space.clone()).unwrap();
let mut c = TensorMap::<f64, 1, 1>::from_vec(vec![0.0; 4], space).unwrap();

tensorcontract_into(
    &mut c,
    &a,
    &b,
    TensorContractSpec::with_default_output_order(&[1], &[0]),
    1.0,
    0.0,
)
.unwrap();
assert_eq!(c.data(), &[19.0, 43.0, 22.0, 50.0]);
```

**Dynamic API** (`TensorContractFusionExecutionContext::tensorcontract_fusion_dyn_into`
and `TreeTransformExecutionContext::tree_transform_dyn_into`): rank is a runtime
value with no ceiling. The curated matrix-algebra facade exposes typed
`svd_compact` and its bound input/result types; use `tenet-matrixalgebra` directly
for broader unstable dynamic workflows.

### If you are coming from TensorKit

| TensorKit idea | TeNeT user layer | expert-layer internals |
| --- | --- | --- |
| `TensorMap` | [`prelude::TensorMap`] | [`core::TensorMap`], [`operations::DynamicFusionMapSpace`] + flat data |
| `U1Space(-1 => 2, ...)`, `Vect[...]` | [`prelude::GradedSpace`] with a concrete provider | [`core::SectorLeg`] + per-sector degeneracies |
| `V'` (dual space) | [`prelude::GradedSpace::try_dual`] | dual flag + dualized sectors on [`core::SectorLeg`] |
| `@tensor` | `tensor!` over [`prelude::TensorMap`] (crate `tenet-network`) | planner IR -> pairwise [`operations::tensorcontract_fusion_into`] |
| `permute` / `braid` / `transpose` | [`prelude::TensorMap`] methods of the same names | [`operations::permute_into`] / `braid_into` / `transpose_into` |
| SVD / QR / LQ / orthogonalization / eigensolvers | [`prelude::TensorMap`] methods with the TensorKit 0.17 names | curated [`matrixalgebra::svd_compact`] typed workflow; broader unstable APIs are in `tenet-matrixalgebra` |
| `dot` / `norm` / `axpby` | [`prelude::TensorMap::inner`] / `norm` / `add` / `scale` | weighted block inner products |
| implicit global caches | [`prelude::Runtime`] | [`operations::TensorContractFusionExecutionContext`], tree-transform caches, dense executor |
| hom space / fusion-tree basis | (implicit in `TensorMap` construction) | [`core::FusionTreeHomSpace`], [`core::FusionTensorMapSpace`] |

Two details when translating Julia examples: Julia is one-based, TeNeT
axis lists are zero-based; and TensorKit hides flat block storage behind
array syntax, while [`prelude::TensorMap::data`] shows it directly.

Function rustdoc records TeNeT's contract and any relevant semantic difference;
the pinned source coordinates for external correspondence live in
`tenet/references.md`. The provider-typed public cutover is recorded in issue
#727.

## 7. Runtime, backends, and performance

A [`prelude::Runtime`] is built once with [`prelude::RuntimeBuilder`] and then
carried implicitly by every tensor made from it. The builder is where you pick
execution policy — none of it appears in everyday op code:

- **Device.** `Runtime::builder().cuda(device)` selects CUDA storage (phase 1;
  see the limitations below). The default is the host CPU backend.
- **Dense backends.** [`prelude::RuntimeBuilder::linalg_backend`] /
  `gemm_backend` ([`prelude::LinalgBackend`]) choose the dense factorization /
  GEMM providers. Backends are first-class and selectable — see
  `docs/backend_policy.md`.
- **Threads.** `dense_threads` sizes the dense executor pool and
  `recoupling_threads` the tree-transform recoupling. Ops on a shared
  `Runtime` scale with outer threads: each standalone op leases a per-rule
  context (and a dense executor for factorizations) and runs lock-free, so a
  `Runtime` is cheap to `clone` across threads.
- **Plan cache.** The `tensor!` frontend caches contraction plans keyed by
  network topology ([`prelude::PlanCacheConfig`] / [`prelude::Optimizer`] /
  [`prelude::ReplanPolicy`], set via `plan_cache` / `optimizer`). Reusing the
  same runtime across repeated contractions of the same shape (an iTEBD/CTMRG
  sweep) reuses the cached order and the warm per-rule structural caches.

```rust
use tenet::prelude::*;

// A runtime with an explicit dense-thread budget and a plan-cache policy.
let rt = Runtime::builder()
    .dense_threads(4)
    .plan_cache(PlanCacheConfig::default())
    .build()?;
let v = GradedSpace::try_new_owned(
    U1FusionRule,
    [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    false,
)?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
assert!(a.runtime().shares_state_with(&rt));
# Ok::<(), Error>(())
```

**Performance notes.** The hot loop wants a shared, reused `Runtime`: the plan
cache amortizes order search, and the per-rule recoupling/structure caches warm
up on first use and stay warm. For typed workloads, prefer `compose` or
`tensor!` over hand-spelling `contract` axis lists when the categorical
composition is what you mean (`compose` can skip the fermionic twist). Truncated factorizations are
quantum-dimension weighted, so a `Rank(n)` budget bounds the *weighted* bond
dimension — size budgets against `GradedSpace::dim`, not raw sector counts.

## 8. Current Limitations

Honest list, as of this writing:

- **CUDA support remains deliberately narrow.**
  `Runtime::builder().cuda(device)` + `to_cuda()`/`to_host()` support direct
  dense-`f64` contractions, selected arithmetic and reductions (including
  norms), and the typed compact QR, compact SVD, and truncated-SVD paths.
  Unsupported storage/scalar/provider combinations and unwired operations
  still return an explicit `UnsupportedOnDevice`-style error; nothing falls
  back to the CPU silently.
- **No hyperedge/batch labels in `tensor!`** (a label appearing three or
  more times is a compile error). Partial traces (`a[i, i; j]`) and full
  traces are supported.
- **No automatic dtype promotion**: mixing `f64` and `c64` operands in
  `tensor!` is rejected at compile time; widen explicitly with `to_c64()`.
- **Memory-bounded slicing is planned but not executable yet**: the
  slicing planner IR is ported, the sliced executor over symmetric legs
  is future work (sector-granular slicing).
