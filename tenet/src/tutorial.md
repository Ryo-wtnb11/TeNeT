# Tutorial

Everyday TeNeT code uses the **user layer**: one `use tenet::prelude::*;`
import gives [`prelude::Runtime`], [`prelude::Space`], [`prelude::Tensor`],
and [`prelude::Truncation`]; the `tensor!` macro (from the `tenet-network`
crate) is the contraction frontend. The expert layers ([`core`],
[`operations`], [`dense`], [`matrixalgebra`]) stay available underneath —
see the appendix at the end.

Every code block in this tutorial runs as a doctest, so it is guaranteed to
compile and pass against the current API.

## 1. Quick Start

A [`prelude::Runtime`] is built once and then carried implicitly by every
tensor created from it (it owns the contraction/tree-transform caches and
the dense backend). A [`prelude::Space`] is a graded vector space for one
tensor leg: `(sector, degeneracy)` pairs plus a dual flag, in TensorKit's
`U1Space(-1 => 2, 0 => 3, 1 => 2)` style. A [`prelude::Tensor`] is a
block-sparse symmetric tensor `codomain <- domain` with dynamic rank.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;

// U(1): (charge, degeneracy) pairs. dim = 2 + 3 + 2.
let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);
assert_eq!(v.dim(), 7);
assert_eq!(v.dual().dual(), v);

// SU(2): (twice_spin, degeneracy) pairs; dim is quantum-dimension
// weighted: 2 * 1 (spin 0) + 2 * 2 (spin 1/2).
let s = Space::su2([(0, 2), (1, 2)]).unwrap();
assert_eq!(s.dim(), 6);

// Tensors on codomain <- domain leg lists.
let a = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;
assert_eq!((a.codomain_rank(), a.domain_rank(), a.rank()), (2, 2, 4));
let z = Tensor::zeros(&rt, Dtype::F64, [&v], [&v])?;
assert_eq!(z.norm()?, 0.0);
# Ok::<(), Error>(())
```

[`prelude::Tensor::from_block_fn`] fills every symmetry-allowed block
element from a closure over the block key and block-local degeneracy
indices:

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = Space::z2([(0, 1), (1, 1)]);

// A diagonal Z2 matrix: 2 on the even block, 3 on the odd block.
let a = Tensor::from_block_fn(&rt, [&v], [&v], |key, _indices| match key {
    BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
    _ => 3.0,
})?;
let b = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
    BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 5.0,
    _ => 7.0,
})?;
assert_eq!(a.compose(&b)?.data(), &[10.0, 21.0]);
# Ok::<(), Error>(())
```

### Scalar dtype

User-layer tensors store either `f64` or `c64`, fixed at construction by
the [`prelude::Dtype`] token: `Tensor::zeros(&rt, Dtype::F64, ...)`,
`Tensor::rand(&rt, Dtype::C64, ...)`, and so on (TensorKit's
`rand(ComplexF64, W ← V)` leading type argument).
[`prelude::Tensor::from_block_fn`] needs no token — the dtype follows the
fill closure's return type (`f64` or [`prelude::Complex64`]). TeNeT does
not promote mixed dtypes implicitly: widen explicitly with
[`prelude::Tensor::to_c64`].

Scalar results ([`prelude::Tensor::scalar`], [`prelude::Tensor::inner`],
[`prelude::Tensor::tr`]) return a [`prelude::Scalar`] whose variant matches
the tensor's dtype — real tensors give `Scalar::F64`, so no `.re` noise on
real code paths; use `re()` / `im()` / `try_f64()` / `to_c64()` to unwrap.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);

let re = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
let cx = Tensor::from_block_fn(&rt, [&v], [&v], |_, indices| {
    Complex64::new(indices[0] as f64, -(indices[1] as f64))
})?;
assert_eq!(re.dtype(), Dtype::F64);
assert_eq!(cx.dtype(), Dtype::C64);

// inner on f64 tensors is Scalar::F64: exactly real, try_f64() succeeds.
let inner = re.inner(&re)?.try_f64()?;
assert!((inner - re.norm()?.powi(2)).abs() <= 1e-10 * (1.0 + inner));

// inner on c64 tensors is Scalar::C64.
let cc = cx.inner(&cx)?;
assert!(matches!(cc, Scalar::C64(_)));
assert!(cc.im().abs() <= 1e-12 * (1.0 + cc.re()));

assert!(matches!(re.compose(&cx), Err(Error::DtypeMismatch)));
assert!(re.to_c64().compose(&cx).is_ok());
# Ok::<(), Error>(())
```

### Which legs may contract?

Contraction compatibility follows TensorKit's dual-pairing convention:
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
contraction, and TensorKit-style `flip` conventions.

Space or rule mismatches are **runtime** typed errors ([`prelude::Error`]:
`RuleMismatch`, `RuntimeMismatch`, `InvalidArgument`, or a bubbled-up
expert-layer error). Label mistakes inside `tensor!` (dangling or repeated
labels) are **compile-time** errors.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
// Works for any charge set, including ones that are not symmetric under
// negation (a hardcore boson).
let v = Space::u1([(0, 2), (1, 1)]);

// Codomain-vs-domain legs of the same Space contract directly...
let a = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
let _ = a.compose(&a)?;

// ...domain-vs-domain legs need one side built from the dual space.
let b = Tensor::rand(&rt, Dtype::F64, [&v], [&v.dual()])?;
let _ = a.contract(&b, &[1], &[1])?;

// Mixing fusion rules is a typed runtime error.
let z = Tensor::rand(&rt, Dtype::F64, [&Space::z2([(0, 1), (1, 1)])], [&Space::z2([(0, 1), (1, 1)])])?;
assert!(matches!(a.compose(&z), Err(Error::RuleMismatch)));

// So is mixing runtimes.
let rt2 = Runtime::builder().build()?;
let c = Tensor::rand(&rt2, Dtype::F64, [&v], [&v])?;
assert!(matches!(a.compose(&c), Err(Error::RuntimeMismatch)));
# Ok::<(), Error>(())
```

## 2. Contraction

### `tensor!` — the way to contract

The `tensor!` macro (crate `tenet-network`) is @tensor-style index
notation. The output signature comes first: `[codomain; domain]`; the `;`
is optional (`[a, b]` = all-codomain output) and `[]` is a rank-0 (scalar)
output, read out with [`prelude::Tensor::scalar`]. `conj(x)` marks an
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
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
let a = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;
let b = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;

// Pairwise contraction with an explicit output signature.
let c = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n])?;
assert_eq!((c.codomain_rank(), c.domain_rank()), (2, 2));

// conj() + rank-0 output computes the weighted self inner product.
let n2 = tensor!([] = conj(a)[i, j; k, l] * a[i, j; k, l])?.scalar()?.try_f64()?;
let norm = a.norm()?;
assert!((n2 - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

// A single operand is a permute.
let p = tensor!([j, i; m, n] = c[i, j; m, n])?;
assert_eq!(p.rank(), 4);

// N-ary: an energy contraction; greedy planning picks the order.
let psi = Tensor::rand(&rt, Dtype::F64, [&v], [&v, &v])?;
let h = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
let e = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])?.scalar()?.try_f64()?;
assert!(e.is_finite());
# Ok::<(), Error>(())
```

Label errors do not survive to runtime — this does not compile because `k`
and `j` each appear once without being output labels:

```rust,compile_fail
use tenet::prelude::*;
use tenet_network::tensor;

fn wrong(a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    tensor!([i; m] = a[i; k] * b[j; m])
}
```

A written `;` split that contradicts the tensor's actual codomain rank is
checked at plan time (runtime `InvalidArgument`), since the macro cannot
see the tensor's shape.

### The method API underneath

`tensor!` lowers to pairwise steps over the explicit method API, which is
available directly when you want to spell the axes:

- [`prelude::Tensor::compose`] — categorical composition (TensorKit
  `A * B` / `mul!`), also spelled `&a * &b`. **No** fermionic supertrace
  twist on dual composed legs.
- [`prelude::Tensor::contract`] — contract arbitrary axis pairs (TensorKit
  `tensorcontract!`); output is `a`'s open axes (ascending) as codomain,
  `b`'s open axes as domain. Like `tensor!`, this **twists** dual
  contracted legs on fermionic rules — bosonic results are identical to
  `compose`, fermionic ones can differ by signs; see the fermionic note on
  [`prelude::Tensor::compose`].
- [`prelude::Tensor::contract_ordered`] — same with an explicit output
  axis order (TensorKit's `pAB`).
- [`prelude::Tensor::permute`] / [`prelude::Tensor::braid`] /
  [`prelude::Tensor::transpose`] — TensorKit's leg re-arrangements
  (symmetric braiding / explicit braid levels / planar transpose).
- [`prelude::Tensor::adjoint`] — dagger: swaps codomain and domain.

Axes are zero-based and flat: codomain axes first, then domain axes.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
let a = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;
let b = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;

let c1 = a.compose(&b)?;
let c2 = a.contract(&b, &[2, 3], &[0, 1])?;
assert_eq!(c1.data(), c2.data());
let _c3 = a.contract_ordered(&b, &[2, 3], &[0, 1], &[1, 0, 2, 3])?;

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
[`prelude::Tensor::norm`], [`prelude::Tensor::normalize`],
[`prelude::Tensor::inner`] / [`prelude::Tensor::dot`],
[`prelude::Tensor::scale`], [`prelude::Tensor::add`] (the `α·self + β·other`
combination, covering TensorKit's `axpy!`/`axpby!`),
[`prelude::Tensor::tr`], and [`prelude::Tensor::zeros_like`] (TensorKit
`zerovector`). Structural predicates match TensorKit's
`ishermitian`/`isantihermitian`/`isisometric`/`isunitary`/`isposdef`, with the
`(t ± t†)/2` projectors [`prelude::Tensor::project_hermitian`] /
[`prelude::Tensor::project_antihermitian`].

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
let a = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
let b = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;

// α·a + β·b (TensorKit axpby), scaling, and unit normalization.
let _diff = a.add(&b, 1.0, -1.0)?;    // a - b
let _scaled = a.scale(2.0)?;
let unit = a.normalize()?;
assert!((unit.norm()? - 1.0).abs() <= 1e-12);

// inner / dot agree, and norm² == <a, a>.
let ip = a.inner(&a)?.try_f64()?;
assert!((ip - a.norm()?.powi(2)).abs() <= 1e-10 * (1.0 + ip));

// A same-shape zero (zerovector) and the trace of an endomorphism.
let zero = a.zeros_like()?;
assert_eq!(zero.norm()?, 0.0);
let _trace = a.tr()?;

// Structural predicates: the identity is Hermitian, unitary, positive definite.
let id = Tensor::id(&rt, Dtype::F64, [&v])?;
assert!(id.is_hermitian(1e-12)? && id.is_unitary(1e-12)? && id.is_posdef(1e-12)?);
# Ok::<(), Error>(())
```

### Index operations

Leg rearrangements follow TensorKit's names. Axis lists are flat and
zero-based (codomain axes first). [`prelude::Tensor::permute`] chooses new
codomain/domain axis lists; [`prelude::Tensor::repartition`] re-splits the
legs at a codomain count while keeping their order (TensorKit `repartition`);
[`prelude::Tensor::transpose`] is the planar transpose,
[`prelude::Tensor::adjoint`] the dagger, and
[`prelude::Tensor::twist`] / [`prelude::Tensor::flip`] act on chosen legs.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
let a = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;

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

A [`prelude::Space`] carries `(sector, degeneracy)` content queried through
[`prelude::Tensor`]-free space methods: [`prelude::Space::sectors`],
[`prelude::Space::degeneracy`] (TensorKit `dim(V, c)`),
[`prelude::Space::has_sector`] (TensorKit `hassector`),
[`prelude::Space::fuse`] (`⊗`), and [`prelude::Space::oplus`] (`⊕`). SU(3)
irreps do not fit the [`prelude::SectorLabel`] enum, so they read back through
[`prelude::Space::su3_sectors`] / `su3_degeneracy` as `(p, q)` Dynkin labels.

```rust
use tenet::prelude::*;

let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);

// Enumerate sectors and query membership / degeneracy.
assert_eq!(v.sectors().len(), 3);
assert_eq!(v.degeneracy(SectorLabel::U1(0)), Some(3));
assert!(v.has_sector(SectorLabel::U1(1)));
assert!(!v.has_sector(SectorLabel::U1(9)));

// fuse (⊗) collapses two legs; oplus (⊕) sums per-sector degeneracies.
let w = Space::u1([(0, 1), (1, 1)]);
assert_eq!(v.fuse(&w)?.dim(), v.dim() * w.dim());
assert_eq!(v.oplus(&w)?.degeneracy(SectorLabel::U1(0)), Some(3 + 1));

// SU(2) dims are quantum-dimension weighted; SU(3) reads back (p, q) irreps.
let s = Space::su2([(0, 1), (1, 1)]).unwrap();          // spin 0 ⊕ spin 1/2
assert_eq!(s.dim(), 1 + 2);
let fundamental = Space::su3([((1, 0), 1)])?;  // the 3
assert_eq!(fundamental.su3_sectors()?, vec![((1, 0), 1)]);
# Ok::<(), Error>(())
```

For the complete TensorKit-name lookup — every user-facing 0.17 export, its
TeNeT name, and the rationale for anything spelled or gated differently — see
`docs/tk_api_parity.md`.

## 4. Decompositions

All decomposition names follow TensorKit 0.17 / MatrixAlgebraKit, applied
per coupled sector across the codomain | domain split:

- [`prelude::Tensor::svd_trunc`] — truncated SVD; see below.
- [`prelude::Tensor::svd_compact`] / [`prelude::Tensor::svd_full`] /
  [`prelude::Tensor::svd_vals`].

For generic-fusion SU(3) tensors, the supported decomposition subset is
`svd_compact`, `svd_trunc`, `svd_vals`, `qr_compact`, `lq_compact`,
`left_orth`, and `right_orth`. Full SVD/QR/LQ, null spaces, spectral
decompositions, polar decompositions, and `inv`/`pinv`/`exp` are currently
unsupported. The full and complementary-space operations require a
multiplicity-aware completion; TeNeT returns `Error::UnsupportedForRule`
instead of applying a multiplicity-free construction that would produce the
wrong space.
- [`prelude::Tensor::qr_compact`] / [`prelude::Tensor::qr_full`],
  [`prelude::Tensor::lq_compact`] / [`prelude::Tensor::lq_full`].
- [`prelude::Tensor::left_orth`] / [`prelude::Tensor::right_orth`] —
  TensorKit's default kinds (QR / LQ), including the positive-diagonal
  gauge (`positive = true`, MatrixAlgebraKit's default).
- [`prelude::Tensor::left_null`] / [`prelude::Tensor::right_null`],
  [`prelude::Tensor::left_polar`] / [`prelude::Tensor::right_polar`].
- [`prelude::Tensor::eigh_full`] / [`prelude::Tensor::eigh_trunc`] /
  [`prelude::Tensor::eigh_vals`] — Hermitian eigendecomposition.
- [`prelude::Tensor::eig_full`] / [`prelude::Tensor::eig_trunc`] /
  [`prelude::Tensor::eig_vals`] — general eigendecomposition; outputs are
  `c64` even for real input.
- [`prelude::Tensor::exp`] / [`prelude::Tensor::inv`] /
  [`prelude::Tensor::pinv`] — matrix functions of endomorphisms.

Hermitian `eigh_*` keeps the input dtype and reports real eigenvalues.
General `eig_*` is complex-valued by construction, so the returned
diagonal/eigenvector tensors are always `c64`.

Truncation is controlled by [`prelude::Truncation`]: `Full`,
`Rank(n)` (`Truncation::rank(n)`), `Tolerance { atol, rtol }`
(`absolute_cutoff` / `relative_cutoff`), `DiscardWeight { rtol }`, and
`All(vec)` (intersection of rules). All bounds and reported errors are
**quantum-dimension weighted**: `Rank(n)` bounds the weighted kept bond
dimension, and the `error` field of [`prelude::SvdTrunc`] /
[`prelude::EighTrunc`] is the weighted 2-norm of everything discarded, so
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
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
let t = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;

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

// General eigendecomposition is c64 even for real input.
let (d, w) = t.eig_full()?;
assert_eq!(d.dtype(), Dtype::C64);
assert_eq!(w.dtype(), Dtype::C64);

// Hermitian eigendecomposition keeps the real dtype.
let h = t.add(&t.adjoint()?, 0.5, 0.5)?;
let (evals, vecs) = h.eigh_full()?;
assert_eq!(evals.dtype(), Dtype::F64);
assert_eq!(vecs.dtype(), Dtype::F64);
# Ok::<(), Error>(())
```

To split a tensor along a different bipartition than its current
codomain | domain split, `permute` (or a single-operand `tensor!`) first —
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
let p = Space::u1([(-1, 1), (1, 1)]);
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);

// Two-site wavefunction with two physical and two virtual legs.
let psi = Tensor::rand(&rt, Dtype::F64, [&p, &p], [&v, &v])?;

// Hermitian two-site Hamiltonian and the imaginary-time gate.
let h0 = Tensor::rand(&rt, Dtype::F64, [&p, &p], [&p, &p])?;
let h = h0.add(&h0.adjoint()?, 0.5, 0.5)?;
let tau = 0.05;
let gate = h.scale(-tau)?.exp()?;

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

The user layer is a thin, rule-erased face over four expert modules:

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
- [`matrixalgebra`] — factorizations and matrix functions; the
  `Tensor` decomposition methods pass through to the `*_dyn` entry points
  here.
- `tenet-network` (separate crate) — the `tensor!` macro, the label
  planner (`NetworkIR`, greedy and optional `opt-einsum-path`
  optimizers, slicing types), and the pairwise executor over `Tensor`.

Storage is column-major inside each dense block; symmetric tensors use the
TensorKit-equivalent **coupled-sector matrix layout** ([`prelude::Tensor::data`]
exposes the flat storage). Axis numbers are zero-based, codomain axes
first.

### Two expert APIs

**Typed const-generic API** (`core::TensorMap<T, NOUT, NIN>` +
[`operations`] `*_into` functions): rank is in the type, outputs are
preallocated, contexts are explicit. Use it when the rank is statically
known and you want zero per-call allocation surprises or custom
context/cache management. Example — a plain dense matrix product:

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

**Dynamic `_dyn` API** (`tensorcontract_fusion_dyn_into`,
`tree_transform_dyn_into`, and the `matrixalgebra::*_dyn` functions): rank is
a runtime value with no ceiling. Provider-sensitive matrix algebra takes a
validated [`matrixalgebra::BoundDynamicTensorRef`], which borrows a
[`operations::BoundDynamicFusionMapSpace`] together with its flat data. The
fusion provider is therefore inherited from the space rather than supplied as
an independent argument. This is exactly what [`prelude::Tensor`] calls; drop
to it when you need dynamic rank without the user layer's runtime erasure.

### If you are coming from TensorKit

| TensorKit idea | TeNeT user layer | expert-layer internals |
| --- | --- | --- |
| `TensorMap` | [`prelude::Tensor`] | [`core::TensorMap`], [`operations::DynamicFusionMapSpace`] + flat data |
| `U1Space(-1 => 2, ...)`, `Vect[...]` | [`prelude::Space`] (`u1`/`z2`/`fz2`/`su2`/`product`) | [`core::SectorLeg`] + per-sector degeneracies |
| `V'` (dual space) | [`prelude::Space::dual`] | dual flag + dualized sectors on [`core::SectorLeg`] |
| `@tensor` | `tensor!` (crate `tenet-network`) | planner IR -> pairwise [`operations::tensorcontract_fusion_into`] |
| `permute` / `braid` / `transpose` | [`prelude::Tensor`] methods of the same names | [`operations::permute_into`] / `braid_into` / `transpose_into` |
| SVD / QR / LQ / orthogonalization / eigensolvers | [`prelude::Tensor`] methods with the TensorKit 0.17 names | [`matrixalgebra`] typed + `_dyn` functions |
| `dot` / `norm` / `axpby` | [`prelude::Tensor::inner`] / `norm` / `add` / `scale` | weighted block inner products |
| implicit global caches | [`prelude::Runtime`] | [`operations::TensorContractFusionExecutionContext`], tree-transform caches, dense executor |
| hom space / fusion-tree basis | (implicit in `Tensor` construction) | [`core::FusionTreeHomSpace`], [`core::FusionTensorMapSpace`] |

Two details when translating Julia examples: Julia is one-based, TeNeT
axis lists are zero-based; and TensorKit hides flat block storage behind
array syntax, while [`prelude::Tensor::data`] shows it directly.

For the per-export lookup table (every user-facing TensorKit 0.17 function,
its TeNeT name, and why anything differs), see `docs/tk_api_parity.md`. For the
internal naming correspondences and storage invariants, see
`docs/tensorkit_compatibility_table.md`; for the user-layer design decisions
(why `Runtime`, why no einsum strings, why no index objects), see
`docs/user_api_design.md`.

## 7. Runtime, backends, and performance

A [`prelude::Runtime`] is built once with [`prelude::RuntimeBuilder`] and then
carried implicitly by every tensor made from it. The builder is where you pick
execution policy — none of it appears in everyday op code:

- **Device.** `Runtime::builder().cuda(device)` selects CUDA storage (phase 1;
  see the limitations below). The default is the host CPU backend.
- **Dense backends.** [`prelude::RuntimeBuilder::linalg_backend`] /
  `gemm_backend` ([`prelude::LinalgBackend`]) and `transpose_backend`
  ([`prelude::TransposeBackend`]) choose the dense GEMM / transpose kernels.
  Backends are first-class and selectable, never hardcoded — see
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
let v = Space::u1([(0, 2), (1, 1)]);
let a = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
assert!(a.runtime().shares_state_with(&rt));
# Ok::<(), Error>(())
```

**Performance notes.** The hot loop wants a shared, reused `Runtime`: the plan
cache amortizes order search, and the per-rule recoupling/structure caches warm
up on first use and stay warm. Prefer `compose` / the `tensor!` macro over
hand-spelling `contract` axis lists when the categorical composition is what you
mean (it can skip the fermionic twist). Truncated factorizations are
quantum-dimension weighted, so a `Rank(n)` budget bounds the *weighted* bond
dimension — size budgets against `Space::dim`, not raw sector counts.

## 8. Current Limitations

Honest list, as of this writing:

- **CUDA support is phase 1.** `Runtime::builder().cuda(device)` +
  `to_cuda()`/`to_host()` run fully-direct contractions on device
  (verified on A100 against the host results). Everything else on a
  device tensor — index manipulations, decompositions, norms, `c64` —
  returns an explicit `UnsupportedOnDevice`-style error; nothing falls
  back to the CPU silently.
- **No hyperedge/batch labels in `tensor!`** (a label appearing three or
  more times is a compile error). Partial traces (`a[i, i; j]`) and full
  traces are supported.
- **No automatic dtype promotion**: mixing `f64` and `c64` operands is a
  typed error; widen explicitly with `to_c64()`.
- **Memory-bounded slicing is planned but not executable yet**: the
  slicing planner IR is ported, the sliced executor over symmetric legs
  is future work (sector-granular slicing).
