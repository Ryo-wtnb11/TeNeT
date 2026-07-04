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
let s = Space::su2([(0, 2), (1, 2)]);
assert_eq!(s.dim(), 6);

// Tensors on codomain <- domain leg lists.
let a = Tensor::rand(&rt, [&v, &v], [&v, &v])?;
assert_eq!((a.codomain_rank(), a.domain_rank(), a.rank()), (2, 2, 4));
let z = Tensor::zeros(&rt, [&v], [&v])?;
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

### Which legs may contract?

Contraction compatibility is decided by **Space identity** (the TensorKit
contract): two legs contract when they carry the same `Space` and exactly
one of the two sits on a domain side. A codomain leg built from `v`
contracts a domain leg built from the same `v`; to contract two same-side
legs (e.g. domain against domain), build one of them from `v.dual()`.

Space or rule mismatches are **runtime** typed errors ([`prelude::Error`]:
`RuleMismatch`, `RuntimeMismatch`, `InvalidArgument`, or a bubbled-up
expert-layer error). Label mistakes inside `tensor!` (dangling or repeated
labels) are **compile-time** errors.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);

// Codomain-vs-domain legs of the same Space contract directly...
let a = Tensor::rand(&rt, [&v], [&v])?;
let _ = a.compose(&a)?;

// ...domain-vs-domain legs need one side built from the dual space.
let b = Tensor::rand(&rt, [&v], [&v.dual()])?;
let _ = a.contract(&b, &[1], &[1])?;

// Mixing fusion rules is a typed runtime error.
let z = Tensor::rand(&rt, [&Space::z2([(0, 1), (1, 1)])], [&Space::z2([(0, 1), (1, 1)])])?;
assert!(matches!(a.compose(&z), Err(Error::RuleMismatch)));

// So is mixing runtimes.
let rt2 = Runtime::builder().build()?;
let c = Tensor::rand(&rt2, [&v], [&v])?;
assert!(matches!(a.compose(&c), Err(Error::RuntimeMismatch)));
# Ok::<(), Error>(())
```

Caveat: today this contract is only honored for spaces whose sector
content is closed under dualization (any Z2/fZ2/SU2 space, and U(1)
spaces with charge sets symmetric under negation, like the `{-1, 0, 1}`
above). See the limitations section.

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

```rust
use tenet::prelude::*;
use tenet_network::tensor;

let rt = Runtime::builder().build()?;
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
let a = Tensor::rand(&rt, [&v, &v], [&v, &v])?;
let b = Tensor::rand(&rt, [&v, &v], [&v, &v])?;

// Pairwise contraction with an explicit output signature.
let c = tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n])?;
assert_eq!((c.codomain_rank(), c.domain_rank()), (2, 2));

// conj() + rank-0 output: <a, a> equals the weighted norm squared.
let n2 = tensor!([] = conj(a)[i, j; k, l] * a[i, j; k, l])?.scalar()?;
let norm = a.norm()?;
assert!((n2 - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

// A single operand is a permute.
let p = tensor!([j, i; m, n] = c[i, j; m, n])?;
assert_eq!(p.rank(), 4);

// N-ary: the psi-H-psi energy shape; greedy planning picks the order.
let psi = Tensor::rand(&rt, [&v], [&v, &v])?;
let h = Tensor::rand(&rt, [&v], [&v])?;
let e = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])?.scalar()?;
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

- [`prelude::Tensor::compose`] — categorical composition `a * b` (domain
  of `a` against codomain of `b`, leg by leg).
- [`prelude::Tensor::contract`] — contract arbitrary axis pairs; output is
  `a`'s open axes (ascending) as codomain, `b`'s open axes as domain.
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
let a = Tensor::rand(&rt, [&v, &v], [&v, &v])?;
let b = Tensor::rand(&rt, [&v, &v], [&v, &v])?;

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

## 3. Decompositions

All decomposition names follow TensorKit 0.17 / MatrixAlgebraKit, applied
per coupled sector across the codomain | domain split:

- [`prelude::Tensor::svd_trunc`] — truncated SVD; see below.
- [`prelude::Tensor::svd_compact`] / [`prelude::Tensor::svd_full`] /
  [`prelude::Tensor::svd_vals`].
- [`prelude::Tensor::qr_compact`] / [`prelude::Tensor::qr_full`],
  [`prelude::Tensor::lq_compact`] / [`prelude::Tensor::lq_full`].
- [`prelude::Tensor::left_orth`] / [`prelude::Tensor::right_orth`] —
  TensorKit's default kinds (QR / LQ). **Deviation:** the
  positive-diagonal gauge (`positive = true`) is *not* applied.
- [`prelude::Tensor::left_null`] / [`prelude::Tensor::right_null`],
  [`prelude::Tensor::left_polar`] / [`prelude::Tensor::right_polar`].
- [`prelude::Tensor::eigh_full`] / [`prelude::Tensor::eigh_trunc`] /
  [`prelude::Tensor::eigh_vals`] — Hermitian eigendecomposition.
- [`prelude::Tensor::exp`] / [`prelude::Tensor::inv`] /
  [`prelude::Tensor::pinv`] — matrix functions of endomorphisms.

Truncation is controlled by [`prelude::Truncation`]: `Full`,
`Rank(n)` (`Truncation::rank(n)`), `Tolerance { atol, rtol }`
(`absolute_cutoff` / `relative_cutoff`), `DiscardWeight { rtol }`, and
`All(vec)` (intersection of rules). All bounds and reported errors are
**quantum-dimension weighted**: `Rank(n)` bounds the weighted kept bond
dimension, and the `error` field of [`prelude::SvdTrunc`] /
[`prelude::EighTrunc`] is the weighted 2-norm of everything discarded, so
`|t - u s vh| == error` in the weighted Frobenius norm.

A worked mini-example — split a rank-4 tensor 2 | 2, truncate the bond,
and check the reported error against the actual reconstruction distance:

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
let t = Tensor::rand(&rt, [&v, &v], [&v, &v])?;

// Truncated SVD across the codomain | domain split (weighted bond <= 6).
let svd = t.svd_trunc(&Truncation::rank(6))?;
assert_eq!((svd.u.codomain_rank(), svd.u.domain_rank()), (2, 1));
assert_eq!((svd.vh.codomain_rank(), svd.vh.domain_rank()), (1, 2));

// The kept spectra are reported per coupled sector...
assert!(!svd.singular_values.is_empty());

// ...and the reported error is exactly the reconstruction distance.
let recon = svd.u.compose(&svd.s)?.compose(&svd.vh)?;
let diff = recon.add(&t, 1.0, -1.0)?.norm()?;
assert!((diff - svd.error).abs() <= 1e-8 * (1.0 + svd.error));

// Orthogonality: q from QR is an isometry (q^H q = id), so q q^H t' = t'
// for t' = q r.
let (q, r) = t.qr_compact()?;
let qr = q.compose(&r)?;
let diff = qr.add(&t, 1.0, -1.0)?.norm()?;
assert!(diff <= 1e-10 * (1.0 + t.norm()?));
# Ok::<(), Error>(())
```

To split a tensor along a different bipartition than its current
codomain | domain split, `permute` (or a single-operand `tensor!`) first —
that is exactly what the next section does.

## 4. Worked Example: a U(1) Two-Site Imaginary-Time Step

The simple-update kernel: apply a two-site gate `exp(-tau h)` to a
two-site wavefunction, regroup the legs around the bond, and truncate the
bond back with `svd_trunc`.

```rust
use tenet::prelude::*;
use tenet_network::tensor;

let rt = Runtime::builder().build()?;

// Physical leg: spin-1/2 with U(1) Sz charges +-1. Virtual bond legs.
let p = Space::u1([(-1, 1), (1, 1)]);
let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);

// Two-site wavefunction  psi : p (x) p <- l (x) r.
let psi = Tensor::rand(&rt, [&p, &p], [&v, &v])?;

// Hermitian two-site Hamiltonian and the imaginary-time gate exp(-tau h).
let h0 = Tensor::rand(&rt, [&p, &p], [&p, &p])?;
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

// u and vh are isometries, so the weighted norms satisfy
// |theta|^2 = |s_kept|^2 + error^2 exactly.
let total = theta.norm()?.powi(2);
let kept = weights.norm()?.powi(2);
assert!((total - (kept + svd.error.powi(2))).abs() <= 1e-8 * (1.0 + total));
println!("truncation error: {:.3e}", svd.error);
# Ok::<(), Error>(())
```

In a real simple-update loop this step runs once per bond per sweep, with
the stored bond weights absorbed and re-extracted around each gate.

## 5. Under the Hood: the Expert Layers

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
  planner ([`NetworkIR`], greedy and optional `opt-einsum-path`
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

**Dynamic `_dyn` API** ([`operations::DynamicFusionMapSpace`] + flat data
slices, `tensorcontract_fusion_dyn_into`, `tree_transform_dyn_into`,
`adjoint_dyn`, `matrixalgebra::svd_*_dyn` / `eigh_*_dyn` / ...): rank is a
runtime value with no ceiling. This is exactly what [`prelude::Tensor`]
calls; drop to it when you need dynamic rank without the user layer's
runtime/rule erasure (e.g. your own context management or scalar types).

### If you are coming from TensorKit

| TensorKit idea | TeNeT user layer | expert-layer internals |
| --- | --- | --- |
| `TensorMap` | [`prelude::Tensor`] | [`core::TensorMap`], [`operations::DynamicFusionMapSpace`] + flat data |
| `U1Space(-1 => 2, ...)`, `Vect[...]` | [`prelude::Space`] (`u1`/`z2`/`fz2`/`su2`/`product`) | [`core::SectorLeg`] + per-sector degeneracies |
| `V'` (dual space) | [`prelude::Space::dual`] | dual flag + dualized sectors on [`core::SectorLeg`] |
| `@tensor` | `tensor!` (crate `tenet-network`) | planner IR -> pairwise [`operations::tensorcontract_fusion_into`] |
| `permute` / `braid` / `transpose` | [`prelude::Tensor`] methods of the same names | [`operations::permute_into`] / `braid_into` / `transpose_into` |
| `tsvd` (0.17: `svd_trunc`), `leftorth` (0.17: `left_orth`), ... | [`prelude::Tensor`] methods with the 0.17 names | [`matrixalgebra`] typed + `_dyn` functions |
| `dot` / `norm` / `axpby` | [`prelude::Tensor::inner`] / `norm` / `add` / `scale` | weighted block inner products |
| implicit global caches | [`prelude::Runtime`] | [`operations::TensorContractFusionExecutionContext`], tree-transform caches, dense executor |
| hom space / fusion-tree basis | (implicit in `Tensor` construction) | [`core::FusionTreeHomSpace`], [`core::FusionTensorMapSpace`] |

Two details when translating Julia examples: Julia is one-based, TeNeT
axis lists are zero-based; and TensorKit hides flat block storage behind
array syntax, while [`prelude::Tensor::data`] shows it directly.

For the full mapping (including storage invariants and per-function
correspondences), see `docs/tensorkit_compatibility_table.md`; for the
user-layer design decisions (why `Runtime`, why no einsum strings, why no
index objects), see `docs/user_api_design.md`.

## 6. Current Limitations

Honest list, as of this writing:

- **Scalars are `f64` only.** `c64` is planned; the expert layers are
  generic, the user layer pins `f64`.
- **`eig_full` / `eig_trunc` / `eig_vals` are pending `c64`**: the general
  eigendecomposition is complex-valued. The expert layer
  (`matrixalgebra::eig_*_dyn`) already supports them.
- **CPU only.** [`prelude::Runtime::builder`] exists so a GPU backend can
  land without an API break, but today the default CPU backend is the
  only choice.
- **`tensor!` re-plans on every call.** Greedy planning is cheap, but a
  shape-keyed plan cache for iterative sweeps is still planned.
- **No trace/diagonal inside one `tensor!` operand** (a repeated label on
  one tensor is a compile error), and no hyperedge/batch labels.
- **`left_orth` / `right_orth` do not apply the positive-diagonal gauge**
  (TensorKit's `positive = true`).
- **Dual pairing requires dualization-closed sector content.** For a U(1)
  space with an asymmetric charge set (e.g. `{0, 1}`, a hardcore boson),
  contracting against the same `Space` (codomain-vs-domain) or its
  `.dual()` (same-side) currently fails with a `SectorMismatch`: the
  user-layer lowering does not yet reconcile the stored-dual convention of
  the expert layer with `Space::dual`. Symmetric charge sets and all
  Z2/fZ2/SU2 spaces (self-dual sectors) are unaffected. The same pairing
  check can reject re-composing truncated SVD factors when the *kept*
  coupled-sector set comes out asymmetric (which is why the example above
  verifies the truncation through norms instead of `u * s * vh`).
