# Tutorial

Everyday TeNeT code starts with `use tenet::prelude::*;`. It provides
[`prelude::Runtime`], [`prelude::GradedSpace`], [`prelude::TensorMap`], the
built-in symmetries, and [`prelude::Truncation`]. Import the `tensor!` macro
separately from `tenet-network`.

This page's Rust blocks are doctests unless marked as syntax-only. The
repository examples are compiled separately; use them for complete programs.

## Quick start

A `GradedSpace` describes one tensor leg. Each U(1) `(charge, degeneracy)` pair
specifies the number of states at that charge. A `TensorMap` is a
symmetry-preserving block-sparse map written `codomain <- domain`. A `Runtime`
supplies the resources for its operations.

Clone the repository and run its quickstart example:

```sh
git clone https://github.com/Ryo-wtnb11/TeNeT.git
cd TeNeT
cargo run -p tenet-network --example quickstart
```

`tenet-rs` is the published package; its Rust crate is named `tenet`. The
command above runs a repository example owned by the `tenet-network` package.
It constructs two U(1) maps and checks their contracted squared norm. Next,
read the [iTEBD example](https://github.com/Ryo-wtnb11/TeNeT/blob/main/tenet-network/examples/itebd_heisenberg.rs)
and its [guide](https://github.com/Ryo-wtnb11/TeNeT/blob/main/docs/itebd_heisenberg.md).

### Constructing a map

`TensorMap::from_block_fn` visits every allowed reduced entry. Its closure gets
the fused-tree labels and the index within the current dense block. Returning
zero is often the clearest way to construct a sparse physical operator.

```rust
use tenet::prelude::*;

let runtime = Runtime::builder().build()?;
let spin = GradedSpace::try_new(
    U1FusionRule,
    [(U1Irrep::new(-1), 1), (U1Irrep::new(1), 1)],
)?;
let sz: TensorMap<U1FusionRule, f64> = TensorMap::from_block_fn(
    &runtime,
    [&spin],
    [&spin],
    |trees, indices| {
        if indices[0] != indices[1] {
            return 0.0;
        }
        match trees.coupled().charge() {
            -1 => -0.5,
            1 => 0.5,
            _ => 0.0,
        }
    },
)?;
assert_eq!(sz.codomain_rank(), 1);
assert_eq!(sz.domain_rank(), 1);
# Ok::<(), Error>(())
```

`zeros`, `id`, `rand`, and `rand_with_seed` cover common initial values.
The constructor validates the symmetry layout before it returns, so invalid
spaces and invalid provider labels become `Error` values rather than partial
tensors.

### Scalars and dual spaces

The scalar is the second type parameter: `TensorMap<R, f64>` or
`TensorMap<R, Complex64>`. Mixed scalar operations are rejected; widen with
[`prelude::TensorMap::to_c64`]. `GradedSpace::try_new` creates a nondual space.
Call [`prelude::GradedSpace::try_dual`] when a dual leg is required.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new(
    U1FusionRule,
    [
        (U1Irrep::new(-1), 1),
        (U1Irrep::new(0), 2),
        (U1Irrep::new(1), 1),
    ],
)?;
let re = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
let cx: TensorMap<U1FusionRule, Complex64> =
    TensorMap::from_block_fn(&rt, [&v], [&v], |_, ij| {
        Complex64::new(ij[0] as f64, -(ij[1] as f64))
    })?;

let inner = re.inner(&re)?;
assert!((inner - re.norm()?.powi(2)).abs() <= 1e-10 * (1.0 + inner));
assert!(re.to_c64().compose(&cx).is_ok());
assert!(v.try_dual()?.try_dual()?.sectors()? == v.sectors()?);
# Ok::<(), Error>(())
```

Use `f64` when real arithmetic is sufficient. Use `Complex64` from the
prelude for complex states and operators. A method returning a scalar, such as
`inner`, returns that same payload type. There is no implicit widening in a
network: convert its operands before writing the contraction.

### Blocks and contraction orientation

[`prelude::TensorMap::blocks`] reads reduced blocks with their fusion-tree
labels. To contract two legs, their oriented spaces must be dual. A codomain
leg and a domain leg made from the same space pair directly. For two legs on
the same side, construct one from `v.try_dual()?`. See [`mathematics`] for the
full convention.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new(
    U1FusionRule,
    [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
)?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
assert!(a.compose(&a).is_ok());

let dual = v.try_dual()?;
let b = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&dual])?;
let _ = a.contract(&b, &[1], &[1], &[0, 1])?;

let mut count = 0;
for (_trees, values) in a.blocks()? {
    assert_eq!(values.shape().len(), 2);
    count += 1;
}
assert_eq!(count, a.block_count());
# Ok::<(), Error>(())
```

Different Rust provider types and scalar types are compile-time mismatches.
Two spaces with the same provider type but different rule identities return
[`prelude::Error::RuleMismatch`]. Incompatible spaces and runtimes also return
[`prelude::Error`].

The block buffer is reduced data, so it is not a carrier-basis matrix. In
particular, its length depends on fusion sectors and degeneracies, not just the
product of leg dimensions. Prefer `blocks` for inspection and ordinary tensor
methods for computation. `data` is useful only when the reduced layout itself
is the desired representation.

## Contraction

### `tensor!`

`tensor!` is index notation for homogeneous `TensorMap` operands. Its output
signature is `[codomain; domain]`; `[]` is a scalar output, read with
[`prelude::TensorMap::scalar`]. A label shared by two operands is contracted;
a label used once must occur in the output. `conj(x)` marks an adjoint operand.
For three or more operands, the runtime's configured optimizer chooses the
pairwise order. There are no einsum strings.

The facade crate cannot use `tensor!` in its own rustdoc because the macro is
owned by `tenet-network`. The compiled quickstart and iTEBD examples exercise
it instead.

```rust,ignore
use tenet_network::tensor;

// Syntax only: `a` and `b` are compatible TensorMap values.
let c = tensor!([i; k] = a[i; j] * b[j; k])?;
let expectation = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])?;
```

The macro reports malformed labels at compile time. A written `;` split that
does not match a tensor's runtime rank is reported when the plan is built.

Output labels also define their order. For example, `[i, k; m]` keeps `i` and
`k` as codomain legs, in that order, then places `m` in the domain. A one-input
macro call is therefore a permutation, and a scalar network has `[]` as its
output signature. Write the result orientation deliberately; it determines
the map on which following methods operate.

The default optimizer is greedy. A `RuntimeBuilder` can select another
available optimizer before tensors are constructed. This changes contraction
planning, not the mathematical result. For a repeated network shape, reuse
the same runtime rather than adding a separate planning layer in application
code.

### Method API

Use methods when the contracted axes or output order are more direct than
labels. [`prelude::TensorMap::compose`] is the categorical map composition.
[`prelude::TensorMap::contract`] accepts arbitrary axis pairs and an explicit
flat output order. Axes are zero-based, with codomain axes before domain axes.
`permute`, `repartition`, `transpose`, `adjoint`, `twist`, and `flip` rearrange
legs; their conventions are specified in [`mathematics`].

`compose` is the right operation for ordinary map composition. `contract` is
the general operation; on fermionic rules its selected dual legs carry the
contraction twist, so its result need not match `compose`.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
)?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;
let b = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;
let c = a.compose(&b)?;
let same = a.contract(&b, &[2, 3], &[0, 1], &[0, 1, 2, 3])?;
assert_eq!(c.data(), same.data());

let reordered = c.permute(&[0, 2], &[1, 3])?;
assert_eq!((reordered.codomain_rank(), reordered.domain_rank()), (2, 2));
assert_eq!(c.repartition(1)?.repartition(2)?.data(), c.data());
# Ok::<(), Error>(())
```

For simple relabeling, use `permute`. Use `repartition` only to change where
the existing ordered leg list is split between codomain and domain. `adjoint`
swaps the orientation and conjugates entries; `transpose` is the planar
operation. These are distinct operations even where a real, bosonic example
makes their values look similar.

## Tensor algebra and spaces

`TensorMap` has ordinary vector operations: `norm`, `normalize`, `inner`,
`scale`, `add`, `tr`, and `zeros_like`. It also supplies structural predicates
such as `is_hermitian`, `is_unitary`, and `is_posdef`.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
)?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
let b = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v], [&v])?;
let difference = a.add(&b, 1.0, -1.0)?;
assert!(difference.norm()? >= 0.0);
assert!((a.normalize()?.norm()? - 1.0).abs() <= 1e-12);
assert_eq!(a.zeros_like().norm()?, 0.0);
let id = TensorMap::<U1FusionRule, f64>::id(&rt, [&v])?;
assert!(id.is_hermitian(1e-12)? && id.is_unitary(1e-12)?);
# Ok::<(), Error>(())
```

`add(&other, alpha, beta)` computes `alpha * self + beta * other`; both maps
must have compatible runtime, space, scalar, and storage. `inner` and `norm`
use TeNeT's weighted block inner product. Check `norm()` before `normalize()`
when zero tensors are possible: normalization divides by that norm, so a zero
input produces non-finite values rather than an error.

`GradedSpace` exposes its sectors, per-sector degeneracies, total dimension,
direct sum (`oplus`), and fusion (`fuse`). The total dimension includes each
sector's quantum dimension.

```rust
use tenet::prelude::*;

let v = GradedSpace::try_new(
    U1FusionRule,
    [(-1, 2), (0, 3), (1, 2)].map(|(q, n)| (U1Irrep::new(q), n)),
)?;
let w = GradedSpace::try_new(
    U1FusionRule,
    [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
)?;
assert_eq!(v.degeneracy(&U1Irrep::new(0))?, 3);
assert_eq!(v.fuse(&w)?.dim()?, v.dim()? * w.dim()?);
assert_eq!(v.oplus(&w)?.degeneracy(&U1Irrep::new(0))?, 4);
# Ok::<(), Error>(())
```

Symmetries can be combined with `product`. The order and association are part
of the provider type, so choose an order once for a model. The payload scalar
remains separate from the provider's categorical coefficient scalar.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let rule = FermionParityFusionRule.product(U1FusionRule);
let v = GradedSpace::try_new(
    rule,
    [
        (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
        (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
    ],
)?;
assert_eq!(TensorMap::<_, f64>::zeros(&rt, [&v], [&v])?.block_count(), 2);
# Ok::<(), Error>(())
```

An external provider uses the same user API when it implements the required
traits. Its laws and current capability requirements are in
[`docs/provider_interface.md`](../../docs/provider_interface.md).

Product providers are useful for simultaneous conserved quantities such as
fermion parity and particle number. They do not provide an automatic change of
basis between separately constructed models. Construct the desired sector
labels explicitly, then use the same space and tensor operations as for U(1).

## Decompositions

Decompositions act independently in each coupled sector across the current
codomain | domain split. `svd_trunc` returns `u`, `s`, `vh`, and the discarded
weighted Frobenius norm. `Truncation::rank(n)` bounds the weighted kept bond
dimension; tolerance constructors and `and` combine additional limits.

The main method families are `svd_compact`/`svd_full`/`svd_vals`,
`qr_compact`/`lq_compact`, `left_orth`/`right_orth`,
`eigh_full`/`eigh_trunc`, `eig_full`/`eig_trunc`, and endomorphism methods
`exp`, `inv`, and `pinv`. General eigendecomposition returns `c64` data even
for real input.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new(
    U1FusionRule,
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
)?;
let t = TensorMap::<U1FusionRule, f64>::rand(&rt, [&v, &v], [&v, &v])?;
let svd = t.svd_trunc(&Truncation::rank(6))?;
let reconstructed = svd.u.compose(&svd.s)?.compose(&svd.vh)?;
let error = reconstructed.add(&t, 1.0, -1.0)?.norm()?;
assert!((error - svd.error).abs() <= 1e-8 * (1.0 + svd.error));

let (q, r) = t.qr_compact()?;
assert!(q.compose(&r)?.add(&t, 1.0, -1.0)?.norm()? <= 1e-10 * (1.0 + t.norm()?));
# Ok::<(), Error>(())
```

To factor a different bipartition, use `permute` or `repartition` first.

The returned `s` is a diagonal tensor map on the newly introduced bond space.
Keep it when a tensor-network algorithm needs bond weights, or absorb it into
one neighboring factor when it does not. The `error` field measures discarded
weight for the selected truncation, not convergence of an iterative algorithm.
Check both that error and the observable relevant to the calculation.

## Physical entries

[`prelude::TensorMap::data`] is reduced fusion-tree storage, not ordinary
carrier-basis data. Use [`prelude::TensorMap::to_physical_dense`] to expand to
that basis and [`prelude::TensorMap::project_physical_dense`] to project into
the exact schema of another tensor. Basis alignment is application-specific;
TeNeT does not infer a conversion between different symmetry choices.

Projection takes the receiver tensor as the target schema. It therefore makes
the destination runtime, provider, leg order, and sector content explicit.
When two applications use different physical basis orders, permute that
physical data in application code before projection; no generic ordering can
be inferred from the symmetry names alone.

## Runtime and backends

Build one `Runtime` and reuse or clone it for related tensors. Host execution
is the default. The builder selects thread counts, dense backends, and the
`tensor!` optimizer. With the `cuda` feature, `.cuda(device)` attaches a device
to the runtime; tensors remain on Host storage until `to_cuda()` transfers them
explicitly.

```rust
use tenet::prelude::*;

let rt = Runtime::builder().dense_threads(4).build()?;
let rt_for_worker = rt.clone();
let v = GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), 2)])?;
let a = TensorMap::<U1FusionRule, f64>::rand(&rt_for_worker, [&v], [&v])?;
assert!(a.runtime().shares_state_with(&rt));
# Ok::<(), Error>(())
```

The default build uses the Host `cpu-faer` backend. Select a different enabled
dense backend with the builder when a measured workload requires it. CUDA
operations reject unsupported combinations rather than silently moving work
back to the CPU. Build the runtime once at program setup; `clone` shares its
execution configuration for work submitted from another thread.

`tensor!` does not accept hyperedges: a label may appear at most twice.
It does not promote scalar types automatically. Slicing is explicit rather
than selected automatically by the macro.

For source correspondence and background references, see
[`references.md`](references.md).
