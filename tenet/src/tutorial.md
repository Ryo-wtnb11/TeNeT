# Tutorial

TeNeT provides small operations on symmetric tensor maps. This page shows how
to construct a map, contract its legs, and factorize it. For a complete program
with index notation, run:

```sh
cargo run -p tenet-network --example quickstart
```

The package is `tenet-rs`, imported in Rust as `tenet`. The separate
`tenet-network` package provides `tensor!` and network planning. The Rust
blocks below are doctests; they use the built-in U(1) symmetry, but the tensor
operations use the same API for other supported providers.

## Quick start

A [`typed::GradedSpace`] describes one leg by its symmetry sectors and the
number of states in each sector. A [`typed::TensorMap`] is a map
`codomain <- domain`; TeNeT stores only its symmetry-allowed reduced blocks.
A [`typed::Runtime`] owns execution resources and is shared by related maps.

```rust
use std::sync::Arc;
use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};

let runtime = Runtime::builder().build()?;
let spin = GradedSpace::try_new(
    Arc::new(U1FusionRule),
    [(U1Irrep::new(-1), 1), (U1Irrep::new(1), 1)],
)?;
let sz: TensorMap<U1FusionRule, f64> = TensorMap::from_subblock_fn(
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
assert_eq!((sz.codomain_rank(), sz.domain_rank()), (1, 1));
# Ok::<(), Error>(())
```

`from_subblock_fn` visits each allowed reduced entry. For common initial
values, use `zeros`, `isomorphism`, or `rand_with_seed`. The scalar is the
second type parameter, such as `f64` or [`typed::Complex64`]; use `convert`
when an operation needs a different scalar type. There is no implicit
conversion in a contraction.

`GradedSpace::try_new` creates a nondual leg. Use `try_dual` when the operation
requires a dual leg. In a contraction, a codomain and domain leg built from
the same space pair directly; two legs on the same side require the appropriate
dual orientation. [`mathematics`] explains the
basis, duality, and signs.

## Arrange and contract legs

`compose` composes maps. `contract` uses a [`typed::ContractSpec`] to name the
contracted axes and the codomain/domain order of the open axes. Input axes are
zero-based, with codomain axes before domain axes. Number the remaining open
axes separately: first the left operand's open axes in input order, then the
right operand's. `codomain` and `domain` list each of those open-axis numbers
exactly once. The output split is part of the operation; it need not be a
separate permutation afterward.

```rust
use std::sync::Arc;
use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{ContractSpec, Error, GradedSpace, Runtime, TensorMap};

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new(
    Arc::new(U1FusionRule),
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
)?;
let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 4)?;
let b = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 5)?;
let c = a.compose(&b)?;
let spec = ContractSpec { lhs: &[2, 3], rhs: &[0, 1], codomain: &[0, 1], domain: &[2, 3] };
let same = a.contract(&b, &spec)?;
assert_eq!(c.dense_data()?, same.dense_data()?);

// Put open axes [0, 2, 3] in the codomain and [1] in the domain.
let mixed = a.contract(&b, &ContractSpec { codomain: &[0, 2, 3], domain: &[1], ..spec })?;
let reordered = c.permute(&[0, 2, 3], &[1])?;
assert!(mixed.axpby(1.0, &reordered, -1.0)?.norm(2.0)? < 1e-12);
# Ok::<(), Error>(())
```

Use `permute` to choose leg order and codomain/domain split; use `repartition`
when only the split changes. `adjoint` reverses the map orientation and
conjugates values; `braid` also accounts for the provider's braiding. These
operations have different meanings even when a real U(1) example gives the
same values. [`mathematics`] gives their precise
definitions.

For a network, `tenet_network::tensor!` names legs instead of listing axis
numbers. Its [compiled quickstart](https://github.com/Ryo-wtnb11/TeNeT/blob/main/tenet-network/examples/quickstart.rs)
shows the syntax. The caller still decides the algorithm and may choose or
reuse a network plan; TeNeT handles the symmetry-aware work inside each
operation.

## Factorize and truncate

Factorizations take `(rows, cols)`: the source axes that form each side of the
matrix. TeNeT handles the required leg transformation. A compact SVD returns
`u`, `s`, and `vh` on a new bond space. On Host, `s` has compact diagonal
storage for both multiplicity-free and Checked Generic providers.

```rust
use std::sync::Arc;
use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{Error, GradedSpace, Runtime, Svd, TensorMap};

let rt = Runtime::builder().build()?;
let v = GradedSpace::try_new(
    Arc::new(U1FusionRule),
    [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
)?;
let t = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 10)?;
let Svd { u, s, vh } = t.svd_compact(&[0, 1], &[2, 3])?;
let reconstructed = u.compose(&s)?.compose(&vh)?;
let error = reconstructed.axpby(1.0, &t, -1.0)?.norm(2.0)?;
assert!(error < 1e-8);
# Ok::<(), Error>(())
```

Truncation is a separate choice made by the algorithm. The
[U(1) iTEBD example](https://github.com/Ryo-wtnb11/TeNeT/blob/main/tenet-network/examples/itebd_heisenberg.rs)
shows spectrum selection and bond restriction in context. `qr_compact`,
`eigh_full`, and the truncation methods have their own contracts in rustdoc.

## Storage and execution

`dense_data` and `blocks` expose **reduced** storage, not a dense array in the
physical carrier basis. Use `to_physical_dense` when that full array is needed.
The latter can be much larger than the reduced representation.

Host is the default execution placement. With the `cuda` feature, configure a
device on `Runtime::builder()` and explicitly move tensor payloads with
`to_cuda()` and `to_host()`. TeNeT does not silently transfer an unsupported
operation to Host. Supported operations depend on the provider, scalar type,
and placement; see the [backend policy](https://github.com/Ryo-wtnb11/TeNeT/blob/main/docs/backend_policy.md) and each
method's rustdoc for the current capability boundary.

For more detail, use the [documentation index](https://github.com/Ryo-wtnb11/TeNeT/blob/main/docs/README.md) for
mathematical conventions, provider implementation, complete examples, and
development guidance.
