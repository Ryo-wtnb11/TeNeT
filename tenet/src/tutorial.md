# Tutorial

This tutorial shows the current public facade. TeNeT is split into:

- [`core`]: tensor spaces, tensor storage, sectors, and fusion-tree spaces.
- [`operations`]: TensorOperations-style tensoradd, trace, contraction, and
  fusion-tree transforms.
- [`dense`]: dense block execution boundary.
- [`matrixalgebra`]: matrix decompositions and linear algebra helpers.

The examples below use the facade crate, so user code imports from `tenet`.

## First Mental Model

The main object is [`core::TensorMap`]. It is a tensor, but TeNeT treats it as
a linear map from **domain** legs to **codomain** legs:

`TensorMap<T, NOUT, NIN>`, where:

- `T` is the scalar type.
- `NOUT` is the number of codomain legs.
- `NIN` is the number of domain legs.

For a matrix, `NOUT = 1` and `NIN = 1`. For a rank-2 dense tensor written as
`A[i, j]`, it is often convenient to use `NOUT = 2` and `NIN = 0`: both axes
are just output axes. The split matters for categorical operations such as
adjoint, trace, and symmetric tensor contraction.

A [`core::TensorMapSpace`] describes the ordinary dense dimensions. A
[`core::TensorMap`] owns the data living in that space.

Storage is column-major, because the hot path eventually calls BLAS-like dense
matrix kernels. For example, the matrix

```text
[[1, 2],
 [3, 4]]
```

is stored as:

```rust
let storage = vec![1.0, 3.0, 2.0, 4.0];
```

Axis numbers in the Rust API are zero-based. They are ordered as all codomain
axes first, followed by all domain axes. Thus a `TensorMap<T, 1, 1>` matrix has
axis `0` for its codomain leg and axis `1` for its domain leg.

## If You Are Coming From TensorKit

TensorKit/Julia and TeNeT/Rust expose different amounts of machinery.
TensorKit can rely on Julia's multiple dispatch and macros to hide many
choices. TeNeT keeps the same mathematical objects, but Rust asks us to make
some choices explicit so they can be cached, checked, and sent to CPU/GPU
backends without hidden allocation.

The practical mapping is:

| TensorKit idea | TeNeT object |
| --- | --- |
| `TensorMap` | [`core::TensorMap`] |
| ordinary vector spaces | [`core::TensorMapSpace`] |
| sector labels and dual legs | [`core::SectorLeg`] |
| product of external legs | [`core::FusionProductSpace`] |
| hom space / fusion-tree basis | [`core::FusionTreeHomSpace`] |
| block-sparse tensor space | [`core::FusionTensorMapSpace`] |
| `@tensor` lowering | [`operations`] contraction/trace/tensoradd functions |

The main ergonomic difference is construction. In TensorKit, much of the space
and block structure is inferred from the domain and codomain objects. In TeNeT,
the current explicit path is:

```text
legs -> fusion product spaces -> hom space -> tensor space -> tensor data
```

That is more verbose, but it gives a stable place to cache TensorKit-like
fusion block structures, tree transformers, and dense backend execution
metadata. User-facing helpers should wrap this path; the low-level objects
still exist because they are the performance boundary.

Two details are easy to forget when translating Julia examples:

- Julia index notation is one-based in user code; TeNeT axis lists are
  zero-based.
- TeNeT storage examples show flat column-major data. TensorKit users often do
  not see this because block storage is hidden behind array-like syntax.

## Which Constructor Should I Use?

If there is no symmetry, start with:

1. [`core::TensorMapSpace::from_dims`] for the shape.
2. [`core::TensorMap::from_vec`] or [`core::TensorMap::filled`] for the data.

If there is symmetry, start with:

1. [`core::SectorLeg::new`] for each external leg.
2. [`core::FusionProductSpace::new`] for the list of codomain/domain legs.
3. [`core::FusionTreeHomSpace::new`] for the categorical tensor space.
4. [`core::FusionTensorMapSpace::from_degeneracy_shapes`] for the block layout.
5. [`core::TensorMap::from_vec_with_fusion_space`] for the data.

Use the lower-level constructors only when you already have the corresponding
internal object:

- Use [`core::FusionTensorMapSpace::new`] when a [`core::BlockStructure`] has
  already been prepared.
- Use [`core::TensorMap::from_storage_with_structure`] or
  [`core::TensorMap::from_storage_with_fusion_space`] when the data is not a
  plain `Vec<T>`.
- Use [`core::TensorMap::from_block_fn_with_fusion_space`] when the natural way
  to create the tensor is block-by-block rather than flat-vector order.

## Construction Quick Links

Dense tensors:

- Use [`core::TensorMapSpace::from_dims`] to build a dense codomain/domain
  product space from plain dimensions.
- Use [`core::TensorMap::from_vec`] to build a dense tensor with the default
  trivial block structure.
- Use [`core::TensorMap::filled`] to build a dense tensor filled with one value.
- Use [`core::TensorMap::from_vec_with_structure`] to build a dense tensor with
  an explicit [`core::BlockStructure`].
- Use [`core::TensorMap::from_storage_with_structure`] for custom host storage
  instead of `Vec<T>`.

Symmetric tensors:

- Use [`core::FusionProductSpace::new`] and [`core::SectorLeg::new`] to build
  the sector content of each external leg.
- Use [`core::FusionTreeHomSpace::new`] to combine codomain/domain fusion
  product spaces.
- Use [`core::FusionTreeHomSpace::from_sectors`] and
  [`core::FusionTreeHomSpace::from_sector_ids`] as shorter constructors when
  each leg has exactly one sector.
- Use [`core::FusionTensorMapSpace::from_degeneracy_shapes`] as the default
  symmetric tensor space constructor. This uses the coupled-sector matrix
  layout.
- Use [`core::FusionTensorMapSpace::from_degeneracy_shapes_coupled`] when the
  coupled layout should be spelled explicitly.
- Use [`core::FusionTensorMapSpace::new`] to build a symmetric tensor space
  from an already prepared [`core::BlockStructure`].
- Use [`core::TensorMap::from_vec_with_fusion_space`] to attach data to a
  symmetric tensor space.
- Use [`core::TensorMap::from_block_fn_with_fusion_space`] to fill a symmetric
  tensor by iterating block keys and block-local indices. This is useful when
  the physical meaning is block-local rather than flat-storage-local.
- Use [`core::TensorMap::from_storage_with_fusion_space`] to attach custom
  storage to a symmetric tensor space.

## Dense tensors

A dense matrix is a [`core::TensorMap`] with one codomain leg and one domain
leg. Storage is column-major, matching the dense block convention used by
BLAS-like backends.

The example below computes `C = A * B`. In TeNeT axis notation, this contracts
`A`'s domain axis `1` with `B`'s codomain axis `0`.

```
use tenet::core::{TensorMap, TensorMapSpace};
use tenet::operations::{tensorcontract_into, TensorContractAxisSpec};

let space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();

// Column-major storage for [[1, 2], [3, 4]].
let a = TensorMap::<f64, 1, 1>::from_vec(
    vec![1.0, 3.0, 2.0, 4.0],
    space.clone(),
)
.unwrap();
// Column-major storage for [[5, 6], [7, 8]].
let b = TensorMap::<f64, 1, 1>::from_vec(
    vec![5.0, 7.0, 6.0, 8.0],
    space.clone(),
)
.unwrap();
let mut c = TensorMap::<f64, 1, 1>::from_vec(vec![0.0; 4], space).unwrap();

// Contract A's domain axis 1 with B's codomain axis 0.
tensorcontract_into(
    &mut c,
    &a,
    &b,
    TensorContractAxisSpec::canonical(&[1], &[0]),
    1.0,
    0.0,
)
.unwrap();

// [[19, 22], [43, 50]] in column-major storage.
assert_eq!(c.data(), &[19.0, 43.0, 22.0, 50.0]);
```

## A Small Tensor Network

For a slightly larger dense tensor network, contract a triangle

```text
sum_{i,j,k} A[i,j] B[j,k] C[k,i]
```

This is the same contraction as `ij,jk,ki->`, written as two explicit binary
contractions. The first contraction builds the open intermediate `AB[i,k]`;
the second closes the remaining two legs against `C[k,i]`.

This example is intentionally written without an einsum macro so the axis
semantics are visible. A higher-level contraction frontend can choose the same
intermediate automatically, but the low-level operation is still a sequence of
binary contractions.

```
use tenet::core::{TensorMap, TensorMapSpace};
use tenet::operations::{tensorcontract_into, TensorContractAxisSpec};

let matrix = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();

// A[i, j] = [[1, 2], [3, 4]]
let a = TensorMap::<f64, 2, 0>::from_vec(
    vec![1.0, 3.0, 2.0, 4.0],
    matrix.clone(),
)
.unwrap();
// B[j, k] = [[5, 6], [7, 8]]
let b = TensorMap::<f64, 2, 0>::from_vec(
    vec![5.0, 7.0, 6.0, 8.0],
    matrix.clone(),
)
.unwrap();
// C[k, i] = [[2, 1], [0, 3]]
let c = TensorMap::<f64, 2, 0>::from_vec(
    vec![2.0, 0.0, 1.0, 3.0],
    matrix,
)
.unwrap();

let mut ab = TensorMap::<f64, 2, 0>::from_vec(
    vec![0.0; 4],
    TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap(),
)
.unwrap();

// Contract A's j leg (axis 1) with B's j leg (axis 0).
// The canonical output order is A's open legs followed by B's open legs:
// AB has axes [i, k].
tensorcontract_into(
    &mut ab,
    &a,
    &b,
    TensorContractAxisSpec::canonical(&[1], &[0]),
    1.0,
    0.0,
)
.unwrap();
assert_eq!(ab.data(), &[19.0, 43.0, 22.0, 50.0]);

let mut scalar = TensorMap::<f64, 0, 0>::from_vec(
    vec![0.0],
    TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
)
.unwrap();

// Close i with C's axis 1 and k with C's axis 0.
tensorcontract_into(
    &mut scalar,
    &ab,
    &c,
    TensorContractAxisSpec::canonical(&[0, 1], &[1, 0]),
    1.0,
    0.0,
)
.unwrap();

assert_eq!(scalar.data(), &[231.0]);
```

## Symmetric tensors

A symmetric tensor additionally carries a fusion-tree hom space and a block
structure. The default constructor
[`core::FusionTensorMapSpace::from_degeneracy_shapes`] uses the
TensorKit-equivalent coupled-sector matrix layout. That is the normal
hot-path layout: canonical matrix views can go to contraction and
decomposition routines without first packing every subblock.

The example uses `Z2`, whose sectors are even and odd. The tensor has dense
dimension `2 x 2`, but only charge-conserving blocks are stored. That is why the
data vector has length `2` rather than `4`: one scalar for the even block and
one scalar for the odd block.

The vocabulary is:

- A sector is a charge label, such as `Z2` even/odd, a `U1` charge, or an
  `SU2` irrep.
- A [`core::SectorLeg`] is the set of sectors allowed on one external tensor
  leg, plus its dual-orientation flag.
- A [`core::FusionProductSpace`] is an ordered list of sector legs.
- A [`core::FusionTreeHomSpace`] is the codomain/domain fusion-tree space.
- Degeneracy shapes are the dense multiplicity dimensions attached to each
  allowed fusion-tree block.

```
use tenet::core::{
    FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SectorLeg,
    TensorMap, TensorMapSpace, Z2FusionRule, Z2Irrep,
};
use tenet::operations::{tensorcontract_fusion_into, TensorContractAxisSpec};

let rule = Z2FusionRule;
let leg = || SectorLeg::new([Z2Irrep::EVEN, Z2Irrep::ODD], false);
let space = || {
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap()
};

let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
    vec![2.0, 3.0],
    space(),
)
.unwrap();
let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
    vec![5.0, 7.0],
    space(),
)
.unwrap();
let mut dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
    vec![0.0, 0.0],
    space(),
)
.unwrap();

tensorcontract_fusion_into(
    &rule,
    &mut dst,
    &lhs,
    &rhs,
    TensorContractAxisSpec::canonical(&[1], &[0]),
    1.0,
    0.0,
)
.unwrap();

assert_eq!(dst.data(), &[10.0, 21.0]);
```

## Execution model

TeNeT separates three layers:

1. Categorical structure: sector labels, dual flags, fusion trees, and
   recoupling coefficients.
2. Block layout: dense block shape, strides, offsets, and coupled-sector
   matrix layout.
3. Execution: strided tensoradd, pack/GEMM/scatter when a tree transform
   needs it, and dense backend calls.

The default symmetric layout is coupled-sector matrix layout. Pack/scatter
still exists, but it is a temporary replay mechanism for noncanonical
tree-transform paths, not the default storage layout for ordinary tensors.

For repeated operations, use the explicit context/cache APIs in
[`operations`] instead of rebuilding structures in a tight loop.
