//! General-axes contraction fixtures and their independent oracles, shared by
//! the Host pin (`typed_contract_host_oracle.rs`, ungated) and the device gate
//! (`typed_cuda_contract.rs`, G2c-1a #1345).
//!
//! Two oracles, neither of which passes through the Host `DynamicTree`
//! compiler whose device replay is under test:
//!
//! * [`blas_contract_oracle`] — TensorKit's `blas_contract!` step sequence
//!   (tensoroperations.jl:383-455 @cfaa073) on Host typed operations: permute
//!   A to `(open; contracted)`, permute B to `(contracted; open)`, canonical
//!   `compose` (TensorKit `mul!`), permute the result by `pAB`. Bosonic
//!   providers only: TensorKit inserts a `twist!` for fermions here.
//! * [`dense_oracle`] — the physical-basis expansion (U(1), SU(2)): an index
//!   sum over the contracted legs followed by an axis permutation, for the
//!   cases where A contracts only domain legs and B only codomain legs, so no
//!   leg bends and the physical array only moves.
//!
//! Fixture entries are dyadic rationals, exact in `f32`. The payload set is
//! the shared `common::Payload` harness, so every including test binary also
//! declares `mod common;`.

#![allow(dead_code)]

use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, PhysicalFusionBasis,
    ProductFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorCodec, U1FusionRule,
    U1Irrep,
};
use tenet::typed::{BlockFusionTrees, GradedSpace, Runtime, TensorMap};

pub use crate::common::Payload;

/// `terms` is the longest sum an entry of the result combines; the bound is
/// `64 * sqrt(terms) * eps(D)` relative to the largest reference magnitude.
pub fn assert_close<D: Payload>(actual: &[D], expected: &[D], terms: usize, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: payload length");
    let scale = expected
        .iter()
        .map(|value| value.magnitude())
        .fold(0.0_f64, f64::max);
    assert!(
        scale > 0.0,
        "{what}: the reference is all zero, so it proves nothing"
    );
    let tolerance = 64.0 * (terms.max(1) as f64).sqrt() * D::EPS * (1.0 + scale);
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(right) <= tolerance,
            "{what} [{}]: element {index} is {left:?}, expected {right:?} (tolerance {tolerance:e})",
            D::NAME
        );
    }
}

/// Dyadic counting fill, distinct per `salt`.
pub fn fill<S, D: Payload>(salt: usize) -> impl FnMut(&BlockFusionTrees<S>, &[usize]) -> D {
    let mut next = salt;
    move |_, _| {
        next += 1;
        let real = ((next * 37) % 17) as f64 / 8.0 - 1.0;
        let imaginary = ((next * 11) % 13) as f64 / 16.0 - 0.375;
        D::entry(real, imaginary)
    }
}

pub fn u1(charges: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        charges
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

pub fn su2() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

pub type U1Su2 = ProductFusionRule<U1FusionRule, SU2FusionRule>;

pub fn u1_su2() -> GradedSpace<U1Su2> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule.product(SU2FusionRule)),
        [
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(U1Irrep::new(-1), SU2Irrep::from_twice_spin(1)),
                2,
            ),
        ],
    )
    .unwrap()
}

/// One contraction: operands, axes, and whether the physical-basis oracle
/// applies (A contracts only domain legs, B only codomain legs).
pub struct Case<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    pub name: &'static str,
    pub lhs: TensorMap<R, D>,
    pub rhs: TensorMap<R, D>,
    pub lhs_axes: Vec<usize>,
    pub rhs_axes: Vec<usize>,
    pub output_axes: Vec<usize>,
    pub dense: bool,
}

impl<R, D> Case<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    /// The longest sum an output entry combines: the contracted physical
    /// dimension (a bound, not the reduced count).
    pub fn terms(&self) -> usize {
        self.lhs_axes.len().max(1) * 64
    }

    pub fn host(&self) -> TensorMap<R, D> {
        self.lhs
            .contract(&self.rhs, &self.lhs_axes, &self.rhs_axes, &self.output_axes)
            .unwrap()
    }
}

fn tensor<R, D>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
    salt: usize,
) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    TensorMap::from_block_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        fill(salt),
    )
    .unwrap()
}

/// Rank 5 against rank 4: a codomain leg against a domain leg and a domain
/// leg against a codomain leg, contracted out of order, output permuted
/// across both sources.
pub fn u1_rank_five<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    Case {
        name: "U(1) rank 5 x 4, mixed sides",
        lhs: tensor(runtime, &[&v, &v, &v], &[&v, &v], 1),
        rhs: tensor(runtime, &[&v, &v], &[&v, &v], 2),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
        dense: false,
    }
}

/// Whole domain against whole codomain in reversed order, codomain output
/// permuted; `v` carries charge 2 that `w` lacks, so a coupled sector of the
/// result has no contributing GEMM.
pub fn u1_reordered<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(0, 1), (1, 2), (2, 1)]);
    let w = u1(&[(-1, 1), (0, 2), (1, 2)]);
    Case {
        name: "U(1) reordered whole-side",
        lhs: tensor(runtime, &[&v, &v], &[&w, &w], 3),
        rhs: tensor(runtime, &[&w, &w], &[&v], 4),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![0, 1],
        output_axes: vec![1, 0, 2],
        dense: true,
    }
}

/// SU(2) recoupling of the reordered whole-side contraction.
pub fn su2_reordered<D: Payload>(runtime: &Runtime) -> Case<SU2FusionRule, D> {
    let s = su2();
    Case {
        name: "SU(2) reordered whole-side",
        lhs: tensor(runtime, &[&s, &s], &[&s, &s], 5),
        rhs: tensor(runtime, &[&s, &s], &[&s], 6),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![0, 1],
        output_axes: vec![1, 0, 2],
        dense: true,
    }
}

/// SU(2): a codomain leg of A against a domain leg of B.
pub fn su2_bent<D: Payload>(runtime: &Runtime) -> Case<SU2FusionRule, D> {
    let s = su2();
    Case {
        name: "SU(2) codomain against domain",
        lhs: tensor(runtime, &[&s, &s], &[&s], 7),
        rhs: tensor(runtime, &[&s], &[&s, &s], 8),
        lhs_axes: vec![0],
        rhs_axes: vec![2],
        output_axes: vec![3, 0, 2, 1],
        dense: false,
    }
}

/// Bosonic product provider, two pairs across sides.
pub fn product_general<D: Payload>(runtime: &Runtime) -> Case<U1Su2, D> {
    let p = u1_su2();
    Case {
        name: "U(1) x SU(2) mixed sides",
        lhs: tensor(runtime, &[&p, &p], &[&p], 9),
        rhs: tensor(runtime, &[&p, &p], &[&p], 10),
        lhs_axes: vec![2, 1],
        rhs_axes: vec![0, 2],
        output_axes: vec![1, 0],
        dense: false,
    }
}

/// A already `(open; contracted)`: its source transform is the identity and
/// it can be read in place; B needs one.
pub fn u1_lhs_identity<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    let w = u1(&[(0, 2), (1, 1)]);
    Case {
        name: "U(1) identity lhs transform",
        lhs: tensor(runtime, &[&v, &v], &[&w], 11),
        rhs: tensor(runtime, &[&v, &w], &[&v], 12),
        lhs_axes: vec![2],
        rhs_axes: vec![1],
        output_axes: vec![0, 1, 2, 3],
        dense: false,
    }
}

/// B already `(contracted; open)`: its source transform is the identity.
pub fn u1_rhs_identity<D: Payload>(runtime: &Runtime) -> Case<U1FusionRule, D> {
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    let w = u1(&[(0, 2), (1, 1)]);
    let w_dual = w.try_dual().unwrap();
    Case {
        name: "U(1) identity rhs transform",
        lhs: tensor(runtime, &[&v, &w], &[&v], 13),
        rhs: tensor(runtime, &[&w_dual], &[&v, &v], 14),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1, 2, 3],
        dense: false,
    }
}

/// `L^H` contracted with `L`, and `L` with `L^H`, on a subset of legs in a
/// non-canonical order. Every leg of `L^H` is the dual of the matching leg
/// of `L`, so the pairing is valid for any provider; `L^H` stays a lazy
/// adjoint over `L`'s payload.
pub fn lazy_cases<R, D>(lhs: &TensorMap<R, D>, name: &'static str) -> [Case<R, D>; 2]
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let nout = lhs.codomain_rank();
    let nin = lhs.domain_rank();
    assert!(nout >= 2 && nin >= 1, "{name}: fixture too small");
    // `L^H` axis `nin + j` is `L` codomain leg `j`; `L^H` axis `i < nin` is
    // `L` domain leg `nout + i`.
    let adjoint_axes = vec![nin + 1, 0];
    let direct_axes = vec![1, nout];
    let output: Vec<usize> = (0..lhs.rank() * 2 - 4).rev().collect();
    let adjoint = lhs.adjoint().unwrap();
    [
        Case {
            name,
            lhs: adjoint.clone(),
            rhs: lhs.clone(),
            lhs_axes: adjoint_axes.clone(),
            rhs_axes: direct_axes.clone(),
            output_axes: output.clone(),
            dense: false,
        },
        Case {
            name,
            lhs: lhs.clone(),
            rhs: adjoint,
            lhs_axes: direct_axes,
            rhs_axes: adjoint_axes,
            output_axes: output,
            dense: false,
        },
    ]
}

/// The class the Host resolves to its dense `Structure` route, which the
/// device replaces by the prelowered `DynamicTree` artifact: a lazy adjoint
/// over SU(2) (every sector self-dual), contracted in core-form source order
/// (lhs whole domain, rhs whole codomain, in order), with a non-identity
/// output. Lazy lhs, then lazy rhs; multi-block, degeneracy > 1. That the
/// Host really takes `Structure` for this geometry is pinned in
/// `tenet-tensors/src/contract/storage_contract_tests.rs`.
pub fn su2_structure_cases<D: Payload>(runtime: &Runtime) -> [Case<SU2FusionRule, D>; 2] {
    let s = su2();
    let x: TensorMap<_, D> = tensor(runtime, &[&s, &s], &[&s, &s], 15);
    let y: TensorMap<_, D> = tensor(runtime, &[&s], &[&s, &s], 17);
    [
        Case {
            name: "SU(2) Structure class, lazy lhs",
            lhs: x.adjoint().unwrap(),
            rhs: tensor(runtime, &[&s, &s], &[&s], 16),
            lhs_axes: vec![2, 3],
            rhs_axes: vec![0, 1],
            output_axes: vec![2, 0, 1],
            dense: false,
        },
        Case {
            name: "SU(2) Structure class, lazy rhs",
            lhs: tensor(runtime, &[&s], &[&s, &s], 18),
            rhs: y.adjoint().unwrap(),
            lhs_axes: vec![1, 2],
            rhs_axes: vec![0, 1],
            output_axes: vec![1, 0],
            dense: false,
        },
    ]
}

/// TensorKit `blas_contract!` on Host typed operations.
pub fn blas_contract_oracle<R, D>(case: &Case<R, D>) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let open = |rank: usize, contracted: &[usize]| -> Vec<usize> {
        (0..rank)
            .filter(|axis| !contracted.contains(axis))
            .collect()
    };
    let lhs_open = open(case.lhs.rank(), &case.lhs_axes);
    let rhs_open = open(case.rhs.rank(), &case.rhs_axes);
    let a = case.lhs.permute(&lhs_open, &case.lhs_axes).unwrap();
    let b = case.rhs.permute(&case.rhs_axes, &rhs_open).unwrap();
    let c = a.compose(&b).unwrap();
    let split = lhs_open.len();
    c.permute(&case.output_axes[..split], &case.output_axes[split..])
        .unwrap()
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut running = 1;
    for &dim in shape {
        strides.push(running);
        running *= dim;
    }
    strides
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    let total: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        visit(&index);
        for (axis, value) in index.iter_mut().enumerate() {
            *value += 1;
            if *value < shape[axis] {
                break;
            }
            *value = 0;
        }
    }
}

/// Physical-basis contraction: `C[out] = sum_k A[.., k, ..] B[.., k, ..]`,
/// output axes permuted by `output_axes`, all arrays column-major.
pub fn dense_oracle<R, D>(case: &Case<R, D>) -> (Vec<usize>, Vec<D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    let a = case.lhs.to_physical_dense().unwrap();
    let b = case.rhs.to_physical_dense().unwrap();
    let a_strides = column_major_strides(&a.shape);
    let b_strides = column_major_strides(&b.shape);
    let a_open: Vec<usize> = (0..a.shape.len())
        .filter(|axis| !case.lhs_axes.contains(axis))
        .collect();
    let b_open: Vec<usize> = (0..b.shape.len())
        .filter(|axis| !case.rhs_axes.contains(axis))
        .collect();
    let joint_shape: Vec<usize> = a_open
        .iter()
        .map(|&axis| a.shape[axis])
        .chain(b_open.iter().map(|&axis| b.shape[axis]))
        .collect();
    let contracted_shape: Vec<usize> = case.lhs_axes.iter().map(|&axis| a.shape[axis]).collect();
    let output_shape: Vec<usize> = case
        .output_axes
        .iter()
        .map(|&axis| joint_shape[axis])
        .collect();
    let output_strides = column_major_strides(&output_shape);
    let mut output = vec![D::entry(0.0, 0.0); output_shape.iter().product()];
    for_each_index(&joint_shape, |joint| {
        let mut total = D::entry(0.0, 0.0);
        for_each_index(&contracted_shape, |k| {
            let mut a_offset = 0;
            for (position, &axis) in a_open.iter().enumerate() {
                a_offset += joint[position] * a_strides[axis];
            }
            let mut b_offset = 0;
            for (position, &axis) in b_open.iter().enumerate() {
                b_offset += joint[a_open.len() + position] * b_strides[axis];
            }
            for (pair, (&a_axis, &b_axis)) in case.lhs_axes.iter().zip(&case.rhs_axes).enumerate() {
                a_offset += k[pair] * a_strides[a_axis];
                b_offset += k[pair] * b_strides[b_axis];
            }
            total = total + a.data[a_offset] * b.data[b_offset];
        });
        let mut offset = 0;
        for (position, &axis) in case.output_axes.iter().enumerate() {
            offset += joint[axis] * output_strides[position];
        }
        output[offset] = total;
    });
    (output_shape, output)
}
