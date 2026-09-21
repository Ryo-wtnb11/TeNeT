//! Host pin of the general-contraction oracles the device gate
//! (`typed_cuda_contract.rs`, G2c-1a #1345) compares against.
//!
//! An oracle is only independent evidence for the device once it is shown to
//! agree with the Host on the same fixtures; this file does that without a
//! device, so it runs in ordinary CI:
//!
//! * TensorKit's `blas_contract!` step sequence on Host typed ops, for every
//!   fixture and payload dtype;
//! * the physical-basis dense contraction, for the U(1) and SU(2) fixtures
//!   whose contracted legs do not bend.

mod common;
mod contract_cases;

use contract_cases::{
    assert_close, blas_contract_oracle, dense_oracle, lazy_cases, product_general, su2_bent,
    su2_reordered, u1_lhs_identity, u1_rank_five, u1_reordered, u1_rhs_identity, Case, Payload,
};
use num_complex::{Complex32, Complex64};
use tenet::core::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use tenet::typed::Runtime;

fn check_blas<R, D>(case: Case<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let host = case.host();
    let oracle = blas_contract_oracle(&case);
    assert_eq!(
        host.codomain_rank(),
        oracle.codomain_rank(),
        "{}",
        case.name
    );
    assert_close(host.data(), oracle.data(), case.terms(), case.name);
}

fn every_fixture<D: Payload>() {
    let runtime = Runtime::builder().build().unwrap();
    check_blas(u1_rank_five::<D>(&runtime));
    check_blas(u1_reordered::<D>(&runtime));
    check_blas(su2_reordered::<D>(&runtime));
    check_blas(su2_bent::<D>(&runtime));
    check_blas(product_general::<D>(&runtime));
    check_blas(u1_lhs_identity::<D>(&runtime));
    check_blas(u1_rhs_identity::<D>(&runtime));
    for case in lazy_cases(&u1_rank_five::<D>(&runtime).lhs, "U(1) lazy") {
        check_blas(case);
    }
    for case in lazy_cases(&su2_reordered::<D>(&runtime).lhs, "SU(2) lazy") {
        check_blas(case);
    }
    for case in lazy_cases(&product_general::<D>(&runtime).lhs, "U(1) x SU(2) lazy") {
        check_blas(case);
    }
}

#[test]
fn host_contract_matches_the_tensorkit_blas_contract_sequence_at_every_dtype() {
    every_fixture::<f64>();
    every_fixture::<Complex64>();
    every_fixture::<f32>();
    every_fixture::<Complex32>();
}

fn check_dense<R, D>(case: Case<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + tenet::core::PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    assert!(case.dense, "{}", case.name);
    let (shape, expected) = dense_oracle(&case);
    let actual = case.host().to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape, "{}", case.name);
    assert_close(&actual.data, &expected, case.terms(), case.name);
}

#[test]
fn host_contract_matches_the_physical_basis_contraction() {
    let runtime = Runtime::builder().build().unwrap();
    check_dense(u1_reordered::<f64>(&runtime));
    check_dense(u1_reordered::<Complex64>(&runtime));
    check_dense(su2_reordered::<f64>(&runtime));
    check_dense(su2_reordered::<Complex64>(&runtime));
}
