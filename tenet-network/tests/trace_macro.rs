//! Typed Host partial traces in `tensor!`.
//!
//! These tests keep the macro lowering pinned to the canonical typed
//! `TensorMap::trace_pairs` semantics: quantum dimensions for SU(2), the fZ2
//! supertrace sign, lazy-adjoint trace axes, and compact full-trace storage.

use std::sync::Arc;

use tenet::core::{
    CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep,
    SectorCodec, TypedSectorAdmission, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{Complex64, Error};
use tenet::typed::{GradedSpace, Runtime, SectorSpectrum, TensorMap};
use tenet_network::tensor;

fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "data lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
            "element {index} differs: {a} vs {b}"
        );
    }
}

fn assert_close_c64(lhs: &[Complex64], rhs: &[Complex64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "data lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            (*a - *b).norm() <= tol * (1.0 + a.norm().max(b.norm())),
            "element {index} differs: {a} vs {b}"
        );
    }
}

fn u1_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_shared(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

fn su2_space() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_shared(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

fn fz2_space() -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new_shared(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap()
}

fn eye<R>(runtime: &Runtime, space: &GradedSpace<R>) -> TensorMap<R, f64>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    TensorMap::from_block_fn(runtime, [space], [space], |_, indices| {
        if indices[0] == indices[1] {
            1.0
        } else {
            0.0
        }
    })
    .unwrap()
}

fn assert_partial_trace<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let dual = space.try_dual().unwrap();
    let real = TensorMap::<R, f64>::rand_with_seed(runtime, [space, &dual], [space], seed).unwrap();
    let traced = tensor!([; j] = real[i, i; j]).unwrap();
    let expected = real.trace_pairs(&[(0, 1)]).unwrap();
    assert_close(traced.data(), expected.data(), 1e-12);
    assert_eq!(traced.codomain_rank(), 0);
    assert_eq!(traced.domain_rank(), 1);

    let complex =
        TensorMap::<R, Complex64>::rand_with_seed(runtime, [space, &dual], [space], seed + 1)
            .unwrap();
    let traced = tensor!([; j] = complex[i, i; j]).unwrap();
    let expected = complex.trace_pairs(&[(0, 1)]).unwrap();
    assert_close_c64(traced.data(), expected.data(), 1e-12);
}

/// Macro trace and the typed categorical primitive agree for bosonic,
/// non-Abelian, and fermionic providers in both supported Host dtypes.
#[test]
fn partial_trace_matches_typed_trace_pairs_elementwise() {
    let runtime = Runtime::builder().build().unwrap();
    assert_partial_trace(&runtime, &u1_space(), 201);
    assert_partial_trace(&runtime, &su2_space(), 203);
    assert_partial_trace(&runtime, &fz2_space(), 205);
}

fn assert_identity_oracle<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let dual = space.try_dual().unwrap();
    let tensor =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, &dual], [space], seed).unwrap();
    let identity = eye(runtime, space);
    let traced = tensor!([; j] = tensor[i, i; j]).unwrap();
    let via_identity = tensor!([; j] = tensor[i, k; j] * identity[k; i]).unwrap();
    assert_close(traced.data(), via_identity.data(), 1e-12);
}

/// For twist-free U(1) and SU(2), a partial trace equals contraction with an
/// identity endomorphism.
#[test]
fn partial_trace_matches_identity_contraction_for_twist_free_rules() {
    let runtime = Runtime::builder().build().unwrap();
    assert_identity_oracle(&runtime, &u1_space(), 211);
    assert_identity_oracle(&runtime, &su2_space(), 213);
}

fn assert_identity_quantum_dimension<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let identity = eye(runtime, space);
    let trace = tensor!([] = identity[i; i]).unwrap().scalar().unwrap();
    let dimension = space.dim().unwrap();
    assert!((trace - dimension).abs() <= 1e-12, "{trace} vs {dimension}");
    assert!((identity.tr().unwrap() - trace).abs() <= 1e-12);
}

/// In particular, SU(2) includes its irrep quantum-dimension factors.
#[test]
fn full_trace_of_identity_is_quantum_dimension() {
    let runtime = Runtime::builder().build().unwrap();
    assert_identity_quantum_dimension(&runtime, &u1_space());
    assert_identity_quantum_dimension(&runtime, &su2_space());
}

/// The fZ2 tensor-contraction trace is the supertrace, while `tr()` is the
/// ordinary positive trace. Off-diagonal noise must not contribute.
#[test]
fn fz2_macro_trace_is_supertrace_while_tensor_tr_is_ordinary() {
    let runtime = Runtime::builder().build().unwrap();
    let space = fz2_space();
    let tensor = TensorMap::from_block_fn(&runtime, [&space], [&space], |sectors, indices| {
        if indices[0] != indices[1] {
            return 9.0;
        }
        if *sectors.coupled() == Z2Irrep::EVEN {
            2.0 + indices[0] as f64
        } else {
            5.0 + indices[0] as f64
        }
    })
    .unwrap();

    let trace = tensor!([] = tensor[i; i]).unwrap().scalar().unwrap();
    assert!((trace - (-13.0)).abs() <= 1e-12, "supertrace = {trace}");
    assert!((tensor.tr().unwrap() - 23.0).abs() <= 1e-12);
}

fn assert_trace_and_contract<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let dual = space.try_dual().unwrap();
    let traced_input =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, &dual], [space], seed).unwrap();
    let rhs = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], seed + 1).unwrap();
    let combined = tensor!([; m] = traced_input[i, i; j] * rhs[j; m]).unwrap();
    let manual = traced_input
        .trace_pairs(&[(0, 1)])
        .unwrap()
        .contract(&rhs, &[0], &[0], &[0])
        .unwrap();
    assert_close(combined.data(), manual.data(), 1e-12);
}

#[test]
fn trace_and_contract_combined_matches_manual_two_step() {
    let runtime = Runtime::builder().build().unwrap();
    assert_trace_and_contract(&runtime, &u1_space(), 221);
    assert_trace_and_contract(&runtime, &su2_space(), 223);
    assert_trace_and_contract(&runtime, &fz2_space(), 225);
}

/// `conj(a)` traces the lazy adjoint on its logical axes.
#[test]
fn conj_operand_partial_trace_matches_adjoint_trace() {
    let runtime = Runtime::builder().build().unwrap();
    let space = u1_space();
    let tensor =
        TensorMap::<U1FusionRule, Complex64>::rand_with_seed(&runtime, [&space], [&space], 231)
            .unwrap();
    let traced = tensor!([] = conj(tensor)[i; i]).unwrap().scalar().unwrap();
    let expected = tensor.adjoint().unwrap().tr().unwrap();
    assert!((traced - expected).norm() <= 1e-12);
    assert!((traced - tensor.tr().unwrap().conj()).norm() <= 1e-12);
}

fn assert_two_trace_pairs<R>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let tensor =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], seed).unwrap();
    let via_macro = tensor!([] = tensor[i, j; i, j]).unwrap().scalar().unwrap();
    let via_pairs = tensor
        .trace_pairs(&[(0, 2), (1, 3)])
        .unwrap()
        .scalar()
        .unwrap();
    assert!((via_macro - via_pairs).abs() <= 1e-12 * (1.0 + via_pairs.abs()));
}

#[test]
fn two_trace_pairs_reduce_to_scalar() {
    let runtime = Runtime::builder().build().unwrap();
    assert_two_trace_pairs(&runtime, &u1_space(), 241);
    assert_two_trace_pairs(&runtime, &fz2_space(), 243);
}

/// Compact diagonal tensors stay valid inputs to the macro full-trace arm.
#[test]
fn compact_full_trace_preserves_positive_and_supertrace_oracles() {
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new_shared(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let compact = TensorMap::diagonal(
        &runtime,
        &u1,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![2.0, 3.0, 5.0],
        }],
    )
    .unwrap();
    assert!(compact.diagonal_spectrum().unwrap().is_some());
    let trace = tensor!([] = compact[i; i]).unwrap().scalar().unwrap();
    assert!((trace - 10.0).abs() <= 1e-12);
    assert!((compact.tr().unwrap() - 10.0).abs() <= 1e-12);

    let fz2 = fz2_space();
    let compact = TensorMap::diagonal(
        &runtime,
        &fz2,
        [
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![2.0, 3.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![5.0, 6.0, 7.0],
            },
        ],
    )
    .unwrap();
    assert!(compact.diagonal_spectrum().unwrap().is_some());
    let supertrace = tensor!([] = compact[i; i]).unwrap().scalar().unwrap();
    assert!((supertrace - (-13.0)).abs() <= 1e-12);
    assert!((compact.tr().unwrap() - 23.0).abs() <= 1e-12);
}

#[test]
fn trace_error_paths_stay_typed() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let domain =
        GradedSpace::try_new_shared(Arc::clone(&provider), [(U1Irrep::new(0), 3)]).unwrap();
    let codomain = GradedSpace::try_new_shared(provider, [(U1Irrep::new(0), 4)]).unwrap();
    let non_endomorphism =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&codomain], [&domain], 251)
            .unwrap();
    assert!(matches!(
        non_endomorphism.tr(),
        Err(Error::InvalidArgument(_))
    ));

    let endomorphism =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&domain], [&domain], 252)
            .unwrap();
    assert!(matches!(
        endomorphism.trace_pairs(&[(0, 0)]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        endomorphism.trace_pairs(&[(0, 5)]),
        Err(Error::InvalidArgument(_))
    ));
}
