//! Host gates of `PreparedEighFull` (#1499, leaf L3 of #1287).
//!
//! Per member the handle must equal Host eager `eigh_full` (gauge-fixed, so
//! the eigenvectors themselves are compared) and the dense Jacobi oracle of
//! `prepared/eigh.rs`, over U(1), SU(2) and fZ2xU(1) with several coupled
//! sectors and degeneracies above one. A batch fails as a whole and names
//! every failing member.

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use tenet::core::{
    CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep, SectorCodec,
};
use tenet::prelude::Error;
use tenet::typed::{
    BatchError, GradedSpace, MemberFault, PreparedEighFull, Runtime, SignatureField,
    StackedTensorMap, TensorMap,
};

use prepared::eigh::{
    check_member, degenerate_entry, has_degenerate_group, has_plus_minus_tie, hermitian_members,
    plus_minus_entry, single_leg,
};
use prepared::{assert_close, fz2u1_legs, members, su2_legs, u1_legs};

const MEMBER_COUNTS: [usize; 3] = [1, 2, 7];

/// Runs the handle on `inputs` and checks every member against Host eager
/// and the oracle. At `B = 1` the factors must equal eager bit for bit
/// (IEEE `==`); at any `B` the Host handle runs the eager code per member,
/// so its gauge-fixed `v` equals eager's within the numerics rule.
fn check_batch<R>(label: &str, runtime: &Runtime, inputs: &[TensorMap<R, f64>])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0), runtime);
    let stack = StackedTensorMap::pack(inputs).unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    let output = handle.execute(&stack).unwrap();
    assert_eq!(
        (output.d.len(), output.v.len()),
        (inputs.len(), inputs.len())
    );
    assert_eq!(output.spectra.len(), inputs.len());
    for (member, input) in inputs.iter().enumerate() {
        let what = format!("{label} member {member}/{}", inputs.len());
        let (eager_d, eager_v) = input.eigh_full().unwrap();
        let d = output.d.member(member).unwrap();
        let v = output.v.member(member).unwrap();
        assert!(
            d.structure_signature() == eager_d.structure_signature(),
            "{what}: d space"
        );
        assert!(
            v.structure_signature() == eager_v.structure_signature(),
            "{what}: v space"
        );
        if inputs.len() == 1 {
            assert!(d.data() == eager_d.data(), "{what}: d bit-identical");
            assert!(v.data() == eager_v.data(), "{what}: v bit-identical");
        }
        let terms = input.data().len();
        assert_close(
            v.data(),
            eager_v.data(),
            terms,
            &format!("{what}: gauge-fixed v"),
        );
        assert_close(d.data(), eager_d.data(), terms, &format!("{what}: d"));
        assert!(
            output.spectra[member] == input.eigh_vals().unwrap(),
            "{what}: spectra"
        );
        check_member(&what, input, &d, &v, (&eager_d, &eager_v), f64::EPSILON);
    }
}

fn equivalence<R>(label: &str, leg: GradedSpace<R>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
{
    let runtime = Runtime::builder().build().unwrap();
    for count in MEMBER_COUNTS {
        let inputs = hermitian_members(&runtime, &[&leg, &leg], count, 3);
        assert!(inputs[0].block_count() > 3, "{label}: several blocks");
        check_batch(&format!("{label} B={count}"), &runtime, &inputs);
    }
}

#[test]
fn host_members_equal_host_eager_and_the_dense_oracle() {
    equivalence("u1", u1_legs().0);
    equivalence("su2", su2_legs().0);
    equivalence("fz2u1", fz2u1_legs().0);
}

#[test]
fn plus_minus_lambda_and_degenerate_groups_compare_by_value() {
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_legs().0;
    let j = SU2Irrep::from_twice_spin;
    let su2 = GradedSpace::try_new(SU2FusionRule, [(j(0), 3), (j(1), 4)]).unwrap();
    for count in [1, 5] {
        let pm = single_leg(&runtime, &leg, count, plus_minus_entry);
        let (d, _) = pm[0].eigh_full().unwrap();
        assert!(has_plus_minus_tie(&d), "the ±λ fixture has |λ| ties");
        check_batch(&format!("u1 ±λ B={count}"), &runtime, &pm);
        let degenerate = single_leg(&runtime, &su2, count, degenerate_entry);
        let (d, _) = degenerate[0].eigh_full().unwrap();
        assert!(
            has_degenerate_group(&d),
            "the degenerate fixture has a group"
        );
        check_batch(&format!("su2 degenerate B={count}"), &runtime, &degenerate);
    }
}

/// Eager's verdict: `Some(fault)` when eager `eigh_full` rejects the member.
fn eager_fault<R>(input: &TensorMap<R, f64>) -> Option<MemberFault>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let error = input.eigh_full().err()?;
    let message = format!("{error:?}");
    Some(if message.contains("Hermitian") {
        MemberFault::NotHermitian
    } else if message.contains("finite") {
        MemberFault::NonFiniteEigenvalue
    } else {
        panic!("unexpected eager error {message}")
    })
}

fn rejected_members(result: Result<(), BatchError>) -> Vec<(usize, MemberFault)> {
    match result {
        Ok(()) => Vec::new(),
        Err(BatchError::MemberRejected { members }) => members,
        Err(other) => panic!("unexpected batch error {other:?}"),
    }
}

/// Members spanning ±1000 binary orders of magnitude, Hermitian and not.
pub fn mixed_scale_members<R>(runtime: &Runtime, leg: &GradedSpace<R>) -> Vec<TensorMap<R, f64>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let h = hermitian_members(runtime, &[leg, leg], 1, 5).remove(0);
    let x = members::<R, f64>(runtime, &[leg, leg], &[leg, leg], 1, 9).remove(0);
    // `2^-30` relative skew rejects (`> 64 eps`); `2^-60` is admitted.
    let rejected = h.add(&x, 1.0, 2f64.powi(-30)).unwrap();
    let admitted = h.add(&x, 1.0, 2f64.powi(-60)).unwrap();
    let mut out = Vec::new();
    for exponent in [0, -1000, 1000, -1060] {
        let s = 2f64.powi(exponent);
        out.push(h.scale(s));
        out.push(rejected.scale(s));
        out.push(admitted.scale(s));
    }
    out
}

#[test]
fn mixed_scale_admission_verdicts_equal_eager_per_member() {
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let inputs = mixed_scale_members(&runtime, &leg);
    let expected: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter_map(|(member, input)| eager_fault(input).map(|fault| (member, fault)))
        .collect();
    assert!(
        expected.len() >= 3,
        "the fixture rejects some members: {expected:?}"
    );
    assert!(expected.len() < inputs.len(), "and admits others");
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    let got = rejected_members(handle.execute(&stack).map(|_| ()));
    assert_eq!(got, expected);
}

#[test]
fn every_non_hermitian_or_non_finite_member_is_named() {
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = su2_legs();
    let mut inputs = hermitian_members(&runtime, &[&leg, &leg], 6, 1);
    let x = members::<_, f64>(&runtime, &[&leg, &leg], &[&leg, &leg], 1, 4).remove(0);
    inputs[1] = x.clone();
    inputs[4] = x;
    inputs[5] = inputs[5].scale(f64::NAN);
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    let got = rejected_members(handle.execute(&stack).map(|_| ()));
    let fault = MemberFault::NotHermitian;
    assert_eq!(got, vec![(1, fault), (4, fault), (5, fault)]);
    assert!(
        handle.take_output().is_none(),
        "no output after a rejected batch"
    );
}

#[test]
fn a_non_finite_eigenvalue_rejects_its_members() {
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    // `M [[1, 1], [1, 1]]`-like sector matrices with `M` near the largest
    // double: finite and Hermitian, but `2M` overflows.
    let huge = f64::MAX / 1.5;
    let mut inputs = single_leg(&runtime, &leg, 4, |_, _, _| 1.0);
    for member in [0, 2] {
        inputs[member] = inputs[member].scale(huge);
    }
    let expected: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter_map(|(member, input)| eager_fault(input).map(|fault| (member, fault)))
        .collect();
    assert_eq!(
        expected,
        vec![
            (0, MemberFault::NonFiniteEigenvalue),
            (2, MemberFault::NonFiniteEigenvalue)
        ],
        "eager rejects the overflowing members"
    );
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    assert_eq!(
        rejected_members(handle.execute(&stack).map(|_| ())),
        expected
    );
}

#[test]
fn complex_payloads_and_other_signatures_are_typed_errors() {
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let complex = members::<_, Complex64>(&runtime, &[&v], &[&v], 2, 1);
    let stack = StackedTensorMap::pack(&complex).unwrap();
    assert!(matches!(
        PreparedEighFull::new(&stack),
        Err(Error::Operation(error)) if format!("{error:?}").contains("real payloads")
    ));

    let inputs = hermitian_members(&runtime, &[&v], 2, 1);
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    let other = StackedTensorMap::pack(&hermitian_members(&runtime, &[&w], 2, 1)).unwrap();
    assert!(matches!(
        handle.execute(&other).map(|_| ()),
        Err(BatchError::Operation(Error::BatchSignatureMismatch {
            member: None,
            field: SignatureField::HomSpace
        }))
    ));
    let rectangular = members::<_, f64>(&runtime, &[&v], &[&w], 1, 1);
    assert!(PreparedEighFull::new(&StackedTensorMap::pack(&rectangular).unwrap()).is_err());
}

#[test]
fn warm_calls_reuse_the_outputs() {
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = fz2u1_legs();
    let inputs = hermitian_members(&runtime, &[&leg, &leg], 3, 2);
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let mut handle = PreparedEighFull::new(&stack).unwrap();
    let first = handle.execute(&stack).unwrap().v.member(2).unwrap();
    let retained = handle.retained_bytes();
    assert!(retained > 0);
    let second = handle.execute(&stack).unwrap().v.member(2).unwrap();
    assert!(first.data() == second.data(), "deterministic replay");
    assert_eq!(handle.retained_bytes(), retained, "flat on a warm call");
    let (d, v) = handle.take_output().unwrap();
    assert_eq!((d.len(), v.len()), (3, 3));
    assert!(handle.retained_bytes() < retained);
}
