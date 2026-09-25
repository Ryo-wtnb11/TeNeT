//! `StackedTensorMap` on Host (#1497, leaf L1 of #1287).

use num_complex::Complex64;

use tenet::prelude::{Error, Runtime};
use tenet::typed::{
    BatchMemberRepresentation, GradedSpace, SectorSpectrum, SignatureField, StackedTensorMap,
    TensorMap,
};

#[macro_use]
#[path = "stacked/fixtures.rs"]
mod fixtures;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

macro_rules! round_trip {
    ($label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let runtime = runtime();
        round_trip!(@dtype $label, &runtime, &a, f64);
        round_trip!(@dtype $label, &runtime, &a, Complex64);
    }};
    (@dtype $label:expr, $runtime:expr, $a:expr, $d:ty) => {{
        let label = format!("{} {}", $label, stringify!($d));
        let members = members!($runtime, $a, $d, 3);
        let stack = StackedTensorMap::pack(&members).unwrap();
        assert_eq!(stack.len(), 3);
        assert!(*stack.signature() == members[0].structure_signature());
        for (index, member) in members.iter().enumerate() {
            let unpacked = stack.member(index).unwrap();
            assert!(unpacked.structure_signature() == member.structure_signature());
            fixtures::assert_bit_exact(unpacked.data(), member.data(), &label);
        }
        assert!(matches!(stack.member(3), Err(Error::InvalidArgument(_))));
    }};
}

#[test]
fn member_of_pack_is_bit_exact() {
    for_each_symmetry!(round_trip);
}

fn assert_drift(result: Result<StackedTensorMap<impl Sized, f64>, Error>, field: SignatureField) {
    match result {
        Err(error) => assert_eq!(
            error,
            Error::BatchSignatureMismatch {
                member: Some(2),
                field
            }
        ),
        Ok(_) => panic!("drift must be rejected"),
    }
}

macro_rules! space_drift {
    ($label:expr, $leg:expr) => {{
        let leg = $leg;
        let (a, degeneracy) = (leg(0), leg(1));
        let dual = a.try_dual().unwrap();
        let runtime = runtime();
        let base = members!(&runtime, &a, f64, 2);
        for other in [&degeneracy, &dual] {
            let drifted = TensorMap::<_, f64>::zeros(&runtime, [&a, other], [&a]).unwrap();
            assert_drift(
                StackedTensorMap::pack(&[&base[0], &base[1], &drifted]),
                SignatureField::HomSpace,
            );
        }
    }};
}

#[test]
fn space_drift_names_the_member_and_field() {
    for_each_symmetry!(space_drift);
}

#[test]
fn runtime_drift_names_the_member_and_field() {
    let leg = GradedSpace::try_new(
        fixtures::U1FusionRule,
        [
            (fixtures::U1Irrep::new(0), 2),
            (fixtures::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let (first, second) = (runtime(), runtime());
    let base = members!(&first, &leg, f64, 2);
    let drifted = members!(&second, &leg, f64, 1);
    assert_drift(
        StackedTensorMap::pack(&[&base[0], &base[1], &drifted[0]]),
        SignatureField::Runtime,
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn rule_instance_drift_names_the_member_and_field() {
    use std::sync::Arc;
    let runtime = runtime();
    let tensor = |rank: usize, trivial: Vec<i64>| {
        let rule = Arc::new(tenet::typed::SUNFusionRule::new(rank).unwrap());
        let leg = GradedSpace::try_new_with_arc(rule, [(trivial, 2)]).unwrap();
        TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg]).unwrap()
    };
    let su3 = tensor(3, vec![0, 0]);
    assert_drift(
        StackedTensorMap::pack(&[&su3, &su3, &tensor(4, vec![0, 0, 0])]),
        SignatureField::Rule,
    );
}

#[test]
fn lazy_adjoint_and_compact_diagonal_members_are_rejected() {
    let runtime = runtime();
    let q = fixtures::U1Irrep::new;
    let bond = GradedSpace::try_new(fixtures::U1FusionRule, [(q(0), 2), (q(1), 1)]).unwrap();
    let square = TensorMap::<_, f64>::rand(&runtime, [&bond], [&bond]).unwrap();
    let lazy = square.adjoint().unwrap();
    let diagonal = TensorMap::<_, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: q(0),
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: q(1),
                values: vec![3.0],
            },
        ],
    )
    .unwrap();
    for (tensor, representation) in [
        (&lazy, BatchMemberRepresentation::LazyAdjoint),
        (&diagonal, BatchMemberRepresentation::CompactDiagonal),
    ] {
        assert_eq!(
            StackedTensorMap::pack(&[&square, tensor]).err(),
            Some(Error::UnsupportedBatchMember {
                member: 1,
                representation
            })
        );
    }
}

#[test]
fn empty_batch_is_rejected() {
    let empty: [&TensorMap<fixtures::U1FusionRule, f64>; 0] = [];
    assert!(matches!(
        StackedTensorMap::pack(&empty),
        Err(Error::InvalidArgument(_))
    ));
}
