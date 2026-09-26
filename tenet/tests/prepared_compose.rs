//! Host gates of `PreparedCompose` (#1498, leaf L2 of #1287).
//!
//! Per member the handle must equal eager `compose` of the same placement,
//! and both must equal the tree-keyed `mul!` oracle of `prepared/mod.rs`,
//! over U(1), SU(2) and fZ2xU(1), f64 and c64, B in {1, 2, 17}, with inactive
//! destination blocks. `execute_into` must give the same result over a
//! destination prefilled with NaN or garbage (D2).

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use tenet::core::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use tenet::prelude::Error;
use tenet::typed::{GradedSpace, PreparedCompose, Runtime, SignatureField, StackedTensorMap};

use common::Payload;
use prepared::{assert_close, compose_oracle, filled, fz2u1_legs, members, su2_legs, u1_legs};

const MEMBER_COUNTS: [usize; 3] = [1, 2, 17];

fn equivalence<R, D>(label: &str, (v, w): (GradedSpace<R>, GradedSpace<R>))
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
    D: Payload,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let runtime = Runtime::builder().build().unwrap();
    for count in MEMBER_COUNTS {
        let label = format!("{label} {} B={count}", D::NAME);
        let a = members::<R, D>(&runtime, &[&v, &v], &[&w], count, 1);
        let b = members::<R, D>(&runtime, &[&w], &[&v], count, 2);
        let lhs = StackedTensorMap::pack(&a).unwrap();
        let rhs = StackedTensorMap::pack(&b).unwrap();
        let mut handle = PreparedCompose::new(&lhs, &rhs).unwrap();

        // `terms` = len(A), an upper bound on the inner dimension of a block.
        let terms = a[0].data().len();
        let mut oracles = Vec::new();
        for (x, y) in a.iter().zip(&b) {
            let eager = x.compose(y).unwrap();
            let (oracle, unreached) = compose_oracle(x, y, &eager);
            assert!(unreached > 0, "{label}: fixture must have inactive blocks");
            assert!(
                eager.subblock_count() > unreached + 1,
                "{label}: several active blocks"
            );
            assert_close(eager.data(), &oracle, terms, &format!("{label}: eager"));
            oracles.push(oracle);
        }

        let output = handle.execute(&lhs, &rhs).unwrap();
        assert_eq!(output.len(), count);
        assert!(*output.signature() == a[0].compose(&b[0]).unwrap().structure_signature());
        for (index, oracle) in oracles.iter().enumerate() {
            let member = output.member(index).unwrap();
            assert_close(
                member.data(),
                oracle,
                terms,
                &format!("{label}: member {index}"),
            );
        }

        for (fill, name) in [
            (D::entry(f64::NAN, f64::NAN), "NaN"),
            (D::entry(-3.0e17, 7.5), "garbage"),
        ] {
            let mut dst =
                StackedTensorMap::pack(&filled::<R, D>(&runtime, &[&v, &v], &[&v], count, fill))
                    .unwrap();
            assert!(*dst.signature() == *handle.output_signature());
            handle.execute_into(&lhs, &rhs, &mut dst).unwrap();
            for (index, oracle) in oracles.iter().enumerate() {
                let member = dst.member(index).unwrap();
                assert_close(
                    member.data(),
                    oracle,
                    terms,
                    &format!("{label}: execute_into over {name}, member {index}"),
                );
            }
        }
    }
}

#[test]
fn prepared_compose_equals_eager_and_the_tree_oracle_per_member() {
    equivalence::<_, f64>("U1", u1_legs());
    equivalence::<_, Complex64>("U1", u1_legs());
    equivalence::<_, f64>("SU2", su2_legs());
    equivalence::<_, Complex64>("SU2", su2_legs());
    equivalence::<_, f64>("fZ2xU1", fz2u1_legs());
    equivalence::<_, Complex64>("fZ2xU1", fz2u1_legs());
}

#[test]
fn warm_replay_retains_the_same_bytes_and_a_new_member_count_resizes() {
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let stacks = |count| {
        (
            StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 1))
                .unwrap(),
            StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], count, 2)).unwrap(),
        )
    };
    let (lhs, rhs) = stacks(4);
    let mut handle = PreparedCompose::new(&lhs, &rhs).unwrap();
    assert_eq!(
        handle.retained_bytes(),
        0,
        "nothing is allocated before a call"
    );
    handle.execute(&lhs, &rhs).unwrap();
    let cold = handle.retained_bytes();
    assert!(cold > 0);
    for _ in 0..3 {
        handle.execute(&lhs, &rhs).unwrap();
        assert_eq!(
            handle.retained_bytes(),
            cold,
            "a warm replay retains nothing new"
        );
    }
    let (wide_lhs, wide_rhs) = stacks(9);
    handle.execute(&wide_lhs, &wide_rhs).unwrap();
    assert!(
        handle.retained_bytes() > cold,
        "a larger B resizes the output"
    );

    // `execute_into` across a change of B on the same handle: the job list
    // is rebuilt for each B and the result stays the oracle's.
    for count in [4, 9, 4] {
        let a = members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 1);
        let b = members::<_, f64>(&runtime, &[&w], &[&v], count, 2);
        let (lhs, rhs) = (
            StackedTensorMap::pack(&a).unwrap(),
            StackedTensorMap::pack(&b).unwrap(),
        );
        let mut dst = StackedTensorMap::pack(&filled::<_, f64>(
            &runtime,
            &[&v, &v],
            &[&v],
            count,
            f64::NAN,
        ))
        .unwrap();
        handle.execute_into(&lhs, &rhs, &mut dst).unwrap();
        for (index, (x, y)) in a.iter().zip(&b).enumerate() {
            let (oracle, _) = compose_oracle(x, y, &x.compose(y).unwrap());
            assert_close(
                dst.member(index).unwrap().data(),
                &oracle,
                x.data().len(),
                &format!("execute_into at B={count}, member {index}"),
            );
        }
    }
    handle.execute(&wide_lhs, &wide_rhs).unwrap();

    let output = handle.take_output().unwrap();
    assert_eq!(output.len(), 9);
    assert!(
        handle.retained_bytes() < cold,
        "the moved output is no longer retained"
    );
}

#[test]
fn mismatched_stacks_are_typed_errors_before_any_work() {
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let lhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 3, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 3, 2)).unwrap();
    let short = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 2)).unwrap();
    let foreign =
        StackedTensorMap::pack(&members::<_, f64>(&other_runtime, &[&w], &[&v], 3, 2)).unwrap();
    let mut handle = PreparedCompose::new(&lhs, &rhs).unwrap();

    assert_eq!(
        PreparedCompose::new(&lhs, &foreign).err(),
        Some(Error::RuntimeMismatch)
    );
    assert_eq!(
        handle.execute(&rhs, &lhs).err(),
        Some(Error::BatchSignatureMismatch {
            member: None,
            field: SignatureField::HomSpace
        })
    );
    assert_eq!(
        handle.execute(&lhs, &foreign).err(),
        Some(Error::BatchSignatureMismatch {
            member: None,
            field: SignatureField::Runtime
        })
    );
    assert!(matches!(
        handle.execute(&lhs, &short),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(
        handle.retained_bytes(),
        0,
        "a rejected call allocates nothing"
    );

    let mut wrong_dst =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&v], 3, 3)).unwrap();
    assert!(matches!(
        handle.execute_into(&lhs, &rhs, &mut wrong_dst),
        Err(Error::BatchSignatureMismatch { member: None, .. })
    ));
    let mut short_dst =
        StackedTensorMap::pack(&filled::<_, f64>(&runtime, &[&v, &v], &[&v], 2, 1.0)).unwrap();
    assert!(matches!(
        handle.execute_into(&lhs, &rhs, &mut short_dst),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(
        short_dst.member(0).unwrap().data()[0],
        1.0,
        "nothing was written"
    );
}
