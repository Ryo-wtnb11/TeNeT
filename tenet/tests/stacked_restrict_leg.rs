//! `StackedTensorMap::restrict_leg` on Host (#1503, leaf L5b of #1287).
//!
//! The oracle is eager `TensorMap::restrict_leg` applied to each member: the
//! stacked result must equal it bit for bit, with the same space, and reject
//! the same inputs with the same errors.

use num_complex::Complex64;

use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, LegSelection, StackedTensorMap, TensorMap};

#[macro_use]
#[path = "stacked/fixtures.rs"]
mod fixtures;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

macro_rules! restrict_members {
    ($label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let dual = a.try_dual().unwrap();
        let runtime = runtime();
        restrict_members!(@dtype $label, &runtime, &a, &dual, f64);
        restrict_members!(@dtype $label, &runtime, &a, &dual, Complex64);
    }};
    (@dtype $label:expr, $runtime:expr, $a:expr, $dual:expr, $d:ty) => {{
        for count in [1usize, 3] {
            let members = mixed_members!($runtime, $a, $dual, $d, count);
            let stack = StackedTensorMap::pack(&members).unwrap();
            for (axis, parent) in [$a, $dual, $a, $dual].into_iter().enumerate() {
                for (choice, selection) in selections(parent).iter().enumerate() {
                    let label = format!(
                        "{} {} B={count} axis {axis} selection {choice}",
                        $label,
                        stringify!($d)
                    );
                    let restricted = stack.restrict_leg(axis, selection).unwrap();
                    assert_eq!(restricted.len(), count, "{label}");
                    for (i, member) in members.iter().enumerate() {
                        let expected = member.restrict_leg(axis, selection).unwrap();
                        let actual = restricted.member(i).unwrap();
                        assert!(
                            actual.structure_signature() == expected.structure_signature(),
                            "{label}: member {i} space"
                        );
                        assert!(
                            *restricted.signature() == expected.structure_signature(),
                            "{label}: stack signature"
                        );
                        fixtures::assert_bit_exact(actual.data(), expected.data(), &label);
                    }
                }
            }
        }
    }};
}

#[test]
fn restrict_leg_equals_eager_per_member_bitwise() {
    for_each_symmetry!(restrict_members);
}

macro_rules! malformed {
    ($label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let dual = a.try_dual().unwrap();
        let other = leg(1);
        let runtime = runtime();
        let members = mixed_members!(&runtime, &a, &dual, f64, 2);
        let stack = StackedTensorMap::pack(&members).unwrap();
        let on_a = &selections(&a)[0];
        let on_dual = &selections(&dual)[0];
        let on_other = &selections(&other)[0];
        // Out of range, the dual of the selected leg, and a leg of another
        // degeneracy: each is eager's error, word for word.
        for (axis, selection) in [(4, on_a), (0, on_dual), (1, on_a), (2, on_other)] {
            let expected = members[0].restrict_leg(axis, selection).err().unwrap();
            let actual = stack.restrict_leg(axis, selection).err().unwrap();
            assert_eq!(
                format!("{actual:?}"),
                format!("{expected:?}"),
                "{} axis {axis}",
                $label
            );
        }
    }};
}

#[test]
fn restrict_leg_rejects_what_eager_rejects() {
    for_each_symmetry!(malformed);
}

#[cfg(feature = "racah-generated")]
#[test]
fn a_selection_of_another_rule_instance_is_a_rule_mismatch() {
    use std::sync::Arc;
    let runtime = runtime();
    let leg = |rank: usize, trivial: Vec<i64>| {
        let rule = Arc::new(tenet::typed::SUNFusionRule::new(rank).unwrap());
        GradedSpace::try_new_with_arc(rule, [(trivial, 2)]).unwrap()
    };
    let su3 = leg(3, vec![0, 0]);
    let su4 = leg(4, vec![0, 0, 0]);
    let member = TensorMap::<_, f64>::rand(&runtime, [&su3], [&su3]).unwrap();
    let stack = StackedTensorMap::pack(&[&member, &member]).unwrap();
    let foreign = LegSelection::try_new(&su4, [(vec![0i64, 0, 0], 0..1)]).unwrap();
    let expected = member.restrict_leg(0, &foreign).err().unwrap();
    let actual = stack.restrict_leg(0, &foreign).err().unwrap();
    assert!(
        format!("{expected:?}").contains("RuleMismatch"),
        "{expected:?}"
    );
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

#[test]
fn restrict_leg_of_a_blockless_structure_is_empty() {
    let runtime = runtime();
    let q = fixtures::U1Irrep::new;
    let charged = GradedSpace::try_new(fixtures::U1FusionRule, [(q(1), 2)]).unwrap();
    let neutral = GradedSpace::try_new(fixtures::U1FusionRule, [(q(0), 2)]).unwrap();
    let empty = TensorMap::<_, f64>::zeros(&runtime, [&charged], [&neutral]).unwrap();
    let stack = StackedTensorMap::pack(&[&empty, &empty, &empty]).unwrap();
    let selection = LegSelection::try_new(&charged, [(q(1), 1..2)]).unwrap();
    let restricted = stack.restrict_leg(0, &selection).unwrap();
    let expected = empty.restrict_leg(0, &selection).unwrap();
    assert_eq!(restricted.len(), 3);
    assert!(*restricted.signature() == expected.structure_signature());
    assert!(restricted.member(2).unwrap().data().is_empty());
}
