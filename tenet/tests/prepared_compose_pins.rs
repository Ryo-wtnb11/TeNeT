//! Characterization pins of the Host `ComposePlan` (#1775), recorded on the
//! base `ec3dea5d` before `ComposePlan` moved onto `ContractPlan`.
//!
//! Over U(1), SU(2) and fZ2xU(1), f64 and c64, one plan and one workspace run
//! the member counts B = 1, 2, 17, 1 with new dyadic values on every call.
//! Every product and partial sum of the fixtures is exact, so the value
//! digest (±0 and NaN folded) does not depend on the dense kernel's
//! summation order. Each row pins, in this order:
//! `[digest, retained after execute, retained after execute_into,
//! first-execute allocation calls, warm-execute allocation calls,
//! execute_into allocation calls]`. Bytes are reported, not pinned. The
//! error matrix pins the variant and message of every rejected call.
//!
//! One test, so the process-wide caches warm in one fixed order and the
//! allocation counts are reproducible.

#![cfg(all(
    feature = "cpu-faer",
    not(any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl"
    ))
))]

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

use std::fmt::Debug;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use tenet::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use tenet::typed::{ComposePlan, GradedSpace, Runtime, StackedTensorMap};

use common::Payload;
use prepared::{filled, fz2u1_legs, members, su2_legs, u1_legs};

type Row = [u64; 6];

fn fold(state: &mut u64, values: impl IntoIterator<Item = f64>) {
    for value in values {
        let bits = if value.is_nan() {
            f64::NAN.to_bits()
        } else if value == 0.0 {
            0
        } else {
            value.to_bits()
        };
        for byte in bits.to_le_bytes() {
            *state ^= u64::from(byte);
            *state = state.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn stack_digest<R, D>(stack: &StackedTensorMap<R, D>) -> u64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let mut state = 0xcbf2_9ce4_8422_2325;
    for index in 0..stack.len() {
        let member = stack.member(index).unwrap();
        fold(
            &mut state,
            member.dense_data().unwrap().iter().flat_map(|value| {
                let (re, im) = value.parts();
                [re, im]
            }),
        );
    }
    state
}

fn calls<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let (value, allocs) = counting_alloc::measure(f);
    (value, allocs.calls, allocs.bytes)
}

fn record<R, D>(
    label: &str,
    (v, w): (GradedSpace<R>, GradedSpace<R>),
    rows: &mut Vec<(String, Row)>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
    D: Payload,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let stacks = |count, salt| {
        (
            StackedTensorMap::pack(&members::<R, D>(&runtime, &[&v, &v], &[&w], count, salt))
                .unwrap(),
            StackedTensorMap::pack(&members::<R, D>(&runtime, &[&w], &[&v], count, salt + 1))
                .unwrap(),
        )
    };
    let (lhs, rhs) = stacks(1, 1);
    // A first composition on another plan creates the Runtime's context and
    // fills the process-wide caches, whose cold cost depends on the platform
    // and the enabled features; the rows below count this plan's own work.
    let warmup = ComposePlan::new(&lhs, &rhs).unwrap();
    warmup
        .execute(&lhs, &rhs, &mut warmup.workspace().unwrap())
        .unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    for (call, count) in [1usize, 2, 17, 1].into_iter().enumerate() {
        let key = format!("{label} {} call={call} B={count}", D::NAME);
        let (lhs, rhs) = stacks(count, 10 * call + 3);
        let fills = [D::entry(f64::NAN, f64::NAN), D::entry(-3.0e17, 7.5)].map(|fill| {
            StackedTensorMap::pack(&filled::<R, D>(&runtime, &[&v, &v], &[&v], count, fill))
                .unwrap()
        });
        let (_, first, first_bytes) = calls(|| {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        let (_, warm, warm_bytes) = calls(|| {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        let value = stack_digest(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
        let retained = workspace.retained_bytes() as u64;
        let mut into_calls = Vec::new();
        for mut dst in fills {
            let (_, into, into_bytes) = calls(|| {
                plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
            });
            assert_eq!(stack_digest(&dst), value, "{key}: execute_into == execute");
            into_calls.push((into, into_bytes));
        }
        let retained_into = workspace.retained_bytes() as u64;
        eprintln!(
            "    (\"{key}\", [{value:#x}, {retained}, {retained_into}, {first}, {warm}, {}]), \
             // bytes first={first_bytes} warm={warm_bytes} into={:?}",
            into_calls[1].0, into_calls
        );
        rows.push((
            key,
            [value, retained, retained_into, first, warm, into_calls[1].0],
        ));
    }
}

/// Every rejected call of `mismatched_stacks_are_typed_errors_before_any_work`
/// and the foreign workspace, as `Debug` of the error.
fn error_matrix() -> Vec<(&'static str, String)> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let lhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 3, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 3, 2)).unwrap();
    let short = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 2)).unwrap();
    let foreign =
        StackedTensorMap::pack(&members::<_, f64>(&other_runtime, &[&w], &[&v], 3, 2)).unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let other_plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut other_ws = other_plan.workspace().unwrap();
    let mut wrong_dst =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&v], 3, 3)).unwrap();
    let mut short_dst =
        StackedTensorMap::pack(&filled::<_, f64>(&runtime, &[&v, &v], &[&v], 2, 1.0)).unwrap();
    let mut dst =
        StackedTensorMap::pack(&filled::<_, f64>(&runtime, &[&v, &v], &[&v], 3, 1.0)).unwrap();
    let show = |result: Result<(), tenet::typed::Error>| format!("{:?}", result.err());
    vec![
        (
            "new: foreign runtime",
            show(ComposePlan::new(&lhs, &foreign).map(|_| ())),
        ),
        (
            "new: member counts differ",
            show(ComposePlan::new(&lhs, &short).map(|_| ())),
        ),
        (
            "execute: swapped operands",
            show(plan.execute(&rhs, &lhs, &mut ws).map(|_| ())),
        ),
        (
            "execute: foreign runtime",
            show(plan.execute(&lhs, &foreign, &mut ws).map(|_| ())),
        ),
        (
            "execute: member counts differ",
            show(plan.execute(&lhs, &short, &mut ws).map(|_| ())),
        ),
        (
            "execute_into: wrong destination",
            show(plan.execute_into(&lhs, &rhs, &mut wrong_dst, &mut ws)),
        ),
        (
            "execute_into: short destination",
            show(plan.execute_into(&lhs, &rhs, &mut short_dst, &mut ws)),
        ),
        (
            "execute_into: foreign workspace",
            show(plan.execute_into(&lhs, &rhs, &mut dst, &mut other_ws)),
        ),
        (
            "execute: foreign workspace",
            show(plan.execute(&lhs, &rhs, &mut other_ws).map(|_| ())),
        ),
    ]
}

#[test]
fn host_compose_plan_values_bytes_allocations_and_errors_are_pinned() {
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let mut rows = Vec::new();
    record::<_, f64>("U1", u1_legs(), &mut rows);
    record::<_, Complex64>("U1", u1_legs(), &mut rows);
    record::<_, f64>("SU2", su2_legs(), &mut rows);
    record::<_, Complex64>("SU2", su2_legs(), &mut rows);
    record::<_, f64>("fZ2xU1", fz2u1_legs(), &mut rows);
    record::<_, Complex64>("fZ2xU1", fz2u1_legs(), &mut rows);
    let errors = error_matrix();
    for (row, error) in &errors {
        eprintln!("    ({row:?}, {error:?}),");
    }
    let pinned: Vec<_> = ROWS
        .iter()
        .map(|&(key, row)| (key.to_string(), row))
        .collect();
    assert_eq!(rows, pinned, "observed rows are printed above");
    let pinned_errors: Vec<_> = ERRORS
        .iter()
        .map(|&(row, error)| (row, error.to_string()))
        .collect();
    assert_eq!(errors, pinned_errors, "observed errors are printed above");
}

#[rustfmt::skip]
const ROWS: &[(&str, Row)] = &[
    ("U1 f64 call=0 B=1", [0xd76442f7a019dd2, 520, 520, 191, 1, 1]),
    ("U1 f64 call=1 B=2", [0xcc7f92611b071413, 928, 928, 10, 1, 1]),
    ("U1 f64 call=2 B=17", [0x4aea74fa1b49b599, 7048, 7048, 11, 1, 1]),
    ("U1 f64 call=3 B=1", [0x6b6d317cdac66a13, 520, 520, 10, 1, 1]),
    ("U1 c64 call=0 B=1", [0x652cf4b4aef062b0, 832, 832, 187, 1, 1]),
    ("U1 c64 call=1 B=2", [0x297cf63f68f0d68f, 1552, 1552, 10, 1, 1]),
    ("U1 c64 call=2 B=17", [0x2dc5cbbc33f0b32e, 12352, 12352, 11, 1, 1]),
    ("U1 c64 call=3 B=1", [0xb75dc3454851fb8b, 832, 832, 10, 1, 1]),
    ("SU2 f64 call=0 B=1", [0x990d51ba85b0b569, 616, 616, 187, 1, 1]),
    ("SU2 f64 call=1 B=2", [0x9e89efa27937d69d, 1120, 1120, 10, 1, 1]),
    ("SU2 f64 call=2 B=17", [0x5bb560c4813e8837, 8680, 8680, 11, 1, 1]),
    ("SU2 f64 call=3 B=1", [0x43ce177b7d3eedd5, 616, 616, 10, 1, 1]),
    ("SU2 c64 call=0 B=1", [0x6d1183ab6294aee9, 1024, 1024, 187, 1, 1]),
    ("SU2 c64 call=1 B=2", [0x14750eff5eb8112b, 1936, 1936, 10, 1, 1]),
    ("SU2 c64 call=2 B=17", [0x5a1ded85d8111ec2, 15616, 15616, 11, 1, 1]),
    ("SU2 c64 call=3 B=1", [0x93d59f926f684fe5, 1024, 1024, 10, 1, 1]),
    ("fZ2xU1 f64 call=0 B=1", [0xe4b22a8e500beb3f, 496, 496, 187, 1, 1]),
    ("fZ2xU1 f64 call=1 B=2", [0xa0dc11d634d2c0ec, 880, 880, 10, 1, 1]),
    ("fZ2xU1 f64 call=2 B=17", [0x4b5950622b829e71, 6640, 6640, 11, 1, 1]),
    ("fZ2xU1 f64 call=3 B=1", [0x8e144ba44bb62cb5, 496, 496, 10, 1, 1]),
    ("fZ2xU1 c64 call=0 B=1", [0x6d763c21c6ac588b, 784, 784, 187, 1, 1]),
    ("fZ2xU1 c64 call=1 B=2", [0xf59530db98534636, 1456, 1456, 10, 1, 1]),
    ("fZ2xU1 c64 call=2 B=17", [0x758b6a9cb59609e, 11536, 11536, 11, 1, 1]),
    ("fZ2xU1 c64 call=3 B=1", [0xdf19f94b2386f04c, 784, 784, 10, 1, 1]),
];

#[rustfmt::skip]
const ERRORS: &[(&str, &str)] = &[
    ("new: foreign runtime", "Some(RuntimeMismatch)"),
    ("new: member counts differ", "None"),
    ("execute: swapped operands", "Some(BatchSignatureMismatch { member: None, field: HomSpace })"),
    ("execute: foreign runtime", "Some(BatchSignatureMismatch { member: None, field: Runtime })"),
    ("execute: member counts differ", "Some(InvalidArgument(\"operand stacks hold 3 and 2 members\"))"),
    ("execute_into: wrong destination", "Some(BatchSignatureMismatch { member: None, field: HomSpace })"),
    ("execute_into: short destination", "Some(InvalidArgument(\"destination stack holds 2 members, operands 3\"))"),
    ("execute_into: foreign workspace", "Some(InvalidArgument(\"compose workspace belongs to another plan\"))"),
    ("execute: foreign workspace", "Some(InvalidArgument(\"compose workspace belongs to another plan\"))"),
];
