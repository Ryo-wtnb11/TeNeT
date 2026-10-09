//! #1859 Host H2: `ContractPlan` Host member replays of the Core and CopyC
//! routes, pinned on exactly representable data.
//!
//! The digests fold signed zeros and NaNs, whose signs the dense backend's
//! architecture-specific kernels choose, so they are platform-independent;
//! every other bit is pinned. Recorded on `e6e9cac8`, before Core and CopyC
//! joined the one Host executor.

use super::*;
use crate::sector::{U1FusionRule, U1Irrep};
use crate::typed::{ContractSpec, GradedSpace, TensorMap};

fn exact(len: usize, salt: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let value = ((index * 7 + salt) % 5) as f64 - 2.0;
            if value == 0.0 && (index + salt) % 2 == 1 {
                -0.0
            } else {
                value
            }
        })
        .collect()
}

fn digest(digest: &mut u64, values: &[f64]) {
    for &value in values {
        let bits = if value.is_nan() {
            f64::NAN.to_bits()
        } else if value == 0.0 {
            0
        } else {
            value.to_bits()
        };
        for byte in bits.to_le_bytes() {
            *digest ^= u64::from(byte);
            *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Member replays at B = 1, 2, 17, 1 on one workspace into NaN-poisoned
/// destinations, folded into one portable digest.
fn member_digest(lhs_axes: &[usize], codomain: &[usize], copy_c: bool) -> u64 {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap();
    let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 6).unwrap();
    let spec = ContractSpec {
        lhs: lhs_axes,
        rhs: &[0],
        codomain,
        domain: &[2],
    };
    let mut state = 0xcbf2_9ce4_8422_2325;
    let mut plan_and_workspace = None;
    for members in [1, 2, 17, 1] {
        let mut lhs = StackedTensorMap::pack(&vec![&a; members]).unwrap();
        let mut rhs = StackedTensorMap::pack(&vec![&b; members]).unwrap();
        let lhs_len = lhs.storage.len();
        let rhs_len = rhs.storage.len();
        lhs.storage = exact(lhs_len, 1);
        rhs.storage = exact(rhs_len, 3);
        let (plan, workspace) = plan_and_workspace.get_or_insert_with(|| {
            let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
            let workspace = plan.workspace().unwrap();
            (plan, workspace)
        });
        assert_eq!(plan.copy_c().is_some(), copy_c);
        assert!(!plan.resolution.is_dynamic_tree());
        plan.execute(&lhs, &rhs, workspace).unwrap();
        let mut dst = workspace.take_output().unwrap();
        dst.storage.fill(f64::NAN);
        plan.execute_into(&lhs, &rhs, &mut dst, workspace).unwrap();
        digest(&mut state, &dst.storage);
    }
    state
}

#[test]
fn member_core_and_copy_c_bits_are_pinned() {
    let observed = [
        ("core", member_digest(&[2], &[0, 1], false)),
        ("copyC", member_digest(&[2], &[1, 0], true)),
    ];
    assert_eq!(
        observed,
        [
            ("core", 0x07e5_80a2_cdf4_b8b8),
            ("copyC", 0x3b1b_a5df_92e3_eab8)
        ],
        "observed digests: {observed:#x?}"
    );
}
