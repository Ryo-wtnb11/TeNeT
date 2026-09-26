//! User-level decomposition contract not already owned by the typed
//! factorization suites.

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{Complex32, Complex64, Runtime};
use tenet::typed::{GradedSpace, Svd, TensorMap, Truncation};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

/// A truncated bond may drop a whole coupled sector, but recomposition must
/// restore the source layout with a zero block so ordinary typed operations
/// remain compatible. The positive-only charge set also pins the historical
/// non-dualization-closed U(1) case.
#[test]
fn truncated_svd_restores_dropped_sector_in_non_dual_closed_space() {
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();

    // Spectra: charge 0 -> {4, 3}, charge 1 -> {1}, charge 2 -> {2}.
    let tensor =
        TensorMap::<_, f64>::from_block_fn(&runtime, [&space], [&space], |trees, indices| {
            if *trees.coupled() == U1Irrep::new(0) {
                match indices {
                    [0, 0] => 4.0,
                    [1, 1] => 3.0,
                    _ => 0.0,
                }
            } else if *trees.coupled() == U1Irrep::new(1) {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();

    // Reconstruction is backward stable: within the tolerance rule, one term
    // per entry of the largest (2 x 2) block.
    let Svd { u, s, vh } = tensor.svd_compact().unwrap();
    numerics::assert_slices_close(
        "u s vh",
        u.compose(&s).unwrap().compose(&vh).unwrap().data(),
        tensor.data(),
        4,
    );

    // Truncation is the composition `find_truncated` + `restrict_*`.
    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), &Truncation::rank(3))
        .unwrap();
    let u = u.restrict_leg(u.codomain_rank(), &found.selection).unwrap();
    let s = s.restrict_diagonal(&found.selection).unwrap();
    let vh = vh.restrict_leg(0, &found.selection).unwrap();
    let kept: Vec<_> = s
        .diagview()
        .unwrap()
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    assert_eq!(kept, [(U1Irrep::new(0), 2), (U1Irrep::new(2), 1)]);
    assert!((found.error - 1.0).abs() < 1.0e-12);

    let recomposed = u.compose(&s).unwrap().compose(&vh).unwrap();
    numerics::assert_slices_close(
        "truncated u s vh",
        recomposed.data(),
        &[4.0, 0.0, 0.0, 3.0, 0.0, 2.0],
        4,
    );
    let error = tensor
        .axpby(1.0, &recomposed, -1.0)
        .unwrap()
        .norm()
        .unwrap();
    assert!((error - found.error).abs() < 1.0e-12);
}

#[test]
fn solve_right_reuses_left_solve_for_real_complex_and_nonselfdual_u1() {
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs = TensorMap::<_, f64>::from_block_fn(&runtime, [&space], [&space], |trees, _| {
        if *trees.coupled() == U1Irrep::new(0) {
            6.0
        } else {
            10.0
        }
    })
    .unwrap();
    let rhs = TensorMap::<_, f64>::from_block_fn(&runtime, [&space], [&space], |trees, _| {
        if *trees.coupled() == U1Irrep::new(0) {
            2.0
        } else {
            5.0
        }
    })
    .unwrap();
    let solved = lhs.solve_right(&rhs).unwrap();
    assert_eq!(solved.compose(&rhs).unwrap().data(), lhs.data());

    let lhs_c = lhs.to_c64();
    let rhs_c = rhs.to_c64();
    let solved_c = lhs_c.solve_right(&rhs_c).unwrap();
    assert!(solved_c
        .compose(&rhs_c)
        .unwrap()
        .data()
        .iter()
        .zip(lhs_c.data())
        .all(|(a, b)| (*a - *b).norm() < 1.0e-12));
}
