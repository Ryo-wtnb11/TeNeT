//! User-level decomposition contract not already owned by the typed
//! factorization suites.

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap, Truncation};

/// A truncated bond may drop a whole coupled sector, but recomposition must
/// restore the source layout with a zero block so ordinary typed operations
/// remain compatible. The positive-only charge set also pins the historical
/// non-dualization-closed U(1) case.
#[test]
fn truncated_svd_restores_dropped_sector_in_non_dual_closed_space() {
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(2), 1),
        ],
        false,
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

    let (u, s, vh) = tensor.svd_compact().unwrap();
    assert_eq!(
        u.compose(&s).unwrap().compose(&vh).unwrap().data(),
        tensor.data()
    );

    let truncated = tensor.svd_trunc(&Truncation::rank(3)).unwrap();
    let kept: Vec<_> = truncated
        .singular_values
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    assert_eq!(kept, [(U1Irrep::new(0), 2), (U1Irrep::new(2), 1)]);
    assert!((truncated.error - 1.0).abs() < 1.0e-12);

    let recomposed = truncated
        .u
        .compose(&truncated.s)
        .unwrap()
        .compose(&truncated.vh)
        .unwrap();
    assert_eq!(recomposed.data(), &[4.0, 0.0, 0.0, 3.0, 0.0, 2.0]);
    let error = tensor.add(&recomposed, 1.0, -1.0).unwrap().norm().unwrap();
    assert!((error - truncated.error).abs() < 1.0e-12);
}
