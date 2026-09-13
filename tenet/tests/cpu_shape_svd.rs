use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{Runtime, Truncation};
use tenet::typed::{GradedSpace, TensorMap};

#[test]
fn svd_trunc_runtime_reuse_tracks_data_dependent_rank() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space =
        GradedSpace::try_new_with_arc(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let policy = Truncation::absolute_cutoff(0.5).unwrap();

    for (diagonal, kept) in [
        ([4.0, 2.0, 0.1], 2),
        ([4.0, 2.0, 1.0], 3),
        ([4.0, 0.2, 0.1], 1),
        ([4.0, 3.0, 0.0], 2),
    ] {
        let source =
            TensorMap::<_, f64>::from_block_fn(&runtime, [&space], [&space], |_, indices| {
                if indices[0] == indices[1] {
                    diagonal[indices[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        let result = source.svd_trunc(&policy).unwrap();

        assert_eq!(
            result
                .singular_values
                .iter()
                .map(|entry| entry.values.len())
                .sum::<usize>(),
            kept
        );
        assert_eq!(result.u.domain()[0].degeneracies(), &[kept]);
        assert!(result.u.is_isometric(1.0e-12).unwrap());
        assert!(result.vh.adjoint().unwrap().is_isometric(1.0e-12).unwrap());

        let reconstructed = result
            .u
            .compose(&result.s)
            .unwrap()
            .compose(&result.vh)
            .unwrap();
        let residual = source
            .add(&reconstructed, 1.0, -1.0)
            .unwrap()
            .norm()
            .unwrap();
        let discarded = diagonal
            .iter()
            .copied()
            .filter(|value| value.abs() < 0.5)
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!((result.error - discarded).abs() <= 1.0e-12);
        assert!((residual - discarded).abs() <= 1.0e-12);
    }
}
