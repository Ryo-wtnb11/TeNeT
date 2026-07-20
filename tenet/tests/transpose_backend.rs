//! Selectable CPU transpose kernel (#114): `RuntimeBuilder::transpose_backend`
//! chooses the kernel for pure permuted copies (tree-replay and fusion-block
//! pack/scatter). Backend choice is performance-only — routed copies are
//! byte-identical, so every result must match the fused-loop default exactly
//! (not just to rounding). The builder→backend propagation is pinned by an
//! in-crate runtime test; this exercises the selected runtime end to end.

use tenet::prelude::*;

fn u1_space() -> Space {
    Space::u1([(-1, 2), (0, 3), (1, 2)])
}

fn su2_space() -> Space {
    Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap()
}

/// Permute (transpose-heavy tree replay) + swapped-axis contraction
/// (fusion-block pack/scatter) on one runtime; returns the raw block data of
/// both results for exact comparison.
fn transpose_workload(rt: &Runtime, space: &Space) -> (Vec<f64>, Vec<f64>) {
    let a = Tensor::rand_with_seed(rt, Dtype::F64, [space, space], [space, space], 21).unwrap();
    let b = Tensor::rand_with_seed(rt, Dtype::F64, [space, space], [space, space], 22).unwrap();
    let permuted = a.permute(&[1, 0], &[3, 2]).unwrap();
    let contracted = permuted.compose(&b).unwrap();
    (permuted.data().to_vec(), contracted.data().to_vec())
}

#[test]
fn strided_perm_backend_builds_and_matches_fused_default_exactly() {
    let fused = Runtime::builder().build().unwrap();
    let strided = Runtime::builder()
        .transpose_backend(TransposeBackend::StridedPerm)
        .build()
        .unwrap();

    for space in [u1_space(), su2_space()] {
        let (perm_fused, comp_fused) = transpose_workload(&fused, &space);
        let (perm_strided, comp_strided) = transpose_workload(&strided, &space);
        assert!(!perm_fused.is_empty());
        // Byte-identical, not approximately equal: the strided-perm route is a
        // pure permuted copy, so any difference is a routing bug.
        assert_eq!(perm_fused, perm_strided, "permute differs across backends");
        assert_eq!(
            comp_fused, comp_strided,
            "contraction differs across backends"
        );
    }
}

#[test]
fn fused_loops_backend_requested_explicitly_builds_and_computes() {
    // The default, requested explicitly, must behave like the unset default.
    let rt = Runtime::builder()
        .transpose_backend(TransposeBackend::FusedLoops)
        .build()
        .unwrap();
    let (permuted, contracted) = transpose_workload(&rt, &u1_space());
    assert!(!permuted.is_empty());
    assert!(!contracted.is_empty());
}
