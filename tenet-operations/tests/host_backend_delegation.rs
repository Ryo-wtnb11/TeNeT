//! `HostTensorOperations` as a `TreeTransformBackend` runs the one batched-GEMM
//! replay (#1767); the per-block strided driver is only the reference oracle.

mod common;

use common::{recoupling_fixtures, Fixture, TestScalar};
use num_complex::Complex64;
use tenet_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw,
    tree_transform_structure_with_strided_kernel_raw, ConjugateValue, DenseRecouplingScalar,
    DenseTreeTransformOperations, HostTensorOperations, RecouplingCoefficientAction,
    StridedHostKernelAdapter, TreeTransformBackend, TreeTransformScalar, TreeTransformWorkspace,
};

struct Replays<T> {
    host: Vec<T>,
    dense: Vec<T>,
    reference: Vec<T>,
}

fn replay<T>(fixture: &Fixture, overwrite: bool) -> Replays<T>
where
    T: TestScalar
        + TreeTransformScalar
        + RecouplingCoefficientAction<f64>
        + DenseRecouplingScalar
        + ConjugateValue,
{
    let structure = fixture.compile();
    let dst_structure = fixture.dst_structure();
    let src_structure = fixture.src_structure();
    // Non-dyadic payload so the two summation orders round differently.
    let source: Vec<T> = (0..fixture.src_len())
        .map(|i| T::from_parts((i as f64 * 0.731 + 0.3).sin(), (i as f64 * 1.37).cos()))
        .collect();
    let initial: Vec<T> = (0..fixture.dst_len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    let alpha = T::from_parts(0.7, -0.3);
    let beta = T::from_parts(1.0, 0.0);

    let mut host = initial.clone();
    let mut dense = initial.clone();
    let mut reference = initial;
    let mut host_backend = HostTensorOperations;
    let mut dense_backend = DenseTreeTransformOperations::default();
    if overwrite {
        host_backend
            .tree_transform_structure_overwrite_into_raw(
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut host,
                &source,
                alpha,
            )
            .unwrap();
        dense_backend
            .tree_transform_structure_overwrite_into_raw(
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dense,
                &source,
                alpha,
            )
            .unwrap();
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut reference,
            &source,
            alpha,
        )
        .unwrap();
    } else {
        host_backend
            .tree_transform_structure_into_raw(
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut host,
                &source,
                alpha,
                beta,
            )
            .unwrap();
        dense_backend
            .tree_transform_structure_into_raw(
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dense,
                &source,
                alpha,
                beta,
            )
            .unwrap();
        tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut reference,
            &source,
            alpha,
            beta,
        )
        .unwrap();
    }
    Replays {
        host,
        dense,
        reference,
    }
}

fn check<T>() -> bool
where
    T: TestScalar
        + TreeTransformScalar
        + RecouplingCoefficientAction<f64>
        + DenseRecouplingScalar
        + ConjugateValue,
{
    // The scalar-loop oracle and the GEMM path round differently, so a host
    // result bit-identical to the GEMM path on data where they differ shows
    // which of the two the host backend ran.
    let mut paths_differ = false;
    for fixture in recoupling_fixtures() {
        for overwrite in [true, false] {
            let r = replay::<T>(&fixture, overwrite);
            assert_eq!(r.host, r.dense, "{} overwrite={overwrite}", fixture.name);
            for (a, b) in r.host.iter().zip(&r.reference) {
                assert!(
                    a.distance(*b) <= 1e-12,
                    "{} overwrite={overwrite}: {a:?} vs oracle {b:?}",
                    fixture.name
                );
            }
            paths_differ |= r.dense != r.reference;
        }
    }
    paths_differ
}

#[test]
fn host_backend_is_the_batched_gemm_path() {
    let real = check::<f64>();
    let complex = check::<Complex64>();
    assert!(
        real || complex,
        "no fixture separates the GEMM path from the scalar oracle bitwise"
    );
}
