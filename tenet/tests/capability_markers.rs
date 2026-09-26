//! Static-assertion side of the payload-dtype capability split (#1308).
//!
//! Each helper below is a *positive* assertion: the family's representative
//! entry points are callable with exactly the marker named in its bound, and
//! with nothing stronger. The matching *negative* assertions — that a
//! `D: TensorScalar` caller cannot reach a factorization or a matrix function,
//! and that a `D: FactorizationScalar` caller cannot reach a matrix function —
//! are `compile_fail` doctests on the marker traits themselves
//! (`tenet::typed::FactorizationScalar`, `tenet::typed::AdvancedLinalgScalar`),
//! because a `compile_fail` alone can pass for an unrelated reason: the pair is
//! what pins the boundary.
//!
//! The helpers are never executed. They are instantiated as function items in
//! the test below, which is what forces their bodies to be type-checked.

use tenet::prelude::{
    AdvancedLinalgScalar, FactorizationScalar, TensorMap, TensorScalar, U1FusionRule,
};
use tenet_matrixalgebra::FactorScalar;

/// Base family: everything admitted by [`TensorScalar`] alone.
fn base_family<D: TensorScalar>(tensor: &TensorMap<U1FusionRule, D>) {
    let _ = tensor.norm(2.0);
    let _ = tensor.norm(f64::INFINITY);
    let _ = tensor.inner(tensor);
    let _ = tensor.tr();
    let _ = tensor.adjoint();
    let _ = tensor.permute(&[0], &[1]);
    let _ = tensor.transpose();
    let _ = tensor.repartition(1);
    let _ = tensor.twist(&[0]);
    let _ = tensor.compose(tensor);
    let _ = tensor.otimes(tensor);
    let _ = tensor.trace_pairs(&[(0, 1)]);
    let _ = tensor.diagview();
    let _ = tensor.is_hermitian(0.0);
    let _ = tensor.is_unitary(0.0);
    let _ = tensor.blocks();
}

/// Factorization family: needs [`FactorizationScalar`], not more.
fn factorization_family<D: FactorizationScalar>(tensor: &TensorMap<U1FusionRule, D>) {
    let _ = tensor.qr_compact();
    let _ = tensor.qr_full();
    let _ = tensor.lq_compact();
    let _ = tensor.lq_full();
    let _ = tensor.svd_compact();
    let _ = tensor.svd_full();
    let _ = tensor.svd_vals();
    let _ = tensor.eigh_full();
    let _ = tensor.eigh_vals();
    let _ = tensor.left_null();
    let _ = tensor.right_null();
    let _ = tensor.left_polar();
    let _ = tensor.right_polar();
    let _ = tensor.is_posdef(0.0);
}

/// Advanced linear algebra: needs [`AdvancedLinalgScalar`].
fn advanced_family<D: AdvancedLinalgScalar>(tensor: &TensorMap<U1FusionRule, D>) {
    let _ = tensor.exp();
    let _ = tensor.sqrt();
    let _ = tensor.powi(2);
    let _ = tensor.inv();
    let _ = tensor.pinv(0.0);
    let _ = tensor.solve(tensor);
    let _ = tensor.solve_right(tensor);
}

/// The general (non-Hermitian) eigendecomposition also needs its factor dtype
/// `D::Eig` to be a payload, so it gets its own helper rather than widening
/// the marker.
fn general_eig_family<D>(tensor: &TensorMap<U1FusionRule, D>)
where
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    let _ = tensor.eig_full();
    let _ = tensor.eig_vals();
}

/// Instantiating the helpers for every admitted payload dtype is the
/// assertion: it type-checks each body under exactly one marker bound.
#[test]
fn each_family_is_callable_under_exactly_its_marker() {
    let _: fn(&TensorMap<U1FusionRule, f64>) = base_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = base_family;
    // Single precision reaches every family: base (#1315), factorization
    // (#1324) and advanced (#1459).
    let _: fn(&TensorMap<U1FusionRule, f32>) = base_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = base_family;
    let _: fn(&TensorMap<U1FusionRule, f64>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, f32>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, f64>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, f64>) = general_eig_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = general_eig_family;
    let _: fn(&TensorMap<U1FusionRule, f32>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, f32>) = general_eig_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = general_eig_family;
}
