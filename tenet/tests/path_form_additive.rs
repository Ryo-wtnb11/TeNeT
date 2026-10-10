//! The `cuda` feature is additive: every typed method unified over storage
//! (#1756) has one definition per name, so its path form resolves with and
//! without `cuda`. CI's `cuda-check` compiles this file under `cuda`, where a
//! Host/CUDA twin pair would be `E0034`-ambiguous.
//!
//! Each #1756 leaf adds the names it unifies.

use std::sync::Arc;

use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{Direction, Eigh, GradedSpace, HermitianTol, Qr, Runtime, Svd, TensorMap};

#[test]
fn unified_methods_resolve_in_path_form() {
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let t: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v]).unwrap();

    // L1 reductions.
    assert_eq!(TensorMap::norm(&t, f64::INFINITY).unwrap(), 1.0);
    assert_eq!(TensorMap::inner(&t, &t).unwrap(), 2.0);

    // L2 elementwise.
    let twice = TensorMap::scale(&t, 2.0).unwrap();
    let sum = TensorMap::axpby(&t, 1.0, &t, 1.0).unwrap();
    assert_eq!(sum.dense_data().unwrap(), twice.dense_data().unwrap());
    let mut zero = TensorMap::zeros_like(&t).unwrap();
    assert_eq!(TensorMap::norm(&zero, 2.0).unwrap(), 0.0);
    TensorMap::axpby_into(&t, &mut zero, 2.0, 0.0).unwrap();
    assert_eq!(zero.dense_data().unwrap(), twice.dense_data().unwrap());

    // L3 compact factors.
    let Svd { s, .. } = TensorMap::svd_compact(&t, &[0], &[1]).unwrap();
    assert!((TensorMap::norm(&s, 2.0).unwrap() - 2f64.sqrt()).abs() < 1e-12);
    let Qr { r, .. } = TensorMap::qr_compact(&t, &[1], &[0]).unwrap();
    assert!((TensorMap::norm(&r, 2.0).unwrap() - 2f64.sqrt()).abs() < 1e-12);

    // L4 eigh_full.
    let Eigh { d, .. } = TensorMap::eigh_full(&t, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert!((TensorMap::norm(&d, 2.0).unwrap() - 2f64.sqrt()).abs() < 1e-12);

    // L5 twist.
    let twisted = TensorMap::twist(&t, &[0], Direction::Forward).unwrap();
    assert_eq!(twisted.dense_data().unwrap(), t.dense_data().unwrap());
}
