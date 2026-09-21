//! #1361: a compact QR or SVD enters one CPU linear-algebra session per call,
//! however many coupled sectors it factorizes, under both the default and the
//! one-thread CPU layout.
//!
//! Routes: the multiplicity-free fixtures are canonical coupled-sector matrix
//! layouts, so they take the direct-region QR and SVD. The matricization,
//! Generic and checked-Generic QR routes are gated in
//! `tenet-matrixalgebra/tests/factorization_session_scope.rs`.
//!
//! Its own test binary: `cpu_session_stats` is a process-wide counter, so a
//! delta means something only when no unrelated test opens a session in the
//! same process. The mutex serializes the readers inside this binary.

#![cfg(all(feature = "cpu-faer", not(feature = "provider-inject")))]

use std::sync::Mutex;

use tenet::prelude::*;
use tenet_dense::cpu_session_stats;

static COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn sessions_during<T>(call: impl FnOnce() -> T) -> (u64, T) {
    let before = cpu_session_stats().sessions_opened;
    let value = call();
    (cpu_session_stats().sessions_opened - before, value)
}

macro_rules! assert_one_session_per_factorization {
    ($runtime:expr, $provider:expr, $dtype:ty, $sectors:expr, $nc:expr, $nd:expr) => {{
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let leg = GradedSpace::try_new($provider, $sectors.into_iter().map(|s| (s, 2))).unwrap();
        let tensor =
            TensorMap::<_, $dtype>::rand_with_seed($runtime, vec![&leg; $nc], vec![&leg; $nd], 7)
                .unwrap();
        let mut coupled = (0..tensor.block_count())
            .map(|i| tensor.block_fusion_trees(i).unwrap().coupled().clone())
            .collect::<Vec<_>>();
        coupled.sort();
        coupled.dedup();
        assert!(coupled.len() > 2, "fixture must span many sectors");

        let (sessions, _) = sessions_during(|| tensor.qr_compact().unwrap());
        assert_eq!(sessions, 1, "qr_compact");

        let (sessions, _) = sessions_during(|| tensor.left_orth().unwrap());
        assert_eq!(sessions, 1, "left_orth");

        let (sessions, _) = sessions_during(|| tensor.svd_compact().unwrap());
        assert_eq!(sessions, 1, "svd_compact");
    }};
}

fn runtimes() -> [Runtime; 2] {
    [
        Runtime::builder().build().unwrap(),
        Runtime::builder().dense_threads(1).build().unwrap(),
    ]
}

fn centered(count: i32) -> impl Iterator<Item = i32> {
    let low = -((count - 1) / 2);
    (0..count).map(move |i| low + i)
}

#[test]
fn u1_compact_factorizations_open_one_session() {
    let sectors = || centered(8).map(U1Irrep::new).collect::<Vec<_>>();
    for runtime in runtimes() {
        assert_one_session_per_factorization!(&runtime, U1FusionRule, f64, sectors(), 1, 1);
        assert_one_session_per_factorization!(&runtime, U1FusionRule, Complex64, sectors(), 2, 2);
    }
}

#[test]
fn su2_compact_factorizations_open_one_session() {
    let sectors = || (0..4).map(SU2Irrep::from_twice_spin).collect::<Vec<_>>();
    for runtime in runtimes() {
        assert_one_session_per_factorization!(&runtime, SU2FusionRule, f64, sectors(), 2, 1);
        assert_one_session_per_factorization!(&runtime, SU2FusionRule, Complex64, sectors(), 2, 2);
    }
}

#[test]
fn fz2_u1_compact_factorizations_open_one_session() {
    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let sectors = || {
        centered(6)
            .map(|q| {
                let parity = if q.rem_euclid(2) == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                };
                ProductSector::new(parity, U1Irrep::new(q))
            })
            .collect::<Vec<_>>()
    };
    let rule = || Fz2U1::new(FermionParityFusionRule, U1FusionRule);
    for runtime in runtimes() {
        assert_one_session_per_factorization!(&runtime, rule(), f64, sectors(), 1, 1);
        assert_one_session_per_factorization!(&runtime, rule(), Complex64, sectors(), 2, 1);
    }
}
