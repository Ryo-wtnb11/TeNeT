//! Real-device gate for `materialize` on CUDA tensors (#1514).
//!
//! The Host `materialize` is gated against `zeros_like().absorb(&adj)` in
//! `typed_materialize.rs`; here the device result must equal it bit for bit
//! (each entry is `1 * [conj] x`, exact for finite nonzero parts), and the
//! device must allocate exactly one payload with no download.
//!
//! One test only: the transfer counters are process-wide.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_materialize -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::dense::cuda_transfer_stats;
use tenet::prelude::{Runtime, TensorScalar};
use tenet::typed::{GradedSpace, NetworkReuseClass, TensorMap};

trait Bits: TensorScalar + Copy {
    fn bits(self) -> (u64, u64);
}

impl Bits for f64 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl Bits for Complex64 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

fn bits<D: Bits>(data: &[D]) -> Vec<(u64, u64)> {
    data.iter().map(|value| value.bits()).collect()
}

macro_rules! device_materialize_matches_host {
    ($runtime:expr, $leg:expr, $dtype:ty, $what:expr) => {{
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let what: &str = $what;
        let host: TensorMap<_, $dtype> =
            TensorMap::rand_with_seed($runtime, [&leg, &dual], [&dual, &leg], 1514).unwrap();
        assert!(host.block_count() >= 2, "{what}: multi-block fixture");
        let host_lazy = host.adjoint().unwrap();
        let expected = host_lazy.zeros_like().absorb(&host_lazy).unwrap();
        let payload_bytes = std::mem::size_of_val(expected.data()) as u64;

        let device = host.to_cuda().unwrap();
        let lazy = device.adjoint().unwrap();
        assert!(lazy.network_reuse_class(false) == NetworkReuseClass::LazyAdjoint);

        // Warm-up: the region copy's unit scalar operand is uploaded once per
        // dtype per context, which is not part of the per-call contract.
        let _ = lazy.materialize().unwrap();
        let before = cuda_transfer_stats();
        let owned = lazy.materialize().unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(after.device_allocs - before.device_allocs, 1, "{what}");
        assert_eq!(after.h2d_calls - before.h2d_calls, 1, "{what}");
        assert_eq!(after.h2d_bytes - before.h2d_bytes, payload_bytes, "{what}");
        assert_eq!(after.d2h_calls, before.d2h_calls, "{what}: no download");

        assert!(owned.network_reuse_class(false) == NetworkReuseClass::OwnedDense);
        let owned = owned.to_host().unwrap();
        assert_eq!(owned.codomain(), expected.codomain(), "{what}: codomain");
        assert_eq!(owned.domain(), expected.domain(), "{what}: domain");
        assert_eq!(bits(owned.data()), bits(expected.data()), "{what}: payload");

        // An owned device tensor is shared, not copied.
        let before = cuda_transfer_stats();
        let _ = device.materialize().unwrap();
        assert_eq!(cuda_transfer_stats(), before, "{what}: owned is a clone");
    }};
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_materialize_matches_host_with_one_payload_allocation() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let u1 = || {
        GradedSpace::try_new_with_arc(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 3),
            ],
        )
        .unwrap()
    };
    let su2 = || {
        GradedSpace::try_new_with_arc(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 1),
                (SU2Irrep::from_twice_spin(2), 2),
            ],
        )
        .unwrap()
    };
    let fz2_u1 = || {
        GradedSpace::try_new_with_arc(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2),
            ],
        )
        .unwrap()
    };
    device_materialize_matches_host!(&runtime, u1(), f64, "U1/f64");
    device_materialize_matches_host!(&runtime, u1(), Complex64, "U1/c64");
    device_materialize_matches_host!(&runtime, su2(), f64, "SU2/f64");
    device_materialize_matches_host!(&runtime, su2(), Complex64, "SU2/c64");
    device_materialize_matches_host!(&runtime, fz2_u1(), f64, "fZ2xU1/f64");
    device_materialize_matches_host!(&runtime, fz2_u1(), Complex64, "fZ2xU1/c64");
}
