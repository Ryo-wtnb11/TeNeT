//! Real-device gate for `materialize` on CUDA tensors (#1545).
//!
//! The Host `materialize` is checked against independent oracles in
//! `typed_materialize.rs`. Here the device result must equal it bit for bit
//! (each entry is `1 * [conj] x`), allocate exactly one device payload with no
//! download, and own that payload: the device `*_overwrite_into`, which
//! rejects a destination aliasing its source, accepts it, and the input is
//! unchanged.
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
            TensorMap::rand_with_seed($runtime, [&leg, &dual], [&dual, &leg], 1545).unwrap();
        assert!(host.subblock_count() >= 2, "{what}: multi-block fixture");
        let host_lazy = host.adjoint().unwrap();
        let payload_bytes = std::mem::size_of_val(host.data()) as u64;
        let device = host.to_cuda().unwrap();
        let lazy = device.adjoint().unwrap();
        assert!(lazy.network_reuse_class(false) == NetworkReuseClass::LazyAdjoint);

        for (input, expected, case) in [
            (&device, host.materialize().unwrap(), "owned"),
            (&lazy, host_lazy.materialize().unwrap(), "adjoint"),
        ] {
            // Warm-up: the region copy's unit scalar operand is uploaded once
            // per dtype per context, which is not part of the per-call contract.
            let _ = input.materialize().unwrap();
            let before = cuda_transfer_stats();
            let owned = input.materialize().unwrap();
            let after = cuda_transfer_stats();
            assert_eq!(
                after.device_allocs - before.device_allocs,
                1,
                "{what} {case}"
            );
            assert_eq!(after.h2d_calls - before.h2d_calls, 1, "{what} {case}");
            assert_eq!(
                after.h2d_bytes - before.h2d_bytes,
                payload_bytes,
                "{what} {case}"
            );
            assert_eq!(
                after.d2h_calls, before.d2h_calls,
                "{what} {case}: no download"
            );

            assert!(owned.network_reuse_class(false) == NetworkReuseClass::OwnedDense);
            assert_eq!(owned.placement(), input.placement(), "{what} {case}");
            assert!(std::ptr::eq(owned.provider(), input.provider()));
            let back = owned.to_host().unwrap();
            assert_eq!(
                back.codomain(),
                expected.codomain(),
                "{what} {case}: codomain"
            );
            assert_eq!(back.domain(), expected.domain(), "{what} {case}: domain");
            assert_eq!(
                bits(back.data()),
                bits(expected.data()),
                "{what} {case}: payload"
            );
        }

        // Independence: the device overwrite rejects a destination that
        // aliases its source, so accepting the copy proves a fresh payload,
        // and the source keeps its values.
        let mut destination = device.materialize().unwrap();
        device
            .permute_overwrite_into(&mut destination, &[0, 1], &[2, 3], <$dtype>::from(2.0))
            .unwrap();
        assert_eq!(
            bits(device.to_host().unwrap().data()),
            bits(host.data()),
            "{what}: input unchanged"
        );
        let doubled: Vec<$dtype> = host
            .data()
            .iter()
            .map(|&value| value * <$dtype>::from(2.0))
            .collect();
        assert_eq!(
            bits(destination.to_host().unwrap().data()),
            bits(&doubled),
            "{what}: destination written"
        );
        let mut alias = device.clone();
        assert!(device
            .permute_overwrite_into(&mut alias, &[0, 1], &[2, 3], <$dtype>::from(2.0))
            .is_err());
    }};
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_materialize_matches_host_with_one_fresh_payload() {
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
