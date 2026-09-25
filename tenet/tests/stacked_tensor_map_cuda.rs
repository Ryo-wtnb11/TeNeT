//! `StackedTensorMap` transfers on a real device (#1497, leaf L1 of #1287).
//!
//! Its own binary with one test, because `cuda_transfer_stats` is
//! process-wide. Run with `cargo test -p tenet-rs --no-default-features
//! --features cuda,cpu-faer --test stacked_tensor_map_cuda -- --ignored`.

#![cfg(feature = "cuda")]

use num_complex::Complex64;

use tenet::dense::cuda_transfer_stats;
use tenet::prelude::Runtime;
use tenet::typed::StackedTensorMap;

#[macro_use]
#[path = "stacked/fixtures.rs"]
mod fixtures;

macro_rules! device_round_trip {
    ($label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        device_round_trip!(@dtype $label, &runtime, &a, f64);
        device_round_trip!(@dtype $label, &runtime, &a, Complex64);
    }};
    (@dtype $label:expr, $runtime:expr, $a:expr, $d:ty) => {{
        let label = format!("{} {}", $label, stringify!($d));
        let members = members!($runtime, $a, $d, 3);
        let stack = StackedTensorMap::pack(&members).unwrap();
        let bytes = (stack.len() * members[0].data().len() * std::mem::size_of::<$d>()) as u64;

        let before = cuda_transfer_stats();
        let device = stack.to_cuda().unwrap();
        let uploaded = cuda_transfer_stats();
        assert_eq!(uploaded.h2d_calls - before.h2d_calls, 1, "{label}: to_cuda H2D");
        assert_eq!(uploaded.h2d_bytes - before.h2d_bytes, bytes, "{label}: H2D bytes");
        assert_eq!(uploaded.d2h_calls, before.d2h_calls, "{label}: to_cuda D2H");

        let host = device.to_host().unwrap();
        let downloaded = cuda_transfer_stats();
        assert_eq!(downloaded.d2h_calls - uploaded.d2h_calls, 1, "{label}: to_host D2H");
        assert_eq!(downloaded.h2d_calls, uploaded.h2d_calls, "{label}: to_host H2D");

        assert!(*host.signature() == *stack.signature(), "{label}");
        let device_member = members[0].to_cuda().unwrap();
        assert!(
            *device.signature() == device_member.structure_signature(),
            "{label}: device signature"
        );
        assert!(*device.signature() != *stack.signature(), "{label}: placement");
        for (index, member) in members.iter().enumerate() {
            fixtures::assert_bit_exact(host.member(index).unwrap().data(), member.data(), &label);
        }
    }};
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_round_trip_is_bit_exact_with_one_transfer_each_way() {
    for_each_symmetry!(device_round_trip);
}
