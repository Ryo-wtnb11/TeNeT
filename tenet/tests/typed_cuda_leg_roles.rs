//! Device factorizations take leg roles `(rows, cols)` exactly as on Host
//! (#1553): `t.op(rows, cols)` is `t.permute(rows, cols)?.op(current split)`,
//! the permute running on the device.
//!
//! The oracle is that composition on the device, compared bit for bit (the
//! same device kernels on the same bytes), and the Host result for the same
//! roles, compared under a tolerance through the reconstruction.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_leg_roles -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::typed::{Eigh, GradedSpace, Qr, Runtime, Svd, TensorMap};

macro_rules! same {
    ($got:expr, $want:expr, $what:expr) => {{
        let (got, want) = ($got.materialize().unwrap(), $want.materialize().unwrap());
        assert!(
            got.codomain() == want.codomain() && got.domain() == want.domain(),
            "{}",
            $what
        );
        assert!(
            got.dense_data().unwrap() == want.dense_data().unwrap(),
            "{}",
            $what
        );
    }};
}

macro_rules! close {
    ($got:expr, $want:expr, $what:expr) => {{
        let (got, want) = (&$got, &$want);
        let residual = got.axpby(1.0, want, -1.0).unwrap().norm(2.0).unwrap();
        assert!(
            residual <= 1e-10 * want.norm(2.0).unwrap().max(1.0),
            "{}: {residual}",
            $what
        );
    }};
}

macro_rules! check {
    ($runtime:expr, $v:expr, $w:expr, $label:expr) => {{
        let (v, w) = (&$v, &$w);
        let host: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&$runtime, [v, w], [w, v], 1553).unwrap();
        let device = host.to_cuda().unwrap();
        let (rows, cols) = ([2, 0], [3, 1]);
        let permuted = device.permute(&rows, &cols).unwrap();
        let host_permuted = host.permute(&rows, &cols).unwrap();

        let Svd { u, s, vh } = device.svd_compact(&rows, &cols).unwrap();
        let expected = permuted.svd_compact(&[0, 1], &[2, 3]).unwrap();
        same!(u.to_host().unwrap(), expected.u.to_host().unwrap(), $label);
        same!(s.to_host().unwrap(), expected.s.to_host().unwrap(), $label);
        same!(
            vh.to_host().unwrap(),
            expected.vh.to_host().unwrap(),
            $label
        );
        let rebuilt = u
            .to_host()
            .unwrap()
            .compose(&s.to_host().unwrap())
            .unwrap()
            .compose(&vh.to_host().unwrap())
            .unwrap();
        close!(rebuilt, host_permuted, $label);

        let Qr { q, r } = device.qr_compact(&rows, &cols).unwrap();
        let expected = permuted.qr_compact(&[0, 1], &[2, 3]).unwrap();
        same!(q.to_host().unwrap(), expected.q.to_host().unwrap(), $label);
        same!(r.to_host().unwrap(), expected.r.to_host().unwrap(), $label);
        let rebuilt = q.to_host().unwrap().compose(&r.to_host().unwrap()).unwrap();
        close!(rebuilt, host_permuted, $label);

        // Square roles on a Hermitian source: both sides swap their two legs.
        let square: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&$runtime, [v, w], [v, w], 1554).unwrap();
        let hermitian = square
            .axpby(1.0, &square.adjoint().unwrap().materialize().unwrap(), 1.0)
            .unwrap();
        let device = hermitian.to_cuda().unwrap();
        let (rows, cols) = ([1, 0], [3, 2]);
        let Eigh { d, v: vectors } = device.eigh_full(&rows, &cols).unwrap();
        let expected = device
            .permute(&rows, &cols)
            .unwrap()
            .eigh_full(&[0, 1], &[2, 3])
            .unwrap();
        same!(d.to_host().unwrap(), expected.d.to_host().unwrap(), $label);
        same!(
            vectors.to_host().unwrap(),
            expected.v.to_host().unwrap(),
            $label
        );
        let vectors = vectors.to_host().unwrap();
        let rebuilt = vectors
            .compose(&d.to_host().unwrap())
            .unwrap()
            .compose(&vectors.adjoint().unwrap().materialize().unwrap())
            .unwrap();
        close!(rebuilt, hermitian.permute(&rows, &cols).unwrap(), $label);

        // The current split is the unchanged device path.
        let current = hermitian.to_cuda().unwrap();
        assert!(current.svd_compact(&[0, 1], &[2, 3]).is_ok());
        assert!(
            current.svd_compact(&[0, 0], &[2, 3]).is_err(),
            "malformed roles"
        );
    }};
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_leg_roles_equal_the_device_permute_composition() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let u1 = |charges: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            charges.iter().map(|&(q, n)| (U1Irrep::new(q), n)),
        )
        .unwrap()
    };
    check!(
        runtime,
        u1(&[(-1, 1), (0, 2), (1, 2)]),
        u1(&[(0, 1), (1, 2), (2, 1)]).try_dual().unwrap(),
        "u1"
    );
    let su2 = |spins: &[(usize, usize)]| {
        GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            spins
                .iter()
                .map(|&(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
        )
        .unwrap()
    };
    check!(
        runtime,
        su2(&[(0, 2), (1, 2), (2, 1)]),
        su2(&[(1, 1), (2, 2)]).try_dual().unwrap(),
        "su2"
    );
}
