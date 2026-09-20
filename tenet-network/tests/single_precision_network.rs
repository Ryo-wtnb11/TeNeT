//! `tensor!` network execution on single-precision payloads (#1315).
//!
//! Admitting `f32`/`Complex32` to `TensorScalar` opens every `D: TensorScalar`
//! bound in `tenet-network` at once, so the macro path needs its own oracle.
//! It is the same one the typed base suite uses: the `f64`/`Complex64` result
//! of the same network on the widened input, where the twin is filled from one
//! 24-bit stream so the widening is exact.
//!
//! Tolerance is `K * sqrt(n) * eps(f32) * max(1, max|expected|)` with `K = 32`
//! and `n` the payload length of the largest operand — the naive-sum bound for
//! the contraction the network performs.

use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::{Complex32, Complex64};
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::tensor;

const K: f64 = 32.0;

trait Parts: Copy {
    fn parts(re: f32, im: f32) -> Self;
    fn wide(self) -> Complex64;
}

impl Parts for f32 {
    fn parts(re: f32, _im: f32) -> Self {
        re
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl Parts for f64 {
    fn parts(re: f32, _im: f32) -> Self {
        f64::from(re)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Parts for Complex32 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(re, im)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl Parts for Complex64 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(f64::from(re), f64::from(im))
    }
    fn wide(self) -> Complex64 {
        self
    }
}

fn draw(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 40) as f32) / ((1_u32 << 23) as f32) - 1.0
}

fn draw_parts<D: Parts>(state: &mut u64) -> D {
    let re = draw(state);
    let im = draw(state);
    D::parts(re, im)
}

fn assert_payloads_agree<D: Parts, W: Parts>(what: &str, narrow: &[D], wide: &[W], terms: usize) {
    assert_eq!(narrow.len(), wide.len(), "{what}: payload lengths differ");
    let scale = wide
        .iter()
        .map(|&value| value.wide().norm())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let tolerance = K * (terms as f64).sqrt() * f64::from(f32::EPSILON) * scale;
    for (index, (&got, &expected)) in narrow.iter().zip(wide).enumerate() {
        let error = (got.wide() - expected.wide()).norm();
        assert!(
            error <= tolerance,
            "{what}: entry {index} is {:?} against the widened oracle {:?} \
             (error {error:e} > tolerance {tolerance:e})",
            got.wide(),
            expected.wide(),
        );
    }
}

macro_rules! twin {
    ($rt:expr, $narrow:ty, $wide:ty, $codomain:expr, $domain:expr, $seed:expr) => {{
        let mut state = $seed;
        let narrow: TensorMap<_, $narrow> =
            TensorMap::from_block_fn($rt, $codomain, $domain, |_, _| draw_parts(&mut state))
                .unwrap();
        let mut state = $seed;
        let wide: TensorMap<_, $wide> =
            TensorMap::from_block_fn($rt, $codomain, $domain, |_, _| draw_parts(&mut state))
                .unwrap();
        (narrow, wide)
    }};
}

macro_rules! network_suite {
    ($suite:ident, $narrow:ty, $wide:ty) => {
        mod $suite {
            use super::*;

            #[test]
            fn u1_chain_matches_the_widened_oracle() {
                let runtime = Runtime::builder().dense_threads(1).build().unwrap();
                let leg = GradedSpace::try_new_with_arc(
                    Arc::new(U1FusionRule),
                    [
                        (U1Irrep::new(-1), 2),
                        (U1Irrep::new(0), 3),
                        (U1Irrep::new(1), 2),
                    ],
                )
                .unwrap();
                let (a, wa) = twin!(&runtime, $narrow, $wide, [&leg], [&leg], 7_001);
                let (b, wb) = twin!(&runtime, $narrow, $wide, [&leg], [&leg], 7_002);
                let (c, wc) = twin!(&runtime, $narrow, $wide, [&leg], [&leg], 7_003);
                let terms = wa.data().len();

                let got = tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap();
                let expected = tensor!([i; l] = wa[i; j] * wb[j; k] * wc[k; l]).unwrap();
                assert_payloads_agree("u1 chain", got.data(), expected.data(), terms);

                // The plan cache is keyed by structure, not payload dtype; a
                // second execution must therefore still produce the same
                // result at this precision.
                let again = tensor!([i; l] = a[i; j] * b[j; k] * c[k; l]).unwrap();
                assert_eq!(again.data(), got.data());

                // Full trace through the macro, on the single-precision lane.
                let traced = tensor!([] = a[i; i]).unwrap().scalar().unwrap();
                let wide_traced = tensor!([] = wa[i; i]).unwrap().scalar().unwrap();
                let tolerance =
                    K * (terms as f64).sqrt() * f64::from(f32::EPSILON) * wide_traced.wide().norm().max(1.0);
                assert!(
                    (traced.wide() - wide_traced.wide()).norm() <= tolerance,
                    "u1 macro trace: {:?} against {:?}",
                    traced.wide(),
                    wide_traced.wide()
                );
            }

            #[test]
            fn su2_recoupled_network_matches_the_widened_oracle() {
                let runtime = Runtime::builder().dense_threads(1).build().unwrap();
                let leg = GradedSpace::try_new_with_arc(
                    Arc::new(SU2FusionRule),
                    [
                        (SU2Irrep::from_twice_spin(0), 2),
                        (SU2Irrep::from_twice_spin(1), 2),
                    ],
                )
                .unwrap();
                let (a, wa) = twin!(&runtime, $narrow, $wide, [&leg, &leg], [&leg], 7_101);
                let (b, wb) = twin!(&runtime, $narrow, $wide, [&leg], [&leg, &leg], 7_102);
                let terms = wa.data().len();

                let got = tensor!([i, j; l, m] = a[i, j; k] * b[k; l, m]).unwrap();
                let expected = tensor!([i, j; l, m] = wa[i, j; k] * wb[k; l, m]).unwrap();
                assert_payloads_agree("su2 network", got.data(), expected.data(), terms);
                assert!(
                    got.data().iter().any(|&value| value.wide().norm() > 1e-3),
                    "the SU(2) fixture must produce a nonzero result"
                );
            }
        }
    };
}

network_suite!(f32_payload, f32, f64);
network_suite!(complex32_payload, Complex32, Complex64);
