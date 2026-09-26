//! Shared fixtures for the `StackedTensorMap` gates (#1497).

#![allow(dead_code, unused_macros)]

use num_complex::Complex64;

pub use tenet::core::{
    product_sector, FermionParityFusionRule, Fz2SectorLayout, PackedProductCodec,
    ProductFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
};

pub type Fz2U1Rule = ProductFusionRule<
    FermionParityFusionRule,
    U1FusionRule,
    PackedProductCodec<Fz2SectorLayout, U1SectorLayout>,
>;

/// A payload whose bit pattern the gates compare, with deterministic values
/// that include signed zero, a subnormal, an infinity and a NaN payload.
pub trait Payload: Copy {
    fn value(index: usize) -> Self;
    fn bits(self) -> (u64, u64);
}

fn real(index: usize) -> f64 {
    match index % 7 {
        0 => -0.0,
        1 => f64::MIN_POSITIVE / 4.0,
        2 => f64::NEG_INFINITY,
        3 => f64::from_bits(0x7ff8_0000_0000_1234),
        _ => (index as f64 * 0.731_8).sin() * 1.0e3,
    }
}

impl Payload for f64 {
    fn value(index: usize) -> Self {
        real(index)
    }

    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl Payload for Complex64 {
    fn value(index: usize) -> Self {
        Complex64::new(real(index), real(index + 3))
    }

    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

pub fn assert_bit_exact<D: Payload>(left: &[D], right: &[D], label: &str) {
    assert_eq!(left.len(), right.len(), "{label}: length");
    for (index, (&l, &r)) in left.iter().zip(right).enumerate() {
        assert_eq!(l.bits(), r.bits(), "{label}: entry {index}");
    }
}

/// `$count` members `[a, a] <- [a]` of payload `$d`, member `k` filled with
/// `Payload::value(1000 k + n)`.
macro_rules! members {
    ($runtime:expr, $a:expr, $d:ty, $count:expr) => {
        (0..$count)
            .map(|member: usize| {
                let mut index = 1000 * member;
                tenet::typed::TensorMap::<_, $d>::from_block_fn($runtime, [$a, $a], [$a], |_, _| {
                    index += 1;
                    <$d as fixtures::Payload>::value(index)
                })
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
}

/// Calls `$check!(label, leg)` for U(1), SU(2), fZ2xU(1) and, with
/// `racah-generated`, checked Generic SU(3). `leg(0)` is the base leg and
/// `leg(1)` changes one degeneracy.
macro_rules! for_each_symmetry {
    ($check:ident) => {{
        use fixtures::*;
        use std::sync::Arc;
        use tenet::typed::GradedSpace;
        $check!("U1", |variant: usize| {
            let q = U1Irrep::new;
            GradedSpace::try_new(U1FusionRule, [(q(-1), 2), (q(0), 1 + variant), (q(1), 3)])
                .unwrap()
        });
        $check!("SU2", |variant: usize| {
            let j = SU2Irrep::from_twice_spin;
            GradedSpace::try_new(SU2FusionRule, [(j(0), 2), (j(1), 2 + variant), (j(2), 1)])
                .unwrap()
        });
        $check!("fZ2xU1", |variant: usize| {
            let rule = Arc::new(Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule));
            let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
            let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
            GradedSpace::try_new_with_arc(rule, [(even(0), 2), (odd(1), 1 + variant), (odd(-1), 2)])
                .unwrap()
        });
        #[cfg(feature = "racah-generated")]
        $check!("SU3 checked Generic", |variant: usize| {
            let rule = Arc::new(tenet::typed::SUNFusionRule::new(3).unwrap());
            GradedSpace::try_new_with_arc(rule, [(vec![0i64, 0], 1), (vec![1, 1], 2 + variant)])
                .unwrap()
        });
    }};
}

/// Two selections of `leg`, named in its own labels: every sector at a
/// non-leading offset where it has room, with every other one-dimensional
/// sector dropped (several surviving blocks, some removed); and one index of
/// the last sector (a one-dimensional charged leg).
pub fn selections<R>(leg: &tenet::typed::GradedSpace<R>) -> Vec<tenet::typed::LegSelection<R>>
where
    R: tenet::core::TypedSectorAdmission,
    R::Mode: tenet::typed::TypedTensorModeDispatch<R>,
{
    let sectors = leg.sectors().unwrap();
    let degeneracies = leg.degeneracies();
    let partial = sectors
        .iter()
        .zip(degeneracies)
        .enumerate()
        .filter(|&(index, (_, &degeneracy))| index == 0 || degeneracy > 1 || index % 2 == 0)
        .map(|(_, (sector, &degeneracy))| {
            let start = usize::from(degeneracy > 1);
            (sector.clone(), start..degeneracy)
        });
    let last = sectors.len() - 1;
    vec![
        tenet::typed::LegSelection::try_new(leg, partial).unwrap(),
        tenet::typed::LegSelection::try_new(leg, [(sectors[last].clone(), 0..1)]).unwrap(),
    ]
}

/// `$count` members `[a, a*] <- [a, a*]` of payload `$d`: every leg kind
/// (codomain/domain, dual/non-dual) appears once.
macro_rules! mixed_members {
    ($runtime:expr, $a:expr, $dual:expr, $d:ty, $count:expr) => {
        (0..$count)
            .map(|member: usize| {
                let mut index = 1000 * member;
                tenet::typed::TensorMap::<_, $d>::from_block_fn(
                    $runtime,
                    [$a, $dual],
                    [$a, $dual],
                    |_, _| {
                        index += 1;
                        <$d as fixtures::Payload>::value(index)
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
}
