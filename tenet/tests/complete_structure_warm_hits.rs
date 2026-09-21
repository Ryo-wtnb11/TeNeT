//! Warm public eager calls reuse the core complete-structure cache: each call
//! is a hit, never a miss, admission, or eviction, and returns bit-identical
//! data. One test per process keeps the global cache statistics isolated.

use tenet::prelude::*;
use tenet_core::{complete_hom_space_structure_cache_info, CompleteHomSpaceStructureCacheInfo};

fn assert_warm_hits(label: &str, mut call: impl FnMut() -> Vec<u64>) {
    let cold = call();
    for _ in 0..3 {
        let before: CompleteHomSpaceStructureCacheInfo = complete_hom_space_structure_cache_info();
        assert_eq!(call(), cold, "{label}: warm result differs");
        let after = complete_hom_space_structure_cache_info();
        assert!(
            after.hits() > before.hits(),
            "{label}: warm call did not hit"
        );
        assert_eq!(after.misses(), before.misses(), "{label}: warm miss");
        assert_eq!(
            after.admissions(),
            before.admissions(),
            "{label}: warm admission"
        );
        assert_eq!(
            after.evictions(),
            before.evictions(),
            "{label}: warm eviction"
        );
    }
}

fn bits(data: &[f64]) -> Vec<u64> {
    data.iter().map(|value| value.to_bits()).collect()
}

macro_rules! warm_ops {
    ($runtime:expr, $label:literal, $provider:expr, $sectors:expr) => {{
        let sectors: Vec<_> = $sectors;
        let leg = GradedSpace::try_new($provider, sectors.iter().map(|s| (s.clone(), 2))).unwrap();
        let a =
            TensorMap::<_, f64>::rand_with_seed($runtime, [&leg, &leg], [&leg, &leg], 1).unwrap();
        let square =
            TensorMap::<_, f64>::rand_with_seed($runtime, [&leg, &leg], [&leg, &leg], 2).unwrap();
        let matrix = TensorMap::<_, f64>::rand_with_seed($runtime, [&leg], [&leg], 3).unwrap();
        let selection =
            LegSelection::try_new(&leg, sectors.iter().map(|s| (s.clone(), 0..1))).unwrap();
        assert_warm_hits(concat!($label, " permute"), || {
            bits(a.permute(&[1, 2], &[3, 0]).unwrap().data())
        });
        assert_warm_hits(concat!($label, " repartition"), || {
            bits(a.repartition(3).unwrap().data())
        });
        assert_warm_hits(concat!($label, " restrict_leg"), || {
            bits(a.restrict_leg(0, &selection).unwrap().data())
        });
        assert_warm_hits(concat!($label, " compose"), || {
            bits(a.compose(&square).unwrap().data())
        });
        assert_warm_hits(concat!($label, " contract"), || {
            bits(
                a.contract(&matrix, &[0], &[1], &[0, 1, 2, 3])
                    .unwrap()
                    .data(),
            )
        });
    }};
}

#[test]
fn warm_eager_calls_hit_complete_structure_cache() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    warm_ops!(
        &runtime,
        "U1",
        U1FusionRule,
        (-1..=1).map(U1Irrep::new).collect()
    );
    warm_ops!(
        &runtime,
        "SU2",
        SU2FusionRule,
        (0..3).map(SU2Irrep::from_twice_spin).collect()
    );
    warm_ops!(
        &runtime,
        "fZ2xU1",
        ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::new(
            FermionParityFusionRule,
            U1FusionRule
        ),
        (-1..=1)
            .map(|q: i32| {
                let parity = if q.rem_euclid(2) == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                };
                ProductSector::new(parity, U1Irrep::new(q))
            })
            .collect()
    );
}
