use std::collections::VecDeque;
use std::sync::{OnceLock, RwLock};

use rustc_hash::FxHashMap;
use wigner_symbols::regge::CanonicalRegge6j;
use wigner_symbols::Wigner6j;

const PUBLICATION_CACHE_BUDGET_BYTES: usize = 8 * 1024 * 1024;
// Why not charge only `size_of::<(Key, f64)>`: the map buckets and the FIFO's
// duplicate key are retained payload too; 128 bytes is a conservative charge.
const PUBLICATION_CACHE_CHARGE_BYTES: usize = 128;
const PUBLICATION_CACHE_ENTRY_CAP: usize =
    PUBLICATION_CACHE_BUDGET_BYTES / PUBLICATION_CACHE_CHARGE_BYTES;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PublicationKey(CanonicalRegge6j);

#[derive(Default)]
struct PublicationCache {
    values: FxHashMap<PublicationKey, f64>,
    insertion_order: VecDeque<PublicationKey>,
}

impl PublicationCache {
    fn publish(&mut self, key: PublicationKey, value: f64) -> f64 {
        if let Some(published) = self.values.get(&key).copied() {
            return published;
        }
        if self.values.len() == PUBLICATION_CACHE_ENTRY_CAP {
            if let Some(evicted) = self.insertion_order.pop_front() {
                self.values.remove(&evicted);
            }
        }
        self.values.insert(key, value);
        self.insertion_order.push_back(key);
        value
    }

    fn clear(&mut self) {
        self.values.clear();
        self.insertion_order.clear();
    }
}

fn publication_cache() -> &'static RwLock<PublicationCache> {
    static CACHE: OnceLock<RwLock<PublicationCache>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(PublicationCache::default()))
}

pub(crate) fn validate_supported_spins(spins: [usize; 6]) {
    let max = spins.into_iter().max().unwrap_or(0);
    // Why not return `Result`: the existing fusion-symbol trait returns its
    // scalar directly, so an unsupported authority range must fail explicitly.
    assert!(
        max <= crate::SU2_MAX_DOUBLED_SPIN,
        "SU(2) doubled spin {max} exceeds the supported maximum {}",
        crate::SU2_MAX_DOUBLED_SPIN
    );
}

pub(crate) fn wigner_6j_doubled(spins: [usize; 6]) -> f64 {
    validate_supported_spins(spins);
    if !admissible(spins) {
        return 0.0;
    }

    let [tj1, tj2, tj3, tj4, tj5, tj6] =
        spins.map(|spin| i32::try_from(spin).expect("validated SU(2) doubled spin must fit i32"));
    let wigner = Wigner6j {
        tj1,
        tj2,
        tj3,
        tj4,
        tj5,
        tj6,
    };
    let key = PublicationKey(CanonicalRegge6j::from(wigner));
    if let Some(value) = publication_cache()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .values
        .get(&key)
        .copied()
    {
        return value;
    }

    let value = f64::from(wigner.value());

    let mut cache = publication_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.publish(key, value)
}

pub(crate) fn reset_publication_cache() {
    let mut cache = publication_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.clear();
}

fn admissible([tj1, tj2, tj3, tj4, tj5, tj6]: [usize; 6]) -> bool {
    triangle(tj1, tj2, tj3)
        && triangle(tj1, tj5, tj6)
        && triangle(tj4, tj2, tj6)
        && triangle(tj4, tj5, tj3)
}

fn triangle(a: usize, b: usize, c: usize) -> bool {
    let sum = a + b + c;
    sum & 1 == 0 && a <= b + c && b <= a + c && c <= a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FusionRule, ProductFusionRule, RuleIdentity, SU2FusionRule, TensorKitProductCodec,
        U1FusionRule,
    };

    fn dependency_publication_key(spins: [usize; 6]) -> PublicationKey {
        let [tj1, tj2, tj3, tj4, tj5, tj6] = spins.map(|spin| spin as i32);
        PublicationKey(CanonicalRegge6j::from(Wigner6j {
            tj1,
            tj2,
            tj3,
            tj4,
            tj5,
            tj6,
        }))
    }

    fn tetrahedral_equivalents(spins: [usize; 6]) -> Vec<[usize; 6]> {
        let top = [spins[0], spins[1], spins[2]];
        let bottom = [spins[3], spins[4], spins[5]];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let swaps = [
            [false, false, false],
            [false, true, true],
            [true, false, true],
            [true, true, false],
        ];
        let mut equivalents = Vec::with_capacity(24);
        for permutation in permutations {
            for swap in swaps {
                let mut equivalent = [0; 6];
                for column in 0..3 {
                    let source = permutation[column];
                    equivalent[column] = if swap[column] {
                        bottom[source]
                    } else {
                        top[source]
                    };
                    equivalent[column + 3] = if swap[column] {
                        top[source]
                    } else {
                        bottom[source]
                    };
                }
                equivalents.push(equivalent);
            }
        }
        equivalents
    }

    #[test]
    fn exact_authority_matches_supported_high_spin_oracles() {
        let cases: [(usize, f64); 4] = [
            (100, -0.000_112_137_492_362_641_99),
            (150, 0.000_515_980_222_695_226),
            (200, -0.000_469_841_623_298_744_2),
            (208, 0.000_103_652_642_315_091_42),
        ];
        for (spin, expected) in cases {
            let actual = wigner_6j_doubled([spin; 6]);
            assert!(actual.is_finite());
            let error = (actual - expected).abs();
            assert!(
                error <= 4.0 * f64::EPSILON * expected.abs(),
                "2j={spin}: actual={actual:?}, expected={expected:?}, error={error:?}"
            );
        }
    }

    #[test]
    fn exact_authority_matches_low_spin_raw_6j_oracles() {
        let cases = [
            ([0, 0, 0, 0, 0, 0], 1.0),
            ([1, 1, 0, 1, 1, 0], -0.5),
            ([1, 1, 0, 1, 1, 2], 0.5),
            ([1, 1, 2, 1, 1, 2], 1.0 / 6.0),
            ([2, 2, 2, 2, 2, 2], 1.0 / 6.0),
        ];
        for (spins, expected) in cases {
            assert_eq!(wigner_6j_doubled(spins), expected, "spins={spins:?}");
        }
    }

    fn assert_exact_authority_and_canonical_publication_agree_through(max_spin: usize) {
        for tj1 in 0..=max_spin {
            for tj2 in 0..=max_spin {
                for tj3 in 0..=max_spin {
                    for tj4 in 0..=max_spin {
                        for tj5 in 0..=max_spin {
                            for tj6 in 0..=max_spin {
                                let spins = [tj1, tj2, tj3, tj4, tj5, tj6];
                                if !admissible(spins) {
                                    continue;
                                }
                                let expected = f64::from(
                                    Wigner6j {
                                        tj1: tj1 as i32,
                                        tj2: tj2 as i32,
                                        tj3: tj3 as i32,
                                        tj4: tj4 as i32,
                                        tj5: tj5 as i32,
                                        tj6: tj6 as i32,
                                    }
                                    .value(),
                                );
                                assert_eq!(wigner_6j_doubled(spins), expected, "spins={spins:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "release audit over the larger 2j <= 12 domain"]
    fn release_audit_exhaustively_compares_exact_and_canonical_publication_through_twelve() {
        assert_exact_authority_and_canonical_publication_agree_through(12);
    }

    #[test]
    fn exact_authority_preserves_tetrahedral_and_nontrivial_regge_equivalences() {
        let fixture = [2, 3, 3, 4, 3, 3];
        let expected_value = wigner_6j_doubled(fixture);
        let expected_key = dependency_publication_key(fixture);
        let equivalents = tetrahedral_equivalents(fixture);
        assert_eq!(equivalents.len(), 24);
        for equivalent in equivalents {
            assert_eq!(dependency_publication_key(equivalent), expected_key);
            assert_eq!(wigner_6j_doubled(equivalent), expected_value);
        }

        let regge_pairs = [
            ([0, 2, 2, 2, 2, 2], [1, 1, 2, 1, 3, 2]),
            ([0, 2, 2, 3, 3, 3], [1, 1, 2, 2, 4, 3]),
        ];
        for (left, right) in regge_pairs {
            assert!(!tetrahedral_equivalents(left).contains(&right));
            assert_eq!(
                dependency_publication_key(left),
                dependency_publication_key(right)
            );
            assert_eq!(wigner_6j_doubled(left), wigner_6j_doubled(right));
        }
    }

    #[test]
    fn publication_cache_is_fifo_bounded_and_clearable() {
        let mut cache = PublicationCache::default();
        for index in 0..=PUBLICATION_CACHE_ENTRY_CAP {
            let key = PublicationKey(CanonicalRegge6j {
                s: (index & 0xff) as u8,
                b: ((index >> 8) & 0xff) as u8,
                t: ((index >> 16) & 0xff) as u8,
                x: 0,
                l: 0,
                e: 0,
            });
            assert_eq!(cache.publish(key, index as f64), index as f64);
        }
        assert_eq!(cache.values.len(), PUBLICATION_CACHE_ENTRY_CAP);
        assert!(!cache
            .values
            .contains_key(&PublicationKey(CanonicalRegge6j::default())));
        cache.clear();
        assert!(cache.values.is_empty());
        assert!(cache.insertion_order.is_empty());
    }

    #[test]
    fn concurrent_publication_returns_one_exact_value() {
        let fixture = [20, 22, 24, 22, 24, 20];
        let [tj1, tj2, tj3, tj4, tj5, tj6] = fixture.map(|spin| spin as i32);
        let expected = f64::from(
            Wigner6j {
                tj1,
                tj2,
                tj3,
                tj4,
                tj5,
                tj6,
            }
            .value(),
        );
        reset_publication_cache();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    wigner_6j_doubled(fixture)
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), expected);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds the supported maximum 254")]
    fn unsupported_spin_panics_before_i32_arithmetic() {
        wigner_6j_doubled([crate::SU2_MAX_DOUBLED_SPIN + 1; 6]);
    }

    #[test]
    fn canonical_authority_boundary_accepts_254_and_rejects_larger_oracles() {
        assert!(CanonicalRegge6j::len(crate::SU2_MAX_DOUBLED_SPIN as i32) > 0);
        assert!(wigner_6j_doubled([crate::SU2_MAX_DOUBLED_SPIN; 6]).is_finite());
        for spin in [255, 300, 400, 800] {
            assert!(std::panic::catch_unwind(|| wigner_6j_doubled([spin; 6])).is_err());
        }
    }

    #[test]
    fn irrep_construction_and_fusion_closure_enforce_the_authority_boundary() {
        assert!(crate::SU2Irrep::try_from_twice_spin(254).is_some());
        assert!(crate::SU2Irrep::try_from_twice_spin(255).is_none());
        assert!(std::panic::catch_unwind(|| crate::SU2Irrep::from_twice_spin(255)).is_err());

        let supported = SU2FusionRule.fusion_channels(
            crate::SU2Irrep::from_twice_spin(127).sector_id(),
            crate::SU2Irrep::from_twice_spin(127).sector_id(),
        );
        assert_eq!(supported.last().map(|sector| sector.id()), Some(254));
        assert!(std::panic::catch_unwind(|| {
            SU2FusionRule.fusion_channels(
                crate::SU2Irrep::from_twice_spin(128).sector_id(),
                crate::SU2Irrep::from_twice_spin(127).sector_id(),
            )
        })
        .is_err());
    }

    #[test]
    fn exact_authority_identity_rejects_legacy_su2_and_product_identity() {
        let legacy_su2 = RuleIdentity::of_type::<SU2FusionRule>();
        assert_ne!(SU2FusionRule.rule_identity(), legacy_su2);

        let current_product =
            ProductFusionRule::<U1FusionRule, SU2FusionRule>::new(U1FusionRule, SU2FusionRule);
        let legacy_product = RuleIdentity::compose_with_codec::<TensorKitProductCodec>(
            U1FusionRule.rule_identity(),
            RuleIdentity::of_type::<SU2FusionRule>(),
        );
        assert_ne!(current_product.rule_identity(), legacy_product);
    }
}
