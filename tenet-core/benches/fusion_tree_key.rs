//! Zero-cost-claim canary (#153): `FusionTreeKey` hash/eq on the
//! multiplicity-free path (`has_multiplicity == false`, every rule in this
//! crate today — see the big comment on `FusionTreeKey`'s `Hash` impl). No
//! timing assertion here: shared CI runners are too noisy for a
//! pass/fail latency gate. The gate is this bench compiling and running,
//! plus the `size_of` canary in `src/tests.rs`.

use std::hash::{Hash, Hasher};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustc_hash::FxHasher;
use tenet_core::{FusionTreeKey, SectorId};

fn mult_free_key() -> FusionTreeKey {
    FusionTreeKey::from_uncoupled((0..4).map(SectorId::new))
}

fn bench_hash_eq(c: &mut Criterion) {
    let a = mult_free_key();
    let b = mult_free_key();
    c.bench_function("fusion_tree_key_hash_mult_free", |bencher| {
        bencher.iter(|| {
            let mut state = FxHasher::default();
            black_box(&a).hash(&mut state);
            black_box(state.finish())
        })
    });
    c.bench_function("fusion_tree_key_eq_mult_free", |bencher| {
        bencher.iter(|| black_box(&a) == black_box(&b))
    });
}

criterion_group!(benches, bench_hash_eq);
criterion_main!(benches);
