use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, TensorMap};

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
    static LIVE_BYTES: Cell<i64> = const { Cell::new(0) };
    static PEAK_BYTES: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + layout.size() as u64);
            let live = LIVE_BYTES.get() + layout.size() as i64;
            LIVE_BYTES.set(live);
            PEAK_BYTES.set(PEAK_BYTES.get().max(live.max(0) as u64));
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ENABLED.get() {
            LIVE_BYTES.set(LIVE_BYTES.get() - layout.size() as i64);
        }
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            BYTES.set(BYTES.get() + new_size as u64);
            let live = LIVE_BYTES.get() - layout.size() as i64 + new_size as i64;
            LIVE_BYTES.set(live);
            PEAK_BYTES.set(PEAK_BYTES.get().max(live.max(0) as u64));
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

fn measure(f: impl FnOnce()) -> (u64, u64) {
    let (allocations, bytes, _) = measure_peak(f);
    (allocations, bytes)
}

fn measure_peak(f: impl FnOnce()) -> (u64, u64, u64) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    LIVE_BYTES.set(0);
    PEAK_BYTES.set(0);
    ENABLED.set(true);
    f();
    ENABLED.set(false);
    (ALLOCATIONS.get(), BYTES.get(), PEAK_BYTES.get())
}

fn tensor(
    runtime: &Runtime,
    sectors: impl IntoIterator<Item = (i32, usize)>,
    rank: usize,
) -> TensorMap<U1FusionRule, num_complex::Complex64> {
    assert_eq!(rank % 2, 0);
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new_with_arc(
        provider,
        sectors
            .into_iter()
            .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    TensorMap::rand_with_seed(
        runtime,
        std::iter::repeat_n(&space, rank / 2),
        std::iter::repeat_n(&space, rank / 2),
        261 + rank as u64,
    )
    .unwrap()
}

#[test]
fn adjoint_creation_allocates_metadata_not_a_receiver_sized_payload() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: the fallible typed constructor transactionally admits the logical
    // adjoint layout, so metadata may scale with rank and tree count. It must
    // not scale with degeneracy: that would mean copying the parent payload.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (rank, radius) in [(2, 0), (2, 2), (2, 6), (2, 12), (4, 4), (6, 1), (8, 0)] {
        let large_degeneracy = if rank == 2 { 16 } else { 2 };
        let small = tensor(&runtime, (-radius..=radius).map(|charge| (charge, 1)), rank);
        let large = tensor(
            &runtime,
            (-radius..=radius).map(|charge| (charge, large_degeneracy)),
            rank,
        );
        let small_layout = small.adjoint().unwrap();
        let small_cost = measure(|| {
            black_box(small.adjoint().unwrap());
        });
        let large_layout = large.adjoint().unwrap();
        let large_cost = measure(|| {
            black_box(large.adjoint().unwrap());
        });
        assert_eq!(
            small_cost, large_cost,
            "rank={rank}, sector radius={radius}: adjoint creation copied payload-sized state"
        );
        black_box((small_layout, large_layout));
    }
}

#[test]
fn adjoint_involution_does_not_allocate() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: the second dagger restores the parent body without allocating.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let source = tensor(&runtime, (-4..=4).map(|charge| (charge, 2)), 4);
    let adjoint = source.adjoint().unwrap();

    let cost = measure(|| {
        black_box(adjoint.adjoint().unwrap());
    });

    assert_eq!(cost, (0, 0));
}

#[test]
fn labelled_block_inspection_materializes_lazy_adjoint_once_and_borrows_it() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let parent = tensor(&runtime, [(0, 64)], 2);
    let parent_pointer = parent.data().as_ptr();
    let parent_block = parent.block(0).unwrap();
    let adjoint = parent.adjoint().unwrap();
    let payload_bytes =
        (parent.data().len() * std::mem::size_of::<num_complex::Complex64>()) as u64;

    let first = measure(|| {
        let blocks = adjoint.blocks().unwrap().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 1);
        let (_, values) = &blocks[0];
        assert_eq!(values.data().as_ptr(), adjoint.data().as_ptr());
        let expected = parent.data()
            [parent_block.offset() + 5 * parent_block.strides()[0] + 3 * parent_block.strides()[1]]
            .conj();
        assert_eq!(values.get(&[3, 5]).copied(), Some(expected));
    });
    assert!(
        first.1 >= payload_bytes,
        "block inspection did not materialize the lazy adjoint: {first:?}"
    );

    let second = measure(|| {
        let blocks = adjoint.blocks().unwrap().collect::<Vec<_>>();
        assert_eq!(blocks[0].1.data().as_ptr(), adjoint.data().as_ptr());
    });
    assert!(
        second.1 < payload_bytes,
        "block inspection rebuilt the adjoint payload: {second:?}"
    );
    assert_eq!(parent.data().as_ptr(), parent_pointer);
}

#[test]
fn typed_compact_svd_keeps_total_and_peak_below_materialized_baseline() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: the typed wrapper reuses the same parent-factor seam and does not
    // hide a receiver-sized logical-adjoint allocation around it.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 32)]).unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space], 693_695).unwrap();
    black_box(parent.svd_compact().unwrap());

    let input_bytes = (parent.data().len() * std::mem::size_of::<num_complex::Complex64>()) as u64;
    let optimized = parent.adjoint().unwrap();
    let baseline = parent.adjoint().unwrap();
    let optimized_cost = measure_peak(|| {
        black_box(optimized.svd_compact().unwrap());
    });
    let baseline_cost = measure_peak(|| {
        black_box(baseline.data());
        black_box(baseline.svd_compact().unwrap());
    });

    assert!(
        optimized_cost.1 < baseline_cost.1,
        "total bytes: optimized={optimized_cost:?}, baseline={baseline_cost:?}"
    );
    assert!(
        optimized_cost.2 < baseline_cost.2,
        "peak bytes: optimized={optimized_cost:?}, baseline={baseline_cost:?}"
    );
    assert!(
        measure(|| {
            black_box(optimized.data());
        })
        .1 >= input_bytes,
        "optimized compact SVD materialized its lazy input"
    );
    assert_eq!(
        measure(|| {
            black_box(baseline.data());
        }),
        (0, 0),
        "baseline materialization was not retained"
    );
}

#[test]
fn typed_full_svd_keeps_total_and_peak_below_materialized_baseline() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 32)]).unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space], 693_697).unwrap();
    black_box(parent.svd_full().unwrap());

    let input_bytes = (parent.data().len() * std::mem::size_of::<num_complex::Complex64>()) as u64;
    let optimized = parent.adjoint().unwrap();
    let baseline = parent.adjoint().unwrap();
    let optimized_cost = measure_peak(|| {
        black_box(optimized.svd_full().unwrap());
    });
    let baseline_cost = measure_peak(|| {
        black_box(baseline.data());
        black_box(baseline.svd_full().unwrap());
    });
    eprintln!("input={input_bytes} optimized={optimized_cost:?} materialized={baseline_cost:?}");

    assert!(
        optimized_cost.1 < baseline_cost.1,
        "total bytes: optimized={optimized_cost:?}, baseline={baseline_cost:?}"
    );
    assert!(
        optimized_cost.2 < baseline_cost.2,
        "peak bytes: optimized={optimized_cost:?}, baseline={baseline_cost:?}"
    );
    assert!(
        measure(|| {
            black_box(optimized.data());
        })
        .1 >= input_bytes,
        "optimized full SVD materialized its lazy input"
    );
    assert_eq!(
        measure(|| {
            black_box(baseline.data());
        }),
        (0, 0),
        "baseline materialization was not retained"
    );
}

#[test]
fn typed_truncated_svd_keeps_total_and_peak_below_materialized_baseline() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: typed truncation reuses the parent-factor seam without retaining
    // a receiver-sized logical-adjoint input.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 64)]).unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space], 693_696).unwrap();
    let truncation = tenet::typed::Truncation::rank(16);
    black_box(parent.svd_trunc(&truncation).unwrap());

    let input_bytes = (parent.data().len() * std::mem::size_of::<num_complex::Complex64>()) as u64;
    let optimized = parent.adjoint().unwrap();
    let baseline = parent.adjoint().unwrap();
    let optimized_cost = measure_peak(|| {
        black_box(optimized.svd_trunc(&truncation).unwrap());
    });
    let baseline_cost = measure_peak(|| {
        black_box(baseline.data());
        black_box(baseline.svd_trunc(&truncation).unwrap());
    });

    assert!(
        optimized_cost.1 < baseline_cost.1,
        "total bytes: optimized={optimized_cost:?}, baseline={baseline_cost:?}"
    );
    assert!(
        optimized_cost.2 < baseline_cost.2,
        "peak bytes: optimized={optimized_cost:?}, baseline={baseline_cost:?}"
    );
    assert!(
        measure(|| {
            black_box(optimized.data());
        })
        .1 >= input_bytes,
        "optimized truncated SVD materialized its lazy input"
    );
    assert_eq!(
        measure(|| {
            black_box(baseline.data());
        }),
        (0, 0),
        "baseline materialization was not retained"
    );
}

#[test]
fn lazy_scale_and_add_allocate_only_one_input_sized_payload() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space =
        GradedSpace::try_new_with_arc(provider, (-4..=4).map(|charge| (U1Irrep::new(charge), 8)))
            .unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_401).unwrap();
    let other_parent =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_402).unwrap();
    let owned = TensorMap::rand_with_seed(&runtime, [&space], [&space, &space], 666_403).unwrap();
    let payload_bytes = std::mem::size_of_val(parent.data()) as u64;
    let alpha = num_complex::Complex64::new(0.5, 0.0);
    let beta = num_complex::Complex64::new(-0.25, 0.0);

    let warm = parent.adjoint().unwrap();
    let warm_other = other_parent.adjoint().unwrap();
    black_box(warm.scale(alpha));
    black_box(warm.add(&owned, alpha, beta).unwrap());
    black_box(warm.add(&warm_other, alpha, beta).unwrap());

    let scale_lazy = parent.adjoint().unwrap();
    let mixed_lazy = parent.adjoint().unwrap();
    let pair_lazy = parent.adjoint().unwrap();
    let other_pair_lazy = other_parent.adjoint().unwrap();
    for (_, bytes) in [
        measure(|| {
            black_box(scale_lazy.scale(alpha));
        }),
        measure(|| {
            black_box(mixed_lazy.add(&owned, alpha, beta).unwrap());
        }),
        measure(|| {
            black_box(pair_lazy.add(&other_pair_lazy, alpha, beta).unwrap());
        }),
    ] {
        assert!(
            bytes >= payload_bytes && bytes < 2 * payload_bytes,
            "expected one payload allocation ({payload_bytes} bytes), observed {bytes} bytes"
        );
    }
    for lazy in [&scale_lazy, &mixed_lazy, &pair_lazy, &other_pair_lazy] {
        let (_, bytes) = measure(|| {
            black_box(lazy.data().len());
        });
        assert!(
            bytes >= payload_bytes,
            "the operation materialized its lazy operand"
        );
    }
}

#[test]
fn mixed_lazy_add_has_no_rank_dependent_stride_allocation() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut reference = None;
    for rank in [8, 10, 12] {
        let provider = Arc::new(U1FusionRule);
        let space = GradedSpace::try_new_with_arc(provider, [(U1Irrep::new(0), 1)]).unwrap();
        let codomain = std::iter::repeat_n(&space, rank / 2).collect::<Vec<_>>();
        let parent: TensorMap<_, num_complex::Complex64> = TensorMap::rand_with_seed(
            &runtime,
            codomain.clone(),
            codomain.clone(),
            666_500 + rank as u64,
        )
        .unwrap();
        let owned =
            TensorMap::rand_with_seed(&runtime, codomain.clone(), codomain, 666_600 + rank as u64)
                .unwrap();
        let alpha = num_complex::Complex64::new(0.5, 0.0);
        let beta = num_complex::Complex64::new(-0.25, 0.0);
        black_box(parent.adjoint().unwrap().add(&owned, alpha, beta).unwrap());
        let lazy = parent.adjoint().unwrap();
        let cost = measure(|| {
            black_box(lazy.add(&owned, alpha, beta).unwrap());
        });
        assert_eq!(
            cost,
            *reference.get_or_insert(cost),
            "rank={rank} must not heap-allocate stride metadata"
        );
        assert!(
            measure(|| {
                black_box(lazy.data().len());
            })
            .0 > 0,
            "the mixed add materialized its lazy operand"
        );
    }
}

#[test]
fn identity_adjoint_transform_cost_is_independent_of_rank_and_block_count() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: identity permute, braid, and repartition share the lazy view with
    // zero allocations across increasing rank and U1 sector counts.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (rank, radius) in [(2, 0), (2, 6), (4, 4), (6, 1), (8, 0), (10, 0)] {
        let source = tensor(&runtime, (-radius..=radius).map(|charge| (charge, 2)), rank);
        let adjoint = source.adjoint().unwrap();
        let split = adjoint.codomain_rank();
        let codomain_axes = (0..split).collect::<Vec<_>>();
        let domain_axes = (split..rank).collect::<Vec<_>>();
        let levels = (0..rank).map(|axis| rank - axis).collect::<Vec<_>>();

        for cost in [
            measure(|| {
                black_box(adjoint.permute(&codomain_axes, &domain_axes).unwrap());
            }),
            measure(|| {
                black_box(
                    adjoint
                        .braid(&codomain_axes, &domain_axes, &levels)
                        .unwrap(),
                );
            }),
            measure(|| {
                black_box(adjoint.repartition(split).unwrap());
            }),
        ] {
            assert_eq!(cost, (0, 0), "rank={rank}, sector radius={radius}");
        }
    }
}

#[test]
fn ordinary_tensor_clone_does_not_allocate() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: the representation split keeps an owned tensor's value-like Arc clone cost.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let source = tensor(&runtime, (-4..=4).map(|charge| (charge, 2)), 4);

    let cost = measure(|| {
        black_box(source.clone());
    });

    assert_eq!(cost, (0, 0));
}

fn measure_eager_lazy_core_compose(
    rows: &[(i32, usize)],
    contracted: &[(i32, usize)],
    cols: &[(i32, usize)],
    seed: u64,
) -> u64 {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let make_space = |sectors: &[(i32, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            sectors
                .iter()
                .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
        )
        .unwrap()
    };
    let rows = make_space(rows);
    let contracted = make_space(contracted);
    let cols = make_space(cols);
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&contracted], [&rows], seed).unwrap();
    let rhs = TensorMap::rand_with_seed(&runtime, [&contracted], [&cols], seed + 1).unwrap();
    let lhs = parent.adjoint().unwrap();
    black_box(lhs.compose(&rhs).unwrap());
    measure(|| {
        black_box(lhs.compose(&rhs).unwrap());
    })
    .0
}

#[test]
fn typed_single_group_lazy_compose_stays_below_the_measured_engine_margin() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: warmed typed lazy-adjoint compose measured 40 allocations when
    // admitted here; 64 leaves platform headroom without inheriting the erased
    // facade's former 128-allocation ceiling.
    let calls = measure_eager_lazy_core_compose(&[(0, 3)], &[(0, 2)], &[(0, 4)], 272_001);
    assert!(calls <= 64, "typed lazy compose allocated {calls} times");
}

#[test]
fn typed_multigroup_lazy_compose_stays_below_the_measured_engine_margin() {
    let _measurement = MEASUREMENT_LOCK.lock().unwrap();
    // What: the corresponding multigroup route measured 52 allocations; 80
    // keeps explicit headroom while still killing a return to per-term replay.
    let calls = measure_eager_lazy_core_compose(
        &[(-1, 2), (0, 3), (1, 1)],
        &[(-1, 1), (0, 2), (1, 3)],
        &[(-1, 3), (0, 1), (1, 2)],
        272_011,
    );
    assert!(
        calls <= 80,
        "typed multigroup compose allocated {calls} times"
    );
}
