use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tenet::prelude::*;
use tenet_network::{GreedyDenseOptimizer, Network, NetworkExecutionWorkspace, TemporaryLabel};

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure(operation: impl FnOnce()) -> (u64, u64) {
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::SeqCst);
    operation();
    ENABLED.store(false, Ordering::SeqCst);
    (CALLS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}

#[test]
fn warm_intermediate_arena_reduces_allocations_across_chi() {
    for chi in [8, 16, 32, 64] {
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, chi)]);
        let tensors = (0..4)
            .map(|index| {
                Tensor::rand_with_seed(
                    &runtime,
                    Dtype::F64,
                    [&space],
                    [&space],
                    31_000 + chi as u64 * 10 + index,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let label = |name: &str| TemporaryLabel::from(name);
        let network = Network::new(
            vec![
                vec![label("a"), label("b")],
                vec![label("b"), label("c")],
                vec![label("c"), label("d")],
                vec![label("d"), label("e")],
            ],
            vec![false; 4],
            vec![None; 4],
            vec![label("a"), label("e")],
            Some(1),
        )
        .unwrap();
        let refs = tensors.iter().collect::<Vec<_>>();
        let planned = network.plan(&refs, &GreedyDenseOptimizer).unwrap();
        let mut owned_arena = NetworkExecutionWorkspace::default();
        let mut reused_arena = NetworkExecutionWorkspace::default();

        drop(
            planned
                .execute_with_workspace(&refs, &mut owned_arena)
                .unwrap(),
        );
        drop(
            planned
                .execute_with_workspace(&refs, &mut reused_arena)
                .unwrap(),
        );
        drop(
            planned
                .execute_with_workspace(&refs, &mut reused_arena)
                .unwrap(),
        );
        let owned_stats_before = owned_arena.stats();
        let reused_stats_before = reused_arena.stats();

        let owned = measure(|| {
            for _ in 0..3 {
                drop(planned.execute(&refs).unwrap());
            }
        });
        let reused = measure(|| {
            for _ in 0..3 {
                drop(
                    planned
                        .execute_with_workspace(&refs, &mut reused_arena)
                        .unwrap(),
                );
            }
        });
        for _ in 0..3 {
            owned_arena.clear_intermediate_buffers();
            drop(
                planned
                    .execute_with_workspace(&refs, &mut owned_arena)
                    .unwrap(),
            );
        }
        let owned_stats = owned_arena.stats();
        let reused_stats = reused_arena.stats();
        eprintln!("chi={chi} owned={owned:?} reused={reused:?}");

        assert!(
            reused.0 < owned.0,
            "chi={chi}: allocation calls did not decrease: owned={owned:?}, reused={reused:?}"
        );
        assert!(
            reused.1 < owned.1,
            "chi={chi}: allocated bytes did not decrease: owned={owned:?}, reused={reused:?}"
        );
        assert_eq!(
            owned_stats.owned_intermediates - owned_stats_before.owned_intermediates,
            9
        );
        assert_eq!(
            owned_stats.reused_intermediates,
            owned_stats_before.reused_intermediates
        );
        assert_eq!(
            reused_stats.owned_intermediates - reused_stats_before.owned_intermediates,
            3,
            "the final escaping tensor remains owned once per replay"
        );
        assert_eq!(
            reused_stats.reused_intermediates - reused_stats_before.reused_intermediates,
            6
        );
    }
}
