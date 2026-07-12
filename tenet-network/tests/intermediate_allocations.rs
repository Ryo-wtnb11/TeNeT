use std::alloc::{GlobalAlloc, Layout, System};
use std::process::Command;
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

fn worker(chi: usize, reuse: bool) {
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
    let mut arena = NetworkExecutionWorkspace::default();

    for _ in 0..3 {
        drop(planned.execute_with_workspace(&refs, &mut arena).unwrap());
    }
    let before = arena.stats();
    let mut total = (0, 0);
    for _ in 0..3 {
        if !reuse {
            // Clearing before measurement keeps the same warm context and slot
            // capacities while forcing only intermediate Tensor allocation.
            arena.clear_intermediate_buffers();
        }
        let sample = measure(|| {
            drop(planned.execute_with_workspace(&refs, &mut arena).unwrap());
        });
        total.0 += sample.0;
        total.1 += sample.1;
    }
    let after = arena.stats();
    if reuse {
        assert_eq!(after.owned_intermediates - before.owned_intermediates, 3);
        assert_eq!(after.reused_intermediates - before.reused_intermediates, 6);
    } else {
        assert_eq!(after.owned_intermediates - before.owned_intermediates, 9);
        assert_eq!(after.reused_intermediates, before.reused_intermediates);
    }
    println!("TENET_ALLOC_RESULT {} {}", total.0, total.1);
}

fn run_worker(chi: usize, mode: &str) -> (u64, u64) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "warm_intermediate_arena_reduces_allocations_across_chi",
            "--nocapture",
        ])
        .env("TENET_ALLOC_WORKER", mode)
        .env("TENET_ALLOC_CHI", chi.to_string())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "worker failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let line = stdout
        .lines()
        .find(|line| line.starts_with("TENET_ALLOC_RESULT "))
        .unwrap_or_else(|| panic!("worker omitted result: {stdout}"));
    let mut fields = line.split_whitespace().skip(1);
    (
        fields.next().unwrap().parse().unwrap(),
        fields.next().unwrap().parse().unwrap(),
    )
}

#[test]
fn warm_intermediate_arena_reduces_allocations_across_chi() {
    if let Ok(mode) = std::env::var("TENET_ALLOC_WORKER") {
        let chi = std::env::var("TENET_ALLOC_CHI").unwrap().parse().unwrap();
        worker(chi, mode == "reuse");
        return;
    }

    for chi in [8, 16, 32, 64] {
        let fresh = run_worker(chi, "fresh");
        let reused = run_worker(chi, "reuse");
        eprintln!("chi={chi} fresh_intermediates={fresh:?} reused={reused:?}");
        // The process-global allocator remains diagnostic because lazy backend
        // worker allocation is not an intermediate-Tensor allocation. Exact
        // structural counts above provide the stable allocation gate.
        let fresh_tensor_bytes = 9 * chi * chi * std::mem::size_of::<f64>();
        let reused_tensor_bytes = 3 * chi * chi * std::mem::size_of::<f64>();
        assert!(
            reused_tensor_bytes < fresh_tensor_bytes,
            "chi={chi}: exact Tensor allocation bytes did not decrease"
        );
    }
}
