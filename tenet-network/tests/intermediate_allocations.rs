use std::alloc::{GlobalAlloc, Layout, System};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use tenet::prelude::*;
use tenet_network::{GreedyDenseOptimizer, Network, NetworkExecutionWorkspace, TemporaryLabel};

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PAYLOAD_SIZE: AtomicUsize = AtomicUsize::new(0);
static PAYLOAD_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static PAYLOAD_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PAYLOAD_PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
const REGISTRY_CAPACITY: usize = 1 << 16;
const TOMBSTONE: usize = usize::MAX;
static LIVE_POINTERS: [AtomicUsize; REGISTRY_CAPACITY] =
    [const { AtomicUsize::new(0) }; REGISTRY_CAPACITY];
static LIVE_SIZES: [AtomicUsize; REGISTRY_CAPACITY] =
    [const { AtomicUsize::new(0) }; REGISTRY_CAPACITY];
static REGISTRY_LOCK: AtomicBool = AtomicBool::new(false);
static TEST_LOCK: AtomicBool = AtomicBool::new(false);

fn lock_test() {
    while TEST_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        std::hint::spin_loop();
    }
}

fn unlock_test() {
    TEST_LOCK.store(false, Ordering::Release);
}

fn lock_registry() {
    while REGISTRY_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        std::hint::spin_loop();
    }
}

fn unlock_registry() {
    REGISTRY_LOCK.store(false, Ordering::Release);
}

fn pointer_hash(pointer: usize) -> usize {
    pointer.wrapping_mul(0x9e37_79b9_7f4a_7c15) & (REGISTRY_CAPACITY - 1)
}

fn register_live(pointer: *mut u8, size: usize) {
    lock_registry();
    let pointer = pointer as usize;
    let start = pointer_hash(pointer);
    for offset in 0..REGISTRY_CAPACITY {
        let index = (start + offset) & (REGISTRY_CAPACITY - 1);
        let current = LIVE_POINTERS[index].load(Ordering::Relaxed);
        if (current == 0 || current == TOMBSTONE)
            && LIVE_POINTERS[index]
                .compare_exchange(current, pointer, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            LIVE_SIZES[index].store(size, Ordering::Relaxed);
            add_live(size as u64);
            if size == PAYLOAD_SIZE.load(Ordering::Relaxed) {
                PAYLOAD_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                let live =
                    PAYLOAD_LIVE_BYTES.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
                PAYLOAD_PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
            }
            unlock_registry();
            return;
        }
    }
    unlock_registry();
}

fn unregister_live(pointer: *mut u8) -> Option<usize> {
    lock_registry();
    let pointer = pointer as usize;
    let start = pointer_hash(pointer);
    for offset in 0..REGISTRY_CAPACITY {
        let index = (start + offset) & (REGISTRY_CAPACITY - 1);
        let current = LIVE_POINTERS[index].load(Ordering::Relaxed);
        if current == 0 {
            unlock_registry();
            return None;
        }
        if current == pointer
            && LIVE_POINTERS[index]
                .compare_exchange(pointer, TOMBSTONE, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            let size = LIVE_SIZES[index].swap(0, Ordering::Relaxed);
            LIVE_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
            if size == PAYLOAD_SIZE.load(Ordering::Relaxed) {
                PAYLOAD_LIVE_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
            }
            unlock_registry();
            return Some(size);
        }
    }
    unlock_registry();
    None
}

fn reset_live_registry() {
    lock_registry();
    for index in 0..REGISTRY_CAPACITY {
        LIVE_POINTERS[index].store(0, Ordering::Relaxed);
        LIVE_SIZES[index].store(0, Ordering::Relaxed);
    }
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    PAYLOAD_ALLOC_CALLS.store(0, Ordering::Relaxed);
    PAYLOAD_LIVE_BYTES.store(0, Ordering::Relaxed);
    PAYLOAD_PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    unlock_registry();
}

fn add_live(bytes: u64) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            peak,
            live,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if ENABLED.load(Ordering::Relaxed) && !ptr.is_null() {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            register_live(ptr, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unregister_live(ptr);
        if ENABLED.load(Ordering::Relaxed) {
            DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            unregister_live(ptr);
            if ENABLED.load(Ordering::Relaxed) {
                REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
                DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
                register_live(new_ptr, new_size);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug)]
struct AllocationSample {
    alloc_calls: u64,
    realloc_calls: u64,
    dealloc_calls: u64,
    allocated_bytes: u64,
    deallocated_bytes: u64,
    peak_live_delta: u64,
    output_live_bytes: u64,
    retained_live_bytes: u64,
    payload_alloc_calls: u64,
    payload_peak_live_bytes: u64,
    payload_retained_live_bytes: u64,
    payload_output_live_bytes: u64,
}

fn measure_execute(
    planned: &tenet_network::PlannedNetwork,
    tensors: &[&Tensor],
    arena: &mut NetworkExecutionWorkspace,
    oracle: &Tensor,
) -> AllocationSample {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    REALLOC_CALLS.store(0, Ordering::Relaxed);
    DEALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    DEALLOCATED_BYTES.store(0, Ordering::Relaxed);
    PAYLOAD_SIZE.store(
        match oracle.dtype() {
            Dtype::F64 => oracle.data().len() * std::mem::size_of::<f64>(),
            Dtype::C64 => oracle.data_c64().len() * std::mem::size_of::<Complex64>(),
        },
        Ordering::Relaxed,
    );
    reset_live_registry();
    ENABLED.store(true, Ordering::SeqCst);

    let output = planned.execute_with_workspace(tensors, arena).unwrap();
    match output.dtype() {
        Dtype::F64 => assert_eq!(output.data(), oracle.data()),
        Dtype::C64 => assert_eq!(output.data_c64(), oracle.data_c64()),
    }
    let live_with_output = LIVE_BYTES.load(Ordering::Relaxed);
    let payload_live_with_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    let peak_live_delta = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    drop(output);
    let live_after_output = LIVE_BYTES.load(Ordering::Relaxed);
    let payload_live_after_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    ENABLED.store(false, Ordering::SeqCst);

    AllocationSample {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
        dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live_delta,
        output_live_bytes: live_with_output.saturating_sub(live_after_output),
        retained_live_bytes: live_after_output,
        payload_alloc_calls: PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed),
        payload_peak_live_bytes: PAYLOAD_PEAK_LIVE_BYTES.load(Ordering::Relaxed),
        payload_retained_live_bytes: PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed),
        payload_output_live_bytes: payload_live_with_output
            .saturating_sub(payload_live_after_output),
    }
}

#[derive(Clone, Copy)]
enum Workload {
    U1F64,
    U1C64,
    Su2F64,
    Su3C64,
    Su2PermuteF64,
}

impl Workload {
    const ALL: [Self; 5] = [
        Self::U1F64,
        Self::U1C64,
        Self::Su2F64,
        Self::Su3C64,
        Self::Su2PermuteF64,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::U1F64 => "u1-f64",
            Self::U1C64 => "u1-c64",
            Self::Su2F64 => "su2-f64",
            Self::Su3C64 => "su3-c64",
            Self::Su2PermuteF64 => "su2-permute-f64",
        }
    }

    fn parse(name: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|workload| workload.name() == name)
            .unwrap_or_else(|| panic!("unknown allocator workload {name:?}"))
    }

    fn dtype(self) -> Dtype {
        match self {
            Self::U1C64 | Self::Su3C64 => Dtype::C64,
            _ => Dtype::F64,
        }
    }

    fn space(self, chi: usize) -> Space {
        match self {
            Self::U1F64 | Self::U1C64 => Space::u1([(-1, chi), (0, chi), (1, chi)]),
            Self::Su2F64 | Self::Su2PermuteF64 => Space::su2([(0, chi), (1, chi), (2, chi)]),
            Self::Su3C64 => Space::su3([((1, 0), chi), ((0, 1), chi)]).unwrap(),
        }
    }

    fn permutes_intermediate(self) -> bool {
        matches!(self, Self::Su2PermuteF64)
    }
}

fn build_worker(workload: Workload, chi: usize) -> (tenet_network::PlannedNetwork, Vec<Tensor>) {
    let runtime = Runtime::builder().build().unwrap();
    let space = workload.space(chi);
    let count = if workload.permutes_intermediate() {
        3
    } else {
        4
    };
    let tensors = (0..count)
        .map(|index| {
            Tensor::rand_with_seed(
                &runtime,
                workload.dtype(),
                [&space],
                [&space],
                31_000 + chi as u64 * 100 + index as u64,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let label = |name: &str| TemporaryLabel::from(name);
    let labels = (0..count)
        .map(|index| {
            vec![
                label(&format!("l{index}")),
                label(&format!("l{}", index + 1)),
            ]
        })
        .collect::<Vec<_>>();
    let output = if workload.permutes_intermediate() {
        vec![label(&format!("l{count}")), label("l0")]
    } else {
        vec![label("l0"), label(&format!("l{count}"))]
    };
    let network = Network::new(
        labels,
        vec![false; count],
        vec![None; count],
        output,
        Some(1),
    )
    .unwrap();
    let refs = tensors.iter().collect::<Vec<_>>();
    let planned = network.plan(&refs, &GreedyDenseOptimizer).unwrap();
    (planned, tensors)
}

fn worker(workload: Workload, chi: usize, reuse: bool) {
    let (planned, tensors) = build_worker(workload, chi);
    let refs = tensors.iter().collect::<Vec<_>>();
    let mut arena = NetworkExecutionWorkspace::default();
    for _ in 0..3 {
        drop(planned.execute_with_workspace(&refs, &mut arena).unwrap());
    }
    for _ in 0..5 {
        if !reuse {
            arena.clear_intermediate_buffers();
        }
        drop(planned.execute_with_workspace(&refs, &mut arena).unwrap());
    }
    let oracle = planned.execute(&refs).unwrap();
    let structural_before = arena.stats();
    let mut samples = Vec::new();
    for _ in 0..3 {
        if !reuse {
            arena.clear_intermediate_buffers();
        }
        samples.push(measure_execute(&planned, &refs, &mut arena, &oracle));
    }
    let structural_after = arena.stats();
    let owned = structural_after.owned_intermediates - structural_before.owned_intermediates;
    let reused = structural_after.reused_intermediates - structural_before.reused_intermediates;
    let owned_contractions =
        structural_after.owned_contractions - structural_before.owned_contractions;
    let reused_contractions =
        structural_after.reused_contractions - structural_before.reused_contractions;
    let owned_orientations =
        structural_after.owned_orientations - structural_before.owned_orientations;
    let reused_orientations =
        structural_after.reused_orientations - structural_before.reused_orientations;
    assert_eq!(
        structural_after.escaped_outputs - structural_before.escaped_outputs,
        3
    );
    if workload.permutes_intermediate() {
        if reuse {
            assert_eq!((owned, reused), (3, 6));
            assert_eq!((owned_contractions, reused_contractions), (3, 3));
            assert_eq!((owned_orientations, reused_orientations), (0, 3));
        } else {
            assert_eq!((owned, reused), (9, 0));
            assert_eq!((owned_contractions, reused_contractions), (6, 0));
            assert_eq!((owned_orientations, reused_orientations), (3, 0));
        }
    } else if reuse {
        assert_eq!((owned, reused), (3, 6));
        assert_eq!((owned_contractions, reused_contractions), (3, 6));
        assert_eq!((owned_orientations, reused_orientations), (0, 0));
    } else {
        assert_eq!((owned, reused), (9, 0));
        assert_eq!((owned_contractions, reused_contractions), (9, 0));
        assert_eq!((owned_orientations, reused_orientations), (0, 0));
    }
    for sample in samples {
        println!(
            "TENET_ALLOC_SAMPLE {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            workload.name(),
            chi,
            u8::from(reuse),
            sample.alloc_calls,
            sample.realloc_calls,
            sample.dealloc_calls,
            sample.allocated_bytes,
            sample.deallocated_bytes,
            sample.peak_live_delta,
            sample.output_live_bytes,
            sample.retained_live_bytes,
            sample.payload_alloc_calls,
            sample.payload_peak_live_bytes,
            sample.payload_retained_live_bytes,
            sample.payload_output_live_bytes,
            owned * 1_000_000 + reused,
        );
    }
}

fn run_worker(workload: Workload, chi: usize, reuse: bool) -> Vec<AllocationSample> {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "measured_intermediate_arena_accounting",
            "--nocapture",
        ])
        .env("TENET_ALLOC_WORKLOAD", workload.name())
        .env("TENET_ALLOC_MODE", if reuse { "reuse" } else { "fresh" })
        .env("TENET_ALLOC_CHI", chi.to_string())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "worker failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter(|line| line.starts_with("TENET_ALLOC_SAMPLE "))
        .map(|line| {
            let mut fields = line.split_whitespace();
            assert_eq!(fields.next(), Some("TENET_ALLOC_SAMPLE"));
            assert_eq!(fields.next(), Some(workload.name()));
            assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), chi);
            assert_eq!(fields.next().unwrap(), if reuse { "1" } else { "0" });
            let values = fields
                .map(|value| value.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 13);
            let expected_structural = if reuse {
                if workload.permutes_intermediate() {
                    3_000_006
                } else {
                    3_000_006
                }
            } else {
                9_000_000
            };
            assert_eq!(values[12], expected_structural);
            AllocationSample {
                alloc_calls: values[0],
                realloc_calls: values[1],
                dealloc_calls: values[2],
                allocated_bytes: values[3],
                deallocated_bytes: values[4],
                peak_live_delta: values[5],
                output_live_bytes: values[6],
                retained_live_bytes: values[7],
                payload_alloc_calls: values[8],
                payload_peak_live_bytes: values[9],
                payload_retained_live_bytes: values[10],
                payload_output_live_bytes: values[11],
            }
        })
        .collect()
}

fn median3(mut values: [u64; 3]) -> u64 {
    values.sort_unstable();
    values[1]
}

#[test]
fn origin_registry_attributes_output_lifetime() {
    lock_test();
    const SIZE: usize = 4096;
    PAYLOAD_SIZE.store(SIZE, Ordering::Relaxed);
    reset_live_registry();
    ENABLED.store(true, Ordering::SeqCst);

    let output = std::hint::black_box(vec![7u8; SIZE]);
    let live_with_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    drop(output);
    let live_after_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    ENABLED.store(false, Ordering::SeqCst);

    assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(live_with_output, SIZE as u64);
    assert_eq!(live_after_output, 0);
    assert_eq!(live_with_output - live_after_output, SIZE as u64);
    unlock_test();
}

#[test]
fn measured_intermediate_arena_accounting() {
    lock_test();
    if let Ok(name) = std::env::var("TENET_ALLOC_WORKLOAD") {
        let workload = Workload::parse(&name);
        let chi = std::env::var("TENET_ALLOC_CHI").unwrap().parse().unwrap();
        let reuse = match std::env::var("TENET_ALLOC_MODE").unwrap().as_str() {
            "fresh" => false,
            "reuse" => true,
            mode => panic!("unknown allocator mode {mode:?}"),
        };
        worker(workload, chi, reuse);
        unlock_test();
        return;
    }

    for workload in Workload::ALL {
        for chi in [8, 16, 32, 64] {
            let fresh = run_worker(workload, chi, false);
            let reused = run_worker(workload, chi, true);
            assert_eq!(fresh.len(), 3);
            assert_eq!(reused.len(), 3);
            // Backend worker pools may perform one delayed scratch allocation.
            // The median rejects that single outlier, while payload-size
            // classification prevents scratch from satisfying the gate.
            let fresh_payload_allocs = median3([
                fresh[0].payload_alloc_calls,
                fresh[1].payload_alloc_calls,
                fresh[2].payload_alloc_calls,
            ]);
            let reused_payload_allocs = median3([
                reused[0].payload_alloc_calls,
                reused[1].payload_alloc_calls,
                reused[2].payload_alloc_calls,
            ]);
            let fresh_payload_peak = median3([
                fresh[0].payload_peak_live_bytes,
                fresh[1].payload_peak_live_bytes,
                fresh[2].payload_peak_live_bytes,
            ]);
            let reused_payload_peak = median3([
                reused[0].payload_peak_live_bytes,
                reused[1].payload_peak_live_bytes,
                reused[2].payload_peak_live_bytes,
            ]);
            let fresh_payload_retained = median3([
                fresh[0].payload_retained_live_bytes,
                fresh[1].payload_retained_live_bytes,
                fresh[2].payload_retained_live_bytes,
            ]);
            let reused_payload_retained = median3([
                reused[0].payload_retained_live_bytes,
                reused[1].payload_retained_live_bytes,
                reused[2].payload_retained_live_bytes,
            ]);
            eprintln!(
                "workload={} chi={chi} fresh={fresh:?} reused={reused:?}",
                workload.name()
            );
            assert!(fresh_payload_allocs > reused_payload_allocs);
            assert!(fresh_payload_retained > reused_payload_retained);
            assert!(fresh_payload_peak > reused_payload_peak);
            assert!(fresh
                .iter()
                .all(|sample| sample.payload_output_live_bytes > 0));
            assert!(reused
                .iter()
                .all(|sample| sample.payload_output_live_bytes > 0));
        }
    }
    unlock_test();
}
