use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Mutex, MutexGuard};

use tenet::prelude::*;
use tenet_network::{GreedyDenseOptimizer, Network, NetworkExecutionWorkspace, TemporaryLabel};

struct CountingAllocator;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static PROBE_THREAD_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static PROBE_THREAD_ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static PROBE_THREAD_REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PAYLOAD_SIZE: AtomicUsize = AtomicUsize::new(0);
static PAYLOAD_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static PAYLOAD_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PAYLOAD_PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static REGISTRY_OVERFLOWS: AtomicU64 = AtomicU64::new(0);
const REGISTRY_CAPACITY: usize = 1 << 16;
const TOMBSTONE: usize = usize::MAX;
static LIVE_POINTERS: [AtomicUsize; REGISTRY_CAPACITY] =
    [const { AtomicUsize::new(0) }; REGISTRY_CAPACITY];
static LIVE_SIZES: [AtomicUsize; REGISTRY_CAPACITY] =
    [const { AtomicUsize::new(0) }; REGISTRY_CAPACITY];
static REGISTRY_LOCK: AtomicBool = AtomicBool::new(false);
static TEST_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static PROBE_THREAD_ENABLED: Cell<bool> = const { Cell::new(false) };
    static PROBE_THREAD_DEALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
    static PROBE_THREAD_DEALLOCATED_BYTES: Cell<u64> = const { Cell::new(0) };
    static DEALLOC_BOUNDARY_HOOK: RefCell<Option<DeallocBoundaryHook>> = const { RefCell::new(None) };
    #[cfg(test)]
    static REALLOC_TRANSITION_HOOK: RefCell<Option<DeallocBoundaryHook>> = const { RefCell::new(None) };
}

struct DeallocBoundaryHook {
    reached: SyncSender<()>,
    resume: Receiver<()>,
}

#[cfg(test)]
fn cross_realloc_transition_hook() {
    let hook = REALLOC_TRANSITION_HOOK
        .try_with(|slot| slot.borrow_mut().take())
        .ok()
        .flatten();
    if let Some(hook) = hook {
        hook.reached.send(()).unwrap();
        hook.resume.recv().unwrap();
    }
}

struct RegistryGuard;

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        REGISTRY_LOCK.store(false, Ordering::Release);
    }
}

fn lock_unpoisoned(mutex: &Mutex<()>) -> MutexGuard<'_, ()> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lock_registry() -> RegistryGuard {
    while REGISTRY_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        std::hint::spin_loop();
    }
    RegistryGuard
}

fn pointer_hash(pointer: usize, capacity: usize) -> usize {
    pointer.wrapping_mul(0x9e37_79b9_7f4a_7c15) % capacity
}

fn insert_live_with_capacity(
    pointer: *mut u8,
    size: usize,
    capacity: usize,
    account_live: bool,
    count_payload_origin: bool,
) -> bool {
    if pointer.is_null()
        || pointer as usize == TOMBSTONE
        || size == 0
        || capacity == 0
        || capacity > REGISTRY_CAPACITY
    {
        return false;
    }
    let _guard = lock_registry();
    let pointer = pointer as usize;
    let start = pointer_hash(pointer, capacity);
    let mut first_available = None;
    for offset in 0..capacity {
        let index = (start + offset) % capacity;
        let current = LIVE_POINTERS[index].load(Ordering::Relaxed);
        if current == pointer {
            return true;
        }
        if current == TOMBSTONE {
            first_available.get_or_insert(index);
            continue;
        }
        if current == 0 {
            let index = first_available.unwrap_or(index);
            LIVE_POINTERS[index].store(pointer, Ordering::Relaxed);
            LIVE_SIZES[index].store(size, Ordering::Relaxed);
            if account_live {
                add_live(size as u64);
            }
            if account_live && size == PAYLOAD_SIZE.load(Ordering::Relaxed) {
                if count_payload_origin {
                    PAYLOAD_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                }
                let live =
                    PAYLOAD_LIVE_BYTES.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
                PAYLOAD_PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
            }
            return true;
        }
    }
    if let Some(index) = first_available {
        LIVE_POINTERS[index].store(pointer, Ordering::Relaxed);
        LIVE_SIZES[index].store(size, Ordering::Relaxed);
        if account_live {
            add_live(size as u64);
        }
        if account_live && size == PAYLOAD_SIZE.load(Ordering::Relaxed) {
            if count_payload_origin {
                PAYLOAD_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            let live = PAYLOAD_LIVE_BYTES.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
            PAYLOAD_PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        return true;
    }
    REGISTRY_OVERFLOWS.fetch_add(1, Ordering::Relaxed);
    false
}

fn register_live_with_capacity(pointer: *mut u8, size: usize, capacity: usize) -> bool {
    insert_live_with_capacity(pointer, size, capacity, true, true)
}

fn register_live(pointer: *mut u8, size: usize) -> bool {
    register_live_with_capacity(pointer, size, REGISTRY_CAPACITY)
}

fn insert_live_without_accounting(pointer: *mut u8, size: usize) -> bool {
    // Why not call register_live: restoring a failed realloc revives the same
    // allocation origin and must not report a second payload allocation.
    insert_live_with_capacity(pointer, size, REGISTRY_CAPACITY, false, false)
}

fn take_live_with_capacity(pointer: *mut u8, capacity: usize, release_live: bool) -> Option<usize> {
    if pointer.is_null()
        || pointer as usize == TOMBSTONE
        || capacity == 0
        || capacity > REGISTRY_CAPACITY
    {
        return None;
    }
    let _guard = lock_registry();
    let pointer = pointer as usize;
    let start = pointer_hash(pointer, capacity);
    for offset in 0..capacity {
        let index = (start + offset) % capacity;
        let current = LIVE_POINTERS[index].load(Ordering::Relaxed);
        if current == 0 {
            return None;
        }
        if current == pointer
            && LIVE_POINTERS[index]
                .compare_exchange(pointer, TOMBSTONE, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            let size = LIVE_SIZES[index].swap(0, Ordering::Relaxed);
            if release_live {
                LIVE_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
                if size == PAYLOAD_SIZE.load(Ordering::Relaxed) {
                    PAYLOAD_LIVE_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
                }
            }
            return Some(size);
        }
    }
    None
}

fn unregister_live_with_capacity(pointer: *mut u8, capacity: usize) -> Option<usize> {
    take_live_with_capacity(pointer, capacity, true)
}

fn unregister_live(pointer: *mut u8) -> Option<usize> {
    unregister_live_with_capacity(pointer, REGISTRY_CAPACITY)
}

fn registered_size(pointer: *const u8) -> Option<usize> {
    if pointer.is_null() || pointer as usize == TOMBSTONE {
        return None;
    }
    let _guard = lock_registry();
    let pointer = pointer as usize;
    let start = pointer_hash(pointer, REGISTRY_CAPACITY);
    for offset in 0..REGISTRY_CAPACITY {
        let index = (start + offset) % REGISTRY_CAPACITY;
        let current = LIVE_POINTERS[index].load(Ordering::Relaxed);
        if current == 0 {
            return None;
        }
        if current == pointer {
            return Some(LIVE_SIZES[index].load(Ordering::Relaxed));
        }
    }
    None
}

fn reset_live_registry() {
    let _guard = lock_registry();
    for index in 0..REGISTRY_CAPACITY {
        LIVE_POINTERS[index].store(0, Ordering::Relaxed);
        LIVE_SIZES[index].store(0, Ordering::Relaxed);
    }
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    PAYLOAD_ALLOC_CALLS.store(0, Ordering::Relaxed);
    PAYLOAD_LIVE_BYTES.store(0, Ordering::Relaxed);
    PAYLOAD_PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    REGISTRY_OVERFLOWS.store(0, Ordering::Relaxed);
}

fn reset_event_counters() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    REALLOC_CALLS.store(0, Ordering::Relaxed);
    DEALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    DEALLOCATED_BYTES.store(0, Ordering::Relaxed);
    PROBE_THREAD_ALLOC_CALLS.store(0, Ordering::Relaxed);
    PROBE_THREAD_ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    PROBE_THREAD_REALLOC_CALLS.store(0, Ordering::Relaxed);
}

#[derive(Clone, Copy)]
struct DetachedReallocOrigin {
    pointer: *mut u8,
    size: usize,
}

fn detach_realloc_origin(pointer: *mut u8) -> Option<DetachedReallocOrigin> {
    // Address identity is removed before System.realloc, but its live metrics
    // remain reserved until the allocator reports success or failure.
    take_live_with_capacity(pointer, REGISTRY_CAPACITY, false)
        .map(|size| DetachedReallocOrigin { pointer, size })
}

fn finish_realloc_result(
    origin: DetachedReallocOrigin,
    new_ptr: *mut u8,
    new_size: usize,
    count_event: bool,
) -> bool {
    if new_ptr.is_null() {
        insert_live_without_accounting(origin.pointer, origin.size);
        return false;
    }
    if !insert_live_without_accounting(new_ptr, new_size) {
        LIVE_BYTES.fetch_sub(origin.size as u64, Ordering::Relaxed);
        if origin.size == PAYLOAD_SIZE.load(Ordering::Relaxed) {
            PAYLOAD_LIVE_BYTES.fetch_sub(origin.size as u64, Ordering::Relaxed);
        }
        return false;
    }

    // Why not subtract then register: a concurrent allocation can complete in
    // that zero-live gap and permanently understate the peak. Publish the new
    // generation first, then replace the reserved old metrics by their delta.
    #[cfg(test)]
    cross_realloc_transition_hook();
    if new_size >= origin.size {
        add_live((new_size - origin.size) as u64);
    } else {
        LIVE_BYTES.fetch_sub((origin.size - new_size) as u64, Ordering::Relaxed);
    }
    let payload_size = PAYLOAD_SIZE.load(Ordering::Relaxed);
    if origin.size == payload_size && new_size != payload_size {
        PAYLOAD_LIVE_BYTES.fetch_sub(origin.size as u64, Ordering::Relaxed);
    } else if origin.size != payload_size && new_size == payload_size {
        PAYLOAD_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        let live =
            PAYLOAD_LIVE_BYTES.fetch_add(new_size as u64, Ordering::Relaxed) + new_size as u64;
        PAYLOAD_PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
    } else if new_size == payload_size {
        PAYLOAD_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    if count_event {
        REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(origin.size as u64, Ordering::Relaxed);
    }
    true
}

fn record_dealloc_result(pointer: *mut u8) -> bool {
    let Some(size) = unregister_live(pointer) else {
        return false;
    };
    // Why not gate unregistering: probe-origin storage can outlive the measurement
    // window, and leaving it registered corrupts retained-live accounting.
    // Why not use infallible TLS access: the allocator also observes frees
    // performed while a worker thread's TLS values are being destroyed.
    let boundary_hook = DEALLOC_BOUNDARY_HOOK
        .try_with(|slot| slot.borrow_mut().take())
        .ok()
        .flatten();
    if let Some(hook) = boundary_hook {
        // Why not leave the hook installed: synchronization may enter the allocator,
        // so the one-shot hook must be removed before crossing the test boundary.
        hook.reached.send(()).unwrap();
        hook.resume.recv().unwrap();
    }
    let count_event = ENABLED.load(Ordering::Relaxed);
    if count_event {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        if PROBE_THREAD_ENABLED.try_with(Cell::get).unwrap_or(false) {
            let _ = PROBE_THREAD_DEALLOC_CALLS.try_with(|calls| calls.set(calls.get() + 1));
            let _ = PROBE_THREAD_DEALLOCATED_BYTES
                .try_with(|bytes| bytes.set(bytes.get() + size as u64));
        }
    }
    true
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
        if ENABLED.load(Ordering::Relaxed) && !ptr.is_null() && layout.size() != 0 {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            if PROBE_THREAD_ENABLED.get() {
                PROBE_THREAD_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                PROBE_THREAD_ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            register_live(ptr, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_dealloc_result(ptr);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // The origin is detached before System can release its address. Why not
        // hold REGISTRY_LOCK across System.realloc: allocator reentrancy would
        // deadlock every registry operation on this thread.
        let origin = detach_realloc_origin(ptr);
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        let enabled = ENABLED.load(Ordering::Relaxed);
        if let Some(origin) = origin {
            finish_realloc_result(origin, new_ptr, new_size, enabled);
        }
        if enabled && !new_ptr.is_null() && PROBE_THREAD_ENABLED.get() {
            PROBE_THREAD_REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
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
    payload_size_bytes: u64,
    registry_overflows: u64,
}

fn measure_execute(
    planned: &tenet_network::PlannedNetwork,
    tensors: &[&Tensor],
    arena: &mut NetworkExecutionWorkspace,
    oracle: &Tensor,
) -> AllocationSample {
    reset_event_counters();
    let payload_size_bytes = match oracle.dtype() {
        Dtype::F64 => oracle.data().len().checked_mul(std::mem::size_of::<f64>()),
        Dtype::C64 => oracle
            .data_c64()
            .len()
            .checked_mul(std::mem::size_of::<Complex64>()),
    };
    let payload_size_bytes = payload_size_bytes.expect("oracle payload byte size overflowed");
    assert!(payload_size_bytes > 0);
    PAYLOAD_SIZE.store(payload_size_bytes, Ordering::Relaxed);
    reset_live_registry();
    ENABLED.store(true, Ordering::SeqCst);

    let output = planned.execute_with_workspace(tensors, arena).unwrap();
    match output.dtype() {
        Dtype::F64 => assert_eq!(output.data(), oracle.data()),
        Dtype::C64 => assert_eq!(output.data_c64(), oracle.data_c64()),
    }
    let (output_pointer, output_payload_bytes) = match output.dtype() {
        Dtype::F64 => (
            output.data().as_ptr().cast::<u8>(),
            output.data().len().checked_mul(std::mem::size_of::<f64>()),
        ),
        Dtype::C64 => (
            output.data_c64().as_ptr().cast::<u8>(),
            output
                .data_c64()
                .len()
                .checked_mul(std::mem::size_of::<Complex64>()),
        ),
    };
    assert_eq!(output_payload_bytes, Some(payload_size_bytes));
    assert_eq!(registered_size(output_pointer), Some(payload_size_bytes));
    let live_with_output = LIVE_BYTES.load(Ordering::Relaxed);
    let payload_live_with_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    let peak_live_delta = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    drop(output);
    assert_eq!(registered_size(output_pointer), None);
    let live_after_output = LIVE_BYTES.load(Ordering::Relaxed);
    let payload_live_after_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    ENABLED.store(false, Ordering::SeqCst);
    assert_eq!(REGISTRY_OVERFLOWS.load(Ordering::Relaxed), 0);

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
        payload_size_bytes: payload_size_bytes as u64,
        registry_overflows: REGISTRY_OVERFLOWS.load(Ordering::Relaxed),
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
        structural_after.contract_layout_preparations
            - structural_before.contract_layout_preparations,
        0
    );
    assert_eq!(
        structural_after.orientation_layout_preparations
            - structural_before.orientation_layout_preparations,
        0
    );
    assert_eq!(
        structural_after.contract_structural_comparisons
            - structural_before.contract_structural_comparisons,
        0
    );
    assert_eq!(
        structural_after.orientation_structural_comparisons
            - structural_before.orientation_structural_comparisons,
        0
    );
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
            "TENET_ALLOC_SAMPLE {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
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
            sample.payload_size_bytes,
            sample.registry_overflows,
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
            assert_eq!(values.len(), 15);
            let expected_structural = if reuse {
                if workload.permutes_intermediate() {
                    3_000_006
                } else {
                    3_000_006
                }
            } else {
                9_000_000
            };
            assert_eq!(values[14], expected_structural);
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
                payload_size_bytes: values[12],
                registry_overflows: values[13],
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
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    const SIZE: usize = 4096;
    PAYLOAD_SIZE.store(SIZE, Ordering::Relaxed);
    reset_live_registry();
    ENABLED.store(true, Ordering::SeqCst);

    let decoy = std::hint::black_box(vec![3u8; SIZE]);
    let output = std::hint::black_box(vec![7u8; SIZE]);
    let output_pointer = output.as_ptr();
    let decoy_pointer = decoy.as_ptr();
    assert_eq!(registered_size(output_pointer), Some(SIZE));
    assert_eq!(registered_size(decoy_pointer), Some(SIZE));
    let live_with_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    drop(output);
    assert_eq!(registered_size(output_pointer), None);
    assert_eq!(registered_size(decoy_pointer), Some(SIZE));
    let live_after_output = PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed);
    ENABLED.store(false, Ordering::SeqCst);

    assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 2);
    assert_eq!(live_with_output, (2 * SIZE) as u64);
    assert_eq!(live_after_output, SIZE as u64);
    assert_eq!(live_with_output - live_after_output, SIZE as u64);
    drop(decoy);
}

#[test]
fn realloc_moved_transition_preserves_concurrently_reused_old_address() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    reset_event_counters();
    reset_live_registry();
    ENABLED.store(false, Ordering::SeqCst);
    let old = 0x1000usize as *mut u8;
    let moved = 0x2000usize as *mut u8;
    assert!(register_live(old, 8));

    let origin = detach_realloc_origin(old).expect("old origin must be tracked");
    // What: detaching address identity reserves the live origin metrics until
    // the allocator reports whether the realloc succeeded.
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 8);
    let old_address = old as usize;
    // What: another allocator thread may reuse the freed address before the moved
    // realloc result is committed to the registry.
    assert!(
        std::thread::spawn(move || register_live(old_address as *mut u8, 32))
            .join()
            .unwrap()
    );
    assert!(finish_realloc_result(origin, moved, 16, true));

    assert_eq!(registered_size(old), Some(32));
    assert_eq!(registered_size(moved), Some(16));
    assert_eq!(REALLOC_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(ALLOCATED_BYTES.load(Ordering::Relaxed), 16);
    assert_eq!(DEALLOCATED_BYTES.load(Ordering::Relaxed), 8);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 48);
}

#[test]
fn realloc_failed_transition_restores_origin_without_duplicate_metrics() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    reset_event_counters();
    PAYLOAD_SIZE.store(8, Ordering::Relaxed);
    reset_live_registry();
    let old = 0x1000usize as *mut u8;
    assert!(register_live(old, 8));
    let origin = detach_realloc_origin(old).expect("old origin must be tracked");
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 8);
    assert_eq!(PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed), 8);

    // What: a failed realloc restores the exact old origin without reporting a
    // second allocation or a successful realloc event.
    assert!(!finish_realloc_result(
        origin,
        std::ptr::null_mut(),
        16,
        true
    ));
    assert_eq!(registered_size(old), Some(8));
    assert_eq!(REALLOC_CALLS.load(Ordering::Relaxed), 0);
    assert_eq!(ALLOCATED_BYTES.load(Ordering::Relaxed), 0);
    assert_eq!(DEALLOCATED_BYTES.load(Ordering::Relaxed), 0);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 8);
    assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed), 8);
}

#[test]
fn realloc_in_place_transition_replaces_only_its_detached_origin() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    reset_event_counters();
    PAYLOAD_SIZE.store(0, Ordering::Relaxed);
    reset_live_registry();
    let pointer = 0x1000usize as *mut u8;
    assert!(register_live(pointer, 8));
    let origin = detach_realloc_origin(pointer).expect("old origin must be tracked");

    // What: an in-place realloc replaces its own generation with the new size.
    assert!(finish_realloc_result(origin, pointer, 4, true));
    assert_eq!(registered_size(pointer), Some(4));
    assert_eq!(REALLOC_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(ALLOCATED_BYTES.load(Ordering::Relaxed), 4);
    assert_eq!(DEALLOCATED_BYTES.load(Ordering::Relaxed), 8);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 4);
}

#[test]
fn realloc_transition_never_exposes_a_zero_live_metrics_window() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    reset_event_counters();
    PAYLOAD_SIZE.store(0, Ordering::Relaxed);
    reset_live_registry();
    let old = 0x1000usize as *mut u8;
    let moved = 0x2000usize as *mut u8;
    let concurrent = 0x3000usize;
    assert!(register_live(old, 8));
    let origin = detach_realloc_origin(old).expect("old origin must be tracked");
    let (reached_tx, reached_rx) = mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = mpsc::sync_channel(0);
    REALLOC_TRANSITION_HOOK.with_borrow_mut(|slot| {
        *slot = Some(DeallocBoundaryHook {
            reached: reached_tx,
            resume: resume_rx,
        });
    });
    let worker = std::thread::spawn(move || {
        reached_rx.recv().unwrap();
        // What: a complete concurrent allocation lifetime overlaps the realloc
        // transition and must overlap either its old or new live metrics.
        let pointer = concurrent as *mut u8;
        assert!(register_live(pointer, 64));
        assert_eq!(unregister_live(pointer), Some(64));
        resume_tx.send(()).unwrap();
    });

    assert!(finish_realloc_result(origin, moved, 16, true));
    worker.join().unwrap();
    assert_eq!(registered_size(moved), Some(16));
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 16);
    assert_eq!(PEAK_LIVE_BYTES.load(Ordering::Relaxed), 72);
}

#[test]
fn dealloc_counts_only_enabled_probe_origin() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    ENABLED.store(false, Ordering::SeqCst);
    reset_event_counters();
    reset_live_registry();
    PROBE_THREAD_DEALLOC_CALLS.set(0);
    PROBE_THREAD_DEALLOCATED_BYTES.set(0);

    let untracked = std::hint::black_box(Box::new([9u8; 128]));
    PROBE_THREAD_ENABLED.set(true);
    ENABLED.store(true, Ordering::SeqCst);

    // What: freeing memory allocated before a probe never enters its counters.
    drop(untracked);
    assert_eq!(PROBE_THREAD_DEALLOC_CALLS.get(), 0);
    assert_eq!(PROBE_THREAD_DEALLOCATED_BYTES.get(), 0);

    let tracked = std::hint::black_box(Box::new([7u8; 256]));
    PROBE_THREAD_DEALLOC_CALLS.set(0);
    PROBE_THREAD_DEALLOCATED_BYTES.set(0);

    // What: a real probe-origin free records its exact allocation size.
    drop(tracked);
    ENABLED.store(false, Ordering::SeqCst);
    PROBE_THREAD_ENABLED.set(false);
    assert_eq!(PROBE_THREAD_DEALLOC_CALLS.get(), 1);
    assert_eq!(PROBE_THREAD_DEALLOCATED_BYTES.get(), 256);
}

#[test]
fn dealloc_snapshots_probe_state_after_unregistering_origin() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    ENABLED.store(true, Ordering::SeqCst);
    reset_live_registry();
    let tracked = std::hint::black_box(Box::new([5u8; 64]));
    let (reached_tx, reached_rx) = mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = mpsc::sync_channel(0);

    let worker = std::thread::spawn(move || {
        PROBE_THREAD_DEALLOC_CALLS.set(0);
        PROBE_THREAD_DEALLOCATED_BYTES.set(0);
        PROBE_THREAD_ENABLED.set(true);
        DEALLOC_BOUNDARY_HOOK.with_borrow_mut(|slot| {
            *slot = Some(DeallocBoundaryHook {
                reached: reached_tx,
                resume: resume_rx,
            });
        });
        drop(tracked);
        DEALLOC_BOUNDARY_HOOK.with_borrow_mut(Option::take);
        PROBE_THREAD_ENABLED.set(false);
        (
            PROBE_THREAD_DEALLOC_CALLS.get(),
            PROBE_THREAD_DEALLOCATED_BYTES.get(),
        )
    });

    // What: disabling the probe after unregister but before attribution excludes the free.
    reached_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("deallocation did not expose its unregister boundary");
    ENABLED.store(false, Ordering::SeqCst);
    resume_tx.send(()).unwrap();
    assert_eq!(worker.join().unwrap(), (0, 0));
}

#[test]
fn registry_overflow_invalidates_a_bounded_probe() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    reset_live_registry();
    assert!(register_live_with_capacity(0x1000usize as *mut u8, 8, 2));
    assert!(register_live_with_capacity(0x2000usize as *mut u8, 8, 2));
    assert!(!register_live_with_capacity(0x3000usize as *mut u8, 8, 2));
    assert_eq!(REGISTRY_OVERFLOWS.load(Ordering::Relaxed), 1);
}

#[test]
fn registry_rejects_zero_sentinels_and_deduplicates_pointers() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    PAYLOAD_SIZE.store(64, Ordering::Relaxed);
    reset_live_registry();
    let pointer = 0x1000usize as *mut u8;

    assert!(!register_live(std::ptr::null_mut(), 64));
    assert!(!register_live(TOMBSTONE as *mut u8, 64));
    assert!(!register_live(pointer, 0));
    assert!(!register_live_with_capacity(pointer, 64, 0));
    assert!(!register_live_with_capacity(
        pointer,
        64,
        REGISTRY_CAPACITY + 1
    ));
    assert_eq!(unregister_live_with_capacity(pointer, 0), None);
    assert_eq!(
        unregister_live_with_capacity(pointer, REGISTRY_CAPACITY + 1),
        None
    );
    assert!(register_live(pointer, 64));
    assert!(register_live(pointer, 64));
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 64);
    assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(unregister_live(pointer), Some(64));
    assert_eq!(unregister_live(pointer), None);
    reset_live_registry();
    assert!(register_live_with_capacity(pointer, 64, 1));
    assert_eq!(unregister_live_with_capacity(pointer, 1), Some(64));
    assert!(register_live_with_capacity(0x2000usize as *mut u8, 32, 1));
}

#[test]
fn rank_nine_cached_permutation_has_no_caller_thread_operation_allocation() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(0, 1)]);
    let source = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space; 9], [], 31_901).unwrap();
    let axes = [8, 7, 6, 5, 4, 3, 2, 1, 0];
    let expected = source.permute(&axes, &[]).unwrap();
    let mut destination = expected.scale(f64::NAN).unwrap();
    let mut context = TensorExecutionContext::for_runtime(&runtime).unwrap();
    let mut cache = PermuteOverwriteCache::default();

    for _ in 0..3 {
        assert_eq!(
            context
                .try_permute_overwrite_into(
                    &mut cache,
                    &mut destination,
                    &source,
                    &axes,
                    &[],
                    Scalar::F64(1.0),
                )
                .unwrap(),
            OverwriteOutcome::Written
        );
    }
    assert_eq!(cache.preparations(), 1);
    let structural_comparisons = cache.structural_comparisons();

    reset_event_counters();
    reset_live_registry();
    PROBE_THREAD_ENABLED.set(true);
    ENABLED.store(true, Ordering::SeqCst);
    let outcome = context
        .try_permute_overwrite_into(
            &mut cache,
            &mut destination,
            &source,
            &axes,
            &[],
            Scalar::F64(1.0),
        )
        .unwrap();
    ENABLED.store(false, Ordering::SeqCst);
    PROBE_THREAD_ENABLED.set(false);

    assert_eq!(outcome, OverwriteOutcome::Written);
    assert_eq!(cache.preparations(), 1);
    assert_eq!(cache.structural_comparisons(), structural_comparisons);
    assert_eq!(PROBE_THREAD_ALLOC_CALLS.load(Ordering::Relaxed), 0);
    assert_eq!(PROBE_THREAD_ALLOCATED_BYTES.load(Ordering::Relaxed), 0);
    assert_eq!(PROBE_THREAD_REALLOC_CALLS.load(Ordering::Relaxed), 0);
    assert_eq!(destination.data(), expected.data());
}

#[test]
fn test_mutex_recovers_after_poisoning() {
    let poisoned = std::panic::catch_unwind(|| {
        let _guard = lock_unpoisoned(&TEST_LOCK);
        panic!("poison test mutex");
    });
    assert!(poisoned.is_err());
    let _recovered = lock_unpoisoned(&TEST_LOCK);
}

#[test]
fn measured_intermediate_arena_accounting() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    if let Ok(name) = std::env::var("TENET_ALLOC_WORKLOAD") {
        let workload = Workload::parse(&name);
        let chi = std::env::var("TENET_ALLOC_CHI").unwrap().parse().unwrap();
        let reuse = match std::env::var("TENET_ALLOC_MODE").unwrap().as_str() {
            "fresh" => false,
            "reuse" => true,
            mode => panic!("unknown allocator mode {mode:?}"),
        };
        worker(workload, chi, reuse);
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
                .all(|sample| sample.payload_output_live_bytes == sample.payload_size_bytes));
            assert!(reused
                .iter()
                .all(|sample| sample.payload_output_live_bytes == sample.payload_size_bytes));
            assert!(fresh.iter().all(|sample| sample.registry_overflows == 0));
            assert!(reused.iter().all(|sample| sample.registry_overflows == 0));
        }
    }
}
