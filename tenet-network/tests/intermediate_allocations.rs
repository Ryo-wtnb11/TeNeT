use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{Complex64, Runtime, TensorScalar};
use tenet::typed::{GradedSpace, SectorSpectrum, TensorMap as TypedTensorMap};
use tenet_network::{
    tensor, ContractionPlan, ContractionStep, Network, NetworkExecutionWorkspace, PlannedNetwork,
    TemporaryLabel, TensorId,
};

#[test]
fn warm_macro_pool_reuses_receiver_sized_dense_payloads() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 7)], false).unwrap();
    let left = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 5)], false).unwrap();
    let right = GradedSpace::try_new(provider, [(U1Irrep::new(0), 11)], false).unwrap();
    let left_dual = left.try_dual().unwrap();
    let a =
        TypedTensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&left], [&bond], 750_001)
            .unwrap();
    let b =
        TypedTensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&bond], [&right], 750_002)
            .unwrap();
    let c = TypedTensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        [&left_dual],
        [&right],
        750_003,
    )
    .unwrap();
    for _ in 0..3 {
        drop(tensor!([c_out; d] = a[a_leg; b_leg] * b[b_leg; c_out] * c[a_leg; d]).unwrap());
    }

    // Depending on the greedy tie break, the sole receiver is either 5x11
    // (A*B) or 7x11 (A*C). Both are distinct from the 11x11 escaped output;
    // neither receiver-sized Vec may be allocated after the pool is warm.
    for receiver_elements in [5 * 11, 7 * 11] {
        reset_event_counters();
        PAYLOAD_SIZE.store(
            receiver_elements * std::mem::size_of::<f64>(),
            Ordering::Relaxed,
        );
        reset_live_registry();
        ENABLED.store(true, Ordering::SeqCst);
        let output = tensor!([c_out; d] = a[a_leg; b_leg] * b[b_leg; c_out] * c[a_leg; d]).unwrap();
        assert_eq!(output.data().len(), 11 * 11);
        drop(output);
        ENABLED.store(false, Ordering::SeqCst);
        assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 0);
        assert_eq!(REGISTRY_OVERFLOWS.load(Ordering::Relaxed), 0);
    }
}

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
    // Why not assert the global event counters in deterministic helper tests:
    // a libtest thread may finish an earlier enabled realloc after their reset.
    static TEST_THREAD_REALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
    static TEST_THREAD_REALLOC_ALLOCATED_BYTES: Cell<u64> = const { Cell::new(0) };
    static TEST_THREAD_REALLOC_DEALLOCATED_BYTES: Cell<u64> = const { Cell::new(0) };
    static BOUNDARY_DEALLOC_COUNTED: Cell<Option<bool>> = const { Cell::new(None) };
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

fn reset_test_thread_realloc_counters() {
    TEST_THREAD_REALLOC_CALLS.set(0);
    TEST_THREAD_REALLOC_ALLOCATED_BYTES.set(0);
    TEST_THREAD_REALLOC_DEALLOCATED_BYTES.set(0);
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
        let _ = TEST_THREAD_REALLOC_CALLS.try_with(|calls| calls.set(calls.get() + 1));
        let _ = TEST_THREAD_REALLOC_ALLOCATED_BYTES
            .try_with(|bytes| bytes.set(bytes.get() + new_size as u64));
        let _ = TEST_THREAD_REALLOC_DEALLOCATED_BYTES
            .try_with(|bytes| bytes.set(bytes.get() + origin.size as u64));
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
    let observe_boundary = boundary_hook.is_some();
    if let Some(hook) = boundary_hook {
        // Why not leave the hook installed: synchronization may enter the allocator,
        // so the one-shot hook must be removed before crossing the test boundary.
        hook.reached.send(()).unwrap();
        hook.resume.recv().unwrap();
    }
    let count_event = ENABLED.load(Ordering::Relaxed);
    if observe_boundary {
        // Why not inspect the thread's aggregate deallocation counters: the
        // synchronization hook may release its own tracked storage on this thread.
        let _ = BOUNDARY_DEALLOC_COUNTED.try_with(|counted| counted.set(Some(count_event)));
    }
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

fn median3(mut values: [u64; 3]) -> u64 {
    values.sort_unstable();
    values[1]
}

trait AllocationScalar: TensorScalar + std::fmt::Debug {
    fn close(lhs: Self, rhs: Self) -> bool;
}

impl AllocationScalar for f64 {
    fn close(lhs: Self, rhs: Self) -> bool {
        (lhs - rhs).abs() <= 1.0e-12 * lhs.abs().max(rhs.abs()).max(1.0)
    }
}

impl AllocationScalar for Complex64 {
    fn close(lhs: Self, rhs: Self) -> bool {
        (lhs - rhs).norm() <= 1.0e-12 * lhs.norm().max(rhs.norm()).max(1.0)
    }
}

fn measure_typed_overwrite_witness<D>(
    planned: &PlannedNetwork,
    tensors: &[TypedTensorMap<U1FusionRule, D>; 3],
    workspace: &mut NetworkExecutionWorkspace<U1FusionRule, D>,
    oracle: &[D],
) -> AllocationSample
where
    D: AllocationScalar,
{
    reset_event_counters();
    let payload_size_bytes = oracle
        .len()
        .checked_mul(std::mem::size_of::<D>())
        .expect("oracle payload byte size overflowed");
    assert!(payload_size_bytes > 0);
    PAYLOAD_SIZE.store(payload_size_bytes, Ordering::Relaxed);
    reset_live_registry();
    ENABLED.store(true, Ordering::SeqCst);

    let refs = tensors.iter().collect::<Vec<_>>();
    let output = planned.execute_with_workspace(&refs, workspace).unwrap();
    assert_eq!(output.data().len(), oracle.len());
    assert!(
        output
            .data()
            .iter()
            .zip(oracle)
            .all(|(&lhs, &rhs)| D::close(lhs, rhs)),
        "typed overwrite result differs from the returning oracle"
    );
    let output_pointer = output.data().as_ptr().cast::<u8>();
    assert!(registered_size(output_pointer).is_some());
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

fn typed_overwrite_worker<D>(chi: usize, reuse: bool)
where
    D: AllocationScalar,
{
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), chi)], false).unwrap();
    let left =
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), chi + 1)], false).unwrap();
    let right = GradedSpace::try_new(provider, [(U1Irrep::new(0), chi + 1)], false).unwrap();
    let left_dual = left.try_dual().unwrap();
    let tensors: [TypedTensorMap<U1FusionRule, D>; 3] = std::array::from_fn(|index| {
        let (codomain, domain) = match index {
            0 => (&left, &bond),
            1 => (&bond, &right),
            _ => (&left_dual, &right),
        };
        TypedTensorMap::rand_with_seed(
            &runtime,
            [codomain],
            [domain],
            74_600 + chi as u64 * 10 + index as u64,
        )
        .unwrap()
    });
    let label = TemporaryLabel::from;
    let output = vec![label("c"), label("d")];
    let network = Network::new(
        vec![
            vec![label("a"), label("b")],
            vec![label("b"), label("c")],
            vec![label("a"), label("d")],
        ],
        vec![false; 3],
        vec![Some(1); 3],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let plan = ContractionPlan::new(
        3,
        output,
        vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                vec![label("a"), label("c")],
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                vec![label("c"), label("d")],
            ),
        ],
    )
    .unwrap();
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    let planned = network.plan_with(&refs, plan).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for _ in 0..8 {
        if !reuse {
            workspace = NetworkExecutionWorkspace::default();
        }
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );
    }
    let oracle = tensors[0]
        .contract(&tensors[1], &[1], &[0], &[1, 0])
        .unwrap()
        .contract(&tensors[2], &[1], &[0], &[0, 1])
        .unwrap()
        .data()
        .to_vec();
    let samples = std::array::from_fn::<_, 3, _>(|_| {
        if !reuse {
            workspace = NetworkExecutionWorkspace::default();
        }
        measure_typed_overwrite_witness(&planned, &tensors, &mut workspace, &oracle)
    });
    for sample in samples {
        println!(
            "TENET_TYPED_ALLOC_SAMPLE {chi} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
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
        );
    }
}

fn run_typed_overwrite_worker(dtype: &str, chi: usize, reuse: bool) -> Vec<AllocationSample> {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "measured_typed_overwrite_witness", "--nocapture"])
        .env("TENET_TYPED_ALLOC_DTYPE", dtype)
        .env(
            "TENET_TYPED_ALLOC_MODE",
            if reuse { "reuse" } else { "fresh" },
        )
        .env("TENET_TYPED_ALLOC_CHI", chi.to_string())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "typed worker failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter(|line| line.starts_with("TENET_TYPED_ALLOC_SAMPLE "))
        .map(|line| {
            let mut fields = line.split_whitespace();
            assert_eq!(fields.next(), Some("TENET_TYPED_ALLOC_SAMPLE"));
            assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), chi);
            assert_eq!(fields.next().unwrap(), if reuse { "1" } else { "0" });
            let values = fields
                .map(|value| value.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 14);
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

fn lazy_conj_worker() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = |degeneracy| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), degeneracy)],
            false,
        )
        .unwrap()
    };
    let (left, bond, right) = (space(5), space(7), space(11));
    let lhs =
        TypedTensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&left], [&bond], 74_700)
            .unwrap();
    let rhs =
        TypedTensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&left], [&right], 74_701)
            .unwrap();
    let label = TemporaryLabel::from;
    let network = Network::new(
        vec![vec![label("x"), label("k")], vec![label("x"), label("r")]],
        vec![true, false],
        vec![Some(1), Some(1)],
        vec![label("k"), label("r")],
        Some(1),
    )
    .unwrap();
    let refs = [&lhs, &rhs];
    let planned = network
        .plan(&refs, &tenet_network::GreedyDenseOptimizer)
        .unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for _ in 0..3 {
        drop(
            planned
                .execute_with_workspace(&refs, &mut workspace)
                .unwrap(),
        );
    }

    let parent_pointer = lhs.data().as_ptr();
    reset_event_counters();
    PAYLOAD_SIZE.store(5 * 7 * std::mem::size_of::<f64>(), Ordering::Relaxed);
    reset_live_registry();
    ENABLED.store(true, Ordering::SeqCst);
    let output = planned
        .execute_with_workspace(&refs, &mut workspace)
        .unwrap();
    assert_eq!(output.rank(), 2);
    drop(output);
    ENABLED.store(false, Ordering::SeqCst);

    // What: 5*7 is the conj receiver only; the final payload is 7*11.
    // PAYLOAD_ALLOC_CALLS includes both allocations and reallocations whose
    // origin or result has the selected size.
    assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 0);
    assert_eq!(lhs.data().as_ptr(), parent_pointer);
    assert_eq!(REGISTRY_OVERFLOWS.load(Ordering::Relaxed), 0);

    let compact_bond = space(13);
    let compact = TypedTensorMap::<U1FusionRule, f64>::diagonal(
        &runtime,
        &compact_bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: (1..=13).map(|value| value as f64).collect(),
        }],
    )
    .unwrap();
    let compact_network = Network::new(
        vec![vec![label("i"), label("j")]],
        vec![false],
        vec![Some(1)],
        vec![label("i"), label("j")],
        Some(1),
    )
    .unwrap();
    let compact_plan = compact_network
        .plan(&[&compact], &tenet_network::GreedyDenseOptimizer)
        .unwrap();
    let mut compact_workspace = NetworkExecutionWorkspace::default();
    for _ in 0..3 {
        drop(
            compact_plan
                .execute_with_workspace(&[&compact], &mut compact_workspace)
                .unwrap(),
        );
    }

    reset_event_counters();
    PAYLOAD_SIZE.store(13 * 13 * std::mem::size_of::<f64>(), Ordering::Relaxed);
    reset_live_registry();
    ENABLED.store(true, Ordering::SeqCst);
    assert_eq!(std::hint::black_box(compact.data()).len(), 13 * 13);
    ENABLED.store(false, Ordering::SeqCst);
    // What: materialization still allocates the dense 13*13 cache after warm
    // replay, proving replay itself did not publish that cache.
    assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(REGISTRY_OVERFLOWS.load(Ordering::Relaxed), 0);
}

#[test]
fn warm_lazy_conj_replay_does_not_materialize_the_input() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    if std::env::var_os("TENET_LAZY_CONJ_ALLOC_WORKER").is_some() {
        lazy_conj_worker();
        return;
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "warm_lazy_conj_replay_does_not_materialize_the_input",
            "--nocapture",
        ])
        .env("TENET_LAZY_CONJ_ALLOC_WORKER", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "lazy-conj allocator worker failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn measured_typed_overwrite_witness() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    if let Ok(dtype) = std::env::var("TENET_TYPED_ALLOC_DTYPE") {
        let chi = std::env::var("TENET_TYPED_ALLOC_CHI")
            .unwrap()
            .parse()
            .unwrap();
        let reuse = std::env::var("TENET_TYPED_ALLOC_MODE").unwrap() == "reuse";
        match dtype.as_str() {
            "f64" => typed_overwrite_worker::<f64>(chi, reuse),
            "c64" => typed_overwrite_worker::<Complex64>(chi, reuse),
            _ => panic!("unknown typed allocator dtype {dtype:?}"),
        }
        return;
    }

    for dtype in ["f64", "c64"] {
        for chi in [8, 16, 32, 64] {
            let fresh = run_typed_overwrite_worker(dtype, chi, false);
            let warm = run_typed_overwrite_worker(dtype, chi, true);
            assert_eq!(fresh.len(), 3);
            assert_eq!(warm.len(), 3);
            let values = |samples: &[AllocationSample], field: fn(&AllocationSample) -> u64| {
                [field(&samples[0]), field(&samples[1]), field(&samples[2])]
            };
            let fresh_peak = values(&fresh, |sample| sample.peak_live_delta);
            let warm_peak = values(&warm, |sample| sample.peak_live_delta);
            let fresh_retained = values(&fresh, |sample| sample.retained_live_bytes);
            let warm_retained = values(&warm, |sample| sample.retained_live_bytes);
            let fresh_payload_retained =
                values(&fresh, |sample| sample.payload_retained_live_bytes);
            let warm_payload_retained = values(&warm, |sample| sample.payload_retained_live_bytes);
            let diagnostics = format!(
                "dtype={dtype}, chi={chi}, fresh peak={fresh_peak:?} (median {}), warm peak={warm_peak:?} (median {}), fresh retained={fresh_retained:?} (median {}), warm retained={warm_retained:?} (median {}), fresh payload retained={fresh_payload_retained:?} (median {}), warm payload retained={warm_payload_retained:?} (median {})",
                median3(fresh_peak),
                median3(warm_peak),
                median3(fresh_retained),
                median3(warm_retained),
                median3(fresh_payload_retained),
                median3(warm_payload_retained),
            );
            assert!(
                fresh.iter().all(|sample| {
                    sample.payload_retained_live_bytes == sample.payload_size_bytes
                        && sample.payload_output_live_bytes == sample.payload_size_bytes
                        && sample.payload_alloc_calls == 2
                        && sample.registry_overflows == 0
                }),
                "{diagnostics}; fresh samples={fresh:?}"
            );
            assert!(
                warm.iter().all(|sample| {
                    sample.payload_retained_live_bytes == 0
                        && sample.payload_output_live_bytes == sample.payload_size_bytes
                        && sample.payload_alloc_calls == 1
                        && sample.registry_overflows == 0
                }),
                "{diagnostics}; warm samples={warm:?}"
            );
        }
    }
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
    reset_test_thread_realloc_counters();
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
    assert_eq!(TEST_THREAD_REALLOC_CALLS.get(), 1);
    assert_eq!(TEST_THREAD_REALLOC_ALLOCATED_BYTES.get(), 16);
    assert_eq!(TEST_THREAD_REALLOC_DEALLOCATED_BYTES.get(), 8);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 48);
}

#[test]
fn realloc_failed_transition_restores_origin_without_duplicate_metrics() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    ENABLED.store(false, Ordering::SeqCst);
    reset_event_counters();
    reset_test_thread_realloc_counters();
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
    assert_eq!(TEST_THREAD_REALLOC_CALLS.get(), 0);
    assert_eq!(TEST_THREAD_REALLOC_ALLOCATED_BYTES.get(), 0);
    assert_eq!(TEST_THREAD_REALLOC_DEALLOCATED_BYTES.get(), 0);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 8);
    assert_eq!(PAYLOAD_ALLOC_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(PAYLOAD_LIVE_BYTES.load(Ordering::Relaxed), 8);
}

#[test]
fn realloc_in_place_transition_replaces_only_its_detached_origin() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    ENABLED.store(false, Ordering::SeqCst);
    reset_event_counters();
    reset_test_thread_realloc_counters();
    PAYLOAD_SIZE.store(0, Ordering::Relaxed);
    reset_live_registry();
    let pointer = 0x1000usize as *mut u8;
    assert!(register_live(pointer, 8));
    let origin = detach_realloc_origin(pointer).expect("old origin must be tracked");

    // What: an in-place realloc replaces its own generation with the new size.
    assert!(finish_realloc_result(origin, pointer, 4, true));
    assert_eq!(registered_size(pointer), Some(4));
    assert_eq!(TEST_THREAD_REALLOC_CALLS.get(), 1);
    assert_eq!(TEST_THREAD_REALLOC_ALLOCATED_BYTES.get(), 4);
    assert_eq!(TEST_THREAD_REALLOC_DEALLOCATED_BYTES.get(), 8);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 4);
}

#[test]
fn realloc_transition_never_exposes_a_zero_live_metrics_window() {
    let _test_guard = lock_unpoisoned(&TEST_LOCK);
    ENABLED.store(false, Ordering::SeqCst);
    reset_event_counters();
    reset_test_thread_realloc_counters();
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
        BOUNDARY_DEALLOC_COUNTED.set(None);
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
        BOUNDARY_DEALLOC_COUNTED.take()
    });

    // What: disabling the probe after unregister but before attribution excludes the free.
    reached_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("deallocation did not expose its unregister boundary");
    ENABLED.store(false, Ordering::SeqCst);
    resume_tx.send(()).unwrap();
    assert_eq!(worker.join().unwrap(), Some(false));
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
fn test_mutex_recovers_after_poisoning() {
    let poisoned = std::panic::catch_unwind(|| {
        let _guard = lock_unpoisoned(&TEST_LOCK);
        panic!("poison test mutex");
    });
    assert!(poisoned.is_err());
    let _recovered = lock_unpoisoned(&TEST_LOCK);
}
