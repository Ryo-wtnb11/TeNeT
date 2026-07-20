use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet::prelude::*;

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATED.set(ALLOCATED.get() + layout.size() as u64);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATED.set(ALLOCATED.get() + new_size as u64);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured_bytes<T>(operation: impl FnOnce() -> T) -> u64 {
    ALLOCATED.set(0);
    ENABLED.set(true);
    let output = black_box(operation());
    ENABLED.set(false);
    black_box(output);
    ALLOCATED.get()
}

#[test]
fn mixed_cat_widens_into_the_final_c64_payload() {
    let runtime = Runtime::builder().build().unwrap();
    let codomain = Space::u1([(0, 16)]);
    let wide = Space::u1([(0, 512)]);
    let narrow = Space::u1([(0, 3)]);
    let real = Tensor::rand_with_seed(&runtime, Dtype::F64, [&codomain], [&wide], 11_058).unwrap();
    let real_c64 = real.to_c64();
    let complex =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&codomain], [&narrow], 11_059).unwrap();

    black_box(real.catdomain(&complex).unwrap());
    black_box(real_c64.catdomain(&complex).unwrap());
    black_box(complex.catdomain(&real).unwrap());
    black_box(complex.catdomain(&real_c64).unwrap());

    let mixed_first = measured_bytes(|| real.catdomain(&complex).unwrap());
    let c64_first = measured_bytes(|| real_c64.catdomain(&complex).unwrap());
    let mixed_second = measured_bytes(|| complex.catdomain(&real).unwrap());
    let c64_second = measured_bytes(|| complex.catdomain(&real_c64).unwrap());
    let promoted_payload = real.data().len() as u64 * std::mem::size_of::<Complex64>() as u64;
    let fixed_allocation_tolerance = promoted_payload / 8;

    assert!(
        mixed_first <= c64_first + fixed_allocation_tolerance,
        "mixed lhs allocated {mixed_first} B versus {c64_first} B for c64; \
         promoted payload is {promoted_payload} B"
    );
    assert!(
        mixed_second <= c64_second + fixed_allocation_tolerance,
        "mixed rhs allocated {mixed_second} B versus {c64_second} B for c64; \
         promoted payload is {promoted_payload} B"
    );
}
