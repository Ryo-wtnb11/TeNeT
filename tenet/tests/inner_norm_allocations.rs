use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use tenet::core::{
    product_sector, FermionParityFusionRule, Fz2SectorLayout, PackedProductCodec,
    ProductFusionRule, ProductSectorLayout, SU2FusionRule, SU2Irrep, Su2SectorLayout, U1FusionRule,
    U1Irrep, U1SectorLayout, Z2Irrep,
};
use tenet::prelude::{Complex64, GradedSpace, Runtime, TensorMap};

type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Su2Codec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type Fz2U1Su2Rule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, Fz2U1Su2Codec>;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, ALLOCATIONS.get())
}

fn non_abelian_space() -> GradedSpace<Fz2U1Su2Rule> {
    let rule = Arc::new(Fz2U1Su2Rule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    ));
    let label = |parity: u8, charge: i32, twice_spin: usize| {
        product_sector(
            product_sector(
                if parity == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                },
                U1Irrep::new(charge),
            ),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    GradedSpace::try_new_shared(
        rule,
        [
            (label(0, -2, 0), 4),
            (label(0, 1, 2), 3),
            (label(1, -1, 1), 4),
            (label(1, 2, 3), 2),
        ],
    )
    .unwrap()
}

#[test]
fn warmed_non_abelian_inner_and_norm_do_not_allocate() {
    // What: cold region compilation is observable, while explicit warm-up
    // leaves both public reductions allocation-free on the caller thread.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let lhs: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 282_401).unwrap();
    let rhs: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 282_402).unwrap();

    let (cold, cold_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    eprintln!("cold coupled-region initialization: {cold_allocations} allocations");
    black_box(cold);
    black_box(lhs.norm().unwrap());

    let (inner, inner_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    let (norm, norm_allocations) = measured(|| lhs.norm().unwrap());
    black_box((inner, norm));

    assert_eq!(inner_allocations, 0);
    assert_eq!(norm_allocations, 0);
}

#[test]
fn warmed_lazy_adjoint_inner_does_not_allocate_in_mixed_or_double_orientation() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let lhs_parent: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_301).unwrap();
    let rhs_parent: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_302).unwrap();
    let owned: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space, &space], 666_303).unwrap();
    let warm_lhs = lhs_parent.adjoint().unwrap();
    let warm_rhs = rhs_parent.adjoint().unwrap();
    black_box(warm_lhs.inner(&owned).unwrap());
    black_box(owned.inner(&warm_lhs).unwrap());
    black_box(warm_lhs.inner(&warm_rhs).unwrap());

    let lhs_mixed_left = lhs_parent.adjoint().unwrap();
    let lhs_mixed_right = lhs_parent.adjoint().unwrap();
    let lhs_double = lhs_parent.adjoint().unwrap();
    let rhs_double = rhs_parent.adjoint().unwrap();
    for (value, allocations) in [
        measured(|| lhs_mixed_left.inner(&owned).unwrap()),
        measured(|| owned.inner(&lhs_mixed_right).unwrap()),
        measured(|| lhs_double.inner(&rhs_double).unwrap()),
    ] {
        black_box(value);
        assert_eq!(allocations, 0);
    }
    for lazy in [&lhs_mixed_left, &lhs_mixed_right, &lhs_double, &rhs_double] {
        let parent_len = lhs_parent.data().len();
        let (materialized_len, allocations) = measured(|| lazy.data().len());
        assert_eq!(materialized_len, parent_len);
        assert!(allocations > 0, "inner materialized its lazy operand");
    }
}
