//! `StructureSignature` is content identity (#1495, leaf L0 of #1287).
//!
//! Every test takes one process-wide lock: the eviction, reset and
//! pointer-sharing cases observe the global intern tables.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tenet::core::{
    block_structure_intern_cache_info, complete_hom_space_structure_cache_info, product_sector,
    reset_core_intern_tables, BlockStructure, FermionParityFusionRule, Fz2SectorLayout,
    PackedProductCodec, ProductFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    U1SectorLayout, Z2Irrep,
};
use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, StructureSignature, TensorMap};

type Fz2U1Rule = ProductFusionRule<
    FermionParityFusionRule,
    U1FusionRule,
    PackedProductCodec<Fz2SectorLayout, U1SectorLayout>,
>;

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

static INTERN_TABLES: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    INTERN_TABLES.lock().unwrap_or_else(PoisonError::into_inner)
}

fn hash_of(signature: &StructureSignature) -> u64 {
    let mut hasher = DefaultHasher::new();
    signature.hash(&mut hasher);
    hasher.finish()
}

fn assert_same(left: &StructureSignature, right: &StructureSignature, label: &str) {
    assert!(left == right, "{label}: {left:?} != {right:?}");
    assert_eq!(hash_of(left), hash_of(right), "{label}: hash");
}

fn assert_differ(left: &StructureSignature, right: &StructureSignature, label: &str) {
    assert!(left != right, "{label}: {left:?} == {right:?}");
}

/// One symmetry fixture: `$leg(variant)` builds an independent `GradedSpace`
/// on every call. Variant 0 is the base leg, 1 changes one degeneracy, and 2
/// changes one sector. A macro because the typed constructors' dispatch
/// bounds differ per admission mode.
macro_rules! check_fixture {
    ($label:expr, $leg:expr) => {{
        let label: &str = $label;
        let leg = $leg;
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let build = |codomain: [_; 2], domain: [_; 1]| {
            TensorMap::<_, f64>::zeros(&runtime, codomain, domain).unwrap()
        };
        let (a, b) = (leg(0), leg(0));
        let base = build([&a, &a], [&a]).structure_signature();
        assert_same(
            &base,
            &build([&b, &b], [&b]).structure_signature(),
            &format!("{label} independent"),
        );

        let dual = a.try_dual().unwrap();
        let (degeneracy, sector) = (leg(1), leg(2));
        for (other, name) in [
            (&dual, "dual"),
            (&degeneracy, "degeneracy"),
            (&sector, "sector"),
        ] {
            assert_differ(
                &base,
                &build([&a, other], [&a]).structure_signature(),
                &format!("{label} {name}"),
            );
        }

        // Same external legs, different coupled sectors and fusion-tree basis.
        let repartitioned = build([&a, &a], [&a]).repartition(1).unwrap();
        assert_differ(
            &base,
            &repartitioned.structure_signature(),
            &format!("{label} fusion-tree basis"),
        );
        assert_same(
            &base,
            &repartitioned.repartition(2).unwrap().structure_signature(),
            &format!("{label} repartition round trip"),
        );
    }};
}

#[test]
fn fixtures_are_equal_iff_structure_is_equal() {
    let _guard = lock();
    check_fixture!("U1", |variant| {
        let q = U1Irrep::new;
        let pairs = match variant {
            0 => [(q(-1), 2), (q(0), 1), (q(1), 3)],
            1 => [(q(-1), 2), (q(0), 2), (q(1), 3)],
            _ => [(q(-1), 2), (q(0), 1), (q(2), 3)],
        };
        GradedSpace::try_new(U1FusionRule, pairs).unwrap()
    });
    check_fixture!("SU2", |variant| {
        let j = SU2Irrep::from_twice_spin;
        let pairs = match variant {
            0 => [(j(0), 2), (j(1), 2), (j(2), 1)],
            1 => [(j(0), 2), (j(1), 3), (j(2), 1)],
            _ => [(j(0), 2), (j(1), 2), (j(3), 1)],
        };
        GradedSpace::try_new(SU2FusionRule, pairs).unwrap()
    });
    check_fixture!("fZ2xU1", |variant| {
        let rule = Arc::new(Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule));
        let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
        let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
        let pairs = match variant {
            0 => [(even(0), 2), (odd(1), 1), (odd(-1), 2)],
            1 => [(even(0), 2), (odd(1), 2), (odd(-1), 2)],
            _ => [(even(0), 2), (odd(1), 1), (even(-2), 2)],
        };
        GradedSpace::try_new_with_arc(rule, pairs).unwrap()
    });
    #[cfg(feature = "racah-generated")]
    check_fixture!("SU3 checked Generic", |variant| {
        let rule = Arc::new(tenet::typed::SUNFusionRule::new(3).unwrap());
        let pairs = match variant {
            0 => [(vec![0i64, 0], 1), (vec![1, 1], 2)],
            1 => [(vec![0i64, 0], 1), (vec![1, 1], 3)],
            _ => [(vec![0i64, 0], 1), (vec![1, 0], 2)],
        };
        GradedSpace::try_new_with_arc(rule, pairs).unwrap()
    });
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_rule_instance_separates_signatures() {
    // What: SU(3) and SU(4) trivial legs share hom space and block structure
    // (the unit test checks the fields), so the rule alone separates them.
    let _guard = lock();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let signature = |rank: usize, trivial: Vec<i64>| {
        let rule = Arc::new(tenet::typed::SUNFusionRule::new(rank).unwrap());
        let leg = GradedSpace::try_new_with_arc(rule, [(trivial, 2)]).unwrap();
        TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg])
            .unwrap()
            .structure_signature()
    };
    let su3 = signature(3, vec![0, 0]);
    assert_same(&su3, &signature(3, vec![0, 0]), "SU3 instances");
    assert_differ(&su3, &signature(4, vec![0, 0, 0]), "SU3 vs SU4");
}

fn u1_leg(charges: std::ops::RangeInclusive<i32>, degeneracy: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        U1FusionRule,
        charges.map(|charge| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn u1_signature(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) -> StructureSignature {
    TensorMap::<_, f64>::zeros(runtime, [leg, leg], [leg])
        .unwrap()
        .structure_signature()
}

#[test]
fn runtime_identity_separates_signatures() {
    let _guard = lock();
    let leg = u1_leg(-1..=1, 2);
    let first = Runtime::builder().dense_threads(1).build().unwrap();
    let second = Runtime::builder().dense_threads(1).build().unwrap();
    let signature = u1_signature(&first, &leg);
    assert_same(
        &signature,
        &u1_signature(&first.clone(), &leg),
        "shared runtime",
    );
    assert_differ(&signature, &u1_signature(&second, &leg), "second runtime");

    // A dead Runtime's signature never matches a Runtime built afterwards.
    drop(first);
    let third = Runtime::builder().dense_threads(1).build().unwrap();
    assert_differ(&signature, &u1_signature(&third, &leg), "after drop");
}

#[test]
fn equal_across_reset_core_intern_tables() {
    let _guard = lock();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let before = u1_signature(&runtime, &u1_leg(-2..=2, 3));
    reset_core_intern_tables();
    let after = u1_signature(&runtime, &u1_leg(-2..=2, 3));
    assert_ne!(
        before.content_id(),
        after.content_id(),
        "premise: fresh content"
    );
    assert_same(&before, &after, "reset");
}

#[test]
fn equal_across_interner_eviction() {
    let _guard = lock();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let before = u1_signature(&runtime, &u1_leg(-3..=3, 5));
    let evictions = block_structure_intern_cache_info().pressure_evictions();
    // Push the space out of the complete-structure cache, then its content
    // out of the block-structure interner, with more distinct entries than
    // each holds.
    for degeneracy in 1..=complete_hom_space_structure_cache_info().entry_capacity() + 16 {
        let leg = GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), degeneracy)]).unwrap();
        TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg]).unwrap();
    }
    for extent in 1..=block_structure_intern_cache_info().entry_capacity() + 16 {
        BlockStructure::trivial(&[extent]).unwrap();
    }
    assert!(block_structure_intern_cache_info().pressure_evictions() > evictions);
    let after = u1_signature(&runtime, &u1_leg(-3..=3, 5));
    assert_ne!(before.content_id(), after.content_id(), "premise: evicted");
    assert_same(&before, &after, "eviction");
}

#[test]
fn equal_for_oversized_structures_that_bypass_the_interner() {
    let _guard = lock();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    // One block per charge; far more block bytes than the 8 MiB entry cap.
    let leg = u1_leg(-30_000..=30_000, 1);
    let signature = || {
        TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg])
            .unwrap()
            .structure_signature()
    };
    let bypasses = block_structure_intern_cache_info().oversized_admission_bypasses();
    let before = signature();
    let after = signature();
    assert!(block_structure_intern_cache_info().oversized_admission_bypasses() >= bypasses + 2);
    assert_ne!(
        before.content_id(),
        after.content_id(),
        "premise: not interned"
    );
    assert_same(&before, &after, "oversized");
}

#[test]
fn shared_structure_comparison_does_not_allocate() {
    let _guard = lock();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = u1_leg(-2..=2, 2);
    let first = u1_signature(&runtime, &leg);
    let second = u1_signature(&runtime, &leg);
    assert_eq!(first.content_id(), second.content_id(), "premise: shared");
    let (equal, allocations) = measured(|| first == second);
    assert!(equal);
    assert_eq!(allocations, 0, "pointer fast path");
    let (_, allocations) = measured(|| hash_of(&first));
    assert_eq!(allocations, 0, "hash");
}
