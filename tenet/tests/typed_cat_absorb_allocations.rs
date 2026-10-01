//! Allocation gates for the typed `cat`/`braid`/`absorb`
//! (#580 PR 4), alongside the compact-storage gates in
//! `typed_diagonal_allocations.rs`: bytes counted through a global allocator
//! while one warmed operation runs.
//!
//! The compact diagonal gates require one dense output allocation and no
//! source-sized dense intermediate per compact operand.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;
use tenet::typed::Side;

use tenet::sector::{U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep};
use tenet::typed::{Complex64, Runtime};
use tenet::typed::{GradedSpace, TensorMap};

struct CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED: Cell<u64> = const { Cell::new(0) };
    static PAYLOAD_BYTES: Cell<usize> = const { Cell::new(0) };
    static PAYLOAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ENABLED.get() {
            ALLOCATED.set(ALLOCATED.get() + layout.size() as u64);
            if layout.size() == PAYLOAD_BYTES.get() {
                PAYLOAD_ALLOCATIONS.set(PAYLOAD_ALLOCATIONS.get() + 1);
            }
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
            if new_size == PAYLOAD_BYTES.get() {
                PAYLOAD_ALLOCATIONS.set(PAYLOAD_ALLOCATIONS.get() + 1);
            }
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured_allocations<T>(payload_bytes: usize, operation: impl FnOnce() -> T) -> (u64, usize) {
    ALLOCATED.set(0);
    PAYLOAD_BYTES.set(payload_bytes);
    PAYLOAD_ALLOCATIONS.set(0);
    ENABLED.set(true);
    let output = black_box(operation());
    ENABLED.set(false);
    black_box(output);
    (ALLOCATED.get(), PAYLOAD_ALLOCATIONS.get())
}

/// Small fixed allowance for descriptor/layout bookkeeping (copy plans,
/// derived-space metadata).
const STRUCTURAL_TOLERANCE: u64 = 128 * 1024;

fn u1_leg(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn pseudo_random(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

fn catcodomain_oracle<D: Copy>(lhs: &[D], lhs_rows: usize, rhs: &[D], rhs_rows: usize) -> Vec<D> {
    let columns = lhs.len() / lhs_rows;
    let mut output = Vec::with_capacity(lhs.len() + rhs.len());
    for column in 0..columns {
        output.extend_from_slice(&lhs[column * lhs_rows..(column + 1) * lhs_rows]);
        output.extend_from_slice(&rhs[column * rhs_rows..(column + 1) * rhs_rows]);
    }
    output
}

#[test]
fn typed_cat_uses_one_output_allocation_without_scratch() {
    // What: a warmed typed catdomain owns exactly one output-sized payload
    // and no per-block allocations — the plan executes into the single final
    // buffer.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let codomain = u1_leg(&provider, &[(0, 127)]);
    let left = u1_leg(&provider, &[(0, 251)]);
    let right = u1_leg(&provider, &[(0, 263)]);
    let mut state = 0x5eed_0580u64;
    let lhs: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&codomain], [&left], |_, _| {
            pseudo_random(&mut state)
        })
        .unwrap();
    let mut state = 0x5eed_0581u64;
    let rhs: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&codomain], [&right], |_, _| {
            pseudo_random(&mut state)
        })
        .unwrap();
    let warm: TensorMap<U1FusionRule, f64> = lhs.cat(&rhs, Side::Domain).unwrap();
    let output_payload = std::mem::size_of_val(warm.dense_data().unwrap());

    let (allocated, payload_allocations) =
        measured_allocations(output_payload, || lhs.cat(&rhs, Side::Domain).unwrap());

    assert_eq!(payload_allocations, 1);
    assert!(
        allocated <= output_payload as u64 + STRUCTURAL_TOLERANCE,
        "typed cat allocated {allocated} B for a {output_payload} B output"
    );
}

#[test]
fn typed_absorb_clones_the_destination_once() {
    // What: a warmed typed absorb allocates the destination clone and nothing
    // per block — the merge-join writes prefixes in place.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let mut state = 0x5eed_0582u64;
    let destination: TensorMap<U1FusionRule, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&u1_leg(&provider, &[(0, 251)])],
        [&u1_leg(&provider, &[(0, 127)])],
        |_, _| pseudo_random(&mut state),
    )
    .unwrap();
    let mut state = 0x5eed_0583u64;
    let source: TensorMap<U1FusionRule, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&u1_leg(&provider, &[(0, 263)])],
        [&u1_leg(&provider, &[(0, 101)])],
        |_, _| pseudo_random(&mut state),
    )
    .unwrap();
    black_box(destination.absorb(&source).unwrap());
    let output_payload = std::mem::size_of_val(destination.dense_data().unwrap());

    let (allocated, payload_allocations) =
        measured_allocations(output_payload, || destination.absorb(&source).unwrap());

    assert_eq!(payload_allocations, 1);
    assert!(
        allocated <= output_payload as u64 + STRUCTURAL_TOLERANCE,
        "typed absorb allocated {allocated} B for a {output_payload} B destination clone"
    );
}

#[test]
fn typed_cat_reads_compact_operands_without_source_sized_buffers() {
    const DEGENERACY: usize = 128;
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, DEGENERACY)]).unwrap();
    let mut state = 0x5eed_0584u64;
    let tensor: TensorMap<Z2FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| pseudo_random(&mut state))
            .unwrap();
    // Warm every layout cache with a throwaway spectrum factor.
    let warmup: TensorMap<Z2FusionRule, f64> = tensor.svd_compact(&[0], &[1]).unwrap().s;
    black_box(warmup.cat(&warmup, Side::Domain).unwrap());

    let s: TensorMap<Z2FusionRule, f64> = tensor.svd_compact(&[0], &[1]).unwrap().s;
    let dense_payload = DEGENERACY * DEGENERACY * std::mem::size_of::<f64>();
    let output_payload = 2 * dense_payload;
    let expected = output_payload as u64;

    for call in ["first", "second"] {
        let (bytes, payload_allocations) = measured_allocations(output_payload, || {
            black_box(s.cat(&s, Side::Domain).unwrap());
        });
        assert_eq!(payload_allocations, 1, "{call} cat");
        assert!(
            (expected..=expected + STRUCTURAL_TOLERANCE).contains(&bytes),
            "{call} cat allocated {bytes} B; expected one owned output ({expected} B)"
        );
    }
}

#[test]
fn typed_braid_and_absorb_allocate_only_one_dense_output_for_compact_input() {
    const DEGENERACY: usize = 128;
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, DEGENERACY)]).unwrap();
    let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [tenet::typed::SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![2.0; DEGENERACY],
        }],
    )
    .unwrap();
    black_box(diagonal.braid(&[1], &[0], &[0, 1]).unwrap());
    black_box(diagonal.absorb(&diagonal).unwrap());
    let payload = DEGENERACY * DEGENERACY * std::mem::size_of::<f64>();
    let mut counts = [0; 2];
    for (operation, name) in ["braid", "absorb"].into_iter().enumerate() {
        let (bytes, payload_allocations) = measured_allocations(payload, || {
            if operation == 0 {
                black_box(diagonal.braid(&[1], &[0], &[0, 1]).unwrap());
            } else {
                black_box(diagonal.absorb(&diagonal).unwrap());
            }
        });
        counts[operation] = payload_allocations;
        assert!(
            bytes >= payload as u64,
            "{name} allocated fewer than the required output: {bytes} B"
        );
    }
    assert_eq!(counts, [1, 1], "braid/absorb P-sized buffer counts");
}

#[test]
fn typed_lazy_adjoint_cat_allocates_only_the_output_payload() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let common = u1_leg(&provider, &[(0, 64)]);
    let left = u1_leg(&provider, &[(0, 192)]);
    let right = u1_leg(&provider, &[(0, 320)]);
    let lhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left], [&common], 388_001).unwrap();
    let rhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&right], [&common], 388_002).unwrap();

    let warm_lhs = lhs_parent.adjoint().unwrap();
    let warm_rhs = rhs_parent.adjoint().unwrap();
    let warm_output = warm_lhs.cat(&warm_rhs, Side::Domain).unwrap();
    let output_payload = std::mem::size_of_val(warm_output.dense_data().unwrap());
    let input_payload = (lhs_parent.dense_data().unwrap().len()
        + rhs_parent.dense_data().unwrap().len()) as u64
        * std::mem::size_of::<Complex64>() as u64;

    let fast_lhs = lhs_parent.adjoint().unwrap();
    let fast_rhs = rhs_parent.adjoint().unwrap();
    let (fast_bytes, payload_allocations) = measured_allocations(output_payload, || {
        fast_lhs.cat(&fast_rhs, Side::Domain).unwrap()
    });

    let eager_lhs = lhs_parent.adjoint().unwrap();
    let eager_rhs = rhs_parent.adjoint().unwrap();
    let (eager_bytes, _) = measured_allocations(output_payload, || {
        let lhs = eager_lhs.materialize().unwrap();
        let rhs = eager_rhs.materialize().unwrap();
        lhs.cat(&rhs, Side::Domain).unwrap()
    });

    assert_eq!(payload_allocations, 1);
    assert!(
        fast_bytes >= output_payload as u64,
        "lazy cat allocated {fast_bytes} B, below its {output_payload} B owned output"
    );
    assert!(
        fast_bytes <= output_payload as u64 + STRUCTURAL_TOLERANCE,
        "lazy cat allocated {fast_bytes} B for a {output_payload} B output"
    );
    assert!(
        eager_bytes + STRUCTURAL_TOLERANCE >= fast_bytes + input_payload,
        "lazy cat saved {} B, below the {input_payload} B input payloads outside the \
         {STRUCTURAL_TOLERANCE} B structural allowance",
        eager_bytes.saturating_sub(fast_bytes)
    );
}

#[test]
fn typed_lazy_adjoint_cat_matches_column_major_oracles_in_every_orientation() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let common = u1_leg(&provider, &[(0, 3)]);
    let left = u1_leg(&provider, &[(0, 2)]);
    let right = u1_leg(&provider, &[(0, 4)]);

    let lhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left], [&common], 773_001).unwrap();
    let rhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&right], [&common], 773_002).unwrap();
    let lhs_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&common], [&left], 773_003).unwrap();
    let rhs_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&common], [&right], 773_004).unwrap();

    let lhs_adjoint = lhs_parent.adjoint().unwrap();
    let rhs_adjoint = rhs_parent.adjoint().unwrap();
    for (lhs, rhs, expected) in [
        (
            &lhs_adjoint,
            &rhs_owned,
            lhs_parent
                .adjoint()
                .unwrap()
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .chain(rhs_owned.dense_data().unwrap())
                .copied()
                .collect::<Vec<_>>(),
        ),
        (
            &lhs_owned,
            &rhs_adjoint,
            lhs_owned
                .dense_data()
                .unwrap()
                .iter()
                .chain(
                    rhs_parent
                        .adjoint()
                        .unwrap()
                        .materialize()
                        .unwrap()
                        .dense_data()
                        .unwrap(),
                )
                .copied()
                .collect(),
        ),
        (
            &lhs_adjoint,
            &rhs_adjoint,
            lhs_parent
                .adjoint()
                .unwrap()
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .chain(
                    rhs_parent
                        .adjoint()
                        .unwrap()
                        .materialize()
                        .unwrap()
                        .dense_data()
                        .unwrap(),
                )
                .copied()
                .collect(),
        ),
    ] {
        assert_eq!(
            lhs.cat(rhs, Side::Domain).unwrap().dense_data().unwrap(),
            expected
        );
    }

    let upper_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&common], [&left], 773_005).unwrap();
    let lower_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&common], [&right], 773_006).unwrap();
    let upper_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left], [&common], 773_007).unwrap();
    let lower_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&right], [&common], 773_008).unwrap();
    let upper_adjoint = upper_parent.adjoint().unwrap();
    let lower_adjoint = lower_parent.adjoint().unwrap();
    for (upper, lower, expected) in [
        (
            &upper_adjoint,
            &lower_owned,
            catcodomain_oracle(
                upper_parent
                    .adjoint()
                    .unwrap()
                    .materialize()
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                2,
                lower_owned.dense_data().unwrap(),
                4,
            ),
        ),
        (
            &upper_owned,
            &lower_adjoint,
            catcodomain_oracle(
                upper_owned.dense_data().unwrap(),
                2,
                lower_parent
                    .adjoint()
                    .unwrap()
                    .materialize()
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                4,
            ),
        ),
        (
            &upper_adjoint,
            &lower_adjoint,
            catcodomain_oracle(
                upper_parent
                    .adjoint()
                    .unwrap()
                    .materialize()
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                2,
                lower_parent
                    .adjoint()
                    .unwrap()
                    .materialize()
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                4,
            ),
        ),
    ] {
        assert_eq!(
            upper
                .cat(lower, Side::Codomain)
                .unwrap()
                .dense_data()
                .unwrap(),
            expected
        );
    }
}
