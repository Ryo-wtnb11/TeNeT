//! Explicit payload precision conversions (#1323).
//!
//! What: `to_f64`/`to_c32`/`to_c64` are exact and `narrow_to_f32`/
//! `narrow_to_c32` round like Rust `as f32`, entry by entry; spaces, block
//! structure and storage form (dense, compact diagonal, lazy adjoint) are
//! unchanged; a conversion makes one payload-sized allocation.
//!
//! Every comparison is bitwise: conversion is data movement. The oracle for a
//! widening is `f64::from` and for a narrowing `as f32`, applied per element
//! to the source payload — independent of the tensor code under test.

mod single_precision_oracle;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use num_complex::{Complex32, Complex64};
use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{GradedSpace, SectorSpectrum, TensorMap};

use single_precision_oracle::{fermion_su2_leg, runtime, u1_leg, u1_leg_with};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static LARGE_THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
    static LARGE: Cell<usize> = const { Cell::new(0) };
}

fn record(size: usize) {
    if COUNTING.get() {
        CALLS.set(CALLS.get() + 1);
        BYTES.set(BYTES.get() + size);
        if size >= LARGE_THRESHOLD.get() {
            LARGE.set(LARGE.get() + 1);
        }
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            record(new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocation calls, bytes, and calls of at least `large` bytes.
fn measured<T>(large: usize, operation: impl FnOnce() -> T) -> (T, usize, usize, usize) {
    CALLS.set(0);
    BYTES.set(0);
    LARGE.set(0);
    LARGE_THRESHOLD.set(large);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, CALLS.get(), BYTES.get(), LARGE.get())
}

/// Every `f32` class: signed zeros, the smallest and largest subnormal, the
/// normal range ends, infinities, a NaN with a payload, and ordinary values.
const F32_SPECIALS: [f32; 12] = [
    0.0,
    -0.0,
    f32::from_bits(1),
    f32::from_bits(0x007f_ffff),
    f32::MIN_POSITIVE,
    f32::MAX,
    -f32::MAX,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::from_bits(0x7fc0_1234),
    1.0 / 3.0,
    -2.5e-3,
];

/// `f64` inputs chosen for the rounding and range edges of `as f32`.
fn f64_narrowing_cases() -> Vec<(f64, Option<f32>)> {
    let one = 1.0f64;
    let half_ulp = 2f64.powi(-24);
    let max = f64::from(f32::MAX);
    vec![
        // tie at 1 + 2^-24 rounds to even (down to 1)
        (one + half_ulp, Some(1.0)),
        // just above the tie rounds up
        (one + half_ulp + 2f64.powi(-52), Some(1.0 + f32::EPSILON)),
        // tie at 1 + 3 * 2^-24 rounds to even (up)
        (one + 3.0 * half_ulp, Some(1.0 + 2.0 * f32::EPSILON)),
        (0.1, None),
        (-0.0, Some(-0.0)),
        (0.0, Some(0.0)),
        // below half the ulp of f32::MAX: rounds back to f32::MAX
        (max + 2f64.powi(102), Some(f32::MAX)),
        // the tie above f32::MAX (odd significand) rounds to infinity
        (max + 2f64.powi(103), Some(f32::INFINITY)),
        (-2.0 * max, Some(f32::NEG_INFINITY)),
        (1e300, Some(f32::INFINITY)),
        (f64::INFINITY, Some(f32::INFINITY)),
        (f64::NEG_INFINITY, Some(f32::NEG_INFINITY)),
        // subnormal range: kept, not flushed
        (f64::from(f32::from_bits(1)), Some(f32::from_bits(1))),
        (3e-39, None),
        // below half the smallest subnormal: signed zero
        (1e-46, Some(0.0)),
        (-1e-46, Some(-0.0)),
        (f64::from_bits(1), Some(0.0)),
        (f64::NAN, None),
    ]
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn f64_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn c32_bits(values: &[Complex32]) -> Vec<(u32, u32)> {
    values
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect()
}

fn c64_bits(values: &[Complex64]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect()
}

fn narrow(value: f64) -> f32 {
    black_box(value) as f32
}

/// Asserts the converted tensor carries the source's spaces and blocks.
macro_rules! assert_same_structure {
    ($what:expr, $got:expr, $source:expr) => {{
        let (got, source) = (&$got, &$source);
        assert_eq!(got.codomain(), source.codomain(), "{}: codomain", $what);
        assert_eq!(got.domain(), source.domain(), "{}: domain", $what);
        assert_eq!(got.block_count(), source.block_count(), "{}: blocks", $what);
        for index in 0..source.block_count() {
            assert_eq!(
                format!("{:?}", got.block_fusion_trees(index).unwrap()),
                format!("{:?}", source.block_fusion_trees(index).unwrap()),
                "{}: fusion trees of block {index}",
                $what
            );
        }
        assert_eq!(got.data().len(), source.data().len(), "{}: length", $what);
        assert!(
            std::ptr::eq(got.provider(), source.provider()),
            "{}: the provider Arc must be shared, not rebuilt",
            $what
        );
    }};
}

/// Runs every conversion on one tensor of any provider and storage form and
/// checks it bitwise against the per-element oracle and the source structure.
/// `$f32` is the single-precision source; the other dtypes are derived from it
/// by the oracle, not by the code under test.
macro_rules! assert_all_conversions {
    ($what:expr, $f32:expr) => {{
        let what = $what;
        let source: TensorMap<_, f32> = $f32;
        let values = source.data().to_vec();

        let wide = source.to_f64();
        assert_same_structure!(what, wide, source);
        let expected: Vec<f64> = values.iter().map(|&value| f64::from(value)).collect();
        assert_eq!(f64_bits(wide.data()), f64_bits(&expected), "{what}: to_f64");
        assert_eq!(
            f32_bits(wide.narrow_to_f32().data()),
            f32_bits(&values),
            "{what}: narrow_to_f32(to_f64(x)) == x"
        );

        let c32 = source.to_c32();
        assert_same_structure!(what, c32, source);
        let expected: Vec<Complex32> = values
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect();
        assert_eq!(c32_bits(c32.data()), c32_bits(&expected), "{what}: to_c32");

        let c64 = source.to_c64();
        assert_same_structure!(what, c64, source);
        let expected: Vec<Complex64> = values
            .iter()
            .map(|&value| Complex64::new(f64::from(value), 0.0))
            .collect();
        assert_eq!(
            c64_bits(c64.data()),
            c64_bits(&expected),
            "{what}: f32 to_c64"
        );
        assert_eq!(
            c64_bits(wide.to_c64().data()),
            c64_bits(&expected),
            "{what}: f64 to_c64"
        );
        assert_eq!(
            c64_bits(c32.to_c64().data()),
            c64_bits(&expected),
            "{what}: Complex32 to_c64"
        );
        assert_eq!(
            c32_bits(c64.narrow_to_c32().data()),
            c32_bits(c32.data()),
            "{what}: narrow_to_c32(to_c64(x)) == x"
        );

        // A genuinely complex payload: both components move independently.
        let complex = c32.scale(Complex32::new(0.75, -1.5));
        let complex_values = complex.data().to_vec();
        let widened = complex.to_c64();
        assert_same_structure!(what, widened, complex);
        let expected: Vec<Complex64> = complex_values
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect();
        assert_eq!(
            c64_bits(widened.data()),
            c64_bits(&expected),
            "{what}: complex to_c64"
        );
        assert_eq!(
            c32_bits(widened.narrow_to_c32().data()),
            c32_bits(&complex_values),
            "{what}: complex round trip"
        );
        (source, wide)
    }};
}

/// Converts the lazy adjoint of `$owned` at every source dtype and compares
/// each result bitwise with the element-wise conversion of the materialized
/// source view. Returns the converted views for storage-form checks.
///
/// `$signless_complex_zeros` is set for a checked-Generic compact diagonal:
/// its complex lazy view materializes the unstored off-diagonal zeros as
/// `0 - 0i` (the conjugate of the dense expansion), while the converted
/// result is the owned compact diagonal the multiplicity-free `adjoint`
/// emits, whose unstored zeros are `+0`. Only the sign of a zero is waived,
/// and only for complex sources; the stored spectrum is compared bitwise by
/// the caller.
macro_rules! assert_adjoint_conversions {
    ($what:expr, $owned:expr) => {
        assert_adjoint_conversions!($what, $owned, false)
    };
    ($what:expr, $owned:expr, $signless_complex_zeros:expr) => {{
        let what = $what;
        let zeros = |bits: Vec<(u64, u64)>| -> Vec<(u64, u64)> {
            let unsign = |b: u64| if b == 1 << 63 { 0 } else { b };
            if $signless_complex_zeros {
                bits.into_iter()
                    .map(|(re, im)| (unsign(re), unsign(im)))
                    .collect()
            } else {
                bits
            }
        };
        let zeros32 = |bits: Vec<(u32, u32)>| -> Vec<(u32, u32)> {
            let unsign = |b: u32| if b == 1 << 31 { 0 } else { b };
            if $signless_complex_zeros {
                bits.into_iter()
                    .map(|(re, im)| (unsign(re), unsign(im)))
                    .collect()
            } else {
                bits
            }
        };
        let owned: TensorMap<_, f32> = $owned;
        let lazy32 = owned.adjoint().unwrap();
        let lazy64 = owned.to_f64().adjoint().unwrap();
        let lazyc32 = owned
            .to_c32()
            .scale(Complex32::new(0.75, -1.5))
            .adjoint()
            .unwrap();
        let lazyc64 = owned
            .to_c64()
            .scale(Complex64::new(0.75, -1.5))
            .adjoint()
            .unwrap();

        let to_f64 = lazy32.to_f64();
        let expected: Vec<f64> = lazy32.data().iter().map(|&v| f64::from(v)).collect();
        assert_eq!(
            f64_bits(to_f64.data()),
            f64_bits(&expected),
            "{what}: f32 to_f64"
        );
        assert_same_structure!(what, to_f64, lazy32);

        let to_c32 = lazy32.to_c32();
        let expected: Vec<Complex32> = lazy32
            .data()
            .iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        assert_eq!(
            c32_bits(to_c32.data()),
            c32_bits(&expected),
            "{what}: f32 to_c32"
        );

        let f32_to_c64 = lazy32.to_c64();
        let expected: Vec<Complex64> = lazy32
            .data()
            .iter()
            .map(|&v| Complex64::new(f64::from(v), 0.0))
            .collect();
        assert_eq!(
            c64_bits(f32_to_c64.data()),
            c64_bits(&expected),
            "{what}: f32 to_c64"
        );

        let f64_to_c64 = lazy64.to_c64();
        let expected: Vec<Complex64> = lazy64
            .data()
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        assert_eq!(
            c64_bits(f64_to_c64.data()),
            c64_bits(&expected),
            "{what}: f64 to_c64"
        );
        assert_same_structure!(what, f64_to_c64, lazy64);

        let narrowed = lazy64.narrow_to_f32();
        let expected: Vec<f32> = lazy64.data().iter().map(|&v| narrow(v)).collect();
        assert_eq!(
            f32_bits(narrowed.data()),
            f32_bits(&expected),
            "{what}: narrow_to_f32"
        );

        let c32_to_c64 = lazyc32.to_c64();
        let expected: Vec<Complex64> = lazyc32
            .data()
            .iter()
            .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)))
            .collect();
        assert_eq!(
            zeros(c64_bits(c32_to_c64.data())),
            zeros(c64_bits(&expected)),
            "{what}: Complex32 to_c64"
        );

        let narrowed_c = lazyc64.narrow_to_c32();
        let expected: Vec<Complex32> = lazyc64
            .data()
            .iter()
            .map(|v| Complex32::new(narrow(v.re), narrow(v.im)))
            .collect();
        assert_eq!(
            zeros32(c32_bits(narrowed_c.data())),
            zeros32(c32_bits(&expected)),
            "{what}: narrow_to_c32"
        );
        assert_same_structure!(what, narrowed_c, lazyc64);

        (to_f64, f64_to_c64, c32_to_c64, narrowed_c)
    }};
}

/// A tensor cycling through [`F32_SPECIALS`], scaled per entry so the
/// stored values are distinct.
macro_rules! filled {
    ([$($codomain:expr),*], [$($domain:expr),*]) => {{
        let mut next = 0usize;
        let tensor: TensorMap<_, f32> =
            TensorMap::from_block_fn(&runtime(), [$($codomain),*], [$($domain),*], |_, _| {
                let value = F32_SPECIALS[next % F32_SPECIALS.len()];
                next += 1;
                value * (1.0 + next as f32 / 64.0)
            })
            .unwrap();
        tensor
    }};
}

#[test]
fn u1_dense_conversions_are_exact_and_keep_structure() {
    let leg = u1_leg();
    let (source, _) = assert_all_conversions!("u1 dense", filled!([&leg, &leg], [&leg]));
    assert!(source.data().iter().any(|value| value.is_nan()));
    assert!(source.data().iter().any(|value| value.is_subnormal()));
    assert!(source
        .data()
        .iter()
        .any(|value| *value == 0.0 && value.is_sign_negative()));
}

#[test]
fn fermionic_su2_dense_conversions_are_exact_and_keep_structure() {
    let leg = fermion_su2_leg();
    assert_all_conversions!(
        "fZ2 x U(1) x SU(2) dense",
        filled!([&leg, &leg], [&leg, &leg])
    );
}

#[test]
fn lazy_adjoint_conversions_match_the_converted_logical_payload() {
    let leg = fermion_su2_leg();
    let source = filled!([&leg, &leg], [&leg]);
    // The logical payload of the lazy view is what gets converted.
    assert_all_conversions!("fermionic lazy adjoint", source.adjoint().unwrap());
    let u1 = u1_leg();
    assert_all_conversions!(
        "u1 lazy adjoint",
        filled!([&u1], [&u1, &u1]).adjoint().unwrap()
    );

    assert_adjoint_conversions!("fermionic dense adjoint", source.clone());
    assert_adjoint_conversions!("u1 dense adjoint", filled!([&u1, &u1], [&u1]));

    // A complex lazy adjoint conjugates: the converted view must too.
    let complex = source.to_c32().scale(Complex32::new(0.5, 2.0));
    let lazy = complex.adjoint().unwrap();
    let expected: Vec<Complex64> = lazy
        .data()
        .iter()
        .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect();
    let widened = complex.adjoint().unwrap().to_c64();
    assert_eq!(c64_bits(widened.data()), c64_bits(&expected));
    assert_eq!(
        c32_bits(complex.to_c64().adjoint().unwrap().narrow_to_c32().data()),
        c32_bits(lazy.data())
    );
}

#[test]
fn compact_diagonal_stays_compact() {
    let leg = u1_leg();
    let mut next = 0usize;
    let spectra = [(-1, 2), (0, 3), (1, 2)].map(|(charge, degeneracy)| SectorSpectrum {
        sector: U1Irrep::new(charge),
        values: (0..degeneracy)
            .map(|_| {
                next += 1;
                F32_SPECIALS[next % F32_SPECIALS.len()]
            })
            .collect(),
    });
    let source: TensorMap<U1FusionRule, f32> =
        TensorMap::diagonal(&runtime(), &leg, spectra.clone()).unwrap();
    let (_, wide) = assert_all_conversions!("compact diagonal", source.clone());

    let widened = wide
        .diagonal_spectrum()
        .unwrap()
        .expect("to_f64 stays compact");
    assert_eq!(widened.len(), spectra.len());
    for got in &widened {
        let expected = spectra
            .iter()
            .find(|entry| entry.sector == got.sector)
            .unwrap();
        let expected: Vec<f64> = expected
            .values
            .iter()
            .map(|&value| f64::from(value))
            .collect();
        assert_eq!(f64_bits(&got.values), f64_bits(&expected));
    }
    assert!(source.to_c32().diagonal_spectrum().unwrap().is_some());
    assert!(source.to_c64().diagonal_spectrum().unwrap().is_some());
    assert!(wide.narrow_to_f32().diagonal_spectrum().unwrap().is_some());
    assert!(wide
        .to_c64()
        .narrow_to_c32()
        .diagonal_spectrum()
        .unwrap()
        .is_some());
    // A multiplicity-free `adjoint` of a compact diagonal is already an owned
    // compact diagonal, so its conversion is the owned case again.
    let adjoint = source.adjoint().unwrap().to_f64();
    assert!(adjoint.diagonal_spectrum().unwrap().is_some());
}

#[test]
fn narrowing_rounds_like_as_f32_per_element() {
    let cases = f64_narrowing_cases();
    let leg = GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), cases.len())]).unwrap();
    let mut entries = cases.iter().map(|&(value, _)| value);
    let source: TensorMap<U1FusionRule, f64> = TensorMap::from_block_fn(
        &runtime(),
        [&leg],
        [&GradedSpace::try_new(U1FusionRule, [(U1Irrep::new(0), 1)]).unwrap()],
        |_, _| entries.next().unwrap(),
    )
    .unwrap();
    // `from_block_fn` fills in the coupled layout; read the order back.
    let values = source.data().to_vec();
    assert_eq!(values.len(), cases.len());

    let narrowed = source.narrow_to_f32();
    assert_same_structure!("narrow_to_f32", narrowed, source);
    for (index, (&got, &value)) in narrowed.data().iter().zip(&values).enumerate() {
        let oracle = narrow(value);
        assert_eq!(got.to_bits(), oracle.to_bits(), "entry {index}: {value:e}");
        assert_eq!(got.is_nan(), value.is_nan(), "entry {index}: NaN stays NaN");
    }
    // Hand values, independent of `as`: the documented rounding and range.
    for &(value, expected) in &cases {
        let position = values
            .iter()
            .position(|stored| stored.to_bits() == value.to_bits())
            .unwrap();
        let got = narrowed.data()[position];
        match expected {
            Some(expected) => assert_eq!(got.to_bits(), expected.to_bits(), "{value:e}"),
            None if value.is_nan() => assert!(got.is_nan()),
            None => assert_eq!(f64::from(got), f64::from(narrow(value))),
        }
    }
    assert!(narrowed.data().iter().any(|value| value.is_subnormal()));

    // Complex narrowing is componentwise with the same rounding.
    let complex = source.to_c64().scale(Complex64::new(0.0, 1.0)).add(
        &source.to_c64(),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    );
    let complex = complex.unwrap();
    let complex_values = complex.data().to_vec();
    let narrowed = complex.narrow_to_c32();
    assert_same_structure!("narrow_to_c32", narrowed, complex);
    let expected: Vec<Complex32> = complex_values
        .iter()
        .map(|value| Complex32::new(narrow(value.re), narrow(value.im)))
        .collect();
    assert_eq!(c32_bits(narrowed.data()), c32_bits(&expected));
}

/// Measures `convert` on a small and a large source of the same structure:
/// the handle cost must not depend on the payload, the byte difference must
/// be exactly `payloads` buffers of the given entry sizes, and exactly that
/// many allocations are payload-sized.
fn assert_payload_allocations<T, O>(
    what: &str,
    small: &T,
    large: &T,
    lengths: (usize, usize),
    entry_sizes: &[usize],
    convert: impl Fn(&T) -> O,
) {
    let smallest = *entry_sizes.iter().min().unwrap();
    let per_entry: usize = entry_sizes.iter().sum();
    black_box(convert(small));
    black_box(convert(large));
    let (_, small_calls, small_bytes, small_payloads) =
        measured(lengths.0 * smallest, || black_box(convert(small)));
    let (_, large_calls, large_bytes, large_payloads) =
        measured(lengths.1 * smallest, || black_box(convert(large)));
    assert_eq!(
        small_calls, large_calls,
        "{what}: handle cost depends on the payload"
    );
    assert_eq!(
        large_bytes - small_bytes,
        (lengths.1 - lengths.0) * per_entry,
        "{what}: payload bytes"
    );
    assert_eq!(
        (small_payloads, large_payloads),
        (entry_sizes.len(), entry_sizes.len()),
        "{what}: payload-sized allocations"
    );
}

#[test]
fn conversion_payload_allocations() {
    let (small_leg, large_leg) = (u1_leg_with([2, 3, 2]), u1_leg_with([9, 10, 9]));
    let small = filled!([&small_leg, &small_leg], [&small_leg]);
    let large = filled!([&large_leg, &large_leg], [&large_leg]);
    let lengths = (small.data().len(), large.data().len());
    assert_payload_allocations(
        "to_f64",
        &small,
        &large,
        lengths,
        &[std::mem::size_of::<f64>()],
        TensorMap::to_f64,
    );
    assert_payload_allocations(
        "to_c32",
        &small,
        &large,
        lengths,
        &[std::mem::size_of::<Complex32>()],
        TensorMap::to_c32,
    );
    assert_payload_allocations(
        "f32 to_c64",
        &small,
        &large,
        lengths,
        &[std::mem::size_of::<Complex64>()],
        |t: &TensorMap<U1FusionRule, f32>| t.to_c64(),
    );

    let (small_wide, large_wide) = (small.to_f64(), large.to_f64());
    assert_payload_allocations(
        "narrow_to_f32",
        &small_wide,
        &large_wide,
        lengths,
        &[std::mem::size_of::<f32>()],
        TensorMap::narrow_to_f32,
    );
    assert_payload_allocations(
        "f64 to_c64",
        &small_wide,
        &large_wide,
        lengths,
        &[std::mem::size_of::<Complex64>()],
        |t: &TensorMap<U1FusionRule, f64>| t.to_c64(),
    );
    let (small_c32, large_c32) = (small.to_c32(), large.to_c32());
    assert_payload_allocations(
        "Complex32 to_c64",
        &small_c32,
        &large_c32,
        lengths,
        &[std::mem::size_of::<Complex64>()],
        |t: &TensorMap<U1FusionRule, Complex32>| t.to_c64(),
    );
    let (small_c64, large_c64) = (small.to_c64(), large.to_c64());
    assert_payload_allocations(
        "narrow_to_c32",
        &small_c64,
        &large_c64,
        lengths,
        &[std::mem::size_of::<Complex32>()],
        TensorMap::narrow_to_c32,
    );

    // A lazy-adjoint source: the result is owned, at the cost of an
    // operation-local materialization in the source dtype plus the output —
    // two payload-sized allocations.
    let (small_lazy, large_lazy) = (small.adjoint().unwrap(), large.adjoint().unwrap());
    assert_payload_allocations(
        "lazy adjoint to_f64",
        &small_lazy,
        &large_lazy,
        lengths,
        &[std::mem::size_of::<f32>(), std::mem::size_of::<f64>()],
        TensorMap::to_f64,
    );
}

/// A converted lazy adjoint is owned, so operations that reject lazy
/// adjoints accept it (#1446 review: `diagview` on U(1) returned `Ok` on main
/// and must keep doing so).
#[test]
fn converted_lazy_adjoints_are_owned() {
    let leg = u1_leg();
    let bond: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime(), [&leg], [&leg], 11).unwrap();
    let lazy = bond.adjoint().unwrap();
    assert!(
        lazy.diagview().is_err(),
        "the fixture must be a lazy adjoint"
    );
    assert_eq!(
        lazy.to_c64().diagview().unwrap().len(),
        bond.diagview().unwrap().len()
    );
    lazy.narrow_to_f32().diagview().unwrap();
    lazy.narrow_to_f32().to_f64().diagview().unwrap();
    lazy.narrow_to_f32().to_c32().diagview().unwrap();
    lazy.to_c64().narrow_to_c32().diagview().unwrap();
    lazy.to_c64()
        .adjoint()
        .unwrap()
        .narrow_to_c32()
        .to_c64()
        .diagview()
        .unwrap();

    // A `permute_overwrite_into` source must be owned.
    let tensor: TensorMap<U1FusionRule, f32> = filled!([&leg, &leg], [&leg]);
    let source = tensor.adjoint().unwrap().to_f64();
    let expected = source.permute(&[2], &[0, 1]).unwrap();
    let mut destination = expected.zeros_like();
    source
        .permute_overwrite_into(&mut destination, &[2], &[0, 1], 1.0)
        .unwrap();
    assert_eq!(f64_bits(destination.data()), f64_bits(expected.data()));
}

/// Checked-Generic provider: SU(3) with an outer-multiplicity vertex.
#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_su3_conversions_are_exact_and_keep_structure() {
    use std::sync::Arc;
    use tenet::typed::SUNFusionRule;

    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(provider, [(vec![2i64, 2], 2)]).unwrap();
    let source = filled!([&leg, &leg], [&leg, &leg]);
    assert!(
        (0..source.block_count()).any(|index| source
            .block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() > 1)),
        "the fixture must carry a Generic multiplicity vertex"
    );
    assert_all_conversions!("SU(3) dense", source.clone());
    assert_all_conversions!("SU(3) lazy adjoint", source.adjoint().unwrap());
    let (_, complex, _, _) = assert_adjoint_conversions!("SU(3) dense adjoint", source);
    // Checked-Generic factorizations reject lazy adjoints, so the converted
    // adjoint must come out owned and factorizable.
    complex.qr_compact().unwrap();
}

/// Checked-Generic compact diagonal under a lazy adjoint (#1446 review):
/// the conversion must come out as the owned compact diagonal, keep the
/// structural zeros `+0 + 0i`, and stay factorizable.
#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_diagonal_adjoint_converts_to_an_owned_compact_diagonal() {
    use std::sync::Arc;
    use tenet::typed::SUNFusionRule;

    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new_with_arc(provider, [(vec![2i64, 1], 3), (vec![0, 0], 2)]).unwrap();
    let spectra = [
        SectorSpectrum {
            sector: vec![2i64, 1],
            values: vec![f32::INFINITY, f32::from_bits(0x7fc0_1234), -0.0],
        },
        SectorSpectrum {
            sector: vec![0, 0],
            values: vec![f32::NEG_INFINITY, f32::from_bits(1)],
        },
    ];
    let source: TensorMap<_, f32> = TensorMap::diagonal(&runtime(), &leg, spectra).unwrap();
    assert!(source.data().iter().any(|value| value.is_nan()));
    assert!(source.data().iter().any(|value| value.is_infinite()));

    let (to_f64, to_c64, c32_to_c64, narrowed_c) =
        assert_adjoint_conversions!("SU(3) diagonal adjoint", source.clone(), true);
    for converted in [&to_f64.to_c64(), &to_c64] {
        assert!(converted.diagonal_spectrum().unwrap().is_some());
    }
    // The stored spectrum of the complex conversions is conj(convert(value)),
    // bitwise.
    let factor = Complex32::new(0.75, -1.5);
    let stored = source
        .to_c32()
        .scale(factor)
        .diagonal_spectrum()
        .unwrap()
        .unwrap();
    let got = c32_to_c64.diagonal_spectrum().unwrap().unwrap();
    for entry in &got {
        let parent = stored.iter().find(|p| p.sector == entry.sector).unwrap();
        let expected: Vec<Complex64> = parent
            .values
            .iter()
            .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)).conj())
            .collect();
        assert_eq!(c64_bits(&entry.values), c64_bits(&expected));
    }
    let stored = source
        .to_c64()
        .scale(Complex64::new(0.75, -1.5))
        .diagonal_spectrum()
        .unwrap()
        .unwrap();
    let got = narrowed_c.diagonal_spectrum().unwrap().unwrap();
    for entry in &got {
        let parent = stored.iter().find(|p| p.sector == entry.sector).unwrap();
        let expected: Vec<Complex32> = parent
            .values
            .iter()
            .map(|v| Complex32::new(narrow(v.re), narrow(-v.im)))
            .collect();
        assert_eq!(c32_bits(&entry.values), c32_bits(&expected));
    }

    // The review repro: finite f64 values, `to_c64` of the lazy adjoint.
    let finite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime(),
        &leg,
        [
            SectorSpectrum {
                sector: vec![2i64, 1],
                values: vec![1.5, -2.0, 0.25],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![3.0, -0.5],
            },
        ],
    )
    .unwrap();
    let lazy = finite.adjoint().unwrap();
    let converted = lazy.to_c64();
    let expected: Vec<Complex64> = lazy
        .data()
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    assert_eq!(c64_bits(converted.data()), c64_bits(&expected));
    converted.qr_compact().unwrap();
}
