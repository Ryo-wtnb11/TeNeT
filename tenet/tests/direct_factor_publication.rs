//! Compact LQ and checked-Generic polar write their factors straight into the
//! output storage (#1478): LQ appends each sector's adjoint instead of
//! zero-filling first, and polar runs its GEMMs into the output regions
//! instead of per-sector temporaries that are then copied.

#[path = "../../tests/support/numerics.rs"]
mod numerics;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use tenet::prelude::{Complex32, Complex64, Runtime};
use tenet::typed::{GradedSpace, TensorMap, TensorScalar};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static ZEROED_BYTES: Cell<usize> = const { Cell::new(0) };
}

fn record(layout: Layout, zeroed: bool) {
    if COUNTING.get() {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        BYTES.set(BYTES.get() + layout.size());
        if zeroed {
            ZEROED_BYTES.set(ZEROED_BYTES.get() + layout.size());
        }
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout, false);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout, true);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            record(
                Layout::from_size_align(new_size, layout.align()).unwrap(),
                false,
            );
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Debug)]
#[cfg_attr(
    not(feature = "racah-generated"),
    expect(
        dead_code,
        reason = "only the checked-Generic polar contract reads calls and bytes"
    )
)]
struct Counts {
    calls: usize,
    bytes: usize,
    zeroed_bytes: usize,
}

fn measured<T>(operation: impl FnOnce() -> T) -> (T, Counts) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    ZEROED_BYTES.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    let counts = Counts {
        calls: ALLOCATIONS.get(),
        bytes: BYTES.get(),
        zeroed_bytes: ZEROED_BYTES.get(),
    };
    (value, counts)
}

trait Scalar: TensorScalar + numerics::Numeric {
    const ZERO: Self;
    const ONE: Self;
    const MINUS_ONE: Self;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MINUS_ONE: Self = -1.0;
}

impl Scalar for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    const MINUS_ONE: Self = Complex64::new(-1.0, 0.0);
}

/// An owned copy of a (possibly lazy) adjoint: checked-Generic `compose`
/// accepts owned operands only.
macro_rules! owned_adjoint {
    ($d:ty, $t:expr) => {{
        let adjoint = $t.adjoint().unwrap();
        adjoint
            .add(&adjoint, <$d as Scalar>::ONE, <$d as Scalar>::ZERO)
            .unwrap()
    }};
}

/// `‖actual − expected‖ ≤ tolerance(terms, ‖expected‖)`; `terms` bounds the
/// floating terms of one entry by the source payload length, and the
/// conditioning factor is one (backward-stable factorizations of random
/// fixtures).
macro_rules! assert_residual {
    ($what:expr, $d:ty, $actual:expr, $expected:expr, $terms:expr) => {{
        let residual = $actual
            .add($expected, <$d as Scalar>::ONE, <$d as Scalar>::MINUS_ONE)
            .unwrap()
            .norm()
            .unwrap();
        let bound = numerics::tolerance::<$d>($terms, $expected.norm().unwrap());
        assert!(
            residual <= bound,
            "{}: residual {residual:e} exceeds {bound:e}",
            $what
        );
    }};
}

/// `A = L Q` and `A Qᴴ = L`, which with a full-rank `L` is `Q Qᴴ = 1`
/// (independent reconstruction/orthogonality oracle).
macro_rules! assert_lq {
    ($d:ty, $a:expr) => {{
        let a = $a;
        let (l, q) = a.lq_compact().unwrap();
        let terms = a.data().len();
        assert_residual!("A = L Q", $d, &l.compose(&q).unwrap(), &a, terms);
        let aqh = a.compose(&owned_adjoint!($d, q)).unwrap();
        assert_residual!("A Qᴴ = L", $d, &aqh, &l, terms);
    }};
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

macro_rules! lq_cases {
    ($d:ty, $runtime:expr) => {{
        let u1 = u1_leg();
        let a: TensorMap<_, $d> =
            TensorMap::rand_with_seed($runtime, [&u1], [&u1, &u1], 1478).unwrap();
        assert_lq!($d, a);
        let su2 = su2_leg();
        let a: TensorMap<_, $d> =
            TensorMap::rand_with_seed($runtime, [&su2], [&su2, &su2], 1479).unwrap();
        assert_lq!($d, a);
    }};
}

#[test]
fn multiplicity_free_compact_lq_matches_reconstruction_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    lq_cases!(f64, &runtime);
    lq_cases!(Complex64, &runtime);
}

// Byte-traffic contract: TeNeT writes the compact LQ outputs and the adjoint
// scratch once, so it requests no zeroed storage for them. The only zeroed
// request left is the dense backend's R factor (Tenferro
// `faer_linalg::upper_triangle_vec_from_mat`), whose total over sectors is
// the payload of L. Before #1478 both outputs and the scratch also came from
// `vec![0.0; len]`, i.e. `alloc_zeroed`, which this counter observes for f64:
// 16928 zeroed bytes before, 4872 (= L) after.
#[test]
fn compact_lq_requests_no_zeroed_output_storage() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = u1_leg();
    let a: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 1480).unwrap();
    let warm = a.lq_compact().unwrap();
    let ((l, q), counts) = measured(|| a.lq_compact().unwrap());
    black_box((&warm, &q));
    let backend_r_bytes = std::mem::size_of_val(l.data());
    assert!(
        counts.zeroed_bytes <= backend_r_bytes,
        "{counts:?}, L bytes {backend_r_bytes}"
    );
}

#[cfg(feature = "racah-generated")]
mod checked_generic {
    use super::*;
    use tenet::typed::SUNFusionRule;

    fn su3_leg(
        provider: &Arc<SUNFusionRule>,
        irreps: &[(Vec<i64>, usize)],
    ) -> GradedSpace<SUNFusionRule> {
        GradedSpace::try_new_with_arc(Arc::clone(provider), irreps.to_vec()).unwrap()
    }

    /// Four coupled sectors, rows >= columns in each.
    fn tall_legs(
        provider: &Arc<SUNFusionRule>,
    ) -> (GradedSpace<SUNFusionRule>, GradedSpace<SUNFusionRule>) {
        let irreps = [vec![0i64, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let rows = irreps.iter().cloned().zip([1, 2, 3, 2]).collect::<Vec<_>>();
        let cols = irreps.iter().cloned().zip([1, 1, 2, 2]).collect::<Vec<_>>();
        (su3_leg(provider, &rows), su3_leg(provider, &cols))
    }

    /// Left: `A = W P`, right: `A = P W`, with `W` an isometry (`Wᴴ W = 1`
    /// left, `W Wᴴ = 1` right: every eigenvalue of the Gram matrix is one)
    /// and `P` Hermitian positive semidefinite (eigenvalues `>= -tol`). This
    /// excludes trivial splittings such as `W = A`, `P = 1`.
    macro_rules! assert_polar {
        ($d:ty, $a:expr, $left:expr) => {{
            let a = $a;
            let terms = a.data().len();
            let (w, p, product, gram) = if $left {
                let (w, p) = a.left_polar().unwrap();
                let product = w.compose(&p).unwrap();
                let gram = owned_adjoint!($d, w).compose(&w).unwrap();
                (w, p, product, gram)
            } else {
                let (p, w) = a.right_polar().unwrap();
                let product = p.compose(&w).unwrap();
                let gram = w.compose(&owned_adjoint!($d, w)).unwrap();
                (w, p, product, gram)
            };
            assert!(!w.data().is_empty());
            assert_residual!("A = polar product", $d, &product, a, terms);
            assert_residual!("P = Pᴴ", $d, &owned_adjoint!($d, p), &p, terms);
            let one = numerics::tolerance::<$d>(terms, 1.0);
            for spectrum in gram.eigh_vals().unwrap() {
                for value in spectrum.values {
                    assert!(
                        (value - 1.0).abs() <= one,
                        "W is not an isometry: Gram eigenvalue {value:e}"
                    );
                }
            }
            let psd = numerics::tolerance::<$d>(terms, p.norm().unwrap());
            let mut p_values = 0;
            for spectrum in p.eigh_vals().unwrap() {
                for value in spectrum.values {
                    p_values += 1;
                    assert!(value >= -psd, "P is not PSD: eigenvalue {value:e}");
                }
            }
            assert!(p_values > 0);
        }};
    }

    macro_rules! cases {
        ($d:ty, $runtime:expr) => {{
            let provider = Arc::new(SUNFusionRule::new(3).unwrap());
            let (rows, cols) = tall_legs(&provider);
            let tall: TensorMap<_, $d> =
                TensorMap::rand_with_seed($runtime, [&rows], [&cols], 1481).unwrap();
            assert_polar!($d, &tall, true);
            let wide: TensorMap<_, $d> =
                TensorMap::rand_with_seed($runtime, [&cols], [&rows], 1482).unwrap();
            assert_polar!($d, &wide, false);
            assert_lq!($d, wide);
            // Vertex multiplicity: 8 ⊗ 8 contains 8 twice.
            let adjoint = su3_leg(&provider, &[(vec![1, 1], 2)]);
            let single = su3_leg(&provider, &[(vec![1, 1], 1)]);
            let tall: TensorMap<_, $d> =
                TensorMap::rand_with_seed($runtime, [&adjoint, &adjoint], [&single], 1483).unwrap();
            assert_eq!(tall.block_count(), 2);
            assert_polar!($d, &tall, true);
            let wide: TensorMap<_, $d> =
                TensorMap::rand_with_seed($runtime, [&single], [&adjoint, &adjoint], 1484).unwrap();
            assert_polar!($d, &wide, false);
            assert_lq!($d, wide);
        }};
    }

    #[test]
    fn checked_generic_polar_and_lq_match_reconstruction_oracles() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        cases!(f64, &runtime);
        cases!(Complex64, &runtime);
    }

    // Allocation contract: polar writes W and P into their output regions.
    // Before #1478 each of the four coupled sectors added three temporaries
    // (W, P and a clone of the scaled factor) that were then copied out.
    // Caller-thread counts of the second call, pinned faer provider, one dense
    // thread: 341 calls / 53362 bytes before, 329 / 53098 after (-3 per sector).
    #[test]
    fn checked_generic_polar_allocates_no_per_sector_temporaries() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let (rows, cols) = tall_legs(&provider);
        let a: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&rows], [&cols], 1485).unwrap();
        let warm = a.left_polar().unwrap();
        let ((w, p), counts) = measured(|| a.left_polar().unwrap());
        black_box((&warm, &w, &p));
        assert!(counts.calls <= 329, "{counts:?}");
        assert!(counts.bytes <= 53098, "{counts:?}");
    }

    // Byte contract (#1494): the leg-degeneracy (facade) layout of a rank-3
    // and a rank-4 multiplicity map is a coupled-sector tiling, so QR, SVD,
    // LQ and `svd_vals` read the payload in place. Before, `FusionTreeKey`
    // Ord admission rejected the facade tree order and each call first packed
    // a copy of the whole payload. Caller-thread calls / bytes of the second
    // call, pinned faer provider, one dense thread, before -> after:
    //   rank 3: QR 351/47555 -> 325/43275, SVD 420/59541 -> 394/55261,
    //           LQ 353/48811 -> 327/44531, svd_vals 40/6616 -> 14/2336;
    //   rank 4: QR 647/121808 -> 583/111248, SVD 778/158444 -> 714/147884,
    //           LQ 657/124632 -> 593/114072, svd_vals 97/21544 -> 33/10984.
    // `svd_vals` no longer zero-fills a packed matrix at all.
    #[test]
    fn checked_generic_facade_factorizations_read_the_payload_in_place() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let leg = su3_leg(&provider, &[(vec![1, 1], 2), (vec![0, 0], 1)]);
        let rank3: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 1494).unwrap();
        let rank4: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 1495).unwrap();
        let budgets = [
            [(325, 43275), (394, 55261), (327, 44531), (14, 2336)],
            [(583, 111248), (714, 147884), (593, 114072), (33, 10984)],
        ];
        for (a, budget) in [&rank3, &rank4].into_iter().zip(budgets) {
            let warm = (a.qr_compact().unwrap(), a.svd_compact().unwrap());
            let (_, qr) = measured(|| black_box(a.qr_compact().unwrap()));
            let (_, svd) = measured(|| black_box(a.svd_compact().unwrap()));
            let (_, lq) = measured(|| black_box(a.lq_compact().unwrap()));
            let (_, vals) = measured(|| black_box(a.svd_vals().unwrap()));
            black_box(&warm);
            for (counts, (calls, bytes)) in [&qr, &svd, &lq, &vals].into_iter().zip(budget) {
                assert!(counts.calls <= calls && counts.bytes <= bytes, "{counts:?}");
            }
            assert_eq!(vals.zeroed_bytes, 0, "{vals:?}");
        }
    }
}
