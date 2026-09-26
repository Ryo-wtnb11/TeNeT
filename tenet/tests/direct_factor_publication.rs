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
use tenet::typed::{GradedSpace, Lq, TensorMap, TensorScalar};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static ZEROED_BYTES: Cell<usize> = const { Cell::new(0) };
}

fn record(layout: Layout, zeroed: bool) {
    if COUNTING.get() {
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
        reason = "only the checked-Generic contracts read total bytes"
    )
)]
struct Counts {
    bytes: usize,
    zeroed_bytes: usize,
}

fn measured<T>(operation: impl FnOnce() -> T) -> (T, Counts) {
    BYTES.set(0);
    ZEROED_BYTES.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    let counts = Counts {
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
            .axpby(<$d as Scalar>::ONE, &adjoint, <$d as Scalar>::ZERO)
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
            .axpby(<$d as Scalar>::ONE, $expected, <$d as Scalar>::MINUS_ONE)
            .unwrap()
            .norm(2.0)
            .unwrap();
        let bound = numerics::tolerance::<$d>($terms, $expected.norm(2.0).unwrap());
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
        let Lq { l, q } = a.lq_compact().unwrap();
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
    let (Lq { l, q }, counts) = measured(|| a.lq_compact().unwrap());
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
    use tenet::typed::{LeftPolar, RightPolar, Svd};

    fn su3_leg(
        provider: &Arc<SUNFusionRule>,
        irreps: &[(Vec<i64>, usize)],
    ) -> GradedSpace<SUNFusionRule> {
        GradedSpace::try_new_with_arc(Arc::clone(provider), irreps.to_vec()).unwrap()
    }

    /// Four coupled sectors, rows >= columns in each; every degeneracy is
    /// multiplied by `scale`, which leaves the tree structure unchanged.
    fn tall_legs(
        provider: &Arc<SUNFusionRule>,
        scale: usize,
    ) -> (GradedSpace<SUNFusionRule>, GradedSpace<SUNFusionRule>) {
        let irreps = [vec![0i64, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let rows = irreps
            .iter()
            .cloned()
            .zip([1, 2, 3, 2].map(|d| d * scale))
            .collect::<Vec<_>>();
        let cols = irreps
            .iter()
            .cloned()
            .zip([1, 1, 2, 2].map(|d| d * scale))
            .collect::<Vec<_>>();
        (su3_leg(provider, &rows), su3_leg(provider, &cols))
    }

    fn payload_bytes(tensor: &TensorMap<SUNFusionRule, f64>) -> usize {
        std::mem::size_of_val(tensor.data())
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
                let LeftPolar { w, p } = a.left_polar().unwrap();
                let product = w.compose(&p).unwrap();
                let gram = owned_adjoint!($d, w).compose(&w).unwrap();
                (w, p, product, gram)
            } else {
                let RightPolar { p, wh: w } = a.right_polar().unwrap();
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
            let psd = numerics::tolerance::<$d>(terms, p.norm(2.0).unwrap());
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
            let (rows, cols) = tall_legs(&provider, 1);
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
            assert_eq!(tall.subblock_count(), 2);
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
    // Before #1478 each coupled sector added three temporaries (W, P and a
    // clone of the scaled factor) that were then copied out. Both polar and
    // `svd_compact` run the same staged per-sector SVD, so absolute counts
    // include backend workspace whose size depends on the dense kernel's SIMD
    // width (macOS and Linux differ by tens of bytes). Scaling every
    // degeneracy keeps the tree structure (and all metadata) fixed, so the
    // growth of the bytes a call allocates beyond its outputs is its
    // payload-proportional scratch. Polar's must not exceed the SVD stage's,
    // which it would by the size of the temporaries before #1478.
    #[test]
    fn checked_generic_polar_allocates_no_per_sector_temporaries() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let scratch = |scale| {
            let (rows, cols) = tall_legs(&provider, scale);
            let a: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&runtime, [&rows], [&cols], 1485).unwrap();
            let warm = (a.left_polar().unwrap(), a.svd_compact().unwrap());
            let (LeftPolar { w, p }, polar) = measured(|| a.left_polar().unwrap());
            let (Svd { u, s, vh }, svd) = measured(|| a.svd_compact().unwrap());
            black_box(&warm);
            let polar_outputs = payload_bytes(&w) + payload_bytes(&p);
            let svd_outputs = payload_bytes(&u) + payload_bytes(&s) + payload_bytes(&vh);
            (polar.bytes - polar_outputs, svd.bytes - svd_outputs)
        };
        let (polar_small, svd_small) = scratch(1);
        let (polar_large, svd_large) = scratch(3);
        assert!(
            polar_large - polar_small <= svd_large - svd_small,
            "polar scratch {polar_small} -> {polar_large}, SVD stage {svd_small} -> {svd_large}"
        );
    }

    // Byte contract (#1494): the leg-degeneracy (facade) layout of a rank-3
    // and a rank-4 multiplicity map is a coupled-sector tiling, so QR, SVD,
    // LQ and `svd_vals` read the payload in place. Before, `FusionTreeKey`
    // Ord admission rejected the facade tree order and each call first packed
    // a copy of the whole payload. The control is the rank-2 map between the
    // fused legs: it has the same coupled blocks, hence the same dense kernel
    // work and output sizes, and was always read in place. The facade call
    // still allocates more tree metadata than the control, but scaling every
    // degeneracy leaves that metadata fixed, so the facade's excess over the
    // control must not grow with the payload; a packed copy grows it by the
    // whole payload growth.
    #[test]
    fn checked_generic_facade_factorizations_read_the_payload_in_place() {
        type Tensor = TensorMap<SUNFusionRule, f64>;
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let excess = |scale: usize, rank4: bool| {
            let leg = su3_leg(&provider, &[(vec![1, 1], 2 * scale), (vec![0, 0], scale)]);
            let fused = leg.fuse(&leg).unwrap();
            let (facade, control): (Tensor, Tensor) = if rank4 {
                (
                    TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 1495).unwrap(),
                    TensorMap::rand_with_seed(&runtime, [&fused], [&fused], 1495).unwrap(),
                )
            } else {
                (
                    TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 1494).unwrap(),
                    TensorMap::rand_with_seed(&runtime, [&fused], [&leg], 1494).unwrap(),
                )
            };
            assert_eq!(payload_bytes(&facade), payload_bytes(&control));
            let counts = |a: &Tensor| {
                black_box((a.qr_compact().unwrap(), a.svd_compact().unwrap()));
                black_box((a.lq_compact().unwrap(), a.svd_vals().unwrap()));
                [
                    measured(|| black_box(a.qr_compact().unwrap())).1,
                    measured(|| black_box(a.svd_compact().unwrap())).1,
                    measured(|| black_box(a.lq_compact().unwrap())).1,
                    measured(|| black_box(a.svd_vals().unwrap())).1,
                ]
            };
            let (facade_counts, control_counts) = (counts(&facade), counts(&control));
            assert_eq!(facade_counts[3].zeroed_bytes, 0, "{:?}", facade_counts[3]);
            let excess = std::array::from_fn::<isize, 4, _>(|op| {
                facade_counts[op].bytes as isize - control_counts[op].bytes as isize
            });
            (payload_bytes(&facade), excess)
        };
        for rank4 in [false, true] {
            let (small_payload, small) = excess(1, rank4);
            let (large_payload, large) = excess(2, rank4);
            let payload_growth = (large_payload - small_payload) as isize;
            for (op, name) in ["qr_compact", "svd_compact", "lq_compact", "svd_vals"]
                .into_iter()
                .enumerate()
            {
                assert!(
                    large[op] - small[op] < payload_growth,
                    "rank4 {rank4} {name}: facade excess {} -> {}, payload growth {payload_growth}",
                    small[op],
                    large[op]
                );
            }
        }
    }
}
