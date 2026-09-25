//! `TensorMap::materialize` (#1514), TensorKit `copy(t)`.
//!
//! The oracle is the pre-existing owning route `adj.zeros_like().absorb(&adj)`,
//! which reaches the payload through a different entry point (`absorb`'s
//! min-prefix block copy over two independent materializations). Values must
//! agree bit for bit: both sides are a conjugating copy with no arithmetic.

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{Runtime, TensorScalar};
use tenet::typed::{GradedSpace, NetworkReuseClass, SectorSpectrum, TensorMap};

trait Bits: TensorScalar + Copy + std::fmt::Debug {
    fn bits(self) -> (u64, u64);
}

impl Bits for f64 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl Bits for Complex64 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

macro_rules! assert_materializes_like_absorb {
    ($tensor:expr, $what:expr) => {{
        let tensor = &$tensor;
        let what: &str = $what;
        assert!(tensor.block_count() >= 2, "{what}: multi-block fixture");
        let lazy = tensor.adjoint().unwrap();
        assert!(
            lazy.network_reuse_class(false) == NetworkReuseClass::LazyAdjoint,
            "{what}: fixture must be a lazy adjoint"
        );
        let expected = lazy.zeros_like().absorb(&lazy).unwrap();
        let owned = lazy.materialize().unwrap();
        assert!(
            owned.network_reuse_class(false) == NetworkReuseClass::OwnedDense,
            "{what}: result must be owned dense"
        );
        assert_eq!(owned.codomain(), lazy.codomain(), "{what}: codomain");
        assert_eq!(owned.domain(), lazy.domain(), "{what}: domain");
        assert_eq!(bits(owned.data()), bits(expected.data()), "{what}: payload");
        // Non-vacuity: the dagger really moved or conjugated entries.
        assert_ne!(bits(owned.data()), bits(tensor.data()), "{what}: payload");
        // Round trip: materializing the adjoint of the owned result restores
        // the original payload.
        let back = owned.adjoint().unwrap().materialize().unwrap();
        assert_eq!(bits(back.data()), bits(tensor.data()), "{what}: round trip");
    }};
}

fn bits<D: Bits>(data: &[D]) -> Vec<(u64, u64)> {
    data.iter().map(|value| value.bits()).collect()
}

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
            (SU2Irrep::from_twice_spin(2), 2),
        ],
    )
    .unwrap()
}

macro_rules! fz2_u1_leg {
    () => {
        GradedSpace::try_new_with_arc(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2),
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
            ],
        )
        .unwrap()
    };
}

macro_rules! check_both_dtypes {
    ($what:expr, $leg:expr) => {{
        let runtime = runtime();
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        // Rank 2 <- 2 with a dual leg on each side, so the dagger swaps
        // multi-leg trees and dualities, not just one matrix block.
        let real: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&leg, &dual], [&leg, &dual], 1514).unwrap();
        assert_materializes_like_absorb!(real, concat!($what, "/f64"));
        let complex: TensorMap<_, Complex64> =
            TensorMap::rand_with_seed(&runtime, [&leg, &dual], [&dual, &leg], 1515).unwrap();
        assert_materializes_like_absorb!(complex, concat!($what, "/c64"));
        // Uneven 1 <- 2 split: codomain and domain ranks trade places.
        let uneven: TensorMap<_, Complex64> =
            TensorMap::rand_with_seed(&runtime, [&leg], [&dual, &leg], 1516).unwrap();
        assert_materializes_like_absorb!(uneven, concat!($what, "/c64 1<-2"));
    }};
}

#[test]
fn materialize_matches_absorb_for_u1() {
    check_both_dtypes!("U1", u1_leg());
}

#[test]
fn materialize_matches_absorb_for_su2() {
    check_both_dtypes!("SU2", su2_leg());
}

#[test]
fn materialize_matches_absorb_for_fermionic_z2_times_u1() {
    check_both_dtypes!("fZ2xU1", fz2_u1_leg!());
}

#[test]
fn materialize_reuses_a_published_lazy_materialization() {
    let runtime = runtime();
    let leg = u1_leg();
    let tensor: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 7).unwrap();
    let lazy = tensor.adjoint().unwrap();
    let published = lazy.data().as_ptr();
    assert_eq!(lazy.materialize().unwrap().data().as_ptr(), published);
}

#[test]
fn materialize_keeps_owned_and_compact_representations() {
    let runtime = runtime();
    let leg = u1_leg();
    let dense: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 8).unwrap();
    // An owned tensor is shared, not copied: bodies are copy-on-write.
    assert_eq!(
        dense.materialize().unwrap().data().as_ptr(),
        dense.data().as_ptr()
    );

    // TensorKit `copy(::DiagonalTensorMap)` stays diagonal; so does this.
    let diagonal: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [(-1, 2), (0, 1), (1, 3)].map(|(charge, n)| SectorSpectrum {
            sector: U1Irrep::new(charge),
            values: (0..n)
                .map(|i| Complex64::new(1.0 + i as f64, -0.5 * charge as f64))
                .collect(),
        }),
    )
    .unwrap();
    let copied = diagonal.materialize().unwrap();
    assert!(copied.network_reuse_class(false) == NetworkReuseClass::Compact);
    // `adjoint` of a compact diagonal is already the owned conjugated
    // diagonal (TensorKit `adjoint(::DiagonalTensorMap)`), never lazy.
    let adjoint = diagonal.adjoint().unwrap();
    assert!(
        adjoint.materialize().unwrap().network_reuse_class(false) == NetworkReuseClass::Compact
    );
}

// Independent oracle (reviewer, #1522): the physical-basis expansion of
// `adjoint().materialize()` is the conjugate transpose of the parent's
// physical expansion. This does not go through the reduced-block adjoint map.

trait Conjugate: Copy {
    fn conjugate(self) -> Self;
    fn distance(self, other: Self) -> f64;
}

impl Conjugate for f64 {
    fn conjugate(self) -> Self {
        self
    }
    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }
}

impl Conjugate for Complex64 {
    fn conjugate(self) -> Self {
        self.conj()
    }
    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

/// Returns how many entries would also match *without* conjugation, so a
/// complex caller can prove the conjugation is observable.
fn assert_physical_dagger<D: Conjugate>(
    what: &str,
    shape: &[usize],
    nout: usize,
    parent: &[D],
    adjoint_shape: &[usize],
    adjoint: &[D],
) -> usize {
    let rows: usize = shape[..nout].iter().product();
    let cols: usize = shape[nout..].iter().product();
    let mut expected_shape = shape[nout..].to_vec();
    expected_shape.extend_from_slice(&shape[..nout]);
    assert_eq!(adjoint_shape, expected_shape.as_slice(), "{what}: shape");
    assert_eq!(adjoint.len(), rows * cols, "{what}: length");
    let mut unconjugated_matches = 0;
    for row in 0..rows {
        for col in 0..cols {
            let parent_entry = parent[row + rows * col];
            let adjoint_entry = adjoint[col + cols * row];
            assert!(
                parent_entry.conjugate().distance(adjoint_entry) < 1e-12,
                "{what}: entry ({row},{col})"
            );
            if parent_entry.distance(adjoint_entry) < 1e-12 {
                unconjugated_matches += 1;
            }
        }
    }
    unconjugated_matches
}

macro_rules! dense_oracle {
    ($leg:expr, $what:expr, $dtype:ty, $codomain:expr, $domain:expr, $seed:expr) => {{
        let runtime = runtime();
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let tensor: TensorMap<_, $dtype> = TensorMap::rand_with_seed(
            &runtime,
            $codomain(&leg, &dual),
            $domain(&leg, &dual),
            $seed,
        )
        .unwrap();
        let lazy = tensor.adjoint().unwrap();
        assert!(lazy.network_reuse_class(false) == NetworkReuseClass::LazyAdjoint);
        let owned = lazy.materialize().unwrap();
        let parent = tensor.to_physical_dense().unwrap();
        let adjoint = owned.to_physical_dense().unwrap();
        let unconjugated = assert_physical_dagger::<$dtype>(
            $what,
            &parent.shape,
            tensor.codomain_rank(),
            &parent.data,
            &adjoint.shape,
            &adjoint.data,
        );
        (unconjugated, adjoint.data.len())
    }};
}

#[test]
fn materialize_matches_the_physical_conjugate_transpose() {
    for (unconjugated, len) in [
        dense_oracle!(
            u1_leg(),
            "U1 c64 2<-2",
            Complex64,
            |l, d| [l, d],
            |d, l| [l, d],
            2
        ),
        dense_oracle!(
            u1_leg(),
            "U1 c64 1<-2",
            Complex64,
            |l, _d| [l],
            |l, d| [d, l],
            3
        ),
        dense_oracle!(
            u1_leg(),
            "U1 c64 3<-1",
            Complex64,
            |l, d| [d, l, l],
            |_l, d| [d],
            4
        ),
        dense_oracle!(
            su2_leg(),
            "SU2 c64 2<-2",
            Complex64,
            |l, d| [l, d],
            |d, l| [l, d],
            6
        ),
        dense_oracle!(
            su2_leg(),
            "SU2 c64 1<-2",
            Complex64,
            |l, _d| [l],
            |l, d| [d, l],
            7
        ),
        dense_oracle!(
            su2_leg(),
            "SU2 c64 3<-1",
            Complex64,
            |l, d| [d, l, l],
            |_l, d| [d],
            8
        ),
    ] {
        // Negative control: a transpose without conjugation must not pass.
        assert!(unconjugated < len, "conjugation is not observable");
    }
    dense_oracle!(
        u1_leg(),
        "U1 f64 2<-2",
        f64,
        |l, d| [l, d],
        |l, d| [l, d],
        1
    );
    dense_oracle!(
        su2_leg(),
        "SU2 f64 2<-2",
        f64,
        |l, d| [l, d],
        |l, d| [l, d],
        5
    );
}
