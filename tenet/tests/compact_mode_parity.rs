//! A compact diagonal (TensorKit `DiagonalTensorMap`) answers every operation
//! the same way in both admission modes (#1866): the same storage form, the
//! same values, and the dense route's answer once materialized.
//!
//! Twist (leaf A). TensorKit `cfaa073e` `twist(t, inds; inv)`
//! (`src/tensors/indexmanipulations.jl:90`) allocates `similar(t, …)`, which
//! for a `DiagonalTensorMap` is `similar_diagonal` (`src/tensors/diagonal.jl:85`),
//! so the result stays diagonal; `twist!` (`indexmanipulations.jl:62`) scales
//! each block by `Π_{i ∈ inds} θ(f.uncoupled[i])`, conjugated for `inv`. On a
//! bond every uncoupled sector is the coupled `c`, so the hand oracle below is
//! `s_c[i] · θ(c)^{|inds|}` for the real θ of these rules. QSpace has no
//! twist.
//!
//! Capability absences (compile-time, not counted as passing cells): checked
//! × CUDA and Fibonacci × CUDA (CUDA ops are bounded
//! `MultiplicityFreeRigidSymbols<Scalar = f64>`), and Fibonacci twist (the
//! multiplicity-free twist needs `Scalar = f64`).

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FermionParityFusionRule, FusionStyleKind, GenericFArray,
    GenericRMatrix, RuleIdentity, SectorId, SectorVec, TypedSectorAdmission, Z2Irrep,
};
use tenet::typed::{Complex64, Direction, GradedSpace, SectorSpectrum, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::host_runtime;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

const EVEN: SectorId = SectorId::new(0);
const ODD: SectorId = SectorId::new(1);

/// A checked Generic twin of fZ2 (`Z2` fusion, real symbols) whose braiding
/// and odd-sector twist are parameters: fermionic with `θ(odd) = -1` is the
/// checked counterpart of [`FermionParityFusionRule`]; anyonic with a
/// non-unit `θ(odd)` makes the twist a non-sign rescaling.
struct CheckedZ2 {
    braiding: BraidingStyleKind,
    odd_twist: f64,
    twist_queries: AtomicUsize,
}

impl CheckedZ2 {
    fn new(braiding: BraidingStyleKind, odd_twist: f64) -> Self {
        Self {
            braiding,
            odd_twist,
            twist_queries: AtomicUsize::new(0),
        }
    }

    fn admissible(left: SectorId, right: SectorId, coupled: SectorId) -> bool {
        (left.id() ^ right.id()) == coupled.id()
    }
}

#[derive(Debug)]
struct InvalidSector;

impl std::fmt::Display for InvalidSector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("invalid Z2 sector")
    }
}

impl std::error::Error for InvalidSector {}

fn checked_z2(sector: SectorId) -> Result<SectorId, InvalidSector> {
    if sector == EVEN || sector == ODD {
        Ok(sector)
    } else {
        Err(InvalidSector)
    }
}

impl CheckedGenericFusion for CheckedZ2 {
    type Error = InvalidSector;

    fn rule_identity(&self) -> RuleIdentity {
        let mut bytes = vec![self.braiding as u8];
        bytes.extend(self.odd_twist.to_bits().to_le_bytes());
        RuleIdentity::from_canonical_bytes::<Self>(0x1866, Arc::<[u8]>::from(bytes))
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.braiding
    }

    fn vacuum(&self) -> SectorId {
        EVEN
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        checked_z2(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        let coupled = SectorId::new(checked_z2(left)?.id() ^ checked_z2(right)?.id());
        Ok([coupled].into_iter().collect())
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(usize::from(Self::admissible(
            checked_z2(left)?,
            checked_z2(right)?,
            checked_z2(coupled)?,
        )))
    }
}

impl CheckedGenericRigidSymbols for CheckedZ2 {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        let n = |l, r, c| usize::from(Self::admissible(l, r, c));
        let shape = (n(a, b, e), n(e, c, d), n(b, c, f), n(a, f, d));
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        Ok(GenericFArray::new(vec![1.0; len], shape))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        let size = usize::from(Self::admissible(a, b, c));
        let value = if a == ODD && b == ODD { -1.0 } else { 1.0 };
        Ok(GenericRMatrix::new(vec![value; size], size, size))
    }
}

impl CheckedGenericPivotal for CheckedZ2 {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.twist_queries.fetch_add(1, Ordering::Relaxed);
        Ok(if checked_z2(sector)? == ODD {
            self.odd_twist
        } else {
            1.0
        })
    }
}

impl TypedSectorAdmission for CheckedZ2 {
    type Sector = Z2Irrep;
    type Error = InvalidSector;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Z2Irrep) -> Result<SectorId, Self::Error> {
        Ok(SectorId::new(usize::from(sector.parity())))
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Z2Irrep, Self::Error> {
        Ok(Z2Irrep::new(checked_z2(sector)?.id() as u8))
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        checked_z2(sector)
    }
}

/// `(parity, k)`: both sectors, nontrivial and distinct degeneracies.
const SECTORS: [(u8, usize); 2] = [(0, 2), (1, 3)];

/// The stored diagonal; binary fractions, so a ±1 twist is exact.
fn value(parity: u8, i: usize) -> f64 {
    [[0.75, -1.5, 0.0], [0.5, 2.25, -0.125]][usize::from(parity)][i]
}

const LEG_SETS: [&[usize]; 3] = [&[0], &[1], &[0, 1]];

fn close(actual: Complex64, expected: Complex64) -> bool {
    (actual - expected).norm() <= 1e-12 * expected.norm().max(1.0)
}

/// Twists one compact fZ2-shaped diagonal on every leg set in both
/// directions, asserting against the hand oracle and the dense route, and
/// returns the twisted spectra in a fixed order for the cross-mode check.
macro_rules! twist_rows {
    ($rule:expr, $odd_twist:expr, $D:ty, $conv:expr) => {{
        let runtime = host_runtime();
        let conv = $conv;
        let odd_twist: f64 = $odd_twist;
        let leg = GradedSpace::try_new(
            Arc::new($rule),
            SECTORS.iter().map(|&(p, k)| (Z2Irrep::new(p), k)),
        )
        .unwrap();
        let spectrum = SECTORS
            .iter()
            .map(|&(p, k)| SectorSpectrum {
                sector: Z2Irrep::new(p),
                values: (0..k).map(|i| conv(value(p, i))).collect::<Vec<$D>>(),
            })
            .collect::<Vec<_>>();
        let compact = TensorMap::<_, $D>::diagonal(&runtime, &leg, spectrum).unwrap();
        let dense_twin = compact.materialize().unwrap();
        let mut out = Vec::new();
        for legs in LEG_SETS {
            for direction in [Direction::Forward, Direction::Inverse] {
                let twisted = compact.twist(legs, direction).unwrap();
                assert!(
                    twisted.dense_data().is_err(),
                    "twist {legs:?} {direction:?} densified a compact diagonal"
                );
                // TensorKit `twist!`: `inv && (θ = θ')`; a real θ is its own
                // conjugate.
                let power = legs.len() as i32;
                let got = twisted.diagview().unwrap();
                assert_eq!(got.len(), SECTORS.len());
                for (entry, &(p, k)) in got.iter().zip(&SECTORS) {
                    assert_eq!(entry.sector, Z2Irrep::new(p));
                    let theta = if p == 1 { odd_twist } else { 1.0 };
                    let factor = theta.powi(power);
                    for i in 0..k {
                        let expected = Complex64::from(conv(value(p, i))) * factor;
                        assert!(
                            close(Complex64::from(entry.values[i]), expected),
                            "twist {legs:?} {direction:?} sector {p} entry {i}"
                        );
                    }
                }
                let dense = dense_twin.twist(legs, direction).unwrap();
                let materialized = twisted.materialize().unwrap();
                let (lhs, rhs) = (
                    materialized.dense_data().unwrap(),
                    dense.dense_data().unwrap(),
                );
                assert_eq!(lhs.len(), rhs.len());
                for (&a, &b) in lhs.iter().zip(rhs) {
                    assert!(close(Complex64::from(a), Complex64::from(b)));
                }
                out.push(got);
            }
        }
        out
    }};
}

#[test]
fn compact_twist_stays_compact_and_agrees_across_modes() {
    // What: a fermionic compact twist is one mode-free arm. The checked
    // Generic fZ2 twin used to densify; now both modes keep the spectrum and
    // give the identical values.
    let real = |x: f64| x;
    let mf = twist_rows!(FermionParityFusionRule, -1.0, f64, real);
    let checked = twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
        -1.0,
        f64,
        real
    );
    assert_eq!(mf, checked);

    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    let mf = twist_rows!(FermionParityFusionRule, -1.0, Complex64, complex);
    let checked = twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
        -1.0,
        Complex64,
        complex
    );
    assert_eq!(mf, checked);
}

#[test]
fn checked_anyonic_compact_twist_scales_by_theta_powers() {
    // What: a non-unit real θ, which only the checked mode can express here,
    // scales the spectrum by θ^{|legs|}, so a two-leg twist is not the
    // identity on the bond.
    let real = |x: f64| x;
    twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6),
        0.6,
        f64,
        real
    );
    let complex = |x: f64| Complex64::new(-0.5 * x, x);
    twist_rows!(
        CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6),
        0.6,
        Complex64,
        complex
    );
}

#[test]
fn checked_compact_twist_queries_the_provider_like_the_dense_route() {
    // What: the compact arm reuses the staged twist values; it adds no
    // provider query beyond the dense route's.
    let runtime = host_runtime();
    let rule = Arc::new(CheckedZ2::new(BraidingStyleKind::Anyonic, 0.6));
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        SECTORS.iter().map(|&(p, k)| (Z2Irrep::new(p), k)),
    )
    .unwrap();
    let spectrum = SECTORS
        .iter()
        .map(|&(p, k)| SectorSpectrum {
            sector: Z2Irrep::new(p),
            values: (0..k).map(|i| value(p, i)).collect::<Vec<f64>>(),
        })
        .collect::<Vec<_>>();
    let compact = TensorMap::<_, f64>::diagonal(&runtime, &leg, spectrum).unwrap();
    let dense = compact.materialize().unwrap();
    let queries = |tensor: &TensorMap<CheckedZ2, f64>| {
        rule.twist_queries.store(0, Ordering::Relaxed);
        black_box(tensor.twist(&[0, 1], Direction::Forward).unwrap());
        rule.twist_queries.load(Ordering::Relaxed)
    };
    let (compact_queries, dense_queries) = (queries(&compact), queries(&dense));
    assert!(dense_queries > 0);
    assert_eq!(compact_queries, dense_queries);
}

#[cfg(feature = "racah-generated")]
#[test]
fn bosonic_su2_compact_twist_is_the_identity_in_both_modes() {
    // What: SU(2) is bosonic, so twist returns the compact receiver as is in
    // both modes.
    use tenet::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
    let runtime = host_runtime();
    let su2 = [(0usize, 2usize), (1, 3), (2, 1)];
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        su2.iter()
            .map(|&(twice, k)| (SU2Irrep::from_twice_spin(twice), k)),
    )
    .unwrap();
    let checked_leg = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(2).unwrap()),
        su2.iter().map(|&(twice, k)| (vec![twice as i64], k)),
    )
    .unwrap();
    let values = |twice: usize, k: usize| (0..k).map(|i| 0.5 + (twice * 4 + i) as f64).collect();
    let mf = TensorMap::<_, f64>::diagonal(
        &runtime,
        &mf_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(twice),
            values: values(twice, k),
        }),
    )
    .unwrap();
    let checked = TensorMap::<_, f64>::diagonal(
        &runtime,
        &checked_leg,
        su2.iter().map(|&(twice, k)| SectorSpectrum {
            sector: vec![twice as i64],
            values: values(twice, k),
        }),
    )
    .unwrap();
    for legs in LEG_SETS {
        let mf_twisted = mf.twist(legs, Direction::Forward).unwrap();
        let checked_twisted = checked.twist(legs, Direction::Forward).unwrap();
        assert!(mf_twisted.dense_data().is_err());
        assert!(checked_twisted.dense_data().is_err());
        let mf_values: Vec<_> = mf_twisted
            .diagview()
            .unwrap()
            .into_iter()
            .map(|e| e.values)
            .collect();
        let checked_values: Vec<_> = checked_twisted
            .diagview()
            .unwrap()
            .into_iter()
            .map(|e| e.values)
            .collect();
        assert_eq!(mf_values, checked_values);
        assert_eq!(
            mf_values,
            su2.iter()
                .map(|&(twice, k)| values(twice, k))
                .collect::<Vec<Vec<f64>>>()
        );
    }
}

/// k = 128 makes one dense payload (`k² · 8` bytes) dominate the warm
/// staging bytes, as in `typed_diagonal_allocations.rs`.
const K: usize = 128;

fn odd_diagonal<R>(rule: R, k: usize) -> TensorMap<R, f64>
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: tenet::typed::TypedTensorConstructionDispatch<R, f64>,
{
    let runtime = host_runtime();
    let leg = GradedSpace::try_new(Arc::new(rule), [(Z2Irrep::ODD, k)]).unwrap();
    TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Z2Irrep::ODD,
            values: (0..k).map(|i| 1.0 + i as f64).collect(),
        }],
    )
    .unwrap()
}

fn warmed_twist_bytes<R>(tensor: &TensorMap<R, f64>) -> u64
where
    R: TypedSectorAdmission,
    R::Mode: tenet::typed::TypedTensorTwistDispatch<R, f64>,
{
    black_box(tensor.twist(&[0], Direction::Forward).unwrap());
    let (twisted, allocs) =
        counting_alloc::measure(|| black_box(tensor.twist(&[0], Direction::Forward).unwrap()));
    assert!(twisted.dense_data().is_err(), "the twist densified");
    allocs.bytes
}

#[test]
fn compact_twist_allocates_linear_in_the_spectrum_in_both_modes() {
    // What: the twist of a compact diagonal costs Σk, never a k² payload: from
    // k to 2k the bytes grow by at most the spectrum's growth, and stay below
    // one dense payload.
    let _measurement = counting_alloc::serial();
    let dense_payload = (K * K * std::mem::size_of::<f64>()) as u64;
    let spectrum_growth = (2 * K * std::mem::size_of::<f64>()) as u64;
    let rows = [
        (
            "multiplicity-free",
            warmed_twist_bytes(&odd_diagonal(FermionParityFusionRule, K)),
            warmed_twist_bytes(&odd_diagonal(FermionParityFusionRule, 2 * K)),
        ),
        (
            "checked",
            warmed_twist_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                K,
            )),
            warmed_twist_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                2 * K,
            )),
        ),
    ];
    for (mode, small, large) in rows {
        assert!(
            large.saturating_sub(small) <= spectrum_growth,
            "{mode} twist is not linear in k: {small} -> {large} bytes"
        );
        assert!(
            large < dense_payload,
            "{mode} twist allocated a dense payload: {large} bytes"
        );
    }
}
