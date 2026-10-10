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
//! Compose and contract (leaf B). TensorKit `cfaa073e` `compose`
//! (`src/tensors/linalg.jl:38`) with `compose_dest(::Diagonal, ::Diagonal)`
//! (`src/tensors/diagonal.jl:325`) keeps `D * D` diagonal, and `t * D` /
//! `D * t` dispatch `mul!` on `block(d, c)::Diagonal` (`diagonal.jl:153`,
//! `330`), i.e. one axis scaled per block. The oracle is the same operation on
//! the materialized operands (the dense route). QSpace has no compact
//! diagonal contraction (`contractQS.cc` expands a diagonal first).
//!
//! Capability absences (compile-time, not counted as passing cells): checked
//! × CUDA and Fibonacci × CUDA (CUDA ops are bounded
//! `MultiplicityFreeRigidSymbols<Scalar = f64>`), and Fibonacci twist (the
//! multiplicity-free twist needs `Scalar = f64`). Fibonacci contract is
//! anyonic and asserted as the unchanged `contract` rejection.

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FermionParityFusionRule, FusionStyleKind, GenericFArray,
    GenericRMatrix, RuleIdentity, SectorId, SectorVec, TypedSectorAdmission, Z2Irrep,
};
use tenet::typed::{
    BlockFusionTrees, Complex64, ContractSpec, Direction, GradedSpace, SectorSpectrum, TensorMap,
    TensorScalar, TypedTensorConstructionDispatch, TypedTensorContractDispatch,
    TypedTensorModeDispatch,
};

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
    /// Once set, every fallible provider query fails: an operation that
    /// still succeeds made none.
    fail_queries: AtomicBool,
}

impl CheckedZ2 {
    fn new(braiding: BraidingStyleKind, odd_twist: f64) -> Self {
        Self {
            braiding,
            odd_twist,
            twist_queries: AtomicUsize::new(0),
            fail_queries: AtomicBool::new(false),
        }
    }

    fn gate(&self) -> Result<(), InvalidSector> {
        if self.fail_queries.load(Ordering::Relaxed) {
            Err(InvalidSector)
        } else {
            Ok(())
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
        self.gate()?;
        checked_z2(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.gate()?;
        let coupled = SectorId::new(checked_z2(left)?.id() ^ checked_z2(right)?.id());
        Ok([coupled].into_iter().collect())
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.gate()?;
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.gate()?;
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
        self.gate()?;
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.gate()?;
        checked_z2(sector).map(|_| 1.0)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.gate()?;
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
        self.gate()?;
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
        self.gate()?;
        let size = usize::from(Self::admissible(a, b, c));
        // Bosonic Z2 is the trivially braided twin; every other style keeps
        // the super-vector-space exchange sign.
        let odd_pair = a == ODD && b == ODD && self.braiding != BraidingStyleKind::Bosonic;
        let value = if odd_pair { -1.0 } else { 1.0 };
        Ok(GenericRMatrix::new(vec![value; size], size, size))
    }
}

impl CheckedGenericPivotal for CheckedZ2 {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.gate()?;
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

// ---- Leaf B: compose and contract ----

/// The compact operand and its dense twin, plus two dense operands that face
/// the bond: `a: W ⊗ W ← W` and `b: W ← W ⊗ W`.
struct BondFixture<R: TypedSectorAdmission, D: TensorScalar> {
    d: TensorMap<R, D>,
    e: TensorMap<R, D>,
    a: TensorMap<R, D>,
    b: TensorMap<R, D>,
}

fn bond_fixture<R, D>(
    leg: &GradedSpace<R>,
    label: impl Fn(&R::Sector) -> f64,
    conv: impl Fn(f64) -> D,
) -> BondFixture<R, D>
where
    R: TypedSectorAdmission,
    R::Sector: Clone,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    let runtime = host_runtime();
    let spectrum = |shift: f64| {
        leg.sectors()
            .unwrap()
            .into_iter()
            .map(|sector| SectorSpectrum {
                values: (0..leg.degeneracy(&sector).unwrap())
                    .map(|i| conv(shift + 0.5 * label(&sector) - 0.75 * i as f64))
                    .collect(),
                sector,
            })
            .collect::<Vec<_>>()
    };
    let fill = |trees: &BlockFusionTrees<R::Sector>, index: &[usize]| {
        let sectors: f64 = trees
            .codomain_uncoupled()
            .iter()
            .chain(trees.domain_uncoupled())
            .enumerate()
            .map(|(position, sector)| (position + 1) as f64 * label(sector))
            .sum();
        let offsets: f64 = index
            .iter()
            .enumerate()
            .map(|(axis, &i)| (axis as f64 + 0.5) * i as f64)
            .sum();
        conv(0.25 + 0.375 * sectors - 0.3125 * offsets)
    };
    BondFixture {
        d: TensorMap::diagonal(&runtime, leg, spectrum(1.5)).unwrap(),
        e: TensorMap::diagonal(&runtime, leg, spectrum(-0.5)).unwrap(),
        a: TensorMap::from_subblock_fn(&runtime, [leg, leg], [leg], fill).unwrap(),
        b: TensorMap::from_subblock_fn(&runtime, [leg], [leg, leg], fill).unwrap(),
    }
}

const fn spec<'a>(
    lhs: &'a [usize],
    rhs: &'a [usize],
    codomain: &'a [usize],
    domain: &'a [usize],
) -> ContractSpec<'a> {
    ContractSpec {
        lhs,
        rhs,
        codomain,
        domain,
    }
}

/// `(what, stays compact, lhs, rhs, spec)`; `None` spec is `compose`. Operand
/// codes: `d`/`e` compact, `a`/`b` dense. Every one-leg contraction is listed
/// in its default output order and in at least one reordered one.
type BondCase = (
    &'static str,
    bool,
    char,
    char,
    Option<ContractSpec<'static>>,
);

const COMPOSE_CASES: [BondCase; 3] = [
    ("compose D * D", true, 'd', 'e', None),
    ("compose t * D", false, 'a', 'd', None),
    ("compose D * t", false, 'd', 'b', None),
];

const CONTRACT_CASES: [BondCase; 8] = [
    (
        "contract D . D",
        true,
        'd',
        'e',
        Some(spec(&[1], &[0], &[0], &[1])),
    ),
    (
        "contract D . D bent",
        false,
        'd',
        'e',
        Some(spec(&[1], &[0], &[0, 1], &[])),
    ),
    (
        "contract t . D",
        false,
        'a',
        'd',
        Some(spec(&[2], &[0], &[0, 1], &[2])),
    ),
    (
        "contract t . D swapped",
        false,
        'a',
        'd',
        Some(spec(&[2], &[0], &[1, 0], &[2])),
    ),
    (
        "contract t . D bent",
        false,
        'a',
        'd',
        Some(spec(&[2], &[0], &[0], &[1, 2])),
    ),
    (
        "contract D . t",
        false,
        'd',
        'b',
        Some(spec(&[1], &[0], &[0], &[1, 2])),
    ),
    (
        "contract D . t swapped",
        false,
        'd',
        'b',
        Some(spec(&[1], &[0], &[0], &[2, 1])),
    ),
    (
        "contract D . t bent",
        false,
        'd',
        'b',
        Some(spec(&[1], &[0], &[0, 1], &[2])),
    ),
];

fn pick<R: TypedSectorAdmission, D: TensorScalar>(
    fixture: &BondFixture<R, D>,
    code: char,
) -> &TensorMap<R, D> {
    match code {
        'd' => &fixture.d,
        'e' => &fixture.e,
        'a' => &fixture.a,
        _ => &fixture.b,
    }
}

/// Runs `cases` on the compact operands and on their materialized twins,
/// asserting the storage form and agreement with the dense route, and
/// returns each result's dense image for the cross-mode comparison. `None`
/// marks a case both storages reject with the same error.
///
/// The one storage-dependent outcome: checked Generic contraction of a
/// non-Bosonic rule is unsupported (U15) on the dense route, while the compact
/// arm, which only scales and permutes, answers it as TensorKit does. The
/// caller checks those values against the multiplicity-free mode instead.
fn bond_rows<R, D>(fixture: &BondFixture<R, D>, cases: &[BondCase]) -> Vec<Option<Vec<Complex64>>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D> + TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar + std::fmt::Debug,
    Complex64: From<D>,
{
    let dense = BondFixture {
        d: fixture.d.materialize().unwrap(),
        e: fixture.e.materialize().unwrap(),
        a: fixture.a.clone(),
        b: fixture.b.clone(),
    };
    let run = |operands: &BondFixture<R, D>, lhs, rhs, spec: &Option<ContractSpec<'_>>| {
        let (lhs, rhs) = (pick(operands, lhs), pick(operands, rhs));
        match spec {
            None => lhs.compose(rhs),
            Some(spec) => lhs.contract(rhs, spec),
        }
    };
    let image = |tensor: &TensorMap<R, D>| -> Vec<Complex64> {
        let dense = tensor.materialize().unwrap();
        dense
            .dense_data()
            .unwrap()
            .iter()
            .map(|&x| Complex64::from(x))
            .collect()
    };
    cases
        .iter()
        .map(|(what, compact, lhs, rhs, spec)| {
            match (
                run(fixture, *lhs, *rhs, spec),
                run(&dense, *lhs, *rhs, spec),
            ) {
                (Ok(got), Ok(want)) => {
                    assert_eq!(got.dense_data().is_err(), *compact, "{what}: storage form");
                    assert_eq!(got.codomain(), want.codomain(), "{what}: codomain");
                    assert_eq!(got.domain(), want.domain(), "{what}: domain");
                    let (got, want) = (image(&got), image(&want));
                    assert_eq!(got.len(), want.len(), "{what}");
                    for (&x, &y) in got.iter().zip(&want) {
                        assert!(close(x, y), "{what}: {x} != dense {y}");
                    }
                    Some(got)
                }
                (Ok(got), Err(rejection)) => {
                    assert!(
                        format!("{rejection}").contains("requires Bosonic braiding"),
                        "{what}: only the U15 boundary may separate the storages: {rejection}"
                    );
                    assert_eq!(got.dense_data().is_err(), *compact, "{what}: storage form");
                    Some(image(&got))
                }
                (Err(got), Err(want)) => {
                    assert_eq!(format!("{got}"), format!("{want}"), "{what}");
                    None
                }
                (Err(got), Ok(_)) => panic!("{what}: compact failed where dense did not: {got}"),
            }
        })
        .collect()
}

fn assert_rows_close(mf: &[Option<Vec<Complex64>>], checked: &[Option<Vec<Complex64>>]) {
    assert_eq!(mf.len(), checked.len());
    for (row, (lhs, rhs)) in mf.iter().zip(checked).enumerate() {
        let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
            panic!("row {row}: one mode rejected it");
        };
        assert_eq!(lhs.len(), rhs.len(), "row {row}");
        for (&x, &y) in lhs.iter().zip(rhs) {
            assert!(
                close(x, y),
                "row {row}: multiplicity-free {x} != checked {y}"
            );
        }
    }
}

fn z2_leg<R>(rule: R) -> GradedSpace<R>
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: TypedTensorModeDispatch<R>,
{
    GradedSpace::try_new(
        Arc::new(rule),
        SECTORS.iter().map(|&(p, k)| (Z2Irrep::new(p), k)),
    )
    .unwrap()
}

fn parity(sector: &Z2Irrep) -> f64 {
    f64::from(sector.parity())
}

#[test]
fn compact_compose_and_contract_agree_across_modes() {
    // What: compose and one-leg contract against a compact diagonal are one
    // mode-free arm. Checked Generic used to densify (`D * D` came back
    // dense); both modes now give the same storage form and values, equal to
    // the dense route, for f64 and Complex64 and for every output order.
    use tenet::sector::Z2FusionRule;
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.5 * x - 0.25);
    assert_rows_close(
        &bond_rows(&bond_fixture(&z2_leg(Z2FusionRule), parity, real), &cases),
        &bond_rows(
            &bond_fixture(
                &z2_leg(CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0)),
                parity,
                real,
            ),
            &cases,
        ),
    );
    assert_rows_close(
        &bond_rows(
            &bond_fixture(&z2_leg(Z2FusionRule), parity, complex),
            &cases,
        ),
        &bond_rows(
            &bond_fixture(
                &z2_leg(CheckedZ2::new(BraidingStyleKind::Bosonic, 1.0)),
                parity,
                complex,
            ),
            &cases,
        ),
    );
}

#[test]
fn fermionic_compact_compose_and_contract_agree_across_modes() {
    // What: the fermionic row. Composition is admitted in both modes and
    // agrees. Checked Generic fermionic contraction is unsupported on the
    // dense route (U15), so where the compact arm answers, its value is
    // checked against the multiplicity-free mode (whose dense route
    // `bond_rows` already checked); the one pattern no arm answers (`D . D`
    // bent) stays rejected in checked mode whatever the storage.
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    for complex in [false, true] {
        let conv = move |x: f64| {
            if complex {
                Complex64::new(x, 0.5 * x - 0.25)
            } else {
                Complex64::new(x, 0.0)
            }
        };
        let mf = bond_rows(
            &bond_fixture(&z2_leg(FermionParityFusionRule), parity, conv),
            &cases,
        );
        let checked = bond_rows(
            &bond_fixture(
                &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
                parity,
                conv,
            ),
            &cases,
        );
        let rejected: Vec<_> = cases
            .iter()
            .zip(&checked)
            .filter(|(_, row)| row.is_none())
            .map(|((what, ..), _)| *what)
            .collect();
        assert_eq!(rejected, ["contract D . D bent"]);
        let answered = |rows: &[Option<Vec<Complex64>>]| -> Vec<Option<Vec<Complex64>>> {
            rows.iter()
                .zip(&checked)
                .filter(|(_, row)| row.is_some())
                .map(|(row, _)| row.clone())
                .collect()
        };
        assert_rows_close(&answered(&mf), &answered(&checked));
    }
}

#[test]
fn multiplicity_free_u1_compact_compose_and_contract_match_the_dense_route() {
    // What: a non-self-dual U(1) bond (charges ±1 swap under duality) takes the
    // same arms, including the bent outputs whose permute moves a dual sector.
    use tenet::sector::{U1FusionRule, U1Irrep};
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let charge = |sector: &U1Irrep| {
        if *sector == U1Irrep::new(1) {
            1.0
        } else {
            -1.0
        }
    };
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    bond_rows(&bond_fixture(&leg, charge, |x: f64| x), &cases);
    bond_rows(
        &bond_fixture(&leg, charge, |x: f64| Complex64::new(-x, 0.25 * x)),
        &cases,
    );
}

#[test]
fn fibonacci_complex_symbol_compact_compose_takes_the_arms() {
    // What: the complex-symbol coefficient lane used to skip the compact arms
    // (#1866 X6); compose now stays compact for `D * D` and scales for
    // `t * D` / `D * t`. Fibonacci is anyonic, so `contract` keeps rejecting
    // it before any arm, compact or not.
    use tenet::sector::{FibonacciFusionRule, FibonacciSector};
    let leg = GradedSpace::try_new(
        Arc::new(FibonacciFusionRule),
        [(FibonacciSector::Vacuum, 2), (FibonacciSector::Tau, 3)],
    )
    .unwrap();
    let tau = |sector: &FibonacciSector| f64::from(u8::from(*sector == FibonacciSector::Tau));
    let fixture = bond_fixture(&leg, tau, |x: f64| Complex64::new(x, 0.5 - x));
    bond_rows(&fixture, &COMPOSE_CASES);
    let dense = fixture.d.materialize().unwrap();
    for (what, _, lhs, rhs, spec) in CONTRACT_CASES {
        let spec = spec.unwrap();
        let compact = pick(&fixture, lhs).contract(pick(&fixture, rhs), &spec);
        let swap = |code| {
            if code == 'd' {
                &dense
            } else {
                pick(&fixture, code)
            }
        };
        let materialized = swap(lhs).contract(swap(rhs), &spec);
        assert_eq!(
            format!("{}", compact.unwrap_err()),
            format!("{}", materialized.unwrap_err()),
            "{what}"
        );
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn su2_compact_compose_and_contract_agree_across_modes() {
    // What: the non-Abelian row. The bent outputs carry SU(2) bending
    // coefficients through the mode's own permute.
    use tenet::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
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
    let mf_label = |sector: &SU2Irrep| sector.twice_spin() as f64;
    let checked_label = |sector: &Vec<i64>| sector[0] as f64;
    let cases: Vec<BondCase> = COMPOSE_CASES.into_iter().chain(CONTRACT_CASES).collect();
    let real = |x: f64| x;
    assert_rows_close(
        &bond_rows(&bond_fixture(&mf_leg, mf_label, real), &cases),
        &bond_rows(&bond_fixture(&checked_leg, checked_label, real), &cases),
    );
    let complex = |x: f64| Complex64::new(0.5 * x, x - 1.0);
    assert_rows_close(
        &bond_rows(&bond_fixture(&mf_leg, mf_label, complex), &cases),
        &bond_rows(&bond_fixture(&checked_leg, checked_label, complex), &cases),
    );
}

#[test]
fn declined_compact_arms_report_the_dense_route_error() {
    // What: a pair the arms cannot answer — providers with different rule
    // identities, or a dense leg that is not the bond — falls through to the
    // dense route, so compact and materialized operands fail alike.
    macro_rules! assert_same_error {
        ($compact:expr, $dense:expr, $what:expr) => {
            assert_eq!(
                format!("{}", $compact.unwrap_err()),
                format!("{}", $dense.unwrap_err()),
                "{}",
                $what
            )
        };
    }
    let real = |x: f64| x;

    let fixture = bond_fixture(
        &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
        parity,
        real,
    );
    let foreign = bond_fixture(
        &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -0.5)),
        parity,
        real,
    );
    let (d, foreign_d) = (&fixture.d, &foreign.d);
    let (dense_d, dense_foreign) = (d.materialize().unwrap(), foreign_d.materialize().unwrap());
    assert_same_error!(
        d.compose(foreign_d),
        dense_d.compose(&dense_foreign),
        "D * D'"
    );
    assert_same_error!(
        fixture.a.compose(foreign_d),
        fixture.a.compose(&dense_foreign),
        "t * D'"
    );
    assert_same_error!(
        d.contract(&foreign.b, &spec(&[1], &[0], &[0], &[1, 2])),
        dense_d.contract(&foreign.b, &spec(&[1], &[0], &[0], &[1, 2])),
        "D . t'"
    );

    for (mode, errors) in [
        (
            "multiplicity-free",
            non_composable_errors(|| FermionParityFusionRule),
        ),
        (
            "checked",
            non_composable_errors(|| CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
        ),
    ] {
        for (what, (compact, dense)) in ["t * D", "D . t"].into_iter().zip(errors) {
            assert_eq!(compact, dense, "non-composable {what} ({mode})");
        }
    }
}

/// `t * D` and `D . t` where `t`'s facing leg is not the bond, on compact and
/// on materialized `D`: `[(compact, dense); 2]` error messages.
fn non_composable_errors<R>(rule: impl Fn() -> R) -> [(String, String); 2]
where
    R: TypedSectorAdmission<Sector = Z2Irrep>,
    R::Mode: TypedTensorContractDispatch<R, f64> + TypedTensorConstructionDispatch<R, f64>,
{
    let other = GradedSpace::try_new(
        Arc::new(rule()),
        [(Z2Irrep::new(0), 1), (Z2Irrep::new(1), 4)],
    )
    .unwrap();
    let t =
        TensorMap::<_, f64>::from_subblock_fn(&host_runtime(), [&other], [&other], |_, index| {
            index[0] as f64 - 0.5 * index[1] as f64
        })
        .unwrap();
    let fixture = bond_fixture(&z2_leg(rule()), parity, |x: f64| x);
    let dense = fixture.d.materialize().unwrap();
    let message = |result: Result<TensorMap<R, f64>, _>| format!("{}", result.unwrap_err());
    let one_leg = spec(&[1], &[0], &[0], &[1]);
    [
        (message(t.compose(&fixture.d)), message(t.compose(&dense))),
        (
            message(fixture.d.contract(&t, &one_leg)),
            message(dense.contract(&t, &one_leg)),
        ),
    ]
}

#[test]
fn checked_compose_and_identity_order_contract_arms_query_no_provider() {
    // What: the compose arms and an arm whose output needs no reordering read
    // only the stored spectrum and the admitted rule identity, so they succeed
    // even when every fallible provider query would fail.
    let rule = CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0);
    let fixture = bond_fixture(&z2_leg(rule), parity, |x: f64| x);
    let provider = fixture.d.provider();
    provider.fail_queries.store(true, Ordering::Relaxed);
    for (what, compact, lhs, rhs, spec) in
        COMPOSE_CASES
            .into_iter()
            .chain([CONTRACT_CASES[0], CONTRACT_CASES[2], CONTRACT_CASES[5]])
    {
        let (lhs, rhs) = (pick(&fixture, lhs), pick(&fixture, rhs));
        let result = match spec {
            None => lhs.compose(rhs),
            Some(spec) => lhs.contract(rhs, &spec),
        };
        let result = result.unwrap_or_else(|error| panic!("{what}: {error}"));
        assert_eq!(result.dense_data().is_err(), compact, "{what}");
    }
    provider.fail_queries.store(false, Ordering::Relaxed);
}

#[test]
fn checked_reordered_contract_arm_failure_is_typed_and_nonpublishing() {
    // What: the reordered `t · D` arm lays its result out with one checked
    // permute staging. A provider failing there is reported as the typed
    // plan error of that staging; the operands are unchanged and no complete
    // layout or transform coefficients are published for the destination.
    const ISOLATED: &str = "TENET_COMPACT_CONTRACT_PUBLICATION_ISOLATED";
    const NAME: &str = "checked_reordered_contract_arm_failure_is_typed_and_nonpublishing";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated publication test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("test result: ok. 1 passed; 0 failed;"));
        return;
    }

    use tenet::typed::{CheckedGenericPlanError, GenericTensorError};
    use tenet_core::{structure_cache_info, StructureCacheKind};
    tenet_core::clear_structure_caches();
    let fixture = bond_fixture(
        &z2_leg(CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0)),
        parity,
        |x: f64| x,
    );
    let source = fixture.a.dense_data().unwrap().to_vec();
    let spectrum = tenet::expert::diagonal_spectrum(&fixture.d).unwrap();
    let published = || {
        [
            StructureCacheKind::DegeneracyStructure,
            StructureCacheKind::CompletedTreeTransformer,
            StructureCacheKind::TreeTransformCoefficients,
        ]
        .map(|kind| {
            let info = structure_cache_info(kind);
            (info.admissions(), info.entries(), info.charged_bytes())
        })
    };
    let before = published();
    fixture
        .d
        .provider()
        .fail_queries
        .store(true, Ordering::Relaxed);
    let result = fixture
        .a
        .contract(&fixture.d, &spec(&[2], &[0], &[0], &[1, 2]));
    fixture
        .d
        .provider()
        .fail_queries
        .store(false, Ordering::Relaxed);
    assert!(
        matches!(
            result,
            Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
                InvalidSector
            )))
        ),
        "{result:?}"
    );
    assert_eq!(published(), before);
    assert_eq!(fixture.a.dense_data().unwrap(), source);
    assert_eq!(
        tenet::expert::diagonal_spectrum(&fixture.d).unwrap(),
        spectrum
    );
}

/// Warm `D * D` bytes and whether the result stayed compact. The caller
/// asserts the byte bound before the storage form, so a densifying
/// regression is caught by the bound on its own.
fn warmed_square_bytes<R>(tensor: &TensorMap<R, f64>) -> (u64, bool)
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, f64>,
{
    black_box(tensor.compose(tensor).unwrap());
    let (square, allocs) = counting_alloc::measure(|| black_box(tensor.compose(tensor).unwrap()));
    (allocs.bytes, square.dense_data().is_err())
}

#[test]
fn compact_square_allocates_linear_in_the_spectrum_in_both_modes() {
    // What: `D * D` of a compact diagonal costs Σk, never a k² payload.
    let _measurement = counting_alloc::serial();
    let dense_payload = (K * K * std::mem::size_of::<f64>()) as u64;
    let spectrum_growth = (2 * K * std::mem::size_of::<f64>()) as u64;
    let rows = [
        (
            "multiplicity-free",
            warmed_square_bytes(&odd_diagonal(FermionParityFusionRule, K)),
            warmed_square_bytes(&odd_diagonal(FermionParityFusionRule, 2 * K)),
        ),
        (
            "checked",
            warmed_square_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                K,
            )),
            warmed_square_bytes(&odd_diagonal(
                CheckedZ2::new(BraidingStyleKind::Fermionic, -1.0),
                2 * K,
            )),
        ),
    ];
    for (mode, (small, _), (large, compact)) in rows {
        assert!(
            large.saturating_sub(small) <= spectrum_growth,
            "{mode} D * D is not linear in k: {small} -> {large} bytes"
        );
        assert!(
            large < dense_payload,
            "{mode} D * D allocated a dense payload: {large} bytes"
        );
        assert!(compact, "{mode} D * D densified");
    }
}
