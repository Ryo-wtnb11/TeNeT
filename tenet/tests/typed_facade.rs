//! Gate suite for the provider-typed facade (`tenet::typed`, issue #557).
//!
//! Every provider here is built from the public vocabulary alone — no sealed
//! lowered codec, no crate-internal machinery — so the suite doubles as proof
//! that a downstream application can drive the typed facade with its own
//! fusion rule.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tenet::core::{
    complete_hom_space_structure_cache_info, fusion_tree_layout_cache_info, BraidingStyleKind,
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FusionAlgebraError, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    ProductFusionRuleExt, RuleIdentity, SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorVec,
};
use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{GradedSpace, TensorMap, Truncation};

/// The fusion-tree layout and complete-structure caches are process-global, so
/// the tests in this binary that snapshot them must not run beside a test that
/// builds a layout. Only this binary shares those globals; other test binaries
/// are separate processes.
static CACHE_LOCK: Mutex<()> = Mutex::new(());

fn cache_lock() -> MutexGuard<'static, ()> {
    CACHE_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------
// External Unique-fusion provider: Z3 charges, addition mod 3.
//
// Z3 rather than Z2/XOR because its non-vacuum charges are not self-dual, so
// every dual-leg assertion in this suite is about a sector that actually
// changes under the dual.
// ---------------------------------------------------------------------------

/// One deliberately broken behaviour, injected into an otherwise valid
/// provider. The rule identity is unaffected, matching how the checked
/// admission tests in `tenet-tensors` inject failures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Quirk {
    /// Encodes every label to the vacuum id, violating codec injectivity.
    AliasLabels,
    /// Fails the checked dual, i.e. mid-staging inside the checked path.
    FailDual,
    /// Refuses to decode charge 2, which the engine reaches by fusing two
    /// charge-1 legs — a violation of the codec's decode-totality law.
    NarrowDecode,
    /// Duals every sector to the vacuum, i.e. a non-injective dual: a broken
    /// rigidity structure rather than an unrepresentable value.
    CollapsingDual,
    /// Labels the charges in the reverse of the engine's `SectorId` order
    /// (`c <-> 2 - c`). Not broken at all — a valid codec whose label order
    /// simply is not the id order, which is the only way to observe that a
    /// facade sorts by label rather than by id.
    ReversedLabels,
}

#[derive(Clone, Copy)]
struct ExternalZ3 {
    quirk: Option<Quirk>,
    /// Distinguishes two otherwise identical provider values, so the facade's
    /// `RuleIdentity` handling can be tested in both directions with one type.
    identity_tag: u8,
}

impl ExternalZ3 {
    fn new() -> Self {
        Self {
            quirk: None,
            identity_tag: 0,
        }
    }

    fn with(quirk: Quirk) -> Self {
        Self {
            quirk: Some(quirk),
            identity_tag: 0,
        }
    }

    fn tagged(identity_tag: u8) -> Self {
        Self {
            quirk: None,
            identity_tag,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Z3Charge(u8);

impl FusionRule for ExternalZ3 {
    fn rule_identity(&self) -> RuleIdentity {
        // Only the tag participates: an injected quirk is a broken provider,
        // not a different fusion algebra, and the failure-injection tests rely
        // on it keeping the identity it claims.
        RuleIdentity::from_canonical_bytes::<Self>(
            0x5a33_0000_0000_0000,
            Arc::<[u8]>::from(vec![self.identity_tag]),
        )
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((3 - sector.id() % 3) % 3)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        core::iter::once(SectorId::new((left.id() + right.id()) % 3)).collect()
    }
}

impl MultiplicityFreeFusionRule for ExternalZ3 {}

impl MultiplicityFreeFusionSymbols for ExternalZ3 {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for ExternalZ3 {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

// Opt-in for the typed unit-leg operations (#580 PR 5): the marker certifies
// that the vacuum obeys the canonical unit laws, which the Z3 vacuum (charge
// 0, self-dual, trivial unitors) does. A downstream provider makes the same
// one-line declaration to unlock `insert_left_unit`/`insert_right_unit`/
// `remove_unit`.
impl tenet::core::CanonicalUnitFusionRule for ExternalZ3 {}

impl CheckedFusionAlgebra for ExternalZ3 {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if self.quirk == Some(Quirk::FailDual) {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        if self.quirk == Some(Quirk::CollapsingDual) {
            return Ok(SectorId::new(0));
        }
        Ok(self.dual(sector))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl SectorCodec for ExternalZ3 {
    type Sector = Z3Charge;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        if self.quirk == Some(Quirk::AliasLabels) {
            return Ok(SectorId::new(0));
        }
        if value.0 < 3 {
            let id = if self.quirk == Some(Quirk::ReversedLabels) {
                2 - value.0
            } else {
                value.0
            };
            Ok(SectorId::new(usize::from(id)))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.rule_identity(),
                label: format!("Z3 charge {}", value.0),
            })
        }
    }

    fn decode_sector(&self, sector: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        let limit = if self.quirk == Some(Quirk::NarrowDecode) {
            2
        } else {
            3
        };
        u8::try_from(sector.id())
            .ok()
            .filter(|&charge| charge < limit)
            .map(|charge| {
                Z3Charge(if self.quirk == Some(Quirk::ReversedLabels) {
                    2 - charge
                } else {
                    charge
                })
            })
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }
}

// ---------------------------------------------------------------------------
// External Simple-fusion provider: SU(2) mathematics behind a distinct type
// that never certifies the sealed lowered codec.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ExternalSu2;

impl FusionRule for ExternalSu2 {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        SU2FusionRule.nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for ExternalSu2 {}

impl MultiplicityFreeFusionSymbols for ExternalSu2 {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        l: SectorId,
        m: SectorId,
        r: SectorId,
        c: SectorId,
        lc: SectorId,
        rc: SectorId,
    ) -> f64 {
        SU2FusionRule.f_symbol_scalar(l, m, r, c, lc, rc)
    }
    fn r_symbol_scalar(&self, l: SectorId, r: SectorId, c: SectorId) -> f64 {
        SU2FusionRule.r_symbol_scalar(l, r, c)
    }
}

impl MultiplicityFreeRigidSymbols for ExternalSu2 {
    fn dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.dim_scalar(s)
    }
    fn inv_dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.inv_dim_scalar(s)
    }
    fn sqrt_dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.sqrt_dim_scalar(s)
    }
    fn inv_sqrt_dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.inv_sqrt_dim_scalar(s)
    }
    fn twist_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.twist_scalar(s)
    }
    fn frobenius_schur_phase_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.frobenius_schur_phase_scalar(s)
    }
}

impl CheckedFusionAlgebra for ExternalSu2 {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        SU2FusionRule.try_dual_sector(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        SU2FusionRule.try_fusion_channels(left, right)
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        SU2FusionRule.try_nsymbol(left, right, coupled)
    }
}

impl SectorCodec for ExternalSu2 {
    type Sector = SU2Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        SU2FusionRule.encode_sector(value)
    }

    fn decode_sector(&self, sector: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        SectorCodec::decode_sector(&SU2FusionRule, sector)
    }
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn z3_leg(provider: &Arc<ExternalZ3>, is_dual: bool) -> GradedSpace<ExternalZ3> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        [(Z3Charge(0), 2), (Z3Charge(1), 3), (Z3Charge(2), 1)],
    )
    .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
    .expect("Z3 leg is well formed")
}

fn su2_leg(provider: &Arc<ExternalSu2>, is_dual: bool) -> GradedSpace<ExternalSu2> {
    GradedSpace::try_new_with_arc(Arc::clone(provider), [(SU2Irrep::from_twice_spin(1), 2)])
        .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
        .expect("SU(2) leg is well formed")
}

fn runtime() -> Runtime {
    Runtime::builder().build().expect("runtime builds")
}

// ---------------------------------------------------------------------------
// Slice 4: `GradedSpace<R>`.
// ---------------------------------------------------------------------------

#[test]
fn graded_space_reports_labels_in_provider_sector_id_order() {
    // What: `sectors()` decodes back to the caller's labels, ordered by the
    // provider's sector id (not by label order), with degeneracies parallel.
    let provider = Arc::new(ExternalZ3::new());
    let space = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(Z3Charge(2), 1), (Z3Charge(0), 2), (Z3Charge(1), 3)],
    )
    .unwrap();

    assert_eq!(
        space.sectors().unwrap(),
        vec![Z3Charge(0), Z3Charge(1), Z3Charge(2)]
    );
    assert_eq!(space.degeneracies(), &[2, 3, 1]);
    assert!(!space.is_dual());
}

#[test]
fn graded_space_drops_zero_degeneracy_sectors() {
    // What: the leg invariant (a zero-degeneracy sector is absent) reaches the
    // typed surface unchanged.
    let provider = Arc::new(ExternalZ3::new());
    let space =
        GradedSpace::try_new_with_arc(provider, [(Z3Charge(0), 2), (Z3Charge(1), 0)]).unwrap();

    assert_eq!(space.sectors().unwrap(), vec![Z3Charge(0)]);
    assert_eq!(space.degeneracies(), &[2]);
}

#[test]
fn graded_space_constructor_queries_and_algebra_match_tensorkit() {
    fn generic_dim<R>(space: &GradedSpace<R>) -> f64
    where
        R: tenet::core::TypedSectorAdmission,
        R::Mode: tenet::typed::TypedSpaceModeDispatch<R>,
    {
        space.dim().ok().unwrap()
    }

    let ordinary = GradedSpace::<tenet::core::U1FusionRule>::try_new(
        tenet::core::U1FusionRule,
        [(tenet::core::U1Irrep::new(0), 1)],
    )
    .unwrap();
    assert_eq!(ordinary.sectors().unwrap(), [tenet::core::U1Irrep::new(0)]);

    let provider = Arc::new(tenet::core::U1FusionRule);
    let shared =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(tenet::core::U1Irrep::new(0), 1)])
            .unwrap();
    assert!(std::ptr::eq(shared.provider(), provider.as_ref()));
    assert_eq!(shared.sectors().unwrap(), ordinary.sectors().unwrap());
    assert_eq!(shared.degeneracies(), ordinary.degeneracies());
    assert_eq!(shared.is_dual(), ordinary.is_dual());
    let dual =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(tenet::core::U1Irrep::new(1), 2)])
            .and_then(|space| space.try_dual())
            .unwrap();
    assert!(std::ptr::eq(dual.provider(), provider.as_ref()));
    assert_eq!(dual.sectors().unwrap(), [tenet::core::U1Irrep::new(-1)]);
    assert_eq!(dual.degeneracy(&tenet::core::U1Irrep::new(-1)).unwrap(), 2);
    assert_eq!(dual.degeneracy(&tenet::core::U1Irrep::new(1)).unwrap(), 0);
    assert!(dual.has_sector(&tenet::core::U1Irrep::new(-1)).unwrap());
    assert!(!dual.has_sector(&tenet::core::U1Irrep::new(1)).unwrap());
    let back = dual.try_dual().unwrap();
    assert!(!back.is_dual());
    assert_eq!(back.sectors().unwrap(), [tenet::core::U1Irrep::new(1)]);
    assert_eq!(
        back.try_dual().unwrap().sectors().unwrap(),
        dual.sectors().unwrap()
    );

    let unit = dual.unitspace().unwrap();
    assert!(std::ptr::eq(unit.provider(), provider.as_ref()));
    assert!(!unit.is_dual());
    assert_eq!(unit.sectors().unwrap(), [tenet::core::U1Irrep::new(0)]);
    assert_eq!(unit.degeneracies(), [1]);

    let left = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let right =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(tenet::core::U1Irrep::new(-1), 3)])
            .unwrap();
    let fused = left.fuse(&right).unwrap();
    assert_eq!(fused.degeneracy(&tenet::core::U1Irrep::new(-1)).unwrap(), 3);
    assert_eq!(fused.degeneracy(&tenet::core::U1Irrep::new(0)).unwrap(), 6);
    assert!(!fused.is_dual());
    assert_eq!(generic_dim(&left), 3.0);

    let summed = left.oplus(&right).unwrap();
    assert_eq!(
        summed.degeneracy(&tenet::core::U1Irrep::new(-1)).unwrap(),
        3
    );
    assert_eq!(summed.degeneracy(&tenet::core::U1Irrep::new(0)).unwrap(), 1);
    assert_eq!(summed.degeneracy(&tenet::core::U1Irrep::new(1)).unwrap(), 2);
    assert!(left.oplus(&right.try_dual().unwrap()).is_err());
    let huge = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(tenet::core::U1Irrep::new(0), usize::MAX)],
    )
    .unwrap();
    assert!(huge.oplus(&ordinary).is_err());
    assert!(huge.fuse(&left).is_err());

    let su2_provider = Arc::new(SU2FusionRule);
    let half = SU2Irrep::from_twice_spin(1);
    let su2_left = GradedSpace::try_new_with_arc(Arc::clone(&su2_provider), [(half, 2)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let su2_right = GradedSpace::try_new_with_arc(Arc::clone(&su2_provider), [(half, 3)]).unwrap();
    assert_eq!(su2_left.sectors().unwrap(), [half]);
    assert!(su2_left.is_dual());
    assert_eq!(su2_left.dim().unwrap(), 4.0);
    let su2_fused = su2_left.fuse(&su2_right).unwrap();
    assert_eq!(
        su2_fused.degeneracy(&SU2Irrep::from_twice_spin(0)).unwrap(),
        6
    );
    assert_eq!(
        su2_fused.degeneracy(&SU2Irrep::from_twice_spin(2)).unwrap(),
        6
    );

    let z2_provider = Arc::new(tenet::core::Z2FusionRule);
    let z2_dual = GradedSpace::try_new_with_arc(z2_provider, [(tenet::core::Z2Irrep::ODD, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    assert_eq!(z2_dual.sectors().unwrap(), [tenet::core::Z2Irrep::ODD]);
    assert!(z2_dual.is_dual());

    let product_provider =
        Arc::new(tenet::core::U1FusionRule.product(tenet::core::FermionParityFusionRule));
    let product_left = GradedSpace::try_new_with_arc(
        Arc::clone(&product_provider),
        [(
            tenet::core::product_sector(tenet::core::U1Irrep::new(1), tenet::core::Z2Irrep::ODD),
            2,
        )],
    )
    .unwrap();
    let product_right = GradedSpace::try_new_with_arc(
        product_provider,
        [(
            tenet::core::product_sector(tenet::core::U1Irrep::new(-1), tenet::core::Z2Irrep::ODD),
            3,
        )],
    )
    .unwrap();
    let product_fused = product_left.fuse(&product_right).unwrap();
    assert_eq!(
        product_fused
            .degeneracy(&tenet::core::product_sector(
                tenet::core::U1Irrep::new(0),
                tenet::core::Z2Irrep::EVEN,
            ))
            .unwrap(),
        6
    );
}

#[test]
fn graded_space_rejects_a_duplicate_label_by_name() {
    // What: the duplicate is reported as the caller's own label, which needs
    // the check to run before the label is encoded away into a `SectorId`.
    let provider = Arc::new(ExternalZ3::new());
    let error =
        GradedSpace::try_new_with_arc(provider, [(Z3Charge(1), 2), (Z3Charge(1), 3)]).unwrap_err();

    let message = error.to_string();
    assert!(message.contains("Z3Charge(1)"), "{message}");
    assert!(message.contains("more than once"), "{message}");
}

#[test]
fn graded_space_reports_aliased_labels_as_a_codec_law_violation() {
    // What: two distinct labels encoding to one id is the provider breaking
    // codec injectivity, not the caller declaring a sector twice, and the two
    // cases must not be conflated in the diagnosis.
    let provider = Arc::new(ExternalZ3::with(Quirk::AliasLabels));
    let error =
        GradedSpace::try_new_with_arc(provider, [(Z3Charge(0), 2), (Z3Charge(1), 3)]).unwrap_err();

    let message = error.to_string();
    assert!(message.contains("SectorCodec"), "{message}");
    assert!(message.contains("Z3Charge(0)"), "{message}");
    assert!(message.contains("Z3Charge(1)"), "{message}");
}

#[test]
fn graded_space_reports_an_unrepresentable_label() {
    // What: an out-of-domain label surfaces the provider's own encode error.
    let provider = Arc::new(ExternalZ3::new());
    let error = GradedSpace::try_new_with_arc(provider, [(Z3Charge(7), 2)]).unwrap_err();

    assert!(error.to_string().contains("Z3 charge 7"), "{error}");
    let provider = Arc::new(ExternalZ3::new());
    let space = z3_leg(&provider, false);
    assert!(space.degeneracy(&Z3Charge(7)).is_err());
    assert!(space.has_sector(&Z3Charge(7)).is_err());
}

#[test]
fn graded_space_dual_flips_the_leg_and_dualizes_non_self_dual_labels() {
    // What: `try_dual` uses the provider's checked dual, so Z3's charge 1 and
    // charge 2 swap (with their degeneracies) and the dual flag flips.
    let provider = Arc::new(ExternalZ3::new());
    let space = z3_leg(&provider, false);
    let dual = space.try_dual().unwrap();

    assert!(dual.is_dual());
    assert_eq!(
        dual.sectors().unwrap(),
        vec![Z3Charge(0), Z3Charge(1), Z3Charge(2)]
    );
    // Charge 1 (degeneracy 3) became charge 2 and vice versa.
    assert_eq!(dual.degeneracies(), &[2, 1, 3]);
    assert_eq!(
        dual.try_dual().unwrap().degeneracies(),
        space.degeneracies()
    );
}

#[test]
fn graded_space_dual_surfaces_a_failing_checked_dual() {
    // What: a provider that cannot dual a sector reports its own typed error
    // rather than producing a partially dualized leg.
    let provider = Arc::new(ExternalZ3::with(Quirk::FailDual));
    let space = z3_leg(&provider, false);

    assert!(space.try_dual().is_err());
}

#[test]
fn graded_space_dual_reports_a_non_injective_dual_instead_of_panicking() {
    // What: a provider whose dual collapses two sectors onto one id is a
    // broken rigidity structure. The leg cannot hold the result, and the
    // failure must come back as a typed error — the constructor underneath
    // used to panic inside this `Result`-returning API.
    let provider = Arc::new(ExternalZ3::with(Quirk::CollapsingDual));
    let space = z3_leg(&provider, false);

    let error = space.try_dual().unwrap_err();

    assert!(error.to_string().contains("not injective"), "{error}");
    assert!(
        GradedSpace::try_new_with_arc(provider, [(Z3Charge(0), 1), (Z3Charge(1), 1)])
            .and_then(|space| space.try_dual())
            .is_err()
    );
}

#[test]
fn graded_space_carries_a_simple_fusion_provider_too() {
    // What: nothing in the typed space is abelian-specific.
    let provider = Arc::new(ExternalSu2);
    let space = su2_leg(&provider, true);

    assert_eq!(space.sectors().unwrap(), vec![SU2Irrep::from_twice_spin(1)]);
    assert!(space.is_dual());
    assert_eq!(runtime().tree_transform_cache_info().entries(), 0);
}

// ---------------------------------------------------------------------------
// Slice 5: `TensorMap<R, D>` ownership and `zeros`.
// ---------------------------------------------------------------------------

#[test]
fn tensor_map_zeros_builds_a_multi_block_checked_layout() {
    // What: `zeros` admits a runtime-rank multi-block layout through the
    // checked path and returns a zero buffer of exactly the layout's length.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalZ3::new());
    let leg = z3_leg(&provider, false);
    let dual = leg.try_dual().unwrap();
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3, f64> =
        TensorMap::zeros(&runtime, [&leg, &leg], [&dual, &dual]).unwrap();

    assert!(tensor.block_count() >= 2);
    assert!(!tensor.data().is_empty());
    assert!(tensor.data().iter().all(|&value| value == 0.0));
}

#[test]
fn tensor_map_zeros_carries_a_complex_payload() {
    // What: the payload dtype is independent of the provider's real
    // categorical coefficient scalar.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);
    let runtime = runtime();

    let tensor: TensorMap<ExternalSu2, Complex64> =
        TensorMap::zeros(&runtime, [&leg, &leg], [&leg, &leg]).unwrap();

    assert!(tensor.block_count() >= 2);
    assert!(tensor
        .data()
        .iter()
        .all(|&value| value == Complex64::new(0.0, 0.0)));
}

#[test]
fn tensor_map_accepts_separately_allocated_equal_identity_providers() {
    // What: two independent allocations of one rule interoperate; the facade
    // keys on `RuleIdentity`, never on `Arc` identity.
    let _guard = cache_lock();
    let first = Arc::new(ExternalZ3::new());
    let second = Arc::new(ExternalZ3::new());
    assert!(!Arc::ptr_eq(&first, &second));
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3, f64> =
        TensorMap::zeros(&runtime, [&z3_leg(&first, false)], [&z3_leg(&second, true)]).unwrap();

    assert!(tensor.block_count() >= 1);
}

#[test]
fn tensor_map_rejects_distinct_rule_identities_before_provider_work() {
    // What: two providers of the same Rust type but different identities are a
    // rule mismatch, reported before any layout is staged — the caches stay
    // exactly as they were.
    let _guard = cache_lock();
    let first = Arc::new(ExternalZ3::tagged(0));
    let second = Arc::new(ExternalZ3::tagged(1));
    assert_ne!(first.rule_identity(), second.rule_identity());
    let runtime = runtime();
    let before = (
        fusion_tree_layout_cache_info(),
        complete_hom_space_structure_cache_info(),
    );

    let error = TensorMap::<ExternalZ3, f64>::zeros(
        &runtime,
        [&z3_leg(&first, false)],
        [&z3_leg(&second, true)],
    )
    .unwrap_err();

    assert!(matches!(error, tenet::prelude::Error::RuleMismatch));
    assert_eq!(
        (
            fusion_tree_layout_cache_info(),
            complete_hom_space_structure_cache_info(),
        ),
        before
    );
}

#[test]
fn tensor_map_zeros_needs_at_least_one_leg() {
    // What: the provider is inferred from the legs, so an empty tensor map has
    // nothing to infer it from.
    let runtime = runtime();
    let empty: [&GradedSpace<ExternalZ3>; 0] = [];

    assert!(TensorMap::<ExternalZ3, f64>::zeros(&runtime, empty, empty).is_err());
}

#[test]
fn checked_construction_failure_publishes_no_cache_state() {
    // What: a provider that fails mid-staging returns a typed error and leaves
    // both process-global layout caches and the runtime's own cache untouched,
    // which is the transactional guarantee the checked path promises.
    let _guard = cache_lock();
    // The broken provider must be the layout authority (the first leg's
    // provider), otherwise the staging never calls its failing primitive.
    let broken = Arc::new(ExternalZ3::with(Quirk::FailDual));
    let codomain = z3_leg(&broken, false);
    let healthy = Arc::new(ExternalZ3::new());
    let domain = GradedSpace::try_new_with_arc(
        healthy,
        [(Z3Charge(0), 2), (Z3Charge(2), 3), (Z3Charge(1), 1)],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let runtime = runtime();
    let before = (
        fusion_tree_layout_cache_info(),
        complete_hom_space_structure_cache_info(),
    );
    let runtime_before = runtime.tree_transform_cache_info();

    let error = TensorMap::<ExternalZ3, f64>::zeros(&runtime, [&codomain], [&domain]).unwrap_err();

    assert!(matches!(
        error,
        tenet::prelude::Error::FusionAlgebra(_) | tenet::prelude::Error::Operation(_)
    ));
    assert_eq!(
        (
            fusion_tree_layout_cache_info(),
            complete_hom_space_structure_cache_info(),
        ),
        before
    );
    assert_eq!(runtime.tree_transform_cache_info(), runtime_before);
}

// ---------------------------------------------------------------------------
// Slice 6: `from_block_fn` and inspection.
// ---------------------------------------------------------------------------

/// Numeric stand-in for a Z3 charge, so a fill value can depend on the labels
/// the closure was handed.
fn z3_weight(charge: Z3Charge) -> f64 {
    f64::from(charge.0) + 1.0
}

#[test]
fn from_block_fn_sees_decoded_labels_and_fills_every_allowed_element() {
    // What: the closure is handed the provider's own labels (not `SectorId`s)
    // for the coupled sector and both sides' uncoupled legs, and every stored
    // element is written.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalZ3::new());
    let leg = z3_leg(&provider, false);
    let dual = leg.try_dual().unwrap();
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&dual], |sectors, indices| {
            assert_eq!(sectors.codomain_uncoupled().len(), 1);
            assert_eq!(sectors.domain_uncoupled().len(), 1);
            z3_weight(*sectors.coupled()) * 100.0
                + z3_weight(sectors.codomain_uncoupled()[0]) * 10.0
                + indices.iter().sum::<usize>() as f64
        })
        .unwrap();

    assert!(tensor.block_count() >= 1);
    assert!(tensor.data().iter().all(|&value| value >= 100.0));
}

#[test]
fn from_block_fn_surfaces_a_decode_failure_as_the_codec_error() {
    // What: a codec that cannot decode an id the engine produced fails the
    // construction with the provider's own error instead of panicking inside
    // the fill.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalZ3::with(Quirk::NarrowDecode));
    let leg =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 2)])
            .unwrap();
    let runtime = runtime();

    // Two charge-1 codomain legs couple to charge 2, the id this codec refuses.
    let error = TensorMap::<ExternalZ3, f64>::from_block_fn(
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |_, _| 1.0,
    )
    .unwrap_err();

    assert!(matches!(error, tenet::prelude::Error::FusionAlgebra(_)));
}

#[test]
fn tensor_map_inspection_round_trips_the_spaces_and_blocks() {
    // What: the legs come back as typed graded spaces with their labels and
    // dual flags intact, and every block reports decoded labels plus a data
    // view consistent with the buffer.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalZ3::new());
    let leg = z3_leg(&provider, false);
    let dual = leg.try_dual().unwrap();
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3, f64> =
        TensorMap::zeros(&runtime, [&leg, &leg], [&dual]).unwrap();

    let codomain = tensor.codomain();
    let domain = tensor.domain();
    assert_eq!(codomain.len(), 2);
    assert_eq!(domain.len(), 1);
    assert_eq!(codomain[0].sectors().unwrap(), leg.sectors().unwrap());
    assert!(!codomain[0].is_dual());
    assert_eq!(domain[0].sectors().unwrap(), dual.sectors().unwrap());
    assert!(domain[0].is_dual());

    let mut elements = 0;
    for index in 0..tensor.block_count() {
        let sectors = tensor.block_fusion_trees(index).unwrap();
        assert_eq!(sectors.codomain_uncoupled().len(), 2);
        assert_eq!(sectors.domain_uncoupled().len(), 1);
        // Unique fusion: the two codomain charges fuse to the coupled charge.
        let sum = (sectors.codomain_uncoupled()[0].0 + sectors.codomain_uncoupled()[1].0) % 3;
        assert_eq!(&Z3Charge(sum), sectors.coupled());

        let block = tensor.block(index).unwrap();
        assert!(block.storage_end_exclusive().unwrap() <= tensor.data().len());
        elements += block.element_count().unwrap();
    }
    assert_eq!(elements, tensor.data().len());
}

#[test]
fn labelled_blocks_borrow_u1_and_su2_values_in_canonical_order() {
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let leg = GradedSpace::try_new_with_arc(
        provider,
        [
            (tenet::core::U1Irrep::new(-1), 1),
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, indices| {
            100.0 * trees.coupled().charge() as f64
                + 10.0 * indices[0] as f64
                + 3.0 * indices[1] as f64
                + indices[2] as f64
        })
        .unwrap();

    let mut u1_blocks = u1.blocks().unwrap();
    assert_eq!(u1_blocks.len(), u1.block_count());
    for (index, (trees, values)) in u1_blocks.by_ref().enumerate() {
        let raw = u1.block(index).unwrap();
        assert_eq!(values.shape(), raw.shape());
        assert_eq!(values.strides(), raw.strides());
        assert_eq!(values.data().as_ptr(), u1.data().as_ptr());
        assert_eq!(trees.codomain_uncoupled().len(), 2);
        assert_eq!(
            trees.domain_uncoupled(),
            std::slice::from_ref(trees.coupled())
        );
        assert_eq!(
            trees.codomain_uncoupled()[0].charge() + trees.codomain_uncoupled()[1].charge(),
            trees.coupled().charge()
        );
        for i in 0..values.shape()[0] {
            for j in 0..values.shape()[1] {
                for k in 0..values.shape()[2] {
                    assert_eq!(
                        values.get(&[i, j, k]),
                        Some(
                            &(100.0 * trees.coupled().charge() as f64
                                + 10.0 * i as f64
                                + 3.0 * j as f64
                                + k as f64)
                        )
                    );
                }
            }
        }
    }
    assert!(matches!(
        u1.block(u1.block_count()),
        Err(tenet::typed::Error::Core(error))
            if matches!(*error, tenet::core::CoreError::BlockIndexOutOfBounds { index, count }
                if index == count && count == u1.block_count())
    ));

    let su2 = su2_tensor_split(&runtime, 2);
    let su2_blocks = su2.blocks().unwrap();
    assert_eq!(su2_blocks.len(), su2.block_count());
    for (index, (trees, values)) in su2_blocks.enumerate() {
        let raw = su2.block(index).unwrap();
        assert_eq!(trees.coupled(), &SU2Irrep::from_twice_spin(0));
        assert!(trees.domain_uncoupled().is_empty());
        assert!(SU2FusionRule
            .fusion_channels(
                SU2FusionRule
                    .encode_sector(&trees.codomain_uncoupled()[0])
                    .unwrap(),
                SU2FusionRule
                    .encode_sector(&trees.codomain_uncoupled()[1])
                    .unwrap(),
            )
            .contains(&SU2FusionRule.encode_sector(trees.coupled()).unwrap()));
        assert_eq!(values.shape(), raw.shape());
        assert_eq!(values.strides(), raw.strides());
        assert_eq!(values.data().as_ptr(), su2.data().as_ptr());
        assert_eq!(
            values.get(&vec![0; values.shape().len()]),
            Some(&su2.data()[raw.offset()])
        );
    }
}

#[test]
fn block_fusion_trees_reports_a_non_self_dual_domain_label() {
    // What: a dual domain leg carrying charge 2 — whose dual is charge 1, so a
    // confusion between the two would show — is decoded as charge 2, matching
    // the convention that a tree labels a domain leg with the space's own
    // sector rather than its dual.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalZ3::new());
    let codomain =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z3Charge(1), 1)]).unwrap();
    // A dual constructor interprets its key through the orientation, so key
    // charge 1 is stored and reported as the external charge 2.
    let domain = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z3Charge(1), 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    assert_eq!(
        domain.try_dual().unwrap().sectors().unwrap(),
        vec![Z3Charge(1)]
    );
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3, f64> =
        TensorMap::zeros(&runtime, [&codomain, &codomain], [&domain]).unwrap();

    assert_eq!(tensor.block_count(), 1);
    let sectors = tensor.block_fusion_trees(0).unwrap();
    assert_eq!(sectors.coupled(), &Z3Charge(2));
    assert_eq!(sectors.codomain_uncoupled(), &[Z3Charge(1), Z3Charge(1)]);
    assert_eq!(sectors.domain_uncoupled(), &[Z3Charge(2)]);
    assert!(tensor.domain()[0].is_dual());
}

// ---------------------------------------------------------------------------
// Slice 7: cross-cutting gates.
// ---------------------------------------------------------------------------

#[test]
fn simple_fusion_provider_round_trips_construction_fill_and_inspection() {
    // What: the whole phase-2 surface works for a non-abelian (Simple) external
    // provider with a complex payload, not just for the abelian fixture.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);
    let runtime = runtime();

    let tensor: TensorMap<ExternalSu2, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |sectors, indices| {
            Complex64::new(
                sectors.coupled().twice_spin() as f64,
                indices.iter().sum::<usize>() as f64,
            )
        })
        .unwrap();

    // Two spin-1/2 legs couple to spin 0 and spin 1 on each side.
    assert!(tensor.block_count() >= 2);
    let coupled: Vec<usize> = (0..tensor.block_count())
        .map(|index| {
            tensor
                .block_fusion_trees(index)
                .unwrap()
                .coupled()
                .twice_spin()
        })
        .collect();
    assert!(coupled.contains(&0) && coupled.contains(&2));
    assert!(tensor
        .data()
        .iter()
        .zip(0..)
        .all(|(value, _)| value.re == 0.0 || value.re == 2.0));
    assert_eq!(tensor.codomain().len(), 2);
    assert_eq!(tensor.domain().len(), 2);
}

/// The same value computed from the typed labels the facade hands the closure.
fn typed_fill_value(
    sectors: &tenet::typed::BlockFusionTrees<tenet::core::Z2Irrep>,
    indices: &[usize],
) -> f64 {
    let mut value = f64::from(sectors.coupled().parity()) * 1000.0;
    for (position, label) in sectors.codomain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 100.0 * (position + 1) as f64;
    }
    for (position, label) in sectors.domain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 10.0 * (position + 1) as f64;
    }
    value
        + indices
            .iter()
            .enumerate()
            .map(|(a, &i)| (a + 1) * i)
            .sum::<usize>() as f64
}

#[test]
fn typed_block_fill_preserves_tree_and_storage_order() {
    // What: each typed block exposes the same tree and column-major indices
    // that drove its fill closure, without borrowing the retired facade as an
    // oracle.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::Z2FusionRule);
    let leg = GradedSpace::try_new_with_arc(
        provider,
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed: TensorMap<tenet::core::Z2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], typed_fill_value).unwrap();

    assert!(typed.block_count() > 1);
    for index in 0..typed.block_count() {
        let block = typed.block(index).unwrap();
        let sectors = typed.block_fusion_trees(index).unwrap();
        for local in 0..block.element_count().unwrap() {
            let mut remainder = local;
            let indices: Vec<_> = block
                .shape()
                .iter()
                .map(|&extent| {
                    let index = remainder % extent;
                    remainder /= extent;
                    index
                })
                .collect();
            let offset = block.offset()
                + indices
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            assert_eq!(typed.data()[offset], typed_fill_value(&sectors, &indices));
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3, slice 2: `TensorMap::permute`.
// ---------------------------------------------------------------------------

/// The second Z3 leg shape, deliberately unlike [`z3_leg`]'s: with four legs of
/// two distinct degeneracy patterns, a permuted leg can be identified by its
/// degeneracies, which a uniform fixture could not distinguish.
fn z3_other_leg(provider: &Arc<ExternalZ3>, is_dual: bool) -> GradedSpace<ExternalZ3> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        [(Z3Charge(0), 1), (Z3Charge(1), 2), (Z3Charge(2), 4)],
    )
    .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
    .expect("Z3 leg is well formed")
}

/// A rank-4 Z3 tensor map whose elements are all distinct, so any leg, block or
/// element reordering is visible in the buffer. The layout is
/// `[wide, narrow] <- [wide', narrow']`, so no two axes share a shape.
fn z3_rank_four(runtime: &Runtime, provider: &Arc<ExternalZ3>) -> TensorMap<ExternalZ3, f64> {
    let wide = z3_leg(provider, false);
    let narrow = z3_other_leg(provider, false);
    let wide_dual = wide.try_dual().unwrap();
    let narrow_dual = narrow.try_dual().unwrap();
    let mut counter = 0.0;
    TensorMap::from_block_fn(
        runtime,
        [&wide, &narrow],
        [&wide_dual, &narrow_dual],
        |_, _| {
            counter += 1.0;
            counter
        },
    )
    .expect("Z3 rank-4 layout is admissible")
}

#[test]
fn permute_moves_legs_of_a_multi_block_external_provider_tensor() {
    // What: a non-identity permute of a runtime-rank multi-block external
    // provider tensor produces the reordered spaces, keeps the element count,
    // and moves the payload (this rule's F/R symbols are all 1, so the permuted
    // buffer is a rearrangement of the source, never a rescaling).
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    // The source is `[wide, narrow] <- [wide', narrow']`; this order sends one
    // leg of each shape across the codomain/domain split, so the degeneracies
    // pin which source leg landed where. Flags alone would not: they are
    // symmetric under swapping the split.
    let permuted = tensor.permute(&[1, 2], &[3, 0]).unwrap();

    assert_eq!(permuted.codomain().len(), 2);
    assert_eq!(permuted.domain().len(), 2);
    // Axis 1 (`narrow`) stays in the codomain unchanged; axis 2 (`wide'`) is
    // bent round from the domain, which conjugates it, so it arrives as the
    // non-dual `wide`. Symmetrically axis 0 (`wide`) bent into the domain
    // arrives as `wide'`, while axis 3 (`narrow'`) is carried across as is.
    assert_eq!(permuted.codomain()[0].degeneracies(), &[1, 2, 4]);
    assert_eq!(permuted.codomain()[1].degeneracies(), &[2, 3, 1]);
    assert_eq!(permuted.domain()[0].degeneracies(), &[1, 4, 2]);
    assert_eq!(permuted.domain()[1].degeneracies(), &[2, 1, 3]);
    assert!(!permuted.codomain()[0].is_dual());
    assert!(!permuted.codomain()[1].is_dual());
    assert!(permuted.domain()[0].is_dual());
    assert!(permuted.domain()[1].is_dual());
    assert_eq!(permuted.data().len(), tensor.data().len());
    assert_ne!(permuted.data(), tensor.data());
    let mut moved: Vec<f64> = permuted.data().to_vec();
    let mut original: Vec<f64> = tensor.data().to_vec();
    moved.sort_by(f64::total_cmp);
    original.sort_by(f64::total_cmp);
    assert_eq!(moved, original);
}

#[test]
fn permute_round_trips_back_to_the_source_layout() {
    // What: permuting and permuting back is the identity on both the spaces and
    // the payload — a stronger statement than "the multiset survived", since it
    // pins where each element landed.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let there = tensor.permute(&[1, 3], &[0, 2]).unwrap();
    let back = there.permute(&[2, 0], &[3, 1]).unwrap();

    assert_eq!(back.data(), tensor.data());
    assert_eq!(back.block_count(), tensor.block_count());
    for index in 0..tensor.block_count() {
        assert_eq!(
            back.block_fusion_trees(index).unwrap(),
            tensor.block_fusion_trees(index).unwrap()
        );
    }
}

#[test]
fn permute_carries_a_simple_fusion_provider_with_a_complex_payload() {
    // What: nothing in the typed transform is abelian- or real-specific; the
    // SU(2) recoupling coefficients reach a `Complex64` payload.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);
    let tensor: TensorMap<ExternalSu2, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |sectors, indices| {
            Complex64::new(
                sectors.coupled().twice_spin() as f64 + 1.0,
                indices.iter().sum::<usize>() as f64 + 1.0,
            )
        })
        .unwrap();

    let permuted = tensor.permute(&[0, 2], &[1, 3]).unwrap();

    assert_eq!(permuted.codomain().len(), 2);
    assert_eq!(permuted.domain().len(), 2);
    assert!(permuted.block_count() >= 1);
    assert!(permuted
        .data()
        .iter()
        .any(|value| *value != Complex64::new(0.0, 0.0)));
}

#[test]
fn permute_rejects_malformed_axes_without_panicking() {
    // What: the expert layer's typed errors are the contract — an out-of-range
    // axis, a repeated axis and a wrong-length axis list all come back as `Err`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    assert!(tensor.permute(&[0, 9], &[2, 3]).is_err());
    assert!(tensor.permute(&[0, 0], &[2, 3]).is_err());
    assert!(tensor.permute(&[0], &[2, 3]).is_err());
}

/// One built-in Z2 layout filled from typed fusion-tree labels.
fn z2_tensor(runtime: &Runtime) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    z2_tensor_split(runtime, 2)
}

/// The same tensor over three identical legs, split into
/// `num_codomain <- rest`. `2` is tall in every coupled sector and `1` is wide,
/// which is what separates the compact factorizations from the full ones:
/// on a tall input LQ-compact and LQ-full coincide.
fn z2_tensor_split(
    runtime: &Runtime,
    num_codomain: usize,
) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let legs = [&leg, &leg, &leg];
    let (codomain, domain) = legs.split_at(num_codomain);
    TensorMap::from_block_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        typed_fill_value,
    )
    .unwrap()
}

/// The c64 sibling of [`z2_tensor`]. The imaginary part is deliberately
/// not proportional to the real one, so a stray conjugation or a real-only
/// path is visible in every comparison this tensor feeds.
fn z2_complex_tensor(runtime: &Runtime) -> TensorMap<tenet::core::Z2FusionRule, Complex64> {
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], |sectors, indices| {
        complex(typed_fill_value(sectors, indices))
    })
    .unwrap()
}

// ---------------------------------------------------------------------------
// Phase 3, slice 3: `TensorMap::contract`.
// ---------------------------------------------------------------------------

/// A single-sector Z3 leg: the whole tensor map is then one dense block, so a
/// contraction result can be checked against a hand-computed matrix product.
fn z3_dense_leg(provider: &Arc<ExternalZ3>, degeneracy: usize) -> GradedSpace<ExternalZ3> {
    GradedSpace::try_new_with_arc(Arc::clone(provider), [(Z3Charge(0), degeneracy)])
        .expect("single-sector Z3 leg is well formed")
}

/// Fills a tensor map's storage with `start, start + 1, ...` in storage order.
fn counting_z3(
    runtime: &Runtime,
    codomain: &GradedSpace<ExternalZ3>,
    domain: &GradedSpace<ExternalZ3>,
    start: f64,
) -> TensorMap<ExternalZ3, f64> {
    let mut next = start - 1.0;
    TensorMap::from_block_fn(runtime, [codomain], [domain], |_, _| {
        next += 1.0;
        next
    })
    .expect("single-block Z3 layout is admissible")
}

#[test]
fn contract_matches_a_hand_computed_product_with_a_reordered_output() {
    // What: a 2x3 by 3x4 contraction with `output_axes = [1, 0]` is the
    // transpose of the matrix product, element for element. The values are the
    // ones the expert-layer helper is pinned to, so the facade is proved to
    // pass the axes and the output order through unaltered.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 7.0);
    assert_eq!(lhs.data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(rhs.data().len(), 12);

    let contracted = lhs.contract(&rhs, &[1], &[0], &[1, 0]).unwrap();

    assert_eq!(
        contracted.data(),
        [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
    );
    assert_eq!(contracted.codomain().len(), 1);
    assert_eq!(contracted.domain().len(), 1);
}

#[test]
fn contract_with_the_default_output_order_keeps_the_open_axes_in_place() {
    // What: `0..open_rank` is the identity output order, so the same product
    // comes back untransposed.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 7.0);

    let contracted = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();

    assert_eq!(
        contracted.data(),
        [76.0, 100.0, 103.0, 136.0, 130.0, 172.0, 157.0, 208.0]
    );
}

#[test]
fn contract_carries_a_simple_fusion_provider_with_a_complex_payload() {
    // What: the contraction seam is neither abelian- nor real-specific.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);
    let mut next = Complex64::new(0.0, 0.0);
    let build = |next: &mut Complex64| {
        let mut step = *next;
        let tensor: TensorMap<ExternalSu2, Complex64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |_, _| {
                step += Complex64::new(1.0, 0.5);
                step
            })
            .unwrap();
        *next = step;
        tensor
    };
    let lhs = build(&mut next);
    let rhs = build(&mut next);

    // Two spin-1/2 legs on each side, so both operands carry the spin-0 and
    // spin-1 blocks; the pair of contracted legs makes the result rank 4.
    let contracted = lhs.contract(&rhs, &[2, 3], &[0, 1], &[2, 0, 3, 1]).unwrap();

    assert_eq!(contracted.codomain().len() + contracted.domain().len(), 4);
    assert!(contracted
        .data()
        .iter()
        .any(|value| *value != Complex64::new(0.0, 0.0)));
}

#[test]
fn contract_rejects_operands_from_different_runtimes() {
    // What: the runtime is a trust boundary — two runtimes own separate
    // execution state, so mixing them is refused before any provider work.
    let _guard = cache_lock();
    let first = runtime();
    let second = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&first, &rows, &shared, 1.0);
    let rhs = counting_z3(&second, &shared, &columns, 1.0);

    let error = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap_err();

    assert!(matches!(error, tenet::prelude::Error::RuntimeMismatch));
}

#[test]
fn contract_accepts_separately_allocated_equal_identity_providers() {
    // What: the counterpart of the distinct-identity rejection — two
    // independent allocations of one rule interoperate in an operation, not
    // just in construction, and produce the same values a single allocation
    // does.
    let _guard = cache_lock();
    let runtime = runtime();
    let first = Arc::new(ExternalZ3::new());
    let second = Arc::new(ExternalZ3::new());
    assert!(!Arc::ptr_eq(&first, &second));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&first, 2),
        &z3_dense_leg(&first, 3),
        1.0,
    );
    let rhs = counting_z3(
        &runtime,
        &z3_dense_leg(&second, 3),
        &z3_dense_leg(&second, 4),
        7.0,
    );

    let contracted = lhs.contract(&rhs, &[1], &[0], &[1, 0]).unwrap();

    assert_eq!(
        contracted.data(),
        [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
    );
}

#[test]
fn contract_rejects_operands_with_distinct_rule_identities() {
    // What: two providers of one Rust type but different identities are a
    // different algebra. The rejection comes from the expert layer, which is
    // why the facade does not pre-check it.
    let _guard = cache_lock();
    let runtime = runtime();
    let first = Arc::new(ExternalZ3::tagged(0));
    let second = Arc::new(ExternalZ3::tagged(1));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&first, 2),
        &z3_dense_leg(&first, 3),
        1.0,
    );
    let rhs = counting_z3(
        &runtime,
        &z3_dense_leg(&second, 3),
        &z3_dense_leg(&second, 4),
        1.0,
    );

    assert!(lhs.contract(&rhs, &[1], &[0], &[0, 1]).is_err());
}

#[test]
fn contract_rejects_malformed_axes_without_panicking() {
    // What: mismatched axis-list lengths, out-of-range axes and a wrong-length
    // output order all come back as `Err`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 1.0);

    assert!(lhs.contract(&rhs, &[1], &[], &[0, 1]).is_err());
    assert!(lhs.contract(&rhs, &[9], &[0], &[0, 1]).is_err());
    assert!(lhs.contract(&rhs, &[1], &[9], &[0, 1]).is_err());
    assert!(lhs.contract(&rhs, &[1], &[0], &[0]).is_err());
}

#[test]
fn otimes_rejects_runtime_and_rule_identity_mismatches() {
    let _guard = cache_lock();
    let first = runtime();
    let second = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let lhs = counting_z3(
        &first,
        &z3_dense_leg(&provider, 2),
        &z3_dense_leg(&provider, 3),
        1.0,
    );
    let other_runtime = counting_z3(
        &second,
        &z3_dense_leg(&provider, 2),
        &z3_dense_leg(&provider, 3),
        1.0,
    );
    assert!(matches!(
        lhs.otimes(&other_runtime).unwrap_err(),
        tenet::prelude::Error::RuntimeMismatch
    ));

    let other_rule = Arc::new(ExternalZ3::tagged(1));
    let other_rule = counting_z3(
        &first,
        &z3_dense_leg(&other_rule, 2),
        &z3_dense_leg(&other_rule, 3),
        1.0,
    );
    assert!(lhs.otimes(&other_rule).is_err());
}

// TensorKitSectors `anyons.jl` PlanarTrivial: one simple object, unique
// fusion, no braiding, N = F = 1, and the object is the canonical unit.
#[derive(Clone, Copy)]
struct PlanarTrivial;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PlanarTrivialSector;

impl FusionRule for PlanarTrivial {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::NoBraiding
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }
}

impl MultiplicityFreeFusionRule for PlanarTrivial {}
impl tenet::core::CanonicalUnitFusionRule for PlanarTrivial {}

impl MultiplicityFreeFusionSymbols for PlanarTrivial {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        panic!("PlanarTrivial has no R symbol")
    }
}

impl MultiplicityFreeRigidSymbols for PlanarTrivial {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

impl CheckedFusionAlgebra for PlanarTrivial {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if sector == SectorId::new(0) {
            Ok(sector)
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        self.try_dual_sector(left)?;
        self.try_dual_sector(right)?;
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.try_dual_sector(left)?;
        self.try_dual_sector(right)?;
        self.try_dual_sector(coupled)?;
        Ok(1)
    }
}

impl SectorCodec for PlanarTrivial {
    type Sector = PlanarTrivialSector;
    fn encode_sector(&self, _: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(0))
    }
    fn decode_sector(&self, sector: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        self.try_dual_sector(sector)?;
        Ok(PlanarTrivialSector)
    }
}

#[test]
fn otimes_matches_tensorkit_planar_trivial_without_requesting_braiding() {
    // What: the #595 NoBraiding oracle is TensorKit's own PlanarTrivial
    // category. The previous contract-plus-output-permute route reached its
    // NoBraiding boundary; the monoidal merge succeeds without an R symbol.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(PlanarTrivial);
    let lhs_cod =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(PlanarTrivialSector, 2)]).unwrap();
    let lhs_dom = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(PlanarTrivialSector, 3)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_cod = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(PlanarTrivialSector, 4)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_dom =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(PlanarTrivialSector, 2)]).unwrap();
    let lhs: TensorMap<PlanarTrivial, f64> =
        TensorMap::from_block_fn(&runtime, [&lhs_cod], [&lhs_dom], |_, indices| {
            (1 + indices[0] + 10 * indices[1]) as f64
        })
        .unwrap();
    let rhs: TensorMap<PlanarTrivial, f64> =
        TensorMap::from_block_fn(&runtime, [&rhs_cod], [&rhs_dom], |_, indices| {
            (2 + indices[0] + 10 * indices[1]) as f64
        })
        .unwrap();
    let expected: TensorMap<PlanarTrivial, f64> = TensorMap::from_block_fn(
        &runtime,
        [&lhs_cod, &rhs_cod],
        [&lhs_dom, &rhs_dom],
        |_, indices| {
            (1 + indices[0] + 10 * indices[2]) as f64 * (2 + indices[1] + 10 * indices[3]) as f64
        },
    )
    .unwrap();

    // The pre-#595 lowering encoded the same operation as empty-axis
    // contraction plus this interleaving output permutation. NoBraiding still
    // rejects that braid-requiring route.
    assert!(lhs.contract(&rhs, &[], &[], &[0, 2, 1, 3]).is_err());

    let actual = lhs.otimes(&rhs).unwrap();

    assert_eq!(actual.data(), expected.data());
    assert_eq!(
        actual
            .codomain()
            .into_iter()
            .chain(actual.domain())
            .map(|space| space.is_dual())
            .collect::<Vec<_>>(),
        [false, true, true, false]
    );
}

#[test]
fn otimes_fz2_complex_oracle_has_no_crossing_phase() {
    // Built-in TensorKit-equivalent FermionParity semantics: complex payload
    // multiplication is sign-free because otimes performs no leg crossing.
    let _guard = cache_lock();
    let runtime = runtime();
    let rule = Arc::new(tenet::core::FermionParityFusionRule);
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |sectors, _| {
            if *sectors.coupled() == tenet::core::Z2Irrep::EVEN {
                Complex64::new(2.0, 1.0)
            } else {
                Complex64::new(-3.0, 2.0)
            }
        })
        .unwrap();
    let rhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |sectors, _| {
            if *sectors.coupled() == tenet::core::Z2Irrep::EVEN {
                Complex64::new(5.0, -1.0)
            } else {
                Complex64::new(1.0, 4.0)
            }
        })
        .unwrap();
    let expected: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |sectors, _| {
            let codomain = sectors.codomain_uncoupled();
            let domain = sectors.domain_uncoupled();
            if codomain != domain {
                return Complex64::new(0.0, 0.0);
            }
            let lhs = if codomain[0] == tenet::core::Z2Irrep::EVEN {
                Complex64::new(2.0, 1.0)
            } else {
                Complex64::new(-3.0, 2.0)
            };
            let rhs = if codomain[1] == tenet::core::Z2Irrep::EVEN {
                Complex64::new(5.0, -1.0)
            } else {
                Complex64::new(1.0, 4.0)
            };
            lhs * rhs
        })
        .unwrap();

    assert_eq!(lhs.otimes(&rhs).unwrap().data(), expected.data());
}

#[test]
fn typed_deligne_product_uses_the_explicit_component_order() {
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(tenet::core::U1FusionRule);
    let fz2_rule = Arc::new(tenet::core::FermionParityFusionRule);
    let charge =
        GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(tenet::core::U1Irrep::new(1), 1)])
            .unwrap();
    let parity =
        GradedSpace::try_new_with_arc(Arc::clone(&fz2_rule), [(tenet::core::Z2Irrep::ODD, 1)])
            .unwrap();
    let lhs = TensorMap::from_block_fn(&runtime, [&charge], [&charge], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_block_fn(&runtime, [&parity], [&parity], |_, _| 3.0).unwrap();
    let product = Arc::new(tenet::core::U1FusionRule.product(tenet::core::FermionParityFusionRule));

    let result = lhs.deligne_product(&rhs, product).unwrap();

    assert_eq!((result.codomain_rank(), result.domain_rank()), (2, 2));
    assert_eq!(result.data(), [6.0]);
    let codomain = result.codomain();
    assert_eq!(
        codomain[0].sectors().unwrap(),
        [tenet::core::product_sector(
            tenet::core::U1Irrep::new(1),
            tenet::core::Z2Irrep::EVEN
        )]
    );
    assert_eq!(
        codomain[1].sectors().unwrap(),
        [tenet::core::product_sector(
            tenet::core::U1Irrep::new(0),
            tenet::core::Z2Irrep::ODD
        )]
    );
}

#[test]
fn typed_deligne_product_rejects_a_component_identity_mismatch() {
    let _guard = cache_lock();
    let runtime = runtime();
    let lhs_rule = Arc::new(ExternalZ3::tagged(0));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&lhs_rule, 1),
        &z3_dense_leg(&lhs_rule, 1),
        2.0,
    );
    let u1_rule = Arc::new(tenet::core::U1FusionRule);
    let u1 =
        GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(tenet::core::U1Irrep::new(0), 1)])
            .unwrap();
    let rhs = TensorMap::from_block_fn(&runtime, [&u1], [&u1], |_, _| 3.0).unwrap();
    let wrong = Arc::new(ExternalZ3::tagged(1).product(tenet::core::U1FusionRule));

    assert!(matches!(
        lhs.deligne_product(&rhs, wrong).unwrap_err(),
        tenet::prelude::Error::RuleMismatch
    ));
}

#[test]
fn typed_deligne_product_checks_runtime_before_both_component_identities() {
    let _guard = cache_lock();
    let first = runtime();
    let second = runtime();
    let left_rule = Arc::new(ExternalZ3::tagged(0));
    let right_rule = Arc::new(ExternalZ3::tagged(2));
    let lhs = counting_z3(
        &first,
        &z3_dense_leg(&left_rule, 1),
        &z3_dense_leg(&left_rule, 1),
        2.0,
    );
    let rhs = counting_z3(
        &second,
        &z3_dense_leg(&right_rule, 1),
        &z3_dense_leg(&right_rule, 1),
        3.0,
    );
    let wrong_both = Arc::new(ExternalZ3::tagged(1).product(ExternalZ3::tagged(3)));

    assert!(matches!(
        lhs.deligne_product(&rhs, wrong_both).unwrap_err(),
        tenet::prelude::Error::RuntimeMismatch
    ));

    let rhs = counting_z3(
        &first,
        &z3_dense_leg(&right_rule, 1),
        &z3_dense_leg(&right_rule, 1),
        3.0,
    );
    let wrong_right = Arc::new(ExternalZ3::tagged(0).product(ExternalZ3::tagged(3)));
    assert!(matches!(
        lhs.deligne_product(&rhs, wrong_right).unwrap_err(),
        tenet::prelude::Error::RuleMismatch
    ));
}

#[test]
fn typed_deligne_product_preserves_duals_multiblocks_and_complex_values() {
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(tenet::core::U1FusionRule);
    let fz2_rule = Arc::new(tenet::core::FermionParityFusionRule);
    let charge_cod = GradedSpace::try_new_with_arc(
        Arc::clone(&u1_rule),
        [
            (tenet::core::U1Irrep::new(-1), 1),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let charge_dom = GradedSpace::try_new_with_arc(
        Arc::clone(&u1_rule),
        [
            (tenet::core::U1Irrep::new(-1), 1),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let parity_cod = GradedSpace::try_new_with_arc(
        Arc::clone(&fz2_rule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let parity_dom = GradedSpace::try_new_with_arc(
        Arc::clone(&fz2_rule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&charge_cod], [&charge_dom], |sectors, _| {
            Complex64::new(sectors.coupled().charge() as f64, 2.0)
        })
        .unwrap();
    let rhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&parity_cod], [&parity_dom], |sectors, _| {
            if *sectors.coupled() == tenet::core::Z2Irrep::EVEN {
                Complex64::new(3.0, -1.0)
            } else {
                Complex64::new(-2.0, 4.0)
            }
        })
        .unwrap();
    let product = Arc::new(tenet::core::U1FusionRule.product(tenet::core::FermionParityFusionRule));

    let actual = lhs.deligne_product(&rhs, product).unwrap();
    let codomain = actual.codomain();
    let domain = actual.domain();
    let expected =
        TensorMap::from_block_fn(&runtime, codomain.iter(), domain.iter(), |sectors, _| {
            let codomain = sectors.codomain_uncoupled();
            let domain = sectors.domain_uncoupled();
            if codomain[0].left() == domain[0].left() && codomain[1].right() == domain[1].right() {
                let lhs = Complex64::new(codomain[0].left().charge() as f64, 2.0);
                let rhs = if *codomain[1].right() == tenet::core::Z2Irrep::EVEN {
                    Complex64::new(3.0, -1.0)
                } else {
                    Complex64::new(-2.0, 4.0)
                };
                lhs * rhs
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();

    assert_eq!(actual.data(), expected.data());
    assert!(actual.block_count() > 1);
    assert_eq!(
        codomain
            .into_iter()
            .chain(domain)
            .map(|space| space.is_dual())
            .collect::<Vec<_>>(),
        [false, true, true, false]
    );
}

#[test]
fn typed_deligne_product_accepts_a_nondefault_product_codec() {
    type Codec =
        tenet::core::PackedProductCodec<tenet::core::U1SectorLayout, tenet::core::Fz2SectorLayout>;
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(tenet::core::U1FusionRule);
    let fz2_rule = Arc::new(tenet::core::FermionParityFusionRule);
    let charge =
        GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(tenet::core::U1Irrep::new(2), 1)])
            .unwrap();
    let parity =
        GradedSpace::try_new_with_arc(Arc::clone(&fz2_rule), [(tenet::core::Z2Irrep::ODD, 1)])
            .unwrap();
    let lhs = TensorMap::from_block_fn(&runtime, [&charge], [&charge], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_block_fn(&runtime, [&parity], [&parity], |_, _| 5.0).unwrap();
    let product = Arc::new(tenet::core::ProductFusionRule::<_, _, Codec>::new(
        tenet::core::U1FusionRule,
        tenet::core::FermionParityFusionRule,
    ));

    let result = lhs.deligne_product(&rhs, product).unwrap();

    assert_eq!(result.data(), [10.0]);
    assert_eq!(
        result.codomain()[0].sectors().unwrap(),
        [tenet::core::product_sector(
            tenet::core::U1Irrep::new(2),
            tenet::core::Z2Irrep::EVEN
        )]
    );
}

#[test]
fn typed_deligne_product_maps_component_innerlines_into_the_product_tree() {
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(tenet::core::U1FusionRule);
    let fz2_rule = Arc::new(tenet::core::FermionParityFusionRule);
    let charges = [1, 2, 3].map(|charge| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&u1_rule),
            [(tenet::core::U1Irrep::new(charge), 1)],
        )
        .unwrap()
    });
    let charge_total =
        GradedSpace::try_new_with_arc(Arc::clone(&u1_rule), [(tenet::core::U1Irrep::new(6), 1)])
            .unwrap();
    let odd =
        GradedSpace::try_new_with_arc(Arc::clone(&fz2_rule), [(tenet::core::Z2Irrep::ODD, 1)])
            .unwrap();
    let lhs =
        TensorMap::from_block_fn(&runtime, charges.iter(), [&charge_total], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_block_fn(&runtime, [&odd, &odd, &odd], [&odd], |_, _| 3.0).unwrap();
    let product = Arc::new(tenet::core::U1FusionRule.product(tenet::core::FermionParityFusionRule));

    let result = lhs.deligne_product(&rhs, product).unwrap();
    let block = result.block_fusion_trees(0).unwrap();

    assert_eq!(result.data(), [6.0]);
    assert_eq!(
        block.codomain_innerlines(),
        [
            tenet::core::product_sector(tenet::core::U1Irrep::new(3), tenet::core::Z2Irrep::EVEN),
            tenet::core::product_sector(tenet::core::U1Irrep::new(6), tenet::core::Z2Irrep::EVEN),
            tenet::core::product_sector(tenet::core::U1Irrep::new(6), tenet::core::Z2Irrep::ODD),
            tenet::core::product_sector(tenet::core::U1Irrep::new(6), tenet::core::Z2Irrep::EVEN),
        ]
    );
}

#[test]
fn typed_deligne_product_prepares_both_embeddings_before_publishing_either() {
    type WrongCodec =
        tenet::core::PackedProductCodec<tenet::core::U1SectorLayout, tenet::core::Fz2SectorLayout>;
    let _guard = cache_lock();
    let runtime = runtime();
    let rule = Arc::new(tenet::core::U1FusionRule);
    let charge_one =
        GradedSpace::try_new_with_arc(Arc::clone(&rule), [(tenet::core::U1Irrep::new(1), 1)])
            .unwrap();
    let lhs = TensorMap::from_block_fn(&runtime, [&charge_one], [&charge_one], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_block_fn(&runtime, [&charge_one], [&charge_one], |_, _| 3.0).unwrap();
    let product = Arc::new(tenet::core::ProductFusionRule::<
        tenet::core::U1FusionRule,
        tenet::core::U1FusionRule,
        WrongCodec,
    >::new(
        tenet::core::U1FusionRule, tenet::core::U1FusionRule
    ));
    let before = (
        fusion_tree_layout_cache_info(),
        complete_hom_space_structure_cache_info(),
    );

    assert!(lhs.deligne_product(&rhs, product).is_err());

    assert_eq!(
        (
            fusion_tree_layout_cache_info(),
            complete_hom_space_structure_cache_info(),
        ),
        before
    );
}

// ---------------------------------------------------------------------------
// Phase 3, slice 4: non-regression gates.
// ---------------------------------------------------------------------------

#[test]
fn a_failing_typed_operation_publishes_no_cache_state() {
    // What: an operation rejected by the expert layer leaves both process-global
    // layout caches and the runtime's tree-transform cache exactly as they were,
    // the same transactional guarantee construction gives.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);
    let before = (
        fusion_tree_layout_cache_info(),
        complete_hom_space_structure_cache_info(),
    );
    let runtime_before = runtime.tree_transform_cache_info();

    assert!(tensor.permute(&[0, 0], &[2, 3]).is_err());
    assert!(tensor
        .contract(&tensor, &[3], &[9], &[0, 1, 2, 3, 4, 5])
        .is_err());

    assert_eq!(
        (
            fusion_tree_layout_cache_info(),
            complete_hom_space_structure_cache_info(),
        ),
        before
    );
    assert_eq!(runtime.tree_transform_cache_info(), runtime_before);
}

// ---------------------------------------------------------------------------
// Phase 4, slice 1: `TensorMap::braid`.
// ---------------------------------------------------------------------------

#[test]
fn braid_moves_legs_of_a_multi_block_external_provider_tensor() {
    // What: an explicit braid with a full level assignment produces the same
    // reordered spaces a permute of the same axes does, and moves the payload.
    //
    // Why not a case where braid differs from permute: the level *values* are
    // unobservable for every provider this facade can host. The symmetric ones
    // (both fixtures here, the built-in Z2/SU(2), and even the fermionic rule
    // used further down) make over- and under-crossing the same morphism, and
    // the one built-in rule that would not — `FibonacciFusionRule` — is
    // excluded by this facade's `Scalar = f64` and `SectorCodec` bounds.
    //
    // What the tests below therefore do and do not prove: they pin how the
    // levels are *split* (by the source codomain rank, which the oracle below
    // pins with an axis list of a different length) and that a wrong-length
    // list is refused. Nothing here can pin the values, and no test in this
    // crate can until an anyonic provider is reachable.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let braided = tensor.braid(&[1, 2], &[3, 0], &[0, 1, 2, 3]).unwrap();
    let permuted = tensor.permute(&[1, 2], &[3, 0]).unwrap();

    assert_eq!(braided.data(), permuted.data());
    assert_ne!(braided.data(), tensor.data());
    assert_eq!(braided.codomain()[0].degeneracies(), &[1, 2, 4]);
    assert_eq!(braided.domain()[1].degeneracies(), &[2, 1, 3]);
    // A different level assignment is the same morphism for a bosonic rule.
    let reversed = tensor.braid(&[1, 2], &[3, 0], &[3, 2, 1, 0]).unwrap();
    assert_eq!(reversed.data(), braided.data());
}

#[test]
fn braid_rejects_a_wrong_length_levels_list() {
    // What: the public pre-check — `levels` must name every source axis — in
    // both directions.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let short = tensor.braid(&[1, 2], &[3, 0], &[0, 1, 2]).unwrap_err();
    let message = short.to_string();
    assert!(message.contains("one level per source axis"), "{message}");
    assert!(message.contains("expected 4"), "{message}");
    assert!(tensor.braid(&[1, 2], &[3, 0], &[0; 5]).is_err());
    // Malformed axes still come back from the expert layer, not from here.
    assert!(tensor.braid(&[0, 0], &[2, 3], &[0, 1, 2, 3]).is_err());
}

// ---------------------------------------------------------------------------
// Phase 4, slice 2: `TensorMap::transpose` and `TensorMap::transpose_axes`.
// ---------------------------------------------------------------------------

/// The `(is_dual, degeneracies)` shape of a typed tensor map's legs, codomain
/// first — enough to pin where each source leg landed and how it was bent.
fn typed_leg_shapes<R, D>(tensor: &TensorMap<R, D>) -> Vec<(bool, Vec<usize>)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: tenet::prelude::TensorScalar,
{
    tensor
        .codomain()
        .iter()
        .chain(tensor.domain().iter())
        .map(|leg| (leg.is_dual(), leg.degeneracies().to_vec()))
        .collect()
}

#[test]
fn transpose_twice_returns_the_source_layout() {
    // What: the planar transpose is an involution — it rotates every leg once
    // round the boundary, so applying it twice restores the source spaces,
    // block order and bytes exactly.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let once = tensor.transpose().unwrap();
    let twice = once.transpose().unwrap();

    assert_ne!(once.data(), tensor.data());
    assert_eq!(typed_leg_shapes(&twice), typed_leg_shapes(&tensor));
    assert_eq!(twice.data(), tensor.data());
    assert_eq!(twice.block_count(), tensor.block_count());
    for index in 0..tensor.block_count() {
        assert_eq!(
            twice.block_fusion_trees(index).unwrap(),
            tensor.block_fusion_trees(index).unwrap()
        );
    }
}

#[test]
fn transpose_axes_rejects_malformed_axes_without_panicking() {
    // What: out-of-range axes, a wrong-length list and a non-planar
    // re-arrangement (a permute, which `transpose_axes` must refuse rather than
    // silently braid) all come back as `Err`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    assert!(tensor.transpose_axes(&[0, 9], &[2, 3]).is_err());
    assert!(tensor.transpose_axes(&[0], &[2, 3]).is_err());
    assert!(tensor.transpose_axes(&[1, 2], &[3, 0]).is_err());
}

// ---------------------------------------------------------------------------
// Phase 4, slice 3: `TensorMap::repartition`.
// ---------------------------------------------------------------------------

#[test]
fn repartition_moves_the_boundary_and_round_trips_at_every_split() {
    // What: every split point of a rank-4 tensor map is reachable, reports the
    // requested codomain/domain sizes, and comes back to the source layout —
    // spaces, block identities and bytes — when repartitioned to the original
    // split.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    for num_codomain in 0..=4 {
        let moved = tensor.repartition(num_codomain).unwrap();
        assert_eq!(moved.codomain().len(), num_codomain);
        assert_eq!(moved.domain().len(), 4 - num_codomain);
        assert_eq!(moved.data().len(), tensor.data().len());

        let back = moved.repartition(2).unwrap();
        assert_eq!(typed_leg_shapes(&back), typed_leg_shapes(&tensor));
        assert_eq!(back.data(), tensor.data());
        for index in 0..tensor.block_count() {
            assert_eq!(
                back.block_fusion_trees(index).unwrap(),
                tensor.block_fusion_trees(index).unwrap()
            );
        }
    }
}

#[test]
fn repartition_rejects_a_split_beyond_the_rank() {
    // What: `num_codomain > rank` has no planar reading at all, so it is
    // rejected rather than clamped.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let error = tensor.repartition(5).unwrap_err();

    assert!(error.to_string().contains("exceeds rank 4"), "{error}");
}

// ---------------------------------------------------------------------------
// Phase 4, slice 4: `use tenet::typed::*` self-sufficiency (issue #557, O7b).
// ---------------------------------------------------------------------------

/// Deliberately imports nothing but the typed facade's glob and the provider
/// this suite defines: if `Error` or `Runtime` were missing from the module,
/// this module would not compile. The provider itself must come from
/// somewhere — a typed facade is parameterised by one — which is exactly the
/// "self-sufficient apart from the provider" claim.
mod typed_glob_is_self_sufficient {
    use std::sync::Arc;
    use tenet::prelude::*;

    use super::{ExternalZ3, Z3Charge};

    #[test]
    fn a_glob_import_runs_an_end_to_end_typed_operation() {
        let _: Option<SvdTrunc<ExternalZ3, f64>> = None;
        let _: Option<EigTrunc<ExternalZ3, f64>> = None;
        let _: Option<EighTrunc<ExternalZ3, f64>> = None;

        let _guard = super::cache_lock();
        let runtime: Runtime = Runtime::builder().build().expect("runtime builds");
        let provider = Arc::new(ExternalZ3::new());
        let leg = GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            [(Z3Charge(0), 2), (Z3Charge(1), 3)],
        )
        .expect("leg is well formed");

        let build = || -> Result<TensorMap<ExternalZ3, f64>, Error> {
            let mut next = 0.0;
            let tensor = TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
                next += 1.0;
                next
            })?;
            tensor.transpose()
        };

        let transposed = build().expect("the typed pipeline runs");
        assert_eq!(transposed.codomain().len(), 1);
        assert_eq!(transposed.domain().len(), 2);

        // The re-exported `Error` is the same type the facade returns, not a
        // lookalike: this only type-checks if the two are one.
        let failure: Error = transposed.repartition(9).unwrap_err();
        assert!(matches!(failure, Error::InvalidArgument(_)));
    }
}

// ---------------------------------------------------------------------------
// Phase 4, slice 5: planar is not permute.
//
// The built-in `FermionParityFusionRule` is the only rule reachable from this
// facade whose braiding is not symmetric (`BraidingStyleKind::Fermionic`), so
// it is the only one that can tell a planar bend apart from a braid. It meets
// every typed-facade bound — it is a genuine external-shaped provider here,
// not crate-internal machinery.
// ---------------------------------------------------------------------------

/// A rank-2 fermionic tensor map, `[odd, odd] <- []`. Both legs odd, so every
/// bend crosses a fermion past a fermion and the sign is observable; the empty
/// domain keeps the layout to a single one-element block, so a sign flip is the
/// only thing a comparison can be reporting.
/// A fermionic leg carrying both parities, degeneracy one each.
fn fermionic_leg() -> GradedSpace<tenet::core::FermionParityFusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::FermionParityFusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .expect("fermionic leg is well formed")
}

/// `[leg, leg] <- [leg]`, counting fill: four elements across two blocks, all
/// distinct, so both the motion and the sign of every element are visible.
fn fermionic_rank_three(runtime: &Runtime) -> TensorMap<tenet::core::FermionParityFusionRule, f64> {
    let leg = fermionic_leg();
    let mut next = 0.0;
    TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .expect("fermionic layout is admissible")
}

#[test]
fn planar_transposes_bend_where_permute_braids_for_a_fermionic_provider() {
    // What: for a provider whose braiding is not symmetric, a planar transpose
    // and a permute of the *same* axes are different morphisms — the permute
    // crosses strands and picks up the fermionic sign, the planar bend does
    // not. Every other test in this suite is blind to the difference, because
    // every other provider it can host is bosonic; spelling `transpose` as a
    // `permute` would pass all of them and be wrong here.
    let _guard = cache_lock();
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);
    assert_eq!(tensor.data(), [1.0, 2.0, 3.0, 4.0]);

    // Full transpose: same element motion either way, opposite signs.
    assert_eq!(tensor.transpose().unwrap().data(), [1.0, 2.0, 4.0, 3.0]);
    assert_eq!(
        tensor.permute(&[2], &[1, 0]).unwrap().data(),
        [1.0, 2.0, -4.0, -3.0]
    );

    // The explicit form, on a different rotation of the planar order.
    assert_eq!(
        tensor.transpose_axes(&[1, 2], &[0]).unwrap().data(),
        [1.0, 4.0, 2.0, 3.0]
    );
    assert_eq!(
        tensor.permute(&[1, 2], &[0]).unwrap().data(),
        [1.0, 4.0, -2.0, -3.0]
    );
}

#[test]
fn repartition_is_sign_free_even_for_a_fermionic_provider() {
    // What: moving the planar boundary never crosses two strands — the cyclic
    // order of the legs is what `repartition` preserves by definition — so
    // unlike the transposes above, its result carries no braiding phase and
    // coincides with the permute of the same axes even fermionically.
    //
    // Recorded because it is not obvious and it bounds what a test can prove:
    // no provider this facade can host makes a `repartition`-only substitution
    // of the planar transform by a braided one observable. The two transposes
    // above are what guard the shared planar helper the three methods route
    // through.
    let _guard = cache_lock();
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);

    assert_eq!(tensor.repartition(0).unwrap().data(), [1.0, 4.0, 3.0, 2.0]);
    assert_eq!(tensor.repartition(1).unwrap().data(), [1.0, 4.0, 3.0, 2.0]);
    assert_eq!(tensor.repartition(3).unwrap().data(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        tensor.repartition(0).unwrap().data(),
        tensor.permute(&[], &[2, 1, 0]).unwrap().data()
    );
}

// ---------------------------------------------------------------------------
// Phase 5: decompositions (issue #567).
//
// Factorizations are checked through their mathematical contracts.
// ---------------------------------------------------------------------------

fn typed_z2_spectrum(
    spectrum: &[tenet::typed::SectorSpectrum<tenet::core::Z2Irrep>],
) -> Vec<(tenet::core::Z2Irrep, Vec<f64>)> {
    spectrum
        .iter()
        .map(|entry| (entry.sector, entry.values.clone()))
        .collect()
}

/// `u * s * vh` through the typed `contract`, for a `[2] <- [1]` factor chain:
/// `u`'s last axis is its bond, `s` is `bond <- bond`, `vh` is `bond <- rest`.
fn recompose(
    u: &TensorMap<tenet::core::Z2FusionRule, f64>,
    s: &TensorMap<tenet::core::Z2FusionRule, f64>,
    vh: &TensorMap<tenet::core::Z2FusionRule, f64>,
) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    let us = u.contract(s, &[2], &[0], &[0, 1, 2]).unwrap();
    us.contract(vh, &[2], &[0], &[0, 1, 2]).unwrap()
}

#[test]
fn svd_compact_reconstructs_the_source_through_the_typed_contract() {
    // What: the factors really are a factorization in this facade's own
    // vocabulary. There is no typed `compose`, so the composition runs through
    // `contract` — bosonic here, where the two agree.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let (u, s, vh) = typed.svd_compact().unwrap();
    let recon = recompose(&u, &s, &vh);

    assert_eq!(recon.data().len(), typed.data().len());
    for (got, want) in recon.data().iter().zip(typed.data()) {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}

#[test]
fn svd_compact_reconstructs_a_complex_payload() {
    // What: c64 takes the complex factorization route. The imaginary part is
    // deliberately not proportional to the real one, so a stray conjugation
    // or a real-only path is visible.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);
    let (tu, ts, tvh) = typed.svd_compact().unwrap();

    let recon = tu
        .contract(&ts, &[2], &[0], &[0, 1, 2])
        .unwrap()
        .contract(&tvh, &[2], &[0], &[0, 1, 2])
        .unwrap();
    for (got, want) in recon.data().iter().zip(typed.data()) {
        assert!(
            (got - want).norm() <= 1e-12 * want.norm().max(1.0),
            "{got} vs {want}"
        );
    }
}

#[test]
fn svd_full_reconstructs_with_unitary_outer_factors() {
    let _guard = cache_lock();
    let runtime = runtime();
    for num_codomain in [2, 1] {
        let typed = z2_tensor_split(&runtime, num_codomain);
        let (u, s, vh) = typed.svd_full().unwrap();

        let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert_data_close_f64(recon.data(), typed.data());
        assert!(u.is_isometric(1e-12).unwrap());
        assert!(vh.adjoint().unwrap().is_isometric(1e-12).unwrap());
        assert_same_legs(&u.codomain(), &typed.codomain());
        assert_same_legs(&vh.domain(), &typed.domain());
        assert_same_legs(&u.domain(), &s.codomain());
        assert_same_legs(&s.domain(), &vh.codomain());
    }
}

#[test]
fn svd_trunc_reconstructs_and_reports_the_discarded_weight() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let truncation = tenet::typed::Truncation::rank(2);
    let typed_out = typed.svd_trunc(&truncation).unwrap();

    assert_same_legs(&typed_out.u.codomain(), &typed.codomain());
    assert_same_legs(&typed_out.vh.domain(), &typed.domain());
    assert_same_legs(&typed_out.u.domain(), &typed_out.s.codomain());
    assert_same_legs(&typed_out.s.domain(), &typed_out.vh.codomain());

    // The reported error is the 2-norm of everything the truncation dropped.
    // Z2 is a group, so every quantum dimension is one and the weighting is
    // the identity — the check is then a plain sum of squares.
    let full = typed.svd_vals().unwrap();
    let kept = typed_z2_spectrum(&typed_out.singular_values);
    let mut discarded = 0.0;
    for entry in &full {
        let kept_here = kept
            .iter()
            .find(|(sector, _)| sector == &entry.sector)
            .map_or(0, |(_, values)| values.len());
        for value in &entry.values[kept_here..] {
            discarded += value * value;
        }
    }
    assert!(discarded > 0.0, "the fixture must actually truncate");
    assert!((typed_out.error - discarded.sqrt()).abs() < 1e-12);

    let recon = recompose(&typed_out.u, &typed_out.s, &typed_out.vh);
    let reconstruction_error = recon
        .data()
        .iter()
        .zip(typed.data())
        .map(|(got, want)| (got - want) * (got - want))
        .sum::<f64>()
        .sqrt();
    assert!((typed_out.error - reconstruction_error).abs() < 1e-12);

    // A degenerate but well-formed policy is a policy, not an error: keeping
    // nothing succeeds and discards the whole spectrum.
    let empty = typed
        .svd_trunc(&tenet::typed::Truncation::Rank(0))
        .expect("Rank(0) is degenerate, not malformed");
    assert!(empty.s.data().is_empty());
    assert!(empty
        .singular_values
        .iter()
        .all(|entry| entry.values.is_empty()));
}

#[test]
fn a_spectrum_decode_failure_comes_back_as_the_codec_error() {
    // What: a codec that cannot decode a coupled sector the engine produced
    // fails the call with the provider's own error instead of panicking inside
    // the label map.
    //
    // This is the Err path worth testing because a degenerate `Truncation` is
    // not one: `Rank(0)`, `Rank(usize::MAX)` and `All(vec![])` are all
    // constructible and all legitimately succeed (see the `Rank(0)` assertion
    // in the truncation test), and the states that would fail validation are
    // unreachable from outside `tenet-matrixalgebra` — their variants are
    // `#[non_exhaustive]` and their constructors are fallible. So the provider
    // is the only input to these methods a caller can actually malform.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::with(Quirk::NarrowDecode));
    let leg =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 2)])
            .unwrap();
    // Two charge-1 codomain legs couple to charge 2, the id this codec refuses.
    // `zeros` never decodes, so the tensor builds and the failure lands in the
    // spectrum decode.
    let tensor = TensorMap::<ExternalZ3, f64>::zeros(&runtime, [&leg, &leg], [&leg, &leg]).unwrap();

    assert!(matches!(
        tensor.svd_vals().unwrap_err(),
        tenet::prelude::Error::FusionAlgebra(_)
    ));
    assert!(matches!(
        tensor
            .svd_trunc(&tenet::typed::Truncation::Full)
            .unwrap_err(),
        tenet::prelude::Error::FusionAlgebra(_)
    ));
}

#[test]
fn svd_vals_reports_exact_per_label_spectra() {
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 3),
            (tenet::core::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    // Each block is diagonal, so its singular values are the absolute diagonal
    // entries. Both sectors are deliberately signed and out of magnitude order.
    let typed = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, indices| {
        if indices[0] != indices[1] {
            return 0.0;
        }
        if *trees.coupled() == tenet::core::Z2Irrep::EVEN {
            [-2.0, 5.0, -1.0][indices[0]]
        } else {
            assert_eq!(*trees.coupled(), tenet::core::Z2Irrep::ODD);
            [3.0, -4.0][indices[0]]
        }
    })
    .unwrap();

    let spectrum = typed.svd_vals().unwrap();
    assert_eq!(
        typed_z2_spectrum(&spectrum),
        [
            (tenet::core::Z2Irrep::EVEN, vec![5.0, 2.0, 1.0]),
            (tenet::core::Z2Irrep::ODD, vec![4.0, 3.0]),
        ]
    );
    // `Z2Irrep` orders exactly as its ids do, so this fixture cannot tell a
    // label sort from an id sort. The next test does.
}

#[test]
fn svd_vals_sorts_by_label_where_that_differs_from_the_id_order() {
    // What: the O2' promise is *label* order, and the only way to see it is a
    // provider whose codec does not order its labels the way the engine orders
    // its ids. `Quirk::ReversedLabels` is exactly that — a valid codec, not a
    // broken one — so the seam hands the spectrum back in the reversed order
    // and only the facade's own sort puts it right.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::with(Quirk::ReversedLabels));
    let tensor = z3_rank_four(&runtime, &provider);

    let labels: Vec<u8> = tensor
        .svd_vals()
        .unwrap()
        .iter()
        .map(|entry| entry.sector.0)
        .collect();

    assert_eq!(labels, [0, 1, 2]);
}

#[test]
fn qr_and_lq_reconstruct_with_the_expected_isometries_and_spaces() {
    // Both splits are exercised deliberately: `2 <- 1` is tall in every coupled
    // sector, where LQ-compact and LQ-full return the same factors and so
    // cannot tell the two seams apart; `1 <- 2` is wide, where they differ (and
    // symmetrically for QR).
    let _guard = cache_lock();
    let runtime = runtime();
    for num_codomain in [2, 1] {
        let typed = z2_tensor_split(&runtime, num_codomain);

        let qr_compact = typed.qr_compact().unwrap();
        let qr_full = typed.qr_full().unwrap();
        for (q, r) in [&qr_compact, &qr_full] {
            assert_data_close_f64(q.compose(r).unwrap().data(), typed.data());
            assert!(q.is_isometric(1e-12).unwrap());
            assert_same_legs(&q.codomain(), &typed.codomain());
            assert_same_legs(&r.domain(), &typed.domain());
            assert_same_legs(&q.domain(), &r.codomain());
        }

        let lq_compact = typed.lq_compact().unwrap();
        let lq_full = typed.lq_full().unwrap();
        for (l, q) in [&lq_compact, &lq_full] {
            assert_data_close_f64(l.compose(q).unwrap().data(), typed.data());
            assert!(q.adjoint().unwrap().is_isometric(1e-12).unwrap());
            assert_same_legs(&l.codomain(), &typed.codomain());
            assert_same_legs(&q.domain(), &typed.domain());
            assert_same_legs(&l.domain(), &q.codomain());
        }

        // Each orientation distinguishes one compact seam from its full
        // sibling through the internal bond space.
        if num_codomain == 2 {
            assert_ne!(
                typed_leg_shapes(&qr_compact.0),
                typed_leg_shapes(&qr_full.0)
            );
        } else {
            assert_ne!(
                typed_leg_shapes(&lq_compact.1),
                typed_leg_shapes(&lq_full.1)
            );
        }
    }
}

#[test]
fn left_and_right_orth_are_the_tensorkit_default_kinds() {
    // TensorKit 0.17 defaults `left_orth` to `:qr` and `right_orth` to `:lq`.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let (v, c) = typed.left_orth().unwrap();
    let (q, r) = typed.qr_compact().unwrap();
    assert_eq!(v.data(), q.data());
    assert_eq!(c.data(), r.data());

    let (c, vh) = typed.right_orth().unwrap();
    let (l, q) = typed.lq_compact().unwrap();
    assert_eq!(c.data(), l.data());
    assert_eq!(vh.data(), q.data());
}

#[test]
fn left_and_right_null_spaces_annihilate_the_source() {
    let _guard = cache_lock();
    let runtime = runtime();
    for num_codomain in [2, 1] {
        let typed = z2_tensor_split(&runtime, num_codomain);
        let left = typed.left_null().unwrap();
        let right = typed.right_null().unwrap();

        assert_same_legs(&left.codomain(), &typed.codomain());
        assert_same_legs(&right.domain(), &typed.domain());
        assert!(left.is_isometric(1e-12).unwrap());
        assert!(right.adjoint().unwrap().is_isometric(1e-12).unwrap());
        assert!(left
            .adjoint()
            .unwrap()
            .compose(&typed)
            .unwrap()
            .data()
            .iter()
            .all(|value| value.abs() < 1e-12));
        assert!(typed
            .compose(&right.adjoint().unwrap())
            .unwrap()
            .data()
            .iter()
            .all(|value| value.abs() < 1e-12));

        let left_null_dimensions = left.domain()[0].degeneracies().to_vec();
        let right_null_dimensions = right.codomain()[0].degeneracies().to_vec();
        // The counting fixture has one rank-deficient sector: its tall form
        // leaves dimensions 11 and 10 on the left bond and one on the right;
        // repartitioning to the wide form exchanges those two nullities.
        if num_codomain == 2 {
            assert_eq!(left_null_dimensions, [11, 10]);
            assert_eq!(right_null_dimensions, [1]);
        } else {
            assert_eq!(left_null_dimensions, [1]);
            assert_eq!(right_null_dimensions, [11, 10]);
        }
    }
}

#[test]
fn decompositions_carry_a_fermionic_provider() {
    // The one provider this facade can host whose braiding is not symmetric.
    // Every quantum dimension is one for `FermionParity`, and the fusion-tree
    // storage of a block *is* its coupled-sector matricization, so the sum of
    // squared singular values is the sum of squared stored elements — a
    // hand-checkable identity that no gauge convention can move.
    let _guard = cache_lock();
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);

    let spectrum = tensor.svd_vals().unwrap();
    let from_spectrum: f64 = spectrum
        .iter()
        .flat_map(|entry| entry.values.iter())
        .map(|value| value * value)
        .sum();
    let from_data: f64 = tensor.data().iter().map(|value| value * value).sum();
    assert!((from_spectrum - from_data).abs() < 1e-12);

    // And the seam is reachable at all for this provider, in both directions.
    let (q, r) = tensor.qr_compact().unwrap();
    assert_eq!(q.codomain().len(), 2);
    assert_eq!(r.domain().len(), 1);
}

#[test]
fn decompositions_carry_an_external_provider_with_its_own_labels() {
    // A downstream provider drives the same surface, and its spectrum comes
    // back in its own labels rather than raw ids.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let spectrum = tensor.svd_vals().unwrap();
    assert!(!spectrum.is_empty());
    assert!(spectrum.iter().all(|entry| entry.sector.0 < 3));
    assert!(spectrum.windows(2).all(|w| w[0].sector < w[1].sector));

    let out = tensor.svd_trunc(&tenet::typed::Truncation::Full).unwrap();
    assert_eq!(out.error, 0.0);
    assert_eq!(
        out.singular_values
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        spectrum
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Phase 5 (issue #568), slice 1: `TensorMap::add` and `TensorMap::scale`.
// ---------------------------------------------------------------------------

#[test]
fn add_applies_each_real_coefficient_to_the_right_operand() {
    // What: `alpha * self + beta * other`, coefficient for coefficient. The two
    // coefficients are deliberately different and
    // neither is 1, so swapping them (or dropping one) moves the buffer.
    //
    // The second operand is a permute of the first: same space, same layout,
    // different values, and no second fixture.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let typed_other = typed.permute(&[1, 0], &[2]).unwrap();

    let typed_sum = typed.add(&typed_other, 2.0, -3.0).unwrap();

    let expected: Vec<_> = typed
        .data()
        .iter()
        .zip(typed_other.data())
        .map(|(&lhs, &rhs)| 2.0 * lhs - 3.0 * rhs)
        .collect();
    assert_eq!(typed_sum.data(), expected);
    // The asymmetry is real: the swapped combination is a different tensor.
    assert_ne!(
        typed.add(&typed_other, -3.0, 2.0).unwrap().data(),
        typed_sum.data()
    );
}

#[test]
fn add_carries_complex_coefficients() {
    // What: `D` is the coefficient type too, so the c64 instantiation carries
    // genuinely complex coefficients through the same method.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);
    let alpha = Complex64::new(0.5, -2.0);
    let beta = Complex64::new(-1.5, 0.25);

    let other = typed.permute(&[1, 0], &[2]).unwrap();
    let typed_sum = typed.add(&other, alpha, beta).unwrap();
    let expected: Vec<_> = typed
        .data()
        .iter()
        .zip(other.data())
        .map(|(&lhs, &rhs)| alpha * lhs + beta * rhs)
        .collect();

    assert_eq!(typed_sum.data(), expected);
}

#[test]
fn add_rejects_a_different_runtime_and_a_different_space() {
    // What: the two checks this facade makes itself, in order — the runtime
    // identity the expert layer never sees, then the space equality that makes
    // the element-wise combination meaningful.
    let _guard = cache_lock();
    let other_runtime = runtime();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let elsewhere = z2_tensor(&other_runtime);
    let other_split = z2_tensor_split(&runtime, 1);

    assert!(matches!(
        typed.add(&elsewhere, 1.0, 1.0).unwrap_err(),
        tenet::prelude::Error::RuntimeMismatch
    ));
    // Keep the established space-mismatch diagnostic.
    assert!(matches!(
        typed.add(&other_split, 1.0, 1.0).unwrap_err(),
        tenet::prelude::Error::InvalidArgument(message)
            if message == "tensors live on different spaces or block layouts"
    ));
}

#[test]
fn scale_multiplies_every_real_and_complex_entry() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let scaled = typed.scale(-2.5);
    for (&actual, &source) in scaled.data().iter().zip(typed.data()) {
        assert_eq!(actual, -2.5 * source);
    }

    let typed_c = z2_complex_tensor(&runtime);
    let factor = Complex64::new(0.25, 3.0);
    let scaled = typed_c.scale(factor);
    for (&actual, &source) in scaled.data().iter().zip(typed_c.data()) {
        assert_eq!(actual, factor * source);
    }
}

// ---------------------------------------------------------------------------
// Phase 5 (issue #568), slice 2: `norm`, `norm_inf`, `normalize`.
// ---------------------------------------------------------------------------

/// An SU(2) tensor over two legs, split into `num_codomain <- rest`.
///
/// Why this fixture exists at all, next to the Z2 one: SU(2) is the only
/// non-abelian rule here, so it is the only one whose coupled sectors have
/// `dim(c) != 1`. Z2 is abelian and takes `weighted_inner`'s `Unique` fast
/// path, where the quantum-dimension weights are all one and therefore
/// invisible — every dimension-weighted operation needs this fixture as well.
///
fn su2_tensor_split(
    runtime: &Runtime,
    num_codomain: usize,
) -> TensorMap<tenet::core::SU2FusionRule, f64> {
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let legs = [&leg, &leg];
    let (codomain, domain) = legs.split_at(num_codomain);
    let mut next = 0.0;
    TensorMap::from_block_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        |_, _| {
            next += 1.0;
            next
        },
    )
    .unwrap()
}

/// The endomorphism split of [`su2_tensor_split`]: `[v] <- [v]`, which is
/// what `tr` needs and what every other SU(2) assertion here happens to use.
fn su2_tensor(runtime: &Runtime) -> TensorMap<tenet::core::SU2FusionRule, f64> {
    su2_tensor_split(runtime, 1)
}

#[test]
fn normalize_divides_by_the_dimension_weighted_norm() {
    let _guard = cache_lock();
    let runtime = runtime();

    // SU(2): normalizing by a dimension-weighted norm is what a plain
    // Frobenius normalization would get wrong, and only a non-abelian fixture
    // can see it.
    let typed = su2_tensor(&runtime);
    let norm = typed.norm().unwrap();
    let unit = typed.normalize().unwrap();
    for (&actual, &source) in unit.data().iter().zip(typed.data()) {
        assert!((actual - source / norm).abs() <= 1e-12 * source.abs().max(1.0));
    }
    assert!((unit.norm().unwrap() - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Phase 5 (issue #568), slice 3: `inner`, `dot`, `tr`.
// ---------------------------------------------------------------------------

/// A Z2 endomorphism, `[v] <- [v]`: the abelian half of the `tr`
/// comparison, where every quantum dimension is one.
fn z2_endomorphism(runtime: &Runtime) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    TensorMap::from_block_fn(runtime, [&leg], [&leg], typed_fill_value).unwrap()
}

#[test]
fn inner_uses_the_same_dimension_weight_as_norm() {
    // What: `inner` is TensorKit's `dot(x, y)` — conjugate-linear in the first
    // argument and quantum-dimension weighted, and it comes back as a `D`.
    // SU(2) is what
    // exercises the weighted branch; Z2 alone would take the abelian fast path.
    let _guard = cache_lock();
    let runtime = runtime();

    let z2_typed = z2_tensor(&runtime);
    let su2_typed = su2_tensor(&runtime);
    let agree = |typed_value: f64, norm: f64| {
        // `<t, t>` is the squared norm, which is the identity that pins this
        // weighting to `norm`'s.
        assert!((typed_value - norm * norm).abs() < 1e-9 * norm * norm);
    };
    agree(z2_typed.inner(&z2_typed).unwrap(), z2_typed.norm().unwrap());
    agree(
        su2_typed.inner(&su2_typed).unwrap(),
        su2_typed.norm().unwrap(),
    );
}

#[test]
fn inner_conjugates_its_first_argument() {
    // What: the conjugation is on `self`, so for a complex payload
    // `<a, b> = conj(<b, a>)` and the two are genuinely different numbers.
    // A dropped conjugation makes both sides equal and this test fail.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);
    // The extra `i` is what makes the product genuinely complex: this fixture's
    // imaginary part is a function of its real one, so the plain permuted
    // partner happens to give a real inner product and could not see a phase.
    let imaginary = Complex64::new(0.0, 1.0);
    let other = typed.permute(&[1, 0], &[2]).unwrap().scale(imaginary);
    let value = typed.inner(&other).unwrap();
    assert_eq!(value, other.inner(&typed).unwrap().conj());
    assert_ne!(value, other.inner(&typed).unwrap());
    assert!(value.im.abs() > 1e-6, "the fixture must have a real phase");
}

#[test]
#[allow(deprecated)]
fn dot_is_inner() {
    // `dot` is an alias for `inner`; the two names must not drift apart.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);
    let other = typed.permute(&[1, 0], &[2]).unwrap();

    assert_eq!(typed.dot(&other).unwrap(), typed.inner(&other).unwrap());
}

#[test]
fn inner_rejects_a_different_runtime_and_a_different_space() {
    let _guard = cache_lock();
    let other_runtime = runtime();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let elsewhere = z2_tensor(&other_runtime);
    let other_split = z2_tensor_split(&runtime, 1);

    assert!(matches!(
        typed.inner(&elsewhere).unwrap_err(),
        tenet::prelude::Error::RuntimeMismatch
    ));
    assert!(matches!(
        typed.inner(&other_split).unwrap_err(),
        tenet::prelude::Error::InvalidArgument(message)
            if message == "tensors live on different spaces or block layouts"
    ));
}

#[test]
fn tr_uses_the_nonabelian_dimension_weight() {
    // What: TensorKit's positive trace `Σ_c dim(c) * tr(b_c)`. SU(2) separates
    // it from the unweighted diagonal sum.
    let _guard = cache_lock();
    let runtime = runtime();

    let typed = su2_tensor(&runtime);
    let trace = typed.tr().unwrap();
    // The unweighted diagonal sum of the same blocks, for contrast: `tr` is
    // not it, which is what a dropped `dim(c)` would make it.
    let unweighted: f64 = (0..typed.block_count())
        .map(|index| {
            let block = typed.block(index).unwrap();
            let size = block.shape()[0];
            (0..size)
                .map(|i| {
                    typed.data()[block.offset() + i * (block.strides()[0] + block.strides()[1])]
                })
                .sum::<f64>()
        })
        .sum();
    assert!(
        (trace - unweighted).abs() > 1.0,
        "the SU(2) fixture must separate the weighted trace from the unweighted one"
    );
}

#[test]
fn tr_requires_an_endomorphism() {
    // Keep the established endomorphism diagnostic.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    assert!(matches!(
        typed.tr().unwrap_err(),
        tenet::prelude::Error::InvalidArgument(message)
            if message == "tr() requires an endomorphism (domain == codomain)"
    ));
}

// ---------------------------------------------------------------------------
// Phase 5 (issue #568), slice 4: `TensorMap::adjoint`.
// ---------------------------------------------------------------------------

#[test]
fn adjoint_swaps_spaces_and_is_an_involution() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let adjoint = typed.adjoint().unwrap();
    assert_same_legs(&adjoint.codomain(), &typed.domain());
    assert_same_legs(&adjoint.domain(), &typed.codomain());

    let roundtrip = adjoint.adjoint().unwrap();
    assert_same_legs(&roundtrip.codomain(), &typed.codomain());
    assert_same_legs(&roundtrip.domain(), &typed.domain());
    assert_eq!(roundtrip.data(), typed.data());
}

#[test]
fn adjoint_conjugates_a_complex_payload() {
    // What: c64 entries come back conjugated, not merely transposed. Compared
    // as multisets, because the transpose moves entries around: the point is
    // that the *values* are the conjugated ones and not the original ones.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);

    let adjoint = typed.adjoint().unwrap();
    let sorted = |values: &mut Vec<Complex64>| {
        values.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    };
    let mut got: Vec<Complex64> = adjoint.data().to_vec();
    let mut conjugated: Vec<Complex64> = typed.data().iter().map(|v| v.conj()).collect();
    let mut plain: Vec<Complex64> = typed.data().to_vec();
    sorted(&mut got);
    sorted(&mut conjugated);
    sorted(&mut plain);
    assert_eq!(got, conjugated);
    assert_ne!(
        got, plain,
        "the fixture must have a non-zero imaginary part"
    );
}

#[test]
fn adjoint_carries_an_external_provider() {
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let adjoint = tensor.adjoint().unwrap();

    assert_eq!(adjoint.data().len(), tensor.data().len());
    assert_eq!(adjoint.adjoint().unwrap().data(), tensor.data());
    // A dagger preserves the dimension-weighted norm.
    assert!((adjoint.norm().unwrap() - tensor.norm().unwrap()).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Phase 5 (issue #568), slice 5: `TensorMap::trace_pairs`.
// ---------------------------------------------------------------------------

/// A fermionic endomorphism `[v] <- [v]` with distinct sector values.
fn fermionic_endo(runtime: &Runtime) -> TensorMap<tenet::core::FermionParityFusionRule, f64> {
    let leg = fermionic_leg();
    let mut next = 0.0;
    TensorMap::from_block_fn(runtime, [&leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap()
}

#[test]
fn trace_pairs_preserves_partial_trace_geometry() {
    // What: the full trace to a rank-0 tensor and a partial trace that leaves a
    // leg open. The partial case is the one
    // that exercises the output-axis derivation and the destination's
    // codomain rank; the full case is the degenerate one.
    let _guard = cache_lock();
    let runtime = runtime();

    let typed = z2_endomorphism(&runtime);
    let full = typed.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(full.data().len(), 1);
    assert_eq!(full.scalar().unwrap(), typed.tr().unwrap());

    // `[v, v] <- [v]`: tracing axis 1 against axis 2 leaves axis 0 open, so the
    // result is `[v] <- []` and the open axis keeps its side.
    let typed = z2_tensor(&runtime);
    let partial = typed.trace_pairs(&[(1, 2)]).unwrap();
    assert_eq!(partial.codomain().len(), 1);
    assert_eq!(partial.domain().len(), 0);
    assert!(partial.data().iter().any(|&value| value != 0.0));

    // The two cases above leave at most one survivor, and it is codomain-side,
    // so neither can see the order of `output_axes` nor the codomain-rank
    // filter that splits the destination. These two can.
    //
    // `[v] <- [v, v]`, tracing (0, 1): the survivor is axis 2, a domain-side
    // leg, so the destination is `[] <- [v]` — a dropped codomain-rank filter
    // would put it in the codomain instead.
    let typed = z2_tensor_split(&runtime, 1);
    let survivor = typed.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(survivor.codomain().len(), 0);
    assert_eq!(survivor.domain().len(), 1);
    assert!(survivor.data().iter().any(|&value| value != 0.0));

    // `[v, v] <- [v, v]`, tracing (0, 3): two survivors, axes 1 and 2, one on
    // each side — so their relative order in `output_axes` is observable, and
    // reversing it changes the bytes.
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed: TensorMap<tenet::core::Z2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], typed_fill_value).unwrap();
    let two_survivors = typed.trace_pairs(&[(0, 3)]).unwrap();
    assert_eq!(two_survivors.codomain().len(), 1);
    assert_eq!(two_survivors.domain().len(), 1);
    assert!(two_survivors.data().iter().any(|&value| value != 0.0));
}

#[test]
fn trace_pairs_of_nothing_is_the_source() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let traced = typed.trace_pairs(&[]).unwrap();

    assert_eq!(traced.data(), typed.data());
    assert_eq!(traced.codomain().len(), 2);
}

#[test]
fn trace_pairs_rejects_malformed_pairs() {
    // Out-of-range and repeated axes both use the public invalid-pair error.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    for pairs in [vec![(0usize, 9usize)], vec![(0, 0)], vec![(0, 1), (1, 0)]] {
        assert!(matches!(
            typed.trace_pairs(&pairs).unwrap_err(),
            tenet::prelude::Error::InvalidArgument(message)
                if message.contains("invalid trace pair list")
        ));
    }
}

#[test]
fn fermionic_trace_pairs_is_the_supertrace_and_tr_is_not() {
    // What: the documented divergence. `tr` is TensorKit's positive trace
    // (`Σ_c dim(c) tr(b_c)`), `trace_pairs` is the tensor-contraction trace,
    // which for a fermionic rule carries the twist — the supertrace. On this
    // fixture the odd sector contributes with opposite signs, so the two
    // numbers differ.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fermionic_endo(&runtime);

    let positive = typed.tr().unwrap();
    let super_trace = typed.trace_pairs(&[(0, 1)]).unwrap();

    assert_ne!(
        super_trace.data(),
        [positive],
        "the fermionic supertrace must not coincide with the positive trace"
    );
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #569), slice 1: `TensorMap::compose`.
// ---------------------------------------------------------------------------

/// A position-weighted fill whose distinct entries expose individual signs.
fn fermionic_typed_fill(
    sectors: &tenet::typed::BlockFusionTrees<tenet::core::Z2Irrep>,
    indices: &[usize],
) -> f64 {
    let mut value = f64::from(sectors.coupled().parity()) * 1000.0;
    for (position, label) in sectors.codomain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 100.0 * (position + 1) as f64;
    }
    for (position, label) in sectors.domain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 10.0 * (position + 1) as f64;
    }
    value + 1.0 + indices.iter().sum::<usize>() as f64
}

/// `a : [v] <- [v*, v]` and `b : [v*, v] <- [v]`.
///
/// Composition contracts two legs here, exactly one of which is **dual** —
/// and a dual leg is the only place a fermionic twist can act. The mixed pair
/// is deliberate: with every contracted leg dual, "twist the dual contracted
/// legs" and "twist all contracted legs" would be the same statement, and the
/// twist identity below would not be pinned to a leg set at all.
fn fermionic_compose_pair(
    runtime: &Runtime,
) -> (
    TensorMap<tenet::core::FermionParityFusionRule, f64>,
    TensorMap<tenet::core::FermionParityFusionRule, f64>,
) {
    let leg = fermionic_leg_with(&[1, 2]);
    let leg_dual = leg.try_dual().unwrap();
    let typed_a =
        TensorMap::from_block_fn(runtime, [&leg], [&leg_dual, &leg], fermionic_typed_fill).unwrap();
    let typed_b =
        TensorMap::from_block_fn(runtime, [&leg_dual, &leg], [&leg], fermionic_typed_fill).unwrap();
    (typed_a, typed_b)
}

/// [`fermionic_leg`] with explicit `[even, odd]` degeneracies.
fn fermionic_leg_with(
    degeneracies: &[usize; 2],
) -> GradedSpace<tenet::core::FermionParityFusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::FermionParityFusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, degeneracies[0]),
            (tenet::core::Z2Irrep::ODD, degeneracies[1]),
        ],
    )
    .expect("fermionic leg is well formed")
}

#[test]
fn fermionic_compose_is_contract_against_a_twisted_right_operand() {
    // What: the exact relation between the two contraction semantics —
    // `compose(a, b) == contract(a, twist(b, b's dual codomain legs))`. The
    // twisted operand is built through `from_block_fn` rather than through the
    // public `twist`: theta is -1 on an odd sector and +1 otherwise, so the
    // twisted tensor is a one-line independent oracle here.
    let _guard = cache_lock();
    let runtime = runtime();
    let (typed_a, typed_b) = fermionic_compose_pair(&runtime);

    let leg = fermionic_leg_with(&[1, 2]);
    let leg_dual = leg.try_dual().unwrap();
    let twisted_b =
        TensorMap::from_block_fn(&runtime, [&leg_dual, &leg], [&leg], |sectors, indices| {
            // Codomain leg 0 is the dual one; leg 1 is contracted too but is
            // not dual, so theta does not act on it.
            let theta = if sectors.codomain_uncoupled()[0] == tenet::core::Z2Irrep::ODD {
                -1.0
            } else {
                1.0
            };
            theta * fermionic_typed_fill(sectors, indices)
        })
        .unwrap();

    assert_eq!(
        typed_a.compose(&typed_b).unwrap().data(),
        typed_a
            .contract(&twisted_b, &[1, 2], &[0, 1], &[0, 1])
            .unwrap()
            .data()
    );
}

#[test]
fn fermionic_compose_and_contract_disagree() {
    // What: the sign is real. If `compose` were routed through the ordinary
    // contraction the two would coincide and this assertion would fail.
    let _guard = cache_lock();
    let runtime = runtime();
    let (typed_a, typed_b) = fermionic_compose_pair(&runtime);

    assert_ne!(
        typed_a.compose(&typed_b).unwrap().data(),
        typed_a
            .contract(&typed_b, &[1, 2], &[0, 1], &[0, 1])
            .unwrap()
            .data()
    );
}

#[test]
fn bosonic_compose_is_contract_with_the_identity_output_order() {
    // What: for a symmetric braiding the supertrace twist is the identity, so
    // the two semantics coincide.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    let composed = typed.compose(&typed).unwrap();

    assert_eq!(
        composed.data(),
        typed.contract(&typed, &[1], &[0], &[0, 1]).unwrap().data()
    );
}

#[test]
fn compose_contracts_the_whole_domain_against_the_whole_codomain() {
    // What: the derived axes. `[v, v] <- [v]` composed with `[v] <- [v, v]`
    // contracts exactly the one shared leg and leaves `[v, v] <- [v, v]`, and
    // the same tensor comes back from the explicit contraction. Perturbing
    // either axis derivation changes the shape or the numbers here.
    let _guard = cache_lock();
    let runtime = runtime();
    let tall = z2_tensor_split(&runtime, 2);
    let wide = z2_tensor_split(&runtime, 1);

    let composed = tall.compose(&wide).unwrap();

    assert_eq!(composed.codomain().len(), 2);
    assert_eq!(composed.domain().len(), 2);
    assert_eq!(
        composed.data(),
        tall.contract(&wide, &[2], &[0], &[0, 1, 2, 3])
            .unwrap()
            .data()
    );
}

#[test]
fn compose_rejects_operands_from_different_runtimes() {
    let _guard = cache_lock();
    let left = z2_endomorphism(&runtime());
    let right = z2_endomorphism(&runtime());

    assert!(matches!(
        left.compose(&right).unwrap_err(),
        tenet::prelude::Error::RuntimeMismatch
    ));
}

#[test]
fn compose_rejects_operands_whose_domain_and_codomain_do_not_meet() {
    // A rank mismatch and a matching-rank but non-dual leg mismatch, both from
    // the expert layer rather than from a pre-check here.
    let _guard = cache_lock();
    let runtime = runtime();
    let endo = z2_endomorphism(&runtime);
    let tall = z2_tensor_split(&runtime, 2);
    let wide = z2_tensor_split(&runtime, 1);

    // Rank mismatch in both directions: one domain leg against two codomain
    // legs, and two domain legs against one codomain leg.
    assert!(endo.compose(&tall).is_err());
    assert!(wide.compose(&endo).is_err());

    // Matching ranks, mismatched legs: the degeneracies differ, so the two do
    // not meet even though the shapes line up.
    let z2 = Arc::new(tenet::core::Z2FusionRule);
    let narrow_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&z2),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let narrow =
        TensorMap::from_block_fn(&runtime, [&narrow_leg], [&narrow_leg], typed_fill_value).unwrap();
    assert!(endo
        .compose(&narrow)
        .unwrap_err()
        .to_string()
        .contains("leg degeneracy mismatch"));

    // Matching ranks and matching degeneracies, opposite dual flags.
    let wide_leg = GradedSpace::try_new_with_arc(
        Arc::clone(&z2),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let dual_endo =
        TensorMap::from_block_fn(&runtime, [&wide_leg], [&wide_leg], typed_fill_value).unwrap();
    assert!(endo
        .compose(&dual_endo)
        .unwrap_err()
        .to_string()
        .contains("contracted fusion leg duality flags do not match"));

    // Matching ranks and flags, different sector content.
    let even_only =
        GradedSpace::try_new_with_arc(Arc::clone(&z2), [(tenet::core::Z2Irrep::EVEN, 2)]).unwrap();
    let even_endo =
        TensorMap::from_block_fn(&runtime, [&even_only], [&even_only], typed_fill_value).unwrap();
    assert!(endo
        .compose(&even_endo)
        .unwrap_err()
        .to_string()
        .contains("dimension mismatch"));
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #569), slice 2: `TensorMap::id`.
// ---------------------------------------------------------------------------

#[test]
fn id_writes_the_nonuniform_fused_diagonal() {
    // What: the identity on a leg list whose per-sector degeneracies differ
    // from leg to leg — the case where the
    // diagonal offsets inside a coupled-sector block are not all the same.
    let _guard = cache_lock();
    let runtime = runtime();
    let z2 = Arc::new(tenet::core::Z2FusionRule);
    let typed_leg = |even, odd| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&z2),
            [
                (tenet::core::Z2Irrep::EVEN, even),
                (tenet::core::Z2Irrep::ODD, odd),
            ],
        )
        .unwrap()
    };

    let wide = typed_leg(2, 3);
    let narrow = typed_leg(1, 4);
    let typed = TensorMap::<_, f64>::id(&runtime, [&wide, &narrow]).unwrap();

    // Not the zero tensor, and not the all-ones one either: a genuine diagonal.
    // 14 even + 11 odd fused states: `2*1 + 3*4` and `2*4 + 3*1`.
    assert_eq!(typed.data().iter().filter(|&&v| v == 1.0).count(), 25);
    assert!(typed.data().contains(&0.0));
}

#[test]
fn id_composes_as_the_identity_on_both_sides() {
    // What: the defining property, as a byte oracle against the source tensor.
    // `[v, v] <- [v]`, so the two sides exercise different ranks.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let left = TensorMap::id(&runtime, &typed.codomain()).unwrap();
    let right = TensorMap::id(&runtime, &typed.domain()).unwrap();

    assert_eq!(left.compose(&typed).unwrap().data(), typed.data());
    assert_eq!(typed.compose(&right).unwrap().data(), typed.data());
}

#[test]
fn id_composes_as_the_identity_for_a_fermionic_provider() {
    // What: no stray sign. A fermionic identity is still the plain diagonal —
    // the twist question belongs to the contracted legs, and composition does
    // not ask it.
    let _guard = cache_lock();
    let runtime = runtime();
    let (typed_a, _) = fermionic_compose_pair(&runtime);

    let left = TensorMap::id(&runtime, &typed_a.codomain()).unwrap();
    let right = TensorMap::id(&runtime, &typed_a.domain()).unwrap();

    assert_eq!(left.compose(&typed_a).unwrap().data(), typed_a.data());
    assert_eq!(typed_a.compose(&right).unwrap().data(), typed_a.data());
}

#[test]
fn id_needs_at_least_one_leg() {
    // The provider is inferred from the legs, exactly as for `zeros`.
    let _guard = cache_lock();
    let runtime = runtime();
    let legs: [&GradedSpace<tenet::core::Z2FusionRule>; 0] = [];

    assert!(matches!(
        TensorMap::<_, f64>::id(&runtime, legs).unwrap_err(),
        tenet::prelude::Error::InvalidArgument(message)
            if message.contains("at least one leg")
    ));
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #570), slice 1: compact diagonal storage and the operations
// that consume it.
//
// Storage claims are measured in `tests/typed_diagonal_allocations.rs`; the
// value gates here compare compact routes with independent dense or pointwise
// oracles.
// ---------------------------------------------------------------------------

#[test]
fn compact_reductions_match_the_forced_dense_route() {
    // What: `norm`, `norm_inf`, `tr` and `inner` read the stored spectrum
    // instead of its materialization, and land on the same numbers.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_bond(&runtime);
    let dense = forced_dense(&typed);

    assert_eq!(typed.norm().unwrap(), dense.norm().unwrap());
    assert_eq!(typed.norm_inf().unwrap(), dense.norm_inf().unwrap());
    assert_eq!(typed.tr().unwrap(), dense.tr().unwrap());
    assert_eq!(typed.inner(&typed).unwrap(), dense.inner(&dense).unwrap());
    // `<s, s>` is the squared norm: the identity that pins the weighting.
    let norm = typed.norm().unwrap();
    assert!((typed.inner(&typed).unwrap() - norm * norm).abs() < 1e-9 * norm * norm);
}

#[test]
fn compact_reductions_carry_the_su2_dimension_weight() {
    // What: the compact reductions apply `dim(c)` exactly where the dense ones
    // do. Z2 alone cannot see this — every `dim(c)` is one there.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = su2_tensor(&runtime);
    let typed = typed.svd_compact().unwrap().1;
    let dense = forced_dense(&typed);

    assert_eq!(typed.norm().unwrap(), dense.norm().unwrap());
    assert_eq!(typed.tr().unwrap(), dense.tr().unwrap());
    assert_eq!(typed.inner(&typed).unwrap(), dense.inner(&dense).unwrap());
    // The unweighted sum, for contrast: a dropped `dim(c)` would make `tr` this.
    let unweighted: f64 = typed.data().iter().sum();
    assert!(
        (typed.tr().unwrap() - unweighted).abs() > 1e-6,
        "the SU(2) spectrum trace is not dimension weighted"
    );
    // `norm_inf` is deliberately *not* weighted.
    assert_eq!(typed.norm_inf().unwrap(), dense.norm_inf().unwrap());
}

#[test]
fn compact_add_matches_pointwise_values_on_both_arms() {
    // What: diagonal + diagonal stays diagonal, and diagonal + dense goes dense
    // — and both arms obey the pointwise linear-combination law.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_bond(&runtime);

    let diagonal_sum = typed.add(&typed, 0.75, -0.5).unwrap();
    for (&actual, &source) in diagonal_sum.data().iter().zip(typed.data()) {
        let expected = 0.25 * source;
        assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
    }

    // A dense operand on the same bond space: `id` is the cheapest one, and it
    // is not diagonal *storage* even though its values are diagonal.
    let typed_dense = TensorMap::id(&runtime, &typed.domain()).unwrap().scale(3.0);

    for (alpha, beta) in [(0.75, -0.5), (1.0, 1.0)] {
        let diagonal_dense = typed.add(&typed_dense, alpha, beta).unwrap();
        let dense_diagonal = typed_dense.add(&typed, alpha, beta).unwrap();
        for (((&diagonal_dense, &dense_diagonal), &diagonal), &dense) in diagonal_dense
            .data()
            .iter()
            .zip(dense_diagonal.data())
            .zip(typed.data())
            .zip(typed_dense.data())
        {
            let expected = alpha * diagonal + beta * dense;
            assert!((diagonal_dense - expected).abs() <= 1e-12 * expected.abs().max(1.0));
            let expected = alpha * dense + beta * diagonal;
            assert!((dense_diagonal - expected).abs() <= 1e-12 * expected.abs().max(1.0));
        }
    }
}

#[test]
fn compose_takes_the_compact_paths_and_reconstructs_the_source() {
    // What: `u * s * vh` now runs through the bond-scaling arms rather than a
    // dense GEMM, and still reproduces the source — plus `s * s`, the
    // diagonal-times-diagonal arm.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let (tu, ts, tvh) = typed.svd_compact().unwrap();

    let tus = tu.compose(&ts).unwrap();
    let svh = ts.compose(&tvh).unwrap();
    let squared = ts.compose(&ts).unwrap();
    // `s * s` is still a spectrum: its entries are the squares.
    for (squared, original) in squared.data().iter().zip(ts.data()) {
        assert!((squared - original * original).abs() < 1e-12);
    }

    // Both associations reconstruct, separately pinning `t * D` and `D * t`.
    for recon in [tus.compose(&tvh).unwrap(), tu.compose(&svh).unwrap()] {
        assert_data_close_f64(recon.data(), typed.data());
    }
}

#[test]
fn compose_declines_a_compact_arm_it_cannot_prove() {
    // What: the compact arms fire on a proved destination, not on the storage
    // alone. Two spectra on different bond spaces are not composable at all,
    // and the guard is what leaves that verdict to the expert layer instead of
    // multiplying two unrelated spectra elementwise.
    let _guard = cache_lock();
    let runtime = runtime();
    let wide = z2_tensor_split(&runtime, 2);
    // A second endomorphism on a leg with different degeneracies, so its bond
    // space genuinely differs from `wide`'s rather than merely being a second
    // allocation of the same one.
    let narrow_leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let mut next = 0.0;
    let narrow = TensorMap::from_block_fn(&runtime, [&narrow_leg], [&narrow_leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap();
    let wide_s = wide.svd_compact().unwrap().1;
    let narrow_s = narrow.svd_compact().unwrap().1;

    assert_ne!(
        wide_s.data().len(),
        narrow_s.data().len(),
        "the fixture's two bond spaces must differ for this to test anything"
    );
    assert!(
        wide_s.compose(&narrow_s).is_err(),
        "composing spectra on mismatched bond spaces must be refused"
    );
    // And the same for a dense operand whose contracted leg does not match the
    // spectrum's bond.
    assert!(wide.compose(&narrow_s).is_err());
    assert!(narrow_s.compose(&wide).is_err());
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #570), slice 2: the Hermitian eigendecompositions.
// ---------------------------------------------------------------------------

/// A typed Hermitian endomorphism: `p = t + t†`, so `eigh` is defined
/// without projecting first.
fn z2_hermitian(runtime: &Runtime) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    let typed = z2_endomorphism(runtime);
    typed.add(&typed.adjoint().unwrap(), 1.0, 1.0).unwrap()
}

#[test]
fn eigh_full_reconstructs_the_source_through_compose() {
    // What: `v * d * v†` is the source. The middle composition takes the
    // compact bond-scaling arm, so this exercises the storage as well as the
    // factorization.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_hermitian(&runtime);

    let (d, v) = typed.eigh_full().unwrap();
    let recon = v
        .compose(&d)
        .unwrap()
        .compose(&v.adjoint().unwrap())
        .unwrap();

    assert_eq!(recon.data().len(), typed.data().len());
    for (got, want) in recon.data().iter().zip(typed.data()) {
        assert!(
            (got - want).abs() <= 1e-10 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}

#[test]
fn eigh_vals_follow_the_provider_label_order() {
    // What: caller-facing spectra use provider labels in canonical order.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_hermitian(&runtime);
    let spectrum = typed.eigh_vals().unwrap();

    assert_eq!(
        spectrum
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        [tenet::core::Z2Irrep::EVEN, tenet::core::Z2Irrep::ODD]
    );
    assert!(spectrum.iter().all(|entry| !entry.values.is_empty()));
}

#[test]
fn eigh_trunc_reports_the_discarded_eigenvalue_norm() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_hermitian(&runtime);
    let truncation = Truncation::rank(3);

    let mut magnitudes: Vec<_> = typed
        .eigh_vals()
        .unwrap()
        .into_iter()
        .flat_map(|entry| entry.values)
        .map(f64::abs)
        .collect();
    magnitudes.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
    let expected_error = magnitudes[3..]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let typed_out = typed.eigh_trunc(&truncation).unwrap();

    assert_eq!(
        typed_out
            .eigenvalues
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        3
    );
    assert!(
        expected_error > 0.0,
        "the fixture must discard nonzero values"
    );
    assert!((typed_out.error - expected_error).abs() < 1e-12 * expected_error.max(1.0));
    assert!(typed_out.d.data().len() < typed.eigh_full().unwrap().0.data().len());
}

#[test]
fn eigh_reports_a_non_hermitian_input_rather_than_a_wrong_answer() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    assert!(typed.eigh_full().is_err());
    assert!(typed.eigh_vals().is_err());
    assert!(typed.eigh_trunc(&Truncation::Full).is_err());
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #570), slice 3: the general eigendecompositions.
// ---------------------------------------------------------------------------

/// The typed c64 endomorphism `eig` needs: [`z2_complex_tensor`] is rank
/// three, and `eig` is defined on square maps only.
fn z2_complex_endo(runtime: &Runtime) -> TensorMap<tenet::core::Z2FusionRule, Complex64> {
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    TensorMap::from_block_fn(runtime, [&leg], [&leg], |trees, indices| {
        complex(typed_fill_value(trees, indices))
    })
    .unwrap()
}

#[test]
fn eig_full_satisfies_the_eigen_equation_for_a_real_payload() {
    // What: a real input promotes to complex factors satisfying A V = V D.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    let (d, v) = typed.eig_full().unwrap();
    let av = typed.to_c64().compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();

    assert_data_close_c64(av.data(), vd.data());
}

#[test]
fn complex_eig_satisfies_the_eigen_equation_and_conjugates_its_spectrum() {
    // What: the native complex route satisfies A V = V D, and the compact
    // spectrum's adjoint conjugates genuinely nonreal eigenvalues.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_endo(&runtime);

    let (d, v) = typed.eig_full().unwrap();
    assert_data_close_c64(
        typed.compose(&v).unwrap().data(),
        v.compose(&d).unwrap().data(),
    );

    // Genuinely complex, so a missing conjugation is observable.
    assert!(
        d.data().iter().any(|value| value.im.abs() > 1e-6),
        "the eig spectrum must be off the real axis for this to test anything"
    );
    let adjoint = d.adjoint().unwrap();
    for (conjugated, original) in adjoint.data().iter().zip(d.data()) {
        assert_eq!(*conjugated, original.conj());
    }
}

#[test]
fn eig_vals_are_label_ordered_and_trunc_reports_the_discarded_norm() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let spectrum = typed.eig_vals().unwrap();
    assert_eq!(
        spectrum
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        [tenet::core::Z2Irrep::EVEN, tenet::core::Z2Irrep::ODD]
    );

    let truncation = Truncation::rank(3);
    let mut magnitudes: Vec<_> = spectrum
        .into_iter()
        .flat_map(|entry| entry.values)
        .map(|value| value.norm())
        .collect();
    magnitudes.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
    let expected_error = magnitudes[3..]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let typed_out = typed.eig_trunc(&truncation).unwrap();

    assert_eq!(
        typed_out
            .eigenvalues
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        3
    );
    assert!(
        expected_error > 0.0,
        "the fixture must discard nonzero values"
    );
    assert!((typed_out.error - expected_error).abs() < 1e-12 * expected_error.max(1.0));
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #570), slice 4: the `is_hermitian` / `project_*` family.
// ---------------------------------------------------------------------------

#[test]
fn hermitian_projections_satisfy_their_identities_and_predicate_truth_table() {
    // What: the complementary projections reconstruct the source and their
    // adjoint symmetries agree with the typed predicate family.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let tol = 1e-10;

    let hermitian = typed.project_hermitian().unwrap();
    let antihermitian = typed.project_antihermitian().unwrap();
    assert_data_close_f64(
        hermitian.add(&antihermitian, 1.0, 1.0).unwrap().data(),
        typed.data(),
    );
    assert_data_close_f64(hermitian.adjoint().unwrap().data(), hermitian.data());
    assert_data_close_f64(
        antihermitian.adjoint().unwrap().data(),
        antihermitian.scale(-1.0).data(),
    );
    assert_data_close_f64(
        hermitian.project_hermitian().unwrap().data(),
        hermitian.data(),
    );
    assert_data_close_f64(
        antihermitian.project_antihermitian().unwrap().data(),
        antihermitian.data(),
    );

    let cases = [
        (&typed, [false, false, false, false, false]),
        (&hermitian, [true, false, false, false, false]),
        (&antihermitian, [false, true, false, false, false]),
    ];
    for (tensor, expected) in cases {
        assert_eq!(
            [
                tensor.is_hermitian(tol).unwrap(),
                tensor.is_antihermitian(tol).unwrap(),
                tensor.is_isometric(tol).unwrap(),
                tensor.is_unitary(tol).unwrap(),
                tensor.is_posdef(tol).unwrap(),
            ],
            expected
        );
    }
}

#[test]
fn isometry_and_posdef_see_their_positive_cases() {
    // What: the two members the fixture above only ever answers `false` for.
    // `u` from an SVD is isometric by construction, and `t† t` is positive
    // definite when `t` has full rank.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let tol = 1e-9;

    let tu = typed.svd_compact().unwrap().0;
    assert!(tu.is_isometric(tol).unwrap());
    // Isometric but not unitary: `u` is tall here.
    assert!(!tu.is_unitary(tol).unwrap());

    // `2 * id` is Hermitian with every eigenvalue at 2: positive definite on
    // any provider, and the cheapest tensor that is.
    let typed_positive = TensorMap::id(&runtime, &typed.domain()).unwrap().scale(2.0);
    assert!(typed_positive.is_hermitian(tol).unwrap());
    assert!(typed_positive.is_posdef(tol).unwrap());
    // Hermitian but not positive definite: the same tensor negated.
    let negated = typed_positive.scale(-1.0);
    assert!(negated.is_hermitian(tol).unwrap());
    assert!(!negated.is_posdef(tol).unwrap());
    // Positive *semi*definite is `false`, not `true`: TensorKit's `isposdef` is
    // Cholesky-based and strict, and this facade's rustdoc promises the same.
    // A real diagonal endomorphism with one entry at exactly zero is the case
    // that separates `>` from `>=` — `eigh` on it returns that zero exactly, so
    // the comparison is not floating-point weather.
    let semidefinite_leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let semidefinite = TensorMap::from_block_fn(
        &runtime,
        [&semidefinite_leg],
        [&semidefinite_leg],
        |_, indices: &[usize]| {
            if indices[0] != indices[1] {
                0.0
            } else {
                // Row 0 of every block is the zero eigenvalue.
                indices[0] as f64
            }
        },
    )
    .unwrap();
    assert!(semidefinite.is_hermitian(0.0).unwrap());
    assert!(semidefinite
        .eigh_vals()
        .unwrap()
        .iter()
        .any(|entry| entry.values.contains(&0.0)));
    assert!(
        !semidefinite.is_posdef(0.0).unwrap(),
        "a positive semidefinite tensor must not be reported positive definite"
    );

    // This rank-deficient fixture's Gram matrix is Hermitian but not strictly
    // positive definite.
    let tgram = typed.adjoint().unwrap().compose(&typed).unwrap();
    assert!(tgram.is_hermitian(tol).unwrap());
    assert!(!tgram.is_posdef(tol).unwrap());
}

#[test]
fn a_non_endomorphism_is_never_hermitian_and_never_errors() {
    // What: predicates are total and false; projections have no endomorphism to
    // return and therefore reject the input.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    assert!(!typed.is_hermitian(1e-9).unwrap());
    assert!(!typed.is_antihermitian(1e-9).unwrap());
    assert!(!typed.is_posdef(1e-9).unwrap());
    assert!(typed.project_hermitian().is_err());
    assert!(typed.project_antihermitian().is_err());
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #576), slice 1: `inv`.
// ---------------------------------------------------------------------------

/// An endomorphism whose every coupled-sector block is nonsingular: the fill
/// used by [`z2_endomorphism`] is position-weighted and produces rank-one
/// blocks, so `inv` on it would be testing the singular path instead. Adding a
/// multiple of the identity is the cheapest fix.
fn z2_invertible(runtime: &Runtime) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    let typed = z2_endomorphism(runtime);
    let typed_id = TensorMap::id(runtime, &typed.domain()).unwrap();
    typed.add(&typed_id, 1.0, 100.0).unwrap()
}

#[test]
fn inv_is_a_two_sided_inverse() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_invertible(&runtime);

    let typed_inverse = typed.inv().unwrap();
    let expected = TensorMap::<_, f64>::id(&runtime, &typed.domain()).unwrap();
    for (name, identity) in [
        ("t * inv(t)", typed.compose(&typed_inverse).unwrap()),
        ("inv(t) * t", typed_inverse.compose(&typed).unwrap()),
    ] {
        let error = identity
            .data()
            .iter()
            .zip(expected.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(error < 1e-9, "{name} is not the identity: {error}");
    }
}

#[test]
fn inv_of_a_compact_spectrum_is_the_elementwise_reciprocal() {
    // What: the O(rank) arm. A spectrum's inverse is `1/s_i` on the stored
    // values.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s;
    // The fixture is rank deficient, so the full spectrum contains zeros that
    // `inv` must refuse; keep only the nonzero part.
    let typed_s = typed_s.svd_trunc(&Truncation::Rank(2)).unwrap().s;

    let inverse = typed_s.inv().unwrap();
    let source_spectrum = typed_s.diagonal_spectrum().unwrap().unwrap();
    let inverse_spectrum = inverse.diagonal_spectrum().unwrap().unwrap();
    assert_eq!(source_spectrum.len(), inverse_spectrum.len());
    for (source, image) in source_spectrum.iter().zip(&inverse_spectrum) {
        assert_eq!(source.sector, image.sector);
        for (&value, &reciprocal) in source.values.iter().zip(&image.values) {
            assert_eq!(reciprocal, value.recip());
        }
    }
    // `s * s^-1` is the identity on the bond.
    let product = typed_s.compose(&inverse).unwrap();
    let expected = TensorMap::<_, f64>::id(&runtime, &typed_s.domain()).unwrap();
    let error = product
        .data()
        .iter()
        .zip(expected.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-9, "s * inv(s) is not the identity: {error}");
}

#[test]
fn inv_reports_a_singular_input_as_a_typed_error() {
    // What: singular input is a `Result`, never a panic, and the two storages
    // report it through different variants because the two arms detect it in
    // different places — the compact one by inspecting the stored value, the
    // dense one inside the LAPACK solve. Both are pinned here, because both are
    // documented.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    // Compact: a spectrum scaled to exactly zero. Why not the tail of a
    // rank-deficient SVD: those singular values come back tiny but nonzero, and
    // the arm under test compares against exact zero, not a tolerance.
    let spectrum = typed.svd_trunc(&Truncation::Full).unwrap().s.scale(0.0);
    match spectrum.inv() {
        Err(tenet::typed::Error::InvalidArgument(message)) => {
            assert!(
                message.contains("singular"),
                "unexpected message: {message}"
            );
        }
        other => panic!("expected an InvalidArgument for a singular spectrum, got {other:?}"),
    }

    // Dense: an all-zero endomorphism.
    let zeros = typed.scale(0.0);
    match zeros.inv() {
        Err(tenet::typed::Error::Operation(_)) => {}
        other => panic!("expected an Operation error for a singular dense block, got {other:?}"),
    }
}

#[test]
fn inv_accepts_isomorphic_but_unequal_codomain_and_domain() {
    // What: TensorKit's `inv` asks for `codomain ≅ domain`, not `==`, and
    // returns `domain <- codomain`. The seam agrees: a rank-one codomain and a
    // rank-two domain with the same coupled-sector dimensions is accepted, and
    // the result carries the swapped spaces. This is a behavior pin — the
    // rustdoc states it, so a seam that tightened to equality must fail here.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::Z2FusionRule);
    let wide = GradedSpace::try_new_with_arc(
        provider.clone(),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    // `narrow ⊗ narrow` has coupled dimensions (even 2, odd 2) as well, so the
    // two sides are isomorphic while the hom spaces differ in rank.
    let narrow = GradedSpace::try_new_with_arc(
        provider,
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let mut next = 0.0;
    let tensor = TensorMap::from_block_fn(&runtime, [&wide], [&narrow, &narrow], |_, _| {
        next += 1.0;
        next * next
    })
    .unwrap();

    let inverse = tensor.inv().unwrap();
    assert_eq!(inverse.codomain().len(), 2);
    assert_eq!(inverse.domain().len(), 1);
    let identity = tensor.compose(&inverse).unwrap();
    let expected = TensorMap::<_, f64>::id(&runtime, [&wide]).unwrap();
    let error = identity
        .data()
        .iter()
        .zip(expected.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-9, "t * inv(t) is not the identity: {error}");
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #576), slice 2: `pinv`.
// ---------------------------------------------------------------------------

#[test]
fn pinv_satisfies_the_moore_penrose_identities() {
    // The fixture is deliberately rank deficient: a full-rank one would let a
    // broken cutoff pass as an ordinary inverse.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    // `rcond` is well above the fixture's numerically-zero singular values and
    // well below its real ones, so the cutoff drops exactly the null directions
    // — inverting those instead would amplify rounding into the millions of ulp.
    let pseudo = typed.pinv(1e-6).unwrap();
    let left_support = typed.compose(&pseudo).unwrap();
    let right_support = pseudo.compose(&typed).unwrap();
    let assert_close =
        |name: &str,
         actual: &TensorMap<tenet::core::Z2FusionRule, f64>,
         expected: &TensorMap<tenet::core::Z2FusionRule, f64>| {
            let error = actual
                .data()
                .iter()
                .zip(expected.data())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                error < 1e-9 * expected.norm().unwrap().max(1.0),
                "{name}: {error}"
            );
        };
    assert_close(
        "t t^+ t != t",
        &left_support.compose(&typed).unwrap(),
        &typed,
    );
    assert_close(
        "t^+ t t^+ != t^+",
        &right_support.compose(&pseudo).unwrap(),
        &pseudo,
    );
    assert_close(
        "(t t^+)† != t t^+",
        &left_support.adjoint().unwrap(),
        &left_support,
    );
    assert_close(
        "(t^+ t)† != t^+ t",
        &right_support.adjoint().unwrap(),
        &right_support,
    );
}

#[test]
fn pinv_of_a_compact_spectrum_is_the_elementwise_cutoff_reciprocal() {
    // What: the O(rank) arm — an elementwise cutoff and reciprocal, whose own
    // singular values are `|entry|`, so no SVD runs at all.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s;
    let source = typed_s.diagonal_spectrum().unwrap().unwrap();
    let sigma_max = source
        .iter()
        .flat_map(|entry| &entry.values)
        .map(|value| value.abs())
        .fold(0.0, f64::max);

    for rcond in [0.0, 1e-12, 1e-3] {
        let image = typed_s
            .pinv(rcond)
            .unwrap()
            .diagonal_spectrum()
            .unwrap()
            .unwrap();
        let cutoff = rcond * sigma_max;
        for (source, image) in source.iter().zip(&image) {
            assert_eq!(source.sector, image.sector);
            for (&value, &actual) in source.values.iter().zip(&image.values) {
                let expected = if value.abs() > cutoff {
                    value.recip()
                } else {
                    0.0
                };
                assert_eq!(actual, expected, "rcond {rcond}, value {value}");
            }
        }
    }
}

#[test]
fn pinv_cuts_a_singular_value_sitting_exactly_on_the_cutoff() {
    // What: the boundary. The comparison is `sigma > rcond * sigma_max`, so a
    // singular value at *exactly* the cutoff is discarded, not kept — on both
    // storages. This is the one bit of the cutoff policy a
    // mutation to `>=` would otherwise slip past.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [(tenet::core::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    // Diagonal with entries 4 and 1: sigma_max is 4, so rcond = 0.25 puts the
    // second singular value exactly on the cutoff. Both are powers of two, so
    // the product is exact and the comparison is not floating-point weather.
    let tensor = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices: &[usize]| {
        if indices[0] != indices[1] {
            0.0
        } else if indices[0] == 0 {
            4.0
        } else {
            1.0
        }
    })
    .unwrap();
    assert_eq!(0.25 * 4.0, 1.0, "the fixture's cutoff must be exact");

    let dense_pinv = tensor.pinv(0.25).unwrap();
    // Kept: 1/4 for the surviving value. Cut: an exact 0 where 1/1 would be.
    let mut kept: Vec<f64> = dense_pinv
        .data()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .collect();
    kept.sort_by(f64::total_cmp);
    assert_eq!(kept, vec![0.25], "the boundary singular value survived");

    // A discarded nonzero mode makes this the exact Moore-Penrose inverse of
    // the hard-thresholded effective-rank tensor, not of the original tensor.
    let thresholded = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices: &[usize]| {
        if indices[0] == 0 && indices[1] == 0 {
            4.0
        } else {
            0.0
        }
    })
    .unwrap();
    let triple = tensor
        .compose(&dense_pinv)
        .unwrap()
        .compose(&tensor)
        .unwrap();
    assert_eq!(triple.data(), thresholded.data());
    assert_ne!(triple.data(), tensor.data());

    // And on the compact arm.
    let spectrum = tensor.svd_trunc(&Truncation::Full).unwrap().s;
    let compact_pinv = spectrum.pinv(0.25).unwrap();
    let mut kept: Vec<f64> = compact_pinv
        .data()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .collect();
    kept.sort_by(f64::total_cmp);
    assert_eq!(
        kept,
        vec![0.25],
        "the boundary value survived the compact arm"
    );
}

#[test]
fn pinv_rejects_a_nonfinite_or_negative_rcond_before_any_work() {
    // What: `rcond` is validated at the facade, so a bad one never reaches the
    // SVD — and the compact arm validates it too, which a guard placed only on
    // the dense route would miss.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let spectrum = typed.svd_trunc(&Truncation::Full).unwrap().s;

    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                typed.pinv(rcond),
                Err(tenet::typed::Error::InvalidArgument(_))
            ),
            "dense pinv accepted rcond {rcond}"
        );
        assert!(
            matches!(
                spectrum.pinv(rcond),
                Err(tenet::typed::Error::InvalidArgument(_))
            ),
            "compact pinv accepted rcond {rcond}"
        );
    }
}

#[test]
fn pinv_uses_one_global_sigma_max_across_every_sector() {
    // What: the cutoff is relative to the largest singular value of the *whole*
    // tensor, not of each coupled sector — the deliberate divergence from
    // TensorKit's per-block `rtol`.
    //
    // The global maximum deliberately lives in the **second** sector. A fold
    // that only ever looks at the first sector reads `sigma_max = 1` here, which
    // puts the cutoff at 0.5 and keeps everything — so that weaker mutant fails
    // this test, as does the per-sector one, which cannot cut anything in a 1x1
    // sector at all. Both were run by hand against this fixture and both fail
    // it; with the maximum in the first sector, the first-sector-only mutant
    // survived, because there the two folds happen to agree.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    // Even sector (stored first): 1. Odd sector: 1024. Each sector is 1x1, so
    // per-sector sigma_max would be the entry itself and nothing could ever be
    // cut; and the global maximum is not in the sector a first-sector-only fold
    // would find.
    let tensor = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
        if *trees.coupled() == tenet::core::Z2Irrep::EVEN {
            1.0
        } else {
            1024.0
        }
    })
    .unwrap();

    let pseudo = tensor.pinv(0.5).unwrap();
    let mut kept: Vec<f64> = pseudo
        .data()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .collect();
    kept.sort_by(f64::total_cmp);
    assert_eq!(
        kept,
        vec![1.0 / 1024.0],
        "a per-sector cutoff kept the small sector"
    );
    // The compact arm's own `max|entry|` is global for the same reason.
    let spectrum = tensor.svd_trunc(&Truncation::Full).unwrap().s;
    let kept = spectrum
        .pinv(0.5)
        .unwrap()
        .data()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .count();
    assert_eq!(kept, 1, "the compact arm used a per-sector cutoff");
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #576), slice 3: `exp`.
// ---------------------------------------------------------------------------

#[test]
fn exp_of_the_identity_is_e_times_the_identity() {
    // `exp(id) = e * id` on every provider.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let identity = TensorMap::<_, f64>::id(&runtime, &typed.domain()).unwrap();

    let expected = identity.scale(std::f64::consts::E);
    let error = identity
        .exp()
        .unwrap()
        .data()
        .iter()
        .zip(expected.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-12, "exp(id) != e * id: {error}");
}

#[test]
fn exp_accepts_a_non_hermitian_endomorphism_and_inverts_under_negation() {
    // What: issue #577 closed the recorded divergence — TensorKit's `exp` is a
    // general per-block Pade approximant with no hermiticity gate, and so is
    // this one now. `exp(A) exp(-A) = id` pins the actual exponential.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    assert!(!typed.is_hermitian(1e-9).unwrap());
    let typed_exp = typed.exp().unwrap();

    // exp(A) exp(-A) = id, evaluated through the typed composition.
    let inverse = typed.scale(-1.0).exp().unwrap();
    let identity = TensorMap::<_, f64>::id(&runtime, &typed.domain()).unwrap();
    let residual = typed_exp
        .compose(&inverse)
        .unwrap()
        .data()
        .iter()
        .zip(identity.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(residual < 1e-11, "exp(A) exp(-A) != id: {residual}");
}

#[test]
fn exp_of_a_compact_spectrum_stays_compact_and_is_elementwise() {
    // What: `exp(s_i)` on the `Σ_c k_c` stored values, matching TensorKit's
    // `exp(::DiagonalTensorMap)` law.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    // Scaled down: the fixture's largest singular value is in the thousands and
    // `exp` of it overflows to infinity, which no comparison can separate from
    // a wrong infinity.
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s.scale(1e-3);

    let typed_exp = typed_s.exp().unwrap();
    // Every stored value is `exp` of the source's: the elementwise claim, read
    // off the materialized diagonal so it does not need a compact accessor.
    for (index, (source, image)) in typed_s.data().iter().zip(typed_exp.data()).enumerate() {
        let expected = if *source == 0.0 && !on_diagonal(&typed_s, index) {
            // Off-diagonal of the block-diagonal materialization: `exp` of a
            // diagonal is diagonal, so these stay zero rather than becoming 1.
            0.0
        } else {
            source.exp()
        };
        assert!(
            (image - expected).abs() < 1e-12,
            "entry {index}: {image} is not exp({source})"
        );
    }
}

/// Whether storage position `index` sits on a block's own diagonal. Used to
/// read a compact tensor's elementwise claim off its dense materialization.
fn on_diagonal<R, D>(tensor: &TensorMap<R, D>, index: usize) -> bool
where
    R: tenet::core::MultiplicityFreeRigidSymbols<Scalar = f64>
        + tenet::core::CheckedFusionAlgebra
        + tenet::typed::SectorCodec,
    D: tenet::prelude::TensorScalar,
{
    (0..tensor.block_count()).any(|block| {
        let block = tensor.block(block).unwrap();
        let shape = block.shape();
        (0..shape[0]).any(|row| {
            index == block.offset() + row * block.strides()[0] + row * block.strides()[1]
        })
    })
}

#[test]
fn exp_of_a_complex_compact_spectrum_takes_the_complex_elementwise_branch() {
    // What: the compact arm is TensorKit's `exp(::DiagonalTensorMap)`, which is
    // unconditionally elementwise — so a c64 spectrum with a nonreal entry, the
    // case the Hermitian dense arm would refuse, comes back as `exp` of that
    // entry. Storage therefore *does* change what `exp` accepts, exactly as it
    // does in TensorKit; the rustdoc says so and this is the pin.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [(tenet::core::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    let dense = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices: &[usize]| {
        if indices[0] == indices[1] {
            Complex64::new(0.0, indices[0] as f64)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
    .unwrap();
    // Dense storage of the very same matrix: since issue #577 it is accepted
    // too, through the general Pade arm — and because this particular matrix is
    // already diagonal, the two arms must agree entry for entry. Storage no
    // longer decides *whether* `exp` is defined, only how it is computed.
    assert!(!dense.is_hermitian(1e-9).unwrap());
    let dense_exponential = dense.exp().unwrap();
    for (index, (source, value)) in dense
        .data()
        .iter()
        .zip(dense_exponential.data())
        .enumerate()
    {
        let expected = if on_diagonal(&dense, index) {
            source.exp()
        } else {
            Complex64::new(0.0, 0.0)
        };
        assert!(
            (value - expected).norm() < 1e-12,
            "dense entry {index}: {value} is not exp({source})"
        );
    }

    // Compact storage of the same values: accepted, elementwise.
    let spectrum = dense.eig_full().unwrap().0.scale(Complex64::new(1.0, 0.0));
    let image = spectrum.exp().unwrap();
    for (index, (source, value)) in spectrum.data().iter().zip(image.data()).enumerate() {
        let expected = if *source == Complex64::new(0.0, 0.0) && !on_diagonal(&spectrum, index) {
            Complex64::new(0.0, 0.0)
        } else {
            source.exp()
        };
        assert!(
            (value - expected).norm() < 1e-12,
            "entry {index}: {value} is not exp({source})"
        );
    }
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #576), slice 4: `sqrt`.
// ---------------------------------------------------------------------------

#[test]
fn sqrt_squares_to_the_source_on_both_storages() {
    // What: the diagonal-bond idiom, `√S · √S = S`, on compact storage and on
    // dense storage of the same spectrum.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s;

    // √S · √S = S.
    let root = typed_s.sqrt().unwrap();
    let squared = root.compose(&root).unwrap();
    let error = squared
        .data()
        .iter()
        .zip(typed_s.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-9, "√S · √S != S: {error}");

    // The dense storage of the same tensor: `add`ing zero to a dense sibling
    // forces the materialized payload, and the block walk must reach the same
    // answer as the compact arm.
    let dense_zero = TensorMap::<_, f64>::id(&runtime, &typed_s.domain())
        .unwrap()
        .scale(0.0);
    let dense_s = typed_s.add(&dense_zero, 1.0, 1.0).unwrap();
    assert_eq!(dense_s.data(), typed_s.data());
    assert_eq!(
        dense_s.sqrt().unwrap().data(),
        typed_s.sqrt().unwrap().data()
    );
}

#[test]
fn sqrt_refuses_anything_that_is_not_a_diagonal_bond_tensor() {
    // What: the scope guard. `sqrt` here is TensorKit's
    // `sqrt(::DiagonalTensorMap)` and nothing wider — a general endomorphism
    // `sqrt` needs a Schur seam that does not exist below this facade — so a
    // rank-two tensor, and a bond-shaped one whose block has a nonzero
    // off-diagonal entry, are both refused.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    // The shape guard, not the off-diagonal walk behind it.
    match typed.sqrt() {
        Err(tenet::typed::Error::InvalidArgument(message)) => {
            assert!(
                message.contains("`[v] <- [v]`"),
                "a non-bond tensor was refused by something other than the shape \
                 guard: {message}"
            );
        }
        other => panic!("a non-bond tensor was accepted: {other:?}"),
    }
    let typed = z2_endomorphism(&runtime);
    // Bond shaped (`[v] <- [v]`) but not diagonal: the off-diagonal check is
    // what refuses it, and it is the check that separates this from a general
    // endomorphism `sqrt`.
    assert!(typed.data().iter().any(|&value| value != 0.0));
    match typed.sqrt() {
        Err(tenet::typed::Error::InvalidArgument(message)) => {
            assert!(message.contains("off-diagonal"), "unexpected: {message}");
        }
        other => panic!("expected an off-diagonal refusal, got {other:?}"),
    }
}

#[test]
fn sqrt_of_a_negative_f64_entry_points_at_the_complex_payload() {
    // What: a real payload has no principal square root of a negative number to
    // return, so both storages refuse and say what to do instead. This is
    // TensorKit's `DiagonalTensorMap` behavior (a `DomainError`); TensorKit's
    // dense path silently returns a complex tensor, which a typed signature
    // cannot express and which disagrees with TensorKit's own diagonal path.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s.scale(-1.0);

    for (name, result) in [
        ("compact", typed_s.sqrt()),
        (
            "dense",
            typed_s
                .add(
                    &TensorMap::<_, f64>::id(&runtime, &typed_s.domain())
                        .unwrap()
                        .scale(0.0),
                    1.0,
                    1.0,
                )
                .unwrap()
                .sqrt(),
        ),
    ] {
        match result {
            Err(tenet::typed::Error::InvalidArgument(message)) => {
                assert!(
                    message.contains("negative") && message.contains("c64"),
                    "the {name} arm's message does not point at the complex \
                     payload: {message}"
                );
            }
            other => panic!("the {name} arm accepted a negative entry: {other:?}"),
        }
    }
}

#[test]
fn sqrt_of_a_complex_payload_takes_the_principal_branch() {
    // What: with a c64 payload there is a root to return, and it is the
    // principal one — `√(-1) = i`, not `-i`.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [(tenet::core::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    let negative = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices: &[usize]| {
        if indices[0] == indices[1] {
            Complex64::new(-1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
    .unwrap();

    let root = negative.sqrt().unwrap();
    let squared = root.compose(&root).unwrap();
    let error = squared
        .data()
        .iter()
        .zip(negative.data())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-12, "√t · √t != t: {error}");
    // The principal branch, not the other one: every diagonal entry is `+i`.
    for (index, value) in root.data().iter().enumerate() {
        let expected = if on_diagonal(&root, index) {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
        assert!(
            (value - expected).norm() < 1e-12,
            "entry {index} is {value}, not the principal root {expected}"
        );
    }
}

#[test]
fn c64_compact_inv_and_pinv_are_elementwise_reciprocals() {
    // A c64 tensor's singular values are stored in the payload dtype, so both
    // compact matrix functions must apply their laws directly to those values.
    let _guard = cache_lock();
    let runtime = runtime();
    // A full-rank c64 `[v] <- [v]` map with a wide spectrum.
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 16),
            (tenet::core::Z2Irrep::ODD, 17),
        ],
    )
    .unwrap();
    let mut state = 0x05ee_dc64_u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        complex(((state >> 33) as f64) / (u32::MAX as f64) + 0.5)
    };
    let typed = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| next()).unwrap();
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s;
    let source = typed_s.diagonal_spectrum().unwrap().unwrap();
    let sigma_max = source
        .iter()
        .flat_map(|entry| &entry.values)
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let rcond = 1e-12;
    let inverse = typed_s.inv().unwrap().diagonal_spectrum().unwrap().unwrap();
    let pseudo = typed_s
        .pinv(rcond)
        .unwrap()
        .diagonal_spectrum()
        .unwrap()
        .unwrap();
    for ((source, inverse), pseudo) in source.iter().zip(&inverse).zip(&pseudo) {
        assert_eq!(source.sector, inverse.sector);
        assert_eq!(source.sector, pseudo.sector);
        for ((&value, &inverse), &pseudo) in source
            .values
            .iter()
            .zip(&inverse.values)
            .zip(&pseudo.values)
        {
            let reciprocal = Complex64::new(1.0, 0.0) / value;
            assert_eq!(inverse, reciprocal);
            let expected = if value.norm() > rcond * sigma_max {
                reciprocal
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert_eq!(pseudo, expected);
        }
    }
}

// ---------------------------------------------------------------------------
// Issue #584: the compact diagonal arm of `contract`.
// ---------------------------------------------------------------------------

/// The three axis patterns the diagonal `contract` arm claims, plus one it must
/// decline, as `(name, lhs_axes, rhs_axes, output_axes)` on a `[v, v] <- [v]`
/// tensor `t` and its own SVD spectrum `s`.
///
/// `t · s` contracts `t`'s domain axis against `s`'s codomain axis (the
/// compose-shaped pairing, which is the only one the engine admits: contracted
/// legs must agree on their duality flag), and `s · t` the mirror.
const DIAGONAL_CONTRACT_CASES: &[(&str, bool, &[usize], &[usize], &[usize])] = &[
    // `t · s`, identity output order: the scaled leg stays last.
    ("t*s", true, &[2], &[0], &[0, 1, 2]),
    // The same, with the output order moving the scaled leg across the split.
    ("t*s reordered", true, &[2], &[0], &[2, 0, 1]),
    // `s · t`: `s`'s domain axis against `t`'s leading codomain axis, so the
    // scaled leg comes first and the destination is `[v] <- [v, v]`.
    ("s*t", false, &[1], &[0], &[0, 1, 2]),
    ("s*t reordered", false, &[1], &[0], &[1, 2, 0]),
    // `s · t` on `t`'s second codomain axis: the arm's leg position is free
    // within the preserved side.
    ("s*t inner leg", false, &[1], &[1], &[0, 1, 2]),
    // Declined by the arm — `s`'s *codomain* axis against `t`'s domain axis is
    // admissible but is not one of the two proved geometries — and computed
    // densely. The expected bytes are pinned below.
    ("s*t dense fallback", false, &[0], &[2], &[0, 1, 2]),
];

#[test]
fn diagonal_contract_preserves_left_provider_authority_on_every_compact_arm() {
    let _guard = cache_lock();
    let runtime = runtime();

    macro_rules! exercise {
        ($name:literal, $rule:expr, $sectors:expr) => {{
            let left_leg = GradedSpace::try_new_with_arc(Arc::new($rule), $sectors).unwrap();
            let right_leg = GradedSpace::try_new_with_arc(Arc::new($rule), $sectors).unwrap();
            let left = TensorMap::<_, f64>::id(&runtime, [&left_leg])
                .unwrap()
                .scale(2.0);
            let right = TensorMap::<_, f64>::id(&runtime, [&right_leg])
                .unwrap()
                .scale(3.0);
            let left_d = left.svd_compact().unwrap().1;
            let right_d = right.svd_compact().unwrap().1;
            assert!(!std::ptr::eq(left.provider(), right.provider()));

            let left_dense = forced_dense(&left_d);
            let right_dense = forced_dense(&right_d);

            let actual = [
                left.contract(&right_d, &[1], &[0], &[0, 1]).unwrap(),
                right_d.contract(&left, &[1], &[0], &[0, 1]).unwrap(),
                right_d.contract(&left_d, &[1], &[0], &[0, 1]).unwrap(),
            ];
            let expected = [
                left.contract(&right_dense, &[1], &[0], &[0, 1]).unwrap(),
                right_dense.contract(&left, &[1], &[0], &[0, 1]).unwrap(),
                right_dense
                    .contract(&left_dense, &[1], &[0], &[0, 1])
                    .unwrap(),
            ];
            let providers = [left.provider(), right_d.provider(), right_d.provider()];

            for (index, arm) in ["t*D", "D*t", "D*D"].into_iter().enumerate() {
                let (actual, expected) = (&actual[index], &expected[index]);
                assert_same_legs(&actual.codomain(), &expected.codomain());
                assert_same_legs(&actual.domain(), &expected.domain());
                assert_eq!(actual.data(), expected.data(), "{} {arm} values", $name);
                assert!(
                    std::ptr::eq(actual.provider(), providers[index]),
                    "{} {arm} lost left authority",
                    $name
                );
                assert_eq!(
                    actual.diagonal_spectrum().unwrap().is_some(),
                    index == 2,
                    "{} {arm} storage",
                    $name
                );
            }
        }};
    }

    exercise!(
        "Z2",
        tenet::core::Z2FusionRule,
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 1),
        ]
    );
    exercise!(
        "U1",
        tenet::core::U1FusionRule,
        [
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 1),
        ]
    );
}

#[test]
fn complex_diagonal_contract_matches_the_typed_dense_route() {
    // What: every complex compact arm agrees with the ordinary typed engine
    // route on the same spaces, including a compact `D · D` result.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);
    let typed_s = typed.svd_compact().unwrap().1;
    let dense_s = forced_dense(&typed_s);

    for &(_name, spectrum_on_the_right, lhs_axes, rhs_axes, output_axes) in DIAGONAL_CONTRACT_CASES
    {
        let (fast_lhs, fast_rhs, dense_lhs, dense_rhs) = if spectrum_on_the_right {
            (&typed, &typed_s, &typed, &dense_s)
        } else {
            (&typed_s, &typed, &dense_s, &typed)
        };
        let expected = dense_lhs
            .contract(dense_rhs, lhs_axes, rhs_axes, output_axes)
            .unwrap();
        let got = fast_lhs
            .contract(fast_rhs, lhs_axes, rhs_axes, output_axes)
            .unwrap();
        assert_same_legs(&got.codomain(), &expected.codomain());
        assert_same_legs(&got.domain(), &expected.domain());
        assert_data_close_c64(got.data(), expected.data());
    }

    let expected = dense_s.contract(&dense_s, &[1], &[0], &[0, 1]).unwrap();
    let got = typed_s.contract(&typed_s, &[1], &[0], &[0, 1]).unwrap();
    assert_same_legs(&got.codomain(), &expected.codomain());
    assert_same_legs(&got.domain(), &expected.domain());
    assert_data_close_c64(got.data(), expected.data());
    assert!(got.diagonal_spectrum().unwrap().is_some());
}

#[test]
fn the_diagonal_contract_arm_keeps_fermionic_signs() {
    // What: bends inside the compact arm's `permute` pick up the same parity
    // signs as the ordinary typed engine route.
    //
    // The supertrace twist `contract` applies to a **dual** contracted leg of
    // the right operand cannot be reached from this facade, so it is not
    // asserted here: a compact spectrum's bond leg is built non-dual
    // (`diagonal_bond_bound_space_like`), the engine admits a contraction only
    // when the two contracted legs agree on their duality flag, and the arm's
    // right-operand leg is codomain-side, where external duality *is* that
    // flag. `TensorMap::try_contract_diagonal` declines rather than assumes it,
    // and that guard is what would have to be tested if a dual bond leg ever
    // became constructible here.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fermionic_endo(&runtime);
    let typed_s = typed.svd_compact().unwrap().1;
    let dense_s = forced_dense(&typed_s);

    for &(name, spectrum_on_the_right, output_axes) in &[
        ("t*s", true, &[0usize, 1][..]),
        ("t*s reordered", true, &[1, 0][..]),
        ("s*t", false, &[0, 1][..]),
    ] {
        let (fast_lhs, fast_rhs, dense_lhs, dense_rhs) = if spectrum_on_the_right {
            (&typed, &typed_s, &typed, &dense_s)
        } else {
            (&typed_s, &typed, &dense_s, &typed)
        };
        let expected = dense_lhs
            .contract(dense_rhs, &[1], &[0], output_axes)
            .unwrap();
        let got = fast_lhs
            .contract(fast_rhs, &[1], &[0], output_axes)
            .unwrap();
        assert_same_legs(&got.codomain(), &expected.codomain());
        assert_same_legs(&got.domain(), &expected.domain());
        assert_eq!(got.data(), expected.data(), "fermionic {name}");
    }

    let expected = dense_s.contract(&dense_s, &[1], &[0], &[0, 1]).unwrap();
    let got = typed_s.contract(&typed_s, &[1], &[0], &[0, 1]).unwrap();
    assert_same_legs(&got.codomain(), &expected.codomain());
    assert_same_legs(&got.domain(), &expected.domain());
    assert_eq!(got.data(), expected.data(), "fermionic s*s");
    assert!(got.diagonal_spectrum().unwrap().is_some());
}

#[test]
fn the_diagonal_contract_arm_declines_an_illegal_contraction() {
    // What: the arm must not answer where the dense route would refuse. A
    // contracted leg whose duality flag does not match the spectrum's bond leg
    // is inadmissible, and the error is the expert layer's, not a scaled tensor
    // on a made-up space.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let typed = TensorMap::from_block_fn(&runtime, [&dual], [&leg], typed_fill_value).unwrap();
    let s = TensorMap::from_block_fn(&runtime, [&leg], [&leg], typed_fill_value)
        .unwrap()
        .svd_compact()
        .unwrap()
        .1;

    // `s`'s domain leg is non-dual, `typed`'s leading codomain leg is dual.
    assert!(s.contract(&typed, &[1], &[0], &[0, 1]).is_err());
    // The mirror direction needs its own case, because it is the `t · D` arm's
    // own leg comparison that has to reject it: `leg <- dual` contracted on its
    // domain axis against `s`, whose bond leg is non-dual by construction, so
    // the two raw flags differ and the engine refuses the pair.
    let dual_domain =
        TensorMap::from_block_fn(&runtime, [&leg], [&dual], typed_fill_value).unwrap();
    assert!(dual_domain.contract(&s, &[1], &[0], &[0, 1]).is_err());
    // A degeneracy mismatch on an otherwise well-oriented pair is the other way
    // the comparison earns its keep: nothing about the axis pattern is wrong, so
    // only the legs themselves say this is not a contraction.
    let narrow = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    let narrow_bond = TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], typed_fill_value)
        .unwrap()
        .svd_compact()
        .unwrap()
        .1;
    let wide = TensorMap::from_block_fn(&runtime, [&leg], [&leg], typed_fill_value).unwrap();
    assert!(wide.contract(&narrow_bond, &[1], &[0], &[0, 1]).is_err());
    assert!(narrow_bond.contract(&wide, &[1], &[0], &[0, 1]).is_err());
    // And a wrong-length axis list or a non-permutation output order is still
    // the expert layer's error rather than a fast-path answer. `[0, 2]` is the
    // out-of-range case, which the arm has to reject *before* indexing its own
    // source order with it.
    assert!(typed.contract(&s, &[1], &[0], &[0]).is_err());
    assert!(typed.contract(&s, &[1], &[0], &[1, 1]).is_err());
    assert!(wide.contract(&s, &[1], &[0], &[0, 2]).is_err());
    assert!(typed.contract(&s, &[9], &[0], &[0, 1]).is_err());
}

/// The codomain/domain split plus every leg's sectors, degeneracies and dual
/// flag. Comparing the split alone is too weak: a reordered output can leave the
/// rank and the codomain length intact and still land on legs with the opposite
/// duality flag, which is exactly what the `D · D` output-order guard refuses.
#[allow(clippy::type_complexity)]
fn space_shape(
    t: &TensorMap<tenet::core::Z2FusionRule, f64>,
) -> (usize, Vec<(Vec<tenet::core::Z2Irrep>, Vec<usize>, bool)>) {
    let legs = t
        .codomain()
        .iter()
        .chain(t.domain().iter())
        .map(|leg| {
            (
                leg.sectors().unwrap(),
                leg.degeneracies().to_vec(),
                leg.is_dual(),
            )
        })
        .collect();
    (t.codomain().len(), legs)
}

/// Every permutation of `0..n`, for the exhaustive output-order sweep below.
fn all_output_orders(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut orders = Vec::new();
    for head in 0..n {
        for mut rest in all_output_orders(n - 1) {
            for axis in &mut rest {
                if *axis >= head {
                    *axis += 1;
                }
            }
            let mut order = vec![head];
            order.extend(rest);
            orders.push(order);
        }
    }
    orders
}

#[test]
fn the_diagonal_contract_arm_is_its_own_dense_route_on_every_axis_pattern() {
    // What: the compact arm never differs from the dense route it replaces.
    // This direct oracle catches a shared fast-path mistake or a divergence
    // between the codomain rank derived by the arm (`self.rank() - 1` for
    // `t · D`, `1` for `D · t`) and the one the engine would build.
    //
    // Here the comparison is fast against dense *inside this facade*:
    // `forced_dense` adds an exact dense zero on the same space, which preserves
    // the values while forcing an ordinary dense payload. Every single-axis
    // pattern and output order is swept on both codomain/domain splits, so
    // `t · D` is covered at an inner domain axis (which `[v, v] <- [v]` cannot
    // reach: it has one domain
    // leg) as well as at the trailing one, and an inadmissible pattern must be
    // refused by both routes rather than answered by one.
    let _guard = cache_lock();
    let runtime = runtime();
    for split in [1, 2] {
        let t = z2_tensor_split(&runtime, split);
        let s = t.svd_compact().unwrap().1;
        let s_dense = forced_dense(&s);

        let mut fired = 0usize;
        for orders in [&all_output_orders(3)] {
            for output_axes in orders {
                for lhs_axis in 0..3 {
                    for rhs_axis in 0..2 {
                        // `t · s`
                        let dense = t.contract(&s_dense, &[lhs_axis], &[rhs_axis], output_axes);
                        let fast = t.contract(&s, &[lhs_axis], &[rhs_axis], output_axes);
                        let label =
                            format!("t*s split={split} {lhs_axis}/{rhs_axis} {output_axes:?}");
                        match (dense, fast) {
                            (Ok(dense), Ok(fast)) => {
                                assert_eq!(
                                    space_shape(&fast),
                                    space_shape(&dense),
                                    "{label} space"
                                );
                                assert_eq!(fast.data(), dense.data(), "{label} payload");
                                fired += 1;
                            }
                            (Err(_), Err(_)) => {}
                            (dense, fast) => panic!(
                                "{label}: dense {:?} but fast {:?}",
                                dense.map(|_| ()),
                                fast.map(|_| ())
                            ),
                        }
                        // `s · t`
                        let dense = s_dense.contract(&t, &[rhs_axis], &[lhs_axis], output_axes);
                        let fast = s.contract(&t, &[rhs_axis], &[lhs_axis], output_axes);
                        let label =
                            format!("s*t split={split} {rhs_axis}/{lhs_axis} {output_axes:?}");
                        match (dense, fast) {
                            (Ok(dense), Ok(fast)) => {
                                assert_eq!(
                                    space_shape(&fast),
                                    space_shape(&dense),
                                    "{label} space"
                                );
                                assert_eq!(fast.data(), dense.data(), "{label} payload");
                                fired += 1;
                            }
                            (Err(_), Err(_)) => {}
                            (dense, fast) => panic!(
                                "{label}: dense {:?} but fast {:?}",
                                dense.map(|_| ()),
                                fast.map(|_| ())
                            ),
                        }
                    }
                }
            }
        }
        assert!(fired > 0, "split={split} swept no admissible pattern");

        // `s · s`, where the surviving bond may stay compact.
        for output_axes in all_output_orders(2) {
            for lhs_axis in 0..2 {
                for rhs_axis in 0..2 {
                    let dense = s_dense.contract(&s_dense, &[lhs_axis], &[rhs_axis], &output_axes);
                    let fast = s.contract(&s, &[lhs_axis], &[rhs_axis], &output_axes);
                    let label = format!("s*s {lhs_axis}/{rhs_axis} {output_axes:?}");
                    match (dense, fast) {
                        (Ok(dense), Ok(fast)) => {
                            assert_eq!(space_shape(&fast), space_shape(&dense), "{label} space");
                            assert_eq!(fast.data(), dense.data(), "{label} payload");
                        }
                        (Err(_), Err(_)) => {}
                        (dense, fast) => panic!(
                            "{label}: dense {:?} but fast {:?}",
                            dense.map(|_| ()),
                            fast.map(|_| ())
                        ),
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Issue #585: compact diagonal parity, round 2.
//
// The value oracles below are what pin the compact arms that follow. They are
// written against the typed dense and pointwise routes those arms replace, so
// they are independent of whatever shared helper the compact arms end up
// calling.
// ---------------------------------------------------------------------------

/// A dense twin of a compact bond factor. Adding an exact dense zero keeps the
/// values and space while forcing the mixed compact/dense arm.
fn forced_dense<R, D>(compact: &TensorMap<R, D>) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: tenet::prelude::TensorScalar + std::fmt::Debug,
{
    let codomain = compact.codomain();
    let domain = compact.domain();
    let zeros = TensorMap::zeros(compact.runtime(), &codomain, &domain)
        .expect("zero tensor on an admitted bond space is total");
    let dense = compact
        .add(&zeros, D::from_real(1.0), D::from_real(1.0))
        .expect("mixed add on one bond space is total");
    assert_eq!(dense.data(), compact.data(), "the dense twin lost values");
    assert!(
        dense.diagonal_spectrum().unwrap().is_none(),
        "the dense twin stayed compact"
    );
    dense
}

fn z2_bond(runtime: &Runtime) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed = TensorMap::from_block_fn(runtime, [&leg], [&leg], typed_fill_value).unwrap();
    typed.svd_compact().unwrap().1
}

/// The three rank-(1,1) re-orderings that reduce to the proved swap, plus the
/// two repartitions that do not — the latter are here so a compact arm that
/// fires too widely is caught by the same sweep.
fn rank_one_reorderings<R, D>(tensor: &TensorMap<R, D>) -> Vec<(&'static str, TensorMap<R, D>)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: tenet::prelude::TensorScalar + std::fmt::Debug,
{
    vec![
        ("permute", tensor.permute(&[1], &[0]).unwrap()),
        ("transpose", tensor.transpose().unwrap()),
        ("transpose_axes", tensor.transpose_axes(&[1], &[0]).unwrap()),
        ("repartition(1)", tensor.repartition(1).unwrap()),
        ("repartition(0)", tensor.repartition(0).unwrap()),
        ("repartition(2)", tensor.repartition(2).unwrap()),
    ]
}

#[test]
fn compact_rank_one_swaps_match_the_forced_dense_route() {
    // What: every re-ordering of a compact bond factor returns the tensor the
    // dense tree transform returns — same legs and same bytes.
    //
    // What this test cannot see: Z2 is self-dual and bosonic, so every swap
    // coefficient is exactly 1 — dropping the coefficient entirely still passes
    // here. The coefficient is carried by the Z3 and fZ2 sweep below, which is
    // where a mutation to it dies.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_bond(&runtime);
    let dense = forced_dense(&typed);

    for ((name, compact), (_, oracle)) in rank_one_reorderings(&typed)
        .into_iter()
        .zip(rank_one_reorderings(&dense))
    {
        assert_eq!(
            typed_leg_shapes(&compact),
            typed_leg_shapes(&oracle),
            "{name} legs"
        );
        assert_eq!(compact.data(), oracle.data(), "{name} payload");
    }
}

#[test]
fn compact_rank_one_swaps_match_the_dense_route_for_dual_and_fermionic_legs() {
    // What: the swap's per-sector coefficient is only observable where the two
    // ends of the bond are not interchangeable. Z3 is not self-dual, so the
    // destination block of every sector is a *different* sector; the fermionic
    // provider adds a braiding phase the bosonic cases cannot show. A dual leg
    // is swept on both, since bending is what fixes which sector labels the
    // destination structure carries.
    let _guard = cache_lock();
    let runtime = runtime();

    for is_dual in [false, true] {
        let provider = Arc::new(ExternalZ3::new());
        let leg = z3_leg(&provider, is_dual);
        let mut next = 0.0;
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| {
            next += 1.0;
            next
        })
        .unwrap();
        let compact = source.svd_compact().unwrap().1;
        let dense = forced_dense(&compact);
        for ((name, actual), (_, expected)) in rank_one_reorderings(&compact)
            .into_iter()
            .zip(rank_one_reorderings(&dense))
        {
            assert_eq!(
                typed_leg_shapes(&actual),
                typed_leg_shapes(&expected),
                "z3 dual={is_dual} {name} legs"
            );
            assert_eq!(
                actual.data(),
                expected.data(),
                "z3 dual={is_dual} {name} payload"
            );
        }

        let leg = GradedSpace::try_new_with_arc(
            Arc::new(tenet::core::FermionParityFusionRule),
            [
                (tenet::core::Z2Irrep::EVEN, 2),
                (tenet::core::Z2Irrep::ODD, 3),
            ],
        )
        .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
        .unwrap();
        let mut next = 0.0;
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| {
            next += 1.0;
            next
        })
        .unwrap();
        let compact = source.svd_compact().unwrap().1;
        let dense = forced_dense(&compact);
        for ((name, actual), (_, expected)) in rank_one_reorderings(&compact)
            .into_iter()
            .zip(rank_one_reorderings(&dense))
        {
            assert_eq!(
                typed_leg_shapes(&actual),
                typed_leg_shapes(&expected),
                "fZ2 dual={is_dual} {name} legs"
            );
            assert_eq!(
                actual.data(),
                expected.data(),
                "fZ2 dual={is_dual} {name} payload"
            );
        }
    }
}

/// A compact bond factor whose spectrum is exactly `values` per sector, built
/// by rescaling a singular-value factor: `scale` stays compact, so the fixture
/// never leaves compact storage on its way to the assertion.
fn z2_spectrum_fixture(
    runtime: &Runtime,
    rank_deficient: bool,
) -> TensorMap<tenet::core::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    // A constant block is rank one, so all but one singular value per sector is
    // zero — the positive *semi*definite fixture `isposdef` must still reject.
    let source = TensorMap::from_block_fn(runtime, [&leg], [&leg], |sectors, indices| {
        if rank_deficient {
            1.0
        } else {
            typed_fill_value(sectors, indices)
        }
    })
    .unwrap();
    source.svd_compact().unwrap().1
}

#[test]
fn compact_is_posdef_matches_the_forced_dense_route() {
    // What: reading the stored spectrum answers exactly what eigendecomposing
    // the materialization answers, on every sign pattern and at every
    // tolerance — including the strictness at zero, which is the one place a
    // `>=` would pass a positive-semidefinite spectrum the dense route rejects.
    let _guard = cache_lock();
    let runtime = runtime();

    let positive = z2_spectrum_fixture(&runtime, false);
    let semidefinite = z2_spectrum_fixture(&runtime, true);
    let negative = positive.scale(-1.0);
    let indefinite = positive.add(&semidefinite, 1.0, -3.0).unwrap();

    for (name, tensor) in [
        ("positive", &positive),
        ("semidefinite", &semidefinite),
        ("negative", &negative),
        ("indefinite", &indefinite),
    ] {
        let oracle = forced_dense(tensor);
        for tol in [0.0, 1e-14, 1e-8, 1e-3, 0.5] {
            assert_eq!(
                tensor.is_posdef(tol).unwrap(),
                oracle.is_posdef(tol).unwrap(),
                "{name} at tol {tol}"
            );
        }
    }

    // The one case whose answer is asserted absolutely rather than only against
    // the oracle: a rank-deficient spectrum is positive semidefinite, and
    // `isposdef` is strict.
    assert!(positive.is_posdef(0.0).unwrap());
    assert!(!semidefinite.is_posdef(0.0).unwrap());
    assert!(!negative.is_posdef(0.0).unwrap());
}

#[test]
fn compact_is_posdef_matches_the_forced_dense_route_for_a_hermitian_c64_spectrum() {
    // What: a c64 payload whose stored values are real up to rounding is the
    // case where a compact branch could disagree with the dense one — the dense
    // route Hermitian-eigendecomposes and so reads only the real part, and the
    // compact branch must make the same choice rather than, say, comparing a
    // modulus. The `d` of `eigh_full` is exactly that payload.
    let _guard = cache_lock();
    let runtime = runtime();
    let real = z2_complex_tensor(&runtime);
    let square = real
        .contract(&real.adjoint().unwrap(), &[2], &[0], &[0, 1, 2, 3])
        .unwrap();
    let hermitian = square.repartition(2).unwrap();
    assert!(hermitian.is_hermitian(1e-10).unwrap());

    let d = hermitian.eigh_full().unwrap().0;

    // The Hermiticity gate is load-bearing and is checked here, because every
    // other fixture in this file answers the same with or without it. Rotating
    // the spectrum by `1 + i` keeps every real part positive — so a predicate
    // that reads only the stored real parts calls it positive definite — while
    // making the tensor not Hermitian at all. The gate is the only thing
    // standing between the compact arm and a wrong `true`.
    // Built from singular values, not from the Gram's eigenvalues: the latter
    // can hold an exact zero, which the compact predicate rejects on its own
    // and which would therefore hide the gate rather than test it.
    let skewed = real
        .svd_compact()
        .unwrap()
        .1
        .scale(Complex64::new(1.0, 1.0));
    assert!(!skewed.is_hermitian(1e-10).unwrap());
    for tol in [0.0, 1e-14, 1e-8, 1e-3] {
        assert!(!skewed.is_posdef(tol).unwrap(), "skewed at tol {tol}");
        assert_eq!(
            skewed.is_posdef(tol).unwrap(),
            forced_dense(&skewed).is_posdef(tol).unwrap(),
            "skewed at tol {tol}"
        );
    }

    let oracle = forced_dense(&d);
    for tol in [0.0, 1e-14, 1e-8, 1e-3] {
        assert_eq!(
            d.is_posdef(tol).unwrap(),
            oracle.is_posdef(tol).unwrap(),
            "gram at tol {tol}"
        );
        let flipped = d.scale(Complex64::new(-1.0, 0.0));
        assert_eq!(
            flipped.is_posdef(tol).unwrap(),
            forced_dense(&flipped).is_posdef(tol).unwrap(),
            "negated gram at tol {tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Issue #604: the compact full-pair trace arm. The typed `trace_pairs` used to
// densify a compact spectrum factor unconditionally. The value sweep here
// covers the same rules, orientations, and variants against a forced-dense
// typed route. The storage claim lives in `typed_diagonal_allocations.rs`.
// ---------------------------------------------------------------------------

/// A compact bond factor from a counter-filled `[v] <- [v]` endomorphism.
fn compact_bond_trace<R, D>(
    runtime: &Runtime,
    typed_leg: &GradedSpace<R>,
    fill: impl Fn(f64) -> D + Copy,
) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: tenet::prelude::TensorScalar + std::fmt::Debug,
{
    let mut next: f64 = 0.0;
    let typed: TensorMap<R, D> =
        TensorMap::from_block_fn(runtime, [typed_leg], [typed_leg], |_, _| {
            next += 1.0;
            fill(next)
        })
        .unwrap();
    typed.svd_compact().unwrap().1
}

/// The typed compact trace against the typed forced-dense engine route, both
/// pair orders. Close, not byte-for-byte: the engine reduces block by block in
/// its own order, so this is the independent value oracle.
fn assert_compact_trace_matches_forced_dense<R, D>(
    label: &str,
    typed: &TensorMap<R, D>,
    widen: impl Fn(D) -> Complex64 + Copy,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: tenet::prelude::TensorScalar + std::fmt::Debug,
{
    let dense: TensorMap<R, D> = forced_dense(typed);
    for pairs in [[(0usize, 1usize)], [(1usize, 0usize)]] {
        let actual: Complex64 = widen(typed.trace_pairs(&pairs).unwrap().scalar().unwrap());
        let oracle: Complex64 = widen(dense.trace_pairs(&pairs).unwrap().scalar().unwrap());
        assert!(
            (actual - oracle).norm() <= 1e-12 * oracle.norm().max(1.0),
            "{label} {pairs:?}: compact {actual:?} vs dense route {oracle:?}"
        );
    }
}

/// The freshly factorized bond, both compact swaps, and the compact adjoint.
fn sweep_compact_trace_variants<R, D>(
    label: &str,
    typed_s: &TensorMap<R, D>,
    widen: impl Fn(D) -> Complex64 + Copy,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: tenet::prelude::TensorScalar + std::fmt::Debug,
{
    let variants: Vec<(&str, TensorMap<R, D>)> = vec![
        ("plain", typed_s.clone()),
        ("transposed", typed_s.transpose().unwrap()),
        ("permuted", typed_s.permute(&[1], &[0]).unwrap()),
        ("adjoint", typed_s.adjoint().unwrap()),
    ];
    for (which, typed) in &variants {
        assert_compact_trace_matches_forced_dense(&format!("{label} {which}"), typed, widen);
    }
}

#[test]
fn compact_full_trace_matches_the_forced_dense_route() {
    // What: the full trace of a rank-(1,1) compact spectrum factor over its
    // only pair is the same number on compact and forced-dense storage — U(1),
    // SU(2) (non-unit quantum dimensions), fZ2 (the twist makes it a
    // supertrace) and the packed U(1) x fZ2 product route, on plain and dual
    // bond legs, f64 and c64, both pair orders, and on legs the compact swaps
    // bent themselves.
    let _guard = cache_lock();
    let runtime = runtime();
    let real = |value: f64| value;
    let complex = |value: f64| Complex64::new(value, 0.25 + value % 3.0);
    let widen_real = |value: f64| Complex64::new(value, 0.0);
    let widen_complex = |value: Complex64| value;

    for is_dual in [false, true] {
        // U(1).
        let mut typed_leg: GradedSpace<tenet::core::U1FusionRule> = GradedSpace::try_new_with_arc(
            Arc::new(tenet::core::U1FusionRule),
            [
                (tenet::core::U1Irrep::new(-1), 2),
                (tenet::core::U1Irrep::new(0), 3),
                (tenet::core::U1Irrep::new(1), 2),
            ],
        )
        .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("u1 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(&format!("u1 dual={is_dual} c64"), &typed_s, widen_complex);

        // SU(2): dim(c) takes the values 1 and 2, so a coefficient-free
        // reduction cannot pass.
        let mut typed_leg: GradedSpace<tenet::core::SU2FusionRule> = GradedSpace::try_new_with_arc(
            Arc::new(tenet::core::SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 3),
            ],
        )
        .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("su2 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(&format!("su2 dual={is_dual} c64"), &typed_s, widen_complex);

        // fZ2: the twist is -1 on the odd sector, so this is where the
        // supertrace coefficient and its orientation live.
        let mut typed_leg: GradedSpace<tenet::core::FermionParityFusionRule> =
            GradedSpace::try_new_with_arc(
                Arc::new(tenet::core::FermionParityFusionRule),
                [
                    (tenet::core::Z2Irrep::EVEN, 2),
                    (tenet::core::Z2Irrep::ODD, 3),
                ],
            )
            .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("fz2 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(&format!("fz2 dual={is_dual} c64"), &typed_s, widen_complex);

        // U(1) x fZ2, using the packed codec from the #589 section's rule
        // aliases: both the charge and parity factor must survive into the
        // coefficient.
        let product_label = |charge: i32, parity: u8| {
            tenet::core::ProductSector::new(tenet::core::U1Irrep::new(charge), parity_irrep(parity))
        };
        let mut typed_leg: GradedSpace<U1Fz2Rule> = GradedSpace::try_new_with_arc(
            Arc::new(U1Fz2Rule::new(
                tenet::core::U1FusionRule,
                tenet::core::FermionParityFusionRule,
            )),
            [(product_label(0, 0), 2), (product_label(1, 1), 3)],
        )
        .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("u1xfz2 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(
            &format!("u1xfz2 dual={is_dual} c64"),
            &typed_s,
            widen_complex,
        );
    }

    // Genuinely complex stored values (a spectrum has none out of `svd`, and
    // the adjoint variant only conjugates): a compact complex rotation stays
    // compact, and the forced-dense oracle covers it on the rule where the
    // twist could interact with the phase.
    let typed_leg: GradedSpace<tenet::core::FermionParityFusionRule> =
        GradedSpace::try_new_with_arc(
            Arc::new(tenet::core::FermionParityFusionRule),
            [
                (tenet::core::Z2Irrep::EVEN, 2),
                (tenet::core::Z2Irrep::ODD, 3),
            ],
        )
        .unwrap();
    let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
    let rotated: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        typed_s.scale(Complex64::new(0.8, -0.6));
    assert_compact_trace_matches_forced_dense("fz2 rotated c64", &rotated, widen_complex);
    assert_compact_trace_matches_forced_dense(
        "fz2 rotated transposed c64",
        &rotated.transpose().unwrap(),
        widen_complex,
    );
}

#[test]
fn compact_full_trace_is_the_supertrace_and_the_transpose_flips_it() {
    // What: on a single fermion-parity sector, `trace_pairs` and `tr` differ by
    // exactly the twist: an odd fZ2 bond flips the sign, an even one does not,
    // and the bosonic Z2 twin of the odd bond pins that the sign is the *twist*
    // and not the parity label. The transpose duals both bond legs, and the
    // traced channel is twisted only where its leg is not dual, so the same
    // tensor traces without the fermionic sign once transposed — which kills a
    // coefficient that reads the sector alone, or swaps the dual and non-dual
    // arms.
    let _guard = cache_lock();
    let runtime = runtime();

    let assert_supertrace =
        |name: &str, sign: f64, s: &TensorMap<tenet::core::FermionParityFusionRule, f64>| {
            let positive: f64 = s.tr().unwrap();
            let traced: f64 = s.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
            assert_eq!(traced, sign * positive, "{name}");
            let transposed: TensorMap<tenet::core::FermionParityFusionRule, f64> =
                s.transpose().unwrap();
            let transposed_positive: f64 = transposed.tr().unwrap();
            let transposed_traced: f64 =
                transposed.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
            assert_eq!(transposed_traced, transposed_positive, "{name} transposed");
        };

    for (name, parity, sign) in [
        ("fz2 even", tenet::core::Z2Irrep::EVEN, 1.0),
        ("fz2 odd", tenet::core::Z2Irrep::ODD, -1.0),
    ] {
        let leg: GradedSpace<tenet::core::FermionParityFusionRule> = GradedSpace::try_new_with_arc(
            Arc::new(tenet::core::FermionParityFusionRule),
            [(parity, 3)],
        )
        .unwrap();
        let mut next: f64 = 0.0;
        let source: TensorMap<tenet::core::FermionParityFusionRule, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| {
                next += 1.0;
                next
            })
            .unwrap();
        let s: TensorMap<tenet::core::FermionParityFusionRule, f64> =
            source.svd_compact().unwrap().1;
        assert_supertrace(name, sign, &s);
    }

    // The bosonic twin of the odd fixture: same parity label, twist +1, so the
    // supertrace *is* the positive trace here.
    let leg: GradedSpace<tenet::core::Z2FusionRule> = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [(tenet::core::Z2Irrep::ODD, 3)],
    )
    .unwrap();
    let mut next: f64 = 0.0;
    let source: TensorMap<tenet::core::Z2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| {
            next += 1.0;
            next
        })
        .unwrap();
    let s: TensorMap<tenet::core::Z2FusionRule, f64> = source.svd_compact().unwrap().1;
    let positive: f64 = s.tr().unwrap();
    let traced: f64 = s.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
    assert_eq!(traced, positive, "z2 odd");
}

#[test]
fn compact_trace_boundary_geometries_keep_their_existing_routes() {
    // What: the compact arm's boundaries. Tracing nothing on a compact factor
    // returns the source (the pre-guard short-circuit), and a malformed pair
    // list errors before the arm can run — the same validation order as on
    // dense storage. The dense geometries
    // outside the guard (rank > 2, partial pairs) are pinned by the Phase 5
    // `trace_pairs` tests above, which this issue must keep green.
    let _guard = cache_lock();
    let runtime = runtime();
    let s = z2_bond(&runtime);

    let untouched: TensorMap<tenet::core::Z2FusionRule, f64> = s.trace_pairs(&[]).unwrap();
    assert_eq!(untouched.data(), s.data());

    for pairs in [vec![(0usize, 9usize)], vec![(0, 0)], vec![(0, 1), (1, 0)]] {
        assert!(matches!(
            s.trace_pairs(&pairs).unwrap_err(),
            tenet::prelude::Error::InvalidArgument(message)
                if message.contains("invalid trace pair list")
        ));
    }
}

// ---------------------------------------------------------------------------
// Issue #589: semantic laws for U(1), U(1) x fZ2, and
// fZ2 x U(1) x SU(2). The two product rules exercise the packed product codec.
// ---------------------------------------------------------------------------

type U1Fz2Codec =
    tenet::core::PackedProductCodec<tenet::core::U1SectorLayout, tenet::core::Fz2SectorLayout>;
type Fz2U1Codec =
    tenet::core::PackedProductCodec<tenet::core::Fz2SectorLayout, tenet::core::U1SectorLayout>;
type Fz2U1Layout =
    tenet::core::ProductSectorLayout<tenet::core::Fz2SectorLayout, tenet::core::U1SectorLayout>;
type Fz2U1Su2Codec = tenet::core::PackedProductCodec<Fz2U1Layout, tenet::core::Su2SectorLayout>;

type U1Fz2Rule = tenet::core::ProductFusionRule<
    tenet::core::U1FusionRule,
    tenet::core::FermionParityFusionRule,
    U1Fz2Codec,
>;
type Fz2U1Rule = tenet::core::ProductFusionRule<
    tenet::core::FermionParityFusionRule,
    tenet::core::U1FusionRule,
    Fz2U1Codec,
>;
type Fz2U1Su2Rule =
    tenet::core::ProductFusionRule<Fz2U1Rule, tenet::core::SU2FusionRule, Fz2U1Su2Codec>;

/// `0` even, `1` odd, matching the public fermion-parity label contract.
// Strict on purpose: an out-of-range parity must not fold onto a valid label.
fn parity_irrep(parity: u8) -> tenet::core::Z2Irrep {
    match parity {
        0 => tenet::core::Z2Irrep::EVEN,
        1 => tenet::core::Z2Irrep::ODD,
        other => panic!("not a fermion parity: {other} (parity is exactly 0 or 1)"),
    }
}

/// `[p, q] <- [p, q]`, filled with a counter starting at `first_value`.
///
/// Why this one geometry for all three families: it is simultaneously a
/// composition (`self`'s domain *is* `other`'s codomain, so `compose` and a
/// two-axis `contract` are both legal on the same pair, which is what pins the
/// fermionic twist), square enough for every factorization, and rank 4, so a
/// nonidentity output order has somewhere to move a leg.
///
/// Why a plain counter rather than a label-derived fill: every element is then
/// distinct, so any reordering of blocks or of elements within a block moves
/// the buffer. `first_value` makes the second operand carry different values;
/// `contract` against a copy of `self` could otherwise hide a swap.
///
/// Why `p != q` in every family: with two distinct legs, permuting them is
/// visible in the leg shapes, so a nonidentity output order cannot be undone
/// by coincidence. One of the two is dual, because a dual leg is the only
/// place a fermionic twist can act.
fn counter_oracle<R>(
    runtime: &Runtime,
    typed_legs: (&GradedSpace<R>, &GradedSpace<R>),
    first_value: f64,
) -> TensorMap<R, f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let mut next = first_value - 1.0;
    let typed = TensorMap::from_block_fn(
        runtime,
        [typed_legs.0, typed_legs.1],
        [typed_legs.0, typed_legs.1],
        |_, _| {
            next += 1.0;
            next
        },
    )
    .unwrap();
    assert!(!typed.data().is_empty());
    typed
}

/// The U(1) oracle. `p` carries three charges with unequal degeneracies
/// and `q` two, so the two legs are distinguishable by shape alone; `q` is
/// dual, so the charge balance of a block is not symmetric under swapping the
/// legs either.
fn u1_oracle(runtime: &Runtime, first_value: f64) -> TensorMap<tenet::core::U1FusionRule, f64> {
    let rule = Arc::new(tenet::core::U1FusionRule);
    let typed_p = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (tenet::core::U1Irrep::new(-1), 1),
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    // Built through `try_dual` because the dual flips the sector labels too.
    let typed_q = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    counter_oracle(runtime, (&typed_p, &typed_q), first_value)
}

/// The U(1) x fZ2 oracle. Both legs mix the two fermion parities with
/// nonzero U(1) charge, so neither the parity factor nor the charge factor is
/// constant on a leg — a product route that dropped either component would
/// still produce a nonempty layout, but not this one.
fn u1_fz2_oracle(runtime: &Runtime, first_value: f64) -> TensorMap<U1Fz2Rule, f64> {
    let rule = Arc::new(U1Fz2Rule::new(
        tenet::core::U1FusionRule,
        tenet::core::FermionParityFusionRule,
    ));
    let label = |charge: i32, parity: u8| {
        tenet::core::ProductSector::new(
            tenet::core::U1Irrep::new(charge),
            if parity == 0 {
                tenet::core::Z2Irrep::EVEN
            } else {
                tenet::core::Z2Irrep::ODD
            },
        )
    };
    let typed_p =
        GradedSpace::try_new_with_arc(Arc::clone(&rule), [(label(0, 0), 1), (label(1, 1), 2)])
            .unwrap();
    let typed_q = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [(label(-1, 1), 1), (label(0, 0), 2), (label(1, 1), 1)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    counter_oracle(runtime, (&typed_p, &typed_q), first_value)
}

/// The fZ2 x U(1) x SU(2) oracle: the only route that exercises the
/// fermionic twist and the quantum-dimension weights at once.
///
/// Every one of the three factors is deliberately nonconstant across the two
/// legs: both parities appear (so the twist has somewhere to act), the charges
/// are `-1, 0, 1, 2` (so the U(1) balance is not automatic), and the spins are
/// `0, 1/2, 1` (so `dim(c)` takes the values 1, 2 and 3 and a weight-free
/// `inner` cannot pass). A fixture with spin 0 everywhere would not exercise
/// the weighted laws below.
fn fz2_u1_su2_oracle(runtime: &Runtime, first_value: f64) -> TensorMap<Fz2U1Su2Rule, f64> {
    let (typed_p, typed_q) = fz2_u1_su2_typed_legs();
    counter_oracle(runtime, (&typed_p, &typed_q), first_value)
}

/// The legs shared with the twist identity test below.
fn fz2_u1_su2_typed_legs() -> (GradedSpace<Fz2U1Su2Rule>, GradedSpace<Fz2U1Su2Rule>) {
    let rule = Arc::new(Fz2U1Su2Rule::new(
        Fz2U1Rule::new(
            tenet::core::FermionParityFusionRule,
            tenet::core::U1FusionRule,
        ),
        SU2FusionRule,
    ));
    let label = |parity: u8, charge: i32, twice_spin: usize| {
        tenet::core::ProductSector::new(
            tenet::core::ProductSector::new(
                if parity == 0 {
                    tenet::core::Z2Irrep::EVEN
                } else {
                    tenet::core::Z2Irrep::ODD
                },
                tenet::core::U1Irrep::new(charge),
            ),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    let typed_p = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [(label(0, 0, 0), 1), (label(1, 1, 1), 2)],
    )
    .unwrap();
    let typed_q = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (label(1, -1, 1), 1),
            (label(0, 0, 2), 1),
            (label(0, 2, 0), 2),
        ],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    (typed_p, typed_q)
}

/// Asserts a nonzero result: a law on an empty or all-zero buffer is vacuous,
/// and the product routes are exactly where an
/// over-restrictive fusion could silently produce one.
fn assert_nonzero(what: &str, data: &[f64]) {
    assert!(
        data.iter().any(|value| *value != 0.0),
        "{what}: result is empty or all zero, so the law proves nothing"
    );
}

/// Contract, compose, and compact-SVD laws shared by all three provider families.
fn assert_contract_compose_compact_laws_hold<R>(
    what: &str,
    nontrivial_twist: bool,
    typed: (&TensorMap<R, f64>, &TensorMap<R, f64>),
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let shapes = typed_leg_shapes(typed.0);
    assert_ne!(shapes[0], shapes[1], "{what}: p and q must differ");
    assert!(shapes[1].0, "{what}: q must be dual");
    assert_ne!(
        typed.0.data(),
        typed.1.data(),
        "{what}: operands must differ"
    );

    // Contracting the full domain against the other codomain leaves four open
    // legs. An explicit output order must equal the default typed contraction
    // followed by the corresponding public permutation.
    let (lhs_axes, rhs_axes) = ([2, 3], [0, 1]);
    let output_axes = [1, 3, 0, 2];
    let default_contract = typed
        .0
        .contract(typed.1, &lhs_axes, &rhs_axes, &[0, 1, 2, 3])
        .unwrap();
    let ordered_contract = typed
        .0
        .contract(typed.1, &lhs_axes, &rhs_axes, &output_axes)
        .unwrap();
    let permuted_default = default_contract.permute(&[1, 3], &[0, 2]).unwrap();
    assert_same_legs(&ordered_contract.codomain(), &permuted_default.codomain());
    assert_same_legs(&ordered_contract.domain(), &permuted_default.domain());
    assert_eq!(
        ordered_contract.data(),
        permuted_default.data(),
        "{what}: order"
    );
    assert!(
        std::ptr::eq(ordered_contract.provider(), typed.0.provider()),
        "{what}: contract lost left provider authority"
    );
    assert_nonzero(what, ordered_contract.data());
    assert_ne!(
        typed_leg_shapes(&default_contract),
        typed_leg_shapes(&ordered_contract),
        "{what}: the nonidentity output order did not move a leg"
    );
    assert_ne!(
        default_contract.data(),
        ordered_contract.data(),
        "{what}: the nonidentity output order left the buffer unchanged"
    );

    // Composition omits the supertrace twist of ordinary contraction. Twisting
    // exactly the right operand's dual codomain legs therefore relates the two.
    let dual_codomain_legs: Vec<usize> = typed
        .1
        .codomain()
        .iter()
        .enumerate()
        .filter_map(|(index, leg)| leg.is_dual().then_some(index))
        .collect();
    let twisted_right = typed.1.twist(&dual_codomain_legs).unwrap();
    let composed = typed.0.compose(typed.1).unwrap();
    let twisted_contract = typed
        .0
        .contract(&twisted_right, &lhs_axes, &rhs_axes, &[0, 1, 2, 3])
        .unwrap();
    assert_same_legs(&composed.codomain(), &typed.0.codomain());
    assert_same_legs(&composed.domain(), &typed.1.domain());
    assert_eq!(composed.data(), twisted_contract.data(), "{what}: compose");
    assert!(
        std::ptr::eq(composed.provider(), typed.0.provider()),
        "{what}: compose lost left provider authority"
    );
    assert_eq!(
        composed.data() != default_contract.data(),
        nontrivial_twist,
        "{what}: compose twist control"
    );
    assert_nonzero(what, composed.data());

    // Absorb the compact spectrum into U, then use the remaining Vh factor to
    // reconstruct the source. This tests the compact arm without comparing an
    // SVD gauge that is not public API.
    let (typed_u, typed_s, typed_vh) = typed
        .0
        .svd_compact()
        .unwrap_or_else(|error| panic!("{what}: svd_compact failed: {error}"));
    let typed_absorbed = typed_u.contract(&typed_s, &[2], &[0], &[0, 1, 2]).unwrap();
    assert_same_legs(&typed_absorbed.codomain(), &typed_u.codomain());
    assert_same_legs(&typed_absorbed.domain(), &typed_s.domain());
    assert!(
        std::ptr::eq(typed_absorbed.provider(), typed.0.provider()),
        "{what}: absorption lost left provider authority"
    );
    assert_nonzero(what, typed_absorbed.data());
    assert_ne!(
        typed_absorbed.data(),
        typed_u.data(),
        "{what}: the diagonal factor was not applied"
    );
    let reconstructed = typed_absorbed
        .contract(&typed_vh, &[2], &[0], &[0, 1, 2, 3])
        .unwrap();
    assert_same_legs(&reconstructed.codomain(), &typed.0.codomain());
    assert_same_legs(&reconstructed.domain(), &typed.0.domain());
    assert_data_close_f64(reconstructed.data(), typed.0.data());
    assert!(std::ptr::eq(reconstructed.provider(), typed.0.provider()));
}

#[test]
fn contract_compose_and_compact_laws_hold_on_u1() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_a = u1_oracle(&runtime, 1.0);
    let typed_b = u1_oracle(&runtime, 100.0);
    assert_contract_compose_compact_laws_hold(
        "U1, [p, q] <- [p, q] with q dual",
        false,
        (&typed_a, &typed_b),
    );
}

#[test]
fn contract_compose_and_compact_laws_hold_on_u1_fz2() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_a = u1_fz2_oracle(&runtime, 1.0);
    let typed_b = u1_fz2_oracle(&runtime, 100.0);
    assert_contract_compose_compact_laws_hold(
        "U1 x fZ2, [p, q] <- [p, q] with q dual",
        true,
        (&typed_a, &typed_b),
    );
}

#[test]
fn contract_compose_and_compact_laws_hold_on_fz2_u1_su2() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_a = fz2_u1_su2_oracle(&runtime, 1.0);
    let typed_b = fz2_u1_su2_oracle(&runtime, 100.0);
    assert_contract_compose_compact_laws_hold(
        "fZ2 x U1 x SU2, [p, q] <- [p, q] with q dual",
        true,
        (&typed_a, &typed_b),
    );
}

#[test]
fn the_fermionic_product_compose_is_contract_against_a_twisted_right_operand() {
    // What: the exact relation on `fZ2 x U(1) x SU(2)`,
    // `compose(a, b) == contract(a, twist(b, b's dual codomain legs))`. The
    // weaker `contract != compose` would pin only that *a* twist exists, not
    // which legs it acts on nor with which sign; this form pins all three, and
    // it is stated inside the product family so the suite does not lean on the
    // plain-fZ2 test above for it. A byte-parity oracle cannot do this
    // job at all: a twist deleted from a shared kernel moves both buffers
    // together. The bosonic family is the control — theta is one there, so the
    // twisted operand is the operand and the two contractions agree.
    let _guard = cache_lock();
    let runtime = runtime();

    let fermionic_a = fz2_u1_su2_oracle(&runtime, 1.0);
    let fermionic_b = fz2_u1_su2_oracle(&runtime, 100.0);
    // `b`'s codomain is `[p, q]` and only `q` is dual, so theta acts on
    // codomain leg 1 alone — a "twist every contracted leg" reading would
    // multiply leg 0 too and fail here. Theta comes from the rule rather than
    // from an assumed `(-1)^F`: the product's twist is the product of its
    // factors', and hard-coding the parity sign here would quietly assume the
    // U(1) and SU(2) factors contribute one.
    let (leg_p, leg_q) = fz2_u1_su2_typed_legs();
    let rule = leg_p.provider().clone();
    let mut next = 99.0;
    let twisted_b = TensorMap::from_block_fn(
        &runtime,
        [&leg_p, &leg_q],
        [&leg_p, &leg_q],
        |sectors, _| {
            next += 1.0;
            let dual_leg = SectorCodec::encode_sector(&rule, &sectors.codomain_uncoupled()[1])
                .expect("the fixture's own labels encode");
            next * MultiplicityFreeRigidSymbols::twist_scalar(&rule, dual_leg)
        },
    )
    .unwrap();

    assert_eq!(
        fermionic_a.compose(&fermionic_b).unwrap().data(),
        fermionic_a
            .contract(&twisted_b, &[2, 3], &[0, 1], &[0, 1, 2, 3])
            .unwrap()
            .data(),
        "fZ2 x U1 x SU2: compose is contract against the twisted right operand"
    );
    // And the twist is not vacuous: without it the two contractions differ.
    assert_ne!(
        fermionic_a
            .contract(&fermionic_b, &[2, 3], &[0, 1], &[0, 1, 2, 3])
            .unwrap()
            .data(),
        fermionic_a.compose(&fermionic_b).unwrap().data(),
        "fZ2 x U1 x SU2: contract and compose must differ on a dual contracted leg"
    );

    let bosonic_a = u1_oracle(&runtime, 1.0);
    let bosonic_b = u1_oracle(&runtime, 100.0);
    assert_eq!(
        bosonic_a
            .contract(&bosonic_b, &[2, 3], &[0, 1], &[0, 1, 2, 3])
            .unwrap()
            .data(),
        bosonic_a.compose(&bosonic_b).unwrap().data(),
        "U1: a bosonic rule has no twist, so the two agree"
    );
}

/// The reduction and factorization half of the oracle matrix.
///
/// `dimension_weighted` says whether the family has a coupled sector with
/// `dim(c) != 1`: `inner` is TensorKit's quantum-dimension-weighted `dot`, so
/// only there does it differ from a plain sum of squares. Stating it per
/// family rather than deriving it keeps the fixture honest — if a fixture were
/// weakened to spin 0 everywhere, this flag would start lying and the test
/// would fail rather than quietly stop covering the weighted branch.
fn assert_reductions_and_factorizations_hold<R>(
    what: &str,
    dimension_weighted: bool,
    typed: (&TensorMap<R, f64>, &TensorMap<R, f64>),
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    // Neither coefficient is 1 and they differ in sign, so dropping or
    // swapping one moves the buffer.
    let typed_sum = typed.0.add(typed.1, 2.0, -3.0).unwrap();
    for ((&actual, &left), &right) in typed_sum
        .data()
        .iter()
        .zip(typed.0.data())
        .zip(typed.1.data())
    {
        assert_eq!(actual, 2.0 * left - 3.0 * right, "{what}: add");
    }
    assert_nonzero(what, typed_sum.data());
    assert_ne!(
        typed.0.add(typed.1, -3.0, 2.0).unwrap().data(),
        typed_sum.data(),
        "{what}: add is not symmetric in its coefficients"
    );

    let typed_scaled = typed.0.scale(-2.5);
    for (&actual, &source) in typed_scaled.data().iter().zip(typed.0.data()) {
        assert_eq!(actual, -2.5 * source, "{what}: scale");
    }
    assert_nonzero(what, typed_scaled.data());

    let typed_inner = typed.0.inner(typed.1).unwrap();
    let reverse_inner = typed.1.inner(typed.0).unwrap();
    assert!(
        (typed_inner - reverse_inner).abs() < 1e-12 * typed_inner.abs().max(1.0),
        "{what}: inner is not conjugate symmetric"
    );
    assert_ne!(typed_inner, 0.0, "{what}: inner is vacuously zero");
    // `<t, t>` is the squared weighted norm, which is the identity that pins
    // this weighting to `norm`'s.
    let norm = typed.0.norm().unwrap();
    let self_inner = typed.0.inner(typed.0).unwrap();
    assert!(
        (self_inner - norm * norm).abs() < 1e-9 * norm * norm,
        "{what}: <t, t> != norm^2"
    );
    // And the weighting is (or is not) visible against the unweighted sum of
    // squares, per the family.
    let unweighted: f64 = typed.0.data().iter().map(|value| value * value).sum();
    assert_eq!(
        (self_inner - unweighted).abs() > 1e-9 * unweighted,
        dimension_weighted,
        "{what}: dimension weighting expected {dimension_weighted}, \
         <t, t> = {self_inner}, sum of squares = {unweighted}"
    );

    let (typed_q, typed_r) = typed.0.qr_compact().unwrap();
    assert_data_close_f64(typed_q.compose(&typed_r).unwrap().data(), typed.0.data());
    assert!(typed_q.is_isometric(1e-12).unwrap(), "{what}: qr q");
    assert_same_legs(&typed_q.codomain(), &typed.0.codomain());
    assert_same_legs(&typed_r.domain(), &typed.0.domain());
    assert_same_legs(&typed_q.domain(), &typed_r.codomain());
    assert_nonzero(what, typed_q.data());
    assert_nonzero(what, typed_r.data());

    // TensorKit's default left orthogonalization is QR.
    let (typed_v, typed_c) = typed.0.left_orth().unwrap();
    assert_eq!(typed_v.data(), typed_q.data(), "{what}: left_orth is qr");
    assert_eq!(typed_c.data(), typed_r.data(), "{what}: left_orth is qr");
    assert_data_close_f64(typed_v.compose(&typed_c).unwrap().data(), typed.0.data());
    assert!(typed_v.is_isometric(1e-12).unwrap(), "{what}: left orth v");
    assert_same_legs(&typed_v.codomain(), &typed.0.codomain());
    assert_same_legs(&typed_c.domain(), &typed.0.domain());
    assert_same_legs(&typed_v.domain(), &typed_c.codomain());

    let (typed_c, typed_vh) = typed.0.right_orth().unwrap();
    assert_data_close_f64(typed_c.compose(&typed_vh).unwrap().data(), typed.0.data());
    assert!(
        typed_vh.adjoint().unwrap().is_isometric(1e-12).unwrap(),
        "{what}: right orth vh"
    );
    assert_same_legs(&typed_c.codomain(), &typed.0.codomain());
    assert_same_legs(&typed_vh.domain(), &typed.0.domain());
    assert_same_legs(&typed_c.domain(), &typed_vh.codomain());
    assert_nonzero(what, typed_c.data());
    assert_nonzero(what, typed_vh.data());

    let typed_spectrum = typed.0.svd_vals().unwrap();
    assert!(
        typed_spectrum
            .windows(2)
            .all(|pair| pair[0].sector < pair[1].sector),
        "{what}: svd_vals labels are not strictly sorted"
    );
    assert!(
        typed_spectrum.iter().all(|entry| {
            entry.values.iter().all(|value| *value >= 0.0)
                && entry.values.windows(2).all(|pair| pair[0] >= pair[1])
        }),
        "{what}: svd_vals are not nonnegative and descending"
    );
    assert!(
        typed_spectrum
            .iter()
            .any(|entry| entry.values.iter().any(|value| *value != 0.0)),
        "{what}: svd_vals is vacuously zero"
    );
    let codomain = typed.0.codomain();
    let provider = codomain[0].provider();
    let spectral_weight: f64 = typed_spectrum
        .iter()
        .map(|entry| {
            let sector = SectorCodec::encode_sector(provider, &entry.sector).unwrap();
            provider.dim_scalar(sector)
                * entry.values.iter().map(|value| value * value).sum::<f64>()
        })
        .sum();
    assert!(
        (spectral_weight - self_inner).abs() < 1e-9 * self_inner.abs().max(1.0),
        "{what}: weighted sum(sigma^2) = {spectral_weight}, <t, t> = {self_inner}"
    );
}

#[test]
fn reductions_and_factorizations_hold_on_u1() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_a = u1_oracle(&runtime, 1.0);
    let typed_b = u1_oracle(&runtime, 100.0);
    assert_reductions_and_factorizations_hold(
        "U1, [p, q] <- [p, q] with q dual",
        false,
        (&typed_a, &typed_b),
    );
}

#[test]
fn reductions_and_factorizations_hold_on_u1_fz2() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_a = u1_fz2_oracle(&runtime, 1.0);
    let typed_b = u1_fz2_oracle(&runtime, 100.0);
    assert_reductions_and_factorizations_hold(
        "U1 x fZ2, [p, q] <- [p, q] with q dual",
        false,
        (&typed_a, &typed_b),
    );
}

#[test]
fn reductions_and_factorizations_hold_on_fz2_u1_su2() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_a = fz2_u1_su2_oracle(&runtime, 1.0);
    let typed_b = fz2_u1_su2_oracle(&runtime, 100.0);
    // The only weighted family here: its SU(2) factor carries spin 1/2 and
    // spin 1, so `dim(c)` is not identically one.
    assert_reductions_and_factorizations_hold(
        "fZ2 x U1 x SU2, [p, q] <- [p, q] with q dual",
        true,
        (&typed_a, &typed_b),
    );
}

// ---------------------------------------------------------------------------
// Issue #610: the generic product provider as the canonical public route.
//
// Everything below is written the way a downstream user would have to write
// it: `ProductFusionRuleExt::product` for the provider, `product_sector` for
// the label, `tenet::typed` for the space and the tensor. No
// fixed product-space constructor appears, so the tests fail — they do not
// silently reroute — if a group-specific constructor is the only working path.
// ---------------------------------------------------------------------------

/// `fZ2 ⊠ U(1)` built from the generic rule product, then run through a typed
/// operation chain: this is a public-path claim, not a sign or permutation
/// detector. Fermionic sign detection is owned by
/// `the_fermionic_product_compose_is_contract_against_a_twisted_right_operand`
/// earlier in this file.
#[test]
fn generic_product_provider_drives_the_typed_facade_without_a_fixed_constructor() {
    let _guard = cache_lock();
    let runtime = runtime();

    let rule = Arc::new(tenet::core::FermionParityFusionRule.product(tenet::core::U1FusionRule));
    let label = |parity: tenet::core::Z2Irrep, charge: i32| {
        tenet::core::product_sector(parity, tenet::core::U1Irrep::new(charge))
    };

    let p = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (label(tenet::core::Z2Irrep::EVEN, 0), 2),
            (label(tenet::core::Z2Irrep::ODD, 1), 1),
        ],
    )
    .unwrap();
    // Dual, so the dual-flag path — admission, dual sector resolution, block
    // layout, composition — runs on a product provider. The identities below
    // are sign-consistent and hold for either flag, so this leg is coverage of
    // that path, not a probe that makes them sign-sensitive.
    let q = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (label(tenet::core::Z2Irrep::ODD, -1), 1),
            (label(tenet::core::Z2Irrep::EVEN, 0), 2),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();

    // A distinct value per element, so multiple blocks are exercised
    // nontrivially.
    let mut next = 0.0_f64;
    let t: TensorMap<_, f64> = TensorMap::from_block_fn(&runtime, [&p, &q], [&p, &q], |_, _| {
        next += 1.0;
        next
    })
    .unwrap();

    assert!(
        t.block_count() > 1,
        "a single block would make the identities below near-vacuous"
    );
    let norm = t.norm().unwrap();
    assert!(norm > 0.0, "zero tensor: the assertions below are vacuous");

    // <t, t> = |t|^2 and tr(t† ∘ t) = |t|^2: three independent code paths
    // (weighted inner, reduction norm, composition + trace) over the product
    // provider's own fusion and dual data.
    assert!((t.inner(&t).unwrap() - norm * norm).abs() < 1e-9 * norm * norm);
    let gram_trace = t.adjoint().unwrap().compose(&t).unwrap().tr().unwrap();
    assert!((gram_trace - norm * norm).abs() < 1e-9 * norm * norm);
}

/// The recursive three-factor spelling: `(fZ2 ⊠ U(1)) ⊠ SU(2)` needs no new
/// core type, and its factor order and association are structure — of the Rust
/// type, of the label, and of the [`tenet::core::RuleIdentity`] — never an
/// automatic equivalence.
#[test]
fn nested_three_factor_product_keeps_its_declared_factor_order() {
    let _guard = cache_lock();

    let left_assoc = Arc::new(
        tenet::core::FermionParityFusionRule
            .product(tenet::core::U1FusionRule)
            .product(SU2FusionRule),
    );
    let label = |parity: tenet::core::Z2Irrep, charge: i32, twice_spin: usize| {
        tenet::core::product_sector(
            tenet::core::product_sector(parity, tenet::core::U1Irrep::new(charge)),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };

    let declared = [
        (label(tenet::core::Z2Irrep::ODD, 1, 1), 1),
        (label(tenet::core::Z2Irrep::EVEN, 0, 2), 2),
    ];
    let v = GradedSpace::try_new_with_arc(Arc::clone(&left_assoc), declared).unwrap();

    // Decoded labels come back nested exactly as declared: parity outermost
    // left, then charge, with the spin as the outer right factor. Compared as
    // `(label, degeneracy)` pairs — labels alone would not catch a label ↔
    // degeneracy mispairing — and as a set, because a leg is stored in
    // `SectorId` order, not declaration order.
    let mut decoded: Vec<_> = v
        .sectors()
        .unwrap()
        .into_iter()
        .zip(v.degeneracies().iter().copied())
        .collect();
    decoded.sort_unstable();
    let mut expected = declared.to_vec();
    expected.sort_unstable();
    assert_eq!(decoded, expected);
    let odd = decoded
        .iter()
        .map(|(label, _)| label)
        .find(|label| *label.left().left() == tenet::core::Z2Irrep::ODD)
        .expect("the odd-parity label survives the round trip");
    assert_eq!(*odd.left().right(), tenet::core::U1Irrep::new(1));
    assert_eq!(*odd.right(), SU2Irrep::from_twice_spin(1));

    // Association is not an equivalence: `fZ2 ⊠ (U(1) ⊠ SU(2))` is a different
    // provider with a different label type, and the identities do not compare
    // equal. The label type difference is what makes the line below compile
    // only when written for the right association.
    let right_assoc = tenet::core::FermionParityFusionRule
        .product(tenet::core::U1FusionRule.product(SU2FusionRule));
    let right_label = tenet::core::product_sector(
        tenet::core::Z2Irrep::ODD,
        tenet::core::product_sector(tenet::core::U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
    );
    assert_ne!(
        left_assoc.rule_identity(),
        right_assoc.rule_identity(),
        "left- and right-associated products must stay distinct providers"
    );
    assert_eq!(*right_label.left(), tenet::core::Z2Irrep::ODD);

    // Factor order is not an equivalence either: swapping the two factors of
    // the inner product gives a third distinct provider.
    let swapped = tenet::core::U1FusionRule
        .product(tenet::core::FermionParityFusionRule)
        .product(SU2FusionRule);
    assert_ne!(left_assoc.rule_identity(), swapped.rule_identity());
}

// ---------------------------------------------------------------------------
// Slice: typed constructors — `rand`/`rand_with_seed` and the structural
// family (`isomorphism`/`unitary`/`isometry`), issue #580 PR 1.
// ---------------------------------------------------------------------------

fn u1_typed_leg() -> GradedSpace<tenet::core::U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::U1FusionRule),
        [
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 1),
            (tenet::core::U1Irrep::new(-1), 3),
        ],
    )
    .unwrap()
}

fn fz2_typed_leg() -> GradedSpace<tenet::core::FermionParityFusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::FermionParityFusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap()
}

#[test]
fn typed_rand_with_seed_is_deterministic_f64() {
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = u1_typed_leg();
    let first: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 7).unwrap();
    let replay: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 7).unwrap();
    let other: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 8).unwrap();

    assert!(!first.data().is_empty());
    assert_eq!(first.data(), replay.data());
    assert_ne!(first.data(), other.data());
}

#[test]
fn typed_rand_with_seed_is_deterministic_c64() {
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = fz2_typed_leg();
    let first: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 11).unwrap();
    let replay: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 11).unwrap();
    let other: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 12).unwrap();

    assert!(first.data().iter().any(|value| value.im != 0.0));
    assert_eq!(first.data(), replay.data());
    assert_ne!(first.data(), other.data());
}

#[test]
fn rand_and_isomorphism_build_on_an_external_provider() {
    // What: nothing in the new constructors needs a built-in rule — a
    // downstream Z3 provider gets a populated random tensor and a lawful
    // structural isomorphism from the same public vocabulary.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let leg = z3_leg(&provider, false);
    let dual = leg.try_dual().unwrap();

    let random: TensorMap<ExternalZ3, f64> = TensorMap::rand(&runtime, [&leg], [&leg]).unwrap();
    assert!(random.data().iter().any(|&value| value != 0.0));

    // The fused non-dual isomorph of the dual leg has the same sector content.
    let f: TensorMap<ExternalZ3, f64> =
        TensorMap::isomorphism(&runtime, [&leg, &dual], [&dual, &leg]).unwrap();
    let roundtrip = f.adjoint().unwrap().compose(&f).unwrap();
    // Explicit `D`: under cuda,cpu-faer serde_json adds a `PartialEq` impl
    // that makes the `assert_eq!` unable to pin `D` on its own (E0283).
    let id: TensorMap<ExternalZ3, f64> = TensorMap::id(&runtime, [&dual, &leg]).unwrap();
    assert_eq!(roundtrip.data(), id.data());
}

#[test]
fn typed_isomorphism_satisfies_the_identity_law_on_a_builtin_rule() {
    // What: `f† ∘ f = id` on the domain product, on a rank-2 <- rank-2
    // isomorphism whose two sides carry the same fused content through
    // opposite dual arrangements ([dual, leg] <- [leg, dual]). The norm-fuser
    // shape (single fused codomain leg) is exercised by the byte-parity test
    // below.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = u1_typed_leg();
    let dual = leg.try_dual().unwrap();

    let f: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::isomorphism(&runtime, [&dual, &leg], [&leg, &dual]).unwrap();
    let roundtrip = f.adjoint().unwrap().compose(&f).unwrap();
    // Explicit `D` for the cuda,cpu-faer feature set (see the ExternalZ3
    // identity-law test above).
    let id: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::id(&runtime, [&leg, &dual]).unwrap();
    assert_eq!(roundtrip.data(), id.data());
}

#[test]
fn typed_unitary_byte_equals_isomorphism() {
    // What: `unitary` is `isomorphism` plus TensorKit's Euclidean
    // inner-product check, which every tenet fusion rule satisfies, so the two
    // typed constructors are byte-identical here.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = fz2_typed_leg();

    let u: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::unitary(&runtime, [&leg], [&leg]).unwrap();
    let f: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::isomorphism(&runtime, [&leg], [&leg]).unwrap();
    assert_eq!(u.data(), f.data());
}

#[test]
fn typed_isometry_embeds_and_satisfies_the_identity_law() {
    // What: on a strictly-embedding fixture (every domain sector strictly
    // smaller than its codomain sibling in at least one sector) the isometry
    // satisfies `w† ∘ w = id(domain)`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let small = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let big = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 3),
            (tenet::core::U1Irrep::new(-1), 1),
        ],
    )
    .unwrap();

    let w: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::isometry(&runtime, [&big], [&small]).unwrap();
    let roundtrip = w.adjoint().unwrap().compose(&w).unwrap();
    // Explicit `D` for the cuda,cpu-faer feature set (see the ExternalZ3
    // identity-law test above).
    let id: TensorMap<tenet::core::U1FusionRule, f64> = TensorMap::id(&runtime, [&small]).unwrap();
    assert_eq!(roundtrip.data(), id.data());
}

#[test]
fn typed_isometry_rejects_a_non_embeddable_pair() {
    let _guard = cache_lock();
    let runtime = runtime();
    let small = u1_typed_leg();
    let big = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::U1FusionRule),
        [(tenet::core::U1Irrep::new(0), 1)],
    )
    .unwrap();

    // Domain strictly larger than the codomain: not embeddable.
    let typed_error =
        TensorMap::<tenet::core::U1FusionRule, f64>::isometry(&runtime, [&big], [&small])
            .unwrap_err();

    assert!(matches!(
        typed_error,
        tenet::prelude::Error::InvalidArgument(_)
    ));
}

#[test]
fn typed_structural_constructors_reject_an_empty_leg_list() {
    // What: the provider is inferred from the legs, so an empty construction
    // has nothing to infer it from — same class as `zeros`.
    let runtime = runtime();
    let empty: [&GradedSpace<ExternalZ3>; 0] = [];

    assert!(TensorMap::<ExternalZ3, f64>::rand(&runtime, empty, empty).is_err());
    assert!(TensorMap::<ExternalZ3, f64>::isomorphism(&runtime, empty, empty).is_err());
    assert!(TensorMap::<ExternalZ3, f64>::isometry(&runtime, empty, empty).is_err());
}

#[test]
fn typed_rand_rejects_mismatched_providers_like_zeros() {
    // What: two providers with distinct rule identities are a rule mismatch on
    // the random constructor exactly as on `zeros`.
    let _guard = cache_lock();
    let first = Arc::new(ExternalZ3::tagged(0));
    let second = Arc::new(ExternalZ3::tagged(1));
    let runtime = runtime();

    let error = TensorMap::<ExternalZ3, f64>::rand(
        &runtime,
        [&z3_leg(&first, false)],
        [&z3_leg(&second, true)],
    )
    .unwrap_err();
    assert!(matches!(error, tenet::prelude::Error::RuleMismatch));
}

#[test]
fn typed_seedless_rand_does_not_advance_the_stream_on_failure() {
    // What: a failing seedless `rand` must not consume a position in the
    // runtime's deterministic random stream. Twin runtimes: on A a failing
    // call precedes the valid one; on B only the valid call runs. The two
    // valid results must agree byte for byte.
    let _guard = cache_lock();
    let runtime_a = runtime();
    let runtime_b = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let leg = z3_leg(&provider, false);
    let empty: [&GradedSpace<ExternalZ3>; 0] = [];

    // Fails after provider inference but before any fill: no legs at all.
    assert!(TensorMap::<ExternalZ3, f64>::rand(&runtime_a, empty, empty).is_err());
    // Fails during space validation: mismatched rule identities.
    let other = Arc::new(ExternalZ3::tagged(9));
    assert!(
        TensorMap::<ExternalZ3, f64>::rand(&runtime_a, [&leg], [&z3_leg(&other, true)]).is_err()
    );

    let on_a: TensorMap<ExternalZ3, f64> = TensorMap::rand(&runtime_a, [&leg], [&leg]).unwrap();
    let on_b: TensorMap<ExternalZ3, f64> = TensorMap::rand(&runtime_b, [&leg], [&leg]).unwrap();
    assert_eq!(on_a.data(), on_b.data());
}

#[test]
fn typed_isomorphism_and_unitary_reject_embeddable_but_not_isomorphic_content() {
    // What: the isomorphism gate is `domain ≅ codomain`, strictly stronger
    // than the isometry embedding — a domain that embeds but is smaller must
    // be rejected by both `isomorphism` and `unitary`. The
    // embeddable-but-not-isomorphic fixture is deliberate: it also proves
    // `unitary` routes through the isomorphism check, not the isometry one.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let small = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let big = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 3),
        ],
    )
    .unwrap();

    for typed_error in [
        TensorMap::<tenet::core::U1FusionRule, f64>::isomorphism(&runtime, [&big], [&small])
            .unwrap_err(),
        TensorMap::<tenet::core::U1FusionRule, f64>::unitary(&runtime, [&big], [&small])
            .unwrap_err(),
    ] {
        assert!(matches!(
            typed_error,
            tenet::prelude::Error::InvalidArgument(_)
        ));
    }
}

#[test]
fn typed_isometry_rejects_a_larger_domain_degeneracy_with_identical_sector_sets() {
    // What: the embedding check is sectorwise `deg_domain <= deg_codomain`,
    // not sector-set containment — identical sector sets with one domain
    // degeneracy exactly one above its codomain sibling must fail. The
    // off-by-one fixture makes a `deg + 1` slip in the
    // comparison visible.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let codomain = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let domain = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();

    let typed_error =
        TensorMap::<tenet::core::U1FusionRule, f64>::isometry(&runtime, [&codomain], [&domain])
            .unwrap_err();
    assert!(matches!(
        typed_error,
        tenet::prelude::Error::InvalidArgument(_)
    ));
}

#[test]
fn typed_isomorphism_is_unitary_on_the_norm_fuser_shape() {
    // What: the norm-fuser shape carries an asymmetric dual arrangement and
    // still satisfies the defining isomorphism law.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let typed_v = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let typed_dual = typed_v.try_dual().unwrap();
    // fuse(dual(v) ⊗ v) by hand: charges -1, 0 (twice), 1.
    let typed_fused = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(-1), 1),
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let typed: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::isomorphism(&runtime, [&typed_fused], [&typed_dual, &typed_v]).unwrap();
    let identity: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::id(&runtime, [&typed_dual, &typed_v]).unwrap();

    assert_eq!(
        typed.adjoint().unwrap().compose(&typed).unwrap().data(),
        identity.data()
    );
}

#[test]
fn typed_c64_unitary_satisfies_the_identity_law() {
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = fz2_typed_leg();
    let leg_dual = leg.try_dual().unwrap();
    let typed: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::unitary(&runtime, [&leg, &leg_dual], [&leg_dual, &leg]).unwrap();
    let identity: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::id(&runtime, [&leg_dual, &leg]).unwrap();

    assert_eq!(
        typed.adjoint().unwrap().compose(&typed).unwrap().data(),
        identity.data()
    );
}

#[test]
fn typed_isometry_on_a_dual_domain_satisfies_the_identity_law() {
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let small = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let big = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::U1Irrep::new(0), 2),
            (tenet::core::U1Irrep::new(1), 3),
            (tenet::core::U1Irrep::new(-1), 3),
        ],
    )
    .unwrap();

    let typed: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::isometry(&runtime, [&big], [&small]).unwrap();
    let identity: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::id(&runtime, [&small]).unwrap();

    assert_eq!(
        typed.adjoint().unwrap().compose(&typed).unwrap().data(),
        identity.data()
    );
}

#[test]
fn typed_su2_isomorphism_satisfies_the_identity_law() {
    // What: nothing in the structural constructors is abelian-specific — a
    // simple-fusion SU(2) provider's isomorphism still satisfies `f† ∘ f =
    // id` through genuinely non-trivial recoupling.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);

    let f: TensorMap<ExternalSu2, f64> =
        TensorMap::isomorphism(&runtime, [&leg, &leg], [&leg, &leg]).unwrap();
    let roundtrip = f.adjoint().unwrap().compose(&f).unwrap();
    // Explicit `D` for the cuda,cpu-faer feature set (see the ExternalZ3
    // identity-law test above).
    let id: TensorMap<ExternalSu2, f64> = TensorMap::id(&runtime, [&leg, &leg]).unwrap();
    assert_eq!(roundtrip.data(), id.data());
}

// ---------------------------------------------------------------------------
// Slice: typed `left_polar` / `right_polar`, issue #580 PR 2.
// ---------------------------------------------------------------------------

/// Elementwise closeness at factorization tolerance, relative to the wanted
/// entry's magnitude (floored at 1).
fn assert_data_close_f64(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-12 * w.abs().max(1.0), "{g} vs {w}");
    }
}

/// The c64 sibling of [`assert_data_close_f64`].
fn assert_data_close_c64(got: &[Complex64], want: &[Complex64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).norm() <= 1e-12 * w.norm().max(1.0), "{g} vs {w}");
    }
}

/// Content-wise leg equality: `GradedSpace` has no `PartialEq` (deliberate,
/// see its rustdoc), so space equality is asserted through the public
/// accessors — labels, degeneracies and duality all have to agree.
fn assert_same_legs<R>(got: &[GradedSpace<R>], want: &[GradedSpace<R>])
where
    R: SectorCodec + CheckedFusionAlgebra,
    R::Sector: std::fmt::Debug + PartialEq,
{
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.sectors().unwrap(), w.sectors().unwrap());
        assert_eq!(g.degeneracies(), w.degeneracies());
        assert_eq!(g.is_dual(), w.is_dual());
    }
}

#[test]
fn typed_polar_reconstructs_the_input_f64_u1_and_c64_fz2() {
    // Gate 1: `t = w ∘ p` (left) and `t = p ∘ w` (right) at factorization
    // tolerance, on both dtypes and on both a bosonic and a fermionic rule.
    let _guard = cache_lock();
    let runtime = runtime();

    let leg = u1_typed_leg();
    let tall: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 3).unwrap();
    let (w, p) = tall.left_polar().unwrap();
    assert_data_close_f64(w.compose(&p).unwrap().data(), tall.data());
    let wide: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg, &leg], 5).unwrap();
    let (p, w) = wide.right_polar().unwrap();
    assert_data_close_f64(p.compose(&w).unwrap().data(), wide.data());

    let leg = fz2_typed_leg();
    let tall: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 7).unwrap();
    let (w, p) = tall.left_polar().unwrap();
    assert_data_close_c64(w.compose(&p).unwrap().data(), tall.data());
    let wide: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg, &leg], 11).unwrap();
    let (p, w) = wide.right_polar().unwrap();
    assert_data_close_c64(p.compose(&w).unwrap().data(), wide.data());
}

#[test]
fn typed_polar_factor_laws_hold() {
    // Gate 2: `w† ∘ w = id(domain)` for `left_polar` (resp. `w ∘ w† = id` on
    // the rows for `right_polar`); `p` Hermitian with non-negative spectrum,
    // read through `eigh_vals`.
    let _guard = cache_lock();
    let runtime = runtime();

    // c64 fermionic left arm: a stray conjugation or sign is visible here.
    let leg = fz2_typed_leg();
    let tall: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 13).unwrap();
    let (w, p) = tall.left_polar().unwrap();
    let id: TensorMap<tenet::core::FermionParityFusionRule, Complex64> =
        TensorMap::id(&runtime, [&leg]).unwrap();
    assert_data_close_c64(w.adjoint().unwrap().compose(&w).unwrap().data(), id.data());
    assert!(p.is_hermitian(1e-12).unwrap());
    for entry in p.eigh_vals().unwrap() {
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }

    // f64 U(1) right arm.
    let leg = u1_typed_leg();
    let wide: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg, &leg], 17).unwrap();
    let (p, w) = wide.right_polar().unwrap();
    let id: TensorMap<tenet::core::U1FusionRule, f64> = TensorMap::id(&runtime, [&leg]).unwrap();
    assert_data_close_f64(w.compose(&w.adjoint().unwrap()).unwrap().data(), id.data());
    assert!(p.is_hermitian(1e-12).unwrap());
    for entry in p.eigh_vals().unwrap() {
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }
}

#[test]
fn typed_polar_factor_spaces_match_tensorkit() {
    // Gate 3: factor *spaces*, not just shapes — TK 0.17
    // `factorizations/matrixalgebrakit.jl:204-214`: left `W` lives on
    // `space(t)` and `P` on `domain ← domain`; right `P` on
    // `codomain ← codomain` and `Wᴴ` on `space(t)`. A dual leg sits on each
    // side so a dropped duality flag is visible.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = u1_typed_leg();
    let dual = leg.try_dual().unwrap();

    let tall: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &dual], [&dual], 19).unwrap();
    let (w, p) = tall.left_polar().unwrap();
    assert_same_legs(&w.codomain(), &tall.codomain());
    assert_same_legs(&w.domain(), &tall.domain());
    assert_same_legs(&p.codomain(), &tall.domain());
    assert_same_legs(&p.domain(), &tall.domain());

    let wide: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&dual], [&leg, &dual], 23).unwrap();
    let (p, w) = wide.right_polar().unwrap();
    assert_same_legs(&p.codomain(), &wide.codomain());
    assert_same_legs(&p.domain(), &wide.codomain());
    assert_same_legs(&w.codomain(), &wide.codomain());
    assert_same_legs(&w.domain(), &wide.domain());
}

#[test]
fn typed_polar_wrong_side_rectangular_reports_the_requested_direction() {
    // Gate 4: the split-2 fixture is tall in every coupled sector, the split-1
    // one wide — so `right_polar` on the former and `left_polar` on the latter
    // return the seam's exact wrong-side errors, unfiltered.
    let _guard = cache_lock();
    let runtime = runtime();

    let typed_tall = z2_tensor_split(&runtime, 2);
    assert!(matches!(
        typed_tall.right_polar().unwrap_err(),
        tenet::typed::Error::Operation(error)
            if matches!(
                error.as_ref(),
                tenet::operations::OperationError::InvalidArgument { message }
                    if *message
                        == "right_polar requires columns >= rows in every coupled-sector matrix"
            )
    ));

    let typed_wide = z2_tensor_split(&runtime, 1);
    assert!(matches!(
        typed_wide.left_polar().unwrap_err(),
        tenet::typed::Error::Operation(error)
            if matches!(
                error.as_ref(),
                tenet::operations::OperationError::InvalidArgument { message }
                    if *message
                        == "left_polar requires rows >= columns in every coupled-sector matrix"
            )
    ));
}

#[test]
fn typed_polar_carries_an_external_provider() {
    // Gate 5: `ExternalZ3` is a downstream-style provider, so this is a
    // provider-generic law check —
    // reconstruction plus the isometry law — driving the same context lane an
    // external provider reaches through `multiplicity_free_lane`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    // Same legs on both sides (no duals): every coupled-sector matrix is
    // square, so both polars are defined. `z3_rank_four`'s dual domain is not
    // usable here — a dual flips each charge, so some coupled sectors come out
    // wider than tall and `left_polar` rightly refuses them.
    let wide = z3_leg(&provider, false);
    let narrow = z3_other_leg(&provider, false);
    let mut counter = 0.0;
    let tensor: TensorMap<ExternalZ3, f64> =
        TensorMap::from_block_fn(&runtime, [&wide, &narrow], [&wide, &narrow], |_, _| {
            counter += 1.0;
            counter
        })
        .unwrap();

    let (w, p) = tensor.left_polar().unwrap();
    assert_data_close_f64(w.compose(&p).unwrap().data(), tensor.data());
    let gram = w.adjoint().unwrap().compose(&w).unwrap();
    let id: TensorMap<ExternalZ3, f64> = TensorMap::id(&runtime, [&wide, &narrow]).unwrap();
    assert_data_close_f64(gram.data(), id.data());
}

// ---------------------------------------------------------------------------
// Slice: typed inspection, scalar, zeros_like and dtype conversions,
// issue #580 PR 3.
// ---------------------------------------------------------------------------

#[test]
#[allow(deprecated)]
fn typed_rank_accessors_pin_the_aliases() {
    // Gate 1: ranks on a mixed 2 <- 1 fixture and the TensorKit-named aliases.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    assert_eq!(typed.codomain_rank(), 2);
    assert_eq!(typed.domain_rank(), 1);
    assert_eq!(typed.rank(), 3);

    assert_eq!(typed.numout(), typed.codomain_rank());
    assert_eq!(typed.numin(), typed.domain_rank());
    assert_eq!(typed.numind(), typed.rank());
}

#[test]
fn typed_leg_dims_include_quantum_dimensions() {
    // Gate 2: `leg_dims`/`leg_dim` values on a built-in abelian rule and on
    // built-in SU(2), whose non-abelian sectors
    // are what make the quantum-dimension weighting visible (spin-1/2 with
    // degeneracy 2 contributes 4, not 2).
    let _guard = cache_lock();
    let runtime = runtime();

    let typed = z2_tensor(&runtime);
    assert_eq!(typed.leg_dims().unwrap(), vec![5, 5, 5]);
    for axis in 0..typed.rank() {
        assert_eq!(typed.leg_dim(axis).unwrap(), 5);
    }

    let su2_typed = su2_tensor(&runtime);
    assert_eq!(su2_typed.leg_dims().unwrap(), vec![5, 5]);
    assert_eq!(su2_typed.leg_dim(1).unwrap(), 5);
}

#[test]
fn typed_leg_dims_carry_an_external_provider_with_nontrivial_dimensions() {
    // Gate 2's external-provider leg: `ExternalSu2` reports quantum dimension
    // 2 for twice-spin 1, so the degeneracy-2 leg weighs 4. This pins an
    // external provider rather than a built-in special case.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);
    let tensor: TensorMap<ExternalSu2, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();

    assert_eq!(tensor.leg_dims().unwrap(), vec![4, 4]);
    assert_eq!(tensor.leg_dim(0).unwrap(), 4);
    assert_eq!(tensor.leg_dim(1).unwrap(), 4);
}

#[test]
fn typed_leg_dim_out_of_range_is_an_invalid_argument() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let typed_error = typed.leg_dim(3).unwrap_err();
    assert!(matches!(
        typed_error,
        tenet::prelude::Error::InvalidArgument(_)
    ));
}

#[test]
#[allow(deprecated)]
fn typed_codomain_and_domain_spaces_alias_the_primary_accessors() {
    // Gate 3: the compatibility names are documented aliases of
    // `codomain()`/`domain()` — same legs, content-wise.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    assert_same_legs(&typed.codomain_spaces(), &typed.codomain());
    assert_same_legs(&typed.domain_spaces(), &typed.domain());
    assert_eq!(typed.codomain_spaces().len(), 2);
    assert_eq!(typed.domain_spaces().len(), 1);
}

#[test]
fn typed_scalar_reads_a_real_full_contraction() {
    // Gate 4: a rank-0 trace reads back as the payload type itself.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    let typed_value: f64 = typed.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
    assert_eq!(typed_value, typed.tr().unwrap());
}

#[test]
fn typed_scalar_reads_a_complex_full_contraction() {
    // Gate 4's c64 leg: a complex endomorphism traced to rank 0.
    let _guard = cache_lock();
    let runtime = runtime();
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed: TensorMap<tenet::core::Z2FusionRule, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |sectors, indices| {
            complex(typed_fill_value(sectors, indices))
        })
        .unwrap();

    let typed_value: Complex64 = typed.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
    assert_eq!(typed_value, typed.tr().unwrap());
}

#[test]
fn typed_scalar_on_a_tensor_with_legs_is_an_invalid_argument() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let typed_error = typed.scalar().unwrap_err();
    assert!(matches!(
        typed_error,
        tenet::prelude::Error::InvalidArgument(_)
    ));
}

#[test]
fn typed_zeros_like_keeps_the_spaces_and_zeroes_the_payload() {
    // Gate 5: same legs, all-zero payload, dtype preserved statically by the
    // annotated binding. The compact-payload behavior is pinned by the
    // allocation gates in `typed_diagonal_allocations.rs`.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let zeros: TensorMap<tenet::core::Z2FusionRule, f64> = typed.zeros_like();
    assert_same_legs(&zeros.codomain(), &typed.codomain());
    assert_same_legs(&zeros.domain(), &typed.domain());
    assert_eq!(zeros.data().len(), typed.data().len());
    assert!(zeros.data().iter().all(|&value| value == 0.0));

    // A compact spectrum factor stays on its bond space with a zero spectrum.
    let s: TensorMap<tenet::core::Z2FusionRule, f64> = typed.svd_compact().unwrap().1;
    let s_zeros: TensorMap<tenet::core::Z2FusionRule, f64> = s.zeros_like();
    assert_same_legs(&s_zeros.codomain(), &s.codomain());
    assert!(s_zeros.data().iter().all(|&value| value == 0.0));
}

#[test]
fn typed_to_c64_widens_dense_and_compact_values_exactly() {
    // Gate 6: widening adds an exactly zero imaginary component on dense and
    // compact diagonal payloads.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let typed_wide: TensorMap<tenet::core::Z2FusionRule, Complex64> = typed.to_c64();
    assert!(typed_wide
        .data()
        .iter()
        .zip(typed.data())
        .all(|(&wide, &real)| wide == Complex64::new(real, 0.0)));

    let typed_s: TensorMap<tenet::core::Z2FusionRule, f64> = typed.svd_compact().unwrap().1;
    let typed_s_wide: TensorMap<tenet::core::Z2FusionRule, Complex64> = typed_s.to_c64();
    assert!(typed_s_wide
        .data()
        .iter()
        .zip(typed_s.data())
        .all(|(&wide, &real)| wide == Complex64::new(real, 0.0)));
}

#[test]
fn typed_re_im_reconstruct_the_complex_tensor_byte_exactly() {
    // Gate 6's law checks: `re(t) + i*im(t)` rebuilds `t` byte-exactly through
    // `to_c64` + `add`,
    // and on a real tensor `re(to_c64(x)) == x`, `im(to_c64(x)) ==
    // zeros_like(x)`, byte-exactly.
    let _guard = cache_lock();
    let runtime = runtime();
    let complex_typed = z2_complex_tensor(&runtime);

    let real_part: TensorMap<tenet::core::Z2FusionRule, f64> = complex_typed.re();
    let imag_part: TensorMap<tenet::core::Z2FusionRule, f64> = complex_typed.im();
    let rebuilt: TensorMap<tenet::core::Z2FusionRule, Complex64> = real_part
        .to_c64()
        .add(
            &imag_part.to_c64(),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        )
        .unwrap();
    assert_eq!(rebuilt.data(), complex_typed.data());

    let real_typed = z2_tensor(&runtime);
    let round_trip: TensorMap<tenet::core::Z2FusionRule, f64> = real_typed.to_c64().re();
    assert_eq!(round_trip.data(), real_typed.data());
    let vanished: TensorMap<tenet::core::Z2FusionRule, f64> = real_typed.to_c64().im();
    assert_eq!(vanished.data(), real_typed.zeros_like().data());
}

#[test]
fn typed_re_im_keep_a_compact_spectrum_on_its_bond_space() {
    // Gate 6's diagonal leg: `re`/`im` of a complex compact spectrum factor
    // map spectrum-to-spectrum, and the law `re + i*im == t` holds on the
    // materialized payloads.
    let _guard = cache_lock();
    let runtime = runtime();
    let complex_typed = z2_complex_tensor(&runtime);
    let s: TensorMap<tenet::core::Z2FusionRule, Complex64> = complex_typed.svd_compact().unwrap().1;

    let real_part: TensorMap<tenet::core::Z2FusionRule, f64> = s.re();
    let imag_part: TensorMap<tenet::core::Z2FusionRule, f64> = s.im();
    assert_same_legs(&real_part.codomain(), &s.codomain());
    let rebuilt: TensorMap<tenet::core::Z2FusionRule, Complex64> = real_part
        .to_c64()
        .add(
            &imag_part.to_c64(),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        )
        .unwrap();
    assert_eq!(rebuilt.data(), s.data());
}

#[test]
fn typed_leg_dim_routes_each_axis_to_its_own_leg() {
    // Gate 2's axis-routing leg: every earlier fixture has homogeneous leg
    // dimensions, so a leg_dim that reads the wrong leg (e.g. domain[0] for
    // the last codomain axis) still passes them. Three pairwise-distinct
    // dimensions in a 2 <- 1 split make any misrouting visible, per axis.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::Z2FusionRule);
    let z2_pairs = |even: usize, odd: usize| {
        [
            (tenet::core::Z2Irrep::EVEN, even),
            (tenet::core::Z2Irrep::ODD, odd),
        ]
    };
    let a = GradedSpace::try_new_with_arc(Arc::clone(&provider), z2_pairs(2, 3)).unwrap();
    let b = GradedSpace::try_new_with_arc(Arc::clone(&provider), z2_pairs(1, 1)).unwrap();
    let c = GradedSpace::try_new_with_arc(Arc::clone(&provider), z2_pairs(3, 4)).unwrap();
    let typed: TensorMap<tenet::core::Z2FusionRule, f64> =
        TensorMap::zeros(&runtime, [&a, &b], [&c]).unwrap();
    let dims = typed.leg_dims().unwrap();
    assert_eq!(dims, vec![5, 2, 7]);
    for (axis, &dim) in dims.iter().enumerate() {
        assert_eq!(typed.leg_dim(axis).unwrap(), dim);
    }
}

// ---------------------------------------------------------------------------
// #580 PR 4: typed catdomain / catcodomain / absorb.
// ---------------------------------------------------------------------------

/// Position-weighted value from typed U(1) labels.
fn u1_typed_fill(
    sectors: &tenet::typed::BlockFusionTrees<tenet::core::U1Irrep>,
    indices: &[usize],
) -> f64 {
    let mut value = f64::from(sectors.coupled().charge()) * 1000.0;
    for (position, label) in sectors.codomain_uncoupled().iter().enumerate() {
        value += f64::from(label.charge()) * 100.0 * (position + 1) as f64;
    }
    for (position, label) in sectors.domain_uncoupled().iter().enumerate() {
        value += f64::from(label.charge()) * 10.0 * (position + 1) as f64;
    }
    value
        + indices
            .iter()
            .enumerate()
            .map(|(a, &i)| (a + 1) * i)
            .sum::<usize>() as f64
}

/// Position-weighted value from typed fZ2 labels.
fn fz2_typed_fill(
    sectors: &tenet::typed::BlockFusionTrees<tenet::core::Z2Irrep>,
    indices: &[usize],
) -> f64 {
    let mut value = f64::from(sectors.coupled().parity()) * 1000.0;
    for (position, label) in sectors.codomain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 100.0 * (position + 1) as f64;
    }
    for (position, label) in sectors.domain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 10.0 * (position + 1) as f64;
    }
    value
        + indices
            .iter()
            .enumerate()
            .map(|(a, &i)| (a + 1) * i)
            .sum::<usize>() as f64
}

fn u1_leg(
    provider: &Arc<tenet::core::U1FusionRule>,
    pairs: &[(i32, usize)],
) -> GradedSpace<tenet::core::U1FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (tenet::core::U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

/// One typed U(1) cat operand: rank `2 <- 1` with a dual codomain leg.
fn u1_cat_tensor(
    runtime: &Runtime,
    domain_pairs: &[(i32, usize)],
) -> TensorMap<tenet::core::U1FusionRule, f64> {
    let provider = Arc::new(tenet::core::U1FusionRule);
    let l1 = u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let l2 = u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap();
    let lv = u1_leg(&provider, domain_pairs);
    TensorMap::from_block_fn(runtime, [&l1, &l2], [&lv], u1_typed_fill).unwrap()
}

#[test]
fn typed_cat_preserves_a_dual_changed_leg_and_its_slabs() {
    // What: the typed direct sum preserves the changed leg's duality, sector
    // union, degeneracies, and lhs-then-rhs slab order directly.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::FermionParityFusionRule);
    let lw = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let build = |pairs: &[(u8, usize)]| {
        let leg = GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            pairs.iter().map(|&(parity, degeneracy)| {
                (
                    if parity == 0 {
                        tenet::core::Z2Irrep::EVEN
                    } else {
                        tenet::core::Z2Irrep::ODD
                    },
                    degeneracy,
                )
            }),
        )
        .unwrap()
        .try_dual()
        .unwrap();
        TensorMap::from_block_fn(&runtime, [&lw], [&leg], |sectors, indices| {
            10.0 * f64::from(sectors.coupled().parity()) + indices[1] as f64
        })
        .unwrap()
    };
    let typed_lhs = build(&[(0, 2)]);
    let typed_rhs = build(&[(0, 1), (1, 2)]);
    let typed_joined: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed_lhs.catdomain(&typed_rhs).unwrap();
    let changed = &typed_joined.domain()[0];
    assert!(changed.is_dual());
    assert_eq!(
        changed.sectors().unwrap(),
        vec![tenet::core::Z2Irrep::EVEN, tenet::core::Z2Irrep::ODD]
    );
    assert_eq!(changed.degeneracies(), &[3, 2]);
    assert_eq!(typed_joined.data(), &[0.0, 1.0, 0.0, 10.0, 11.0]);
}

#[test]
fn typed_cat_pins_the_slab_order_by_value() {
    // What (gate 2): hand-computed payloads pin adjacent column slabs for
    // catdomain and adjacent
    // row slabs for catcodomain — so the slab order is pinned by value, not
    // only by structural parity.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let w = u1_leg(&provider, &[(0, 2)]);
    let v1 = u1_leg(&provider, &[(0, 1)]);
    let v2 = u1_leg(&provider, &[(0, 2)]);

    let a: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&w], [&v1], |_, i| (i[0] + 1) as f64).unwrap();
    let b: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&w], [&v2], |_, i| (i[0] + 2 * i[1] + 3) as f64)
            .unwrap();
    let joined: TensorMap<tenet::core::U1FusionRule, f64> = a.catdomain(&b).unwrap();
    // Column-major: lhs column [1, 2], then rhs columns [3, 4] and [5, 6].
    assert_eq!(joined.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let at: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&v1], [&w], |_, i| (i[1] + 1) as f64).unwrap();
    let bt: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&v2], [&w], |_, i| (i[0] + 2 * i[1] + 3) as f64)
            .unwrap();
    let stacked: TensorMap<tenet::core::U1FusionRule, f64> = at.catcodomain(&bt).unwrap();
    // Row slabs: lhs row first within each column.
    assert_eq!(stacked.data(), &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
}

#[test]
fn typed_absorb_pins_the_common_prefix_by_value() {
    // What (gate 3): absorb's common-prefix semantics against a hand-computed
    // payload. Destination block is 2x3, source block is 3x2; the shared
    // prefix is 2x2, so exactly those four column-major entries change.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let destination: TensorMap<tenet::core::U1FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [&u1_leg(&provider, &[(0, 2)])],
        [&u1_leg(&provider, &[(0, 3)])],
        |_, i| (10 * (i[0] + 1) + i[1] + 1) as f64,
    )
    .unwrap();
    let source: TensorMap<tenet::core::U1FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [&u1_leg(&provider, &[(0, 3)])],
        [&u1_leg(&provider, &[(0, 2)])],
        |_, i| -((10 * (i[0] + 1) + i[1] + 1) as f64),
    )
    .unwrap();
    // destination (column-major 2x3): [11, 21, 12, 22, 13, 23]
    // source (column-major 3x2): [-11, -21, -31, -12, -22, -32]
    let absorbed: TensorMap<tenet::core::U1FusionRule, f64> = destination.absorb(&source).unwrap();
    assert_eq!(absorbed.data(), &[-11.0, -21.0, -12.0, -22.0, 13.0, 23.0]);
}

#[test]
fn typed_absorb_is_total_for_disjoint_zero_extent_and_rank_zero() {
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let zero = u1_leg(&provider, &[(0, 2)]);
    let one = u1_leg(&provider, &[(1, 3)]);
    let destination: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&zero], [&zero], |_, i| (i[0] + 10 * i[1]) as f64)
            .unwrap();
    let disjoint: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&one], [&one], |_, _| 7.0).unwrap();
    assert_eq!(
        destination.absorb(&disjoint).unwrap().data(),
        destination.data()
    );

    let zero_extent = u1_leg(&provider, &[(0, 0)]);
    let empty: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::zeros(&runtime, [&zero_extent], [&zero_extent]).unwrap();
    assert!(empty.absorb(&destination).unwrap().data().is_empty());

    let vector: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&zero], [], |_, _| 2.0).unwrap();
    let covector: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [], [&zero], |_, _| 3.0).unwrap();
    let source_vector: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&zero], [], |_, _| 5.0).unwrap();
    let source_covector: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [], [&zero], |_, _| 7.0).unwrap();
    let scalar_destination = covector.compose(&vector).unwrap();
    let scalar_source = source_covector.compose(&source_vector).unwrap();
    assert_eq!(
        scalar_destination
            .absorb(&scalar_source)
            .unwrap()
            .scalar()
            .unwrap(),
        scalar_source.scalar().unwrap()
    );
}

#[test]
fn typed_cat_and_absorb_validation_and_precedence_are_stable() {
    // What: each operation keeps its own validation classes and order.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_lhs = u1_cat_tensor(&runtime, &[(0, 1), (1, 1)]);
    let assert_invalid = |error: tenet::prelude::Error, fragment: &str| {
        assert!(
            matches!(&error, tenet::prelude::Error::InvalidArgument(_)),
            "{error:?}"
        );
        assert!(format!("{error:?}").contains(fragment), "{error:?}");
    };

    // Wrong rank: a multi-leg changed side.
    let provider = Arc::new(tenet::core::U1FusionRule);
    let two_legs_typed: TensorMap<tenet::core::U1FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [
            &u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]),
            &u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap(),
        ],
        [&u1_leg(&provider, &[(0, 1)]), &u1_leg(&provider, &[(0, 1)])],
        u1_typed_fill,
    )
    .unwrap();
    assert_invalid(
        typed_lhs.catdomain(&two_legs_typed).unwrap_err(),
        "exactly one domain leg",
    );

    // Mismatched unchanged side, catdomain: rank `2 <- 1` on both operands
    // (the one-domain-leg check passes) with different codomain product
    // spaces, so the codomain-equality check itself is what fires.
    let other_codomain_typed: TensorMap<tenet::core::U1FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [
            &u1_leg(&provider, &[(-1, 1), (0, 3), (1, 1)]),
            &u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap(),
        ],
        [&u1_leg(&provider, &[(0, 1), (1, 1)])],
        u1_typed_fill,
    )
    .unwrap();
    assert_invalid(
        typed_lhs.catdomain(&other_codomain_typed).unwrap_err(),
        "identical codomain product spaces",
    );

    // Mismatched unchanged side, catcodomain: rank `1 <- 2` on both operands
    // (the one-codomain-leg check passes) with different domain product
    // spaces, so the domain-equality check itself is what fires.
    let stack_pair = |domain1: &[(i32, usize)]| {
        TensorMap::from_block_fn(
            &runtime,
            [&u1_leg(&provider, &[(0, 2), (1, 1)])],
            [
                &u1_leg(&provider, &[(-1, 1), (0, 1), (1, 1)]),
                &u1_leg(&provider, domain1),
            ],
            u1_typed_fill,
        )
        .unwrap()
    };
    let typed_stack_lhs: TensorMap<tenet::core::U1FusionRule, f64> = stack_pair(&[(0, 1), (1, 1)]);
    let typed_stack_rhs: TensorMap<tenet::core::U1FusionRule, f64> = stack_pair(&[(0, 2), (1, 1)]);
    assert_invalid(
        typed_stack_lhs.catcodomain(&typed_stack_rhs).unwrap_err(),
        "identical domain product spaces",
    );

    // Duality mismatch on the changed leg: the direct sum refuses.
    let dual_domain_typed: TensorMap<tenet::core::U1FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [
            &u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]),
            &u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap(),
        ],
        [&u1_leg(&provider, &[(0, 1), (1, 1)]).try_dual().unwrap()],
        u1_typed_fill,
    )
    .unwrap();
    assert_invalid(
        typed_lhs.catdomain(&dual_domain_typed).unwrap_err(),
        "opposite duality",
    );

    assert_invalid(
        typed_lhs.absorb(&two_legs_typed).unwrap_err(),
        "equal codomain/domain ranks",
    );
    assert_invalid(
        typed_lhs.absorb(&dual_domain_typed).unwrap_err(),
        "equal duality",
    );

    // Runtime mismatch, for all three operations.
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let typed_other = u1_cat_tensor(&other_runtime, &[(0, 1), (1, 1)]);
    for error in [
        typed_lhs.catdomain(&typed_other).unwrap_err(),
        typed_lhs.catcodomain(&typed_other).unwrap_err(),
        typed_lhs.absorb(&typed_other).unwrap_err(),
    ] {
        assert!(matches!(error, tenet::prelude::Error::RuntimeMismatch));
    }
}

#[test]
fn typed_cat_and_absorb_reject_a_foreign_rule_identity_first() {
    // What (gate 4): the rule-identity check fires before any space validation:
    // two providers of the same Rust type but different identities cannot be
    // concatenated or absorbed and report `RuleMismatch`.
    let _guard = cache_lock();
    let runtime = runtime();
    let build = |provider: &Arc<ExternalZ3>| {
        let leg = GradedSpace::try_new_with_arc(
            Arc::clone(provider),
            [(Z3Charge(0), 1), (Z3Charge(1), 1)],
        )
        .unwrap();
        let tensor: TensorMap<ExternalZ3, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
        tensor
    };
    let ours = build(&Arc::new(ExternalZ3::new()));
    let theirs = build(&Arc::new(ExternalZ3::tagged(7)));
    for error in [
        ours.catdomain(&theirs).unwrap_err(),
        ours.catcodomain(&theirs).unwrap_err(),
        ours.absorb(&theirs).unwrap_err(),
    ] {
        assert!(matches!(error, tenet::prelude::Error::RuleMismatch));
    }

    let other_runtime = Runtime::builder().build().unwrap();
    let foreign = Arc::new(ExternalZ3::tagged(7));
    let foreign_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&foreign), [(Z3Charge(0), 1), (Z3Charge(1), 1)])
            .unwrap();
    let foreign_runtime: TensorMap<ExternalZ3, f64> =
        TensorMap::from_block_fn(&other_runtime, [&foreign_leg], [&foreign_leg], |_, _| 1.0)
            .unwrap();
    assert!(matches!(
        ours.absorb(&foreign_runtime).unwrap_err(),
        tenet::prelude::Error::RuleMismatch
    ));

    let foreign_bad_rank: TensorMap<ExternalZ3, f64> = TensorMap::from_block_fn(
        &other_runtime,
        [&foreign_leg, &foreign_leg],
        [&foreign_leg],
        |_, _| 1.0,
    )
    .unwrap();
    assert!(matches!(
        ours.absorb(&foreign_bad_rank).unwrap_err(),
        tenet::prelude::Error::InvalidArgument(_)
    ));
}

#[test]
fn external_z3_cat_and_absorb_hold_by_value() {
    // What (gate 6): typed-only law/value checks on the external Z3 provider.
    // The expected payloads are computed by hand rather than through another
    // facade.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let w =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)])
            .unwrap();
    let v1 =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)])
            .unwrap();
    let v2 =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Z3Charge(1), 1), (Z3Charge(2), 1)])
            .unwrap();
    let a: TensorMap<ExternalZ3, f64> =
        TensorMap::from_block_fn(&runtime, [&w], [&v1], |sectors, _| {
            1.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();
    let b: TensorMap<ExternalZ3, f64> =
        TensorMap::from_block_fn(&runtime, [&w], [&v2], |sectors, _| {
            10.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();

    let joined: TensorMap<ExternalZ3, f64> = a.catdomain(&b).unwrap();
    // Blocks in coupled-sector order: charge 0 holds only the lhs value, the
    // shared charge 1 holds the lhs column then the rhs column; charge 2 has
    // no codomain sector, so the merged leg carries it without a block.
    assert_eq!(joined.data(), &[1.0, 2.0, 11.0]);
    assert_eq!(
        joined.domain()[0].sectors().unwrap(),
        vec![Z3Charge(0), Z3Charge(1), Z3Charge(2)]
    );
    assert_eq!(joined.domain()[0].degeneracies(), &[1, 2, 1]);

    // Absorb: destination charge-0 block is 1x1, source block 1x1 — the
    // shared charge-0 and charge-1 blocks are overwritten; nothing else
    // exists. Then a non-shared key: absorb from a tensor whose only block is
    // charge 1.
    let c: TensorMap<ExternalZ3, f64> =
        TensorMap::from_block_fn(&runtime, [&w], [&v2], |sectors, _| {
            100.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();
    let absorbed: TensorMap<ExternalZ3, f64> = a.absorb(&c).unwrap();
    // a's blocks: charge 0 -> 1.0, charge 1 -> 2.0; c has only charge 1
    // (value 101.0). The non-shared charge-0 block is untouched.
    assert_eq!(absorbed.data(), &[1.0, 101.0]);
}

// ---------------------------------------------------------------------------
// #580 PR 5: typed twist / flip / unit-leg insert & remove.
// ---------------------------------------------------------------------------

/// One typed fZ2 operand: rank `2 <- 1` with a dual codomain leg. The codomain carries a non-dual
/// and a dual leg and the domain a non-dual one, so twist/flip gates can pick
/// each (side, duality) combination off one fixture.
fn fz2_index(runtime: &Runtime) -> TensorMap<tenet::core::FermionParityFusionRule, f64> {
    let provider = Arc::new(tenet::core::FermionParityFusionRule);
    let leg = |pairs: &[(u8, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            pairs.iter().map(|&(parity, degeneracy)| {
                (
                    if parity == 0 {
                        tenet::core::Z2Irrep::EVEN
                    } else {
                        tenet::core::Z2Irrep::ODD
                    },
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let l1 = leg(&[(0, 1), (1, 2)]);
    let l2 = leg(&[(0, 2), (1, 1)]).try_dual().unwrap();
    let lv = leg(&[(0, 1), (1, 1)]);
    TensorMap::from_block_fn(runtime, [&l1, &l2], [&lv], fz2_typed_fill).unwrap()
}

/// The typed flip-doctest fixture: fZ2 `V <- V`, even block 2.0, odd block 3.0.
fn typed_fz2_doctest(runtime: &Runtime) -> TensorMap<tenet::core::FermionParityFusionRule, f64> {
    let provider = Arc::new(tenet::core::FermionParityFusionRule);
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    TensorMap::from_block_fn(runtime, [&leg], [&leg], |sectors, _| {
        if sectors.coupled() == &tenet::core::Z2Irrep::EVEN {
            2.0
        } else {
            3.0
        }
    })
    .unwrap()
}

fn typed_fz2_two_block(
    runtime: &Runtime,
    dual: bool,
) -> TensorMap<tenet::core::FermionParityFusionRule, f64> {
    let provider = Arc::new(tenet::core::FermionParityFusionRule);
    let leg = GradedSpace::try_new_with_arc(
        provider,
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let leg = if dual { leg.try_dual().unwrap() } else { leg };
    TensorMap::from_block_fn(runtime, [&leg], [&leg], |sectors, _| {
        if sectors.coupled() == &tenet::core::Z2Irrep::EVEN {
            2.0
        } else {
            3.0
        }
    })
    .unwrap()
}

macro_rules! assert_same_typed_block_structure {
    ($got:expr, $source:expr) => {{
        let (got, source) = ($got, $source);
        assert!(std::ptr::eq(got.provider(), source.provider()));
        assert_eq!(got.block_count(), source.block_count());
        for index in 0..source.block_count() {
            let (after, before) = (got.block(index).unwrap(), source.block(index).unwrap());
            assert_eq!(
                (after.offset(), after.shape(), after.strides()),
                (before.offset(), before.shape(), before.strides())
            );
            let (after, before) = (
                got.block_fusion_trees(index).unwrap(),
                source.block_fusion_trees(index).unwrap(),
            );
            assert_eq!(after.coupled(), before.coupled());
            assert_eq!(after.codomain_uncoupled(), before.codomain_uncoupled());
            assert_eq!(after.codomain_innerlines(), before.codomain_innerlines());
            assert_eq!(after.codomain_vertices(), before.codomain_vertices());
            assert_eq!(after.domain_uncoupled(), before.domain_uncoupled());
            assert_eq!(after.domain_innerlines(), before.domain_innerlines());
            assert_eq!(after.domain_vertices(), before.domain_vertices());
        }
    }};
}

#[test]
fn typed_inverse_index_ops_pin_values_and_preserve_structure() {
    let _guard = cache_lock();
    let runtime = runtime();
    let simple = typed_fz2_two_block(&runtime, false);
    let dual = typed_fz2_two_block(&runtime, true);
    assert_eq!(simple.flip_inverse(&[0]).unwrap().data(), &[2.0, -3.0]);
    assert_eq!(dual.flip_inverse(&[0]).unwrap().data(), &[2.0, 3.0]);
    assert_eq!(simple.flip_inverse(&[1]).unwrap().data(), &[2.0, 3.0]);
    assert_eq!(dual.flip_inverse(&[1]).unwrap().data(), &[2.0, -3.0]);
    assert_eq!(simple.twist_inverse(&[0]).unwrap().data(), &[2.0, -3.0]);
    assert_ne!(
        simple.flip(&[1]).unwrap().flip(&[1]).unwrap().data(),
        simple.data()
    );
    for restored in [
        simple.flip(&[1]).unwrap().flip_inverse(&[1]).unwrap(),
        simple.flip_inverse(&[1]).unwrap().flip(&[1]).unwrap(),
    ] {
        assert_eq!(restored.data(), simple.data());
        assert_same_legs(&restored.codomain(), &simple.codomain());
        assert_same_legs(&restored.domain(), &simple.domain());
    }
    assert_eq!(
        simple.flip_inverse(&[1, 1]).unwrap().data(),
        simple
            .flip_inverse(&[1])
            .unwrap()
            .flip_inverse(&[1])
            .unwrap()
            .data()
    );

    let complex = simple.to_c64();
    assert_eq!(
        complex.flip_inverse(&[0]).unwrap().data(),
        &[2.0.into(), (-3.0).into()]
    );
    assert_eq!(
        complex.twist_inverse(&[0]).unwrap().data(),
        &[2.0.into(), (-3.0).into()]
    );

    let su2_provider = Arc::new(SU2FusionRule);
    let spin_half =
        GradedSpace::try_new_with_arc(su2_provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let spin_half_dual = spin_half.try_dual().unwrap();
    let su2: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&spin_half_dual], [&spin_half], |_, _| 5.0).unwrap();
    assert_eq!(su2.flip_inverse(&[0]).unwrap().data(), &[5.0]);
    assert_eq!(su2.flip_inverse(&[1]).unwrap().data(), &[-5.0]);

    let structured = fz2_index(&runtime);
    let twisted = structured.twist_inverse(&[0, 1, 2]).unwrap();
    assert!(std::ptr::eq(twisted.provider(), structured.provider()));
    assert_same_legs(&twisted.codomain(), &structured.codomain());
    assert_same_legs(&twisted.domain(), &structured.domain());
    let codomain_flip = structured.flip_inverse(&[0]).unwrap();
    assert_same_typed_block_structure!(&codomain_flip, &structured);
    assert_eq!(
        codomain_flip.codomain()[0].is_dual(),
        !structured.codomain()[0].is_dual()
    );
    assert_eq!(
        codomain_flip.codomain()[1].is_dual(),
        structured.codomain()[1].is_dual()
    );
    assert_same_legs(&codomain_flip.domain(), &structured.domain());
    let domain_flip = structured.flip_inverse(&[2]).unwrap();
    assert_same_typed_block_structure!(&domain_flip, &structured);
    assert_same_legs(&domain_flip.codomain(), &structured.codomain());
    assert_eq!(
        domain_flip.domain()[0].is_dual(),
        !structured.domain()[0].is_dual()
    );

    let u1_provider = Arc::new(tenet::core::U1FusionRule);
    let u1_leg = GradedSpace::try_new_with_arc(
        u1_provider,
        [
            (tenet::core::U1Irrep::new(-1), 1),
            (tenet::core::U1Irrep::new(0), 1),
            (tenet::core::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&u1_leg], [&u1_leg], |_, _| 7.0).unwrap();
    let u1_twist = u1.twist_inverse(&[0, 1]).unwrap();
    assert_eq!(u1_twist.data().as_ptr(), u1.data().as_ptr());
    let u1_flip = u1.flip_inverse(&[0]).unwrap();
    assert_eq!(u1_flip.data(), u1.data());
    assert_same_typed_block_structure!(&u1_flip, &u1);
    assert_eq!(u1_flip.codomain()[0].is_dual(), !u1.codomain()[0].is_dual());
    assert_same_legs(&u1_flip.domain(), &u1.domain());
}

#[test]
fn inverse_index_ops_cover_the_fermionic_simple_product() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_u1_su2_oracle(&runtime, 1.0);
    for legs in [&[0usize][..], &[1][..], &[0, 1][..]] {
        assert_eq!(
            typed
                .twist(legs)
                .unwrap()
                .twist_inverse(legs)
                .unwrap()
                .data(),
            typed.data()
        );
        assert_eq!(
            typed.flip(legs).unwrap().flip_inverse(legs).unwrap().data(),
            typed.data()
        );
    }
}

#[test]
fn typed_flip_and_twist_pin_the_doctest_values() {
    // What: the flip-doctest values ([2.0, 3.0] -> flip(1) -> [2.0, -3.0])
    // and the twist involution θ² = 1, both pinned directly on the typed API.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = typed_fz2_doctest(&runtime);
    let flipped: TensorMap<tenet::core::FermionParityFusionRule, f64> = typed.flip(&[1]).unwrap();
    assert_eq!(flipped.data(), &[2.0, -3.0]);
    assert_eq!(flipped.domain()[0].is_dual(), !typed.domain()[0].is_dual());

    let twisted: TensorMap<tenet::core::FermionParityFusionRule, f64> = typed.twist(&[1]).unwrap();
    assert_eq!(twisted.data(), &[2.0, -3.0]);
    let back: TensorMap<tenet::core::FermionParityFusionRule, f64> = twisted.twist(&[1]).unwrap();
    assert_eq!(back.data(), typed.data());
}

#[test]
fn typed_multi_leg_dense_twist_is_the_per_leg_product_by_value() {
    // What: the multi-leg dense twist coefficient is pinned by a hand-computed
    // value. A mutation that drops the per-leg product in the shared
    // `twist_block_factor` (e.g. keeping only the first leg's θ) survives
    // ordinary cross-operation parity gates, which share the same helper.
    // Fixture: fZ2 `V <- V`, even block 2.0, odd block 3.0, θ(odd) = −1;
    // both legs carry the block's coupled sector.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = typed_fz2_doctest(&runtime);
    // One leg: θ bites, the odd block negates (sanity that the factor is
    // live at all on this fixture).
    let one: TensorMap<tenet::core::FermionParityFusionRule, f64> = typed.twist(&[0]).unwrap();
    assert_eq!(one.data(), &[2.0, -3.0]);
    // Two *different* legs: the odd block scales by θ·θ = (−1)² = +1 — the
    // per-leg product, not a single factor.
    let both: TensorMap<tenet::core::FermionParityFusionRule, f64> = typed.twist(&[0, 1]).unwrap();
    assert_eq!(both.data(), &[2.0, 3.0]);
    // The same leg listed twice: identity by value, for the same θ² reason.
    let twice: TensorMap<tenet::core::FermionParityFusionRule, f64> = typed.twist(&[1, 1]).unwrap();
    assert_eq!(twice.data(), &[2.0, 3.0]);
}

#[test]
fn typed_flip_is_a_fourth_root_of_identity_and_flip_squared_scales_odd_blocks() {
    // What (gate 2): the TensorKit non-involution law on the typed facade —
    // flip² returns to the original spaces but scales the odd block by
    // θ = −1; only flip⁴ = id.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = typed_fz2_doctest(&runtime);
    let f1: TensorMap<tenet::core::FermionParityFusionRule, f64> = typed.flip(&[1]).unwrap();
    let f2: TensorMap<tenet::core::FermionParityFusionRule, f64> = f1.flip(&[1]).unwrap();
    assert_same_legs(&f2.codomain(), &typed.codomain());
    assert_same_legs(&f2.domain(), &typed.domain());
    assert_eq!(f2.data(), &[2.0, -3.0]);
    let f4: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        f2.flip(&[1]).unwrap().flip(&[1]).unwrap();
    assert_eq!(f4.data(), typed.data());
}

#[test]
fn typed_flip_repeated_leg_in_one_call_is_sequential() {
    // What: the same leg listed twice in one call flips it twice sequentially;
    // the second occurrence sees the duality the first one left.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = typed_fz2_doctest(&runtime);
    let typed_twice: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed.flip(&[1, 1]).unwrap();
    assert_eq!(typed_twice.data(), &[2.0, -3.0]);
    let typed_stepwise: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed.flip(&[1]).unwrap().flip(&[1]).unwrap();
    assert_eq!(typed_twice.data(), typed_stepwise.data());
    assert_same_legs(&typed_twice.domain(), &typed.domain());
}

#[test]
fn typed_insert_unit_round_trips_at_every_position_and_shares_the_payload() {
    // What (gate 4): `insert_left_unit`/`insert_right_unit` at every legal
    // position `0..=rank` followed by `remove_unit` at the inserted axis
    // restores the spaces *and* the payload allocation — `data()` returns the
    // same buffer address, the O(1) reuse the #613 contract promises for a
    // dense payload. (The `Arc`-level gate lives with the body layout tests
    // in `typed.rs`, which can see the private fields.)
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_index(&runtime);
    for position in 0..=typed.rank() {
        for left in [true, false] {
            for dual in [false, true] {
                let inserted: TensorMap<tenet::core::FermionParityFusionRule, f64> = if left {
                    typed.insert_left_unit(position, dual).unwrap()
                } else {
                    typed.insert_right_unit(position, dual).unwrap()
                };
                assert_eq!(
                    inserted.data().as_ptr(),
                    typed.data().as_ptr(),
                    "left={left} position={position} dual={dual}"
                );
                let legs: Vec<_> = inserted
                    .codomain()
                    .into_iter()
                    .chain(inserted.domain())
                    .collect();
                assert_eq!(legs[position].is_dual(), dual);
                assert_eq!(legs[position].degeneracies(), &[1]);
                let removed: TensorMap<tenet::core::FermionParityFusionRule, f64> =
                    inserted.remove_unit(position).unwrap();
                assert_eq!(removed.data().as_ptr(), typed.data().as_ptr());
                assert_same_legs(&removed.codomain(), &typed.codomain());
                assert_same_legs(&removed.domain(), &typed.domain());
            }
        }
    }
}

#[test]
fn typed_index_op_error_classes_and_empty_shortcuts_are_stable() {
    // What: each rejected input returns the typed error class and operation-
    // specific message, while an empty leg list is a buffer-sharing clone.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_index(&runtime);

    // Twist / flip: out-of-range leg.
    for (error, message) in [
        (
            typed.twist(&[5]).unwrap_err(),
            "invalid argument: twist leg 5 out of range for rank 3",
        ),
        (
            typed.flip(&[5]).unwrap_err(),
            "invalid argument: flip leg 5 out of range for rank 3",
        ),
        (
            typed.twist_inverse(&[5]).unwrap_err(),
            "invalid argument: twist_inverse leg 5 out of range for rank 3",
        ),
        (
            typed.flip_inverse(&[5]).unwrap_err(),
            "invalid argument: flip_inverse leg 5 out of range for rank 3",
        ),
    ] {
        assert!(matches!(error, tenet::typed::Error::InvalidArgument(_)));
        assert_eq!(error.to_string(), message);
    }

    // Empty leg list: identical clone, shared buffer typed-side.
    let typed_untwisted: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed.twist(&[]).unwrap();
    assert_eq!(typed_untwisted.data().as_ptr(), typed.data().as_ptr());
    let typed_unflipped: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed.flip(&[]).unwrap();
    assert_eq!(typed_unflipped.data().as_ptr(), typed.data().as_ptr());
    let typed_untwisted_inverse: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed.twist_inverse(&[]).unwrap();
    assert_eq!(
        typed_untwisted_inverse.data().as_ptr(),
        typed.data().as_ptr()
    );
    let typed_unflipped_inverse: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed.flip_inverse(&[]).unwrap();
    assert_eq!(
        typed_unflipped_inverse.data().as_ptr(),
        typed.data().as_ptr()
    );

    // Insert: position past the rank.
    assert_eq!(
        typed.insert_left_unit(4, false).unwrap_err().to_string(),
        "invalid argument: TensorMap::insert_left_unit: position 4 exceeds rank 3"
    );
    assert_eq!(
        typed.insert_right_unit(4, false).unwrap_err().to_string(),
        "invalid argument: TensorMap::insert_right_unit: position 4 exceeds rank 3"
    );

    // Remove: out-of-range axis, then a non-unit leg.
    assert_eq!(
        typed.remove_unit(3).unwrap_err().to_string(),
        "invalid argument: TensorMap::remove_unit: axis 3 is out of range for rank 3"
    );
    let error = typed.remove_unit(0).unwrap_err();
    assert!(
        matches!(error, tenet::typed::Error::InvalidArgument(_)),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "invalid argument: TensorMap::remove_unit: axis 0 is not a canonical unit leg"
    );
}

#[test]
fn typed_twist_on_a_compact_spectrum_matches_the_dense_route() {
    // What: the compact twist arm matches the ordinary dense tree transform
    // on an SVD spectrum whose coupled sectors include the odd one.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_index(&runtime);
    let typed_s: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed.svd_compact().unwrap().1;
    let dense = forced_dense(&typed_s);
    let typed_twisted: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed_s.twist(&[0]).unwrap();
    assert_eq!(typed_twisted.data(), dense.twist(&[0]).unwrap().data());
    // And the two-leg twist is the identity on the bond (θ² = 1 per sector).
    let typed_both: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed_s.twist(&[0, 1]).unwrap();
    assert_eq!(typed_both.data(), typed_s.data());
    let typed_inverse: TensorMap<tenet::core::FermionParityFusionRule, f64> =
        typed_s.twist_inverse(&[0]).unwrap();
    assert_eq!(typed_inverse.data(), typed_twisted.data());
}

#[test]
fn external_z3_twist_flip_and_units_hold_by_value() {
    // What (gate 7): typed-only checks on the external Z3 provider. Coverage
    // limit, on purpose: Z3 is bosonic (θ ≡ 1, χ ≡ 1), so `twist` exercises
    // only the identity short-circuit and `flip` only the structural toggle
    // with factor 1 — the θ/χ-bearing arms are covered by the built-in fZ2
    // parity gates above; the harness has no fermionic external provider.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let w = z3_leg(&provider, false);
    let v = z3_leg(&provider, true);
    let t: TensorMap<ExternalZ3, f64> =
        TensorMap::from_block_fn(&runtime, [&w], [&v], |sectors, indices| {
            f64::from(sectors.coupled().0) * 10.0 + (indices[0] * 3 + indices[1]) as f64
        })
        .unwrap();

    // Twist: identity, shared buffer.
    let twisted: TensorMap<ExternalZ3, f64> = t.twist(&[0, 1]).unwrap();
    assert_eq!(twisted.data().as_ptr(), t.data().as_ptr());

    // Flip: values unchanged, duality flags toggled, non-self-dual sector
    // sets preserved as stored (flip toggles the flag, not the labels).
    let flipped: TensorMap<ExternalZ3, f64> = t.flip(&[0, 1]).unwrap();
    assert_eq!(flipped.data(), t.data());
    assert!(flipped.codomain()[0].is_dual());
    assert!(!flipped.domain()[0].is_dual());
    assert_eq!(
        flipped.codomain()[0].sectors().unwrap(),
        t.codomain()[0].sectors().unwrap()
    );

    // Units: insert -> remove round trip on the external provider, O(1)
    // payload reuse observable through `data()`.
    let inserted: TensorMap<ExternalZ3, f64> = t.insert_right_unit(1, true).unwrap();
    assert_eq!(inserted.data().as_ptr(), t.data().as_ptr());
    assert_eq!(inserted.codomain()[1].sectors().unwrap(), vec![Z3Charge(0)]);
    let removed: TensorMap<ExternalZ3, f64> = inserted.remove_unit(1).unwrap();
    assert_eq!(removed.data().as_ptr(), t.data().as_ptr());
    assert_same_legs(&removed.codomain(), &t.codomain());
    assert_same_legs(&removed.domain(), &t.domain());
}

// ---------------------------------------------------------------------------
// #580 PR 5 / PR #620 review: NoBraiding preflight for twist and flip.
//
// External NoBraiding provider: planar Z2 (the tenet-core test fixture
// `PlanarZ2Rule`, rebuilt from the public vocabulary with a codec). The
// These gates verify that planar providers need no braiding capability;
// twist/flip still route through the shared structural preflight.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct PlanarZ2;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PlanarParity(u8);

impl FusionRule for PlanarZ2 {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x9a2f_0620_0000_0000, Arc::<[u8]>::from(vec![]))
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::NoBraiding
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(left.id() ^ right.id())).collect()
    }
}

impl MultiplicityFreeFusionRule for PlanarZ2 {}
impl tenet::core::CanonicalUnitFusionRule for PlanarZ2 {}

impl MultiplicityFreeFusionSymbols for PlanarZ2 {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for PlanarZ2 {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

impl CheckedFusionAlgebra for PlanarZ2 {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl SectorCodec for PlanarZ2 {
    type Sector = PlanarParity;
    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        if value.0 < 2 {
            Ok(SectorId::new(usize::from(value.0)))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.rule_identity(),
                label: format!("planar parity {}", value.0),
            })
        }
    }
    fn decode_sector(&self, sector: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        if sector.id() < 2 {
            Ok(PlanarParity(sector.id() as u8))
        } else {
            Err(FusionAlgebraError::InvalidSector { sector })
        }
    }
}

#[test]
fn external_nobraiding_twist_and_flip_reject_nontrivial_sectors() {
    // What (PR #620 review P2): under `BraidingStyleKind::NoBraiding` the
    // twist eigenvalue is undefined, so twist/flip on a leg carrying any
    // non-unit sector must fail — TensorKit `has_shared_twist`
    // (`tensors/indexmanipulations.jl:34-41`) throws `SectorMismatch` there
    // — instead of silently applying θ ≡ 1. The compact spectrum arm must
    // hit the same preflight before its own dispatch.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(PlanarZ2);
    let mixed = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(PlanarParity(0), 1), (PlanarParity(1), 2)],
    )
    .unwrap();
    let t: TensorMap<PlanarZ2, f64> =
        TensorMap::from_block_fn(&runtime, [&mixed], [&mixed], |sectors, indices| {
            f64::from(sectors.coupled().0) * 10.0 + (indices[0] * 3 + indices[1]) as f64
        })
        .unwrap();

    for error in [
        t.twist(&[0]).unwrap_err(),
        t.flip(&[1]).unwrap_err(),
        t.twist_inverse(&[0]).unwrap_err(),
        t.flip_inverse(&[1]).unwrap_err(),
    ] {
        assert!(
            matches!(error, tenet::typed::Error::InvalidArgument(_)),
            "{error:?}"
        );
        assert!(error.to_string().contains("no braiding"), "{error}");
    }
    // The compact diagonal arm rejects too: an SVD spectrum factor lives on
    // the mixed bond space, so its twist must fail before the compact
    // per-sector scaling ever runs.
    let s: TensorMap<PlanarZ2, f64> = t.svd_compact().unwrap().1;
    let compact_error = s.twist(&[0]).unwrap_err();
    assert!(
        matches!(compact_error, tenet::typed::Error::InvalidArgument(_)),
        "{compact_error:?}"
    );
    assert!(matches!(
        s.twist_inverse(&[0]),
        Err(tenet::typed::Error::InvalidArgument(_))
    ));
}

#[test]
fn external_nobraiding_vacuum_only_legs_twist_passes_flip_rejects() {
    // What (PR #620 review P2, second round): the TK asymmetry on
    // vacuum-only legs under NoBraiding. `twist` carries an explicit
    // unit-sector carve-out (`has_shared_twist`,
    // `tensors/indexmanipulations.jl:34-41`) and is the identity (shared
    // buffer). `flip` has NO such exception — TK's fusion-tree flip
    // unconditionally evaluates `frobenius_schur_phase(a)` and `twist(a)`
    // (`fusiontrees/braiding_manipulations.jl:384-412`), neither of which a
    // NoBraiding sector defines, so flip fails in TK even on the vacuum and
    // must fail here. The boundary stays: `flip(&[])` is the empty-list
    // identical clone and never reaches the guard.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(PlanarZ2);
    let unit_only =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(PlanarParity(0), 2)]).unwrap();
    let t: TensorMap<PlanarZ2, f64> =
        TensorMap::from_block_fn(&runtime, [&unit_only], [&unit_only], |_, indices| {
            (indices[0] * 2 + indices[1]) as f64
        })
        .unwrap();

    let twisted: TensorMap<PlanarZ2, f64> = t.twist(&[0, 1]).unwrap();
    assert_eq!(twisted.data().as_ptr(), t.data().as_ptr());
    let twisted_inverse: TensorMap<PlanarZ2, f64> = t.twist_inverse(&[0, 1]).unwrap();
    assert_eq!(twisted_inverse.data().as_ptr(), t.data().as_ptr());

    let flip_error = t.flip(&[0]).unwrap_err();
    assert!(
        matches!(flip_error, tenet::typed::Error::InvalidArgument(_)),
        "{flip_error:?}"
    );
    assert!(
        flip_error.to_string().contains("no braiding"),
        "{flip_error}"
    );
    assert!(matches!(
        t.flip_inverse(&[0]),
        Err(tenet::typed::Error::InvalidArgument(_))
    ));

    let unflipped: TensorMap<PlanarZ2, f64> = t.flip(&[]).unwrap();
    assert_eq!(unflipped.data().as_ptr(), t.data().as_ptr());
    let unflipped_inverse: TensorMap<PlanarZ2, f64> = t.flip_inverse(&[]).unwrap();
    assert_eq!(unflipped_inverse.data().as_ptr(), t.data().as_ptr());
}

#[test]
fn cu1_typed_rank_three_permutation_pins_the_gauge_contract_and_recoupling_values() {
    // What: CU(1) does not certify a trivial associator gauge, and this
    // rank-three charged fixture independently pins its recoupled payload.
    let _guard = cache_lock();
    let runtime = runtime();
    let rule = Arc::new(CU1FusionRule);
    assert!(!rule.has_trivial_associator_gauge());
    let q = CU1Irrep::from_twice_charge(1);
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&rule), [(q, 1)]).unwrap();
    let tensor: TensorMap<CU1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg, &leg], [&leg], |_, _| 1.0).unwrap();
    assert_eq!(tensor.codomain().len(), 3);
    assert_eq!(tensor.domain().len(), 1);
    assert!(tensor
        .codomain()
        .iter()
        .chain(tensor.domain().iter())
        .all(|space| space.degeneracies() == [1]));
    assert_eq!(tensor.data(), [1.0, 1.0, 1.0]);
    let permuted = tensor.permute(&[2, 0, 1], &[3]).unwrap();
    assert_eq!(permuted.codomain().len(), 3);
    assert_eq!(permuted.domain().len(), 1);
    assert_eq!(permuted.data().len(), 3);
    for (got, expected) in
        permuted
            .data()
            .iter()
            .zip([2.0_f64.sqrt() / 2.0, -2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt()])
    {
        assert!((got - expected).abs() <= 1e-12, "{got} vs {expected}");
    }
}

// ---------------------------------------------------------------------------
// Issue #580, group 6: ordered contraction through canonical `contract`.
// ---------------------------------------------------------------------------

#[test]
fn contract_error_classes_and_their_both_defect_precedence() {
    // What: the typed contract reports
    // a bad output order before a simultaneous contracted-leg mismatch.
    let _guard = cache_lock();
    let second = runtime();
    let runtime = runtime();
    let typed_fixture = |runtime: &Runtime| {
        let leg = GradedSpace::try_new_with_arc(
            Arc::new(tenet::core::Z2FusionRule),
            [
                (tenet::core::Z2Irrep::EVEN, 2),
                (tenet::core::Z2Irrep::ODD, 3),
            ],
        )
        .unwrap();
        TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], typed_fill_value).unwrap()
    };
    let typed: TensorMap<tenet::core::Z2FusionRule, f64> = typed_fixture(&runtime);

    let cases: &[(&str, &[usize], &[usize], &[usize], &str)] = &[
        (
            "len mismatch",
            &[2],
            &[0, 1],
            &[0, 1, 2, 3],
            "ContractAxisCountMismatch",
        ),
        (
            "lhs out of range",
            &[9],
            &[0],
            &[0, 1, 2, 3],
            "InvalidAxisSet",
        ),
        (
            "rhs out of range",
            &[2],
            &[9],
            &[0, 1, 2, 3],
            "InvalidAxisSet",
        ),
        (
            "output wrong length",
            &[2],
            &[0],
            &[0, 1, 2],
            "InvalidPermutation",
        ),
        (
            "output duplicate",
            &[2],
            &[0],
            &[0, 0, 1, 2],
            "InvalidPermutation",
        ),
        (
            "output out of range",
            &[2],
            &[0],
            &[0, 1, 2, 9],
            "InvalidPermutation",
        ),
    ];
    for &(name, lhs_axes, rhs_axes, output_axes, class) in cases {
        let typed_error = typed
            .contract(&typed, lhs_axes, rhs_axes, output_axes)
            .unwrap_err();
        assert!(
            matches!(typed_error, tenet::typed::Error::Operation(_)),
            "{name}: {typed_error:?}"
        );
        assert!(
            format!("{typed_error:?}").contains(class),
            "{name}: {typed_error:?} does not carry {class}"
        );
    }

    // Mismatched contracted legs retain the expert layer's class.
    let narrow_leg = GradedSpace::try_new_with_arc(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    let typed_narrow: TensorMap<tenet::core::Z2FusionRule, f64> = TensorMap::from_block_fn(
        &runtime,
        [&narrow_leg, &narrow_leg],
        [&narrow_leg],
        typed_fill_value,
    )
    .unwrap();
    let typed_error = typed
        .contract(&typed_narrow, &[2], &[0], &[0, 1, 2, 3])
        .unwrap_err();
    assert!(
        format!("{typed_error:?}").contains("LegDegeneracyMismatch"),
        "{typed_error:?}"
    );

    // Both defects at once: mismatched legs AND a non-permutation output order.
    let typed_both = typed
        .contract(&typed_narrow, &[2], &[0], &[0, 0, 1, 2])
        .unwrap_err();
    assert!(
        format!("{typed_both:?}").contains("InvalidPermutation"),
        "typed both-defect precedence moved: {typed_both:?}"
    );

    // Runtime mismatch stays a facade-level error.
    let typed_second: TensorMap<tenet::core::Z2FusionRule, f64> = typed_fixture(&second);
    assert!(matches!(
        typed
            .contract(&typed_second, &[2], &[0], &[0, 1, 2, 3])
            .unwrap_err(),
        tenet::typed::Error::RuntimeMismatch
    ));

    // Rule-identity mismatch stays the expert layer's rejection through the
    // canonical name too.
    let first = Arc::new(ExternalZ3::tagged(0));
    let second_rule = Arc::new(ExternalZ3::tagged(1));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&first, 2),
        &z3_dense_leg(&first, 3),
        1.0,
    );
    let rhs = counting_z3(
        &runtime,
        &z3_dense_leg(&second_rule, 3),
        &z3_dense_leg(&second_rule, 4),
        1.0,
    );
    let identity_error = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap_err();
    assert!(
        format!("{identity_error:?}").contains("FusionRuleMismatch"),
        "{identity_error:?}"
    );
}

#[test]
#[allow(deprecated)]
fn contract_ordered_delegates_with_a_nonidentity_output_order() {
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(tenet::core::U1FusionRule);
    let leg = u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let lhs: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [], [&leg], 224_303).unwrap();
    let rhs: TensorMap<tenet::core::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg, &leg], 224_304).unwrap();

    let actual = lhs.contract_ordered(&rhs, &[0], &[0], &[1, 0]).unwrap();
    let expected = lhs
        .contract(&rhs, &[0], &[0], &[0, 1])
        .unwrap()
        .permute(&[], &[1, 0])
        .unwrap();
    assert_eq!(actual.data(), expected.data());
}

#[test]
fn typed_contract_parallel_su2_replay_matches_serial() {
    let _guard = cache_lock();

    fn run(runtime: &Runtime) -> Vec<f64> {
        let provider = Arc::new(SU2FusionRule);
        let leg = GradedSpace::try_new_with_arc(
            provider,
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 3),
                (SU2Irrep::from_twice_spin(2), 2),
            ],
        )
        .unwrap();
        let lhs: TensorMap<SU2FusionRule, f64> =
            TensorMap::rand_with_seed(runtime, [&leg, &leg], [&leg, &leg], 224_401).unwrap();
        let rhs: TensorMap<SU2FusionRule, f64> =
            TensorMap::rand_with_seed(runtime, [&leg, &leg], [&leg, &leg], 224_402).unwrap();
        lhs.contract(&rhs, &[3, 2], &[0, 1], &[2, 0, 3, 1])
            .unwrap()
            .data()
            .to_vec()
    }

    let serial = Runtime::builder().recoupling_threads(1).build().unwrap();
    let parallel = Runtime::builder().recoupling_threads(2).build().unwrap();
    let serial = run(&serial);
    let parallel = run(&parallel);
    assert_eq!(serial.len(), parallel.len());
    assert!(serial
        .iter()
        .zip(parallel)
        .all(|(&serial, parallel)| (serial - parallel).abs() < 1e-12));
}

#[test]
fn contract_on_the_external_z3_provider_matches_the_hand_product() {
    // What (gate 5): a typed-only ordered-contraction value check on the
    // external provider. Same fixture as the
    // `contract` hand-product gate: `output_axes = [1, 0]` is the transpose of
    // the 2x3 · 3x4 counting product, `[0, 1]` the product itself.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 7.0);

    let transposed: TensorMap<ExternalZ3, f64> = lhs.contract(&rhs, &[1], &[0], &[1, 0]).unwrap();
    assert_eq!(
        transposed.data(),
        [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
    );
    let identity: TensorMap<ExternalZ3, f64> = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(
        identity.data(),
        [76.0, 100.0, 103.0, 136.0, 130.0, 172.0, 157.0, 208.0]
    );
}
