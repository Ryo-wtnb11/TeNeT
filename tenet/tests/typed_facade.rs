//! Gate suite for the provider-typed facade (`tenet::typed`, issue #557).
//!
//! Every provider here is built from the public vocabulary alone — no sealed
//! lowered codec, no crate-internal machinery — so the suite doubles as proof
//! that a downstream application can drive the typed facade with its own
//! fusion rule.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tenet::core::{
    complete_hom_space_structure_cache_info, fusion_tree_layout_cache_info, BraidingStyleKind,
    CheckedFusionAlgebra, FusionAlgebraError, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorVec,
};
use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{GradedSpace, TensorMap};

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
    fn scalar_one(&self) -> f64 {
        1.0
    }
    fn scalar_conj(&self, value: f64) -> f64 {
        value
    }
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

impl CheckedFusionAlgebra for ExternalZ3 {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if self.quirk == Some(Quirk::FailDual) {
            return Err(FusionAlgebraError::InvalidSector { sector });
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
            Ok(SectorId::new(usize::from(value.0)))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.rule_identity(),
                label: format!("Z3 charge {}", value.0),
            })
        }
    }

    fn decode_sector(&self, sector: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        u8::try_from(sector.id())
            .ok()
            .filter(|&charge| charge < 3)
            .map(Z3Charge)
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
    fn scalar_one(&self) -> f64 {
        SU2FusionRule.scalar_one()
    }
    fn scalar_conj(&self, value: f64) -> f64 {
        SU2FusionRule.scalar_conj(value)
    }
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
    GradedSpace::try_new(
        Arc::clone(provider),
        [(Z3Charge(0), 2), (Z3Charge(1), 3), (Z3Charge(2), 1)],
        is_dual,
    )
    .expect("Z3 leg is well formed")
}

fn su2_leg(provider: &Arc<ExternalSu2>, is_dual: bool) -> GradedSpace<ExternalSu2> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [(SU2Irrep::from_twice_spin(1), 2)],
        is_dual,
    )
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
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Z3Charge(2), 1), (Z3Charge(0), 2), (Z3Charge(1), 3)],
        false,
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
        GradedSpace::try_new(provider, [(Z3Charge(0), 2), (Z3Charge(1), 0)], false).unwrap();

    assert_eq!(space.sectors().unwrap(), vec![Z3Charge(0)]);
    assert_eq!(space.degeneracies(), &[2]);
}

#[test]
fn graded_space_rejects_a_duplicate_label_by_name() {
    // What: the duplicate is reported as the caller's own label, which needs
    // the check to run before the label is encoded away into a `SectorId`.
    let provider = Arc::new(ExternalZ3::new());
    let error =
        GradedSpace::try_new(provider, [(Z3Charge(1), 2), (Z3Charge(1), 3)], false).unwrap_err();

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
        GradedSpace::try_new(provider, [(Z3Charge(0), 2), (Z3Charge(1), 3)], false).unwrap_err();

    let message = error.to_string();
    assert!(message.contains("SectorCodec"), "{message}");
    assert!(message.contains("Z3Charge(0)"), "{message}");
    assert!(message.contains("Z3Charge(1)"), "{message}");
}

#[test]
fn graded_space_reports_an_unrepresentable_label() {
    // What: an out-of-domain label surfaces the provider's own encode error.
    let provider = Arc::new(ExternalZ3::new());
    let error = GradedSpace::try_new(provider, [(Z3Charge(7), 2)], false).unwrap_err();

    assert!(error.to_string().contains("Z3 charge 7"), "{error}");
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
    let domain = GradedSpace::try_new(
        Arc::clone(&broken),
        [(Z3Charge(0), 2), (Z3Charge(2), 3), (Z3Charge(1), 1)],
        true,
    )
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
