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
fn graded_space_dual_reports_a_non_injective_dual_instead_of_panicking() {
    // What: a provider whose dual collapses two sectors onto one id is a
    // broken rigidity structure. The leg cannot hold the result, and the
    // failure must come back as a typed error — the constructor underneath
    // used to panic inside this `Result`-returning API.
    let provider = Arc::new(ExternalZ3::with(Quirk::CollapsingDual));
    let space = z3_leg(&provider, false);

    let error = space.try_dual().unwrap_err();

    assert!(error.to_string().contains("not injective"), "{error}");
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
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Z3Charge(0), 1), (Z3Charge(1), 2)],
        false,
    )
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
fn block_fusion_trees_reports_a_non_self_dual_domain_label() {
    // What: a dual domain leg carrying charge 2 — whose dual is charge 1, so a
    // confusion between the two would show — is decoded as charge 2, matching
    // the convention that a tree labels a domain leg with the space's own
    // sector rather than its dual.
    let _guard = cache_lock();
    let provider = Arc::new(ExternalZ3::new());
    let codomain = GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(1), 1)], false).unwrap();
    let domain = GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(2), 1)], true).unwrap();
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

/// Fill value from the erased fusion-tree key: parities weighted by position
/// so any reordering of legs, blocks or elements changes the buffer.
fn erased_fill_value(key: &tenet::prelude::BlockKey, indices: &[usize]) -> f64 {
    let pair = key.as_fusion_tree_pair().expect("fusion-tree block");
    let parity = |id| {
        SectorCodec::decode_sector(&tenet::core::Z2FusionRule, id)
            .expect("built-in codec decodes its own ids")
            .parity()
    };
    let mut value = f64::from(parity(pair.codomain_tree().coupled())) * 1000.0;
    for (position, &id) in pair.codomain_tree().uncoupled().iter().enumerate() {
        value += f64::from(parity(id)) * 100.0 * (position + 1) as f64;
    }
    for (position, &id) in pair.domain_tree().uncoupled().iter().enumerate() {
        value += f64::from(parity(id)) * 10.0 * (position + 1) as f64;
    }
    value
        + indices
            .iter()
            .enumerate()
            .map(|(a, &i)| (a + 1) * i)
            .sum::<usize>() as f64
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
fn typed_and_erased_block_fill_produce_identical_storage_on_a_builtin_rule() {
    // What: for a built-in rule reachable from both facades, the typed checked
    // construction and the erased one agree byte for byte and block for block —
    // same layout, same block order, same element order.
    let _guard = cache_lock();
    let runtime = runtime();

    let space = tenet::prelude::Space::z2([(0, 2), (1, 3)]);
    let mut erased_keys: Vec<tenet::prelude::BlockKey> = Vec::new();
    let erased = tenet::prelude::Tensor::from_block_fn(
        &runtime,
        [&space, &space],
        [&space],
        |key, indices| {
            if erased_keys.last() != Some(key) {
                erased_keys.push(key.clone());
            }
            erased_fill_value(key, indices)
        },
    )
    .unwrap();

    let provider = Arc::new(tenet::core::Z2FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        false,
    )
    .unwrap();
    let typed: TensorMap<tenet::core::Z2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], typed_fill_value).unwrap();

    assert_eq!(typed.data(), erased.data());
    assert_eq!(typed.block_count(), erased_keys.len());
    for (index, key) in erased_keys.iter().enumerate() {
        let pair = key.as_fusion_tree_pair().unwrap();
        let sectors = typed.block_fusion_trees(index).unwrap();
        let expected: Vec<tenet::core::Z2Irrep> = pair
            .codomain_tree()
            .uncoupled()
            .iter()
            .map(|&id| SectorCodec::decode_sector(&tenet::core::Z2FusionRule, id).unwrap())
            .collect();
        assert_eq!(sectors.codomain_uncoupled(), expected.as_slice());
        assert_eq!(
            sectors.coupled(),
            &SectorCodec::decode_sector(&tenet::core::Z2FusionRule, pair.codomain_tree().coupled())
                .unwrap()
        );
    }
}

// ---------------------------------------------------------------------------
// Phase 3, slice 2: `TensorMap::permute`.
// ---------------------------------------------------------------------------

/// The second Z3 leg shape, deliberately unlike [`z3_leg`]'s: with four legs of
/// two distinct degeneracy patterns, a permuted leg can be identified by its
/// degeneracies, which a uniform fixture could not distinguish.
fn z3_other_leg(provider: &Arc<ExternalZ3>, is_dual: bool) -> GradedSpace<ExternalZ3> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [(Z3Charge(0), 1), (Z3Charge(1), 2), (Z3Charge(2), 4)],
        is_dual,
    )
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

/// The erased/typed pair of the byte-oracle fixture: one built-in Z2 layout
/// filled identically through both facades.
fn z2_oracle_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, f64>,
) {
    z2_oracle_pair_split(runtime, 2)
}

/// The same oracle pair over three identical legs, split into
/// `num_codomain <- rest`. `2` is tall in every coupled sector and `1` is wide,
/// which is what separates the compact factorizations from the full ones:
/// on a tall input LQ-compact and LQ-full coincide.
fn z2_oracle_pair_split(
    runtime: &Runtime,
    num_codomain: usize,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, f64>,
) {
    let space = tenet::prelude::Space::z2([(0, 2), (1, 3)]);
    let spaces = [&space, &space, &space];
    let (codomain, domain) = spaces.split_at(num_codomain);
    let erased = tenet::prelude::Tensor::from_block_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        erased_fill_value,
    )
    .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        false,
    )
    .unwrap();
    let legs = [&leg, &leg, &leg];
    let (codomain, domain) = legs.split_at(num_codomain);
    let typed = TensorMap::from_block_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        typed_fill_value,
    )
    .unwrap();
    (erased, typed)
}

/// The c64 sibling of [`z2_oracle_pair`]. The imaginary part is deliberately
/// not proportional to the real one, so a stray conjugation or a real-only
/// path is visible in every comparison this pair feeds.
fn z2_complex_oracle_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, Complex64>,
) {
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let space = tenet::prelude::Space::z2([(0, 2), (1, 3)]);
    let erased = tenet::prelude::Tensor::from_block_fn(
        runtime,
        [&space, &space],
        [&space],
        |key: &tenet::prelude::BlockKey, indices: &[usize]| {
            complex(erased_fill_value(key, indices))
        },
    )
    .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        false,
    )
    .unwrap();
    let typed = TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], |sectors, indices| {
        complex(typed_fill_value(sectors, indices))
    })
    .unwrap();
    (erased, typed)
}

#[test]
fn typed_and_erased_permute_agree_byte_for_byte_on_a_builtin_rule() {
    // What: the typed permute is the erased permute, not a lookalike — same
    // destination layout, same block order, same bytes. A non-identity order is
    // used deliberately, so neither side can take an identity shortcut.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let erased_permuted = erased.permute(&[2, 0], &[1]).unwrap();
    let typed_permuted = typed.permute(&[2, 0], &[1]).unwrap();

    assert_eq!(typed_permuted.data(), erased_permuted.data());
    assert_ne!(typed_permuted.data(), typed.data());
}

// ---------------------------------------------------------------------------
// Phase 3, slice 3: `TensorMap::contract`.
// ---------------------------------------------------------------------------

/// A single-sector Z3 leg: the whole tensor map is then one dense block, so a
/// contraction result can be checked against a hand-computed matrix product.
fn z3_dense_leg(provider: &Arc<ExternalZ3>, degeneracy: usize) -> GradedSpace<ExternalZ3> {
    GradedSpace::try_new(Arc::clone(provider), [(Z3Charge(0), degeneracy)], false)
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
    // What: `0..open_rank` is the identity output order (the erased facade's
    // default `pAB`), so the same product comes back untransposed.
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
fn typed_and_erased_contract_agree_byte_for_byte_on_a_builtin_rule() {
    // What: the typed contraction is the erased `contract_ordered`, bytes and
    // layout, on a non-identity output order.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let erased_contracted = erased
        .contract_ordered(&erased, &[2], &[0], &[1, 0, 3, 2])
        .unwrap();
    let typed_contracted = typed.contract(&typed, &[2], &[0], &[1, 0, 3, 2]).unwrap();

    assert_eq!(typed_contracted.data(), erased_contracted.data());
    assert!(typed_contracted.data().iter().any(|&value| value != 0.0));
}

// ---------------------------------------------------------------------------
// Phase 3, slice 4: non-regression gates.
// ---------------------------------------------------------------------------

#[test]
fn a_typed_operation_reads_the_runtime_owned_transform_store() {
    // What: the typed permute runs on a context leased from the tensor's own
    // runtime, so it reads the completed transform the erased permute of the
    // same built-in layout already put in that runtime's store — one entry,
    // and a hit rather than a second computation. A typed path that executed
    // on an unbound default context would miss and add its own entry.
    //
    // What this deliberately does not claim: that the two facades use the same
    // execution lane. Lane identity has no observable — the store is owned by
    // the Runtime and keyed by rule identity, operation and interned structure
    // ids, so any leased context sees the same entries.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);
    runtime.clear_tree_transform_cache();
    assert_eq!(runtime.tree_transform_cache_info().entries(), 0);

    erased.permute(&[2, 0], &[1]).unwrap();
    let after_erased = runtime.tree_transform_cache_info().entries();
    assert_eq!(after_erased, 1);

    let hits_before = runtime.tree_transform_cache_info().hits();
    typed.permute(&[2, 0], &[1]).unwrap();

    let after_typed = runtime.tree_transform_cache_info();
    assert_eq!(after_typed.entries(), after_erased);
    // Non-vacuous: the typed run did not merely fail to add an entry, it read
    // the entry the erased run left behind.
    assert!(after_typed.hits() > hits_before);
}

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
    // What: the one facade-level pre-check — `levels` must name every source
    // axis — with the erased layer's own diagnosis, in both directions.
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

#[test]
fn typed_and_erased_braid_agree_byte_for_byte_on_a_builtin_rule() {
    // What: the typed braid is the erased braid — same destination layout, same
    // bytes — including how `levels` is split by the *source* codomain rank.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let erased_braided = erased.braid(&[2, 0], &[1], &[2, 0, 1]).unwrap();
    let typed_braided = typed.braid(&[2, 0], &[1], &[2, 0, 1]).unwrap();

    assert_eq!(typed_braided.data(), erased_braided.data());
    assert_ne!(typed_braided.data(), typed.data());

    // The source is `2 <- 1`, so this destination split (`1 <- 2`) has a
    // codomain axis list of a different length. That is what pins the levels
    // being split by the *source* codomain rank: splitting by the requested
    // codomain length instead is a plausible misreading, and one the case
    // above cannot see because there the two coincide.
    let erased_moved = erased.braid(&[2], &[0, 1], &[2, 0, 1]).unwrap();
    let typed_moved = typed.braid(&[2], &[0, 1], &[2, 0, 1]).unwrap();

    assert_eq!(typed_moved.data(), erased_moved.data());
    assert_eq!(typed_moved.codomain().len(), 1);
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

/// The same shape summary for an erased tensor, so a typed result can be
/// checked against the erased sibling rather than against a guess.
fn erased_leg_shapes(tensor: &tenet::prelude::Tensor) -> Vec<(bool, Vec<usize>)> {
    tensor
        .codomain_spaces()
        .iter()
        .chain(tensor.domain_spaces().iter())
        .map(|space| {
            (
                space.is_dual(),
                space
                    .sectors()
                    .into_iter()
                    .map(|(_, degeneracy)| degeneracy)
                    .collect(),
            )
        })
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
fn typed_and_erased_transpose_agree_byte_for_byte_on_a_builtin_rule() {
    // What: the typed transpose is the erased planar transpose — same bent
    // spaces (dual flags included) and same bytes. It must not be a permute in
    // disguise: on this rank-3 layout the two differ, which the assertion
    // against the permuted buffer pins.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let erased_transposed = erased.transpose().unwrap();
    let typed_transposed = typed.transpose().unwrap();

    assert_eq!(typed_transposed.data(), erased_transposed.data());
    assert_eq!(
        typed_leg_shapes(&typed_transposed),
        erased_leg_shapes(&erased_transposed)
    );
    assert_eq!(typed_transposed.codomain().len(), 1);
    assert_eq!(typed_transposed.domain().len(), 2);
}

#[test]
fn typed_and_erased_transpose_axes_agree_byte_for_byte_on_a_builtin_rule() {
    // What: an explicit cyclic rotation other than the full transpose agrees
    // with the erased sibling, bytes and bent spaces.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    // Planar source order is `0, 1, 2` (codomain then reversed domain); this is
    // the one-step rotation of it.
    let erased_rotated = erased.transpose_axes(&[1, 2], &[0]).unwrap();
    let typed_rotated = typed.transpose_axes(&[1, 2], &[0]).unwrap();

    assert_eq!(typed_rotated.data(), erased_rotated.data());
    assert_eq!(
        typed_leg_shapes(&typed_rotated),
        erased_leg_shapes(&erased_rotated)
    );
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
fn typed_and_erased_repartition_agree_on_bytes_and_on_the_bent_spaces() {
    // What: every split point matches the erased sibling in bytes and in the
    // resulting spaces. The space comparison is the load-bearing half: a leg
    // that crosses the boundary is bent, which flips its dual flag and dualizes
    // its sectors (so a Z2-even/odd degeneracy pair can reorder), and the
    // erased result — not an assumption about which way that goes — is the
    // reference.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    for num_codomain in 0..=3 {
        let erased_moved = erased.repartition(num_codomain).unwrap();
        let typed_moved = typed.repartition(num_codomain).unwrap();

        assert_eq!(typed_moved.data(), erased_moved.data(), "{num_codomain}");
        assert_eq!(
            typed_leg_shapes(&typed_moved),
            erased_leg_shapes(&erased_moved),
            "{num_codomain}"
        );
    }
    // Non-vacuous: at least one split really does flip a dual flag relative to
    // the source, so the comparison above is not comparing three copies of the
    // identity.
    let flipped = typed.repartition(0).unwrap();
    assert!(typed_leg_shapes(&flipped)
        .iter()
        .any(|(is_dual, _)| *is_dual));
    assert!(typed_leg_shapes(&typed)
        .iter()
        .all(|(is_dual, _)| !*is_dual));
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
    use tenet::typed::*;

    use super::{ExternalZ3, Z3Charge};

    #[test]
    fn a_glob_import_runs_an_end_to_end_typed_operation() {
        let _guard = super::cache_lock();
        let runtime: Runtime = Runtime::builder().build().expect("runtime builds");
        let provider = Arc::new(ExternalZ3::new());
        let leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [(Z3Charge(0), 2), (Z3Charge(1), 3)],
            false,
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
    GradedSpace::try_new(
        Arc::new(tenet::core::FermionParityFusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
        false,
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
// The byte oracles here are exact rather than gauge-tolerant: both facades
// call the same `*_dyn` seams, whose gauge fixing is deterministic, so a
// difference of a single bit is a real divergence and not floating-point
// weather.
// ---------------------------------------------------------------------------

/// Decodes an erased spectrum's raw ids into `Z2Irrep` labels, so an erased
/// spectrum can be compared to a typed one label-for-label.
fn erased_z2_spectrum(
    spectrum: &[tenet::prelude::SectorSpectrum],
) -> Vec<(tenet::core::Z2Irrep, Vec<f64>)> {
    let mut decoded: Vec<_> = spectrum
        .iter()
        .map(|entry| {
            (
                SectorCodec::decode_sector(&tenet::core::Z2FusionRule, entry.sector).unwrap(),
                entry.values.clone(),
            )
        })
        .collect();
    decoded.sort_by_key(|(sector, _)| *sector);
    decoded
}

fn typed_z2_spectrum(
    spectrum: &[tenet::typed::SectorSpectrum<tenet::core::Z2Irrep>],
) -> Vec<(tenet::core::Z2Irrep, Vec<f64>)> {
    spectrum
        .iter()
        .map(|entry| (entry.sector, entry.values.clone()))
        .collect()
}

#[test]
fn typed_and_erased_svd_compact_agree_byte_for_byte() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let (eu, es, evh) = erased.svd_compact().unwrap();
    let (tu, ts, tvh) = typed.svd_compact().unwrap();

    assert_eq!(tu.data(), eu.data());
    assert_eq!(tvh.data(), evh.data());
    // Both `s` factors are compact diagonal storage now (#570 closed), so they
    // are byte-comparable through their common materialization — which the
    // previous revision of this test could only assert a size ceiling on.
    // `data()` deliberately reports the dense buffer on both facades: the
    // `Σ_c k_c` storage claim underneath is what
    // `tests/typed_diagonal_allocations.rs` measures, since neither facade
    // publishes a compact accessor.
    assert_eq!(ts.data(), es.data());
    // One dense block per coupled sector, k_c² elements each: 2² + 3² = 13.
    // Kept as the shape of the materialization, not as a storage ceiling.
    assert_eq!(ts.data().len(), 13);
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
    let (_, typed) = z2_oracle_pair(&runtime);

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
fn typed_and_erased_svd_compact_agree_byte_for_byte_on_a_complex_payload() {
    // What: the payload dtype is a type parameter here and a stored `Dtype`
    // there, so c64 takes a different route through both facades. The
    // imaginary part is deliberately not proportional to the real one, so a
    // stray conjugation or a real-only path is visible.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_complex_oracle_pair(&runtime);
    assert_eq!(typed.data(), erased.try_data_c64().unwrap());

    let (eu, es, evh) = erased.svd_compact().unwrap();
    let (tu, ts, tvh) = typed.svd_compact().unwrap();

    assert_eq!(tu.data(), eu.try_data_c64().unwrap());
    assert_eq!(tvh.data(), evh.try_data_c64().unwrap());
    // Both `s` factors are compact diagonal storage (#570 closed), so `s`
    // compares bitwise through its materialization like the f64 sibling above,
    // not only through its spectrum.
    assert_eq!(ts.data(), es.try_data_c64().unwrap());
    assert_eq!(
        typed
            .svd_vals()
            .unwrap()
            .iter()
            .map(|entry| entry.values.clone())
            .collect::<Vec<_>>(),
        erased_z2_spectrum(&erased.svd_vals().unwrap())
            .iter()
            .map(|(_, values)| values.clone())
            .collect::<Vec<_>>()
    );

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
fn typed_and_erased_svd_full_agree_byte_for_byte() {
    // `svd_full`'s `s` is dense rectangular on both sides — TensorKit's own
    // shape — so unlike `svd_compact` all three factors compare bitwise.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let (eu, es, evh) = erased.svd_full().unwrap();
    let (tu, ts, tvh) = typed.svd_full().unwrap();

    assert_eq!(tu.data(), eu.data());
    assert_eq!(ts.data(), es.data());
    assert_eq!(tvh.data(), evh.data());
}

#[test]
fn typed_and_erased_svd_trunc_agree_and_report_the_discarded_weight() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let truncation = tenet::typed::Truncation::rank(2);
    let erased_out = erased.svd_trunc(&truncation).unwrap();
    let typed_out = typed.svd_trunc(&truncation).unwrap();

    assert_eq!(typed_out.u.data(), erased_out.u.data());
    assert_eq!(typed_out.vh.data(), erased_out.vh.data());
    assert_eq!(typed_out.error, erased_out.error);
    assert_eq!(
        typed_z2_spectrum(&typed_out.singular_values),
        erased_z2_spectrum(&erased_out.singular_values)
    );

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
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Z3Charge(0), 1), (Z3Charge(1), 2)],
        false,
    )
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
fn svd_vals_reports_decoded_labels_sorted_by_label() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let spectrum = typed.svd_vals().unwrap();
    assert_eq!(spectrum.len(), 2);
    // The oracle: same values as the erased facade, label for label.
    assert_eq!(
        typed_z2_spectrum(&spectrum),
        erased_z2_spectrum(&erased.svd_vals().unwrap())
    );
    assert!(spectrum.windows(2).all(|w| w[0].sector < w[1].sector));
    // Descending by magnitude within a sector, as the seam guarantees.
    for entry in &spectrum {
        assert!(entry.values.windows(2).all(|w| w[0] >= w[1]));
    }
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
fn typed_and_erased_qr_and_lq_agree_byte_for_byte() {
    // Both splits are exercised deliberately: `2 <- 1` is tall in every coupled
    // sector, where LQ-compact and LQ-full return the same factors and so
    // cannot tell the two seams apart; `1 <- 2` is wide, where they differ (and
    // symmetrically for QR).
    let _guard = cache_lock();
    let runtime = runtime();
    for num_codomain in [2, 1] {
        let (erased, typed) = z2_oracle_pair_split(&runtime, num_codomain);

        for (erased_out, typed_out) in [
            (erased.qr_compact().unwrap(), typed.qr_compact().unwrap()),
            (erased.qr_full().unwrap(), typed.qr_full().unwrap()),
            (erased.lq_compact().unwrap(), typed.lq_compact().unwrap()),
            (erased.lq_full().unwrap(), typed.lq_full().unwrap()),
        ] {
            assert_eq!(typed_out.0.data(), erased_out.0.data());
            assert_eq!(typed_out.1.data(), erased_out.1.data());
        }
    }
}

#[test]
fn left_and_right_orth_are_the_tensorkit_default_kinds() {
    // TensorKit 0.17 defaults `left_orth` to `:qr` and `right_orth` to `:lq`.
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_oracle_pair(&runtime);

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
fn typed_and_erased_null_spaces_agree_byte_for_byte() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    // `[v, v] <- [v]` is tall per coupled sector, so the left null space is
    // non-empty; the right one is empty, which is itself a shape the two
    // facades must agree on.
    assert_eq!(
        typed.left_null().unwrap().data(),
        erased.left_null().unwrap().data()
    );
    assert!(!typed.left_null().unwrap().data().is_empty());
    assert_eq!(
        typed.right_null().unwrap().data(),
        erased.right_null().unwrap().data()
    );
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
fn typed_and_erased_add_agree_byte_for_byte() {
    // What: `alpha * self + beta * other` is the erased combination, coefficient
    // for coefficient. The two coefficients are deliberately different and
    // neither is 1, so swapping them (or dropping one) moves the buffer.
    //
    // The second operand is a permute of the first: same space, same layout,
    // different values, and no second fixture — `permute` is already pinned
    // against the erased facade byte for byte.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);
    let erased_other = erased.permute(&[1, 0], &[2]).unwrap();
    let typed_other = typed.permute(&[1, 0], &[2]).unwrap();

    let erased_sum = erased.add(&erased_other, 2.0, -3.0).unwrap();
    let typed_sum = typed.add(&typed_other, 2.0, -3.0).unwrap();

    assert_eq!(typed_sum.data(), erased_sum.data());
    // The asymmetry is real: the swapped combination is a different tensor.
    assert_ne!(
        typed.add(&typed_other, -3.0, 2.0).unwrap().data(),
        typed_sum.data()
    );
}

#[test]
fn add_carries_complex_coefficients() {
    // What: `D` is the coefficient type too, so the c64 instantiation covers
    // the erased facade's separate `add_c64` with no second method here.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_complex_oracle_pair(&runtime);
    let alpha = Complex64::new(0.5, -2.0);
    let beta = Complex64::new(-1.5, 0.25);

    let erased_sum = erased
        .add_c64(&erased.permute(&[1, 0], &[2]).unwrap(), alpha, beta)
        .unwrap();
    let typed_sum = typed
        .add(&typed.permute(&[1, 0], &[2]).unwrap(), alpha, beta)
        .unwrap();

    assert_eq!(typed_sum.data(), erased_sum.try_data_c64().unwrap());
}

#[test]
fn add_rejects_a_different_runtime_and_a_different_space() {
    // What: the two checks this facade makes itself, in order — the runtime
    // identity the expert layer never sees, then the space equality that makes
    // the element-wise combination meaningful.
    let _guard = cache_lock();
    let other_runtime = runtime();
    let runtime = runtime();
    let (_, typed) = z2_oracle_pair(&runtime);
    let (_, elsewhere) = z2_oracle_pair(&other_runtime);
    let (_, other_split) = z2_oracle_pair_split(&runtime, 1);

    assert!(matches!(
        typed.add(&elsewhere, 1.0, 1.0).unwrap_err(),
        tenet::prelude::Error::RuntimeMismatch
    ));
    // The erased facade's `check_same_space` message, verbatim: one mistake
    // must not be reported two ways across the two facades.
    assert!(matches!(
        typed.add(&other_split, 1.0, 1.0).unwrap_err(),
        tenet::prelude::Error::InvalidArgument(message)
            if message == "tensors live on different spaces or block layouts"
    ));
}

#[test]
fn typed_and_erased_scale_agree_byte_for_byte() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    assert_eq!(typed.scale(-2.5).data(), erased.scale(-2.5).unwrap().data());

    let (erased_c, typed_c) = z2_complex_oracle_pair(&runtime);
    let factor = Complex64::new(0.25, 3.0);
    assert_eq!(
        typed_c.scale(factor).data(),
        erased_c.scale_c64(factor).unwrap().try_data_c64().unwrap()
    );
}

// ---------------------------------------------------------------------------
// Phase 5 (issue #568), slice 2: `norm`, `norm_inf`, `normalize`.
// ---------------------------------------------------------------------------

/// An SU(2) oracle pair over two legs, split into `num_codomain <- rest`: the
/// same tensor built through both facades on the built-in `SU2FusionRule`.
///
/// Why this fixture exists at all, next to the Z2 one: SU(2) is the only
/// non-abelian rule here, so it is the only one whose coupled sectors have
/// `dim(c) != 1`. Z2 is abelian and takes `weighted_inner`'s `Unique` fast
/// path, where the quantum-dimension weights are all one and therefore
/// invisible — every dimension-weighted operation needs this pair as well.
///
/// The fill is a plain counter, so the two buffers agree only if the two
/// facades walk blocks and elements in the same order; the helper asserts that
/// before handing the pair out.
fn su2_oracle_pair_split(
    runtime: &Runtime,
    num_codomain: usize,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::SU2FusionRule, f64>,
) {
    let space = tenet::prelude::Space::su2([(0, 1), (1, 2)]).unwrap();
    let spaces = [&space, &space];
    let (codomain, domain) = spaces.split_at(num_codomain);
    let mut next = 0.0;
    let erased = tenet::prelude::Tensor::from_block_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        |_: &tenet::prelude::BlockKey, _: &[usize]| {
            next += 1.0;
            next
        },
    )
    .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
        false,
    )
    .unwrap();
    let legs = [&leg, &leg];
    let (codomain, domain) = legs.split_at(num_codomain);
    let mut next = 0.0;
    let typed = TensorMap::from_block_fn(
        runtime,
        codomain.iter().copied(),
        domain.iter().copied(),
        |_, _| {
            next += 1.0;
            next
        },
    )
    .unwrap();
    assert_eq!(typed.data(), erased.data());
    (erased, typed)
}

/// The endomorphism split of [`su2_oracle_pair_split`]: `[v] <- [v]`, which is
/// what `tr` needs and what every other SU(2) assertion here happens to use.
fn su2_oracle_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::SU2FusionRule, f64>,
) {
    su2_oracle_pair_split(runtime, 1)
}

#[test]
fn typed_and_erased_norm_agree_including_the_dimension_weighted_branch() {
    // What: `norm` is TensorKit's quantum-dimension-weighted Frobenius norm on
    // both facades. The SU(2) half is the one that matters: there
    // `norm^2 != sum |x|^2`, so a weight-free implementation would pass the Z2
    // half and fail here.
    let _guard = cache_lock();
    let runtime = runtime();

    let (erased, typed) = z2_oracle_pair(&runtime);
    let unweighted: f64 = typed.data().iter().map(|value| value * value).sum();
    assert_eq!(typed.norm().unwrap(), erased.norm().unwrap());
    // Z2 is abelian: every dim(c) is one, so the weighting is the identity and
    // this fixture alone cannot see the weight at all.
    assert!((typed.norm().unwrap() - unweighted.sqrt()).abs() < 1e-12);

    let (erased, typed) = su2_oracle_pair(&runtime);
    let unweighted: f64 = typed.data().iter().map(|value| value * value).sum();
    assert_eq!(typed.norm().unwrap(), erased.norm().unwrap());
    assert!(
        (typed.norm().unwrap() - unweighted.sqrt()).abs() > 1.0,
        "the SU(2) fixture must actually separate the weighted norm from the \
         unweighted one, or it cannot kill a dropped dim(c)"
    );
}

#[test]
fn typed_and_erased_norm_inf_agree_and_are_not_dimension_weighted() {
    // What: TensorKit's `norm(t, Inf)` — the largest absolute stored entry,
    // with no quantum-dimension weighting. Checked on SU(2), where a weighted
    // implementation would differ.
    let _guard = cache_lock();
    let runtime = runtime();

    let (erased, typed) = su2_oracle_pair(&runtime);
    let largest = typed
        .data()
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    assert_eq!(typed.norm_inf().unwrap(), erased.norm_inf().unwrap());
    assert_eq!(typed.norm_inf().unwrap(), largest);
    assert_ne!(typed.norm_inf().unwrap(), typed.norm().unwrap());

    // c64: the modulus, not the real part.
    let (erased, typed) = z2_complex_oracle_pair(&runtime);
    assert_eq!(typed.norm_inf().unwrap(), erased.norm_inf().unwrap());
    assert!(typed
        .data()
        .iter()
        .all(|value| value.norm() <= typed.norm_inf().unwrap()));
}

#[test]
fn normalize_returns_a_unit_norm_tensor_matching_the_erased_facade() {
    let _guard = cache_lock();
    let runtime = runtime();

    // SU(2): normalizing by a dimension-weighted norm is what a plain
    // Frobenius normalization would get wrong, and only a non-abelian fixture
    // can see it.
    let (erased, typed) = su2_oracle_pair(&runtime);
    let unit = typed.normalize().unwrap();
    assert_eq!(unit.data(), erased.normalize().unwrap().data());
    assert!((unit.norm().unwrap() - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Phase 5 (issue #568), slice 3: `inner`, `dot`, `tr`.
// ---------------------------------------------------------------------------

/// A Z2 endomorphism oracle pair, `[v] <- [v]`: the abelian half of the `tr`
/// comparison, where every quantum dimension is one.
fn z2_endo_oracle_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, f64>,
) {
    let space = tenet::prelude::Space::z2([(0, 2), (1, 3)]);
    let erased =
        tenet::prelude::Tensor::from_block_fn(runtime, [&space], [&space], erased_fill_value)
            .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        false,
    )
    .unwrap();
    let typed = TensorMap::from_block_fn(runtime, [&leg], [&leg], typed_fill_value).unwrap();
    (erased, typed)
}

#[test]
fn typed_and_erased_inner_agree_including_the_dimension_weighted_branch() {
    // What: `inner` is TensorKit's `dot(x, y)` — conjugate-linear in the first
    // argument and quantum-dimension weighted — on both facades, and it comes
    // back as a `D` rather than the erased `Scalar` enum. SU(2) is what
    // exercises the weighted branch; Z2 alone would take the abelian fast path.
    let _guard = cache_lock();
    let runtime = runtime();

    let (z2_erased, z2_typed) = z2_oracle_pair(&runtime);
    let (su2_erased, su2_typed) = su2_oracle_pair(&runtime);
    // The two providers are different types, so the shared assertions live in a
    // closure over the pair of scalars rather than in a loop over the pairs.
    let agree = |typed_value: f64, erased_value: f64, norm: f64| {
        assert_eq!(typed_value, erased_value);
        // `<t, t>` is the squared norm, which is the identity that pins this
        // weighting to `norm`'s.
        assert!((typed_value - norm * norm).abs() < 1e-9 * norm * norm);
    };
    agree(
        z2_typed.inner(&z2_typed).unwrap(),
        z2_erased.inner(&z2_erased).unwrap().re(),
        z2_typed.norm().unwrap(),
    );
    agree(
        su2_typed.inner(&su2_typed).unwrap(),
        su2_erased.inner(&su2_erased).unwrap().re(),
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
    let (erased, typed) = z2_complex_oracle_pair(&runtime);
    // The extra `i` is what makes the product genuinely complex: this fixture's
    // imaginary part is a function of its real one, so the plain permuted
    // partner happens to give a real inner product and could not see a phase.
    let imaginary = Complex64::new(0.0, 1.0);
    let other = typed.permute(&[1, 0], &[2]).unwrap().scale(imaginary);
    let erased_other = erased
        .permute(&[1, 0], &[2])
        .unwrap()
        .scale_c64(imaginary)
        .unwrap();

    let value = typed.inner(&other).unwrap();
    assert_eq!(value, erased.inner(&erased_other).unwrap().to_c64());
    assert_eq!(value, other.inner(&typed).unwrap().conj());
    assert_ne!(value, other.inner(&typed).unwrap());
    assert!(value.im.abs() > 1e-6, "the fixture must have a real phase");
}

#[test]
fn dot_is_inner() {
    // The erased `dot` is a plain alias for `inner`; so is this one, and the
    // two names must not be able to come to mean different things.
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_complex_oracle_pair(&runtime);
    let other = typed.permute(&[1, 0], &[2]).unwrap();

    assert_eq!(typed.dot(&other).unwrap(), typed.inner(&other).unwrap());
}

#[test]
fn inner_rejects_a_different_runtime_and_a_different_space() {
    let _guard = cache_lock();
    let other_runtime = runtime();
    let runtime = runtime();
    let (_, typed) = z2_oracle_pair(&runtime);
    let (_, elsewhere) = z2_oracle_pair(&other_runtime);
    let (_, other_split) = z2_oracle_pair_split(&runtime, 1);

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
fn typed_and_erased_tr_agree_including_the_dimension_weighted_branch() {
    // What: TensorKit's positive trace `Σ_c dim(c) * tr(b_c)`. The SU(2) half
    // separates it from the unweighted diagonal sum; the Z2 half is where the
    // two coincide.
    let _guard = cache_lock();
    let runtime = runtime();

    let (erased, typed) = z2_endo_oracle_pair(&runtime);
    assert_eq!(typed.tr().unwrap(), erased.tr().unwrap().re());

    let (erased, typed) = su2_oracle_pair(&runtime);
    let trace = typed.tr().unwrap();
    assert_eq!(trace, erased.tr().unwrap().re());
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
    // The erased facade's own message, verbatim.
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_oracle_pair(&runtime);

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
fn typed_and_erased_adjoint_agree_byte_for_byte() {
    // What: the eager typed adjoint is the erased lazy view's materialized
    // buffer, byte for byte — the divergence is in when the work happens, not
    // in what comes out. The spaces swap sides, which the shape assertions pin.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    let adjoint = typed.adjoint().unwrap();

    assert_eq!(adjoint.data(), erased.adjoint().unwrap().data());
    assert_eq!(adjoint.codomain().len(), typed.domain().len());
    assert_eq!(adjoint.domain().len(), typed.codomain().len());
    // Its own inverse, as a dagger must be.
    assert_eq!(adjoint.adjoint().unwrap().data(), typed.data());
}

#[test]
fn adjoint_conjugates_a_complex_payload() {
    // What: c64 entries come back conjugated, not merely transposed. Compared
    // as multisets, because the transpose moves entries around: the point is
    // that the *values* are the conjugated ones and not the original ones.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_complex_oracle_pair(&runtime);

    let adjoint = typed.adjoint().unwrap();
    assert_eq!(
        adjoint.data(),
        erased.adjoint().unwrap().try_data_c64().unwrap()
    );

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

/// The erased sibling of [`fermionic_rank_three`], as an endomorphism
/// `[v] <- [v]`: the shape `tr` needs, on the one provider here whose braiding
/// is not symmetric.
fn fermionic_endo_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::FermionParityFusionRule, f64>,
) {
    let space = tenet::prelude::Space::fz2([(0, 1), (1, 1)]).unwrap();
    let mut next = 0.0;
    let erased = tenet::prelude::Tensor::from_block_fn(
        runtime,
        [&space],
        [&space],
        |_: &tenet::prelude::BlockKey, _: &[usize]| {
            next += 1.0;
            next
        },
    )
    .unwrap();
    let leg = fermionic_leg();
    let mut next = 0.0;
    let typed = TensorMap::from_block_fn(runtime, [&leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap();
    assert_eq!(typed.data(), erased.data());
    (erased, typed)
}

#[test]
fn typed_and_erased_trace_pairs_agree_byte_for_byte() {
    // What: the full trace to a rank-0 tensor and a partial trace that leaves a
    // leg open, both against the erased sibling. The partial case is the one
    // that exercises the output-axis derivation and the destination's
    // codomain rank; the full case is the degenerate one.
    let _guard = cache_lock();
    let runtime = runtime();

    let (erased, typed) = z2_endo_oracle_pair(&runtime);
    let full = typed.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(full.data(), erased.trace_pairs(&[(0, 1)]).unwrap().data());
    assert_eq!(full.data().len(), 1);

    // `[v, v] <- [v]`: tracing axis 1 against axis 2 leaves axis 0 open, so the
    // result is `[v] <- []` and the open axis keeps its side.
    let (erased, typed) = z2_oracle_pair(&runtime);
    let partial = typed.trace_pairs(&[(1, 2)]).unwrap();
    assert_eq!(
        partial.data(),
        erased.trace_pairs(&[(1, 2)]).unwrap().data()
    );
    assert_eq!(partial.codomain().len(), 1);
    assert_eq!(partial.domain().len(), 0);

    // The two cases above leave at most one survivor, and it is codomain-side,
    // so neither can see the order of `output_axes` nor the codomain-rank
    // filter that splits the destination. These two can.
    //
    // `[v] <- [v, v]`, tracing (0, 1): the survivor is axis 2, a domain-side
    // leg, so the destination is `[] <- [v]` — a dropped codomain-rank filter
    // would put it in the codomain instead.
    let (erased, typed) = z2_oracle_pair_split(&runtime, 1);
    let survivor = typed.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(
        survivor.data(),
        erased.trace_pairs(&[(0, 1)]).unwrap().data()
    );
    assert_eq!(survivor.codomain().len(), 0);
    assert_eq!(survivor.domain().len(), 1);

    // `[v, v] <- [v, v]`, tracing (0, 3): two survivors, axes 1 and 2, one on
    // each side — so their relative order in `output_axes` is observable, and
    // reversing it changes the bytes.
    let space = tenet::prelude::Space::z2([(0, 2), (1, 3)]);
    let erased = tenet::prelude::Tensor::from_block_fn(
        &runtime,
        [&space, &space],
        [&space, &space],
        erased_fill_value,
    )
    .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        false,
    )
    .unwrap();
    let typed: TensorMap<tenet::core::Z2FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], typed_fill_value).unwrap();
    let two_survivors = typed.trace_pairs(&[(0, 3)]).unwrap();
    assert_eq!(
        two_survivors.data(),
        erased.trace_pairs(&[(0, 3)]).unwrap().data()
    );
    assert_eq!(two_survivors.codomain().len(), 1);
    assert_eq!(two_survivors.domain().len(), 1);
}

#[test]
fn trace_pairs_agrees_with_the_erased_facade_on_a_non_abelian_rule() {
    // SU(2): the categorical trace coefficients are not all one here, so this
    // is where a coefficient-free partial trace would diverge.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = su2_oracle_pair(&runtime);

    assert_eq!(
        typed.trace_pairs(&[(0, 1)]).unwrap().data(),
        erased.trace_pairs(&[(0, 1)]).unwrap().data()
    );
}

#[test]
fn trace_pairs_of_nothing_is_the_source() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_oracle_pair(&runtime);

    let traced = typed.trace_pairs(&[]).unwrap();

    assert_eq!(traced.data(), typed.data());
    assert_eq!(traced.codomain().len(), 2);
}

#[test]
fn trace_pairs_rejects_malformed_pairs() {
    // Out of range and repeated axes, both with the erased facade's message.
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_endo_oracle_pair(&runtime);

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
    // numbers differ, and both match their erased siblings.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = fermionic_endo_pair(&runtime);

    let positive = typed.tr().unwrap();
    let super_trace = typed.trace_pairs(&[(0, 1)]).unwrap();

    assert_eq!(positive, erased.tr().unwrap().re());
    assert_eq!(
        super_trace.data(),
        erased.trace_pairs(&[(0, 1)]).unwrap().data()
    );
    assert_ne!(
        super_trace.data(),
        [positive],
        "the fermionic supertrace must not coincide with the positive trace"
    );
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #569), slice 1: `TensorMap::compose`.
// ---------------------------------------------------------------------------

/// Fill value from an fz2 fusion-tree key, weighted by position exactly as
/// [`erased_fill_value`] is: the fermionic pair below needs a fill whose every
/// element is distinct, so a sign flip on any single block is visible.
fn fermionic_erased_fill(key: &tenet::prelude::BlockKey, indices: &[usize]) -> f64 {
    let pair = key.as_fusion_tree_pair().expect("fusion-tree block");
    let parity = |id| {
        SectorCodec::decode_sector(&tenet::core::FermionParityFusionRule, id)
            .expect("built-in codec decodes its own ids")
            .parity()
    };
    let mut value = f64::from(parity(pair.codomain_tree().coupled())) * 1000.0;
    for (position, &id) in pair.codomain_tree().uncoupled().iter().enumerate() {
        value += f64::from(parity(id)) * 100.0 * (position + 1) as f64;
    }
    for (position, &id) in pair.domain_tree().uncoupled().iter().enumerate() {
        value += f64::from(parity(id)) * 10.0 * (position + 1) as f64;
    }
    value + 1.0 + indices.iter().sum::<usize>() as f64
}

/// The same value from the typed labels, so the two facades' operands are the
/// same tensor before either composition runs.
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

/// `a : [v] <- [v*, v]` and `b : [v*, v] <- [v]` on both facades.
///
/// Composition contracts two legs here, exactly one of which is **dual** —
/// and a dual leg is the only place a fermionic twist can act. The mixed pair
/// is deliberate: with every contracted leg dual, "twist the dual contracted
/// legs" and "twist all contracted legs" would be the same statement, and the
/// twist identity below would not be pinned to a leg set at all.
#[allow(clippy::type_complexity)]
fn fermionic_compose_pair(
    runtime: &Runtime,
) -> (
    (tenet::prelude::Tensor, tenet::prelude::Tensor),
    (
        TensorMap<tenet::core::FermionParityFusionRule, f64>,
        TensorMap<tenet::core::FermionParityFusionRule, f64>,
    ),
) {
    let space = tenet::prelude::Space::fz2([(0, 1), (1, 2)]).unwrap();
    let dual = space.dual();
    let erased_a = tenet::prelude::Tensor::from_block_fn(
        runtime,
        [&space],
        [&dual, &space],
        fermionic_erased_fill,
    )
    .unwrap();
    let erased_b = tenet::prelude::Tensor::from_block_fn(
        runtime,
        [&dual, &space],
        [&space],
        fermionic_erased_fill,
    )
    .unwrap();

    let leg = fermionic_leg_with(&[1, 2]);
    let leg_dual = leg.try_dual().unwrap();
    let typed_a =
        TensorMap::from_block_fn(runtime, [&leg], [&leg_dual, &leg], fermionic_typed_fill).unwrap();
    let typed_b =
        TensorMap::from_block_fn(runtime, [&leg_dual, &leg], [&leg], fermionic_typed_fill).unwrap();

    // The operands themselves must be one tensor on both facades, or a
    // difference downstream would not be about composition at all.
    assert_eq!(typed_a.data(), erased_a.data());
    assert_eq!(typed_b.data(), erased_b.data());
    ((erased_a, erased_b), (typed_a, typed_b))
}

/// [`fermionic_leg`] with explicit `[even, odd]` degeneracies.
fn fermionic_leg_with(
    degeneracies: &[usize; 2],
) -> GradedSpace<tenet::core::FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(tenet::core::FermionParityFusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, degeneracies[0]),
            (tenet::core::Z2Irrep::ODD, degeneracies[1]),
        ],
        false,
    )
    .expect("fermionic leg is well formed")
}

#[test]
fn typed_and_erased_compose_agree_byte_for_byte_on_a_fermionic_provider() {
    // What: the typed compose is the erased compose, on the one provider whose
    // braiding is not symmetric — so this pins the fermionic signs, not just
    // the arithmetic. A bosonic-only compose passes every other test here and
    // fails this one.
    let _guard = cache_lock();
    let runtime = runtime();
    let ((erased_a, erased_b), (typed_a, typed_b)) = fermionic_compose_pair(&runtime);

    let typed = typed_a.compose(&typed_b).unwrap();

    assert_eq!(typed.data(), erased_a.compose(&erased_b).unwrap().data());
    assert_eq!(typed.codomain().len(), 1);
    assert_eq!(typed.domain().len(), 1);
}

#[test]
fn fermionic_compose_is_contract_against_a_twisted_right_operand() {
    // What: the exact relation between the two contraction semantics —
    // `compose(a, b) == contract(a, twist(b, b's dual codomain legs))`. The
    // twisted operand is built through `from_block_fn` rather than through the
    // erased facade's `twist`: theta is -1 on an odd sector and +1 otherwise,
    // so the twisted tensor is a one-line fill here, and building it typed
    // keeps the identity inside this facade instead of borrowing the erased
    // one to state it.
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, (typed_a, typed_b)) = fermionic_compose_pair(&runtime);

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
    let (_, (typed_a, typed_b)) = fermionic_compose_pair(&runtime);

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
    // the two semantics coincide — and the typed result still matches the
    // erased sibling byte for byte.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);

    let composed = typed.compose(&typed).unwrap();

    assert_eq!(composed.data(), erased.compose(&erased).unwrap().data());
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
    let (_, tall) = z2_oracle_pair_split(&runtime, 2);
    let (_, wide) = z2_oracle_pair_split(&runtime, 1);

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
    let (_, left) = z2_endo_oracle_pair(&runtime());
    let (_, right) = z2_endo_oracle_pair(&runtime());

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
    let (_, endo) = z2_endo_oracle_pair(&runtime);
    let (_, tall) = z2_oracle_pair_split(&runtime, 2);
    let (_, wide) = z2_oracle_pair_split(&runtime, 1);

    // Rank mismatch in both directions: one domain leg against two codomain
    // legs, and two domain legs against one codomain leg.
    assert!(endo.compose(&tall).is_err());
    assert!(wide.compose(&endo).is_err());

    // Matching ranks, mismatched legs: the degeneracies differ, so the two do
    // not meet even though the shapes line up.
    let z2 = Arc::new(tenet::core::Z2FusionRule);
    let narrow_leg = GradedSpace::try_new(
        Arc::clone(&z2),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
        false,
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
    let wide_leg = GradedSpace::try_new(
        Arc::clone(&z2),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        true,
    )
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
        GradedSpace::try_new(Arc::clone(&z2), [(tenet::core::Z2Irrep::EVEN, 2)], false).unwrap();
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
fn typed_and_erased_id_agree_byte_for_byte() {
    // What: the typed identity is the erased one, including on a leg list whose
    // per-sector degeneracies differ from leg to leg — the case where the
    // diagonal offsets inside a coupled-sector block are not all the same.
    let _guard = cache_lock();
    let runtime = runtime();
    let z2 = Arc::new(tenet::core::Z2FusionRule);
    let typed_leg = |even, odd| {
        GradedSpace::try_new(
            Arc::clone(&z2),
            [
                (tenet::core::Z2Irrep::EVEN, even),
                (tenet::core::Z2Irrep::ODD, odd),
            ],
            false,
        )
        .unwrap()
    };

    let wide = typed_leg(2, 3);
    let narrow = typed_leg(1, 4);
    let typed = TensorMap::<_, f64>::id(&runtime, [&wide, &narrow]).unwrap();

    let erased = tenet::prelude::Tensor::id(
        &runtime,
        tenet::prelude::Dtype::F64,
        [
            &tenet::prelude::Space::z2([(0, 2), (1, 3)]),
            &tenet::prelude::Space::z2([(0, 1), (1, 4)]),
        ],
    )
    .unwrap();

    assert_eq!(typed.data(), erased.data());
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
    let (_, typed) = z2_oracle_pair(&runtime);

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
    let (_, (typed_a, _)) = fermionic_compose_pair(&runtime);

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
// Every oracle here is against the erased facade, which keeps diagonal storage
// on exactly the same paths — so an assertion that only checked values would
// pass even if this facade had densified. The storage claim itself is measured
// in `tests/typed_diagonal_allocations.rs`; what these pin is that the compact
// route computes the same tensor the dense one does.
// ---------------------------------------------------------------------------

/// The `(erased, typed)` `s` factors of one Z2 oracle pair: two spectrum
/// tensors on the same bond space, one per facade.
fn z2_spectrum_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, f64>,
) {
    let (erased, typed) = z2_oracle_pair(runtime);
    (
        erased.svd_compact().unwrap().1,
        typed.svd_compact().unwrap().1,
    )
}

#[test]
fn compact_scale_and_adjoint_agree_with_the_erased_facade() {
    // What: both keep the payload compact on both facades, so `data()` is the
    // shared materialization of the same spectrum. `adjoint` on a real spectrum
    // is the identity — a bond space is its own adjoint — and the erased
    // sibling says so too.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_spectrum_pair(&runtime);

    assert_eq!(typed.scale(-2.5).data(), erased.scale(-2.5).unwrap().data());
    assert_eq!(
        typed.adjoint().unwrap().data(),
        erased.adjoint().unwrap().data()
    );
    // The values themselves, not just their agreement: a dropped factor would
    // agree with a sibling that dropped it too, but not with the source.
    for (scaled, original) in typed.scale(-2.5).data().iter().zip(typed.data()) {
        assert_eq!(*scaled, -2.5 * original);
    }
}

#[test]
fn compact_reductions_agree_with_the_erased_facade() {
    // What: `norm`, `norm_inf`, `tr` and `inner` read the stored spectrum
    // instead of its materialization, and land on the same numbers.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_spectrum_pair(&runtime);

    assert_eq!(typed.norm().unwrap(), erased.norm().unwrap());
    assert_eq!(typed.norm_inf().unwrap(), erased.norm_inf().unwrap());
    assert_eq!(typed.tr().unwrap(), erased.tr().unwrap().re());
    assert_eq!(
        typed.inner(&typed).unwrap(),
        erased.inner(&erased).unwrap().re()
    );
    assert_eq!(typed.dot(&typed).unwrap(), typed.inner(&typed).unwrap());
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
    let (erased, typed) = su2_oracle_pair(&runtime);
    let erased = erased.svd_compact().unwrap().1;
    let typed = typed.svd_compact().unwrap().1;

    assert_eq!(typed.norm().unwrap(), erased.norm().unwrap());
    assert_eq!(typed.tr().unwrap(), erased.tr().unwrap().re());
    assert_eq!(
        typed.inner(&typed).unwrap(),
        erased.inner(&erased).unwrap().re()
    );
    // The unweighted sum, for contrast: a dropped `dim(c)` would make `tr` this.
    let unweighted: f64 = typed.data().iter().sum();
    assert!(
        (typed.tr().unwrap() - unweighted).abs() > 1e-6,
        "the SU(2) spectrum trace is not dimension weighted"
    );
    // `norm_inf` is deliberately *not* weighted, on either facade.
    assert_eq!(typed.norm_inf().unwrap(), erased.norm_inf().unwrap());
}

#[test]
fn compact_add_agrees_with_the_erased_facade_on_both_arms() {
    // What: diagonal + diagonal stays diagonal, and diagonal + dense goes dense
    // — the same two arms the erased facade takes, with the same values.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_spectrum_pair(&runtime);

    assert_eq!(
        typed.add(&typed, 0.75, -0.5).unwrap().data(),
        erased.add(&erased, 0.75, -0.5).unwrap().data()
    );

    // A dense operand on the same bond space: `id` is the cheapest one, and it
    // is not diagonal *storage* on either facade even though its values are.
    let erased_dense = tenet::prelude::Tensor::id(
        &runtime,
        tenet::prelude::Dtype::F64,
        &erased.domain_spaces(),
    )
    .unwrap()
    .scale(3.0)
    .unwrap();
    let typed_dense = TensorMap::id(&runtime, &typed.domain()).unwrap().scale(3.0);
    assert_eq!(typed_dense.data(), erased_dense.data());

    for (alpha, beta) in [(0.75, -0.5), (1.0, 1.0)] {
        assert_eq!(
            typed.add(&typed_dense, alpha, beta).unwrap().data(),
            erased.add(&erased_dense, alpha, beta).unwrap().data(),
            "diagonal + dense disagrees at ({alpha}, {beta})"
        );
        assert_eq!(
            typed_dense.add(&typed, alpha, beta).unwrap().data(),
            erased_dense.add(&erased, alpha, beta).unwrap().data(),
            "dense + diagonal disagrees at ({alpha}, {beta})"
        );
    }
}

#[test]
fn compose_takes_the_compact_paths_and_reconstructs_the_source() {
    // What: `u * s * vh` now runs through the bond-scaling arms rather than a
    // dense GEMM, and still reproduces the source — plus `s * s`, the
    // diagonal-times-diagonal arm, against its erased sibling.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);
    let (eu, es, evh) = erased.svd_compact().unwrap();
    let (tu, ts, tvh) = typed.svd_compact().unwrap();

    // `t * D`, `D * t` and `D * D`, each against the erased facade's own arm.
    assert_eq!(
        tu.compose(&ts).unwrap().data(),
        eu.compose(&es).unwrap().data()
    );
    assert_eq!(
        ts.compose(&tvh).unwrap().data(),
        es.compose(&evh).unwrap().data()
    );
    assert_eq!(
        ts.compose(&ts).unwrap().data(),
        es.compose(&es).unwrap().data()
    );
    // `s * s` is still a spectrum: its entries are the squares.
    for (squared, original) in ts.compose(&ts).unwrap().data().iter().zip(ts.data()) {
        assert!((squared - original * original).abs() < 1e-12);
    }

    let recon = tu.compose(&ts).unwrap().compose(&tvh).unwrap();
    assert_eq!(recon.data().len(), typed.data().len());
    for (got, want) in recon.data().iter().zip(typed.data()) {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "{got} vs {want}"
        );
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
    let (_, wide) = z2_oracle_pair_split(&runtime, 2);
    // A second endomorphism on a leg with different degeneracies, so its bond
    // space genuinely differs from `wide`'s rather than merely being a second
    // allocation of the same one.
    let narrow_leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
        false,
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

/// A Hermitian endomorphism through both facades: `p = t + t†`, which is
/// Hermitian by construction on every provider, so `eigh` is defined on it
/// without either facade having to project first.
fn z2_hermitian_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, f64>,
) {
    let (erased, typed) = z2_endo_oracle_pair(runtime);
    (
        erased.add(&erased.adjoint().unwrap(), 1.0, 1.0).unwrap(),
        typed.add(&typed.adjoint().unwrap(), 1.0, 1.0).unwrap(),
    )
}

fn su2_hermitian_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::SU2FusionRule, f64>,
) {
    let (erased, typed) = su2_oracle_pair(runtime);
    (
        erased.add(&erased.adjoint().unwrap(), 1.0, 1.0).unwrap(),
        typed.add(&typed.adjoint().unwrap(), 1.0, 1.0).unwrap(),
    )
}

#[test]
fn typed_and_erased_eigh_full_agree_byte_for_byte() {
    // What: the same seam on both facades, so `v` compares bitwise and `d` —
    // compact on both — compares through its shared materialization. The
    // return is `(d, v)`, which is what the destructuring here pins.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_hermitian_pair(&runtime);

    let (ed, ev) = erased.eigh_full().unwrap();
    let (td, tv) = typed.eigh_full().unwrap();

    assert_eq!(td.data(), ed.data());
    assert_eq!(tv.data(), ev.data());
    // `d` really is the eigenvalue factor and `v` the eigenbasis, not the other
    // way round: `v` is an isometry and `d` is diagonal. A swapped return would
    // fail both.
    assert_eq!(td.data().len(), 13, "d is the bond <- bond factor");
    assert_eq!(tv.data().len(), 13, "v is the codomain <- bond factor");
    assert_eq!(
        td.eigh_vals().unwrap(),
        typed.eigh_vals().unwrap(),
        "d carries the source's own spectrum on its diagonal"
    );
}

#[test]
fn eigh_full_reconstructs_the_source_through_compose() {
    // What: `v * d * v†` is the source. The middle composition takes the
    // compact bond-scaling arm, so this exercises the storage as well as the
    // factorization.
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_hermitian_pair(&runtime);

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
fn typed_and_erased_eigh_vals_agree_including_the_su2_branch() {
    // What: the spectra match label for label. SU(2) is here because its
    // dimension-weighted machinery runs underneath the eigenvalue enumeration
    // even though the eigenvalues themselves are not weighted.
    let _guard = cache_lock();
    let runtime = runtime();

    let (erased, typed) = z2_hermitian_pair(&runtime);
    assert_eq!(
        typed_z2_spectrum(&typed.eigh_vals().unwrap()),
        erased_z2_spectrum(&erased.eigh_vals().unwrap())
    );

    let (erased, typed) = su2_hermitian_pair(&runtime);
    let typed_values: Vec<Vec<f64>> = typed
        .eigh_vals()
        .unwrap()
        .iter()
        .map(|entry| entry.values.clone())
        .collect();
    let mut erased_entries = erased.eigh_vals().unwrap();
    erased_entries.sort_by_key(|entry| {
        SectorCodec::decode_sector(&tenet::core::SU2FusionRule, entry.sector).unwrap()
    });
    let erased_values: Vec<Vec<f64>> = erased_entries
        .iter()
        .map(|entry| entry.values.clone())
        .collect();
    assert_eq!(typed_values, erased_values);
    assert!(
        erased_values.len() > 1,
        "the SU(2) fixture must span more than one coupled sector"
    );
}

#[test]
fn typed_and_erased_eigh_trunc_agree_and_report_the_discarded_weight() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_hermitian_pair(&runtime);
    let truncation = Truncation::rank(3);

    let erased_out = erased.eigh_trunc(&truncation).unwrap();
    let typed_out = typed.eigh_trunc(&truncation).unwrap();

    assert_eq!(typed_out.d.data(), erased_out.d.data());
    assert_eq!(typed_out.v.data(), erased_out.v.data());
    assert_eq!(typed_out.error, erased_out.error);
    assert_eq!(
        typed_z2_spectrum(&typed_out.eigenvalues),
        erased_z2_spectrum(&erased_out.eigenvalues)
    );
    // Something was actually discarded, so `error` is not vacuously zero.
    assert!(typed_out.error > 0.0);
    assert!(typed_out.d.data().len() < typed.eigh_full().unwrap().0.data().len());
}

#[test]
fn eigh_reports_a_non_hermitian_input_rather_than_a_wrong_answer() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_endo_oracle_pair(&runtime);

    assert!(typed.eigh_full().is_err());
    assert!(typed.eigh_vals().is_err());
    assert!(typed.eigh_trunc(&Truncation::Full).is_err());
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #570), slice 3: the general eigendecompositions.
// ---------------------------------------------------------------------------

/// The c64 endomorphism pair `eig` needs: [`z2_complex_oracle_pair`] is
/// rank three, and `eig` is defined on square maps only.
fn z2_complex_endo_oracle_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, Complex64>,
) {
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let space = tenet::prelude::Space::z2([(0, 2), (1, 3)]);
    let erased = tenet::prelude::Tensor::from_block_fn(
        runtime,
        [&space],
        [&space],
        |key: &tenet::prelude::BlockKey, indices: &[usize]| {
            complex(erased_fill_value(key, indices))
        },
    )
    .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        false,
    )
    .unwrap();
    let typed = TensorMap::from_block_fn(runtime, [&leg], [&leg], |trees, indices| {
        complex(typed_fill_value(trees, indices))
    })
    .unwrap();
    (erased, typed)
}

#[test]
fn typed_and_erased_eig_full_agree_on_a_real_payload() {
    // What: the factors are complex for a real input on both facades —
    // TensorKit's `eigen` promotes too — and `d` is the compact spectrum.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);

    let (ed, ev) = erased.eig_full().unwrap();
    let (td, tv) = typed.eig_full().unwrap();

    assert_eq!(td.data(), ed.try_data_c64().unwrap());
    assert_eq!(tv.data(), ev.try_data_c64().unwrap());
}

#[test]
fn typed_and_erased_eig_agree_on_a_complex_payload_and_conjugate_the_adjoint() {
    // What: the c64 route, where the spectrum is genuinely complex rather than
    // a real one widened. `d.adjoint()` must conjugate it — on a real spectrum
    // that is invisible, which is why the check lives here.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_complex_endo_oracle_pair(&runtime);

    let (ed, ev) = erased.eig_full().unwrap();
    let (td, tv) = typed.eig_full().unwrap();
    assert_eq!(td.data(), ed.try_data_c64().unwrap());
    assert_eq!(tv.data(), ev.try_data_c64().unwrap());

    // Genuinely complex, so a missing conjugation is observable.
    assert!(
        td.data().iter().any(|value| value.im.abs() > 1e-6),
        "the eig spectrum must be off the real axis for this to test anything"
    );
    let adjoint = td.adjoint().unwrap();
    assert_eq!(
        adjoint.data(),
        ed.adjoint().unwrap().try_data_c64().unwrap()
    );
    for (conjugated, original) in adjoint.data().iter().zip(td.data()) {
        assert_eq!(*conjugated, original.conj());
    }

    // The compact reductions on a genuinely complex spectrum. Every other
    // compact oracle in this suite reads an SVD spectrum, which is real, so a
    // dropped conjugation in the compact inner product is invisible there:
    // `Σ conj(a) a` and `Σ a a` agree on the reals. Here they do not.
    assert_eq!(td.inner(&td).unwrap(), ed.inner(&ed).unwrap().to_c64());
    assert_eq!(td.norm().unwrap(), ed.norm().unwrap());
    assert_eq!(td.tr().unwrap(), ed.tr().unwrap().to_c64());
    // `<d, d>` is real and positive precisely because the first argument is
    // conjugated; the unconjugated sum of squares is not.
    let unconjugated: Complex64 = td.data().iter().map(|value| value * value).sum();
    assert!(
        unconjugated.im.abs() > 1e-6,
        "the spectrum must make the unconjugated sum complex for this to test anything"
    );
    assert_eq!(td.inner(&td).unwrap().im, 0.0);
    let norm = td.norm().unwrap();
    assert!((td.inner(&td).unwrap().re - norm * norm).abs() < 1e-9 * norm * norm);
}

#[test]
fn typed_and_erased_eig_vals_and_eig_trunc_agree() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);

    let decode = |spectrum: &[tenet::prelude::SectorSpectrum<Complex64>]| {
        let mut decoded: Vec<_> = spectrum
            .iter()
            .map(|entry| {
                (
                    SectorCodec::decode_sector(&tenet::core::Z2FusionRule, entry.sector).unwrap(),
                    entry.values.clone(),
                )
            })
            .collect();
        decoded.sort_by_key(|(sector, _): &(tenet::core::Z2Irrep, _)| *sector);
        decoded
    };
    let typed_decoded: Vec<_> = typed
        .eig_vals()
        .unwrap()
        .iter()
        .map(|entry| (entry.sector, entry.values.clone()))
        .collect();
    assert_eq!(typed_decoded, decode(&erased.eig_vals().unwrap()));

    let truncation = Truncation::rank(3);
    let erased_out = erased.eig_trunc(&truncation).unwrap();
    let typed_out = typed.eig_trunc(&truncation).unwrap();
    assert_eq!(typed_out.d.data(), erased_out.d.try_data_c64().unwrap());
    assert_eq!(typed_out.v.data(), erased_out.v.try_data_c64().unwrap());
    assert_eq!(typed_out.error, erased_out.error);
    assert_eq!(
        typed_out
            .eigenvalues
            .iter()
            .map(|entry| (entry.sector, entry.values.clone()))
            .collect::<Vec<_>>(),
        decode(&erased_out.eigenvalues)
    );
    assert!(typed_out.error > 0.0);
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #570), slice 4: the `is_hermitian` / `project_*` family.
// ---------------------------------------------------------------------------

#[test]
fn the_hermitian_family_agrees_with_the_erased_predicates() {
    // What: all seven members, oracled against the erased facade on the same
    // tensor — a general endomorphism, its Hermitian part and its
    // anti-Hermitian part, so each predicate sees both verdicts.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);
    let tol = 1e-10;

    let erased_cases = [
        erased.clone(),
        erased.project_hermitian().unwrap(),
        erased.project_antihermitian().unwrap(),
    ];
    let typed_cases = [
        typed.clone(),
        typed.project_hermitian().unwrap(),
        typed.project_antihermitian().unwrap(),
    ];

    for (index, (erased, typed)) in erased_cases.iter().zip(&typed_cases).enumerate() {
        // The projections themselves agree bitwise before anything is asked
        // about them.
        assert_eq!(typed.data(), erased.data(), "case {index} payload");
        assert_eq!(
            typed.is_hermitian(tol).unwrap(),
            erased.is_hermitian(tol).unwrap(),
            "case {index} is_hermitian"
        );
        assert_eq!(
            typed.is_antihermitian(tol).unwrap(),
            erased.is_antihermitian(tol).unwrap(),
            "case {index} is_antihermitian"
        );
        assert_eq!(
            typed.is_isometric(tol).unwrap(),
            erased.is_isometric(tol).unwrap(),
            "case {index} is_isometric"
        );
        assert_eq!(
            typed.is_unitary(tol).unwrap(),
            erased.is_unitary(tol).unwrap(),
            "case {index} is_unitary"
        );
        assert_eq!(
            typed.is_posdef(tol).unwrap(),
            erased.is_posdef(tol).unwrap(),
            "case {index} is_posdef"
        );
    }

    // The verdicts are not all the same value, so the agreement above is not
    // vacuous: the projections really are (anti-)Hermitian and the source is
    // neither.
    assert!(!typed_cases[0].is_hermitian(tol).unwrap());
    assert!(typed_cases[1].is_hermitian(tol).unwrap());
    assert!(!typed_cases[1].is_antihermitian(tol).unwrap());
    assert!(typed_cases[2].is_antihermitian(tol).unwrap());
}

#[test]
fn isometry_and_posdef_see_their_positive_cases() {
    // What: the two members the fixture above only ever answers `false` for.
    // `u` from an SVD is isometric by construction, and `t† t` is positive
    // definite when `t` has full rank.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);
    let tol = 1e-9;

    let eu = erased.svd_compact().unwrap().0;
    let tu = typed.svd_compact().unwrap().0;
    assert!(tu.is_isometric(tol).unwrap());
    assert_eq!(tu.is_isometric(tol).unwrap(), eu.is_isometric(tol).unwrap());
    // Isometric but not unitary: `u` is tall here.
    assert!(!tu.is_unitary(tol).unwrap());
    assert_eq!(tu.is_unitary(tol).unwrap(), eu.is_unitary(tol).unwrap());

    // `2 * id` is Hermitian with every eigenvalue at 2: positive definite on
    // any provider, and the cheapest tensor that is.
    let erased_positive = tenet::prelude::Tensor::id(
        &runtime,
        tenet::prelude::Dtype::F64,
        &erased.domain_spaces(),
    )
    .unwrap()
    .scale(2.0)
    .unwrap();
    let typed_positive = TensorMap::id(&runtime, &typed.domain()).unwrap().scale(2.0);
    assert!(typed_positive.is_hermitian(tol).unwrap());
    assert!(typed_positive.is_posdef(tol).unwrap());
    assert_eq!(
        typed_positive.is_posdef(tol).unwrap(),
        erased_positive.is_posdef(tol).unwrap()
    );
    // Hermitian but not positive definite: the same tensor negated.
    let negated = typed_positive.scale(-1.0);
    assert!(negated.is_hermitian(tol).unwrap());
    assert!(!negated.is_posdef(tol).unwrap());
    assert_eq!(
        negated.is_posdef(tol).unwrap(),
        erased_positive.scale(-1.0).unwrap().is_posdef(tol).unwrap()
    );
    // Positive *semi*definite is `false`, not `true`: TensorKit's `isposdef` is
    // Cholesky-based and strict, and this facade's rustdoc promises the same.
    // A real diagonal endomorphism with one entry at exactly zero is the case
    // that separates `>` from `>=` — `eigh` on it returns that zero exactly, so
    // the comparison is not floating-point weather.
    let semidefinite_leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 3),
        ],
        false,
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

    // A Gram matrix agrees with the erased verdict whichever way it falls.
    let egram = erased.adjoint().unwrap().compose(&erased).unwrap();
    let tgram = typed.adjoint().unwrap().compose(&typed).unwrap();
    assert!(tgram.is_hermitian(tol).unwrap());
    assert_eq!(tgram.is_posdef(tol).unwrap(), egram.is_posdef(tol).unwrap());
}

#[test]
fn a_non_endomorphism_is_never_hermitian_and_never_errors() {
    // What: TensorKit throws here; both facades answer `false`. The projections
    // do error, because there is no tensor to return.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

    assert!(!typed.is_hermitian(1e-9).unwrap());
    assert!(!typed.is_antihermitian(1e-9).unwrap());
    assert!(!typed.is_posdef(1e-9).unwrap());
    assert_eq!(
        typed.is_hermitian(1e-9).unwrap(),
        erased.is_hermitian(1e-9).unwrap()
    );
    assert!(typed.project_hermitian().is_err());
    assert!(typed.project_antihermitian().is_err());
}

#[test]
fn the_hermitian_family_carries_the_su2_dimension_weight() {
    // What: every member reduces through `norm`, which is dimension weighted on
    // a non-abelian provider. Z2 alone cannot separate the two.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = su2_oracle_pair(&runtime);
    let tol = 1e-9;

    assert_eq!(
        typed.project_hermitian().unwrap().data(),
        erased.project_hermitian().unwrap().data()
    );
    assert!(typed
        .project_hermitian()
        .unwrap()
        .is_hermitian(tol)
        .unwrap());
    assert_eq!(
        typed.is_hermitian(tol).unwrap(),
        erased.is_hermitian(tol).unwrap()
    );
    let (_, v) = typed.project_hermitian().unwrap().eigh_full().unwrap();
    assert!(v.is_unitary(tol).unwrap());
}

// ---------------------------------------------------------------------------
// Phase 6 (issue #576), slice 1: `inv`.
// ---------------------------------------------------------------------------

/// An endomorphism whose every coupled-sector block is nonsingular: the fill
/// used by [`z2_endo_oracle_pair`] is position-weighted and produces rank-one
/// blocks, so `inv` on it would be testing the singular path instead. Adding a
/// multiple of the identity is the cheapest fix that stays byte-identical
/// across the two facades.
fn z2_invertible_pair(
    runtime: &Runtime,
) -> (
    tenet::prelude::Tensor,
    TensorMap<tenet::core::Z2FusionRule, f64>,
) {
    let (erased, typed) = z2_endo_oracle_pair(runtime);
    let erased_id =
        tenet::prelude::Tensor::id(runtime, tenet::prelude::Dtype::F64, &erased.domain_spaces())
            .unwrap();
    let typed_id = TensorMap::id(runtime, &typed.domain()).unwrap();
    (
        erased.add(&erased_id, 1.0, 100.0).unwrap(),
        typed.add(&typed_id, 1.0, 100.0).unwrap(),
    )
}

#[test]
fn typed_and_erased_inv_agree_byte_for_byte() {
    // What: the typed `inv` is the erased one — the same per-sector dense solve
    // through the same seam — so the payloads compare bitwise, not just within
    // a tolerance.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_invertible_pair(&runtime);

    let erased_inverse = erased.inv().unwrap();
    let typed_inverse = typed.inv().unwrap();
    assert_eq!(typed_inverse.data(), erased_inverse.data());

    // And it is an inverse: `t * t^-1` is the identity on the codomain.
    let identity = typed.compose(&typed_inverse).unwrap();
    let expected = TensorMap::<_, f64>::id(&runtime, &typed.domain()).unwrap();
    let error = identity
        .data()
        .iter()
        .zip(expected.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-9, "t * inv(t) is not the identity: {error}");
}

#[test]
fn inv_of_a_compact_spectrum_is_the_elementwise_reciprocal() {
    // What: the O(rank) arm. A spectrum's inverse is `1/s_i` on the stored
    // values, and it agrees with the erased facade's own diagonal arm.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);

    let erased_s = erased
        .svd_trunc(&tenet::prelude::Truncation::Full)
        .unwrap()
        .s;
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s;
    // The fixture is rank deficient, so the full spectrum contains zeros that
    // `inv` must refuse; keep only the nonzero part.
    let erased_s = erased_s
        .svd_trunc(&tenet::prelude::Truncation::Rank(2))
        .unwrap()
        .s;
    let typed_s = typed_s.svd_trunc(&Truncation::Rank(2)).unwrap().s;

    assert_eq!(
        typed_s.inv().unwrap().data(),
        erased_s.inv().unwrap().data()
    );
    // `s * s^-1` is the identity on the bond.
    let product = typed_s.compose(&typed_s.inv().unwrap()).unwrap();
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
    let (_, typed) = z2_endo_oracle_pair(&runtime);

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
    let wide = GradedSpace::try_new(
        provider.clone(),
        [
            (tenet::core::Z2Irrep::EVEN, 2),
            (tenet::core::Z2Irrep::ODD, 2),
        ],
        false,
    )
    .unwrap();
    // `narrow ⊗ narrow` has coupled dimensions (even 2, odd 2) as well, so the
    // two sides are isomorphic while the hom spaces differ in rank.
    let narrow = GradedSpace::try_new(
        provider,
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
        false,
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
fn typed_and_erased_pinv_agree_byte_for_byte() {
    // What: the same SVD-and-fold seam on both facades, so the payloads compare
    // bitwise. The fixture is deliberately rank deficient — a full-rank one
    // would let a broken cutoff pass.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);

    for rcond in [0.0, 1e-12, 1e-3] {
        assert_eq!(
            typed.pinv(rcond).unwrap().data(),
            erased.pinv(rcond).unwrap().data(),
            "pinv payloads diverge at rcond {rcond}"
        );
    }

    // Moore-Penrose: `t t^+ t = t`, the identity that a wrong fold would break.
    // `rcond` is well above the fixture's numerically-zero singular values and
    // well below its real ones, so the cutoff drops exactly the null directions
    // — inverting those instead would amplify rounding into the millions of ulp.
    let pseudo = typed.pinv(1e-6).unwrap();
    let round_trip = typed.compose(&pseudo).unwrap().compose(&typed).unwrap();
    let error = round_trip
        .data()
        .iter()
        .zip(typed.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        error < 1e-9 * typed.norm().unwrap(),
        "t t^+ t != t: {error}"
    );
}

#[test]
fn pinv_of_a_compact_spectrum_stays_compact_and_agrees_with_the_erased_arm() {
    // What: the O(rank) arm — an elementwise cutoff and reciprocal, whose own
    // singular values are `|entry|`, so no SVD runs at all.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);
    let erased_s = erased
        .svd_trunc(&tenet::prelude::Truncation::Full)
        .unwrap()
        .s;
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s;

    for rcond in [0.0, 1e-12, 1e-3] {
        assert_eq!(
            typed_s.pinv(rcond).unwrap().data(),
            erased_s.pinv(rcond).unwrap().data(),
            "compact pinv payloads diverge at rcond {rcond}"
        );
    }
}

#[test]
fn pinv_cuts_a_singular_value_sitting_exactly_on_the_cutoff() {
    // What: the boundary. The comparison is `sigma > rcond * sigma_max`, so a
    // singular value at *exactly* the cutoff is discarded, not kept — on both
    // storages and on both facades. This is the one bit of the cutoff policy a
    // mutation to `>=` would otherwise slip past.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [(tenet::core::Z2Irrep::EVEN, 2)],
        false,
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

    // And on the compact arm, whose comparison is the erased facade's own.
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
    let (_, typed) = z2_endo_oracle_pair(&runtime);
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
    // TensorKit's per-block `rtol`. A per-sector cutoff would keep the small
    // sector's value here; the global one discards it.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [
            (tenet::core::Z2Irrep::EVEN, 1),
            (tenet::core::Z2Irrep::ODD, 1),
        ],
        false,
    )
    .unwrap();
    // Even sector: 1024. Odd sector: 1. Each sector is 1x1, so per-sector
    // sigma_max would be the entry itself and nothing could ever be cut.
    let tensor = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
        if *trees.coupled() == tenet::core::Z2Irrep::EVEN {
            1024.0
        } else {
            1.0
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
fn typed_and_erased_exp_agree_byte_for_byte_on_a_dense_hermitian_input() {
    // What: the dense arm is the erased one — the same eigh-and-fold seam — so
    // the payloads compare bitwise.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_hermitian_pair(&runtime);

    assert_eq!(typed.exp().unwrap().data(), erased.exp().unwrap().data());

    // And the SU(2) branch, where the recoupling weights are not all one.
    let (erased, typed) = su2_hermitian_pair(&runtime);
    assert_eq!(typed.exp().unwrap().data(), erased.exp().unwrap().data());
}

#[test]
fn exp_of_the_identity_is_e_times_the_identity() {
    // What: the value oracle that does not go through the erased facade.
    // `exp(id) = e * id` on every provider, which pins the `V exp(D) V^H`
    // assembly rather than just its agreement with a sibling.
    let _guard = cache_lock();
    let runtime = runtime();
    let (_, typed) = z2_endo_oracle_pair(&runtime);
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
fn exp_rejects_a_non_hermitian_endomorphism() {
    // What: this facade's `exp` is Hermitian-only — a recorded divergence from
    // TensorKit, whose `exp` is a general per-block Pade approximant. The
    // refusal is the visible half of that divergence, so it is pinned.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);

    assert!(!typed.is_hermitian(1e-9).unwrap());
    assert!(typed.exp().is_err(), "a non-Hermitian exp was accepted");
    assert!(erased.exp().is_err(), "the erased facade disagrees");
}

#[test]
fn exp_of_a_compact_spectrum_stays_compact_and_is_elementwise() {
    // What: the arm this phase adds. The erased facade densifies a diagonal
    // `exp` and runs a full eigendecomposition on the block-diagonal buffer;
    // here it is `exp(s_i)` on the `Σ_c k_c` stored values, which is
    // TensorKit's own `exp(::DiagonalTensorMap)`. The two must still agree
    // numerically — the point of the arm is the complexity, not the answer.
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_endo_oracle_pair(&runtime);
    // Scaled down: the fixture's largest singular value is in the thousands and
    // `exp` of it overflows to infinity, which no comparison can separate from
    // a wrong infinity.
    let erased_s = erased
        .svd_trunc(&tenet::prelude::Truncation::Full)
        .unwrap()
        .s
        .scale(1e-3)
        .unwrap();
    let typed_s = typed.svd_trunc(&Truncation::Full).unwrap().s.scale(1e-3);

    let typed_exp = typed_s.exp().unwrap();
    let erased_exp = erased_s.exp().unwrap();
    let error = typed_exp
        .data()
        .iter()
        .zip(erased_exp.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        error < 1e-12,
        "the compact exp arm disagrees with the erased densified one: {error}"
    );
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
    let leg = GradedSpace::try_new(
        Arc::new(tenet::core::Z2FusionRule),
        [(tenet::core::Z2Irrep::EVEN, 2)],
        false,
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
    // Dense storage of the very same matrix: refused, because it is not
    // Hermitian.
    assert!(!dense.is_hermitian(1e-9).unwrap());
    assert!(dense.exp().is_err());

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
