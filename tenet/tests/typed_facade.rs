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
    /// Refuses to decode charge 2, which the engine reaches by fusing two
    /// charge-1 legs — a violation of the codec's decode-totality law.
    NarrowDecode,
    /// Duals every sector to the vacuum, i.e. a non-injective dual: a broken
    /// rigidity structure rather than an unrepresentable value.
    CollapsingDual,
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
            Ok(SectorId::new(usize::from(value.0)))
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
    let space = tenet::prelude::Space::z2([(0, 2), (1, 3)]);
    let erased = tenet::prelude::Tensor::from_block_fn(
        runtime,
        [&space, &space],
        [&space],
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
    let typed = TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], typed_fill_value).unwrap();
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

    let (eu, _, evh) = erased.svd_compact().unwrap();
    let (tu, ts, tvh) = typed.svd_compact().unwrap();

    assert_eq!(tu.data(), eu.data());
    assert_eq!(tvh.data(), evh.data());
    // The erased `s` is diagonal storage and the typed one is dense (#570), so
    // the two `s` factors are not byte-comparable. What `s` holds is pinned by
    // the `svd_vals` oracle above and the reconstruction test below.
    //
    // Dense `s`: one block per coupled sector, k_c² elements each — the ceiling
    // #570 records. If typed diagonal storage lands this is what changes.
    assert!(ts.data().len() > typed.svd_vals().unwrap()[0].values.len());
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
}

#[test]
fn a_spectrum_decode_failure_comes_back_as_the_codec_error() {
    // What: the one input a caller of these methods can actually malform is
    // the provider itself — every `Truncation` state that fails validation is
    // unconstructible outside `tenet-matrixalgebra` (the fallible constructors
    // reject them and the variants are `#[non_exhaustive]`), so there is no
    // malformed policy to feed. A codec that cannot decode a coupled sector the
    // engine produced fails the call with the provider's own error instead of
    // panicking inside the label map.
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
    // Why no test that label order actually *differs* from the engine's
    // `SectorId` order: no provider this suite hosts separates the two —
    // `Z3Charge`, `Z2Irrep` and `SU2Irrep` all order exactly as their ids do.
    // Standing up a provider whose `Ord` is deliberately reversed would be a
    // second full rule implementation to observe a sort that this test already
    // pins by its own predicate.
}

#[test]
fn typed_and_erased_qr_and_lq_agree_byte_for_byte() {
    let _guard = cache_lock();
    let runtime = runtime();
    let (erased, typed) = z2_oracle_pair(&runtime);

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
