use super::*;
use crate::tests::GenericMultiplicityRule;
use crate::BoundDynamicTensorRef;
use std::cell::Cell;
use tenet_core::{
    block_structure_intern_cache_info, complete_hom_space_structure_cache_info,
    fusion_tree_layout_cache_info, reset_core_intern_tables, BlockSpec, BraidingStyleKind,
    CoupledSectorFold, FermionParityFusionRule, FusionAlgebraError, FusionProductSpace,
    FusionTreePairKey, Fz2SectorLayout, InfallibleGeneric, PackedProductCodec,
    ProductFusionRule, ProductSectorCodec, ProductSectorLayout, SU2FusionRule, SU2Irrep,
    SectorId, SectorLeg, SectorVec, Su2SectorLayout, TensorMapSpace, U1FusionRule, U1Irrep,
    U1SectorLayout, Z2FusionRule, Z2Irrep,
};

type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

fn lowered_product_rule() -> TripleRule {
    TripleRule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    )
}

fn matrix_space() -> DynamicFusionMapSpace {
    let rule = Z2FusionRule;
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace, [vec![1, 1]]).unwrap()
}

fn typed_z2_matrix_space() -> FusionTensorMapSpace<1, 1> {
    let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
    FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::from_dims([2], [2]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &Z2FusionRule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckedGenericSpyError(usize);

impl fmt::Display for CheckedGenericSpyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "checked Generic query {} failed", self.0)
    }
}

impl std::error::Error for CheckedGenericSpyError {}

struct CheckedGenericSpy {
    identity: RuleIdentity,
    style: Cell<FusionStyleKind>,
    calls: Cell<usize>,
    fail_at: Option<usize>,
}

impl CheckedGenericSpy {
    fn new() -> Self {
        Self {
            identity: RuleIdentity::of_type::<Self>(),
            style: Cell::new(FusionStyleKind::Generic),
            calls: Cell::new(0),
            fail_at: None,
        }
    }

    fn hit(&self) -> Result<(), CheckedGenericSpyError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at == Some(call) {
            Err(CheckedGenericSpyError(call))
        } else {
            Ok(())
        }
    }
}

impl FusionRule for CheckedGenericSpy {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.style.get()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.calls.set(self.calls.get() + 1);
        sector
    }

    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        self.calls.set(self.calls.get() + 1);
        [SectorId::new(0)].into_iter().collect()
    }

    fn nsymbol(&self, _: SectorId, _: SectorId, _: SectorId) -> usize {
        self.calls.set(self.calls.get() + 1);
        2
    }
}

impl CheckedGenericFusion for CheckedGenericSpy {
    type Error = CheckedGenericSpyError;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.style.get()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.hit()?;
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        let _ = (left, right);
        self.hit()?;
        Ok([SectorId::new(0)].into_iter().collect())
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        let _ = (left, right);
        self.hit()?;
        Ok([SectorId::new(0)].into_iter().collect())
    }

    fn try_coupled_sector_fold(
        &self,
        effective: &[SectorId],
    ) -> Result<CoupledSectorFold, Self::Error> {
        let _ = effective;
        self.hit()?;
        Ok(CoupledSectorFold::complete(vec![SectorId::new(0)]))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        let _ = (left, right, coupled);
        self.hit()?;
        Ok(2)
    }
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // Bound ownership requires Arc; this local equality checker borrows a non-Sync rule.
fn checked_generic_equal_identity_checker_commits_under_source_arc() {
    let rule = GenericMultiplicityRule;
    let charge = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(charge, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(charge, 1)], false)]),
    );
    let source_provider = Arc::new(InfallibleGeneric::new(&rule));
    let checker_provider = Arc::new(InfallibleGeneric::new(&rule));
    assert!(!Arc::ptr_eq(&source_provider, &checker_provider));
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&source_provider),
        homspace.clone(),
    )
    .unwrap();

    let prepared = source
        .prepare_final_homspace_generic_with_checked(checker_provider.as_ref(), homspace)
        .unwrap();
    let output = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap();

    assert!(Arc::ptr_eq(output.provider_arc(), &source_provider));
    assert!(!Arc::ptr_eq(output.provider_arc(), &checker_provider));
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; these local guard spies use Cell counters.
fn checked_generic_bound_guards_preserve_typed_errors_before_commit() {
    let homspace = || FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1)], [(0, 1)]);

    let wrong_root_style = Arc::new(CheckedGenericSpy::new());
    wrong_root_style.style.set(FusionStyleKind::Unique);
    let error = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&wrong_root_style),
        homspace(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(wrong_root_style.calls.get(), 0);

    let mut failing_root = CheckedGenericSpy::new();
    failing_root.fail_at = Some(1);
    let failing_root = Arc::new(failing_root);
    let error = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&failing_root),
        homspace(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(CheckedGenericSpyError(1))
    ));

    let source_provider = Arc::new(CheckedGenericSpy::new());
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&source_provider),
        homspace(),
    )
    .unwrap();

    let mut wrong_identity = CheckedGenericSpy::new();
    wrong_identity.identity = RuleIdentity::of_type::<Z2FusionRule>();
    let error = source
        .prepare_final_homspace_generic_with_checked(&wrong_identity, homspace())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::FusionRuleMismatch { .. })
    ));
    assert_eq!(wrong_identity.calls.get(), 0);

    let wrong_checker_style = CheckedGenericSpy::new();
    wrong_checker_style.style.set(FusionStyleKind::Unique);
    let error = source
        .prepare_final_homspace_generic_with_checked(&wrong_checker_style, homspace())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(wrong_checker_style.calls.get(), 0);

    let mut failing_checker = CheckedGenericSpy::new();
    failing_checker.fail_at = Some(1);
    let error = source
        .prepare_final_homspace_generic_with_checked(&failing_checker, homspace())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(CheckedGenericSpyError(1))
    ));

    let checker = CheckedGenericSpy::new();
    let prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    let legacy = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(CheckedGenericSpy::new()),
        homspace(),
    )
    .unwrap();
    let error = legacy
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap_err();
    assert!(matches!(error, OperationError::StructureMismatch { .. }));

    let mut prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    prepared.identity = RuleIdentity::of_type::<Z2FusionRule>();
    let error = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));

    let prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    // Contract-violating adversarial mutation proves the defensive final style guard.
    source_provider.style.set(FusionStyleKind::Unique);
    let error = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    source_provider.style.set(FusionStyleKind::Generic);

    let prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    let output = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap();
    assert!(Arc::ptr_eq(output.provider_arc(), &source_provider));
    assert!(std::ptr::eq(output.provider(), source_provider.as_ref()));
    assert_eq!(output.space().homspace(), &homspace());
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; the single-threaded spy uses Cell counters.
fn checked_generic_preparation_rejects_legacy_binding_before_checker_queries() {
    let source_provider = Arc::new(CheckedGenericSpy::new());
    let checker = CheckedGenericSpy::new();
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        source_provider,
        homspace.clone(),
    )
    .unwrap();

    let error = match source.prepare_final_homspace_generic_with_checked(&checker, homspace) {
        Ok(_) => panic!("legacy binding was accepted as checked authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::MalformedFusionTree { .. })
    ));
    assert_eq!(checker.calls.get(), 0);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; the spy and rule are single-threaded test values.
fn checked_generic_prepared_structure_constructor_matches_root_constructor() {
    // What: committing a structure the caller already enumerated yields
    // the same checked Generic root as enumerating inside the constructor
    // (blocks, required length, HomSpace, provider Arc, checked binding),
    // and the commit itself issues no provider query.
    let rule = GenericMultiplicityRule;
    let charge = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(charge, 2)], false),
            SectorLeg::new([(charge, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(charge, 1)], false),
            SectorLeg::new([(charge, 3)], false),
        ]),
    );
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let root = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )
    .unwrap();
    let prepared = homspace
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(
            provider.as_ref(),
        )
        .unwrap();
    let committed = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
        prepared,
    )
    .unwrap();
    let rows = |space: &DynamicFusionMapSpace| {
        let structure = space.structure();
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(root.space().structure().block_count(), 5);
    assert_eq!(rows(root.space()), rows(committed.space()));
    assert_eq!(
        root.space().required_len().unwrap(),
        committed.space().required_len().unwrap()
    );
    assert_eq!(root.space().homspace(), committed.space().homspace());
    assert_eq!(root.space().nout(), committed.space().nout());
    assert_eq!(root.space().nin(), committed.space().nin());
    assert!(Arc::ptr_eq(committed.provider_arc(), &provider));
    // Only a `CheckedGeneric` binding stages a checked final HomSpace.
    committed
        .prepare_final_homspace_generic_with_checked(provider.as_ref(), homspace)
        .unwrap();

    let spy = Arc::new(CheckedGenericSpy::new());
    let vacuum_hom = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
    let prepared = vacuum_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(spy.as_ref())
        .unwrap();
    let enumerated = spy.calls.get();
    assert!(enumerated > 0);
    BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(&spy),
        vacuum_hom,
        prepared,
    )
    .unwrap();
    assert_eq!(spy.calls.get(), enumerated);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; the isolated spy uses Cell counters.
fn checked_generic_bound_space_commits_the_staged_layout_without_reenumeration() {
    const ISOLATED: &str = "TENET_CHECKED_GENERIC_BOUND_SPACE_ISOLATED";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "contract::dynamic_space::bound_invariant_tests::checked_generic_bound_space_commits_the_staged_layout_without_reenumeration",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
            "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        return;
    }

    reset_core_intern_tables();
    let provider = Arc::new(CheckedGenericSpy::new());
    let source_hom = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::clone(&provider),
        source_hom,
    )
    .unwrap();
    provider.calls.set(0);
    let final_hom = || FusionTreeHomSpace::from_sector_ids([(0, 2), (0, 3)], [(0, 5)]);
    let snapshots = || {
        (
            fusion_tree_layout_cache_info(),
            complete_hom_space_structure_cache_info(),
            block_structure_intern_cache_info(),
        )
    };

    for (identity, style) in [
        (
            RuleIdentity::of_type::<Z2FusionRule>(),
            FusionStyleKind::Generic,
        ),
        (
            FusionRule::rule_identity(provider.as_ref()),
            FusionStyleKind::Unique,
        ),
    ] {
        let mut spy = CheckedGenericSpy::new();
        spy.identity = identity;
        spy.style.set(style);
        let before = snapshots();
        let error = source
            .prepare_final_homspace_generic_checked(&spy, final_hom())
            .err()
            .unwrap();
        assert!(matches!(
            error,
            CheckedGenericStructureError::Core(
                CoreError::FusionRuleMismatch { .. } | CoreError::UnsupportedFusionStyle { .. }
            )
        ));
        assert_eq!(spy.calls.get(), 0);
        assert_eq!(provider.calls.get(), 0);
        assert_eq!(snapshots(), before);
    }

    let before_prepare = snapshots();
    let spy = CheckedGenericSpy::new();
    let prepared = source
        .prepare_final_homspace_generic_checked(&spy, final_hom())
        .unwrap();
    let prepare_calls = spy.calls.get();
    assert!(prepare_calls > 0);
    assert_eq!(provider.calls.get(), 0);
    let preview = prepared.structure().clone();
    let required_len = prepared.required_len();
    assert_eq!(snapshots(), before_prepare);

    let before_commit = snapshots();
    let committed = source
        .commit_final_homspace_generic_checked(prepared)
        .unwrap();
    assert_eq!(spy.calls.get(), prepare_calls);
    assert_eq!(provider.calls.get(), 0);
    assert!(Arc::ptr_eq(committed.provider_arc(), &provider));
    assert_eq!(committed.space().structure().as_ref(), &preview);
    assert_eq!(committed.space().required_len().unwrap(), required_len);
    assert_eq!(committed.space().homspace(), &final_hom());
    assert!(matches!(
        committed.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert_eq!(
        block_structure_intern_cache_info().entries(),
        before_commit.2.entries() + 1
    );

    let complete = CheckedGenericSpy::new();
    let complete_calls = {
        let _staged = source
            .prepare_final_homspace_generic_checked(&complete, final_hom())
            .unwrap();
        complete.calls.get()
    };
    let mut failing = CheckedGenericSpy::new();
    failing.fail_at = Some(complete_calls);
    let before_failure = snapshots();
    let error = source
        .prepare_final_homspace_generic_checked(&failing, final_hom())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(CheckedGenericSpyError(call))
            if call == complete_calls
    ));
    assert_eq!(failing.calls.get(), complete_calls);
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(snapshots(), before_failure);

    let spy = CheckedGenericSpy::new();
    let prepared = source
        .prepare_final_homspace_generic_checked(&spy, final_hom())
        .unwrap();
    provider.style.set(FusionStyleKind::Unique);
    let before_style_rejected_commit = snapshots();
    let error = source
        .commit_final_homspace_generic_checked(prepared)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(snapshots(), before_style_rejected_commit);
    provider.style.set(FusionStyleKind::Generic);

    let spy = CheckedGenericSpy::new();
    let mut corrupted = source
        .prepare_final_homspace_generic_checked(&spy, final_hom())
        .unwrap();
    corrupted.identity = RuleIdentity::of_type::<Z2FusionRule>();
    let before_rejected_commit = snapshots();
    let error = source
        .commit_final_homspace_generic_checked(corrupted)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
    assert_eq!(snapshots(), before_rejected_commit);
}

#[test]
fn expert_sparse_layout_is_a_subset_with_structural_zeros() {
    // What: filtering an expert layout drops completeness, while an omitted
    // valid tree remains an accepted structural zero.
    let complete = typed_z2_matrix_space();
    let keys = complete.homspace().fusion_tree_keys(&Z2FusionRule);
    assert!(matches!(
        complete.admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    let first = complete.subblock_structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let raw = FusionTensorMapSpace::new_unbound(
        complete.dense_space().clone(),
        complete.homspace().clone(),
        structure,
    )
    .unwrap();
    assert_eq!(raw.admission(), &FusionSpaceAdmission::Unbound);
    assert!(matches!(
        raw.validate_rule(&Z2FusionRule),
        Err(CoreError::MissingFusionRuleIdentity)
    ));

    let subset = raw.try_bind_rule(&Z2FusionRule).unwrap();
    assert!(matches!(
        subset.admission(),
        FusionSpaceAdmission::Subset(_)
    ));
    assert!(subset.find_subblock_index(&keys[0]).is_some());
    assert!(subset.find_subblock_index(&keys[1]).is_none());
    assert_eq!(
        subset.adjoint_view().unwrap().admission(),
        subset.admission()
    );
    assert_eq!(
        DynamicFusionMapSpace::from_typed(&subset).admission(),
        subset.admission()
    );
}

#[test]
fn sparse_subset_operands_keep_only_present_canonical_keys() {
    let complete = typed_z2_matrix_space();
    let first = complete.subblock_structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let subset = FusionTensorMapSpace::new_unbound(
        complete.dense_space().clone(),
        complete.homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(&Z2FusionRule)
    .unwrap();
    let storage = DynamicFusionMapSpace::from_typed(&subset);
    let direct = FusionOperand::direct(&storage)
        .prepare(&Z2FusionRule, encoded_layout_primer::<Z2FusionRule>)
        .unwrap();
    let adjoint = FusionOperand::adjoint(&storage)
        .prepare(&Z2FusionRule, encoded_layout_primer::<Z2FusionRule>)
        .unwrap();

    // What: Direct retains the parent-owned order, while Adjoint retains
    // only parent-present keys in canonical logical order.
    assert_eq!(direct.logical_block_count(), 1);
    assert_eq!(adjoint.logical_block_count(), 1);
    assert_eq!(direct.storage_index(0).unwrap(), 0);
    assert_eq!(adjoint.storage_index(0).unwrap(), 0);
    assert!(direct.is_direct());
    assert_eq!(adjoint.adjoint_storage_indices().unwrap(), [0].as_slice());
}

#[test]
fn complete_and_subset_spaces_are_mathematically_equal() {
    // What: proof strength survives typed erasure and adjoint views without
    // changing mathematical equality.
    let complete = typed_z2_matrix_space();
    let subset = FusionTensorMapSpace::from_shared_subblock_structure(
        complete.dense_space().clone(),
        complete.homspace().clone(),
        Arc::clone(complete.subblock_structure()),
    )
    .unwrap()
    .try_bind_rule(&Z2FusionRule)
    .unwrap();
    assert_eq!(complete, subset);

    let complete = DynamicFusionMapSpace::from_typed(&complete);
    let subset = DynamicFusionMapSpace::from_typed(&subset);
    assert_eq!(complete, subset);
    assert!(matches!(
        complete.admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(matches!(
        subset.admission(),
        FusionSpaceAdmission::Subset(_)
    ));
    assert!(matches!(
        complete.adjoint_view().unwrap().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(matches!(
        subset.adjoint_view().unwrap().admission(),
        FusionSpaceAdmission::Subset(_)
    ));

    let bound =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(subset, Arc::new(Z2FusionRule))
            .unwrap();
    assert!(matches!(
        bound.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
}

#[test]
fn bind_revalidates_complete_without_replacing_layout() {
    // What: a legacy Complete stamp gains checked algebra proof, commits
    // one lowered capability, and retains the caller's exact block layout.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    reset_scratch_publication_observations();
    let raw = DynamicFusionMapSpace::from_typed(&typed_z2_matrix_space());
    let structure = Arc::clone(raw.structure());

    let bound =
        BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(Z2FusionRule))
            .unwrap();

    assert!(matches!(
        bound.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(Arc::ptr_eq(&structure, &bound.space.subblock_structure));
    assert_eq!(scratch_publication_observations(), (0, 0, 0));

    let rule = lowered_product_rule();
    let pair = Fz2U1Codec::encode(Z2Irrep::ODD.sector_id(), U1Irrep::new(2).sector_id());
    let sector = TripleCodec::encode(pair, SU2Irrep::from_twice_spin(1).sector_id());
    let typed = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sectors([(sector, 1)], [(sector, 1)]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let raw = DynamicFusionMapSpace::from_typed(&typed);
    let structure = Arc::clone(raw.structure());
    reset_scratch_publication_observations();

    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(rule))
        .unwrap();

    assert!(matches!(
        bound.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(Arc::ptr_eq(&structure, &bound.space.subblock_structure));
    assert_eq!(scratch_publication_observations(), (0, 0, 0));
}

#[test]
fn lowered_bind_failure_publishes_no_layout_or_admission() {
    // What: finite U1 nonclosure revalidates matching legacy Subset and
    // Complete stamps, returns the exact algebra failure, and commits no state.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let max = U1Irrep::new(i32::MAX).sector_id();
    let one = U1Irrep::new(1).sector_id();
    let vacuum = U1Irrep::new(0).sector_id();
    let pair = FusionTreePairKey::try_pair_from_sector_ids(
        [max.id(), one.id()],
        [],
        vacuum.id(),
        [false; 2],
        [],
        [],
        [],
        [1],
        [],
    )
    .unwrap();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(max, 1)], false),
            SectorLeg::new([(one, 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let structure =
        crate::tests::packed_fixture_structure(2, [(BlockKey::FusionTree(pair), vec![1, 1])])
            .unwrap()
            .into_shared();
    let base = DynamicFusionMapSpace {
        nout: 2,
        nin: 0,
        homspace: Arc::new(homspace),
        subblock_structure: structure,
        admission: FusionSpaceAdmission::Unbound,
        adjoint: OnceLock::new(),
    };

    reset_core_intern_tables();
    reset_scratch_publication_observations();
    let mut mismatched = base.clone();
    mismatched.admission = FusionSpaceAdmission::Subset(Z2FusionRule.rule_identity());
    assert!(matches!(
        BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(
            mismatched,
            Arc::new(U1FusionRule),
        ),
        Err(OperationError::Core(CoreError::FusionRuleMismatch { .. }))
    ));
    assert_eq!(scratch_publication_observations(), (0, 0, 0));

    for admission in [
        FusionSpaceAdmission::Subset(U1FusionRule.rule_identity()),
        FusionSpaceAdmission::Complete(U1FusionRule.rule_identity()),
    ] {
        reset_core_intern_tables();
        reset_scratch_publication_observations();
        let mut raw = base.clone();
        raw.admission = admission.clone();
        let original = raw.clone();
        let error = BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(
            raw,
            Arc::new(U1FusionRule),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::FusionAlgebra(cause)
                if *cause == FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1,
                }
        ));
        assert_eq!(original.admission(), &admission);
        assert_eq!(scratch_publication_observations(), (0, 0, 0));
    }
}

#[test]
fn lowered_bind_preserves_omitted_product_codec_failure() {
    // What: an invalid packed-product sector omitted by a valid Subset is
    // reported exactly when complete-grid preparation reaches it, without publication.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = lowered_product_rule();
    let valid = rule.vacuum();
    let invalid = SectorId::new(usize::MAX);
    let expected = TripleCodec::decode_checked(invalid).unwrap_err();
    let pair = FusionTreePairKey::try_pair_from_sector_ids(
        [valid.id()],
        [valid.id()],
        valid.id(),
        [false],
        [false],
        [],
        [],
        [],
        [],
    )
    .unwrap();
    let leg = || SectorLeg::new([(valid, 1), (invalid, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let structure =
        crate::tests::packed_fixture_structure(2, [(BlockKey::FusionTree(pair), vec![1, 1])])
            .unwrap()
            .into_shared();
    let raw = DynamicFusionMapSpace {
        nout: 1,
        nin: 1,
        homspace: Arc::new(homspace),
        subblock_structure: structure,
        admission: FusionSpaceAdmission::Subset(rule.rule_identity()),
        adjoint: OnceLock::new(),
    };
    reset_core_intern_tables();
    reset_scratch_publication_observations();

    let error = BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(rule))
        .unwrap_err();
    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::ProductCodec(expected)))
    );
    assert_eq!(scratch_publication_observations(), (0, 0, 0));
}

#[test]
fn lowered_subset_to_complete_validates_grid_before_commit() {
    // What: a checked sparse subset remains a subset when complete-grid
    // validation fails, and the prepared lowered layout is not published.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let complete = typed_z2_matrix_space();
    let first = complete.subblock_structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let subset = FusionTensorMapSpace::new_unbound(
        complete.dense_space().clone(),
        complete.homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule_checked(&Z2FusionRule)
    .unwrap();
    let raw = DynamicFusionMapSpace::from_typed(&subset);
    let original = raw.clone();
    reset_scratch_publication_observations();

    assert!(matches!(
        BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(Z2FusionRule),),
        Err(OperationError::Core(CoreError::BlockCountMismatch { .. }))
    ));
    assert!(matches!(
        original.admission(),
        FusionSpaceAdmission::Subset(_)
    ));
    assert_eq!(scratch_publication_observations(), (0, 0, 0));
}

#[test]
fn derived_and_validated_paths_reject_subset_proofs() {
    // What: internal derived construction and validated-layout rebinding
    // require an executable Complete proof in release builds.
    let complete = typed_z2_matrix_space();
    let subset = FusionTensorMapSpace::from_shared_subblock_structure(
        complete.dense_space().clone(),
        complete.homspace().clone(),
        Arc::clone(complete.subblock_structure()),
    )
    .unwrap()
    .try_bind_rule(&Z2FusionRule)
    .unwrap();
    let complete = DynamicFusionMapSpace::from_typed(&complete);
    let subset = DynamicFusionMapSpace::from_typed(&subset);
    let provider = Arc::new(Z2FusionRule);
    let expected = OperationError::StructureMismatch {
        tensor: "complete fusion layout",
    };

    assert_eq!(
        BoundDynamicFusionMapSpace::from_derived(Arc::clone(&provider), subset.clone())
            .unwrap_err(),
        expected
    );
    let authority = BoundDynamicFusionMapSpace::from_derived(provider, complete).unwrap();
    assert_eq!(
        authority
            .rebind_validated(&ValidatedDynamicFusionLayout(subset))
            .unwrap_err(),
        expected
    );
}

#[test]
fn bound_space_rejects_incoherent_axis_split() {
    // What: bound admission and the public borrowed tensor proof both reject
    // a dynamic rank that disagrees with its HomSpace.
    let raw = DynamicFusionMapSpace {
        nout: 0,
        nin: 1,
        adjoint: OnceLock::new(),
        ..matrix_space()
    };

    let corrupted = BoundDynamicFusionMapSpace {
        space: raw.clone(),
        provider: Arc::new(Z2FusionRule),
        layout_build: LayoutBuildCapability::encoded(),
    };
    assert!(matches!(
        BoundDynamicTensorRef::<_, f64>::try_new(&corrupted, &[]),
        Err(OperationError::Core(
            CoreError::StructureRankMismatch { .. }
        ))
    ));

    let error = BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::new(Z2FusionRule))
        .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionSpaceSplitMismatch { .. })
    ));
}

#[test]
fn bound_space_rejects_incoherent_structure_rank() {
    // What: bound admission and the public borrowed tensor proof both reject
    // storage whose rank differs from its HomSpace.
    let raw = matrix_space();
    let block = raw.structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::with_key(block.key().clone(), vec![1], vec![1], 0).unwrap()],
    )
    .unwrap();
    let raw = DynamicFusionMapSpace {
        subblock_structure: Arc::new(structure),
        adjoint: OnceLock::new(),
        ..raw
    };

    let corrupted = BoundDynamicFusionMapSpace {
        space: raw.clone(),
        provider: Arc::new(Z2FusionRule),
        layout_build: LayoutBuildCapability::encoded(),
    };
    assert!(matches!(
        BoundDynamicTensorRef::<_, f64>::try_new(&corrupted, &[]),
        Err(OperationError::Core(
            CoreError::StructureRankMismatch { .. }
        ))
    ));

    let error = BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::new(Z2FusionRule))
        .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::StructureRankMismatch { .. })
    ));
}

#[test]
fn direct_bound_builders_keep_coherent_split_and_rank() {
    // What: direct multiplicity-free and Generic builders satisfy bound
    // invariants, and final-HomSpace Generic layout exactly matches the
    // caller-shape expert layout for a genuine outer multiplicity.
    let provider = Arc::new(Z2FusionRule);
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let multiplicity_free = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        Arc::clone(&provider),
        homspace.clone(),
        [vec![1, 1]],
    )
    .unwrap();
    let generic_provider = Arc::new(GenericMultiplicityRule);
    let charge = SectorId::new(1);
    let generic_homspace = FusionTreeHomSpace::from_sector_ids(
        [(charge.id(), 2), (charge.id(), 3)],
        [(charge.id(), 4)],
    );
    let generic_key_count = generic_homspace
        .fusion_tree_keys_generic(generic_provider.as_ref())
        .unwrap()
        .len();
    assert_eq!(generic_key_count, 2);
    let expert = BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(
        Arc::clone(&generic_provider),
        generic_homspace.clone(),
        vec![vec![2, 3, 4]; generic_key_count],
    )
    .unwrap();
    let generic = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        generic_provider,
        generic_homspace,
    )
    .unwrap();

    assert_eq!(multiplicity_free.space().nout(), 1);
    assert_eq!(generic.space().nin(), 1);
    assert_eq!(multiplicity_free.space().structure().rank(), 2);
    assert_eq!(generic.space().structure().rank(), 3);
    assert_eq!(generic.space().structure(), expert.space().structure());
    assert_eq!(
        generic.space().required_len().unwrap(),
        expert.space().required_len().unwrap()
    );
    for index in 0..expert.space().structure().block_count() {
        let actual = generic.space().structure().block(index).unwrap();
        let expected = expert.space().structure().block(index).unwrap();
        assert_eq!(actual.key(), expected.key());
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.strides(), expected.strides());
        assert_eq!(actual.offset(), expected.offset());
    }
}

#[test]
fn generic_completeness_requires_every_vertex_label() {
    // What: N(1,1,1)=2 requires both vertex labels 1 and 2 in a Complete grid.
    let provider = Arc::new(GenericMultiplicityRule);
    let homspace = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1)]);
    let keys = homspace
        .fusion_tree_keys_generic(provider.as_ref())
        .unwrap();
    assert_eq!(
        keys.iter()
            .map(|key| key.codomain_vertices()[0].get())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let complete = DynamicFusionMapSpace::from_degeneracy_shapes_generic(
        provider.as_ref(),
        homspace,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    assert_eq!(complete.structure().block_count(), 2);
    assert!(matches!(
        complete.admission(),
        FusionSpaceAdmission::Complete(_)
    ));

    let first = complete.structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        complete.rank(),
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let subset = DynamicFusionMapSpace {
        subblock_structure: Arc::new(structure),
        admission: FusionSpaceAdmission::Subset(provider.rule_identity()),
        adjoint: OnceLock::new(),
        ..complete
    };

    let error = BoundDynamicFusionMapSpace::bind_generic(subset, provider).unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::BlockCountMismatch {
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn generic_adjoint_rejects_multiplicity_free_before_final_layout_build() {
    // What: a Generic adjoint rejects a multiplicity-free provider before
    // constructing its target layout.
    let source = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        matrix_space(),
        Arc::new(Z2FusionRule),
    )
    .unwrap();
    reset_final_result_layout_builds();

    let error = crate::adjoint::adjoint_bound_space_dyn_generic(&source).unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(final_result_layout_builds(), 0);
}

#[test]
fn memoized_adjoint_view_reads_the_current_admission() {
    // What: admission is reassigned after construction when a layout is
    // promoted, so the adjoint view reads it at call time; only the
    // admission-independent hom space and structure are memoized.
    let complete = DynamicFusionMapSpace::from_typed(&typed_z2_matrix_space());
    let first = complete.adjoint_view().unwrap();
    let mut unbound = complete.clone();
    unbound.admission = FusionSpaceAdmission::Unbound;
    let second = unbound.adjoint_view().unwrap();
    assert_eq!(second.admission(), &FusionSpaceAdmission::Unbound);
    assert_eq!(first.admission(), complete.admission());
    assert!(Arc::ptr_eq(second.structure(), first.structure()));
    assert!(Arc::ptr_eq(second.homspace_arc(), first.homspace_arc()));
}
