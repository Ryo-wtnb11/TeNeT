use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BraidingStyleKind, CheckedGenericRigidSymbols, CoreError, FusionStyleKind, FusionTreeHomSpace,
    StructurallyValidatedFusionTreeSubset,
};
use tenet_operations::{DenseTreeTransformOperations, TensorContractSpec, TreeTransformBackend};

use crate::tree_transform::{
    build_checked_generic_tree_pair_transform_group_plan, CheckedGenericPlanError,
};
use crate::{ConjugateValue, DenseRecouplingScalar, OperationError, RecouplingCoefficientAction};

use super::dynamic_space::{BoundDynamicFusionMapSpace, PreparedCheckedGenericDynamicSpace};
use super::fusion::{
    compile_tensorcontract_fusion_plan_from_ranks, orient_fusion_contract_plan,
    ContractAxisOrderCandidate, FusionContractOrientation,
};
use super::fusion_block::{
    compile_checked_generic_core_plan, BackendRank2Gemm, FusionBlockContractWorkspace, Rank2Gemm,
};
use super::structure::TensorContractAxisPlan;

type CheckedContractResult<P, D> = Result<
    (BoundDynamicFusionMapSpace<P>, Vec<D>),
    CheckedGenericPlanError<<P as tenet_core::CheckedGenericFusion>::Error>,
>;
type CheckedStagedTransform<E> = Result<
    (
        PreparedCheckedGenericDynamicSpace,
        tenet_operations::TreeTransformStructure<f64>,
        Arc<tenet_core::BlockStructure>,
    ),
    CheckedGenericPlanError<E>,
>;

fn same_axes(lhs: &[usize], rhs: &[usize]) -> bool {
    let mut lhs = lhs.to_vec();
    let mut rhs = rhs.to_vec();
    lhs.sort_unstable();
    rhs.sort_unstable();
    lhs == rhs
}

fn validate_source_structure<E>(
    space: &super::DynamicFusionMapSpace,
) -> Result<(), CheckedGenericPlanError<E>> {
    StructurallyValidatedFusionTreeSubset::try_new(space.homspace(), space.structure())?;
    Ok(())
}

fn staged_transform<P>(
    authority: &BoundDynamicFusionMapSpace<P>,
    provider: &P,
    source: &BoundDynamicFusionMapSpace<P>,
    operation: &crate::TreeTransformOperation,
) -> CheckedStagedTransform<P::Error>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    let homspace = source.space().homspace().try_permute_generic_checked(
        provider,
        operation.codomain_permutation(),
        operation.domain_permutation(),
    )?;
    let prepared = authority.prepare_final_homspace_generic_with_checked(provider, homspace)?;
    let destination = Arc::new(prepared.structure().clone());
    let plan = build_checked_generic_tree_pair_transform_group_plan(
        provider,
        operation.clone(),
        source.space().structure(),
    )?;
    let replay = plan.compile_structures(&destination, source.space().structure())?;
    Ok((prepared, replay, destination))
}

fn execute_transform<D>(
    backend: &mut DenseTreeTransformOperations,
    workspace: &mut <DenseTreeTransformOperations as TreeTransformBackend<D, f64>>::Workspace,
    replay: &tenet_operations::TreeTransformStructure<f64>,
    destination: &Arc<tenet_core::BlockStructure>,
    source: &Arc<tenet_core::BlockStructure>,
    destination_data: &mut [D],
    source_data: &[D],
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    backend.tree_transform_structure_into_raw(
        workspace,
        replay,
        destination,
        source,
        destination_data,
        source_data,
        D::one(),
        D::zero(),
    )
}

/// Runs exactly one caller-selected checked Generic contraction candidate.
///
/// Every destination remains a read-only staged structure until replay and
/// backend execution succeed. The sole publication is the final commit under
/// the left operand's provider allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_owned_checked_generic_preselected<P, D>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
    candidate: &ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + ConjugateValue + Copy + Zero,
{
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = Default::default();
    tensorcontract_owned_checked_generic_preselected_with_core_gemm(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        axes,
        dst_nout,
        candidate,
        orientation,
        &mut BackendRank2Gemm {
            backend: &mut backend,
            workspace: &mut workspace,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn tensorcontract_owned_checked_generic_preselected_with_core_gemm<P, D, G>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
    candidate: &ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
    core_gemm: &mut G,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + ConjugateValue + Copy + Zero,
    G: Rank2Gemm<D>,
{
    if axes.lhs_conjugate() || axes.rhs_conjugate() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "checked Generic contraction currently requires eager direct operands",
        }
        .into());
    }
    if !same_axes(axes.lhs_contracting_axes(), candidate.lhs())
        || !same_axes(axes.rhs_contracting_axes(), candidate.rhs())
    {
        return Err(OperationError::InvalidArgument {
            message: "preselected candidate must preserve contracted axis sets",
        }
        .into());
    }
    let output_rank = lhs_space
        .space()
        .rank()
        .checked_sub(candidate.lhs().len())
        .and_then(|rank| {
            rhs_space
                .space()
                .rank()
                .checked_sub(candidate.rhs().len())
                .and_then(|rhs| rank.checked_add(rhs))
        })
        .ok_or(OperationError::ElementCountOverflow)?;
    let candidate_axes =
        TensorContractSpec::new(candidate.lhs(), candidate.rhs(), axes.output_permutation());
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.space().rank(),
        rhs_space.space().rank(),
        output_rank,
        candidate_axes,
    )?;
    if dst_nout > output_rank {
        return Err(CoreError::StructureRankMismatch {
            expected: output_rank,
            actual: dst_nout,
        }
        .into());
    }
    if lhs_data.len() != lhs_space.space().required_len()? {
        return Err(OperationError::ElementCountMismatch {
            expected: lhs_space.space().required_len()?,
            actual: lhs_data.len(),
        }
        .into());
    }
    if rhs_data.len() != rhs_space.space().required_len()? {
        return Err(OperationError::ElementCountMismatch {
            expected: rhs_space.space().required_len()?,
            actual: rhs_data.len(),
        }
        .into());
    }
    validate_source_structure::<P::Error>(lhs_space.space())?;
    validate_source_structure::<P::Error>(rhs_space.space())?;

    let plan = orient_fusion_contract_plan(
        compile_tensorcontract_fusion_plan_from_ranks(
            dst_nout,
            output_rank,
            lhs_space.space().rank(),
            rhs_space.space().rank(),
            candidate_axes,
            false,
            false,
        )?,
        orientation,
    );

    let provider = lhs_space.provider();
    let lhs_identity = provider.rule_identity();
    let rhs_identity = rhs_space.provider().rule_identity();
    if lhs_identity != rhs_identity {
        return Err(CoreError::FusionRuleMismatch {
            expected: lhs_identity,
            actual: rhs_identity,
        }
        .into());
    }
    for actual in [provider.fusion_style(), rhs_space.provider().fusion_style()] {
        if actual != FusionStyleKind::Generic {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual,
            }
            .into());
        }
    }
    for actual in [
        provider.braiding_style(),
        rhs_space.provider().braiding_style(),
    ] {
        if actual != BraidingStyleKind::Bosonic {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "checked Generic contraction requires Bosonic braiding",
            }
            .into());
        }
    }

    let destination_homspace = FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
        provider,
        lhs_space.space().homspace(),
        rhs_space.space().homspace(),
        candidate.lhs(),
        candidate.rhs(),
        &axis_plan.output_axes,
        dst_nout,
    )?;
    let destination =
        lhs_space.prepare_final_homspace_generic_with_checked(provider, destination_homspace)?;

    let (lhs_prepared, lhs_replay, lhs_structure) =
        staged_transform(lhs_space, provider, lhs_space, plan.lhs_transform())?;
    let (rhs_prepared, rhs_replay, rhs_structure) =
        staged_transform(lhs_space, provider, rhs_space, plan.rhs_transform())?;

    let (core_left, core_right, core_left_structure, core_right_structure) = match orientation {
        FusionContractOrientation::LhsRhs => {
            (&lhs_prepared, &rhs_prepared, &lhs_structure, &rhs_structure)
        }
        FusionContractOrientation::RhsLhs => {
            (&rhs_prepared, &lhs_prepared, &rhs_structure, &lhs_structure)
        }
    };
    let core_axes = plan.core_axes().as_spec();
    let core_axis_plan = TensorContractAxisPlan::compile(
        core_left.homspace().rank(),
        core_right.homspace().rank(),
        output_rank,
        core_axes,
    )?;
    let core_homspace = FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
        provider,
        core_left.homspace(),
        core_right.homspace(),
        core_axes.lhs_contracting_axes(),
        core_axes.rhs_contracting_axes(),
        &core_axis_plan.output_axes,
        plan.core_dst_open_lhs_rank(),
    )?;
    let core_destination =
        lhs_space.prepare_final_homspace_generic_with_checked(provider, core_homspace)?;
    let core_structure = Arc::new(core_destination.structure().clone());
    let core_plan = compile_checked_generic_core_plan(
        &core_structure,
        core_destination.nout(),
        core_left_structure,
        core_left.nout(),
        core_right_structure,
        core_right.nout(),
        core_axes,
    )?;

    let destination_structure = Arc::new(destination.structure().clone());
    let output_plan = build_checked_generic_tree_pair_transform_group_plan(
        provider,
        plan.output_transform().clone(),
        &core_structure,
    )?;
    let output_replay = output_plan.compile_structures(&destination_structure, &core_structure)?;

    let mut transform_backend = DenseTreeTransformOperations::default();
    let mut transform_workspace = Default::default();
    let mut lhs_transformed = vec![D::zero(); lhs_prepared.required_len()];
    execute_transform(
        &mut transform_backend,
        &mut transform_workspace,
        &lhs_replay,
        &lhs_structure,
        lhs_space.space().structure(),
        &mut lhs_transformed,
        lhs_data,
    )?;
    let mut rhs_transformed = vec![D::zero(); rhs_prepared.required_len()];
    execute_transform(
        &mut transform_backend,
        &mut transform_workspace,
        &rhs_replay,
        &rhs_structure,
        rhs_space.space().structure(),
        &mut rhs_transformed,
        rhs_data,
    )?;

    let (core_lhs_data, core_rhs_data) = match orientation {
        FusionContractOrientation::LhsRhs => (&lhs_transformed, &rhs_transformed),
        FusionContractOrientation::RhsLhs => (&rhs_transformed, &lhs_transformed),
    };
    let mut core_data = vec![D::zero(); core_destination.required_len()];
    let mut fusion_workspace = FusionBlockContractWorkspace::default();
    let mut kernels = crate::StridedHostKernelAdapter::default();
    core_plan.execute_raw(
        &mut kernels,
        core_gemm,
        &mut fusion_workspace,
        &core_structure,
        &mut core_data,
        core_left_structure,
        core_lhs_data,
        core_right_structure,
        core_rhs_data,
        D::one(),
        D::zero(),
    )?;

    let mut data = vec![D::zero(); destination.required_len()];
    execute_transform(
        &mut transform_backend,
        &mut transform_workspace,
        &output_replay,
        &destination_structure,
        &core_structure,
        &mut data,
        &core_data,
    )?;
    let destination = lhs_space.commit_final_homspace_generic_bound_checked(destination)?;
    Ok((destination, data))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use tenet_core::{
        BraidingStyleKind, CheckedGenericFusion, CoupledSectorFold, FusionProductSpace, FusionRule,
        FusionStyleKind, GenericFArray, GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols,
        RuleIdentity, SectorId, SectorLeg, SectorVec, Su3FusionRule,
    };

    use super::*;
    use crate::contract::fusion::contracted_axis_order_candidates;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Query {
        Dual,
        Channel,
        N,
        F,
        R,
        Rigidity,
    }

    impl Query {
        const COUNT: usize = 6;
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Event {
        Identity,
        Style,
        Braiding,
        Query(Query),
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SpyError(Query);

    impl std::fmt::Display for SpyError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "injected {:?} failure", self.0)
        }
    }

    impl std::error::Error for SpyError {}

    /// Checked-only wrapper: deliberately does not implement `FusionRule`.
    struct CheckedSu3Spy {
        rule: Su3FusionRule,
        calls: Cell<[usize; Query::COUNT]>,
        events: RefCell<Vec<Event>>,
        fail: Cell<Option<Query>>,
        malformed: Cell<Option<Query>>,
    }

    impl CheckedSu3Spy {
        fn new() -> Self {
            Self {
                rule: Su3FusionRule::new(),
                calls: Cell::new([0; Query::COUNT]),
                events: RefCell::new(Vec::new()),
                fail: Cell::new(None),
                malformed: Cell::new(None),
            }
        }

        fn hit(&self, query: Query) -> Result<(), SpyError> {
            self.events.borrow_mut().push(Event::Query(query));
            let mut calls = self.calls.get();
            calls[query as usize] += 1;
            self.calls.set(calls);
            if self.fail.get() == Some(query) {
                Err(SpyError(query))
            } else {
                Ok(())
            }
        }

        fn reset(&self) {
            self.calls.set([0; Query::COUNT]);
            self.events.borrow_mut().clear();
            self.fail.set(None);
            self.malformed.set(None);
        }

        fn count(&self, query: Query) -> usize {
            self.calls.get()[query as usize]
        }

        fn algebra_calls(&self) -> usize {
            self.calls.get().into_iter().sum()
        }
    }

    impl CheckedGenericFusion for CheckedSu3Spy {
        type Error = SpyError;

        fn rule_identity(&self) -> RuleIdentity {
            self.events.borrow_mut().push(Event::Identity);
            FusionRule::rule_identity(&self.rule)
        }

        fn fusion_style(&self) -> FusionStyleKind {
            self.events.borrow_mut().push(Event::Style);
            FusionRule::fusion_style(&self.rule)
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            self.events.borrow_mut().push(Event::Braiding);
            FusionRule::braiding_style(&self.rule)
        }

        fn vacuum(&self) -> SectorId {
            FusionRule::vacuum(&self.rule)
        }

        fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
            self.hit(Query::Dual)?;
            Ok(FusionRule::dual(&self.rule, sector))
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.hit(Query::Channel)?;
            Ok(FusionRule::fusion_channels(&self.rule, left, right))
        }

        fn try_fusion_channels_in_table(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.hit(Query::Channel)?;
            Ok(FusionRule::fusion_channels_in_table(
                &self.rule, left, right,
            ))
        }

        fn try_coupled_sector_fold(
            &self,
            effective: &[SectorId],
        ) -> Result<CoupledSectorFold, Self::Error> {
            self.hit(Query::Channel)?;
            Ok(FusionRule::coupled_sector_fold(&self.rule, effective))
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, Self::Error> {
            self.hit(Query::N)?;
            Ok(FusionRule::nsymbol(&self.rule, left, right, coupled))
        }
    }

    impl CheckedGenericRigidSymbols for CheckedSu3Spy {
        type Scalar = f64;

        fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            Ok(GenericRigidSymbols::sqrt_dim_scalar(&self.rule, sector))
        }

        fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            Ok(GenericRigidSymbols::inv_sqrt_dim_scalar(&self.rule, sector))
        }

        fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            Ok(GenericRigidSymbols::frobenius_schur_phase_scalar(
                &self.rule, sector,
            ))
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
            self.hit(Query::F)?;
            let symbol = GenericFusionSymbols::f_symbol_generic(&self.rule, a, b, c, d, e, f);
            if self.malformed.get() == Some(Query::F) {
                Ok(GenericFArray::new(
                    symbol.data().to_vec(),
                    (1, 1, symbol.data().len(), 1),
                ))
            } else {
                Ok(symbol)
            }
        }

        fn try_r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> Result<GenericRMatrix<f64>, Self::Error> {
            self.hit(Query::R)?;
            let symbol = GenericFusionSymbols::r_symbol_generic(&self.rule, a, b, c);
            if self.malformed.get() == Some(Query::R) {
                Ok(GenericRMatrix::new(
                    symbol.data().to_vec(),
                    1,
                    symbol.data().len(),
                ))
            } else {
                Ok(symbol)
            }
        }
    }

    struct FailingGemm;

    impl Rank2Gemm<f64> for FailingGemm {
        #[allow(clippy::too_many_arguments)]
        fn matmul_rank2(
            &mut self,
            _dst: &mut [f64],
            _lhs: &[f64],
            _rhs: &[f64],
            _rows: usize,
            _contracted: usize,
            _cols: usize,
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            Err(OperationError::StridedKernel {
                message: "injected checked Generic core failure".into(),
            })
        }
    }

    fn homspace(rule: &Su3FusionRule, nout: usize, nin: usize) -> FusionTreeHomSpace {
        let eight = rule.sector_of(1, 1).unwrap();
        let leg = || SectorLeg::new([(eight, 1)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new((0..nout).map(|_| leg())),
            FusionProductSpace::new((0..nin).map(|_| leg())),
        )
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn bound_pair(
        nout: usize,
        nin: usize,
    ) -> (
        Arc<CheckedSu3Spy>,
        BoundDynamicFusionMapSpace<CheckedSu3Spy>,
        Arc<CheckedSu3Spy>,
        BoundDynamicFusionMapSpace<CheckedSu3Spy>,
    ) {
        let left = Arc::new(CheckedSu3Spy::new());
        let right = Arc::new(CheckedSu3Spy::new());
        let homspace = homspace(&left.rule, nout, nin);
        let lhs = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&left),
            homspace.clone(),
        )
        .unwrap();
        let rhs = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&right),
            homspace,
        )
        .unwrap();
        left.reset();
        right.reset();
        (left, lhs, right, rhs)
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_uses_left_authority_and_commits_left_owner() {
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        let candidate = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];

        let (output, data) = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap();

        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert!(!Arc::ptr_eq(output.provider_arc(), &right));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.algebra_calls() > 0);
        assert_eq!(data.len(), output.space().required_len().unwrap());
        assert!(left
            .events
            .borrow()
            .ends_with(&[Event::Identity, Event::Style]));
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_rejects_local_axes_before_provider_queries() {
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        let candidate = contracted_axis_order_candidates(&[9], &[0]).remove(0);
        let error = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &vec![0.0; lhs.space().required_len().unwrap()],
            &rhs,
            &vec![0.0; rhs.space().required_len().unwrap()],
            TensorContractSpec::with_default_output_order(&[9], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap_err();

        assert!(matches!(error, CheckedGenericPlanError::Operation(_)));
        assert_eq!(left.algebra_calls(), 0);
        assert_eq!(right.algebra_calls(), 0);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_executes_one_transforming_candidate() {
        let (left, lhs, right, rhs) = bound_pair(2, 2);
        let candidate = contracted_axis_order_candidates(&[3, 2], &[0, 1]).remove(0);
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];

        let (output, data) = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
            2,
            &candidate,
            FusionContractOrientation::RhsLhs,
        )
        .unwrap();

        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.count(Query::F) > 0);
        assert!(left.count(Query::R) > 0);
        assert_eq!(data.len(), output.space().required_len().unwrap());
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_preserves_late_provider_and_shape_errors() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_PROVIDER_FAILURE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::preselected_checked_generic_preserves_late_provider_and_shape_errors",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated provider-failure test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(2, 2);
        tenet_core::reset_core_intern_tables();
        let candidate = contracted_axis_order_candidates(&[3, 2], &[0, 1]).remove(0);
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];

        for query in [Query::Dual, Query::Channel, Query::N, Query::F, Query::R] {
            left.reset();
            right.reset();
            left.fail.set(Some(query));
            let error = tensorcontract_owned_checked_generic_preselected(
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
                2,
                &candidate,
                FusionContractOrientation::LhsRhs,
            )
            .unwrap_err();
            assert!(matches!(
                error,
                CheckedGenericPlanError::Provider(SpyError(actual)) if actual == query
            ));
            assert_eq!(right.algebra_calls(), 0);
            assert_eq!(tenet_core::block_structure_intern_cache_info().entries(), 0);
        }

        left.reset();
        right.reset();
        left.malformed.set(Some(Query::R));
        let error = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
            2,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::SymbolShape { symbol: "R", .. }
        ));
        assert_eq!(right.algebra_calls(), 0);
        assert_eq!(tenet_core::block_structure_intern_cache_info().entries(), 0);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_backend_failure_does_not_commit() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_FAILURE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::preselected_checked_generic_backend_failure_does_not_commit",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated failure-atomicity test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::reset_core_intern_tables();
        let candidate = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let error = tensorcontract_owned_checked_generic_preselected_with_core_gemm(
            &lhs,
            &vec![1.0; lhs.space().required_len().unwrap()],
            &rhs,
            &vec![2.0; rhs.space().required_len().unwrap()],
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
            &mut FailingGemm,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
        ));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.algebra_calls() > 0);
        assert_eq!(tenet_core::fusion_tree_layout_cache_info().entries(), 0);
        assert_eq!(
            tenet_core::complete_hom_space_structure_cache_info().entries(),
            0
        );
        assert_eq!(tenet_core::block_structure_intern_cache_info().entries(), 0);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_success_commits_once() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_SUCCESS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::preselected_checked_generic_success_commits_once",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated commit test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::reset_core_intern_tables();
        let candidate = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let (output, _) = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &vec![1.0; lhs.space().required_len().unwrap()],
            &rhs,
            &vec![2.0; rhs.space().required_len().unwrap()],
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap();

        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert_eq!(tenet_core::block_structure_intern_cache_info().entries(), 1);
        assert!(left
            .events
            .borrow()
            .ends_with(&[Event::Identity, Event::Style]));
    }
}
