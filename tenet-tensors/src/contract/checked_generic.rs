use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockStructure, BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericRigidSymbols,
    CoreError, FusionTreeHomSpace, RuleIdentity, StructurallyValidatedFusionTreeSubset,
};
use tenet_operations::{ContractDestinationInit, TensorContractSpec, TreeTransformBackend};

use crate::tree_transform::{CheckedGenericPlanError, CheckedPendingCoefficients};
use crate::{
    zeroed_payload, ConjugateValue, DenseRecouplingScalar, OperationError,
    RecouplingCoefficientAction, ZeroBytes,
};

use super::backend::TensorContractBackend;
use super::context::{
    plan_compose_in, plan_contract_in, PlanTarget, TensorContractFusionExecutionContext,
};
use super::dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand,
    PreparedCheckedGenericDynamicSpace,
};
use super::resolution::{HostEagerExecutor, StorageContractResolution};
use super::structure::TensorContractAxisPlan;

type CheckedContractResult<P, D> = Result<
    (BoundDynamicFusionMapSpace<P>, Vec<D>),
    CheckedGenericPlanError<<P as tenet_core::CheckedGenericFusion>::Error>,
>;

/// The U15 boundary of general checked contraction.
pub(crate) const CHECKED_CONTRACTION_REQUIRES_BOSONIC: OperationError =
    OperationError::UnsupportedTensorContractScope {
        message: "checked Generic contraction requires Bosonic braiding",
    };

/// Checked operands are eager and direct; lazy checked adjoints are #1865.
pub(crate) const CHECKED_REQUIRES_DIRECT_OPERANDS: OperationError =
    OperationError::UnsupportedTensorContractScope {
        message: "checked Generic contraction currently requires eager direct operands",
    };

/// What the checked planner derives spaces under (its
/// `PlanningAlgebra::SpaceAuthority`): the left binding, whose provider
/// admission stages every derived space, and the entry's answer to whether
/// a contraction twist exists.
///
/// Why the answer travels here rather than being read where the planner
/// asks: a provider read inside the planner would follow the destination's
/// staging (#2046).
pub(crate) struct CheckedAuthority<'a, P> {
    pub(crate) binding: &'a BoundDynamicFusionMapSpace<P>,
    twist_free: bool,
}

// Why manual: a derive would bound `P: Copy`.
impl<P> Clone for CheckedAuthority<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P> Copy for CheckedAuthority<'_, P> {}

impl<'a, P> CheckedAuthority<'a, P>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    /// Canonical composition (TensorKit `mul!`) crosses no legs, so it never
    /// asks the twist question and the authority holds no answer: asked, it
    /// fails closed.
    pub(crate) fn composition(binding: &'a BoundDynamicFusionMapSpace<P>) -> Self {
        Self {
            binding,
            twist_free: false,
        }
    }

    /// General contraction may braid and twist; the checked engine
    /// implements neither TensorKit `blas_contract!`'s fermionic supertrace
    /// branch nor non-symmetric braiding, so it admits Bosonic braiding only
    /// (U15), read once from the left provider.
    pub(crate) fn contraction(
        binding: &'a BoundDynamicFusionMapSpace<P>,
    ) -> Result<Self, CheckedGenericPlanError<P::Error>> {
        if binding.provider().braiding_style() != BraidingStyleKind::Bosonic {
            return Err(CHECKED_CONTRACTION_REQUIRES_BOSONIC.into());
        }
        Ok(Self {
            binding,
            twist_free: true,
        })
    }

    /// Whether the entry established that no contraction twist exists.
    pub(crate) fn twist_free(self) -> bool {
        self.twist_free
    }
}

impl<'a, P> PlanTarget<'a, P, CheckedAuthority<'a, P>> {
    /// The checked destination `space` (a staged preview) planned under
    /// `authority`, whose binding's provider is the rule.
    pub(crate) fn checked(
        authority: CheckedAuthority<'a, P>,
        space: &'a DynamicFusionMapSpace,
    ) -> Self {
        Self {
            rule: authority.binding.provider(),
            space,
            authority,
        }
    }
}

/// One checked contraction's transaction (#2063): its staged intermediate
/// spaces and its staged coefficients and transformers. Nothing staged is
/// published before [`Self::commit`]; dropping it publishes nothing.
pub(crate) struct CheckedContractTxn {
    coefficients: CheckedPendingCoefficients,
    // Why inline three: a contraction stages at most its two sources and its
    // core destination, or one copyC temporary.
    staged: smallvec::SmallVec<[(PreparedCheckedGenericDynamicSpace, Arc<BlockStructure>); 3]>,
}

impl CheckedContractTxn {
    /// Captures the reset epoch: create it before the call's first build.
    pub(crate) fn new() -> Self {
        Self {
            coefficients: CheckedPendingCoefficients::new(),
            staged: smallvec::SmallVec::new(),
        }
    }

    pub(crate) fn coefficients(&mut self) -> &mut CheckedPendingCoefficients {
        &mut self.coefficients
    }

    /// Stages one derived space and returns the preview the planner reads.
    pub(crate) fn stage(
        &mut self,
        prepared: PreparedCheckedGenericDynamicSpace,
    ) -> DynamicFusionMapSpace {
        let preview = prepared.preview();
        self.staged
            .push((prepared, Arc::clone(preview.structure())));
        preview
    }

    /// Publishes after the fallible destination commit (`destination`:
    /// its preview and committed structures): commits only the staged
    /// intermediates `resolution` replays over, in staging order, then
    /// flushes coefficients and transformers keyed by committed ids. A
    /// stage the planner declined is dropped unpublished.
    pub(crate) fn commit(
        self,
        destination: (Arc<BlockStructure>, Arc<BlockStructure>),
        resolution: &StorageContractResolution<f64>,
    ) {
        let used = resolution.derived_structures();
        let mut committed: smallvec::SmallVec<[_; 4]> = smallvec::smallvec![destination];
        for (prepared, preview) in self.staged {
            if used
                .iter()
                .any(|structure| Arc::ptr_eq(structure, &preview))
            {
                committed.push((preview, prepared.commit_structure()));
            }
        }
        self.coefficients.flush_committed(&committed);
    }

    /// Publishes after an eager transform's destination commit
    /// (`destination`: its preview and committed structures); a transform
    /// stages no intermediate.
    pub(crate) fn commit_transform(self, destination: (Arc<BlockStructure>, Arc<BlockStructure>)) {
        self.coefficients.flush_committed(&[destination]);
    }
}

fn validate_source_structure<E>(
    space: &DynamicFusionMapSpace,
) -> Result<(), CheckedGenericPlanError<E>> {
    StructurallyValidatedFusionTreeSubset::try_new(space.homspace(), space.structure())?;
    Ok(())
}

fn validate_contract_local<P, D>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
) -> Result<TensorContractAxisPlan, CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    if axes.lhs_conjugate() || axes.rhs_conjugate() {
        return Err(CHECKED_REQUIRES_DIRECT_OPERANDS.into());
    }
    let output_rank = lhs_space
        .space()
        .rank()
        .checked_sub(axes.lhs_contracting_axes().len())
        .and_then(|rank| {
            rhs_space
                .space()
                .rank()
                .checked_sub(axes.rhs_contracting_axes().len())
                .and_then(|rhs| rank.checked_add(rhs))
        })
        .ok_or(OperationError::ElementCountOverflow)?;
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.space().rank(),
        rhs_space.space().rank(),
        output_rank,
        axes,
    )?;
    if dst_nout > output_rank {
        return Err(CoreError::StructureRankMismatch {
            expected: output_rank,
            actual: dst_nout,
        }
        .into());
    }
    for (data, space) in [(lhs_data, lhs_space), (rhs_data, rhs_space)] {
        let expected = space.space().required_len()?;
        if data.len() != expected {
            return Err(OperationError::ElementCountMismatch {
                expected,
                actual: data.len(),
            }
            .into());
        }
        validate_source_structure::<P::Error>(space.space())?;
    }
    Ok(axis_plan)
}

/// Contracts two direct checked Generic tensors, the result split after its
/// first `codomain_rank` output axes (TensorOperations `pAB`) and owned by
/// the left provider allocation.
///
/// It plans through the shared [`plan_contract`] ladder in the checked mode
/// (the canonical core, TensorKit's `copyC`, the `DynamicTree` artifact)
/// and replays on the Host route executor. The destination and every
/// intermediate are staged and planned over as previews; only after the
/// replay succeeds is the destination committed, then the intermediates the
/// route used, then the staged transformers (#2063).
///
/// [`plan_contract`]: TensorContractFusionExecutionContext::plan_contract
#[doc(hidden)]
pub fn tensorcontract_owned_checked_generic_in_context<P, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleIdentity>,
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    codomain_rank: usize,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
{
    tensorcontract_checked_generic(
        context,
        (lhs_space, lhs_data),
        (rhs_space, rhs_data),
        axes,
        codomain_rank,
    )
}

/// [`tensorcontract_owned_checked_generic_in_context`] over any context
/// backends (a test injects failing ones).
fn tensorcontract_checked_generic<P, D, BT, BC>(
    context: &mut TensorContractFusionExecutionContext<D, RuleIdentity, BT, BC>,
    (lhs_space, lhs_data): (&BoundDynamicFusionMapSpace<P>, &[D]),
    (rhs_space, rhs_data): (&BoundDynamicFusionMapSpace<P>, &[D]),
    axes: TensorContractSpec<'_>,
    codomain_rank: usize,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
{
    let axis_plan = validate_contract_local(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        axes,
        codomain_rank,
    )?;
    let provider = crate::admission::admit_checked_generic_pair(lhs_space, rhs_space)?;
    let authority = CheckedAuthority::contraction(lhs_space)?;
    let destination = lhs_space.prepare_final_homspace_generic_from_checked(provider, || {
        FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
            provider,
            lhs_space.space().homspace(),
            rhs_space.space().homspace(),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &axis_plan.output_axes,
            codomain_rank,
        )
        .map_err(CheckedGenericPlanError::from)
    })?;
    let preview = destination.preview();
    let mut txn = CheckedContractTxn::new();
    let resolution = plan_contract_in::<HostEagerExecutor, CheckedGenericAdmissionMode, P>(
        &mut txn,
        PlanTarget::checked(authority, &preview),
        FusionOperand::direct(lhs_space.space()),
        FusionOperand::direct(rhs_space.space()),
        (
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &axis_plan.output_axes,
        ),
        None,
    )?;
    #[cfg(test)]
    context.record_contract_route(&resolution);
    let mut data = zeroed_payload(destination.required_len());
    context.execute_contract_route_host(
        &resolution,
        preview.structure(),
        &mut data,
        (lhs_space.space().structure(), lhs_data),
        (rhs_space.space().structure(), rhs_data),
        D::one(),
        ContractDestinationInit::Zeroed,
    )?;
    let destination = lhs_space.commit_final_homspace_generic_bound_checked(destination)?;
    // Only after the fallible destination commit: a failed call commits no
    // intermediate and publishes nothing. These commits cannot fail; a lost
    // or refused admission leaves its preview unpublishable.
    txn.commit(
        (
            Arc::clone(preview.structure()),
            Arc::clone(destination.space().structure()),
        ),
        &resolution,
    );
    Ok((destination, data))
}

/// Canonical composition (TensorKit `mul!`) of two direct checked Generic
/// tensors: `lhs.domain` is glued to `rhs.codomain` in order.
///
/// It plans through the shared [`plan_compose`](super::plan_compose) rung in
/// the checked mode and replays on the Host route executor: one
/// coefficient-free block GEMM per coupled sector, or the checked irregular
/// core for a non-canonical tiling. No leg crosses another, hence no
/// braiding style is required. The destination is staged, planned over as
/// a preview, and committed only after the replay succeeds (#2063).
#[doc(hidden)]
pub fn tensorcompose_owned_checked_generic_in_context<P, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleIdentity>,
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
{
    let lhs_nout = lhs_space.space().nout();
    let lhs_axes = (lhs_nout..lhs_space.space().rank()).collect::<Vec<_>>();
    let rhs_axes = (0..rhs_space.space().nout()).collect::<Vec<_>>();
    let axes = TensorContractSpec::with_default_output_order(&lhs_axes, &rhs_axes);
    let axis_plan =
        validate_contract_local(lhs_space, lhs_data, rhs_space, rhs_data, axes, lhs_nout)?;
    let provider = crate::admission::admit_checked_generic_pair(lhs_space, rhs_space)?;
    let destination = lhs_space.prepare_final_homspace_generic_from_checked(provider, || {
        FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
            provider,
            lhs_space.space().homspace(),
            rhs_space.space().homspace(),
            &lhs_axes,
            &rhs_axes,
            &axis_plan.output_axes,
            lhs_nout,
        )
        .map_err(CheckedGenericPlanError::from)
    })?;
    let preview = destination.preview();
    let resolution = plan_compose_in::<HostEagerExecutor, CheckedGenericAdmissionMode, P>(
        PlanTarget::checked(CheckedAuthority::composition(lhs_space), &preview),
        FusionOperand::direct(lhs_space.space()),
        FusionOperand::direct(rhs_space.space()),
    )?;
    #[cfg(test)]
    context.record_contract_route(&resolution);
    let mut data = zeroed_payload(destination.required_len());
    context.execute_contract_route_host(
        &resolution,
        preview.structure(),
        &mut data,
        (lhs_space.space().structure(), lhs_data),
        (rhs_space.space().structure(), rhs_data),
        D::one(),
        ContractDestinationInit::Zeroed,
    )?;
    let destination = lhs_space.commit_final_homspace_generic_bound_checked(destination)?;
    Ok((destination, data))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use crate::tests::GenericMultiplicityRule;
    use tenet_core::{
        BraidingStyleKind, CheckedGenericFusion, CoupledSectorFold, FusionProductSpace, FusionRule,
        FusionStyleKind, GenericFArray, GenericRMatrix, InfallibleGeneric, RuleIdentity, SectorId,
        SectorLeg, SectorVec,
    };

    use super::*;
    use crate::contract::backend::TensorContractBackend;
    use crate::contract::structure::TensorContractStructure;
    use crate::DenseTreeTransformOperations;
    use tenet_core::{HostReadableStorage, HostWritableStorage, TensorMap};

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
    struct CheckedGenericSpy {
        rule: GenericMultiplicityRule,
        calls: Cell<[usize; Query::COUNT]>,
        events: RefCell<Vec<Event>>,
        fail: Cell<Option<Query>>,
        malformed: Cell<Option<Query>>,
        braiding: Cell<BraidingStyleKind>,
    }

    impl CheckedGenericSpy {
        fn new() -> Self {
            Self {
                rule: GenericMultiplicityRule,
                calls: Cell::new([0; Query::COUNT]),
                events: RefCell::new(Vec::new()),
                fail: Cell::new(None),
                malformed: Cell::new(None),
                braiding: Cell::new(BraidingStyleKind::Bosonic),
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

    impl CheckedGenericFusion for CheckedGenericSpy {
        type Error = SpyError;

        // Why its own identity rather than the wrapped rule's: the composed
        // coefficients are cached per identity, and a sibling test's
        // publication under the shared rule's identity would skip this
        // spy's provider queries.
        fn rule_identity(&self) -> RuleIdentity {
            self.events.borrow_mut().push(Event::Identity);
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            self.events.borrow_mut().push(Event::Style);
            FusionRule::fusion_style(&self.rule)
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            self.events.borrow_mut().push(Event::Braiding);
            self.braiding.get()
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
            Ok(FusionRule::fusion_channels(&self.rule, left, right))
        }

        // Counts one query for the fold itself, as the engine's failure
        // budgets are stated per provider call rather than per inner channel
        // lookup. The classification is the trait default over this rule.
        fn try_coupled_sector_fold(
            &self,
            effective: &[SectorId],
        ) -> Result<CoupledSectorFold, Self::Error> {
            self.hit(Query::Channel)?;
            match InfallibleGeneric::new(&self.rule).try_coupled_sector_fold(effective) {
                Ok(fold) => Ok(fold),
                Err(never) => match never {},
            }
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

    impl CheckedGenericRigidSymbols for CheckedGenericSpy {
        type Scalar = f64;

        fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
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
            let shape = (
                self.rule.nsymbol(a, b, e),
                self.rule.nsymbol(e, c, d),
                self.rule.nsymbol(b, c, f),
                self.rule.nsymbol(a, f, d),
            );
            let mut data = vec![0.0; shape.0 * shape.1 * shape.2 * shape.3];
            let cols = shape.0 * shape.1;
            let rows = shape.2 * shape.3;
            for index in 0..cols.min(rows) {
                data[index * rows + index] = 1.0;
            }
            let symbol = GenericFArray::new(data, shape);
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
            let size = self.rule.nsymbol(a, b, c);
            let mut data = vec![0.0; size * size];
            for index in 0..size {
                data[index * size + index] = 1.0;
            }
            let symbol = GenericRMatrix::new(data, size, size);
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

    /// The default contract backend, failing its `fail_at`-th rank-2 GEMM
    /// job (counted from 0) when set. Batches run serially through the
    /// trait default, so every job is counted.
    #[derive(Default)]
    struct FailingBackend {
        inner: DenseTreeTransformOperations,
        jobs: usize,
        fail_at: Option<usize>,
    }

    impl TensorContractBackend<f64, f64> for FailingBackend {
        type Workspace =
            <DenseTreeTransformOperations as TensorContractBackend<f64, f64>>::Workspace;

        fn tensorcontract_structure_into<
            const DST_NOUT: usize,
            const DST_NIN: usize,
            const LHS_NOUT: usize,
            const LHS_NIN: usize,
            const RHS_NOUT: usize,
            const RHS_NIN: usize,
            SDst,
            SLhs,
            SRhs,
            DDst,
            DLhs,
            DRhs,
        >(
            &mut self,
            _workspace: &mut Self::Workspace,
            _structure: &TensorContractStructure<f64>,
            _dst: &mut TensorMap<f64, DST_NOUT, DST_NIN, SDst, DDst>,
            _lhs: &TensorMap<f64, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
            _rhs: &TensorMap<f64, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError>
        where
            DDst: HostWritableStorage<f64>,
            DLhs: HostReadableStorage<f64>,
            DRhs: HostReadableStorage<f64>,
        {
            unreachable!("the route executor runs rank-2 GEMM jobs only")
        }

        fn tensorcontract_structure_into_raw(
            &mut self,
            _workspace: &mut Self::Workspace,
            _structure: &TensorContractStructure<f64>,
            _dst_structure: &Arc<BlockStructure>,
            _lhs_structure: &Arc<BlockStructure>,
            _rhs_structure: &Arc<BlockStructure>,
            _dst_data: &mut [f64],
            _lhs_data: &[f64],
            _rhs_data: &[f64],
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            unreachable!("the route executor runs rank-2 GEMM jobs only")
        }

        fn matmul_rank2_into_raw(
            &mut self,
            workspace: &mut Self::Workspace,
            dst_data: &mut [f64],
            lhs_data: &[f64],
            rhs_data: &[f64],
            rows: usize,
            contracted: usize,
            cols: usize,
        ) -> Result<(), OperationError> {
            self.matmul_rank2_axpby_into_raw(
                workspace, dst_data, lhs_data, rhs_data, rows, contracted, cols, 1.0, 0.0,
            )
        }

        fn matmul_rank2_axpby_into_raw(
            &mut self,
            workspace: &mut Self::Workspace,
            dst_data: &mut [f64],
            lhs_data: &[f64],
            rhs_data: &[f64],
            rows: usize,
            contracted: usize,
            cols: usize,
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            let job = self.jobs;
            self.jobs += 1;
            if self.fail_at == Some(job) {
                return Err(OperationError::StridedKernel {
                    message: "injected checked Generic core failure".into(),
                });
            }
            TensorContractBackend::<f64, f64>::matmul_rank2_axpby_into_raw(
                &mut self.inner,
                workspace,
                dst_data,
                lhs_data,
                rhs_data,
                rows,
                contracted,
                cols,
                alpha,
                beta,
            )
        }
    }

    type FailingContext = TensorContractFusionExecutionContext<
        f64,
        RuleIdentity,
        DenseTreeTransformOperations,
        FailingBackend,
    >;

    fn failing_context(fail_at: Option<usize>) -> FailingContext {
        let mut context = FailingContext::new(
            DenseTreeTransformOperations::default(),
            FailingBackend::default(),
        );
        context.contract_backend_mut().fail_at = fail_at;
        context
    }

    type SpySpace = BoundDynamicFusionMapSpace<CheckedGenericSpy>;

    /// One public checked contraction on a fresh default context.
    fn contract(
        (lhs, lhs_data): (&SpySpace, &[f64]),
        (rhs, rhs_data): (&SpySpace, &[f64]),
        axes: TensorContractSpec<'_>,
        codomain_rank: usize,
    ) -> CheckedContractResult<CheckedGenericSpy, f64> {
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        tensorcontract_owned_checked_generic_in_context(
            &mut context,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            axes,
            codomain_rank,
        )
    }

    /// The route the last call of `context` planned.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Route {
        Core,
        CopyC,
        DynamicTree,
    }

    fn last_route<BC>(
        context: &TensorContractFusionExecutionContext<
            f64,
            RuleIdentity,
            DenseTreeTransformOperations,
            BC,
        >,
    ) -> Route
    where
        BC: TensorContractBackend<f64, f64>,
    {
        if context.last_resolution_is_core() {
            Route::Core
        } else if context.last_resolution_orientation().is_some() {
            Route::DynamicTree
        } else {
            Route::CopyC
        }
    }

    fn ramp(len: usize, scale: f64, offset: f64) -> Vec<f64> {
        (0..len)
            .map(|index| index as f64 * scale + offset)
            .collect()
    }

    fn homspace(_rule: &GenericMultiplicityRule, nout: usize, nin: usize) -> FusionTreeHomSpace {
        let leg = || SectorLeg::new([(SectorId::new(1), 1)], false);
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
        Arc<CheckedGenericSpy>,
        BoundDynamicFusionMapSpace<CheckedGenericSpy>,
        Arc<CheckedGenericSpy>,
        BoundDynamicFusionMapSpace<CheckedGenericSpy>,
    ) {
        let left = Arc::new(CheckedGenericSpy::new());
        let right = Arc::new(CheckedGenericSpy::new());
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

    /// What: checked compose plans the shared Core route over a preview of
    /// its staged destination. Its provider events are the pair admission,
    /// the destination stage (identity read, admission style, and cold its
    /// structure queries) and the commit guard's style (#2046): planning
    /// and replay query nothing, and no second (core) destination is
    /// derived. The left binding owns the result; a failing destination
    /// query surfaces exactly and publishes nothing.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_compose_plans_the_core_route_without_provider_events_after_staging() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_COMPOSE_EVENTS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::checked_compose_plans_the_core_route_without_provider_events_after_staging",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated compose test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::clear_structure_caches();
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        // A failing destination query surfaces exactly and publishes nothing.
        left.fail.set(Some(Query::Channel));
        let error = tensorcompose_owned_checked_generic_in_context(
            &mut context,
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::Provider(SpyError(Query::Channel))
        ));
        assert_eq!(
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            0
        );
        use Event::{Identity, Style};
        let channel = Event::Query(Query::Channel);
        let expected = [
            (
                "cold",
                vec![Identity, Style, Identity, Style, channel, channel, Style],
            ),
            ("warm", vec![Identity, Style, Identity, Style, Style]),
        ];
        for (call, events) in expected {
            left.reset();
            right.reset();
            let (output, data) = tensorcompose_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
            )
            .unwrap();
            assert!(context.last_resolution_is_core(), "{call}");
            assert!(Arc::ptr_eq(output.provider_arc(), &left), "{call}");
            assert_eq!(right.algebra_calls(), 0, "{call}");
            assert_eq!(data.len(), output.space().required_len().unwrap());
            assert_eq!(*left.events.borrow(), events, "{call}");
            assert_eq!(*right.events.borrow(), [Identity, Style], "{call}");
        }
    }

    const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_ISOLATED";

    /// Reruns test `name` alone in a child process, whose cache assertions
    /// then see no concurrent clear; true inside that child.
    fn isolated(name: &str) -> bool {
        if std::env::var_os(ISOLATED).is_some() {
            return true;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("contract::checked_generic::tests::{name}"),
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated {name} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        false
    }

    use Event::{Braiding, Identity, Style};

    /// What: a canonical contraction plans the shared Core rung (zero-copy
    /// candidate, identity output). Its provider events are the pair
    /// admission, the U15 braiding read, the destination stage and the
    /// commit guard's style (#2046): planning and replay query nothing, and
    /// no second (core) destination is derived. The left binding owns the
    /// result; a failing destination query surfaces exactly and publishes
    /// nothing.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_plans_the_core_route_without_provider_events_after_staging() {
        if !isolated(
            "checked_contraction_plans_the_core_route_without_provider_events_after_staging",
        ) {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::clear_structure_caches();
        let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.5, -1.0);
        let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.25, 2.0);
        let axes = || TensorContractSpec::with_default_output_order(&[1], &[0]);
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        left.fail.set(Some(Query::Channel));
        let error = tensorcontract_owned_checked_generic_in_context(
            &mut context,
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            axes(),
            1,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::Provider(SpyError(Query::Channel))
        ));
        assert_eq!(cache_entries(), (0, 0));
        let channel = Event::Query(Query::Channel);
        let expected = [
            (
                "cold",
                vec![
                    Identity, Style, Braiding, Identity, Style, channel, channel, Style,
                ],
            ),
            (
                "warm",
                vec![Identity, Style, Braiding, Identity, Style, Style],
            ),
        ];
        let mut first = None;
        for (call, events) in expected {
            left.reset();
            right.reset();
            let (output, data) = tensorcontract_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                axes(),
                1,
            )
            .unwrap();
            assert_eq!(last_route(&context), Route::Core, "{call}");
            assert!(Arc::ptr_eq(output.provider_arc(), &left), "{call}");
            assert!(!Arc::ptr_eq(output.provider_arc(), &right), "{call}");
            assert_eq!(right.algebra_calls(), 0, "{call}");
            assert_eq!(data.len(), output.space().required_len().unwrap());
            assert_eq!(*left.events.borrow(), events, "{call}");
            assert_eq!(*right.events.borrow(), [Identity, Style], "{call}");
            // The destination and nothing else is committed.
            assert_eq!(cache_entries(), (1, 0), "{call}");
            let bits = data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(*first.get_or_insert_with(|| bits.clone()), bits, "{call}");
        }
    }

    /// What: local request errors (axes, conjugation, rank, payload length)
    /// are reported before any provider query.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_rejects_local_errors_before_provider_queries() {
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        let lhs_data = vec![0.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![0.0; rhs.space().required_len().unwrap()];
        let short = vec![0.0; rhs_data.len() - 1];
        let cases: [(&str, TensorContractSpec<'_>, usize, &[f64]); 4] = [
            (
                "axis",
                TensorContractSpec::with_default_output_order(&[9], &[0]),
                1,
                &rhs_data,
            ),
            (
                "conjugate",
                TensorContractSpec::new_with_conjugation(
                    &[1],
                    &[0],
                    tenet_operations::OutputAxisOrder::identity(),
                    true,
                    false,
                ),
                1,
                &rhs_data,
            ),
            (
                "rank",
                TensorContractSpec::with_default_output_order(&[1], &[0]),
                3,
                &rhs_data,
            ),
            (
                "length",
                TensorContractSpec::with_default_output_order(&[1], &[0]),
                1,
                &short,
            ),
        ];
        for (name, axes, nout, rhs_data) in cases {
            left.reset();
            right.reset();
            let error = contract((&lhs, &lhs_data), (&rhs, rhs_data), axes, nout).unwrap_err();
            assert!(
                !matches!(error, CheckedGenericPlanError::Provider(_)),
                "{name}: {error:?}"
            );
            assert!(left.events.borrow().is_empty(), "{name}");
            assert!(right.events.borrow().is_empty(), "{name}");
        }
    }

    /// What: a contraction whose operands are not in core form takes the
    /// `DynamicTree` artifact; only the lhs is recoupled (F and R queries),
    /// the rhs borrowed in place.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_executes_one_transforming_candidate() {
        if !isolated("checked_contraction_executes_one_transforming_candidate") {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(2, 2);
        tenet_core::clear_structure_caches();
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        crate::tree_transform::take_completed_transformer_activity();
        let (output, data) = tensorcontract_owned_checked_generic_in_context(
            &mut context,
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
            2,
        )
        .unwrap();
        assert_eq!(last_route(&context), Route::DynamicTree);
        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.count(Query::F) > 0);
        assert!(left.count(Query::R) > 0);
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (1, 1));
        assert_eq!(data.len(), output.space().required_len().unwrap());
    }

    /// The Core, CopyC and `DynamicTree` (one and three transforms) checked
    /// fixtures: `(name, nout = nin, lhs axes, rhs axes, output, codomain
    /// rank, route)`.
    type Fixture = (
        &'static str,
        usize,
        &'static [usize],
        &'static [usize],
        Option<&'static [usize]>,
        usize,
        Route,
    );

    const FIXTURES: [Fixture; 4] = [
        ("core", 1, &[1], &[0], None, 1, Route::Core),
        (
            "copy_c",
            2,
            &[2, 3],
            &[0, 1],
            Some(&[1, 0, 2, 3]),
            2,
            Route::CopyC,
        ),
        ("lhs_only", 2, &[3, 2], &[0, 1], None, 2, Route::DynamicTree),
        (
            "three_stages",
            2,
            &[3, 1],
            &[0, 3],
            Some(&[2, 0, 3, 1]),
            2,
            Route::DynamicTree,
        ),
    ];

    fn fixture_axes<'a>(
        lhs: &'a [usize],
        rhs: &'a [usize],
        output: Option<&'a [usize]>,
    ) -> TensorContractSpec<'a> {
        match output {
            None => TensorContractSpec::with_default_output_order(lhs, rhs),
            Some(output) => {
                TensorContractSpec::new(lhs, rhs, tenet_operations::OutputAxisOrder::Axes(output))
            }
        }
    }

    /// What: on every route a failing provider query (destination, source
    /// spaces and structures, core destination, CopyC temporary, output
    /// structure) or a malformed symbol surfaces exactly, and nothing is
    /// published: no layout (cache 2), no transformer (cache 3), no
    /// composed coefficient, no output.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_preserves_late_provider_and_shape_errors_on_every_route() {
        if !isolated("checked_contraction_preserves_late_provider_and_shape_errors_on_every_route")
        {
            return;
        }
        for (name, n, lhs_axes, rhs_axes, output, nout, route) in FIXTURES {
            let (left, lhs, right, rhs) = bound_pair(n, n);
            let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
            let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
            let axes = || fixture_axes(lhs_axes, rhs_axes, output);
            let mut failed = 0;
            for query in [Query::Dual, Query::Channel, Query::N, Query::F, Query::R] {
                tenet_core::clear_structure_caches();
                left.reset();
                right.reset();
                left.fail.set(Some(query));
                crate::tree_transform::take_completed_transformer_activity();
                match contract((&lhs, &lhs_data), (&rhs, &rhs_data), axes(), nout) {
                    Err(error) => {
                        failed += 1;
                        assert!(
                            matches!(
                                error,
                                CheckedGenericPlanError::Provider(SpyError(actual)) if actual == query
                            ),
                            "{name} {query:?}: {error:?}"
                        );
                    }
                    // A route that never asks this query succeeds.
                    Ok(_) => {
                        assert_eq!(left.count(query), 0, "{name} {query:?}");
                        continue;
                    }
                }
                assert_eq!(right.algebra_calls(), 0, "{name} {query:?}");
                assert_eq!(cache_entries(), (0, 0), "{name} {query:?}");
                assert_eq!(coefficient_admissions(), 0, "{name} {query:?}");
                let transformers = crate::tree_transform::take_completed_transformer_activity();
                assert_eq!(transformers.publications, 0, "{name} {query:?}");
            }
            assert!(failed >= 1, "{name}: some query fails");
            if route == Route::DynamicTree {
                tenet_core::clear_structure_caches();
                left.reset();
                left.malformed.set(Some(Query::R));
                let error =
                    contract((&lhs, &lhs_data), (&rhs, &rhs_data), axes(), nout).unwrap_err();
                assert!(
                    matches!(
                        error,
                        CheckedGenericPlanError::SymbolShape { symbol: "R", .. }
                    ),
                    "{name}"
                );
                assert_eq!(cache_entries(), (0, 0), "{name}");
            }
        }
    }

    /// What: each fixture plans its route, and the GEMM failing on the first
    /// or a later job of the replay returns the error without publishing:
    /// no layout, transformer, composed coefficient or output, so a failed
    /// call leaves nothing behind (#2063). The pooled scratch a warm call
    /// sized is reused, not grown, by a failing call.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_backend_failure_publishes_nothing_and_keeps_retained_scratch() {
        if !isolated(
            "checked_contraction_backend_failure_publishes_nothing_and_keeps_retained_scratch",
        ) {
            return;
        }
        for (name, n, lhs_axes, rhs_axes, output, nout, route) in FIXTURES {
            let (_left, lhs, _right, rhs) = bound_pair(n, n);
            let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
            let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
            let axes = || fixture_axes(lhs_axes, rhs_axes, output);

            // Cold failure: nothing resident afterwards.
            tenet_core::clear_structure_caches();
            crate::tree_transform::take_completed_transformer_activity();
            let mut context = failing_context(Some(0));
            let error = tensorcontract_checked_generic(
                &mut context,
                (&lhs, &lhs_data),
                (&rhs, &rhs_data),
                axes(),
                nout,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
                ),
                "{name}"
            );
            assert_eq!(last_route(&context), route, "{name}");
            assert_eq!(cache_entries(), (0, 0), "{name}");
            assert_eq!(coefficient_admissions(), 0, "{name}");
            let transformers = crate::tree_transform::take_completed_transformer_activity();
            assert_eq!(transformers.publications, 0, "{name}");

            // Warm failure on the last job, after a success sized the scratch.
            context.contract_backend_mut().fail_at = None;
            let reference = tensorcontract_checked_generic(
                &mut context,
                (&lhs, &lhs_data),
                (&rhs, &rhs_data),
                axes(),
                nout,
            )
            .unwrap()
            .1;
            let jobs = context.contract_backend_mut().jobs;
            let resident = cache_entries();
            let retained = context.retained_host_scratch_bytes();
            let backend = context.contract_backend_mut();
            backend.fail_at = Some(backend.jobs + (jobs - 1) / 2);
            let error = tensorcontract_checked_generic(
                &mut context,
                (&lhs, &lhs_data),
                (&rhs, &rhs_data),
                axes(),
                nout,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
                ),
                "{name}"
            );
            assert_eq!(cache_entries(), resident, "{name}");
            assert_eq!(context.retained_host_scratch_bytes(), retained, "{name}");
            context.contract_backend_mut().fail_at = None;
            let repeat = tensorcontract_checked_generic(
                &mut context,
                (&lhs, &lhs_data),
                (&rhs, &rhs_data),
                axes(),
                nout,
            )
            .unwrap()
            .1;
            assert_eq!(
                repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    /// What: a successful canonical contraction commits its destination
    /// once under the left provider and publishes nothing else.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_success_commits_once() {
        if !isolated("checked_contraction_success_commits_once") {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::clear_structure_caches();
        let (output, _) = contract(
            (&lhs, &vec![1.0; lhs.space().required_len().unwrap()]),
            (&rhs, &vec![2.0; rhs.space().required_len().unwrap()]),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
        )
        .unwrap();
        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert_eq!(cache_entries(), (1, 0));
        // The commit guard's style is the last provider event: commit reuses
        // the admitted identity (#2046).
        assert_eq!(left.events.borrow().last(), Some(&Style));
    }

    /// What: the three-stage fixture's staged and output transforms build
    /// their groups, then the core GEMM fails before the commit. Nothing is
    /// published, so a retry replays the identical F/R provider ledger and
    /// error. A committed call then publishes its groups, its intermediates
    /// (cache 2, with the destination) and its three transformers (cache
    /// 3); a repeat binds those transformers, so it makes no group lookup
    /// and no F or R query and returns the same bits.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_publishes_composed_coefficients_only_after_its_commit() {
        if !isolated("checked_contraction_publishes_composed_coefficients_only_after_its_commit") {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(2, 2);
        tenet_core::clear_structure_caches();
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
        let axes = || fixture_axes(&[3, 1], &[0, 3], Some(&[2, 0, 3, 1]));
        let run = |fail_at| {
            let mut context = failing_context(fail_at);
            let result = tensorcontract_checked_generic(
                &mut context,
                (&lhs, &lhs_data),
                (&rhs, &rhs_data),
                axes(),
                2,
            );
            assert_eq!(last_route(&context), Route::DynamicTree);
            result
        };

        let mut ledgers = Vec::new();
        for _attempt in 0..2 {
            left.reset();
            right.reset();
            crate::tree_transform::take_coefficient_group_activity();
            let error = run(Some(0)).unwrap_err();
            assert!(matches!(
                error,
                CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
            ));
            let groups = crate::tree_transform::take_coefficient_group_activity();
            assert!(groups.misses > 0);
            assert_eq!((groups.hits, groups.publications), (0, 0));
            assert_eq!(coefficient_admissions(), 0);
            let transformers = crate::tree_transform::take_completed_transformer_activity();
            assert_eq!((transformers.builds, transformers.publications), (3, 0));
            assert_eq!(cache_entries(), (0, 0));
            assert!(left.count(Query::F) > 0 && left.count(Query::R) > 0);
            // The F/R ledger: layout walks may publish pure-data layouts
            // (#2030), so structural queries can shrink on a retry.
            ledgers.push(
                left.events
                    .borrow()
                    .iter()
                    .copied()
                    .filter(|event| matches!(event, Event::Query(Query::F | Query::R)))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(ledgers[0], ledgers[1]);

        left.reset();
        let (first_space, first) = run(None).unwrap();
        assert!(coefficient_admissions() > 0);
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (3, 3));
        // The destination and the three intermediates span three HomSpaces.
        let committed = cache_entries();
        assert_eq!(committed, (3, 3));
        left.reset();
        crate::tree_transform::take_coefficient_group_activity();
        let (repeat_space, repeat) = run(None).unwrap();
        let groups = crate::tree_transform::take_coefficient_group_activity();
        assert_eq!((groups.hits, groups.misses, groups.publications), (0, 0, 0));
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!(
            (
                transformers.hits,
                transformers.builds,
                transformers.publications
            ),
            (3, 0, 0)
        );
        assert_eq!(cache_entries(), committed);
        assert_eq!((left.count(Query::F), left.count(Query::R)), (0, 0));
        assert_eq!(first_space.space(), repeat_space.space());
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// What (#2063, F7 of the #1860 design review): the three-stage fixture
    /// plans a `DynamicTree` artifact that stages the lhs, the rhs and the
    /// core destination; two of its four committed spaces (the destination
    /// and three intermediates) share a HomSpace, so within the first call
    /// the later preview loses its admission to the earlier and its
    /// transformer publishes under the winner's id: the second call builds
    /// nothing.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn intermediates_sharing_a_homspace_converge_on_one_id_within_the_call() {
        if !isolated("intermediates_sharing_a_homspace_converge_on_one_id_within_the_call") {
            return;
        }
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        tenet_core::clear_structure_caches();
        let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.75, -1.5);
        let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 2.0);
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let mut run = || {
            let result = tensorcontract_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                fixture_axes(&[3, 1], &[0, 3], Some(&[2, 0, 3, 1])),
                2,
            )
            .unwrap();
            assert_eq!(last_route(&context), Route::DynamicTree);
            result
        };
        crate::tree_transform::take_completed_transformer_activity();
        let (cold_space, cold) = run();
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.hits, transformers.builds), (0, 3));
        let committed = cache_entries();
        // Four committed spaces, three resident structures.
        assert_eq!(committed.0, 3);
        for _ in 0..3 {
            let (space, data) = run();
            let transformers = crate::tree_transform::take_completed_transformer_activity();
            assert_eq!(
                (
                    transformers.hits,
                    transformers.builds,
                    transformers.publications
                ),
                (3, 0, 0)
            );
            assert_eq!(cache_entries(), committed);
            assert_eq!(space.space(), cold_space.space());
            assert_eq!(
                data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                cold.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    /// What (U15): general checked contraction with a fermionic or anyonic
    /// provider returns the typed `Unsupported` after the pair admission and
    /// one braiding read, before any structural query; canonical
    /// composition crosses no legs and stays admitted.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_requires_bosonic_braiding_and_compose_does_not() {
        for braiding in [BraidingStyleKind::Fermionic, BraidingStyleKind::Anyonic] {
            let (left, lhs, right, rhs) = bound_pair(1, 1);
            left.braiding.set(braiding);
            right.braiding.set(braiding);
            let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
            let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
            let error = contract(
                (&lhs, &lhs_data),
                (&rhs, &rhs_data),
                TensorContractSpec::with_default_output_order(&[1], &[0]),
                1,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckedGenericPlanError::Operation(
                        OperationError::UnsupportedTensorContractScope {
                            message: "checked Generic contraction requires Bosonic braiding",
                        }
                    )
                ),
                "{braiding:?}: {error:?}"
            );
            assert_eq!(*left.events.borrow(), [Identity, Style, Braiding]);
            assert_eq!(*right.events.borrow(), [Identity, Style]);
            assert_eq!(left.algebra_calls(), 0);

            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            let (_, data) = tensorcompose_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
            )
            .unwrap();
            assert!(data.iter().any(|&value| value != 0.0), "{braiding:?}");
        }
    }

    /// What: a staged space the planned route does not replay over, as a
    /// declined `copyC` temporary would be, is dropped by the transaction's
    /// commit unpublished; the temporary the CopyC route uses is committed.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_declined_staged_space_is_not_published() {
        if !isolated("a_declined_staged_space_is_not_published") {
            return;
        }
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        tenet_core::clear_structure_caches();
        let provider = lhs.provider();
        let authority = CheckedAuthority::contraction(&lhs).unwrap();
        let (lhs_axes, rhs_axes, output) = ([2, 3], [0, 1], [1, 0, 2, 3]);
        let destination = lhs
            .prepare_final_homspace_generic_from_checked(provider, || {
                FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
                    provider,
                    lhs.space().homspace(),
                    rhs.space().homspace(),
                    &lhs_axes,
                    &rhs_axes,
                    &output,
                    2,
                )
                .map_err(CheckedGenericPlanError::from)
            })
            .unwrap();
        let preview = destination.preview();
        let mut txn = CheckedContractTxn::new();
        let resolution = plan_contract_in::<HostEagerExecutor, CheckedGenericAdmissionMode, _>(
            &mut txn,
            PlanTarget::checked(authority, &preview),
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            (&lhs_axes, &rhs_axes, &output),
            None,
        )
        .unwrap();
        assert!(resolution.copy_c().is_some());
        let declined = || {
            lhs.prepare_final_homspace_generic_from_checked(provider, || {
                lhs.space()
                    .homspace()
                    .try_permute_generic_checked(provider, &[0, 1, 2], &[3])
                    .map_err(CheckedGenericPlanError::from)
            })
            .unwrap()
        };
        drop(txn.stage(declined()));
        assert_eq!(cache_entries(), (0, 0));
        txn.commit(
            (
                Arc::clone(preview.structure()),
                Arc::clone(preview.structure()),
            ),
            &resolution,
        );
        // The temporary only. The declined space is dropped, and the output
        // transformer stays unpublished: its destination (this test's
        // uncommitted preview) is not canonical.
        assert_eq!(cache_entries(), (1, 0));
        // A complete-cache hit lends the one resident structure; a miss
        // previews a fresh copy each time.
        let resident = |prepared: PreparedCheckedGenericDynamicSpace| {
            Arc::ptr_eq(&prepared.shared_structure(), &prepared.shared_structure())
        };
        assert!(!resident(declined()));
        let temporary = lhs
            .prepare_final_homspace_generic_from_checked(provider, || {
                Ok::<_, CheckedGenericPlanError<SpyError>>(preview.homspace().clone())
            })
            .unwrap();
        assert!(resident(temporary));
    }

    /// Resident `(cache 2, cache 3)` entries.
    fn cache_entries() -> (usize, usize) {
        let entries = |kind| tenet_core::structure_cache_info(kind).entries();
        (
            entries(tenet_core::StructureCacheKind::DegeneracyStructure),
            entries(tenet_core::StructureCacheKind::CompletedTreeTransformer),
        )
    }

    fn coefficient_admissions() -> u64 {
        tenet_core::structure_cache_info(tenet_core::StructureCacheKind::TreeTransformCoefficients)
            .admissions()
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_warm_checked_generic_contraction_converts_at_most_one_pack_per_stage() {
        // Isolated: a concurrent cache clear would force a rebuild.
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_PACKS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::a_warm_checked_generic_contraction_converts_at_most_one_pack_per_stage",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated pack test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // Why (#2101): each stage replays into its own context workspace, so
        // no stage evicts another's Multi pack: a warm call converts at most
        // one pack per stage (a shared workspace would convert three into one
        // slot). #2063 tightens this to none: a warm call binds the completed
        // transformers of the first call, whose packs are already converted.
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        let lhs_data = (0..lhs.space().required_len().unwrap())
            .map(|index| index as f64 - 1.5)
            .collect::<Vec<_>>();
        let rhs_data = (0..rhs.space().required_len().unwrap())
            .map(|index| 2.0 - index as f64)
            .collect::<Vec<_>>();
        let mut context =
            TensorContractFusionExecutionContext::<f64, tenet_core::RuleIdentity>::default();
        let mut run = || {
            let (_, data) = tensorcontract_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::new(
                    &[3, 1],
                    &[0, 3],
                    tenet_operations::OutputAxisOrder::Axes(&[2, 0, 3, 1]),
                ),
                2,
            )
            .unwrap();
            (data, context.eager_coefficient_pack_builds())
        };
        let (cold_data, cold) = run();
        assert_eq!(cold, [1, 1, 1], "all three stages recouple");
        let mut previous = cold;
        for _ in 0..3 {
            let (data, builds) = run();
            assert_eq!(data, cold_data);
            assert_eq!(builds, previous, "a warm call converts no pack");
            previous = builds;
        }
    }

    /// Rank-1 legs with the listed `(sector, degeneracy)` pairs on each side.
    fn homspace_with(codomain: &[(usize, usize)], domain: &[(usize, usize)]) -> FusionTreeHomSpace {
        let leg = |sectors: &[(usize, usize)]| {
            SectorLeg::new(
                sectors
                    .iter()
                    .map(|&(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
                false,
            )
        };
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(codomain)]),
            FusionProductSpace::new([leg(domain)]),
        )
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn bound_space(homspace: FusionTreeHomSpace) -> BoundDynamicFusionMapSpace<CheckedGenericSpy> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::new(CheckedGenericSpy::new()),
            homspace,
        )
        .unwrap()
    }

    fn assert_inactive_sector_zero_and_active_product<D>(
        one: D,
        two: D,
        six: D,
        is_positive_zero: fn(D) -> bool,
    ) where
        D: DenseRecouplingScalar
            + RecouplingCoefficientAction<f64>
            + ConjugateValue
            + Copy
            + Zero
            + ZeroBytes
            + std::fmt::Debug,
    {
        // lhs `[0:1, 1:2] <- [1:3]` and rhs `[1:3] <- [0:1, 1:2]` couple only
        // to sector 1; the destination `[0:1, 1:2] <- [0:1, 1:2]` also has the
        // sector-0 block, which no GEMM writes.
        let lhs = bound_space(homspace_with(&[(0, 1), (1, 2)], &[(1, 3)]));
        let rhs = bound_space(homspace_with(&[(1, 3)], &[(0, 1), (1, 2)]));
        let lhs_data = vec![one; lhs.space().required_len().unwrap()];
        let rhs_data = vec![two; rhs.space().required_len().unwrap()];
        let mut context = TensorContractFusionExecutionContext::<D, RuleIdentity>::default();
        let (output, data) = tensorcontract_owned_checked_generic_in_context(
            &mut context,
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
        )
        .unwrap();
        let structure = output.space().structure();
        assert_eq!(structure.block_count(), 2);
        assert_eq!(data.len(), 1 + 4);
        for index in 0..structure.block_count() {
            // The sector-0 block is the 1x1 one, the sector-1 block the 2x2 one.
            let block = structure.block(index).unwrap();
            let range = block.offset()..block.offset() + block.element_count().unwrap();
            match range.len() {
                1 => assert!(data[range].iter().all(|&v| is_positive_zero(v))),
                4 => assert!(data[range].iter().all(|&v| v == six)),
                other => panic!("unexpected block length {other}"),
            }
        }
    }

    #[test]
    fn owned_checked_generic_contract_leaves_inactive_sectors_exactly_zero() {
        assert_inactive_sector_zero_and_active_product::<f64>(1.0, 2.0, 6.0, |v| v.to_bits() == 0);
        assert_inactive_sector_zero_and_active_product::<num_complex::Complex64>(
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(2.0, 0.0),
            num_complex::Complex64::new(6.0, 0.0),
            |v| v.re.to_bits() == 0 && v.im.to_bits() == 0,
        );
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_reset_straddling_the_publication_leaves_nothing_resident() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_RESET_STRADDLE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::a_reset_straddling_the_publication_leaves_nothing_resident",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated reset-straddle test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // What: a clear between the commits and the first transformer
        // publication refuses every staged publication (the call's epoch is
        // older), the result is unchanged, and the next call rebuilds.
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        let lhs_data = (0..lhs.space().required_len().unwrap())
            .map(|index| index as f64 - 1.5)
            .collect::<Vec<_>>();
        let rhs_data = (0..rhs.space().required_len().unwrap())
            .map(|index| 2.0 - index as f64)
            .collect::<Vec<_>>();
        let mut context =
            TensorContractFusionExecutionContext::<f64, tenet_core::RuleIdentity>::default();
        let mut run = || {
            tensorcontract_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::new(
                    &[3, 1],
                    &[0, 3],
                    tenet_operations::OutputAxisOrder::Axes(&[2, 0, 3, 1]),
                ),
                2,
            )
            .unwrap()
            .1
        };
        tenet_core::clear_structure_caches();
        let reference = run();
        tenet_core::clear_structure_caches();
        crate::tree_transform::take_completed_transformer_activity();
        crate::tree_transform::before_next_completed_publication(Box::new(
            tenet_core::clear_structure_caches,
        ));
        let straddled = run();
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (3, 3));
        assert_eq!(cache_entries().1, 0);
        assert_eq!(straddled, reference);
        let rebuilt = run();
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.hits, transformers.builds), (0, 3));
        assert_eq!(rebuilt, reference);
    }
}
