//! The mode-specific coefficient steps of tenet-tensors' operations, behind
//! one sealed family implemented by the two admission-mode markers (#1855).
//!
//! An operation is otherwise mode-free: only the categorical coefficients a
//! plan carries, the adjoint space and the quantum dimension depend on
//! whether the provider is multiplicity-free or checked Generic, as in
//! TensorKit, where each fusion-tree move branches on `FusionStyle` only
//! inside its coefficient (`braiding_manipulations.jl:132,157`).
//!
//! Why three traits rather than one: a trait impl has one provider bound per
//! mode, and the checked steps need different provider capabilities. The
//! adjoint space needs only `CheckedGenericFusion`; `dim` needs
//! `CheckedGenericRigidSymbols`; the trace needs the twist of
//! `CheckedGenericPivotal`. Bounding one checked impl by the strongest trait
//! would narrow the facade dispatch that forwards here (for example
//! `TypedAdjointSpace` or `TypedSpaceModeDispatch`) to pivotal
//! providers. So `RigidCoefficientAlgebra` and `PivotalCoefficientAlgebra`
//! extend `CoefficientAlgebra` along the provider capability hierarchy; they
//! are not a second mode axis.
//!
//! [`PlanningAlgebra`] holds the two planning steps (#2024). Why a sibling
//! rather than methods of `RigidCoefficientAlgebra`: the multiplicity-free
//! planner runs for any `R::Scalar` (for example `Complex64` rules), while
//! the multiplicity-free impls above are `Scalar = f64`. Relaxing those would
//! widen the facade dispatch that forwards to them (`TypedAdjointSpace`,
//! `TypedSpaceModeDispatch`) to complex-scalar rules.

use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockKey, BlockStructure, BraidingStyleKind, CheckedFusionAlgebra, CheckedGenericAdmissionMode,
    CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols, CoreError,
    FusionStyleKind, FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeAdmissionMode,
    MultiplicityFreeRigidSymbols, OrientedFusionTreeHomSpace, RuleIdentity, SectorId,
};
use tenet_operations::TreeTransformStructure;

use crate::contract::{
    compile_checked_generic_core_plan_general, compile_contract_twist, compile_core_dst,
    compile_derived_core_plan, compile_fusion_block_contract_plan_prelowered_validated,
    compile_prelowered_dynamic_tree, compile_transformed_source, contract_axes_require_twist,
    core_homspace_matches, prepare_tensorcontract_fusion_plan_dyn_raw_canonical,
    rhs_contract_twist_factor_oriented, select_complete_tensorcontract_fusion_plan,
    tree_transform_operation_axes, validate_fusion_contract_rule, CheckedAuthority,
    CheckedContractTxn, DynamicFusionCoreDstEntry, DynamicFusionMapSpace,
    DynamicFusionTransformedSourceEntry, DynamicTreeExecutionArtifact, FusionContractPlan,
    FusionOperand, FusionOperandLayout, LayoutKeyBuilder, PlanTarget,
    PreparedCheckedGenericDynamicSpace, StorageContractResolution, ValidatedCoreContract,
    CHECKED_CONTRACTION_REQUIRES_BOSONIC, CHECKED_REQUIRES_DIRECT_OPERANDS,
};
use crate::tree_transform::{
    build_checked_generic_tree_pair_transform_group_plan_validated, lookup_bound, publishable,
    validate_checked_generic_tree_pair_plan_preflight, CheckedGenericTreePairPreflight,
    CheckedPendingCoefficients, CoefficientGroupReuse, CompletedTransformerKey, OrientedBasisOrder,
    TransformerMode, TreeTransformPlanning, TreeTransformScope,
};
use crate::{
    adjoint_bound_space_dyn, adjoint_bound_space_dyn_generic_checked,
    validate_oriented_fusion_layout, BoundDynamicFusionMapSpace, CheckedGenericPlanError,
    DenseBlockScalar, OperationError, TensorTraceAxisSpec, TensorTraceFusionStructure,
    TreeTransformOperation, TreeTransformRuleCacheKey,
};
use tenet_operations::fusion_replay::FusionBlockContractPlan;
use tenet_operations::fusion_replay::MatrixOp;
use tenet_operations::{OutputAxisOrder, TensorContractFusionProfile, TensorContractSpec};

mod sealed {
    pub trait Sealed {}
    impl Sealed for tenet_core::MultiplicityFreeAdmissionMode {}
    impl Sealed for tenet_core::CheckedGenericAdmissionMode {}
}

/// Mode-specific steps every provider of a mode supports.
#[doc(hidden)]
pub trait CoefficientAlgebra<R>: sealed::Sealed {
    /// The categorical coefficient type of this mode's plans.
    type Coeff;
    type Error: From<OperationError>;

    /// The space of the adjoint, codomain and domain exchanged.
    fn adjoint_space(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error>;
}

/// Mode-specific steps that need the rigid data (quantum dimensions, F and R
/// symbols).
#[doc(hidden)]
pub trait RigidCoefficientAlgebra<R>: CoefficientAlgebra<R> {
    /// The quantum dimension `dim(c)`: the one weight of truncation budgets
    /// and quantum-dimension-weighted reductions in every mode, exact where
    /// the provider knows it (`CheckedGenericRigidSymbols::try_dim_scalar`).
    ///
    /// TensorKit `cfaa073e` `findtruncated(::SectorVector, ...)` weights by
    /// `dim(c)` (`src/factorizations/truncation.jl:187,239`); QSpace
    /// `d2d3d7da` `SVD_Data::dmrgTruncate` weights its norm by the integer
    /// multiplet dimension `qdim_tot` (`Source/mpsortho.cc:616`).
    fn dim(provider: &R, sector: SectorId) -> Result<f64, Self::Error>;
}

/// Mode-specific steps that need a pivotal provider (the twist).
#[doc(hidden)]
pub trait PivotalCoefficientAlgebra<R>: RigidCoefficientAlgebra<R> {
    /// The trace terms, coefficients and strides `dst <- tr(src)` replays.
    fn trace_terms(
        dst: &BoundDynamicFusionMapSpace<R>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<TensorTraceFusionStructure<Self::Coeff>, Self::Error>;
}

impl<R> CoefficientAlgebra<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    type Coeff = f64;
    type Error = OperationError;

    fn adjoint_space(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error> {
        adjoint_bound_space_dyn(space)
    }
}

impl<R> RigidCoefficientAlgebra<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    fn dim(provider: &R, sector: SectorId) -> Result<f64, Self::Error> {
        Ok(provider.dim_scalar(sector))
    }
}

impl<R> PivotalCoefficientAlgebra<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra,
{
    fn trace_terms(
        dst: &BoundDynamicFusionMapSpace<R>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<TensorTraceFusionStructure<f64>, Self::Error> {
        TensorTraceFusionStructure::compile_fusion_dyn_checked(dst, src, axes)
    }
}

impl<R> CoefficientAlgebra<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericFusion,
{
    type Coeff = f64;
    type Error = CheckedGenericPlanError<R::Error>;

    fn adjoint_space(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error> {
        adjoint_bound_space_dyn_generic_checked(space)
    }
}

impl<R> RigidCoefficientAlgebra<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
{
    fn dim(provider: &R, sector: SectorId) -> Result<f64, Self::Error> {
        provider
            .try_dim_scalar(sector)
            .map_err(CheckedGenericPlanError::Provider)
    }
}

impl<R> PivotalCoefficientAlgebra<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericPivotal<Scalar = f64>,
{
    fn trace_terms(
        dst: &BoundDynamicFusionMapSpace<R>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<TensorTraceFusionStructure<f64>, Self::Error> {
        // Compile-only: nothing commits, so a successful compile publishes.
        let mut coefficients = CheckedPendingCoefficients::new();
        let structure = crate::tensortrace::compile_fusion_dyn_generic_checked(
            dst,
            src,
            axes,
            &mut coefficients,
        )?;
        coefficients.flush();
        Ok(structure)
    }
}

/// The source a tree-pair transform structure is compiled against.
pub(crate) enum TreeStructureSource<'a> {
    /// Blocks in storage order; `storage_conjugate` replays a lazy adjoint
    /// stored as its parent.
    Stored {
        structure: &'a Arc<BlockStructure>,
        storage_conjugate: bool,
    },
    /// A lazy adjoint addressed through its parent's storage blocks.
    Oriented(&'a FusionOperandLayout<'a>),
    /// A lazy adjoint whose `logical` blocks map one by one onto the storage
    /// blocks of `operand`'s parent: the checked Generic eager transform's
    /// source. Why not `Oriented`: that layout is prepared by the
    /// multiplicity-free layout primer, which a checked provider lacks.
    StorageMapped {
        logical: &'a Arc<BlockStructure>,
        operand: FusionOperand<'a>,
    },
}

/// One eager owned transform between its staged destination and commit.
pub(crate) struct StagedTransform<S, P> {
    /// The destination structure replay writes: committed for
    /// multiplicity-free, an uninterned preview for checked Generic.
    pub(crate) preview: Arc<BlockStructure>,
    pub(crate) nout: usize,
    /// What [`PlanningAlgebra::commit_transform`] consumes.
    pub(crate) stage: S,
    /// The source admission the staging proved, handed to
    /// [`PlanningAlgebra::tree_structure_with`].
    pub(crate) proof: P,
}

/// The mode-dependent planning steps of contraction and tree transforms:
/// each has one call site in the multiplicity-free planner, and the checked
/// Generic engine calls the same method.
pub(crate) trait PlanningAlgebra<R>: sealed::Sealed {
    type Scalar;
    type Error: From<OperationError>;
    /// The caller-owned planning state of one resolution scope (completed
    /// transformers and composed coefficients themselves are
    /// process-global): per context for multiplicity-free resolution, per
    /// contraction call for checked Generic, whose pending coefficients
    /// publish only after that call's commit.
    type StructureCache;
    /// What an eager transform's staging proved about its source.
    type SourceProof<'a>
    where
        R: 'a;
    /// What an eager transform holds between staging and commit.
    type TransformStage;
    /// The per-call planning state a contraction owns beside the context's:
    /// none for multiplicity-free, the call's transaction for checked
    /// Generic.
    type ContractTxn;
    /// What the planner derives spaces under: the multiplicity-free layout
    /// primer, or the checked binding whose provider admission stages them.
    type SpaceAuthority<'a>: Copy
    where
        R: 'a;

    /// The core coefficient of one RHS coupled sector, whose core-codomain
    /// row trees are `row_trees`: `Some(alpha)` when it is uniform over them,
    /// `None` when the core cannot carry it as one per-job alpha.
    fn core_alpha<'t>(
        rule: &R,
        rhs: OrientedFusionTreeHomSpace<'_>,
        rhs_contracting_axes: &[usize],
        row_trees: impl IntoIterator<Item = &'t FusionTreeKey>,
    ) -> Result<Option<Self::Scalar>, Self::Error>
    where
        Self::Scalar: DenseBlockScalar;

    /// The tree-pair transform structure `dst <- operation(src)`.
    fn tree_structure(
        cache: &mut Self::StructureCache,
        rule: &R,
        operation: &TreeTransformOperation,
        dst: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
    ) -> Result<TreeTransformStructure<Self::Scalar>, Self::Error> {
        Self::tree_structure_with(cache, rule, operation, dst, src, None)
    }

    /// [`Self::tree_structure`] whose source admission, when `proof` is
    /// given, the caller already ran: a miss builds without repeating it.
    fn tree_structure_with(
        cache: &mut Self::StructureCache,
        rule: &R,
        operation: &TreeTransformOperation,
        dst: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
        proof: Option<&Self::SourceProof<'_>>,
    ) -> Result<TreeTransformStructure<Self::Scalar>, Self::Error>;

    /// Admits an eager owned transform of `logical` (stored as the parent
    /// `adjoint_of` when it is that parent's lazy adjoint, `src_len`
    /// elements) and stages its destination.
    fn stage_transform<'a>(
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
    ) -> Result<StagedTransform<Self::TransformStage, Self::SourceProof<'a>>, Self::Error>;

    /// The planning state an eager transform resolves its structure with:
    /// the context's own, or the one its staging opened.
    fn transform_cache<'c>(
        planning: &'c mut TreeTransformPlanning,
        stage: &'c mut Self::TransformStage,
    ) -> &'c mut Self::StructureCache;

    /// The planning state a contraction resolves its plan with: the
    /// context's own, or the call's transaction.
    fn contract_cache<'c>(
        planning: &'c mut TreeTransformPlanning,
        txn: &'c mut Self::ContractTxn,
    ) -> &'c mut Self::StructureCache;

    /// Commits the staged destination after replay, then publishes what the
    /// transform staged.
    fn commit_transform(
        logical: &BoundDynamicFusionMapSpace<R>,
        stage: Self::TransformStage,
        preview: Arc<BlockStructure>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error>;

    /// Checks that the destination and both operand spaces belong to `rule`
    /// before the planner reads sectors through it.
    fn validate_spaces(
        rule: &R,
        authority: Self::SpaceAuthority<'_>,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
    ) -> Result<(), Self::Error>;

    /// Whether a contraction may carry TensorKit's fermionic supertrace twist
    /// (`blas_contract!`, `tensoroperations.jl:398-410` @cfaa073).
    fn contract_twist_possible(
        rule: &R,
        authority: Self::SpaceAuthority<'_>,
    ) -> Result<bool, Self::Error>;

    /// Whether `dst` is the contracted HomSpace of the two oriented
    /// operands, as the core rung requires of its destination.
    fn destination_matches(
        rule: &R,
        lhs: OrientedFusionTreeHomSpace<'_>,
        rhs: OrientedFusionTreeHomSpace<'_>,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst: &FusionTreeHomSpace,
    ) -> Result<bool, Self::Error>;

    /// The packed core plan of a validated core whose coupled-sector tiling
    /// is not canonical (#1517), for an executor with `IRREGULAR_CORE`.
    fn irregular_core(
        validated: ValidatedCoreContract<'_, R>,
        authority: Self::SpaceAuthority<'_>,
        dst: &DynamicFusionMapSpace,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
    ) -> Result<FusionBlockContractPlan<Self::Scalar>, Self::Error>
    where
        Self::Scalar: DenseBlockScalar;

    /// The `DynamicTree` candidate of two direct operands: TensorKit
    /// `contract!`'s `_contract_memcost` selection (`tensoroperations.jl:314-378`).
    fn dynamic_tree_plan(
        rule: &R,
        authority: Self::SpaceAuthority<'_>,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<FusionContractPlan, Self::Error>
    where
        Self::Scalar: DenseBlockScalar;

    /// The core-layout space of one stored operand and its transformer:
    /// TensorKit `blas_contract!`'s `copyA`/`copyB` (`tensoroperations.jl:411-433`).
    #[allow(clippy::too_many_arguments)]
    fn transformed_source(
        cache: &mut Self::StructureCache,
        authority: Self::SpaceAuthority<'_>,
        rule: &R,
        source: &DynamicFusionMapSpace,
        source_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperation,
        source_conjugate: bool,
    ) -> Result<DynamicFusionTransformedSourceEntry<Self::Scalar>, Self::Error>;

    /// The core destination of a derived plan and the output transformer
    /// from it into `output_dst`.
    #[allow(clippy::too_many_arguments)]
    fn core_destination(
        cache: &mut Self::StructureCache,
        authority: Self::SpaceAuthority<'_>,
        rule: &R,
        core_left: &DynamicFusionMapSpace,
        core_right: &DynamicFusionMapSpace,
        plan: &FusionContractPlan,
        output_dst: &DynamicFusionMapSpace,
    ) -> Result<DynamicFusionCoreDstEntry<Self::Scalar>, Self::Error>;

    /// TensorKit `copyC`'s temporary (`tensoroperations.jl:436-446`): the
    /// default-order contraction of `first` and `second`.
    #[allow(clippy::too_many_arguments)]
    fn copy_c_temporary(
        cache: &mut Self::StructureCache,
        authority: Self::SpaceAuthority<'_>,
        rule: &R,
        first: FusionOperand<'_>,
        second: FusionOperand<'_>,
        first_axes: &[usize],
        second_axes: &[usize],
        open_axes: &[usize],
        first_open: usize,
    ) -> Result<DynamicFusionMapSpace, Self::Error>;

    /// The fermionic contraction twist of one materialized core operand, as
    /// `(block offset, θ_b ≠ 1)` sorted by offset.
    fn contract_twist_scales(
        rule: &R,
        authority: Self::SpaceAuthority<'_>,
        space: &DynamicFusionMapSpace,
        core_right: &FusionTreeHomSpace,
        space_is_core_left: bool,
        rhs_contracting_axes: &[usize],
    ) -> Result<Vec<(usize, Self::Scalar)>, Self::Error>
    where
        Self::Scalar: DenseBlockScalar;

    /// The core plan of operands a [`FusionContractPlan`] derived, whose core
    /// geometry holds by construction.
    fn derived_core_plan(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        core_axes: TensorContractSpec<'_>,
    ) -> Result<Arc<FusionBlockContractPlan<Self::Scalar>>, Self::Error>
    where
        Self::Scalar: DenseBlockScalar;

    /// The `DynamicTree` artifact of a contraction with a lazy-adjoint
    /// operand, whose sources are read through logical-key projections.
    fn prelowered_dynamic_tree_artifact(
        cache: &mut Self::StructureCache,
        target: PlanTarget<'_, R, Self::SpaceAuthority<'_>>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: TensorContractSpec<'_>,
        profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<DynamicTreeExecutionArtifact<Self::Scalar>, Self::Error>
    where
        Self::Scalar: DenseBlockScalar;
}

/// What the one owned contraction entry is asked for.
#[derive(Clone, Copy)]
pub(crate) enum ContractRequest<'s> {
    /// `lhs·rhs` over `axes`, whose conjugation flags are the operands',
    /// split after the first `codomain_rank` output axes.
    Contract {
        axes: TensorContractSpec<'s>,
        codomain_rank: usize,
    },
    /// TensorKit `mul!`: `lhs.domain` glued to `rhs.codomain` in order.
    Compose,
}

/// One contraction operand: its logical space (the authority its
/// destination derives from), its storage operand and its payload length.
pub(crate) type ContractSide<'a, R> = (&'a BoundDynamicFusionMapSpace<R>, FusionOperand<'a>, usize);

/// One owned contraction between its staged destination and commit.
pub(crate) struct StagedContraction<S, A, T> {
    /// What [`ContractStaging::commit_contraction`] consumes.
    pub(crate) stage: S,
    /// What the planner derives spaces under.
    pub(crate) authority: A,
    /// The call's planning state ([`PlanningAlgebra::ContractTxn`]).
    pub(crate) txn: T,
    /// The destination payload length.
    pub(crate) len: usize,
}

/// The destination steps of the one owned contraction/composition entry
/// (#1864): multiplicity-free derives and publishes its destination at
/// once; checked Generic admits, stages it, and commits after replay
/// (#2063).
///
/// Why a sibling of [`PlanningAlgebra`]: the multiplicity-free destination
/// of a lazy adjoint derives through the checked fusion algebra
/// (`contracted_multiplicity_free_oriented`), a provider bound the
/// multiplicity-free planner does not carry.
pub(crate) trait ContractStaging<R>: PlanningAlgebra<R> {
    /// What a contraction holds between staging and commit.
    type ContractStage;

    /// Admits the operands and derives or stages the destination of
    /// `request`.
    #[allow(clippy::type_complexity)]
    fn stage_contraction<'a>(
        lhs: ContractSide<'a, R>,
        rhs: ContractSide<'a, R>,
        request: ContractRequest<'_>,
    ) -> Result<
        StagedContraction<Self::ContractStage, Self::SpaceAuthority<'a>, Self::ContractTxn>,
        Self::Error,
    >;

    /// The provider and destination space the plan targets.
    fn contract_destination<'s>(
        stage: &'s Self::ContractStage,
        lhs: &'s BoundDynamicFusionMapSpace<R>,
    ) -> (&'s R, &'s DynamicFusionMapSpace);

    /// Commits the staged destination after replay, then publishes what the
    /// call staged for the route `resolution` replayed.
    fn commit_contraction(
        lhs: &BoundDynamicFusionMapSpace<R>,
        stage: Self::ContractStage,
        txn: Self::ContractTxn,
        resolution: &StorageContractResolution<Self::Scalar>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error>;
}

/// The coefficient bounds of the multiplicity-free tree-structure cache.
/// Why not `DenseBlockScalar`: the eager transform entries share the
/// multiplicity-free [`PlanningAlgebra`] impl without it; only the core
/// steps need it.
pub(crate) trait MultiplicityFreePlanningScalar:
    Copy + Add<Output = Self> + Mul<Output = Self> + Zero + Send + Sync + 'static
{
}

impl<T> MultiplicityFreePlanningScalar for T where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync + 'static
{
}

impl<R> PlanningAlgebra<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: MultiplicityFreePlanningScalar,
{
    type Scalar = R::Scalar;
    type Error = OperationError;
    type StructureCache = TreeTransformPlanning;
    type SourceProof<'a>
        = ()
    where
        R: 'a;
    /// The destination, derived and published by staging.
    type TransformStage = BoundDynamicFusionMapSpace<R>;
    type ContractTxn = ();
    type SpaceAuthority<'a>
        = LayoutKeyBuilder<R>
    where
        R: 'a;

    /// Unit unless fermionic; a fermionic twist that varies within the
    /// sector declines, so the `DynamicTree` artifact applies it per block.
    fn core_alpha<'t>(
        rule: &R,
        rhs: OrientedFusionTreeHomSpace<'_>,
        rhs_contracting_axes: &[usize],
        row_trees: impl IntoIterator<Item = &'t FusionTreeKey>,
    ) -> Result<Option<R::Scalar>, OperationError>
    where
        R::Scalar: DenseBlockScalar,
    {
        let mut row_trees = row_trees.into_iter();
        let Some(first) = row_trees.next() else {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "canonical RHS coupled region has no row fusion tree",
            });
        };
        let alpha = rhs_contract_twist_factor_oriented(rule, rhs, rhs_contracting_axes, first)?;
        for tree in row_trees {
            if rhs_contract_twist_factor_oriented(rule, rhs, rhs_contracting_axes, tree)? != alpha {
                return Ok(None);
            }
        }
        Ok(Some(alpha))
    }

    fn tree_structure_with(
        cache: &mut Self::StructureCache,
        rule: &R,
        operation: &TreeTransformOperation,
        dst: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
        _proof: Option<&()>,
    ) -> Result<TreeTransformStructure<R::Scalar>, OperationError> {
        match src {
            TreeStructureSource::Stored {
                structure,
                storage_conjugate,
            } => cache.resolve_tree_pair(rule, operation, dst, structure, storage_conjugate),
            TreeStructureSource::Oriented(source) => cache.resolve_tree_pair_oriented(
                rule,
                operation,
                dst,
                source
                    .adjoint_logical_keys()
                    .expect("only adjoint sources use the oriented transform compiler"),
                || source.adjoint_storage_indices(),
                source.storage_space().structure(),
                source.orientation(),
                source.basis_order(),
                source.rank(),
                |axis| source.storage_axis(axis),
            ),
            TreeStructureSource::StorageMapped { .. } => {
                Err(OperationError::UnsupportedTensorContractScope {
                    message: "multiplicity-free tree structure reads a lazy adjoint through its oriented layout",
                })
            }
        }
    }

    fn stage_transform<'a>(
        logical: &'a BoundDynamicFusionMapSpace<R>,
        _adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        _src_len: usize,
        operation: &TreeTransformOperation,
    ) -> Result<StagedTransform<BoundDynamicFusionMapSpace<R>, ()>, OperationError> {
        let destination = logical.transformed_multiplicity_free(operation)?;
        Ok(StagedTransform {
            preview: Arc::clone(destination.space().structure()),
            nout: destination.space().nout(),
            stage: destination,
            proof: (),
        })
    }

    fn transform_cache<'c>(
        planning: &'c mut TreeTransformPlanning,
        _stage: &'c mut BoundDynamicFusionMapSpace<R>,
    ) -> &'c mut TreeTransformPlanning {
        planning
    }

    fn contract_cache<'c>(
        planning: &'c mut TreeTransformPlanning,
        _txn: &'c mut (),
    ) -> &'c mut TreeTransformPlanning {
        planning
    }

    fn commit_transform(
        _logical: &BoundDynamicFusionMapSpace<R>,
        stage: BoundDynamicFusionMapSpace<R>,
        _preview: Arc<BlockStructure>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError> {
        Ok(stage)
    }

    fn validate_spaces(
        rule: &R,
        _primer: LayoutKeyBuilder<R>,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
    ) -> Result<(), OperationError> {
        validate_fusion_contract_rule(rule, dst, lhs, rhs)
    }

    fn contract_twist_possible(
        rule: &R,
        _primer: LayoutKeyBuilder<R>,
    ) -> Result<bool, OperationError> {
        Ok(rule.braiding_style() == BraidingStyleKind::Fermionic)
    }

    fn destination_matches(
        rule: &R,
        lhs: OrientedFusionTreeHomSpace<'_>,
        rhs: OrientedFusionTreeHomSpace<'_>,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst: &FusionTreeHomSpace,
    ) -> Result<bool, OperationError> {
        core_homspace_matches(
            rule,
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst,
        )
    }

    /// The packed plan over logical-key projections laid out by the primer.
    fn irregular_core(
        validated: ValidatedCoreContract<'_, R>,
        primer: LayoutKeyBuilder<R>,
        dst: &DynamicFusionMapSpace,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
    ) -> Result<FusionBlockContractPlan<R::Scalar>, OperationError>
    where
        R::Scalar: DenseBlockScalar,
    {
        let rule = validated.rule();
        compile_fusion_block_contract_plan_prelowered_validated(
            validated,
            dst,
            &lhs.prepare(rule, primer)?,
            &rhs.prepare(rule, primer)?,
        )
    }

    fn dynamic_tree_plan(
        rule: &R,
        _primer: LayoutKeyBuilder<R>,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<FusionContractPlan, OperationError>
    where
        R::Scalar: DenseBlockScalar,
    {
        prepare_tensorcontract_fusion_plan_dyn_raw_canonical(rule, dst, lhs, rhs, axes)
    }

    fn transformed_source(
        planning: &mut TreeTransformPlanning,
        primer: LayoutKeyBuilder<R>,
        rule: &R,
        source: &DynamicFusionMapSpace,
        source_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperation,
        source_conjugate: bool,
    ) -> Result<DynamicFusionTransformedSourceEntry<R::Scalar>, OperationError> {
        compile_transformed_source(
            planning,
            rule,
            source,
            source_structure,
            operation,
            source_conjugate,
            primer,
        )
    }

    fn core_destination(
        planning: &mut TreeTransformPlanning,
        primer: LayoutKeyBuilder<R>,
        rule: &R,
        core_left: &DynamicFusionMapSpace,
        core_right: &DynamicFusionMapSpace,
        plan: &FusionContractPlan,
        output_dst: &DynamicFusionMapSpace,
    ) -> Result<DynamicFusionCoreDstEntry<R::Scalar>, OperationError> {
        compile_core_dst(
            planning, rule, core_left, core_right, plan, output_dst, primer,
        )
    }

    fn copy_c_temporary(
        _planning: &mut TreeTransformPlanning,
        primer: LayoutKeyBuilder<R>,
        rule: &R,
        first: FusionOperand<'_>,
        second: FusionOperand<'_>,
        first_axes: &[usize],
        second_axes: &[usize],
        open_axes: &[usize],
        first_open: usize,
    ) -> Result<DynamicFusionMapSpace, OperationError> {
        DynamicFusionMapSpace::from_final_homspace_with_primer(
            rule,
            OrientedFusionTreeHomSpace::tensorcontract_homspace(
                rule,
                first.oriented_homspace(),
                second.oriented_homspace(),
                first_axes,
                second_axes,
                open_axes,
                first_open,
            )
            .map_err(OperationError::from_core_preserving_context)?,
            primer,
        )
    }

    fn contract_twist_scales(
        rule: &R,
        _primer: LayoutKeyBuilder<R>,
        space: &DynamicFusionMapSpace,
        core_right: &FusionTreeHomSpace,
        space_is_core_left: bool,
        rhs_contracting_axes: &[usize],
    ) -> Result<Vec<(usize, R::Scalar)>, OperationError>
    where
        R::Scalar: DenseBlockScalar,
    {
        compile_contract_twist(
            rule,
            space,
            core_right,
            space_is_core_left,
            rhs_contracting_axes,
        )
    }

    fn derived_core_plan(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        core_axes: TensorContractSpec<'_>,
    ) -> Result<Arc<FusionBlockContractPlan<R::Scalar>>, OperationError>
    where
        R::Scalar: DenseBlockScalar,
    {
        compile_derived_core_plan(rule, dst, lhs, rhs, core_axes)
    }

    fn prelowered_dynamic_tree_artifact(
        planning: &mut TreeTransformPlanning,
        target: PlanTarget<'_, R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: TensorContractSpec<'_>,
        profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<DynamicTreeExecutionArtifact<R::Scalar>, OperationError>
    where
        R::Scalar: DenseBlockScalar,
    {
        compile_prelowered_dynamic_tree(planning, target, lhs, rhs, axes, profile)
    }
}

impl<R> ContractStaging<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + TreeTransformRuleCacheKey,
    R::Scalar: MultiplicityFreePlanningScalar,
{
    /// The destination, derived and published by staging.
    type ContractStage = BoundDynamicFusionMapSpace<R>;

    /// Two direct operands derive through the owned derivation, a lazy
    /// adjoint through the oriented one.
    fn stage_contraction<'a>(
        (lhs, lhs_operand, _): ContractSide<'a, R>,
        (rhs, rhs_operand, _): ContractSide<'a, R>,
        request: ContractRequest<'_>,
    ) -> Result<
        StagedContraction<BoundDynamicFusionMapSpace<R>, LayoutKeyBuilder<R>, ()>,
        OperationError,
    > {
        let direct = !lhs_operand.storage_conjugate() && !rhs_operand.storage_conjugate();
        let destination = match request {
            ContractRequest::Contract {
                axes,
                codomain_rank,
            } if direct => BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
                lhs,
                rhs,
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                axes.output_permutation(),
                codomain_rank,
            )?,
            ContractRequest::Contract {
                axes,
                codomain_rank,
            } => BoundDynamicFusionMapSpace::contracted_multiplicity_free_oriented(
                lhs,
                lhs_operand,
                rhs,
                rhs_operand,
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                axes.output_permutation(),
                Some(codomain_rank),
            )?,
            ContractRequest::Compose => {
                let lhs_axes = (lhs.space().nout()..lhs.space().rank()).collect::<Vec<_>>();
                let rhs_axes = (0..rhs.space().nout()).collect::<Vec<_>>();
                if direct {
                    BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
                        lhs,
                        rhs,
                        &lhs_axes,
                        &rhs_axes,
                        OutputAxisOrder::identity(),
                    )?
                } else {
                    BoundDynamicFusionMapSpace::contracted_multiplicity_free_oriented(
                        lhs,
                        lhs_operand,
                        rhs,
                        rhs_operand,
                        &lhs_axes,
                        &rhs_axes,
                        OutputAxisOrder::identity(),
                        None,
                    )?
                }
            }
        };
        let len = destination.space().required_len()?;
        Ok(StagedContraction {
            authority: destination.layout_primer(),
            stage: destination,
            txn: (),
            len,
        })
    }

    fn contract_destination<'s>(
        stage: &'s BoundDynamicFusionMapSpace<R>,
        _lhs: &'s BoundDynamicFusionMapSpace<R>,
    ) -> (&'s R, &'s DynamicFusionMapSpace) {
        (stage.provider(), stage.space())
    }

    fn commit_contraction(
        _lhs: &BoundDynamicFusionMapSpace<R>,
        stage: BoundDynamicFusionMapSpace<R>,
        _txn: (),
        _resolution: &StorageContractResolution<R::Scalar>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError> {
        Ok(stage)
    }
}

impl<R> PlanningAlgebra<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
{
    type Scalar = f64;
    type Error = CheckedGenericPlanError<R::Error>;
    /// The contraction call's transaction: its staged intermediates,
    /// composed coefficients and completed transformers, published only
    /// after its commits (#2063).
    type StructureCache = CheckedContractTxn;
    /// The admitted rule identity and the plan preflight of the logical
    /// source.
    type SourceProof<'a>
        = (RuleIdentity, CheckedGenericTreePairPreflight<'a, 'a, R>)
    where
        R: 'a;
    /// The staged destination and the call's transaction.
    type TransformStage = (PreparedCheckedGenericDynamicSpace, CheckedContractTxn);
    type ContractTxn = CheckedContractTxn;
    /// The binding whose provider admission stages derived spaces, with the
    /// entry's twist answer.
    type SpaceAuthority<'a>
        = CheckedAuthority<'a, R>
    where
        R: 'a;

    /// Unit for Bosonic braiding, independent of the trees; the checked
    /// engine implements neither the fermionic twist branch of TensorKit's
    /// `blas_contract!` nor non-symmetric braiding (its `SymmetricBraiding`
    /// guard).
    fn core_alpha<'t>(
        rule: &R,
        _rhs: OrientedFusionTreeHomSpace<'_>,
        _rhs_contracting_axes: &[usize],
        _row_trees: impl IntoIterator<Item = &'t FusionTreeKey>,
    ) -> Result<Option<f64>, Self::Error> {
        if rule.braiding_style() != BraidingStyleKind::Bosonic {
            return Err(CHECKED_CONTRACTION_REQUIRES_BOSONIC.into());
        }
        Ok(Some(f64::one()))
    }

    fn tree_structure_with(
        txn: &mut CheckedContractTxn,
        rule: &R,
        operation: &TreeTransformOperation,
        dst: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
        proof: Option<&Self::SourceProof<'_>>,
    ) -> Result<TreeTransformStructure<f64>, Self::Error> {
        let (structure, storage_conjugate, mapped) = match src {
            TreeStructureSource::Stored {
                structure,
                storage_conjugate,
            } => (structure, storage_conjugate, None),
            TreeStructureSource::StorageMapped { logical, operand } => (
                operand.storage_space().structure(),
                true,
                Some((logical, operand)),
            ),
            TreeStructureSource::Oriented(_) => {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "checked Generic tree structure requires a stored source",
                }
                .into())
            }
        };
        let logical = mapped.map(|(logical, _)| logical);
        // A warm call's intermediates are committed, so their canonical ids
        // hit; a miss builds and stages its publication until the call's
        // commits (#2063).
        let identity = match proof {
            Some((identity, _)) => identity.clone(),
            None => rule.rule_identity(),
        };
        let key = CompletedTransformerKey::new::<f64>(
            identity.clone(),
            TransformerMode::CheckedGeneric,
            TreeTransformScope::TreePair,
            operation,
            tenet_core::FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            storage_conjugate,
            logical.map(|logical| logical.as_ref()),
            dst,
            structure,
        );
        if let Some(hit) = lookup_bound(&key, dst, structure) {
            return Ok(hit);
        }
        // Admission precedes any composed-coefficient lookup. Why the
        // cold call's uncommitted intermediates reuse coefficients: their keys
        // hold sectors and trees, never content ids.
        let preflight;
        let source_proof = match proof {
            Some((_, source_proof)) => source_proof,
            None => {
                preflight = validate_checked_generic_tree_pair_plan_preflight(
                    rule,
                    operation,
                    logical.unwrap_or(structure),
                )?;
                &preflight
            }
        };
        // Why the logical source's groups key the coefficients: the plan is
        // built on it; storage conjugation applies at binding.
        let reuse = CoefficientGroupReuse::<f64>::new(
            identity,
            TransformerMode::CheckedGeneric,
            TreeTransformScope::TreePair,
            operation,
            tenet_core::FusionTreePairOrientation::Direct,
        );
        let plan = build_checked_generic_tree_pair_transform_group_plan_validated(
            operation.clone(),
            source_proof,
            &reuse,
        )?;
        let coefficients = txn.coefficients();
        coefficients.stage(reuse.into_pending());
        let built = match mapped {
            Some((logical, operand)) => plan.compile_shared_structures_with_storage_mapping(
                Arc::clone(dst),
                logical,
                Arc::clone(structure),
                |logical_index| {
                    let BlockKey::FusionTree(logical_key) = logical.block(logical_index)?.key()
                    else {
                        return Err(OperationError::StructureMismatch {
                            tensor: "checked logical source",
                        });
                    };
                    operand.storage_block_index(logical_key)
                },
                |axis| operand.storage_axis(axis),
                true,
            )?,
            None => plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst),
                Arc::clone(structure),
                storage_conjugate,
            )?,
        };
        // A non-canonical logical source keys nothing publishable.
        if logical.is_none_or(|logical| publishable([logical.as_ref()])) {
            coefficients.stage_transformer(key, &built);
        }
        Ok(built)
    }

    /// The identity, style and storage-relation checks, then the plan
    /// preflight of the logical source, inside the staged destination's
    /// producer, so preflight errors precede destination errors and the
    /// structure build reuses the proof.
    fn stage_transform<'a>(
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
    ) -> Result<StagedTransform<Self::TransformStage, Self::SourceProof<'a>>, Self::Error> {
        let source = logical.space();
        let storage = adjoint_of.unwrap_or(logical);
        let storage_source = storage.space();
        let provider = logical.provider();
        let expected = storage_source.required_len()?;
        if src_len != expected {
            return Err(OperationError::ElementCountMismatch {
                expected,
                actual: src_len,
            }
            .into());
        }
        // Before admission: a clear during this request leaves its
        // transformer and composed coefficients unpublished.
        let txn = CheckedContractTxn::new();
        let identity = std::cell::OnceCell::new();
        let destination = std::cell::OnceCell::new();
        let source_proof = std::cell::OnceCell::new();
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let structure =
            FusionTreeHomSpace::prepare_complete_coupled_subblock_structure_generic_checked_with::<
                _,
                CheckedGenericPlanError<R::Error>,
                _,
            >(provider, |provider_identity| {
                let actual =
                    source.validate_transformed_generic_checked_identity(provider_identity)?;
                if provider.fusion_style() != FusionStyleKind::Generic {
                    return Err(CoreError::UnsupportedFusionStyle {
                        expected: FusionStyleKind::Generic,
                        actual: provider.fusion_style(),
                    }
                    .into());
                }
                if let Some(parent) = adjoint_of {
                    if !Arc::ptr_eq(logical.provider_arc(), parent.provider_arc()) {
                        return Err(OperationError::StructureMismatch {
                            tensor: "checked adjoint provider",
                        }
                        .into());
                    }
                    if storage_source
                        .validate_transformed_generic_checked_identity(provider_identity)?
                        != actual
                    {
                        return Err(OperationError::StructureMismatch {
                            tensor: "checked adjoint identity",
                        }
                        .into());
                    }
                    if source.nout() != storage_source.nin()
                        || source.nin() != storage_source.nout()
                        || source.homspace().codomain() != storage_source.homspace().domain()
                        || source.homspace().domain() != storage_source.homspace().codomain()
                    {
                        return Err(OperationError::StructureMismatch {
                            tensor: "checked adjoint relation",
                        }
                        .into());
                    }
                    validate_oriented_fusion_layout(
                        source.structure(),
                        FusionOperand::adjoint(storage_source),
                    )?;
                }
                let proof = validate_checked_generic_tree_pair_plan_preflight(
                    provider,
                    operation,
                    source.structure(),
                )?;
                let homspace = source.homspace().try_permute_generic_checked(
                    provider,
                    codomain_axes,
                    domain_axes,
                )?;
                identity.set(actual).expect("checked producer runs once");
                if source_proof.set(proof).is_err() {
                    unreachable!("checked producer runs once");
                }
                destination
                    .set(homspace.clone())
                    .expect("checked producer runs once");
                Ok(homspace)
            })?;
        let identity = identity
            .into_inner()
            .expect("successful checked producer records identity");
        let prepared = PreparedCheckedGenericDynamicSpace::from_complete_parts(
            codomain_axes.len(),
            domain_axes.len(),
            destination
                .into_inner()
                .expect("successful checked producer records destination"),
            structure,
            identity.clone(),
        );
        let source_proof = source_proof
            .into_inner()
            .expect("successful checked producer records source proof");
        Ok(StagedTransform {
            preview: prepared.shared_structure(),
            nout: codomain_axes.len(),
            stage: (prepared, txn),
            proof: (identity, source_proof),
        })
    }

    fn transform_cache<'c>(
        _planning: &'c mut TreeTransformPlanning,
        stage: &'c mut Self::TransformStage,
    ) -> &'c mut CheckedContractTxn {
        &mut stage.1
    }

    fn contract_cache<'c>(
        _planning: &'c mut TreeTransformPlanning,
        txn: &'c mut CheckedContractTxn,
    ) -> &'c mut CheckedContractTxn {
        txn
    }

    /// Publication follows commit: only now are the destination's ids
    /// committed, and only a resident (canonical) destination is keyed.
    fn commit_transform(
        logical: &BoundDynamicFusionMapSpace<R>,
        (prepared, txn): Self::TransformStage,
        preview: Arc<BlockStructure>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error> {
        let destination = logical.commit_final_homspace_generic_bound_checked(prepared)?;
        txn.commit_transform((preview, Arc::clone(destination.space().structure())));
        Ok(destination)
    }

    /// Compares the admissions the spaces carry with the authority's.
    /// Why not `validate_rule`: it reads the provider's identity again, a
    /// provider event after the destination is staged (#2046); the pair
    /// admission already compared that identity once.
    fn validate_spaces(
        _rule: &R,
        authority: CheckedAuthority<'_, R>,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
    ) -> Result<(), Self::Error> {
        let held = authority.binding.space().admission().rule_identity();
        for space in [dst, lhs, rhs] {
            match (space.admission().rule_identity(), held) {
                (Some(expected), Some(actual)) if expected == actual => {}
                (Some(expected), Some(actual)) => {
                    return Err(CoreError::FusionRuleMismatch {
                        expected: expected.clone(),
                        actual: actual.clone(),
                    }
                    .into())
                }
                _ => return Err(CoreError::MissingFusionRuleIdentity.into()),
            }
        }
        Ok(())
    }

    /// The entry's answer (U15): a contraction entry read the braiding style
    /// once, before staging, and admits Bosonic only. Composition never
    /// asks; asked without an answer, it fails closed.
    fn contract_twist_possible(
        _rule: &R,
        authority: CheckedAuthority<'_, R>,
    ) -> Result<bool, Self::Error> {
        if authority.twist_free() {
            Ok(false)
        } else {
            Err(CHECKED_CONTRACTION_REQUIRES_BOSONIC.into())
        }
    }

    /// True by construction: the checked destination is derived from these
    /// operands by the same call. Why not a leg comparison: it reads duals
    /// through the provider after the destination is staged (#2046); a
    /// caller-supplied checked destination (#1870) needs a query-free one.
    fn destination_matches(
        _rule: &R,
        _lhs: OrientedFusionTreeHomSpace<'_>,
        _rhs: OrientedFusionTreeHomSpace<'_>,
        _lhs_contracting_axes: &[usize],
        _rhs_contracting_axes: &[usize],
        _output_axes: &[usize],
        _dst: &FusionTreeHomSpace,
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }

    /// The structure-only re-base of the stored tilings; checked operands
    /// are direct, so no logical-key projection is needed.
    fn irregular_core(
        _validated: ValidatedCoreContract<'_, R>,
        _authority: CheckedAuthority<'_, R>,
        dst: &DynamicFusionMapSpace,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
    ) -> Result<FusionBlockContractPlan<f64>, Self::Error> {
        if lhs.storage_conjugate() || rhs.storage_conjugate() {
            return Err(CHECKED_REQUIRES_DIRECT_OPERANDS.into());
        }
        let (lhs, rhs) = (lhs.storage_space(), rhs.storage_space());
        Ok(compile_checked_generic_core_plan_general(
            dst.structure(),
            dst.nout(),
            lhs.structure(),
            lhs.nout(),
            rhs.structure(),
            rhs.nout(),
        )?)
    }

    /// The Complete scorer over the planner's previews: checked spaces are
    /// always Complete, and the twist is the entry's answer, so no provider
    /// is read.
    fn dynamic_tree_plan(
        rule: &R,
        authority: CheckedAuthority<'_, R>,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<FusionContractPlan, Self::Error> {
        select_complete_tensorcontract_fusion_plan(
            |homspace, axes| {
                contract_axes_require_twist::<Self, R>(rule, authority, homspace, axes)
            },
            dst,
            lhs,
            rhs,
            axes,
        )
    }

    /// An identity permutation borrows the source and derives nothing
    /// (TensorKit `has_shared_permute`); any other stages its permuted
    /// HomSpace under the authority, then its transformer.
    fn transformed_source(
        txn: &mut CheckedContractTxn,
        authority: CheckedAuthority<'_, R>,
        rule: &R,
        source: &DynamicFusionMapSpace,
        source_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperation,
        source_conjugate: bool,
    ) -> Result<DynamicFusionTransformedSourceEntry<f64>, Self::Error> {
        if source_conjugate {
            return Err(CHECKED_REQUIRES_DIRECT_OPERANDS.into());
        }
        if operation.is_identity_for(source.nout(), source.nin()) {
            return Ok(DynamicFusionTransformedSourceEntry {
                space: Arc::new(source.clone()),
                replay_structure: Arc::clone(source_structure),
                transform_structure: None,
            });
        }
        let prepared = authority
            .binding
            .prepare_final_homspace_generic_from_checked(rule, || {
                source
                    .homspace()
                    .try_permute_generic_checked(
                        rule,
                        operation.codomain_permutation(),
                        operation.domain_permutation(),
                    )
                    .map_err(CheckedGenericPlanError::from)
            })?;
        let space = txn.stage(prepared);
        let transform_structure = Self::tree_structure(
            txn,
            rule,
            operation,
            space.structure(),
            TreeStructureSource::Stored {
                structure: source_structure,
                storage_conjugate: false,
            },
        )?;
        Ok(DynamicFusionTransformedSourceEntry {
            space: Arc::new(space),
            replay_structure: Arc::clone(source_structure),
            transform_structure: Some(transform_structure),
        })
    }

    fn core_destination(
        txn: &mut CheckedContractTxn,
        authority: CheckedAuthority<'_, R>,
        rule: &R,
        core_left: &DynamicFusionMapSpace,
        core_right: &DynamicFusionMapSpace,
        plan: &FusionContractPlan,
        output_dst: &DynamicFusionMapSpace,
    ) -> Result<DynamicFusionCoreDstEntry<f64>, Self::Error> {
        let core_axes = plan.core_axes().as_spec();
        // The plan's core output is the default order (its source transforms
        // place every open leg).
        let open_axes: smallvec::SmallVec<[usize; 16]> =
            (0..plan.core_dst_open_lhs_rank() + plan.core_dst_open_rhs_rank()).collect();
        let prepared = authority
            .binding
            .prepare_final_homspace_generic_from_checked(rule, || {
                FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
                    rule,
                    core_left.homspace(),
                    core_right.homspace(),
                    core_axes.lhs_contracting_axes(),
                    core_axes.rhs_contracting_axes(),
                    &open_axes,
                    plan.core_dst_open_lhs_rank(),
                )
                .map_err(CheckedGenericPlanError::from)
            })?;
        let space = txn.stage(prepared);
        let output_transform_structure = Self::tree_structure(
            txn,
            rule,
            plan.output_transform(),
            output_dst.structure(),
            TreeStructureSource::Stored {
                structure: space.structure(),
                storage_conjugate: false,
            },
        )?;
        Ok(DynamicFusionCoreDstEntry {
            space: Arc::new(space),
            output_transform_structure,
        })
    }

    fn copy_c_temporary(
        txn: &mut CheckedContractTxn,
        authority: CheckedAuthority<'_, R>,
        rule: &R,
        first: FusionOperand<'_>,
        second: FusionOperand<'_>,
        first_axes: &[usize],
        second_axes: &[usize],
        open_axes: &[usize],
        first_open: usize,
    ) -> Result<DynamicFusionMapSpace, Self::Error> {
        let prepared = authority
            .binding
            .prepare_final_homspace_generic_from_checked(rule, || {
                OrientedFusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
                    rule,
                    first.oriented_homspace(),
                    second.oriented_homspace(),
                    first_axes,
                    second_axes,
                    open_axes,
                    first_open,
                )
                .map_err(CheckedGenericPlanError::from)
            })?;
        Ok(txn.stage(prepared))
    }

    /// None: a twist is never possible once the entry admitted Bosonic
    /// braiding (U15).
    fn contract_twist_scales(
        rule: &R,
        authority: CheckedAuthority<'_, R>,
        _space: &DynamicFusionMapSpace,
        _core_right: &FusionTreeHomSpace,
        _space_is_core_left: bool,
        _rhs_contracting_axes: &[usize],
    ) -> Result<Vec<(usize, f64)>, Self::Error> {
        if Self::contract_twist_possible(rule, authority)? {
            return Err(CHECKED_CONTRACTION_REQUIRES_BOSONIC.into());
        }
        Ok(Vec::new())
    }

    /// The canonical coupled-region plan, else the structure-only re-base
    /// (the same builders as the core rung). Why no core-form check: the
    /// planner derived these operands in core form from one validated
    /// request.
    fn derived_core_plan(
        _rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        _core_axes: TensorContractSpec<'_>,
    ) -> Result<Arc<FusionBlockContractPlan<f64>>, Self::Error> {
        let plan =
            match FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_generic(
                dst.structure(),
                dst.nout(),
                lhs.structure(),
                lhs.nout(),
                rhs.structure(),
                rhs.nout(),
                MatrixOp::Identity,
                MatrixOp::Identity,
            )? {
                Some(plan) => plan,
                None => compile_checked_generic_core_plan_general(
                    dst.structure(),
                    dst.nout(),
                    lhs.structure(),
                    lhs.nout(),
                    rhs.structure(),
                    rhs.nout(),
                )?,
            };
        Ok(Arc::new(plan))
    }

    /// Lazy checked adjoints are #1865.
    fn prelowered_dynamic_tree_artifact(
        _txn: &mut CheckedContractTxn,
        _target: PlanTarget<'_, R, CheckedAuthority<'_, R>>,
        _lhs: FusionOperand<'_>,
        _rhs: FusionOperand<'_>,
        _axes: TensorContractSpec<'_>,
        _profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<DynamicTreeExecutionArtifact<f64>, Self::Error> {
        Err(CHECKED_REQUIRES_DIRECT_OPERANDS.into())
    }
}
