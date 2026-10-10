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
    BlockStructure, BraidingStyleKind, CheckedFusionAlgebra, CheckedGenericAdmissionMode,
    CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols, CoreError,
    FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols,
    OrientedFusionTreeHomSpace, SectorId,
};
use tenet_operations::TreeTransformStructure;

use crate::contract::{
    compile_checked_generic_core_plan_general,
    compile_fusion_block_contract_plan_prelowered_validated, core_homspace_matches,
    rhs_contract_twist_factor_oriented, validate_fusion_contract_rule, DynamicFusionMapSpace,
    FusionOperand, FusionOperandLayout, LayoutKeyBuilder, ValidatedCoreContract,
};
use crate::tree_transform::{
    build_checked_generic_tree_pair_transform_group_plan_validated, lookup_bound,
    validate_checked_generic_tree_pair_plan_preflight, CheckedPendingCoefficients,
    CoefficientGroupReuse, CompletedTransformerKey, OrientedBasisOrder, TransformerMode,
    TreeTransformPlanning, TreeTransformScope,
};
use crate::{
    adjoint_bound_space_dyn, adjoint_bound_space_dyn_generic_checked, BoundDynamicFusionMapSpace,
    CheckedGenericPlanError, DenseBlockScalar, OperationError, TensorTraceAxisSpec,
    TensorTraceFusionStructure, TreeTransformOperation, TreeTransformRuleCacheKey,
};
use tenet_operations::fusion_replay::FusionBlockContractPlan;

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
    type Structure;
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
    ) -> Result<Self::Structure, Self::Error>;

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
}

impl<R> PlanningAlgebra<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    // Why only the tree-structure cache's bounds here: the eager transform
    // entries share this impl without `DenseBlockScalar`, which only
    // `core_alpha` needs.
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync + 'static,
{
    type Scalar = R::Scalar;
    type Error = OperationError;
    type StructureCache = TreeTransformPlanning;
    type Structure = TreeTransformStructure<R::Scalar>;
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

    fn tree_structure(
        cache: &mut Self::StructureCache,
        rule: &R,
        operation: &TreeTransformOperation,
        dst: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
    ) -> Result<Self::Structure, OperationError> {
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
        }
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
}

impl<R> PlanningAlgebra<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
{
    type Scalar = f64;
    type Error = CheckedGenericPlanError<R::Error>;
    /// The contraction call's composed coefficients and completed
    /// transformers, flushed only after its commits.
    type StructureCache = CheckedPendingCoefficients;
    type Structure = TreeTransformStructure<f64>;
    /// The binding whose provider admission stages derived spaces and
    /// commits them after the call succeeds (#2063).
    type SpaceAuthority<'a>
        = &'a BoundDynamicFusionMapSpace<R>
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
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "checked Generic contraction requires Bosonic braiding",
            }
            .into());
        }
        Ok(Some(f64::one()))
    }

    fn tree_structure(
        coefficients: &mut CheckedPendingCoefficients,
        rule: &R,
        operation: &TreeTransformOperation,
        dst: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
    ) -> Result<Self::Structure, Self::Error> {
        let TreeStructureSource::Stored {
            structure,
            storage_conjugate,
        } = src
        else {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "checked Generic tree structure requires a stored source",
            }
            .into());
        };
        // A warm call's intermediates are committed, so their canonical ids
        // hit; a miss builds and stages its publication until the call's
        // commits (#2063).
        let identity = rule.rule_identity();
        let key = CompletedTransformerKey::new::<f64>(
            identity.clone(),
            TransformerMode::CheckedGeneric,
            TreeTransformScope::TreePair,
            operation,
            tenet_core::FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            storage_conjugate,
            None,
            dst,
            structure,
        );
        if let Some(hit) = lookup_bound(&key, dst, structure) {
            return Ok(hit);
        }
        // Admission precedes any composed-coefficient lookup. Why the
        // cold call's uncommitted intermediates reuse coefficients: their keys
        // hold sectors and trees, never content ids.
        let source_proof =
            validate_checked_generic_tree_pair_plan_preflight(rule, operation, structure)?;
        let reuse = CoefficientGroupReuse::<f64>::new(
            identity,
            TransformerMode::CheckedGeneric,
            TreeTransformScope::TreePair,
            operation,
            tenet_core::FusionTreePairOrientation::Direct,
        );
        let plan = build_checked_generic_tree_pair_transform_group_plan_validated(
            operation.clone(),
            &source_proof,
            &reuse,
        )?;
        coefficients.stage(reuse.into_pending());
        let built = plan.compile_shared_structures_with_storage_conjugation(
            Arc::clone(dst),
            Arc::clone(structure),
            storage_conjugate,
        )?;
        coefficients.stage_transformer(key, &built);
        Ok(built)
    }

    /// Compares the admissions the spaces carry with the authority's.
    /// Why not `validate_rule`: it reads the provider's identity again, a
    /// provider event after the destination is staged (#2046); the pair
    /// admission already compared that identity once.
    fn validate_spaces(
        _rule: &R,
        authority: &BoundDynamicFusionMapSpace<R>,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
    ) -> Result<(), Self::Error> {
        let held = authority.space().admission().rule_identity();
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

    /// Why an error rather than reading the braiding style here: a provider
    /// read inside the planner would follow the destination's staging
    /// (#2046). Checked composition, the one checked caller of the shared
    /// core rung, never asks; checked contraction will carry its entry's U15
    /// answer (#1860 leaf B). Until then a twisted question fails closed.
    fn contract_twist_possible(
        _rule: &R,
        _authority: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<bool, Self::Error> {
        Err(OperationError::UnsupportedTensorContractScope {
            message: "checked Generic contraction requires Bosonic braiding",
        }
        .into())
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
        _authority: &BoundDynamicFusionMapSpace<R>,
        dst: &DynamicFusionMapSpace,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
    ) -> Result<FusionBlockContractPlan<f64>, Self::Error> {
        if lhs.storage_conjugate() || rhs.storage_conjugate() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "checked Generic contraction currently requires eager direct operands",
            }
            .into());
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
}
