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
    CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols, FusionTreeKey,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, OrientedFusionTreeHomSpace,
    SectorId,
};
use tenet_operations::TreeTransformStructure;

use crate::contract::{rhs_contract_twist_factor_oriented, FusionOperandLayout};
use crate::tree_transform::{
    build_checked_generic_tree_pair_transform_group_plan, publishable, resolve,
    CompletedTransformerKey, OrientedBasisOrder, TransformerMode, TreeTransformPlanning,
    TreeTransformScope,
};
use crate::{
    adjoint_bound_space_dyn, adjoint_bound_space_dyn_generic_checked, BoundDynamicFusionMapSpace,
    CheckedGenericPlanError, DenseBlockScalar, OperationError, TensorTraceAxisSpec,
    TensorTraceFusionStructure, TreeTransformOperation, TreeTransformRuleCacheKey,
};

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
        crate::tensortrace::compile_fusion_dyn_generic_checked(dst, src, axes)
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
    /// The per-context planning state a structure resolution reads (the
    /// completed transformers themselves are process-global).
    type StructureCache;
    type Structure;

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
    type StructureCache = TreeTransformPlanning<R::Scalar>;
    type Structure = TreeTransformStructure<R::Scalar>;

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
}

impl<R> PlanningAlgebra<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
{
    type Scalar = f64;
    type Error = CheckedGenericPlanError<R::Error>;
    /// Checked Generic contraction reads no per-context planning state; it
    /// resolves through the process-global completed-transformer cache.
    type StructureCache = ();
    type Structure = TreeTransformStructure<f64>;

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
        _cache: &mut (),
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
        // Staged and core intermediates are uncommitted, hence not
        // canonical: their keys are lookup-only until #2014-3c commits them.
        let epoch = tenet_core::core_reset_epoch();
        let key = CompletedTransformerKey::new::<f64>(
            rule.rule_identity(),
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
        let may_publish = publishable([dst.as_ref(), structure.as_ref()]);
        resolve(key, may_publish, epoch, dst, structure, || {
            let plan = build_checked_generic_tree_pair_transform_group_plan(
                rule,
                operation.clone(),
                structure,
            )?;
            Ok(plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst),
                Arc::clone(structure),
                storage_conjugate,
            )?)
        })
    }
}
