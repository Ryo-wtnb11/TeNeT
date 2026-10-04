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

use tenet_core::{
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols,
    SectorId,
};

use crate::{
    adjoint_bound_space_dyn, adjoint_bound_space_dyn_generic_checked, BoundDynamicFusionMapSpace,
    CheckedGenericPlanError, OperationError, TensorTraceAxisSpec, TensorTraceFusionStructure,
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
    /// The quantum dimension `dim(c)`.
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
        // Why not an exact dimension query: the checked provider exposes only
        // `sqrt(dim)` today (exact `dim(c)` is V7 / #1871).
        provider
            .try_sqrt_dim_scalar(sector)
            .map(|sqrt_dim| sqrt_dim * sqrt_dim)
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
