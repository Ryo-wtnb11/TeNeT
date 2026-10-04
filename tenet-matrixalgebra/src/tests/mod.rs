//! Shared fixtures, spy executors and imports for tenet-matrixalgebra's
//! test suite, split by responsibility into sibling files (#1596).

use tenet_core::{
    product_fusion_rule, BlockKey, BlockSpec, BlockStructure, BraidingStyleKind,
    CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, CoreError,
    CoupledSectorFold, FermionParityFusionRule, FusionProductSpace, FusionRule, FusionStyleKind,
    FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey, GenericFArray, GenericFusionSymbols,
    GenericRMatrix, GenericRigidSymbols, InfallibleGeneric, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, RuleIdentity, SU2FusionRule,
    SU2Irrep, SectorId, SectorLeg, SectorVec, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
    Z2FusionRule,
};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseTreeTransformOperations, DynamicFusionMapSpace,
    OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

use crate::factorize::{
    dyn_space_of, map_square_sectors_dyn_into, pinv_cutoff, typed_from_bound_factor,
    typed_from_dyn, validate_eigenvector_singular_values, validate_inverse_region_routes_for_test,
    BoundTensorMap,
};
use crate::test_numerics::numerics;
use crate::*;
use crate::{LeftPolar, Lq, Qr, RightPolar, Svd};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::{cell::Cell, convert::Infallible, fmt, sync::Arc};
use tenet_dense::{
    DenseBackend, DenseError, DenseExecutor, DenseOwned, DenseRead, DenseTensor, DenseWrite,
};

macro_rules! bound_tensor_ref {
    ($provider:expr, $tensor:expr) => {
        bound_tensor($provider, $tensor).as_ref()
    };
}

mod fixtures;
mod generic_fixtures;
pub(crate) mod scripted_executor;
mod spies;
use fixtures::*;
use generic_fixtures::*;
use spies::*;

// The checked values entries under their pre-#1862 names: the one entry per
// family in checked mode, on a source or on dense storage.
fn svd_vals_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(lease, source)
}

fn eig_vals_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    eig_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(lease, source)
}

fn qr_compact_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Qr<FactorOutput<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    qr_compact_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(lease, source)
}

fn left_null_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    left_null_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(lease, source)
}

fn dense_source<'a, R, D>(input: &'a BoundDynamicTensorRef<'_, R, D>) -> FactorSource<'a, R, D> {
    FactorSource::Dense(BoundDynamicTensorRef::try_new(input.space(), input.data()).unwrap())
}

fn svd_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_vals_checked_generic(dense, dense_source(input))
}

fn eigh_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    eigh_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(dense, dense_source(input))
}

fn eig_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    eig_vals_checked_generic(dense, dense_source(input))
}

mod eigh_eig;
mod generic_dispatch;
mod matrix_functions;
mod null_space;
mod polar;
mod qr_lq;
mod svd;
