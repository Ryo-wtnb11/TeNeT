//! Shared fixtures, spy executors and imports for tenet-matrixalgebra's
//! test suite, split by responsibility into sibling files (#1596).

use tenet_core::{
    product_fusion_rule, BlockKey, BlockSpec, BlockStructure, BraidingStyleKind,
    CheckedGenericFusion, CheckedGenericRigidSymbols, CoreError, CoupledSectorFold,
    FermionParityFusionRule, FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace,
    FusionTreeHomSpace, FusionTreeKey, GenericFArray, GenericFusionSymbols, GenericRMatrix,
    GenericRigidSymbols, InfallibleGeneric, MultiplicityFreeFusionRule,
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
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseOwned, DenseRead, DenseTensor,
    DenseWrite,
};

macro_rules! bound_tensor_ref {
    ($provider:expr, $tensor:expr) => {
        bound_tensor($provider, $tensor).as_ref()
    };
}

mod fixtures;
mod generic_fixtures;
mod spies;
use fixtures::*;
use generic_fixtures::*;
use spies::*;

mod eigh_eig;
mod generic_dispatch;
mod matrix_functions;
mod null_space;
mod polar;
mod qr_lq;
mod svd;
