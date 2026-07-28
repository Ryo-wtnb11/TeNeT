//! External-crate import contract for the curated expert facades.
use tenet::matrixalgebra::{svd_compact, BoundTensorMap, BoundTensorMapRef, SectorSpectrum, SvdCompact};
use tenet::operations::{
    braid_into, permute_into, tensoradd_into, tensorcontract_fusion_into, tensorcontract_into,
    tensortrace_into, transpose_into, BoundDynamicFusionMapSpace, DynamicFusionMapSpace,
    OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TensorTraceAxisSpec, TreeTransformExecutionContext, TreeTransformOperation,
};

#[test]
fn retained_symbols_are_importable_externally() {
    let _: Option<fn()> = None;
    let _: Option<OperationError> = None;
    let _: Option<OutputAxisOrder> = None;
    let _: Option<TensorContractSpec> = None;
    let _: Option<TensorTraceAxisSpec> = None;
    let _: Option<SectorSpectrum> = None;
    // Importing the generic context and bound types is the firewall contract;
    // concrete providers are selected by callers.
}
