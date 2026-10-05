use num_traits::One;
use tenet_core::{HostReadableStorage, HostWritableStorage, TensorMap};

use crate::{
    DenseBlockScalar, DenseTreeTransformOperations, OperationError, RecouplingCoefficientAction,
};
use tenet_operations::{OutputAxisOrder, TensorContractSpec};

use super::backend::{TensorContractBackend, TensorContractWorkspace};
use super::structure::{tensorcontract_structure, TensorContractStructure};

pub fn tensorcontract_into<
    D,
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
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let mut backend = DenseTreeTransformOperations::default_executor();
    let mut workspace = TensorContractWorkspace::default();
    tensorcontract_into_with(
        &mut backend,
        &mut workspace,
        dst,
        lhs,
        rhs,
        axes,
        alpha,
        beta,
    )
}

pub fn tensorproduct_into<
    D,
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
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    output_permutation: OutputAxisOrder<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    tensorcontract_into(
        dst,
        lhs,
        rhs,
        TensorContractSpec::new(&[], &[], output_permutation),
        alpha,
        beta,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the public in-place tensor product exposes operands, conjugation, output order, and alpha/beta explicitly"
)]
pub fn tensorproduct_into_with_conjugation<
    D,
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
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    output_permutation: OutputAxisOrder<'_>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    tensorcontract_into(
        dst,
        lhs,
        rhs,
        TensorContractSpec::new_with_conjugation(
            &[],
            &[],
            output_permutation,
            lhs_conjugate,
            rhs_conjugate,
        ),
        alpha,
        beta,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the backend-explicit contraction keeps execution resources, operands, TensorContractSpec, and alpha/beta separate"
)]
pub fn tensorcontract_into_with<
    B,
    D,
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
    backend: &mut B,
    workspace: &mut B::Workspace,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let structure = tensorcontract_structure(dst, lhs, rhs, axes)?;
    tensorcontract_execute_with(backend, workspace, &structure, dst, lhs, rhs, alpha, beta)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the compiled execution boundary keeps backend, workspace, structure, operands, and alpha/beta explicit"
)]
pub fn tensorcontract_execute_with<
    B,
    D,
    C,
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
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TensorContractStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    structure.execute_with(backend, workspace, dst, lhs, rhs, alpha, beta)
}
