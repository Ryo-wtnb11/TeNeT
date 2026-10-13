use num_traits::One;
use tenet_core::{HostReadableStorage, HostWritableStorage, TensorMap};

use crate::{DenseBlockScalar, OperationError, RecouplingCoefficientAction};

use super::backend::TensorContractBackend;
use super::structure::TensorContractStructure;

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
