use num_traits::One;
use std::sync::Arc;
use tenet_core::{BlockStructure, TensorMap};
use tenet_dense::{DenseExecutor, DenseView, DenseViewMut};

use crate::{
    tensoradd_raw_strided_kernel, DenseBlockScalar, DenseTreeTransformOperations, OperationError,
    RecouplingCoefficientAction,
};

use super::structure::TensorContractStructure;

pub trait TensorContractBackend<D, C = f64>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    type Workspace;

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
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;

    fn tensorcontract_structure_into_raw(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;
}

#[derive(Clone, Debug)]
pub struct TensorContractWorkspace<T> {
    output: Vec<T>,
    zero_strides: Vec<isize>,
}

impl<T> Default for TensorContractWorkspace<T> {
    fn default() -> Self {
        Self {
            output: Vec::new(),
            zero_strides: Vec::new(),
        }
    }
}

impl<T> TensorContractWorkspace<T> {
    #[inline]
    pub fn output_len(&self) -> usize {
        self.output.len()
    }
}

impl<E, D, C> TensorContractBackend<D, C> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    type Workspace = TensorContractWorkspace<D>;

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
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        tensorcontract_structure_with_dense_executor(
            self.dense_mut(),
            workspace,
            structure,
            dst,
            lhs,
            rhs,
            alpha,
            beta,
        )
    }

    fn tensorcontract_structure_into_raw(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        tensorcontract_structure_with_dense_executor_raw(
            self.dense_mut(),
            workspace,
            structure,
            dst_structure,
            lhs_structure,
            rhs_structure,
            dst_data,
            lhs_data,
            rhs_data,
            alpha,
            beta,
        )
    }
}

fn tensorcontract_structure_with_dense_executor<
    E,
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
>(
    dense: &mut E,
    workspace: &mut TensorContractWorkspace<D>,
    structure: &TensorContractStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    let dst_structure = Arc::clone(dst.structure());
    let lhs_structure = Arc::clone(lhs.structure());
    let rhs_structure = Arc::clone(rhs.structure());
    tensorcontract_structure_with_dense_executor_raw(
        dense,
        workspace,
        structure,
        &dst_structure,
        &lhs_structure,
        &rhs_structure,
        dst.data_mut(),
        lhs.data(),
        rhs.data(),
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_structure_with_dense_executor_raw<E, D, C>(
    dense: &mut E,
    workspace: &mut TensorContractWorkspace<D>,
    structure: &TensorContractStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    lhs_structure: &Arc<BlockStructure>,
    rhs_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    structure.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
    let descriptor = structure.descriptor();
    for term in descriptor.terms() {
        workspace.output.resize(term.workspace_len, D::zero());
        let lhs = D::dense_read(
            DenseView::new(
                lhs_data,
                descriptor.lhs_shape(term),
                descriptor.lhs_strides(term),
                term.lhs_offset,
            )
            .map_err(OperationError::Dense)?,
        );
        let rhs = D::dense_read(
            DenseView::new(
                rhs_data,
                descriptor.rhs_shape(term),
                descriptor.rhs_strides(term),
                term.rhs_offset,
            )
            .map_err(OperationError::Dense)?,
        );
        let output = D::dense_write(
            DenseViewMut::new(
                &mut workspace.output,
                descriptor.output_shape(term),
                descriptor.output_strides(term),
                0,
            )
            .map_err(OperationError::Dense)?,
        );
        dense
            .dot_general_into(output, lhs, rhs, descriptor.dot_config())
            .map_err(OperationError::Dense)?;

        let term_alpha = alpha.scale_by_coefficient(term.coefficient);
        let term_beta = if term.apply_beta { beta } else { D::one() };
        tensoradd_raw_strided_kernel(
            &mut workspace.zero_strides,
            dst_data,
            &workspace.output,
            descriptor.scatter_shape(term),
            descriptor.dst_strides(term),
            descriptor.workspace_strides(term),
            term.dst_offset,
            0,
            term_alpha,
            term_beta,
        )?;
    }
    Ok(())
}
