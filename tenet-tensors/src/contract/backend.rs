use num_traits::One;
use std::sync::Arc;
use tenet_core::{
    BlockStructure, HostReadableStorage, HostWritableStorage, Placement, ScratchStorage,
    SimilarStorage, TensorMap,
};
use tenet_dense::{DenseExecutor, DenseView, DenseViewMut};

use crate::host_scratch::HostScratchBuffer;
use crate::storage_scratch::StorageTensorContractWorkspace;
use crate::{
    tensoradd_raw_strided_kernel, ConjugateValue, DenseBlockScalar, DenseTreeTransformOperations,
    OperationError, RecouplingCoefficientAction, ReportsPlacement,
};

use super::structure::{
    TensorContractDenseRouteOrder, TensorContractDescriptor, TensorContractDescriptorTerm,
    TensorContractStructure,
};

/// Legacy/current tensor-contraction execution contract over host-accessible data.
///
/// The raw replay methods take host slices. New code that specifically depends
/// on this host-slice contract may use `HostTensorContractBackend`; future
/// placement-aware/device/MPI contraction traits should not inherit from this
/// raw-slice API.
pub trait TensorContractBackend<D, C = f64>
where
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
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
        DDst,
        DLhs,
        DRhs,
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>;

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

    /// Writes the full rank-2 matrix product `lhs * rhs` into `dst_data`.
    ///
    /// Implementations must overwrite every element of the `rows x cols`
    /// output buffer; callers may reuse dirty workspace and must not rely on
    /// pre-cleared destination storage.
    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_into_raw(
        &mut self,
        workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError>;
}

/// Explicit marker for the legacy host-slice tensor-contract backend family.
///
/// `TensorContractBackend` keeps the existing method-bearing public trait for
/// source compatibility. This marker means “implements the host-slice replay
/// contract,” not necessarily “physically CPU-native.” Future device/MPI
/// contraction backends should use separate placement-aware execution traits.
pub trait HostTensorContractBackend<D, C = f64>: TensorContractBackend<D, C>
where
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
}

impl<B, D, C> HostTensorContractBackend<D, C> for B
where
    B: TensorContractBackend<D, C> + ?Sized,
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
}

/// Host scratch/replay workspace backed by `Vec<T>`.
///
/// Raw contraction replay using this workspace operates on host slices. Device
/// execution should use a separate device workspace instead of hiding device
/// storage behind this type.
#[derive(Clone, Debug)]
pub struct HostTensorContractWorkspace<T> {
    output: HostScratchBuffer<T>,
    zero_strides: Vec<isize>,
}

pub type TensorContractWorkspace<T> = HostTensorContractWorkspace<T>;

impl<T> Default for HostTensorContractWorkspace<T> {
    fn default() -> Self {
        Self {
            output: HostScratchBuffer::default(),
            zero_strides: Vec::new(),
        }
    }
}

impl<T> HostTensorContractWorkspace<T> {
    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    #[inline]
    pub fn is_host_workspace(&self) -> bool {
        self.placement() == Placement::Host
    }

    #[inline]
    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    fn prepare_output(&mut self, len: usize, zero: T)
    where
        T: Clone,
    {
        self.output.resize_filled(len, zero);
    }
}

impl<T> ReportsPlacement for HostTensorContractWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<E, D, C> TensorContractBackend<D, C> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
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
        DDst,
        DLhs,
        DRhs,
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
    {
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

    fn matmul_rank2_into_raw(
        &mut self,
        _workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        let lhs_shape = [rows, contracted];
        let lhs_strides = [1, rows];
        let rhs_shape = [contracted, cols];
        let rhs_strides = [1, contracted];
        let dst_shape = [rows, cols];
        let dst_strides = [1, rows];
        let lhs = D::dense_read(
            DenseView::new(lhs_data, &lhs_shape, &lhs_strides, 0).map_err(OperationError::Dense)?,
        );
        let rhs = D::dense_read(
            DenseView::new(rhs_data, &rhs_shape, &rhs_strides, 0).map_err(OperationError::Dense)?,
        );
        let output = D::dense_write(
            DenseViewMut::new(dst_data, &dst_shape, &dst_strides, 0)
                .map_err(OperationError::Dense)?,
        );
        self.dense_mut()
            .matmul_into(output, lhs, rhs)
            .map_err(OperationError::Dense)
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
    DDst,
    DLhs,
    DRhs,
>(
    dense: &mut E,
    workspace: &mut TensorContractWorkspace<D>,
    structure: &TensorContractStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
    C: Copy + One,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
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

/// Replays a prepared tensor contraction structure on host slices.
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
        workspace.prepare_output(term.workspace_len, D::zero());
        tensorcontract_descriptor_term_with_output_scratch(
            dense,
            &mut workspace.zero_strides,
            descriptor,
            term,
            dst_data,
            workspace.output.as_mut_slice(),
            lhs_data,
            rhs_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_structure_with_storage_workspace_dense_executor<
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
    DDst,
    DLhs,
    DRhs,
>(
    dense: &mut E,
    workspace: &mut StorageTensorContractWorkspace<DDst::Similar>,
    structure: &TensorContractStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
    DDst: HostWritableStorage<D> + SimilarStorage<D>,
    DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let lhs_structure = Arc::clone(lhs.structure());
    let rhs_structure = Arc::clone(rhs.structure());
    structure.validate_replay_structures(&dst_structure, &lhs_structure, &rhs_structure)?;
    let descriptor = structure.descriptor();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    for term in descriptor.terms() {
        workspace.prepare_from_dst_storage(dst.storage(), term.workspace_len, D::zero());
        let (zero_strides, output_scratch) = workspace.replay_parts_mut();
        tensorcontract_descriptor_term_with_output_scratch(
            dense,
            zero_strides,
            descriptor,
            term,
            dst.data_mut(),
            output_scratch.as_mut_slice(),
            lhs_data,
            rhs_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tensorcontract_descriptor_term_with_output_scratch<E, D, C>(
    dense: &mut E,
    zero_strides: &mut Vec<isize>,
    descriptor: &TensorContractDescriptor<C>,
    term: &TensorContractDescriptorTerm<C>,
    dst_data: &mut [D],
    output_scratch: &mut [D],
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
    if descriptor.lhs_conjugate() || descriptor.rhs_conjugate() {
        tensorcontract_conjugating_dot_into_workspace(
            descriptor,
            term,
            output_scratch,
            lhs_data,
            rhs_data,
        )?;
    } else {
        let logical_lhs = DenseView::new(
            lhs_data,
            descriptor.lhs_shape(term),
            descriptor.lhs_strides(term),
            term.lhs_offset,
        )
        .map_err(OperationError::Dense)?;
        let logical_rhs = DenseView::new(
            rhs_data,
            descriptor.rhs_shape(term),
            descriptor.rhs_strides(term),
            term.rhs_offset,
        )
        .map_err(OperationError::Dense)?;
        let (lhs, rhs) = match descriptor.dense_route_order() {
            TensorContractDenseRouteOrder::LhsRhs => {
                (D::dense_read(logical_lhs), D::dense_read(logical_rhs))
            }
            TensorContractDenseRouteOrder::RhsLhs => {
                (D::dense_read(logical_rhs), D::dense_read(logical_lhs))
            }
        };
        let output = D::dense_write(
            DenseViewMut::new(
                output_scratch,
                descriptor.output_shape(term),
                descriptor.output_strides(term),
                0,
            )
            .map_err(OperationError::Dense)?,
        );
        dense
            .dot_general_into(output, lhs, rhs, descriptor.dot_config())
            .map_err(OperationError::Dense)?;
    }

    let term_alpha = alpha.scale_by_coefficient(term.coefficient);
    let term_beta = if term.apply_beta { beta } else { D::one() };
    tensoradd_raw_strided_kernel(
        zero_strides,
        dst_data,
        output_scratch,
        descriptor.scatter_shape(term),
        descriptor.dst_strides(term),
        descriptor.workspace_strides(term),
        term.dst_offset,
        0,
        false,
        term_alpha,
        term_beta,
    )?;
    Ok(())
}

fn tensorcontract_conjugating_dot_into_workspace<D, C>(
    descriptor: &super::structure::TensorContractDescriptor<C>,
    term: &super::structure::TensorContractDescriptorTerm<C>,
    output: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + ConjugateValue,
    C: Copy + One,
{
    let output_shape = descriptor.output_shape(term);
    let contract_shape = descriptor
        .lhs_contracting_axes()
        .iter()
        .map(|&axis| descriptor.lhs_shape(term)[axis])
        .collect::<Vec<_>>();
    let output_len = crate::strided::element_count(output_shape)?;
    let contract_len = crate::strided::element_count(&contract_shape)?;
    for value in output.iter_mut() {
        *value = D::zero();
    }
    for output_linear in 0..output_len {
        let mut sum = D::zero();
        for contract_linear in 0..contract_len {
            let lhs_index = contract_lhs_offset(descriptor, term, output_linear, contract_linear)?;
            let rhs_index = contract_rhs_offset(descriptor, term, output_linear, contract_linear)?;
            sum = sum
                + lhs_data[lhs_index].maybe_conj(descriptor.lhs_conjugate())
                    * rhs_data[rhs_index].maybe_conj(descriptor.rhs_conjugate());
        }
        let dst_index = linear_strided_offset(
            output_linear,
            output_shape,
            descriptor.output_strides(term),
            0,
        )?;
        output[dst_index] = sum;
    }
    Ok(())
}

fn contract_lhs_offset<C>(
    descriptor: &super::structure::TensorContractDescriptor<C>,
    term: &super::structure::TensorContractDescriptorTerm<C>,
    output_linear: usize,
    contract_linear: usize,
) -> Result<usize, OperationError>
where
    C: Copy + One,
{
    let lhs_shape = descriptor.lhs_shape(term);
    let lhs_strides = descriptor.lhs_strides(term);
    let mut offset = term.lhs_offset;
    let mut output_coords = unravel_index(output_linear, descriptor.output_shape(term))?;
    let contract_coords = unravel_index_for_axes(
        contract_linear,
        lhs_shape,
        descriptor.lhs_contracting_axes(),
    )?;
    for &axis in descriptor.lhs_open_axes() {
        let coord = output_coords.remove(0);
        offset = offset
            .checked_add(
                coord
                    .checked_mul(lhs_strides[axis])
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    for (&axis, coord) in descriptor
        .lhs_contracting_axes()
        .iter()
        .zip(contract_coords.into_iter())
    {
        offset = offset
            .checked_add(
                coord
                    .checked_mul(lhs_strides[axis])
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(offset)
}

fn contract_rhs_offset<C>(
    descriptor: &super::structure::TensorContractDescriptor<C>,
    term: &super::structure::TensorContractDescriptorTerm<C>,
    output_linear: usize,
    contract_linear: usize,
) -> Result<usize, OperationError>
where
    C: Copy + One,
{
    let rhs_shape = descriptor.rhs_shape(term);
    let rhs_strides = descriptor.rhs_strides(term);
    let mut offset = term.rhs_offset;
    let output_coords = unravel_index(output_linear, descriptor.output_shape(term))?;
    let rhs_output_start = descriptor.lhs_open_axes().len();
    let contract_coords = unravel_index_for_axes(
        contract_linear,
        rhs_shape,
        descriptor.rhs_contracting_axes(),
    )?;
    for (i, &axis) in descriptor.rhs_open_axes().iter().enumerate() {
        let coord = output_coords[rhs_output_start + i];
        offset = offset
            .checked_add(
                coord
                    .checked_mul(rhs_strides[axis])
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    for (&axis, coord) in descriptor
        .rhs_contracting_axes()
        .iter()
        .zip(contract_coords.into_iter())
    {
        offset = offset
            .checked_add(
                coord
                    .checked_mul(rhs_strides[axis])
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(offset)
}

fn unravel_index(mut linear: usize, shape: &[usize]) -> Result<Vec<usize>, OperationError> {
    let mut coords = Vec::with_capacity(shape.len());
    for &dim in shape {
        let coord = if dim == 0 { 0 } else { linear % dim };
        if dim != 0 {
            linear /= dim;
        }
        coords.push(coord);
    }
    Ok(coords)
}

fn unravel_index_for_axes(
    mut linear: usize,
    shape: &[usize],
    axes: &[usize],
) -> Result<Vec<usize>, OperationError> {
    let mut coords = Vec::with_capacity(axes.len());
    for &axis in axes {
        let dim = shape[axis];
        let coord = if dim == 0 { 0 } else { linear % dim };
        if dim != 0 {
            linear /= dim;
        }
        coords.push(coord);
    }
    Ok(coords)
}

fn linear_strided_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[usize],
    base: usize,
) -> Result<usize, OperationError> {
    let mut offset = base;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let coord = if dim == 0 { 0 } else { linear % dim };
        if dim != 0 {
            linear /= dim;
        }
        offset = offset
            .checked_add(
                coord
                    .checked_mul(stride)
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tensor_contract_backend<B, D, C>()
    where
        B: TensorContractBackend<D, C>,
        D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
        C: Copy + One,
    {
    }

    fn assert_host_tensor_contract_backend<B, D, C>()
    where
        B: HostTensorContractBackend<D, C>,
        D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
        C: Copy + One,
    {
    }

    #[test]
    fn dense_tree_transform_operations_keeps_contract_backend_names() {
        assert_tensor_contract_backend::<DenseTreeTransformOperations, f64, f64>();
        assert_host_tensor_contract_backend::<DenseTreeTransformOperations, f64, f64>();
    }
}
