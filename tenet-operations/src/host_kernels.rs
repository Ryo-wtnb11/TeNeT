use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{BlockStructure, BlockView, BlockViewMut, TensorMap};
use tenet_dense::{DenseExecutor, DenseView, DenseViewMut};

use crate::strided::{
    error as strided_error, offset_to_isize, read as strided_read, write as strided_write,
};
use crate::tensoradd::{TensorAddDescriptor, TensorAddDescriptorTerm};
use crate::{
    ConjugateValue, DenseRecouplingScalar, HostAllocator, OperationError,
    RecouplingCoefficientAction, TensorAddStructure, TreeTransformBlock, TreeTransformLayout,
    TreeTransformLayoutTable, TreeTransformReplayProfile, TreeTransformStructure,
};

#[derive(Clone, Debug)]
pub struct TreeTransformWorkspace<T> {
    zero_strides: Vec<isize>,
    source: Vec<T>,
    destination: Vec<T>,
    coefficients: Vec<T>,
    coefficient_cache_key: Option<CoefficientCacheKey>,
    coefficient_cache_refreshes: usize,
}

impl<T> Default for TreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            source: Vec::new(),
            destination: Vec::new(),
            coefficients: Vec::new(),
            coefficient_cache_key: None,
            coefficient_cache_refreshes: 0,
        }
    }
}

impl<T> TreeTransformWorkspace<T> {
    pub fn source_len(&self) -> usize {
        self.source.len()
    }

    pub fn destination_len(&self) -> usize {
        self.destination.len()
    }

    pub fn coefficient_len(&self) -> usize {
        self.coefficients.len()
    }

    pub fn coefficient_cache_refreshes(&self) -> usize {
        self.coefficient_cache_refreshes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CoefficientCacheKey {
    ptr: *const (),
    len: usize,
}

impl<T> TreeTransformWorkspace<T>
where
    T: Copy,
{
    fn prepare_coefficients_from<C>(&mut self, coefficients: &[C])
    where
        T: RecouplingCoefficientAction<C>,
        C: Copy,
    {
        let key = CoefficientCacheKey {
            ptr: coefficients.as_ptr().cast::<()>(),
            len: coefficients.len(),
        };
        if self.coefficient_cache_key == Some(key) {
            return;
        }
        self.coefficients.clear();
        self.coefficients
            .extend(coefficients.iter().copied().map(T::coefficient_as_data));
        self.coefficient_cache_key = Some(key);
        self.coefficient_cache_refreshes += 1;
    }
}

pub(crate) fn copy_block_with_strided_kernel<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;
    strided_kernel::copy_into(&mut dst, &src).map_err(strided_error)
}

pub(crate) fn tensoradd_structure_with_strided_kernel<T, const NOUT: usize, const NIN: usize, S>(
    allocator: &mut HostAllocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let descriptor = structure.descriptor();
    structure.validate_replay_structures(dst.structure(), src.structure())?;
    if dst.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: dst.structure().block_count(),
            src: descriptor.terms().len(),
        });
    }
    if src.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: descriptor.terms().len(),
            src: src.structure().block_count(),
        });
    }

    let zero_strides = &mut allocator.zero_strides;
    let dst_data = dst.data_mut();
    let src_data = src.data();
    for term in descriptor.terms() {
        tensoradd_prepared_block_with_strided_kernel(
            zero_strides,
            descriptor,
            term,
            dst_data,
            src_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

pub(crate) fn tree_transform_structure_with_strided_kernel<
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_with_strided_kernel_raw(
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

pub(crate) fn tree_transform_structure_with_strided_kernel_raw<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                &mut workspace.zero_strides,
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => tree_transform_multi_with_pack_gemm_scatter(
                workspace,
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.coefficients_src_by_dst,
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
        }
    }
    Ok(())
}

pub(crate) fn tree_transform_structure_with_dense_recoupling<
    E,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_with_dense_recoupling_raw(
        dense,
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

pub(crate) fn tree_transform_structure_with_dense_recoupling_raw<E, D, C>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                &mut workspace.zero_strides,
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => tree_transform_multi_with_dense_recoupling(
                dense,
                workspace,
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.coefficients_src_by_dst,
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tree_transform_structure_with_dense_recoupling_raw_profiled<E, D, C>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let total_start = std::time::Instant::now();

    let start = std::time::Instant::now();
    structure.validate_replay_structures(dst_structure, src_structure)?;
    profile.validate += start.elapsed();

    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => {
                profile.single_blocks += 1;
                let start = std::time::Instant::now();
                tree_transform_single_with_strided_kernel_profiled(
                    &mut workspace.zero_strides,
                    &structure.layouts,
                    structure.layouts.entry(dst_layout),
                    structure.layouts.entry(src_layout),
                    structure.coefficient(coefficient),
                    structure.storage_conjugate(),
                    dst_data,
                    src_data,
                    alpha,
                    beta,
                    profile,
                )?;
                profile.single_total += start.elapsed();
            }
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => {
                profile.multi_blocks += 1;
                tree_transform_multi_with_dense_recoupling_profiled(
                    dense,
                    workspace,
                    &structure.layouts,
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                    &structure.coefficients_src_by_dst,
                    structure.storage_conjugate(),
                    dst_data,
                    src_data,
                    alpha,
                    beta,
                    profile,
                )?;
            }
        }
    }

    profile.total += total_start.elapsed();
    Ok(())
}

pub(crate) fn tensoradd_block_with_strided_kernel<T>(
    allocator: &mut HostAllocator,
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;

    if dst.dims() != src.dims() {
        return Err(OperationError::ShapeMismatch {
            dst: dst.dims().to_vec(),
            src: src.dims().to_vec(),
        });
    }

    if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(&mut allocator.zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    }
}

fn tensoradd_prepared_block_with_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    descriptor: &TensorAddDescriptor,
    term: &TensorAddDescriptorTerm,
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    tensoradd_raw_strided_kernel(
        zero_strides,
        dst_data,
        src_data,
        descriptor.shape(term),
        descriptor.dst_strides(term),
        descriptor.src_strides(term),
        term.dst_offset,
        term.src_offset,
        descriptor.source_conjugate(),
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensoradd_raw_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    if source_conjugate {
        return tensoradd_raw_strided_conjugating_kernel(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        );
    }
    let mut dst = strided_kernel::StridedViewMut::new(dst_data, shape, dst_strides, dst_offset)
        .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(src_data, shape, src_strides, src_offset)
        .map_err(strided_error)?;

    if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    }
}

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_kernel_profiled<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    beta: T,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    if source_conjugate {
        let start = std::time::Instant::now();
        let result = tensoradd_raw_strided_conjugating_kernel(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        );
        profile.strided_kernel += start.elapsed();
        return result;
    }

    let start = std::time::Instant::now();
    let mut dst = strided_kernel::StridedViewMut::new(dst_data, shape, dst_strides, dst_offset)
        .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(src_data, shape, src_strides, src_offset)
        .map_err(strided_error)?;
    profile.strided_view_setup += start.elapsed();

    let start = std::time::Instant::now();
    let result = if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    };
    profile.strided_kernel += start.elapsed();
    result
}

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_conjugating_kernel<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
{
    let len = crate::strided::element_count(shape)?;
    if len == 0 {
        return Ok(());
    }
    for linear in 0..len {
        let dst_index = strided_offset(linear, shape, dst_strides, dst_offset)?;
        let src_index = strided_offset(linear, shape, src_strides, src_offset)?;
        let value = alpha * src_data[src_index].maybe_conj(true);
        dst_data[dst_index] = if beta.is_zero() {
            value
        } else {
            beta * dst_data[dst_index] + value
        };
    }
    zero_strides.clear();
    Ok(())
}

fn tree_transform_single_with_strided_kernel<D, C>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    dst_layout: &TreeTransformLayout,
    src_layout: &TreeTransformLayout,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let shape = layouts.shape(dst_layout);
    let scale = alpha.scale_by_coefficient(coefficient);
    tensoradd_raw_strided_kernel(
        zero_strides,
        dst_data,
        src_data,
        shape,
        layouts.strides(dst_layout),
        layouts.strides(src_layout),
        dst_layout.offset,
        src_layout.offset,
        source_conjugate,
        scale,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_single_with_strided_kernel_profiled<D, C>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    dst_layout: &TreeTransformLayout,
    src_layout: &TreeTransformLayout,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let shape = layouts.shape(dst_layout);
    let scale = alpha.scale_by_coefficient(coefficient);
    tensoradd_raw_strided_kernel_profiled(
        zero_strides,
        dst_data,
        src_data,
        shape,
        layouts.strides(dst_layout),
        layouts.strides(src_layout),
        dst_layout.offset,
        src_layout.offset,
        source_conjugate,
        scale,
        beta,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_pack_gemm_scatter<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
            source_conjugate,
        )?;
    }

    apply_recoupling_matrix_src_times_u_transpose(
        &mut workspace.destination,
        &workspace.source,
        coefficients_src_by_dst,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
    )?;

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            &workspace.destination,
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_dense_recoupling<E, D, C>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
            source_conjugate,
        )?;
    }

    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }
    workspace.prepare_coefficients_from(coefficients_src_by_dst);

    apply_recoupling_matrix_with_dense_executor(
        dense,
        &mut workspace.destination,
        &workspace.source,
        &workspace.coefficients,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
    )?;

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            &workspace.destination,
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_dense_recoupling_profiled<E, D, C>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    let start = std::time::Instant::now();
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());
    profile.multi_workspace_prepare += start.elapsed();

    let start = std::time::Instant::now();
    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column_profiled(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
            source_conjugate,
            profile,
        )?;
        profile.packed_columns += 1;
    }
    profile.multi_pack += start.elapsed();

    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }

    let start = std::time::Instant::now();
    workspace.prepare_coefficients_from(coefficients_src_by_dst);
    profile.multi_coefficient_prepare += start.elapsed();

    apply_recoupling_matrix_with_dense_executor_profiled(
        dense,
        &mut workspace.destination,
        &workspace.source,
        &workspace.coefficients,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
        profile,
    )?;

    let start = std::time::Instant::now();
    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout_profiled(
            &mut workspace.zero_strides,
            layouts,
            layout,
            &workspace.destination,
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
            profile,
        )?;
        profile.scattered_columns += 1;
    }
    profile.multi_scatter += start.elapsed();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_recoupling_matrix_src_times_u_transpose<D, C>(
    destination: &mut [D],
    source: &[D],
    coefficients_src_by_dst: &[C],
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError>
where
    D: Copy + Add<D, Output = D> + Zero + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source.len() != source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: source_len,
            actual: source.len(),
        });
    }
    if destination.len() != destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: destination_len,
            actual: destination.len(),
        });
    }
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }

    // TensorKit's dense-vector GenericTreeTransformer uses `U[dst, src]` and
    // computes `buffer_dst = buffer_src * transpose(U)` after packing source
    // trees as columns. Keep this as the backend-replaceable boundary.
    for dst_index in 0..dst_count {
        let dst_column_start = dst_index * element_count;
        let coefficient_row_start = coefficient_start + dst_index * src_count;
        for element in 0..element_count {
            let mut sum = D::zero();
            for src_index in 0..src_count {
                let coeff = coefficients_src_by_dst[coefficient_row_start + src_index];
                let src_value = source[element + src_index * element_count];
                sum = sum + src_value.scale_by_coefficient(coeff);
            }
            destination[dst_column_start + element] = sum;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_recoupling_matrix_with_dense_executor<E, T>(
    dense: &mut E,
    destination: &mut [T],
    source: &[T],
    coefficients_src_by_dst: &[T],
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    T: DenseRecouplingScalar,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source.len() != source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: source_len,
            actual: source.len(),
        });
    }
    if destination.len() != destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: destination_len,
            actual: destination.len(),
        });
    }
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }

    let source_shape = [element_count, src_count];
    let source_strides = [1, element_count];
    let coefficient_shape = [src_count, dst_count];
    let coefficient_strides = [1, src_count];
    let destination_shape = [element_count, dst_count];
    let destination_strides = [1, element_count];

    let lhs = T::dense_read(
        DenseView::new(source, &source_shape, &source_strides, 0).map_err(OperationError::Dense)?,
    );
    let rhs = T::dense_read(
        DenseView::new(
            coefficients_src_by_dst,
            &coefficient_shape,
            &coefficient_strides,
            coefficient_start,
        )
        .map_err(OperationError::Dense)?,
    );
    let output = T::dense_write(
        DenseViewMut::new(destination, &destination_shape, &destination_strides, 0)
            .map_err(OperationError::Dense)?,
    );
    dense
        .matmul_into(output, lhs, rhs)
        .map_err(OperationError::Dense)
}

#[allow(clippy::too_many_arguments)]
fn apply_recoupling_matrix_with_dense_executor_profiled<E, T>(
    dense: &mut E,
    destination: &mut [T],
    source: &[T],
    coefficients_src_by_dst: &[T],
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    T: DenseRecouplingScalar,
{
    let total_start = std::time::Instant::now();

    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source.len() != source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: source_len,
            actual: source.len(),
        });
    }
    if destination.len() != destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: destination_len,
            actual: destination.len(),
        });
    }
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }

    let source_shape = [element_count, src_count];
    let source_strides = [1, element_count];
    let coefficient_shape = [src_count, dst_count];
    let coefficient_strides = [1, src_count];
    let destination_shape = [element_count, dst_count];
    let destination_strides = [1, element_count];

    let start = std::time::Instant::now();
    let lhs = T::dense_read(
        DenseView::new(source, &source_shape, &source_strides, 0).map_err(OperationError::Dense)?,
    );
    let rhs = T::dense_read(
        DenseView::new(
            coefficients_src_by_dst,
            &coefficient_shape,
            &coefficient_strides,
            coefficient_start,
        )
        .map_err(OperationError::Dense)?,
    );
    let output = T::dense_write(
        DenseViewMut::new(destination, &destination_shape, &destination_strides, 0)
            .map_err(OperationError::Dense)?,
    );
    profile.multi_dense_view_setup += start.elapsed();

    let start = std::time::Instant::now();
    let result = dense
        .matmul_into(output, lhs, rhs)
        .map_err(OperationError::Dense);
    profile.multi_dense_matmul_call += start.elapsed();
    profile.multi_matmul_total += total_start.elapsed();
    result
}

fn pack_layout_into_column<T>(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
) -> Result<(), OperationError>
where
    T: Copy + ConjugateValue + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    if source_conjugate {
        let packed_offset = offset_to_isize(packed_offset)?;
        let len = crate::strided::element_count(shape)?;
        let packed_strides = layouts.packed_strides(layout);
        let src_strides = layouts.strides(layout);
        for linear in 0..len {
            let dst_index = strided_offset(linear, shape, packed_strides, packed_offset)?;
            let src_index = strided_offset(linear, shape, src_strides, layout.offset)?;
            packed[dst_index] = src_data[src_index].maybe_conj(true);
        }
        return Ok(());
    }

    let mut dst = strided_kernel::StridedViewMut::new(
        packed,
        shape,
        layouts.packed_strides(layout),
        offset_to_isize(packed_offset)?,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(
        src_data,
        shape,
        layouts.strides(layout),
        layout.offset,
    )
    .map_err(strided_error)?;
    strided_kernel::copy_into(&mut dst, &src).map_err(strided_error)
}

fn pack_layout_into_column_profiled<T>(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    T: Copy + ConjugateValue + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    if source_conjugate {
        let start = std::time::Instant::now();
        let packed_offset = offset_to_isize(packed_offset)?;
        let len = crate::strided::element_count(shape)?;
        let packed_strides = layouts.packed_strides(layout);
        let src_strides = layouts.strides(layout);
        for linear in 0..len {
            let dst_index = strided_offset(linear, shape, packed_strides, packed_offset)?;
            let src_index = strided_offset(linear, shape, src_strides, layout.offset)?;
            packed[dst_index] = src_data[src_index].maybe_conj(true);
        }
        profile.strided_kernel += start.elapsed();
        return Ok(());
    }

    let start = std::time::Instant::now();
    let mut dst = strided_kernel::StridedViewMut::new(
        packed,
        shape,
        layouts.packed_strides(layout),
        offset_to_isize(packed_offset)?,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(
        src_data,
        shape,
        layouts.strides(layout),
        layout.offset,
    )
    .map_err(strided_error)?;
    profile.strided_view_setup += start.elapsed();

    let start = std::time::Instant::now();
    let result = strided_kernel::copy_into(&mut dst, &src).map_err(strided_error);
    profile.strided_kernel += start.elapsed();
    result
}

fn scatter_column_into_layout<T>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    let mut dst = strided_kernel::StridedViewMut::new(
        dst_data,
        shape,
        layouts.strides(layout),
        layout.offset,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(
        packed,
        shape,
        layouts.packed_strides(layout),
        offset_to_isize(packed_offset)?,
    )
    .map_err(strided_error)?;

    if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_column_into_layout_profiled<T>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    beta: T,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    let start = std::time::Instant::now();
    let mut dst = strided_kernel::StridedViewMut::new(
        dst_data,
        shape,
        layouts.strides(layout),
        layout.offset,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(
        packed,
        shape,
        layouts.packed_strides(layout),
        offset_to_isize(packed_offset)?,
    )
    .map_err(strided_error)?;
    profile.strided_view_setup += start.elapsed();

    let start = std::time::Instant::now();
    let result = if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    };
    profile.strided_kernel += start.elapsed();
    result
}

fn scale_destination<T>(
    zero_strides: &mut Vec<isize>,
    dst: &mut strided_kernel::StridedViewMut<'_, T>,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T> + strided_kernel::MaybeSendSync,
{
    let scalar = [beta];
    zero_strides.clear();
    zero_strides.resize(dst.ndim(), 0);
    let beta_view =
        strided_kernel::StridedView::<T>::new(&scalar, dst.dims(), zero_strides.as_slice(), 0)
            .map_err(strided_error)?;
    strided_kernel::mul(dst, &beta_view).map_err(strided_error)
}

fn strided_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[isize],
    base: isize,
) -> Result<usize, OperationError> {
    let mut offset = base;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let coord = if dim == 0 { 0 } else { linear % dim };
        if dim != 0 {
            linear /= dim;
        }
        let coord = isize::try_from(coord).map_err(|_| OperationError::ElementCountOverflow)?;
        offset = offset
            .checked_add(
                coord
                    .checked_mul(stride)
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    usize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })
}
