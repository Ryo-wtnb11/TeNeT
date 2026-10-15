use tenet_core::{BlockKey, BlockSpec, BlockStructure, FusionTreePairKey};

use crate::OperationError;
use tenet_operations::{
    permutation_axes_inline, AxisVec, OutputAxisOrder, TensorContractSpec, TensorTraceAxisSpec,
};

#[cfg(test)]
thread_local! {
    static ADJOINT_VIEW_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_adjoint_view_build_count() {
    ADJOINT_VIEW_BUILDS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn adjoint_view_build_count() -> usize {
    ADJOINT_VIEW_BUILDS.with(std::cell::Cell::get)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweredTensorTraceAxisSpec {
    output_axes: Vec<usize>,
    trace_lhs_axes: Vec<usize>,
    trace_rhs_axes: Vec<usize>,
    storage_conjugate: bool,
}

impl LoweredTensorTraceAxisSpec {
    #[inline]
    pub(crate) fn as_spec(&self) -> TensorTraceAxisSpec<'_> {
        TensorTraceAxisSpec::new_with_conjugation(
            &self.output_axes,
            &self.trace_lhs_axes,
            &self.trace_rhs_axes,
            self.storage_conjugate,
        )
    }
}

pub(crate) fn lower_tensortrace_source_adjoint_axes_dyn(
    src_nout: usize,
    src_nin: usize,
    axes: TensorTraceAxisSpec<'_>,
) -> Result<LoweredTensorTraceAxisSpec, OperationError> {
    if axes.source_conjugate() {
        Ok(LoweredTensorTraceAxisSpec {
            output_axes: adjoint_tensor_axes(src_nout, src_nin, axes.output_axes())?,
            trace_lhs_axes: adjoint_tensor_axes(src_nout, src_nin, axes.trace_lhs_axes())?,
            trace_rhs_axes: adjoint_tensor_axes(src_nout, src_nin, axes.trace_rhs_axes())?,
            storage_conjugate: true,
        })
    } else {
        Ok(LoweredTensorTraceAxisSpec {
            output_axes: axes.output_axes().to_vec(),
            trace_lhs_axes: axes.trace_lhs_axes().to_vec(),
            trace_rhs_axes: axes.trace_rhs_axes().to_vec(),
            storage_conjugate: false,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweredTensorContractSpec {
    lhs_contracting_axes: AxisVec,
    rhs_contracting_axes: AxisVec,
    output_axes: AxisVec,
    lhs_storage_conjugate: bool,
    rhs_storage_conjugate: bool,
}

impl LoweredTensorContractSpec {
    #[inline]
    pub(crate) fn as_spec(&self) -> TensorContractSpec<'_> {
        TensorContractSpec::new_with_conjugation(
            &self.lhs_contracting_axes,
            &self.rhs_contracting_axes,
            OutputAxisOrder::from_axes(&self.output_axes),
            self.lhs_storage_conjugate,
            self.rhs_storage_conjugate,
        )
    }

    #[inline]
    pub(crate) fn lhs_storage_conjugate(&self) -> bool {
        self.lhs_storage_conjugate
    }

    #[inline]
    pub(crate) fn rhs_storage_conjugate(&self) -> bool {
        self.rhs_storage_conjugate
    }
}

pub(crate) fn lower_tensorcontract_adjoint_axes(
    lhs_nout: usize,
    lhs_nin: usize,
    rhs_nout: usize,
    rhs_nin: usize,
    axes: TensorContractSpec<'_>,
) -> Result<LoweredTensorContractSpec, OperationError> {
    if axes.lhs_contracting_axes().len() != axes.rhs_contracting_axes().len() {
        return Err(OperationError::ContractAxisCountMismatch {
            lhs: axes.lhs_contracting_axes().len(),
            rhs: axes.rhs_contracting_axes().len(),
        });
    }
    let lhs_rank = lhs_nout
        .checked_add(lhs_nin)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let rhs_rank = rhs_nout
        .checked_add(rhs_nin)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let contract_count = axes.lhs_contracting_axes().len();
    let core_output_rank = lhs_rank
        .checked_sub(contract_count)
        .and_then(|lhs_open| {
            rhs_rank
                .checked_sub(contract_count)
                .and_then(|rhs_open| lhs_open.checked_add(rhs_open))
        })
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    Ok(LoweredTensorContractSpec {
        lhs_contracting_axes: if axes.lhs_conjugate() {
            adjoint_tensor_axes_inline(lhs_nout, lhs_nin, axes.lhs_contracting_axes())?
        } else {
            AxisVec::from_slice(axes.lhs_contracting_axes())
        },
        rhs_contracting_axes: if axes.rhs_conjugate() {
            adjoint_tensor_axes_inline(rhs_nout, rhs_nin, axes.rhs_contracting_axes())?
        } else {
            AxisVec::from_slice(axes.rhs_contracting_axes())
        },
        output_axes: permutation_axes_inline(axes.output_permutation(), core_output_rank)?,
        lhs_storage_conjugate: axes.lhs_conjugate(),
        rhs_storage_conjugate: axes.rhs_conjugate(),
    })
}

pub(crate) fn adjoint_tensor_axis(
    nout: usize,
    nin: usize,
    axis: usize,
) -> Result<usize, OperationError> {
    let rank = nout
        .checked_add(nin)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    if axis >= rank {
        return Err(OperationError::InvalidAxisSet {
            tensor: "adjoint source",
            axes: vec![axis],
            rank,
        });
    }
    Ok(if axis < nout { nin + axis } else { axis - nout })
}

pub(crate) fn adjoint_tensor_axes(
    nout: usize,
    nin: usize,
    axes: &[usize],
) -> Result<Vec<usize>, OperationError> {
    adjoint_tensor_axes_inline(nout, nin, axes).map(AxisVec::into_vec)
}

fn adjoint_tensor_axes_inline(
    nout: usize,
    nin: usize,
    axes: &[usize],
) -> Result<AxisVec, OperationError> {
    axes.iter()
        .copied()
        .map(|axis| adjoint_tensor_axis(nout, nin, axis))
        .collect()
}

pub(crate) fn adjoint_block_structure_view(
    nout: usize,
    nin: usize,
    source: &BlockStructure,
) -> Result<BlockStructure, OperationError> {
    #[cfg(test)]
    ADJOINT_VIEW_BUILDS.with(|count| count.set(count.get() + 1));
    let rank = nout
        .checked_add(nin)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    if source.rank() != rank {
        return Err(OperationError::StructureRankMismatch {
            expected: rank,
            actual: source.rank(),
        });
    }

    let mut blocks = Vec::with_capacity(source.block_count());
    for index in 0..source.block_count() {
        let block = source.block(index)?;
        let key = adjoint_block_key(block.key())?;
        let mut shape = Vec::with_capacity(rank);
        shape.extend_from_slice(&block.shape()[nout..]);
        shape.extend_from_slice(&block.shape()[..nout]);
        let mut strides = Vec::with_capacity(rank);
        strides.extend_from_slice(&block.strides()[nout..]);
        strides.extend_from_slice(&block.strides()[..nout]);
        blocks.push(BlockSpec::with_key(key, shape, strides, block.offset())?);
    }
    BlockStructure::from_blocks_with_rank(rank, blocks)
        .map_err(OperationError::from_core_preserving_context)
}

fn adjoint_block_key(key: &BlockKey) -> Result<BlockKey, OperationError> {
    match key {
        BlockKey::Dense => Ok(BlockKey::Dense),
        BlockKey::Opaque(key) => Ok(BlockKey::from(key.clone())),
        BlockKey::FusionTree(tree) => Ok(BlockKey::from(FusionTreePairKey::pair(
            tree.domain_tree().clone(),
            tree.codomain_tree().clone(),
        ))),
        _ => Err(OperationError::InvalidArgument {
            message: "unsupported block key kind in adjoint lowering",
        }),
    }
}
