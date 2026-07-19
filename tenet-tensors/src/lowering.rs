use std::array;

use tenet_core::{
    BlockKey, BlockSpec, BlockStructure, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreePairKey, TensorMapSpace,
};

use crate::DynamicFusionMapSpace;
use crate::{OperationError, TreeTransformOperation};
use tenet_operations::{
    permutation_axes, OutputAxisOrder, TensorContractSpec, TensorTraceAxisSpec,
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
pub(crate) struct LoweredTensorAddSourceOperation {
    operation: TreeTransformOperation,
    storage_conjugate: bool,
}

impl LoweredTensorAddSourceOperation {
    #[inline]
    pub(crate) fn into_operation(self) -> TreeTransformOperation {
        self.operation
    }

    #[inline]
    pub(crate) fn storage_conjugate(&self) -> bool {
        self.storage_conjugate
    }
}

pub(crate) fn lower_tensoradd_source_operation<const SRC_NOUT: usize, const SRC_NIN: usize>(
    operation: TreeTransformOperation,
    source_conjugate: bool,
) -> Result<LoweredTensorAddSourceOperation, OperationError> {
    if !source_conjugate {
        return Ok(LoweredTensorAddSourceOperation {
            operation,
            storage_conjugate: false,
        });
    }

    let operation = match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        } => TreeTransformOperation::permute(
            adjoint_tensor_axes(SRC_NOUT, SRC_NIN, &codomain_permutation)?,
            adjoint_tensor_axes(SRC_NOUT, SRC_NIN, &domain_permutation)?,
        ),
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => TreeTransformOperation::transpose(
            adjoint_tensor_axes(SRC_NOUT, SRC_NIN, &codomain_permutation)?,
            adjoint_tensor_axes(SRC_NOUT, SRC_NIN, &domain_permutation)?,
        ),
        TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        } => {
            validate_braid_source_axes::<SRC_NOUT, SRC_NIN>(
                &codomain_permutation,
                &domain_permutation,
            )?;
            let (codomain_levels, domain_levels) =
                lower_adjoint_braid_levels::<SRC_NOUT, SRC_NIN>(&codomain_levels, &domain_levels)?;
            TreeTransformOperation::braid(
                adjoint_tensor_axes(SRC_NOUT, SRC_NIN, &codomain_permutation)?,
                adjoint_tensor_axes(SRC_NOUT, SRC_NIN, &domain_permutation)?,
                codomain_levels,
                domain_levels,
            )
        }
    };

    Ok(LoweredTensorAddSourceOperation {
        operation,
        storage_conjugate: true,
    })
}

fn lower_adjoint_braid_levels<const SRC_NOUT: usize, const SRC_NIN: usize>(
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), OperationError> {
    if codomain_levels.len() != SRC_NOUT {
        return Err(OperationError::RankMismatch {
            expected: SRC_NOUT,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != SRC_NIN {
        return Err(OperationError::RankMismatch {
            expected: SRC_NIN,
            actual: domain_levels.len(),
        });
    }
    validate_distinct_braid_levels(codomain_levels, domain_levels)?;

    let min_level = codomain_levels.iter().chain(domain_levels).copied().min();
    let max_level = codomain_levels.iter().chain(domain_levels).copied().max();
    let Some((min_level, max_level)) = min_level.zip(max_level) else {
        return Ok((Vec::new(), Vec::new()));
    };

    let mut lowered_codomain_levels = vec![usize::MAX; SRC_NIN];
    let mut lowered_domain_levels = vec![usize::MAX; SRC_NOUT];
    for (source_axis, &level) in codomain_levels.iter().enumerate() {
        set_adjoint_reflected_braid_level::<SRC_NOUT, SRC_NIN>(
            source_axis,
            reflect_level(min_level, max_level, level),
            &mut lowered_codomain_levels,
            &mut lowered_domain_levels,
        )?;
    }
    for (source_domain_axis, &level) in domain_levels.iter().enumerate() {
        set_adjoint_reflected_braid_level::<SRC_NOUT, SRC_NIN>(
            SRC_NOUT + source_domain_axis,
            reflect_level(min_level, max_level, level),
            &mut lowered_codomain_levels,
            &mut lowered_domain_levels,
        )?;
    }

    debug_assert!(!lowered_codomain_levels.contains(&usize::MAX));
    debug_assert!(!lowered_domain_levels.contains(&usize::MAX));
    Ok((lowered_codomain_levels, lowered_domain_levels))
}

#[inline]
fn reflect_level(min_level: usize, max_level: usize, level: usize) -> usize {
    max_level - (level - min_level)
}

fn set_adjoint_reflected_braid_level<const SRC_NOUT: usize, const SRC_NIN: usize>(
    source_axis: usize,
    reflected_level: usize,
    lowered_codomain_levels: &mut [usize],
    lowered_domain_levels: &mut [usize],
) -> Result<(), OperationError> {
    let lowered_axis = adjoint_tensor_axis(SRC_NOUT, SRC_NIN, source_axis)?;
    if lowered_axis < SRC_NIN {
        lowered_codomain_levels[lowered_axis] = reflected_level;
    } else {
        lowered_domain_levels[lowered_axis - SRC_NIN] = reflected_level;
    }
    Ok(())
}

fn validate_braid_source_axes<const SRC_NOUT: usize, const SRC_NIN: usize>(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(), OperationError> {
    let rank = SRC_NOUT
        .checked_add(SRC_NIN)
        .ok_or(OperationError::ElementCountOverflow)?;
    let mut axes = Vec::with_capacity(codomain_permutation.len() + domain_permutation.len());
    axes.extend_from_slice(codomain_permutation);
    axes.extend_from_slice(domain_permutation);
    if axes.len() != rank {
        return Err(OperationError::InvalidPermutation { axes, rank });
    }

    let mut seen = vec![false; rank];
    for &axis in &axes {
        if axis >= rank || seen[axis] {
            return Err(OperationError::InvalidPermutation { axes, rank });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_distinct_braid_levels(
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<(), OperationError> {
    let mut levels = Vec::with_capacity(codomain_levels.len() + domain_levels.len());
    levels.extend_from_slice(codomain_levels);
    levels.extend_from_slice(domain_levels);
    for index in 0..levels.len() {
        if levels[..index].contains(&levels[index]) {
            return Err(OperationError::InvalidAxisSet {
                tensor: "braid level set",
                axes: levels,
                rank: codomain_levels.len() + domain_levels.len(),
            });
        }
    }
    Ok(())
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

pub(crate) fn lower_tensortrace_source_adjoint_axes<const SRC_NOUT: usize, const SRC_NIN: usize>(
    axes: TensorTraceAxisSpec<'_>,
) -> Result<LoweredTensorTraceAxisSpec, OperationError> {
    if axes.source_conjugate() {
        Ok(LoweredTensorTraceAxisSpec {
            output_axes: adjoint_tensor_axes(SRC_NOUT, SRC_NIN, axes.output_axes())?,
            trace_lhs_axes: adjoint_tensor_axes(SRC_NOUT, SRC_NIN, axes.trace_lhs_axes())?,
            trace_rhs_axes: adjoint_tensor_axes(SRC_NOUT, SRC_NIN, axes.trace_rhs_axes())?,
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
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_axes: Vec<usize>,
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
        .ok_or(OperationError::ElementCountOverflow)?;
    let rhs_rank = rhs_nout
        .checked_add(rhs_nin)
        .ok_or(OperationError::ElementCountOverflow)?;
    let contract_count = axes.lhs_contracting_axes().len();
    let core_output_rank = lhs_rank
        .checked_sub(contract_count)
        .and_then(|lhs_open| {
            rhs_rank
                .checked_sub(contract_count)
                .and_then(|rhs_open| lhs_open.checked_add(rhs_open))
        })
        .ok_or(OperationError::ElementCountOverflow)?;
    Ok(LoweredTensorContractSpec {
        lhs_contracting_axes: if axes.lhs_conjugate() {
            adjoint_tensor_axes(lhs_nout, lhs_nin, axes.lhs_contracting_axes())?
        } else {
            axes.lhs_contracting_axes().to_vec()
        },
        rhs_contracting_axes: if axes.rhs_conjugate() {
            adjoint_tensor_axes(rhs_nout, rhs_nin, axes.rhs_contracting_axes())?
        } else {
            axes.rhs_contracting_axes().to_vec()
        },
        output_axes: permutation_axes(axes.output_permutation(), core_output_rank)?,
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
        .ok_or(OperationError::ElementCountOverflow)?;
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
    axes.iter()
        .copied()
        .map(|axis| adjoint_tensor_axis(nout, nin, axis))
        .collect()
}

pub(crate) fn adjoint_fusion_space_view<const NOUT: usize, const NIN: usize>(
    source: &FusionTensorMapSpace<NOUT, NIN>,
) -> Result<FusionTensorMapSpace<NIN, NOUT>, OperationError> {
    let codomain_dims = array::from_fn(|index| source.dense_space().domain().dims()[index]);
    let domain_dims = array::from_fn(|index| source.dense_space().codomain().dims()[index]);
    let dense_space = TensorMapSpace::<NIN, NOUT>::from_dims(codomain_dims, domain_dims)
        .map_err(OperationError::from_core_preserving_context)?;
    let homspace = FusionTreeHomSpace::new(
        source.homspace().domain().clone(),
        source.homspace().codomain().clone(),
    );
    let structure =
        adjoint_block_structure_view(NOUT, NIN, source.subblock_structure())?.into_shared();
    FusionTensorMapSpace::from_shared_subblock_structure(dense_space, homspace, structure)
        .map_err(OperationError::from_core_preserving_context)?
        .try_inherit_rule_identity(source)
        .map_err(OperationError::from_core_preserving_context)
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
        .ok_or(OperationError::ElementCountOverflow)?;
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

pub(crate) fn prelowered_storage_block_index<'a>(
    logical_space: &'a DynamicFusionMapSpace,
    storage_space: &'a DynamicFusionMapSpace,
    storage_conjugate: bool,
) -> impl Fn(usize) -> Result<usize, OperationError> + 'a {
    move |index| {
        if !storage_conjugate {
            return Ok(index);
        }
        let logical = logical_space.structure().block(index)?;
        let storage_key = adjoint_block_key(logical.key())?;
        storage_space
            .structure()
            .find_block_index_by_key(&storage_key)
            .ok_or_else(|| {
                OperationError::Core(tenet_core::CoreError::MissingBlockKey {
                    key: Box::new(storage_key),
                })
            })
    }
}

pub(crate) fn prelowered_storage_axis<'a>(
    logical_space: &'a DynamicFusionMapSpace,
    storage_space: &'a DynamicFusionMapSpace,
    storage_conjugate: bool,
) -> impl Fn(usize) -> Result<usize, OperationError> + 'a {
    move |axis| {
        if axis >= logical_space.rank() {
            return Err(OperationError::InvalidAxisSet {
                tensor: "logical src",
                axes: vec![axis],
                rank: logical_space.rank(),
            });
        }
        if !storage_conjugate {
            return Ok(axis);
        }
        Ok(if axis < storage_space.nin() {
            storage_space.nout() + axis
        } else {
            axis - storage_space.nin()
        })
    }
}
