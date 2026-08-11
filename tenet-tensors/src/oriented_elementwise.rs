use core::ops::{Add, Mul, Range};

use num_traits::{One, Zero};
use tenet_core::{BlockKey, BlockStructure, FusionTreePairKey, SectorId};
use tenet_operations::{
    bilinear_raw_strided_kernel_mapped, tensoradd_raw_strided_kernel_mapped, ConjugateValue,
    OperationError,
};

use crate::FusionOperand;

fn storage_block_index(
    operand: FusionOperand<'_>,
    logical_key: &FusionTreePairKey,
) -> Result<usize, OperationError> {
    let structure = operand.storage_space().structure();
    let index = if operand.storage_conjugate() {
        structure.find_block_index_by_adjoint_fusion_tree_pair(logical_key)
    } else {
        structure.find_block_index_by_fusion_tree_pair(logical_key)
    };
    index.ok_or_else(|| OperationError::MissingBlockKey {
        key: Box::new(BlockKey::from(logical_key.clone())),
    })
}

fn checked_offset(offset: usize) -> Result<isize, OperationError> {
    isize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: offset })
}

fn validate_logical_block_shape(
    operand: FusionOperand<'_>,
    logical_shape: &[usize],
    storage_shape: &[usize],
) -> Result<(), OperationError> {
    if logical_shape.len() != storage_shape.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented elementwise block",
        });
    }
    for (axis, &extent) in logical_shape.iter().enumerate() {
        if storage_shape[operand.storage_axis(axis)?] != extent {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented elementwise block",
            });
        }
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_oriented_fusion_layout(
    logical: &BlockStructure,
    operand: FusionOperand<'_>,
) -> Result<(), OperationError> {
    if logical.block_count() != operand.storage_space().structure().block_count() {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented elementwise layout",
        });
    }
    for logical_index in 0..logical.block_count() {
        let logical_block = logical.block(logical_index)?;
        let BlockKey::FusionTree(logical_key) = logical_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented elementwise logical layout",
            });
        };
        let storage_block = operand
            .storage_space()
            .structure()
            .block(storage_block_index(operand, logical_key)?)?;
        validate_logical_block_shape(operand, logical_block.shape(), storage_block.shape())?;
    }
    Ok(())
}

/// Copies one logical degeneracy rectangle from an owned or lazy-adjoint
/// fusion tensor into a compact destination.
#[doc(hidden)]
pub fn oriented_fusion_restrict_into<D>(
    destination: &BlockStructure,
    destination_data: &mut [D],
    source: FusionOperand<'_>,
    source_data: &[D],
    logical_starts: &[usize],
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    if destination_data.len() != destination.required_len()?
        || source_data.len() != source.storage_space().required_len()?
        || logical_starts.len() != destination.rank()
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented degeneracy restriction storage",
        });
    }
    for destination_index in 0..destination.block_count() {
        let destination_block = destination.block(destination_index)?;
        let BlockKey::FusionTree(logical_key) = destination_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented degeneracy restriction destination",
            });
        };
        let source_block = source
            .storage_space()
            .structure()
            .block(storage_block_index(source, logical_key)?)?;
        let destination_stride = |axis| {
            isize::try_from(destination_block.strides()[axis])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let source_stride = |axis| {
            isize::try_from(source_block.strides()[source.storage_axis(axis)?])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let mut source_offset = source_block.offset();
        for (axis, &logical_start) in logical_starts.iter().take(destination.rank()).enumerate() {
            let storage_axis = source.storage_axis(axis)?;
            let source_extent = source_block.shape()[storage_axis];
            let end = logical_start
                .checked_add(destination_block.shape()[axis])
                .ok_or(OperationError::ElementCountOverflow)?;
            if end > source_extent {
                return Err(OperationError::StructureMismatch {
                    tensor: "oriented degeneracy restriction rectangle",
                });
            }
            source_offset = source_offset
                .checked_add(
                    logical_start
                        .checked_mul(source_block.strides()[storage_axis])
                        .ok_or(OperationError::ElementCountOverflow)?,
                )
                .ok_or(OperationError::ElementCountOverflow)?;
            destination_stride(axis)?;
            source_stride(axis)?;
        }
        tensoradd_raw_strided_kernel_mapped(
            destination_data,
            source_data,
            destination_block.shape(),
            destination_stride,
            source_stride,
            checked_offset(destination_block.offset())?,
            checked_offset(source_offset)?,
            source.storage_conjugate(),
            D::one(),
            D::zero(),
        )?;
    }
    Ok(())
}

struct ScatterBlock {
    shape: Vec<usize>,
    destination_strides: Vec<isize>,
    source_strides: Vec<isize>,
    destination_offset: isize,
    source_offset: isize,
}

fn preflight_scatter_bounds(
    len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<(), OperationError> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or(OperationError::ElementCountOverflow)
    })?;
    if shape.contains(&0) {
        return Ok(());
    }
    let maximum = shape
        .iter()
        .zip(strides)
        .try_fold(offset, |maximum, (&extent, &stride)| {
            let steps =
                isize::try_from(extent - 1).map_err(|_| OperationError::ElementCountOverflow)?;
            maximum
                .checked_add(
                    stride
                        .checked_mul(steps)
                        .ok_or(OperationError::ElementCountOverflow)?,
                )
                .ok_or(OperationError::ElementCountOverflow)
        })?;
    let maximum = usize::try_from(maximum)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    if maximum >= len {
        return Err(OperationError::OffsetOverflow { value: maximum });
    }
    Ok(())
}

/// Adds source fusion-tree blocks into logical rectangles of a full destination.
///
/// All keys, ranks, shapes, ranges, strides, offsets, and storage bounds are
/// preflighted for every block before the first destination element is changed.
#[doc(hidden)]
pub fn fusion_scatter_add_assign<D>(
    destination: &BlockStructure,
    destination_data: &mut [D],
    logical_source: &BlockStructure,
    source: FusionOperand<'_>,
    source_data: &[D],
    ranges: &[Option<Range<usize>>],
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    if destination_data.len() != destination.required_len()?
        || source_data.len() != source.storage_space().required_len()?
        || destination.rank() != logical_source.rank()
        || ranges.len() != destination.rank()
    {
        return Err(OperationError::StructureMismatch {
            tensor: "fusion scatter accumulation storage",
        });
    }
    let mut blocks = Vec::with_capacity(logical_source.block_count());
    for source_index in 0..logical_source.block_count() {
        let logical_block = logical_source.block(source_index)?;
        let BlockKey::FusionTree(logical_key) = logical_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion scatter accumulation source key",
            });
        };
        let destination_index = destination
            .find_block_index_by_key(logical_block.key())
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(logical_block.key().clone()),
            })?;
        let destination_block = destination.block(destination_index)?;
        let storage_block = source
            .storage_space()
            .structure()
            .block(storage_block_index(source, logical_key)?)?;
        validate_logical_block_shape(source, logical_block.shape(), storage_block.shape())?;
        if logical_block.shape().len() != destination.rank()
            || destination_block.shape().len() != destination.rank()
        {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion scatter accumulation block rank",
            });
        }
        let mut destination_offset = destination_block.offset();
        for (axis, range) in ranges.iter().enumerate() {
            let source_extent = logical_block.shape()[axis];
            let destination_extent = destination_block.shape()[axis];
            let start = match range {
                None if source_extent == destination_extent => 0,
                None => {
                    return Err(OperationError::StructureMismatch {
                        tensor: "fusion scatter unsliced extent",
                    });
                }
                Some(range)
                    if range.end.checked_sub(range.start) == Some(source_extent)
                        && range.end <= destination_extent =>
                {
                    range.start
                }
                Some(_) => {
                    return Err(OperationError::StructureMismatch {
                        tensor: "fusion scatter sliced extent",
                    });
                }
            };
            destination_offset = destination_offset
                .checked_add(
                    start
                        .checked_mul(destination_block.strides()[axis])
                        .ok_or(OperationError::ElementCountOverflow)?,
                )
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        let destination_strides = destination_block
            .strides()
            .iter()
            .map(|&stride| {
                isize::try_from(stride)
                    .map_err(|_| OperationError::StrideOverflow { value: stride })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_strides = (0..destination.rank())
            .map(|axis| {
                let stride = storage_block.strides()[source.storage_axis(axis)?];
                isize::try_from(stride)
                    .map_err(|_| OperationError::StrideOverflow { value: stride })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let destination_offset = checked_offset(destination_offset)?;
        let source_offset = checked_offset(storage_block.offset())?;
        preflight_scatter_bounds(
            destination_data.len(),
            logical_block.shape(),
            &destination_strides,
            destination_offset,
        )?;
        preflight_scatter_bounds(
            source_data.len(),
            logical_block.shape(),
            &source_strides,
            source_offset,
        )?;
        blocks.push(ScatterBlock {
            shape: logical_block.shape().to_vec(),
            destination_strides,
            source_strides,
            destination_offset,
            source_offset,
        });
    }
    for block in &blocks {
        tensoradd_raw_strided_kernel_mapped(
            destination_data,
            source_data,
            &block.shape,
            |axis| Ok(block.destination_strides[axis]),
            |axis| Ok(block.source_strides[axis]),
            block.destination_offset,
            block.source_offset,
            source.storage_conjugate(),
            D::one(),
            D::one(),
        )?;
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn oriented_fusion_add_into<D>(
    destination: &BlockStructure,
    destination_data: &mut [D],
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
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
        + strided_kernel::MaybeSendSync,
{
    if destination_data.len() != destination.required_len()?
        || lhs_data.len() != lhs.storage_space().required_len()?
        || rhs_data.len() != rhs.storage_space().required_len()?
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented elementwise storage",
        });
    }
    validate_oriented_fusion_layout(destination, lhs)?;
    validate_oriented_fusion_layout(destination, rhs)?;
    for destination_index in 0..destination.block_count() {
        let destination_block = destination.block(destination_index)?;
        let BlockKey::FusionTree(logical_key) = destination_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented elementwise destination",
            });
        };
        let lhs_block = lhs
            .storage_space()
            .structure()
            .block(storage_block_index(lhs, logical_key)?)?;
        let rhs_block = rhs
            .storage_space()
            .structure()
            .block(storage_block_index(rhs, logical_key)?)?;
        let destination_stride = |axis| {
            isize::try_from(destination_block.strides()[axis])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let lhs_stride = |axis| {
            isize::try_from(lhs_block.strides()[lhs.storage_axis(axis)?])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let rhs_stride = |axis| {
            isize::try_from(rhs_block.strides()[rhs.storage_axis(axis)?])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let destination_offset = checked_offset(destination_block.offset())?;
        let lhs_offset = checked_offset(lhs_block.offset())?;
        let rhs_offset = checked_offset(rhs_block.offset())?;
        for axis in 0..destination_block.shape().len() {
            destination_stride(axis)?;
            lhs_stride(axis)?;
            rhs_stride(axis)?;
        }
        if !alpha.is_zero() {
            tensoradd_raw_strided_kernel_mapped(
                destination_data,
                lhs_data,
                destination_block.shape(),
                destination_stride,
                lhs_stride,
                destination_offset,
                lhs_offset,
                lhs.storage_conjugate(),
                alpha,
                D::zero(),
            )?;
        }
        if !beta.is_zero() {
            tensoradd_raw_strided_kernel_mapped(
                destination_data,
                rhs_data,
                destination_block.shape(),
                destination_stride,
                rhs_stride,
                destination_offset,
                rhs_offset,
                rhs.storage_conjugate(),
                beta,
                if alpha.is_zero() { D::zero() } else { D::one() },
            )?;
        }
    }
    if alpha.is_zero() && beta.is_zero() {
        destination_data.fill(D::zero());
    }
    Ok(())
}

#[doc(hidden)]
pub fn oriented_fusion_inner<D>(
    logical: &BlockStructure,
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
    mut sector_weight: impl FnMut(SectorId) -> D,
) -> Result<D, OperationError>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + Zero + ConjugateValue,
{
    if lhs_data.len() != lhs.storage_space().required_len()?
        || rhs_data.len() != rhs.storage_space().required_len()?
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented inner storage",
        });
    }
    validate_oriented_fusion_layout(logical, lhs)?;
    validate_oriented_fusion_layout(logical, rhs)?;
    let mut total = D::zero();
    for logical_index in 0..logical.block_count() {
        let logical_block = logical.block(logical_index)?;
        let BlockKey::FusionTree(logical_key) = logical_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented inner logical layout",
            });
        };
        let lhs_block = lhs
            .storage_space()
            .structure()
            .block(storage_block_index(lhs, logical_key)?)?;
        let rhs_block = rhs
            .storage_space()
            .structure()
            .block(storage_block_index(rhs, logical_key)?)?;
        let lhs_stride = |axis| {
            let storage_axis = lhs.storage_axis(axis)?;
            isize::try_from(lhs_block.strides()[storage_axis])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let rhs_stride = |axis| {
            let storage_axis = rhs.storage_axis(axis)?;
            isize::try_from(rhs_block.strides()[storage_axis])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let partial = bilinear_raw_strided_kernel_mapped(
            lhs_data,
            rhs_data,
            logical_block.shape(),
            lhs_stride,
            rhs_stride,
            checked_offset(lhs_block.offset())?,
            checked_offset(rhs_block.offset())?,
            !lhs.storage_conjugate(),
            rhs.storage_conjugate(),
        )?;
        total = total + partial * sector_weight(logical_key.codomain_tree().coupled());
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tenet_core::{
        BlockSpec, BlockStructure, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
        SectorId, SectorLeg, TensorMapSpace, Z2FusionRule,
    };

    fn padded_fixture() -> (
        crate::DynamicFusionMapSpace,
        crate::DynamicFusionMapSpace,
        Vec<Complex64>,
        Vec<Complex64>,
    ) {
        let rule = Z2FusionRule;
        let vacuum = SectorId::new(0);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(vacuum, 3)], false)]),
        );
        let canonical = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
            homspace.clone(),
            &rule,
            [vec![2, 3]],
        )
        .unwrap();
        let canonical = crate::DynamicFusionMapSpace::from_typed(&canonical);
        let block = canonical.structure().only_block().unwrap();
        let padded_structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(block.key().clone(), vec![2, 3], vec![2, 5], 1).unwrap()],
        )
        .unwrap();
        let padded = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
            homspace,
            padded_structure,
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let padded = crate::DynamicFusionMapSpace::from_typed(&padded);
        let (logical, _) =
            crate::adjoint::adjoint_dyn(&rule, &canonical, &[Complex64::zero(); 6]).unwrap();
        let direct = (0..6)
            .map(|index| Complex64::new(index as f64 + 1.0, 0.5 - index as f64))
            .collect();
        let mut parent = vec![Complex64::new(99.0, 99.0); padded.required_len().unwrap()];
        for column in 0..3 {
            for row in 0..2 {
                parent[1 + 2 * row + 5 * column] =
                    Complex64::new((row + 2 * column) as f64, row as f64 - column as f64);
            }
        }
        (logical, padded, direct, parent)
    }

    fn two_key_destination() -> BlockStructure {
        let rule = Z2FusionRule;
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(even, 3), (odd, 3)], false)]),
            FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
        );
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([6], [4]).unwrap(),
            homspace,
            &rule,
            [vec![3, 2], vec![3, 2]],
        )
        .unwrap()
        .subblock_structure()
        .as_ref()
        .clone()
    }

    fn bind_source(
        structure: BlockStructure,
        shapes: &[Vec<usize>],
    ) -> crate::DynamicFusionMapSpace {
        let rule = Z2FusionRule;
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(even, shapes[0][0]), (odd, shapes[1][0])],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(even, shapes[0][1]), (odd, shapes[1][1])],
                false,
            )]),
        );
        let typed = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<1, 1>::from_dims(
                [shapes[0][0] + shapes[1][0]],
                [shapes[0][1] + shapes[1][1]],
            )
            .unwrap(),
            homspace,
            structure,
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        crate::DynamicFusionMapSpace::from_typed(&typed)
    }

    fn source_with_shapes(
        destination: &BlockStructure,
        shapes: &[Vec<usize>],
    ) -> crate::DynamicFusionMapSpace {
        let mut offset = 0;
        let structure = BlockStructure::from_blocks_with_rank(
            2,
            shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| {
                    let key = destination.block(index).unwrap().key().clone();
                    let block =
                        BlockSpec::column_major_with_key(key, shape.clone(), offset).unwrap();
                    offset = block.storage_end_exclusive().unwrap();
                    block
                })
                .collect(),
        )
        .unwrap();
        bind_source(structure, shapes)
    }

    #[test]
    fn fusion_scatter_adds_nonprefix_rectangles_for_exact_tree_keys() {
        let destination = two_key_destination();
        let source = source_with_shapes(&destination, &[vec![1, 2], vec![1, 2]]);
        let mut output = vec![0.0; destination.required_len().unwrap()];
        fusion_scatter_add_assign(
            &destination,
            &mut output,
            source.structure(),
            FusionOperand::direct(&source),
            &[1.0, 2.0, 3.0, 4.0],
            &[Some(1..2), None],
        )
        .unwrap();
        assert_eq!(
            output,
            vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0]
        );
    }

    #[test]
    fn fusion_scatter_preflights_all_shapes_before_mutation() {
        let destination = two_key_destination();
        let source = source_with_shapes(&destination, &[vec![1, 2], vec![2, 2]]);
        let mut output = vec![7.0; destination.required_len().unwrap()];
        let before = output.clone();
        assert!(matches!(
            fusion_scatter_add_assign(
                &destination,
                &mut output,
                source.structure(),
                FusionOperand::direct(&source),
                &[1.0; 6],
                &[Some(1..2), None],
            ),
            Err(OperationError::StructureMismatch { .. })
        ));
        assert_eq!(output, before);
    }

    #[test]
    fn fusion_scatter_preflights_stride_overflow_before_mutation() {
        let canonical = two_key_destination();
        let first_key = canonical.block(0).unwrap().key().clone();
        let second_key = canonical.block(1).unwrap().key().clone();
        let destination = canonical;
        let source_structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::column_major_with_key(first_key, vec![1, 2], 0).unwrap(),
                BlockSpec::with_key(second_key, vec![1, 2], vec![usize::MAX, 1], 2).unwrap(),
            ],
        )
        .unwrap();
        let source = bind_source(source_structure, &[vec![1, 2], vec![1, 2]]);
        let mut output = vec![11.0; destination.required_len().unwrap()];
        let before = output.clone();
        assert!(matches!(
            fusion_scatter_add_assign(
                &destination,
                &mut output,
                source.structure(),
                FusionOperand::direct(&source),
                &[1.0, 2.0, 3.0, 4.0],
                &[Some(1..2), None],
            ),
            Err(OperationError::StrideOverflow { value: usize::MAX })
        ));
        assert_eq!(output, before);
    }

    #[test]
    fn oriented_add_and_inner_use_padded_parent_strides_in_both_orders() {
        let (logical, padded, direct, parent) = padded_fixture();
        let direct_operand = FusionOperand::direct(&logical);
        let adjoint_operand = FusionOperand::adjoint(&padded);
        let alpha = Complex64::new(0.5, -0.25);
        let beta = Complex64::new(-1.0, 0.75);
        let mut expected = vec![Complex64::zero(); 6];
        for column in 0..2 {
            for row in 0..3 {
                let logical_index = row + 3 * column;
                let parent_index = 1 + 2 * column + 5 * row;
                expected[logical_index] =
                    alpha * direct[logical_index] + beta * parent[parent_index].conj();
            }
        }
        for (lhs, lhs_data, rhs, rhs_data, lhs_factor, rhs_factor) in [
            (
                direct_operand,
                direct.as_slice(),
                adjoint_operand,
                parent.as_slice(),
                alpha,
                beta,
            ),
            (
                adjoint_operand,
                parent.as_slice(),
                direct_operand,
                direct.as_slice(),
                beta,
                alpha,
            ),
        ] {
            let mut output = vec![Complex64::zero(); 6];
            oriented_fusion_add_into(
                logical.structure(),
                &mut output,
                lhs,
                lhs_data,
                rhs,
                rhs_data,
                lhs_factor,
                rhs_factor,
            )
            .unwrap();
            assert_eq!(output, expected);
        }

        let expected_inner = (0..6).fold(Complex64::zero(), |sum, logical_index| {
            let row = logical_index % 3;
            let column = logical_index / 3;
            let parent_index = 1 + 2 * column + 5 * row;
            sum + direct[logical_index].conj() * parent[parent_index].conj()
        });
        assert_eq!(
            oriented_fusion_inner(
                logical.structure(),
                direct_operand,
                &direct,
                adjoint_operand,
                &parent,
                |_| Complex64::one(),
            )
            .unwrap(),
            expected_inner
        );
        assert_eq!(
            oriented_fusion_inner(
                logical.structure(),
                adjoint_operand,
                &parent,
                direct_operand,
                &direct,
                |_| Complex64::one(),
            )
            .unwrap(),
            expected_inner.conj()
        );

        let mut output = vec![Complex64::zero(); direct.len()];
        let inactive = vec![Complex64::new(f64::NAN, f64::NAN); parent.len()];
        oriented_fusion_add_into(
            logical.structure(),
            &mut output,
            direct_operand,
            &direct,
            adjoint_operand,
            &inactive,
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
        assert_eq!(output, direct);

        let mut output = vec![Complex64::zero(); direct.len()];
        oriented_fusion_add_into(
            logical.structure(),
            &mut output,
            direct_operand,
            &vec![Complex64::new(f64::NAN, f64::NAN); direct.len()],
            adjoint_operand,
            &parent,
            Complex64::zero(),
            Complex64::one(),
        )
        .unwrap();
        let expected = (0..6)
            .map(|logical_index| {
                let row = logical_index % 3;
                let column = logical_index / 3;
                parent[1 + 2 * column + 5 * row].conj()
            })
            .collect::<Vec<_>>();
        assert_eq!(output, expected);

        let mut output = vec![Complex64::new(f64::NAN, f64::NAN); direct.len()];
        oriented_fusion_add_into(
            logical.structure(),
            &mut output,
            direct_operand,
            &vec![Complex64::new(f64::NAN, f64::NAN); direct.len()],
            adjoint_operand,
            &inactive,
            Complex64::zero(),
            Complex64::zero(),
        )
        .unwrap();
        assert_eq!(output, vec![Complex64::zero(); direct.len()]);
    }
}
