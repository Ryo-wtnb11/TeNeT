use core::ops::{Add, Mul};

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
        tensoradd_raw_strided_kernel_mapped(
            destination_data,
            lhs_data,
            destination_block.shape(),
            destination_stride,
            lhs_stride,
            checked_offset(destination_block.offset())?,
            checked_offset(lhs_block.offset())?,
            lhs.storage_conjugate(),
            alpha,
            D::zero(),
        )?;
        tensoradd_raw_strided_kernel_mapped(
            destination_data,
            rhs_data,
            destination_block.shape(),
            destination_stride,
            rhs_stride,
            checked_offset(destination_block.offset())?,
            checked_offset(rhs_block.offset())?,
            rhs.storage_conjugate(),
            beta,
            D::one(),
        )?;
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
    }
}
