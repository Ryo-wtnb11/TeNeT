use core::ops::{Add, Mul, Range};

use num_traits::{One, Zero};
use smallvec::SmallVec;
use tenet_core::{BlockKey, BlockStructure, FusionTreePairKey, SectorId};
use tenet_operations::{
    bilinear_raw_strided_kernel_mapped, ConjugateValue, OperationError,
    RecouplingCoefficientAction, StridedHostKernelAdapter, WideScalar,
};

use crate::FusionOperand;

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
            .block(operand.storage_block_index(logical_key)?)?;
        validate_logical_block_shape(operand, logical_block.shape(), storage_block.shape())?;
    }
    Ok(())
}

/// The uncoupled sector a fusion-tree pair key carries on one logical axis.
///
/// Codomain axes index the codomain tree, the remaining axes the domain tree,
/// which is the same split the block shape uses.
fn logical_axis_sector(key: &FusionTreePairKey, axis: usize) -> Result<SectorId, OperationError> {
    let codomain = key.codomain_uncoupled();
    if let Some(&sector) = codomain.get(axis) {
        return Ok(sector);
    }
    key.domain_uncoupled()
        .get(axis - codomain.len())
        .copied()
        .ok_or_else(|| OperationError::StructureMismatch {
            tensor: "oriented elementwise axis sector",
        })
}

/// Looks a block's sector up in a per-axis selection table.
///
/// Tables are sorted by [`SectorId`]; an unsorted table can only make this
/// miss, which every caller turns into `StructureMismatch`, never into a
/// different sector's entry.
fn selected_for_sector<T>(table: &[(SectorId, T)], sector: SectorId) -> Option<&T> {
    table
        .binary_search_by_key(&sector, |(candidate, _)| *candidate)
        .ok()
        .map(|index| &table[index].1)
}

/// One axis' destination ranges in a scatter, keyed by the sector the block
/// carries on that axis, or `None` when the axis is not sliced.
#[doc(hidden)]
pub type SectorRangeTable<'a> = Option<&'a [(SectorId, Range<usize>)]>;

/// One axis' source starts in a restriction, keyed by the sector the
/// destination block carries on that axis, or `None` for a start of zero.
#[doc(hidden)]
pub type SectorStartTable<'a> = Option<&'a [(SectorId, usize)]>;

/// Copies one logical degeneracy rectangle per block from an owned or
/// lazy-adjoint fusion tensor into a compact destination.
///
/// `logical_starts` holds one entry per logical axis: `None` starts that axis
/// at zero for every sector, `Some(table)` gives the start per uncoupled
/// sector of the destination block's own key, sorted by [`SectorId`]. A block
/// whose sector is absent from a `Some` table is a `StructureMismatch` rather
/// than an implicit zero start, so a caller that forgot a sector cannot get a
/// silently misaligned copy. The extent always comes from the destination
/// block's own shape, and `start + destination extent <= source extent` is
/// checked per block and axis.
#[doc(hidden)]
pub fn oriented_fusion_restrict_into<D>(
    destination: &BlockStructure,
    destination_data: &mut [D],
    source: FusionOperand<'_>,
    source_data: &[D],
    logical_starts: &[SectorStartTable<'_>],
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
    let mut kernels = StridedHostKernelAdapter::default();
    let mut destination_strides = SmallVec::<[isize; 8]>::new();
    let mut source_strides = SmallVec::<[isize; 8]>::new();
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
            .block(source.storage_block_index(logical_key)?)?;
        let stride = |stride: usize| {
            isize::try_from(stride).map_err(|_| OperationError::ElementCountOverflow)
        };
        destination_strides.clear();
        source_strides.clear();
        let mut source_offset = source_block.offset();
        for (axis, table) in logical_starts.iter().take(destination.rank()).enumerate() {
            let logical_start = match table {
                None => 0,
                Some(table) => *selected_for_sector(table, logical_axis_sector(logical_key, axis)?)
                    .ok_or_else(|| OperationError::StructureMismatch {
                        tensor: "oriented degeneracy restriction sector",
                    })?,
            };
            let storage_axis = source.storage_axis(axis)?;
            let source_extent = source_block.shape()[storage_axis];
            let end = logical_start
                .checked_add(destination_block.shape()[axis])
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
            if end > source_extent {
                return Err(OperationError::StructureMismatch {
                    tensor: "oriented degeneracy restriction rectangle",
                });
            }
            source_offset = source_offset
                .checked_add(
                    logical_start
                        .checked_mul(source_block.strides()[storage_axis])
                        .ok_or_else(|| OperationError::ElementCountOverflow)?,
                )
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
            destination_strides.push(stride(destination_block.strides()[axis])?);
            source_strides.push(stride(source_block.strides()[storage_axis])?);
        }
        kernels.tensoradd_strided_checked(
            destination_data,
            source_data,
            destination_block.shape(),
            &destination_strides,
            &source_strides,
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
            .ok_or_else(|| OperationError::ElementCountOverflow)
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
                        .ok_or_else(|| OperationError::ElementCountOverflow)?,
                )
                .ok_or_else(|| OperationError::ElementCountOverflow)
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
///
/// `ranges` holds one entry per logical axis: `None` requires the source and
/// destination extents to be equal, `Some(table)` gives the destination range
/// per uncoupled sector of the source block's own key, sorted by [`SectorId`].
/// A block whose sector is absent from a `Some` table is a preflight
/// `StructureMismatch`. Accumulation is `beta = 1`.
#[doc(hidden)]
pub fn fusion_scatter_add_assign<D>(
    destination: &BlockStructure,
    destination_data: &mut [D],
    logical_source: &BlockStructure,
    source: FusionOperand<'_>,
    source_data: &[D],
    ranges: &[SectorRangeTable<'_>],
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
            .block(source.storage_block_index(logical_key)?)?;
        validate_logical_block_shape(source, logical_block.shape(), storage_block.shape())?;
        if logical_block.shape().len() != destination.rank()
            || destination_block.shape().len() != destination.rank()
        {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion scatter accumulation block rank",
            });
        }
        let mut destination_offset = destination_block.offset();
        for (axis, table) in ranges.iter().enumerate() {
            let source_extent = logical_block.shape()[axis];
            let destination_extent = destination_block.shape()[axis];
            let range = match table {
                None => None,
                Some(table) => Some(
                    selected_for_sector(table, logical_axis_sector(logical_key, axis)?)
                        .ok_or_else(|| OperationError::StructureMismatch {
                            tensor: "fusion scatter sliced sector",
                        })?,
                ),
            };
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
                        .ok_or_else(|| OperationError::ElementCountOverflow)?,
                )
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
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
    let mut kernels = StridedHostKernelAdapter::default();
    for block in &blocks {
        kernels.tensoradd_strided_checked(
            destination_data,
            source_data,
            &block.shape,
            &block.destination_strides,
            &block.source_strides,
            block.destination_offset,
            block.source_offset,
            source.storage_conjugate(),
            D::one(),
            D::one(),
        )?;
    }
    Ok(())
}

/// One logical block's shape and `isize` strides against one operand, in the
/// form [`StridedHostKernelAdapter::tensoradd_strided_checked`] takes.
///
/// Extent-one axes are dropped: they move no element and reach no extra
/// offset. Why not keep them: these inline buffers would then spill at a lower
/// rank than the adapter's own normalization scratch, which drops them too.
/// Every axis' stride is still converted, so an unrepresentable stride is an
/// error before any write, as it was per element before.
///
/// Past eight non-unit axes the buffers spill to the heap: three per operand,
/// once per op and reused across blocks, on top of the adapter's own three.
/// The per-element kernel this replaced allocated nothing at any rank; the
/// count is pinned in `tenet/tests/adjoint_view_allocations.rs`.
#[derive(Default)]
pub(crate) struct CheckedBlockAxes {
    shape: SmallVec<[usize; 8]>,
    destination_strides: SmallVec<[isize; 8]>,
    source_strides: SmallVec<[isize; 8]>,
}

impl CheckedBlockAxes {
    pub(crate) fn fill(
        &mut self,
        shape: &[usize],
        destination_strides: &[usize],
        source: FusionOperand<'_>,
        source_strides: &[usize],
    ) -> Result<(), OperationError> {
        let stride = |stride: usize| {
            isize::try_from(stride).map_err(|_| OperationError::ElementCountOverflow)
        };
        self.shape.clear();
        self.destination_strides.clear();
        self.source_strides.clear();
        for (axis, &extent) in shape.iter().enumerate() {
            let destination = stride(destination_strides[axis])?;
            let source = stride(source_strides[source.storage_axis(axis)?])?;
            if extent != 1 {
                self.shape.push(extent);
                self.destination_strides.push(destination);
                self.source_strides.push(source);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tensoradd<D>(
        &self,
        kernels: &mut StridedHostKernelAdapter,
        destination_data: &mut [D],
        source_data: &[D],
        destination_offset: isize,
        source_offset: isize,
        source_conjugate: bool,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
    {
        kernels.tensoradd_strided_checked(
            destination_data,
            source_data,
            &self.shape,
            &self.destination_strides,
            &self.source_strides,
            destination_offset,
            source_offset,
            source_conjugate,
            alpha,
            beta,
        )
    }
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
    let mut kernels = StridedHostKernelAdapter::default();
    let mut lhs_axes = CheckedBlockAxes::default();
    let mut rhs_axes = CheckedBlockAxes::default();
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
            .block(lhs.storage_block_index(logical_key)?)?;
        let rhs_block = rhs
            .storage_space()
            .structure()
            .block(rhs.storage_block_index(logical_key)?)?;
        // Both sources' layouts are converted before the first write of the
        // block, whichever coefficients are zero.
        lhs_axes.fill(
            destination_block.shape(),
            destination_block.strides(),
            lhs,
            lhs_block.strides(),
        )?;
        rhs_axes.fill(
            destination_block.shape(),
            destination_block.strides(),
            rhs,
            rhs_block.strides(),
        )?;
        let destination_offset = checked_offset(destination_block.offset())?;
        let lhs_offset = checked_offset(lhs_block.offset())?;
        let rhs_offset = checked_offset(rhs_block.offset())?;
        if !alpha.is_zero() {
            lhs_axes.tensoradd(
                &mut kernels,
                destination_data,
                lhs_data,
                destination_offset,
                lhs_offset,
                lhs.storage_conjugate(),
                alpha,
                D::zero(),
            )?;
        }
        if !beta.is_zero() {
            rhs_axes.tensoradd(
                &mut kernels,
                destination_data,
                rhs_data,
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

/// Quantum-dimension-weighted oriented inner product.
///
/// `sector_weight` supplies `dim(c)` per logical block, as the provider's own
/// `f64` scalar: both the per-block sums and the weighted total accumulate in
/// [`WideScalar::Wide`], so a single-precision payload neither narrows the
/// weight nor sums the products in single precision. The result is narrowed to
/// the payload type once, at the end. For `f64`/`Complex64` `Wide` is the
/// payload type and every step is the arithmetic this function always did.
#[doc(hidden)]
pub fn oriented_fusion_inner<D>(
    logical: &BlockStructure,
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
    mut sector_weight: impl FnMut(SectorId) -> f64,
) -> Result<D, OperationError>
where
    D: WideScalar,
{
    oriented_fusion_inner_with(logical, lhs, lhs_data, rhs, rhs_data, |sector| {
        Ok::<_, OperationError>(sector_weight(sector))
    })
}

/// Fallible variant of [`oriented_fusion_inner`] for checked providers.
#[doc(hidden)]
pub fn oriented_fusion_inner_with<D, E>(
    logical: &BlockStructure,
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
    mut sector_weight: impl FnMut(SectorId) -> Result<f64, E>,
) -> Result<D, E>
where
    D: WideScalar,
    E: From<OperationError>,
{
    if lhs_data.len()
        != lhs
            .storage_space()
            .required_len()
            .map_err(OperationError::from)?
        || rhs_data.len()
            != rhs
                .storage_space()
                .required_len()
                .map_err(OperationError::from)?
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented inner storage",
        }
        .into());
    }
    validate_oriented_fusion_layout(logical, lhs)?;
    validate_oriented_fusion_layout(logical, rhs)?;
    let mut total = D::Wide::zero();
    for logical_index in 0..logical.block_count() {
        let logical_block = logical.block(logical_index).map_err(OperationError::from)?;
        let BlockKey::FusionTree(logical_key) = logical_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented inner logical layout",
            }
            .into());
        };
        let lhs_block = lhs
            .storage_space()
            .structure()
            .block(lhs.storage_block_index(logical_key)?)
            .map_err(OperationError::from)?;
        let rhs_block = rhs
            .storage_space()
            .structure()
            .block(rhs.storage_block_index(logical_key)?)
            .map_err(OperationError::from)?;
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
        // `coefficient_as_data` then `*`, not `scale_by_coefficient`: for a
        // complex accumulator the former is `partial * (w + 0i)`, the exact
        // expression this loop used before the accumulator was widened, so
        // `Complex64` keeps its results bit for bit — including the signed
        // zeros and non-finite cases where the two forms differ.
        let weight = <D::Wide as RecouplingCoefficientAction<f64>>::coefficient_as_data(
            sector_weight(logical_key.codomain_tree().coupled())?,
        );
        total = total + partial * weight;
    }
    Ok(D::narrow(total))
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

    /// Both Z2 sectors sliced to the same destination range, sorted by id, as
    /// every scatter caller passes it.
    fn sliced_both_sectors() -> [(SectorId, Range<usize>); 2] {
        [(SectorId::new(0), 1..2), (SectorId::new(1), 1..2)]
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
            &[Some(&sliced_both_sectors()[..]), None],
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
                &[Some(&sliced_both_sectors()[..]), None],
            ),
            Err(OperationError::StructureMismatch { .. })
        ));
        assert_eq!(output, before);
    }

    #[test]
    fn per_sector_tables_reject_a_missing_sector_instead_of_starting_at_zero() {
        // A table that names only one of the two Z2 sectors must not let the
        // other block through with an implicit start of 0.
        let destination = two_key_destination();
        let source = source_with_shapes(&destination, &[vec![1, 2], vec![1, 2]]);
        let mut output = vec![7.0; destination.required_len().unwrap()];
        let before = output.clone();
        assert!(matches!(
            fusion_scatter_add_assign(
                &destination,
                &mut output,
                source.structure(),
                FusionOperand::direct(&source),
                &[1.0, 2.0, 3.0, 4.0],
                &[Some(&[(SectorId::new(0), 1..2)][..]), None],
            ),
            Err(OperationError::StructureMismatch { .. })
        ));
        assert_eq!(output, before);

        let mut restricted = vec![0.0; source.required_len().unwrap()];
        assert!(matches!(
            oriented_fusion_restrict_into(
                source.structure(),
                &mut restricted,
                FusionOperand::direct(&destination_operand()),
                &[0.0; 12],
                &[Some(&[(SectorId::new(0), 1)][..]), None],
            ),
            Err(OperationError::StructureMismatch { .. })
        ));
    }

    /// The full two-sector space, read as a restriction source.
    fn destination_operand() -> crate::DynamicFusionMapSpace {
        bind_source(two_key_destination(), &[vec![3, 2], vec![3, 2]])
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
                &[Some(&sliced_both_sectors()[..]), None],
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
                |_| 1.0,
            )
            .unwrap(),
            expected_inner
        );
        assert_eq!(
            oriented_fusion_inner_with(
                logical.structure(),
                direct_operand,
                &direct,
                adjoint_operand,
                &parent,
                |_| Ok::<_, OperationError>(1.0),
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
                |_| 1.0,
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
