use core::mem::MaybeUninit;
use core::ops::{Add, Mul};

use num_traits::Zero;
use tenet_core::BlockStructure;

use crate::owned_overwrite_buffer::initialize_owned;
use crate::{ConjugateValue, OperationError, RecouplingCoefficientAction};

const OWNED_TRACE_TILE_ELEMENTS: usize = 256;

#[cfg(test)]
thread_local! {
    static TEST_WRITE_BITMAP: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// Borrowed producer geometry for the internal owned trace writer.
///
/// Destination addresses and physical coverage are derived from the supplied
/// block structure; this type carries only source traversal and coefficient
/// data compiled by `tenet-tensors`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct OwnedTraceTerm<'a, C> {
    dst_block: usize,
    src_block: usize,
    output_shape: &'a [usize],
    trace_shape: &'a [usize],
    src_output_strides: &'a [isize],
    src_trace_strides: &'a [isize],
    coefficient: C,
}

impl<'a, C> OwnedTraceTerm<'a, C> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dst_block: usize,
        src_block: usize,
        output_shape: &'a [usize],
        trace_shape: &'a [usize],
        src_output_strides: &'a [isize],
        src_trace_strides: &'a [isize],
        coefficient: C,
    ) -> Self {
        Self {
            dst_block,
            src_block,
            output_shape,
            trace_shape,
            src_output_strides,
            src_trace_strides,
            coefficient,
        }
    }
}

struct OwnedTracePlan<'s, 't, C, F> {
    dst_structure: &'s BlockStructure,
    src_structure: &'s BlockStructure,
    required_len: usize,
    source_conjugate: bool,
    producer_indices: &'s [usize],
    producer_offsets: &'s [usize],
    term_at: F,
    _term: core::marker::PhantomData<&'t C>,
}

impl<'s, 't, C: Copy + 't, F> OwnedTracePlan<'s, 't, C, F>
where
    F: Fn(usize) -> OwnedTraceTerm<'t, C>,
{
    #[allow(clippy::too_many_arguments)]
    fn compile(
        dst_structure: &'s BlockStructure,
        dst_nout: usize,
        src_structure: &'s BlockStructure,
        src_len: usize,
        source_conjugate: bool,
        term_count: usize,
        producer_indices: &'s [usize],
        producer_offsets: &'s [usize],
        term_at: F,
    ) -> Result<Option<Self>, OperationError> {
        let required_len = dst_structure
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        let expected_src_len = src_structure
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        if src_len != expected_src_len {
            return Err(OperationError::ElementCountMismatch {
                expected: expected_src_len,
                actual: src_len,
            });
        }

        let Some(regions) = dst_structure
            .coupled_sector_regions(dst_nout)
            .map_err(OperationError::from_core_preserving_context)?
        else {
            return Ok(None);
        };
        let mut covered_end = 0usize;
        for region in regions.iter() {
            let range = region.range();
            if range.start != covered_end || range.end > required_len {
                return Ok(None);
            }
            covered_end = range.end;
        }
        if covered_end != required_len {
            return Ok(None);
        }

        let block_count = dst_structure.block_count();
        let expected_offsets = block_count
            .checked_add(1)
            .ok_or(OperationError::ElementCountOverflow)?;
        if producer_offsets.len() != expected_offsets {
            return Err(OperationError::InvalidArgument {
                message: "invalid owned trace producer offsets",
            });
        }
        if producer_indices.len() != term_count {
            return Err(OperationError::CoefficientCountMismatch {
                expected: term_count,
                actual: producer_indices.len(),
            });
        }
        let mut next_producer = 0usize;
        for block_index in 0..block_count {
            let start = producer_offsets[block_index];
            let end = producer_offsets[block_index + 1];
            if start != next_producer || end > producer_indices.len() {
                return Err(OperationError::InvalidArgument {
                    message: "invalid owned trace producer partition",
                });
            }
            let mut previous = None;
            for &term_index in &producer_indices[start..end] {
                if term_index >= term_count || previous.is_some_and(|known| term_index <= known) {
                    return Err(OperationError::InvalidArgument {
                        message: "invalid owned trace producer order",
                    });
                }
                let term = term_at(term_index);
                if term.dst_block != block_index {
                    return Err(OperationError::StructureMismatch {
                        tensor: "trace destination",
                    });
                }
                previous = Some(term_index);
            }
            next_producer = end;
        }
        if next_producer != producer_indices.len() {
            return Err(OperationError::InvalidArgument {
                message: "incomplete owned trace producer partition",
            });
        }

        for term_index in 0..term_count {
            validate_term(&term_at(term_index), dst_structure, src_structure, src_len)?;
        }

        Ok(Some(Self {
            dst_structure,
            src_structure,
            required_len,
            source_conjugate,
            producer_indices,
            producer_offsets,
            term_at,
            _term: core::marker::PhantomData,
        }))
    }

    fn execute<D>(&self, src: &[D], alpha: D) -> Result<Vec<D>, OperationError>
    where
        D: Copy
            + Add<D, Output = D>
            + Mul<D, Output = D>
            + Zero
            + ConjugateValue
            + RecouplingCoefficientAction<C>,
    {
        // Why not size scratch to the largest block: that can become another
        // output-sized allocation and zero pass. This fixed tile is reused for
        // every block and bounds stack use independently of tensor geometry.
        let mut tile = [D::zero(); OWNED_TRACE_TILE_ELEMENTS];
        initialize_owned(self.required_len, |dst| {
            self.write_tiles(dst, &mut tile, src, alpha);
            Ok(())
        })
    }

    fn write_tiles<D>(
        &self,
        dst: &mut [MaybeUninit<D>],
        tile: &mut [D; OWNED_TRACE_TILE_ELEMENTS],
        src: &[D],
        alpha: D,
    ) where
        D: Copy
            + Add<D, Output = D>
            + Mul<D, Output = D>
            + Zero
            + ConjugateValue
            + RecouplingCoefficientAction<C>,
    {
        for block_index in 0..self.dst_structure.block_count() {
            let block = self
                .dst_structure
                .block(block_index)
                .expect("owned trace destination blocks were preflighted");
            let element_count = block
                .element_count()
                .expect("owned trace destination sizes were preflighted");
            for tile_start in (0..element_count).step_by(OWNED_TRACE_TILE_ELEMENTS) {
                let tile_end = element_count.min(tile_start + OWNED_TRACE_TILE_ELEMENTS);
                let active_tile = &mut tile[..tile_end - tile_start];
                active_tile.fill(D::zero());

                let producer_range =
                    self.producer_offsets[block_index]..self.producer_offsets[block_index + 1];
                for &producer_index in &self.producer_indices[producer_range] {
                    let term = (self.term_at)(producer_index);
                    let src_offset = isize::try_from(
                        self.src_structure
                            .block(term.src_block)
                            .expect("owned trace source block was preflighted")
                            .offset(),
                    )
                    .expect("owned trace source offset was preflighted");
                    let trace_len = element_count_infallible(term.trace_shape);
                    for (tile_index, value) in active_tile.iter_mut().enumerate() {
                        let output_linear = tile_start + tile_index;
                        let src_base = strided_offset(
                            output_linear,
                            term.output_shape,
                            term.src_output_strides,
                            src_offset,
                        );
                        let mut sum = D::zero();
                        for trace_linear in 0..trace_len {
                            let src_index = strided_offset(
                                trace_linear,
                                term.trace_shape,
                                term.src_trace_strides,
                                src_base,
                            ) as usize;
                            sum = sum + src[src_index].maybe_conj(self.source_conjugate);
                        }
                        *value = *value + (alpha * sum).scale_by_coefficient(term.coefficient);
                    }
                }

                // Why not append in physical order: one canonical sector matrix
                // interleaves tree blocks by column, requiring a tree-pair
                // lookup and producer restart for every block-column slice.
                for (tile_index, &value) in active_tile.iter().enumerate() {
                    let output_linear = tile_start + tile_index;
                    let dst_index = unsigned_strided_offset(
                        output_linear,
                        block.shape(),
                        block.strides(),
                        block.offset(),
                    );
                    #[cfg(test)]
                    TEST_WRITE_BITMAP.with(|bitmap| {
                        if let Some(bitmap) = bitmap.borrow_mut().as_mut() {
                            bitmap[dst_index] += 1;
                        }
                    });
                    dst[dst_index].write(value);
                }
            }
        }
    }
}

/// Attempts the private-initialization owned CPU trace path.
///
/// `Ok(None)` means canonical physical coverage was not proved and no output
/// allocation occurred. Callers must use their initialized fallback with the
/// already-compiled trace structure.
///
/// Why not require a concrete cross-crate term slice: geometry and
/// coefficients live in separate compiled structures, so materializing one
/// would allocate on every call. `term_at` is a read-only metadata accessor;
/// it never receives output storage, an address writer, or commit control.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn try_tensortrace_owned_raw<'s, 't, D, C, F>(
    dst_structure: &'s BlockStructure,
    dst_nout: usize,
    src_structure: &'s BlockStructure,
    src: &[D],
    source_conjugate: bool,
    term_count: usize,
    producer_indices: &'s [usize],
    producer_offsets: &'s [usize],
    term_at: F,
    alpha: D,
) -> Result<Option<Vec<D>>, OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + Zero
        + ConjugateValue
        + RecouplingCoefficientAction<C>,
    C: Copy + 't,
    F: Fn(usize) -> OwnedTraceTerm<'t, C>,
{
    let Some(plan) = OwnedTracePlan::compile(
        dst_structure,
        dst_nout,
        src_structure,
        src.len(),
        source_conjugate,
        term_count,
        producer_indices,
        producer_offsets,
        term_at,
    )?
    else {
        return Ok(None);
    };
    plan.execute(src, alpha).map(Some)
}

fn validate_term<C>(
    term: &OwnedTraceTerm<'_, C>,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    src_len: usize,
) -> Result<(), OperationError> {
    let dst_block = dst_structure
        .block(term.dst_block)
        .map_err(OperationError::from_core_preserving_context)?;
    let src_block = src_structure
        .block(term.src_block)
        .map_err(OperationError::from_core_preserving_context)?;
    if term.output_shape != dst_block.shape() {
        return Err(OperationError::ShapeMismatch {
            dst: dst_block.shape().to_vec(),
            src: term.output_shape.to_vec(),
        });
    }
    if term.output_shape.len() != term.src_output_strides.len() {
        return Err(OperationError::RankMismatch {
            expected: term.output_shape.len(),
            actual: term.src_output_strides.len(),
        });
    }
    if term.trace_shape.len() != term.src_trace_strides.len() {
        return Err(OperationError::RankMismatch {
            expected: term.trace_shape.len(),
            actual: term.src_trace_strides.len(),
        });
    }
    let output_len = crate::strided::element_count(term.output_shape)?;
    let dst_len = dst_block
        .element_count()
        .map_err(OperationError::from_core_preserving_context)?;
    if output_len != dst_len {
        return Err(OperationError::ElementCountMismatch {
            expected: dst_len,
            actual: output_len,
        });
    }
    let trace_len = crate::strided::element_count(term.trace_shape)?;
    if output_len == 0 || trace_len == 0 {
        return Ok(());
    }
    let src_offset =
        isize::try_from(src_block.offset()).map_err(|_| OperationError::OffsetOverflow {
            value: src_block.offset(),
        })?;
    let (output_min, output_max) =
        strided_bounds(src_offset, term.output_shape, term.src_output_strides)?;
    if output_min < 0 {
        return Err(OperationError::OffsetOverflow { value: usize::MAX });
    }
    let _ = usize::try_from(output_max)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    let (trace_min, trace_max) = strided_bounds(0, term.trace_shape, term.src_trace_strides)?;
    let source_min = output_min
        .checked_add(trace_min)
        .ok_or(OperationError::ElementCountOverflow)?;
    let source_max = output_max
        .checked_add(trace_max)
        .ok_or(OperationError::ElementCountOverflow)?;
    let source_min = usize::try_from(source_min)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    let source_max = usize::try_from(source_max)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    if source_min >= src_len || source_max >= src_len {
        return Err(OperationError::OffsetOverflow { value: source_max });
    }
    Ok(())
}

fn element_count_infallible(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn strided_bounds(
    offset: isize,
    shape: &[usize],
    strides: &[isize],
) -> Result<(isize, isize), OperationError> {
    let mut min = offset;
    let mut max = offset;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if dim <= 1 {
            continue;
        }
        let extent = isize::try_from(dim - 1).map_err(|_| OperationError::ElementCountOverflow)?;
        let span = stride
            .checked_mul(extent)
            .ok_or(OperationError::ElementCountOverflow)?;
        if span < 0 {
            min = min
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        } else {
            max = max
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    Ok((min, max))
}

fn strided_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[isize],
    mut offset: isize,
) -> isize {
    for (&dim, &stride) in shape.iter().zip(strides) {
        let coordinate = linear % dim;
        linear /= dim;
        offset += coordinate as isize * stride;
    }
    offset
}

fn unsigned_strided_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[usize],
    mut offset: usize,
) -> usize {
    for (&dim, &stride) in shape.iter().zip(strides) {
        let coordinate = linear % dim;
        linear /= dim;
        offset += coordinate * stride;
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::{BlockKey, BlockSpec, FusionTreePairKey};

    #[test]
    fn physical_write_bitmap_covers_each_address_once_before_commit() {
        // What: the real tiled scatter writes every physical address once
        // across sector blocks, including one structural-zero destination.
        let key = |sector| {
            BlockKey::from(
                FusionTreePairKey::try_pair_from_sector_ids(
                    [sector],
                    [sector],
                    sector,
                    [false],
                    [false],
                    [],
                    [],
                    [],
                    [],
                )
                .unwrap(),
            )
        };
        let structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::with_key(key(1), vec![20, 20], vec![1, 20], 0).unwrap(),
                BlockSpec::with_key(key(2), vec![20, 20], vec![1, 20], 400).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            structure.coupled_sector_regions(1).unwrap().unwrap().len(),
            2
        );
        let source = (0..800).map(|value| value as f64).collect::<Vec<_>>();
        let producer_indices = [0usize];
        let producer_offsets = [0, 1, 1];
        TEST_WRITE_BITMAP.with(|bitmap| *bitmap.borrow_mut() = Some(vec![0; 800]));

        let output = try_tensortrace_owned_raw(
            &structure,
            1,
            &structure,
            &source,
            false,
            1,
            &producer_indices,
            &producer_offsets,
            |_| OwnedTraceTerm::new(0, 0, &[20, 20], &[], &[1, 20], &[], 1.0),
            1.0,
        )
        .unwrap()
        .unwrap();
        let bitmap = TEST_WRITE_BITMAP.with(|bitmap| bitmap.borrow_mut().take().unwrap());

        assert_eq!(&output[..400], &source[..400]);
        assert_eq!(&output[400..], &[0.0; 400]);
        assert!(bitmap.into_iter().all(|writes| writes == 1));
    }
}
