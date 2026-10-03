use super::*;

#[cfg(test)]
thread_local! {
    static EXACT_STORAGE_FALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_exact_storage_fallback_count() {
    EXACT_STORAGE_FALLBACKS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn exact_storage_fallback_count() -> usize {
    EXACT_STORAGE_FALLBACKS.with(std::cell::Cell::get)
}

/// Checks that every logical block element owns a distinct storage offset.
///
/// General [`BlockStructure`] values may describe aliased read views. Owning
/// symmetric spaces and operation destinations call this explicit admission
/// boundary before publishing writable storage.
///
/// A structure whose constructor recorded
/// [`BlockStructure::storage_tiling_proven`] is admitted in O(1); every other
/// layout, including expert and imported ones, keeps the exact check.
#[doc(hidden)]
pub fn validate_block_storage_injective(structure: &BlockStructure) -> Result<(), CoreError> {
    // Why not re-check the canonical coupled-sector layout: its subblocks
    // interleave inside one sector matrix, so the interval sweep below
    // degenerates to hashing every element, while the tiling proof already
    // states that each offset is reached exactly once.
    if structure.storage_tiling_proven() {
        return Ok(());
    }
    #[derive(Clone, Copy)]
    struct BoundedBlock {
        block: usize,
        start: usize,
        end: usize,
    }

    let mut bounded = Vec::with_capacity(structure.block_count());
    for block in 0..structure.block_count() {
        let layout = structure.block(block)?;
        let Some((start, end)) =
            block_layout_bounds(layout.shape(), layout.strides(), layout.offset())?
        else {
            continue;
        };
        let proven_injective = block_layout_is_proven_injective(layout.shape(), layout.strides());
        if !proven_injective {
            #[cfg(test)]
            EXACT_STORAGE_FALLBACKS.with(|count| count.set(count.get() + 1));
            let mut offsets = FxHashMap::<usize, ()>::default();
            if let Some(offset) = visit_block_layout_offsets(
                layout.shape(),
                layout.strides(),
                layout.offset(),
                |at| offsets.insert(at, ()).is_some(),
            )? {
                return Err(CoreError::OverlappingBlockStorage {
                    first_block: block,
                    second_block: block,
                    offset,
                });
            }
        }
        bounded.push(BoundedBlock { block, start, end });
    }
    bounded.sort_by_key(|entry| entry.start);

    let mut component_start = 0;
    while component_start < bounded.len() {
        let mut component_end = component_start + 1;
        let mut max_end = bounded[component_start].end;
        while component_end < bounded.len() && bounded[component_end].start <= max_end {
            max_end = max_end.max(bounded[component_end].end);
            component_end += 1;
        }
        let component = &mut bounded[component_start..component_end];
        if component.len() == 1 {
            component_start = component_end;
            continue;
        }

        #[cfg(test)]
        EXACT_STORAGE_FALLBACKS.with(|count| count.set(count.get() + 1));
        component.sort_by_key(|entry| entry.block);
        let mut owners = FxHashMap::<usize, usize>::default();
        for entry in component {
            let layout = structure.block(entry.block)?;
            let mut collision = None;
            visit_block_layout_offsets(
                layout.shape(),
                layout.strides(),
                layout.offset(),
                |offset| {
                    if let Some(&first_block) = owners.get(&offset) {
                        collision = Some((first_block, offset));
                        true
                    } else {
                        owners.insert(offset, entry.block);
                        false
                    }
                },
            )?;
            if let Some((first_block, offset)) = collision {
                return Err(CoreError::OverlappingBlockStorage {
                    first_block,
                    second_block: entry.block,
                    offset,
                });
            }
        }
        component_start = component_end;
    }
    Ok(())
}

fn block_layout_is_proven_injective(shape: &[usize], strides: &[usize]) -> bool {
    let mut axes = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|&(extent, _)| extent > 1)
        .collect::<SmallVec<[(usize, usize); 8]>>();
    axes.sort_unstable_by_key(|&(_, stride)| stride);
    let mut lower_span = 0usize;
    for (extent, stride) in axes {
        if stride == 0 || stride <= lower_span {
            return false;
        }
        let Some(span) = (extent - 1).checked_mul(stride) else {
            return false;
        };
        let Some(next_span) = lower_span.checked_add(span) else {
            return false;
        };
        lower_span = next_span;
    }
    true
}

fn block_layout_bounds(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<Option<(usize, usize)>, CoreError> {
    if shape.contains(&0) {
        return Ok(None);
    }
    let end = storage_end_exclusive(shape, strides, offset)?
        .checked_sub(1)
        .ok_or_else(|| CoreError::ElementCountOverflow)?;
    Ok(Some((offset, end)))
}

fn visit_block_layout_offsets<F>(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    mut stop: F,
) -> Result<Option<usize>, CoreError>
where
    F: FnMut(usize) -> bool,
{
    if shape.contains(&0) {
        return Ok(None);
    }
    let mut indices = SmallVec::<[usize; 8]>::from_elem(0, shape.len());
    let mut physical = offset;
    loop {
        if stop(physical) {
            return Ok(Some(physical));
        }

        let mut axis = 0;
        while axis < indices.len() {
            indices[axis] += 1;
            if indices[axis] < shape[axis] {
                physical = physical
                    .checked_add(strides[axis])
                    .ok_or_else(|| CoreError::ElementCountOverflow)?;
                break;
            }
            indices[axis] = 0;
            physical = physical
                .checked_sub(
                    shape[axis]
                        .saturating_sub(1)
                        .checked_mul(strides[axis])
                        .ok_or_else(|| CoreError::ElementCountOverflow)?,
                )
                .ok_or_else(|| CoreError::ElementCountOverflow)?;
            axis += 1;
        }
        if axis == indices.len() {
            return Ok(None);
        }
    }
}
