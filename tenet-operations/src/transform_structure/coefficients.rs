use super::*;

/// One plan's coefficients, shared by every binding of that plan.
///
/// Holds the Single scalars, the Multi matrices as the specs' own `Arc`s
/// (spec order, which is also `coefficient_start` order), and an O(1) index
/// from a Multi block's `dst_layout_start` to its matrix. Every part is
/// layout-independent: compilation pushes `dst_count + src_count` layouts per
/// spec in spec order, so a spec's `dst_layout_start` is the same in every
/// binding, and binding an existing plan is one `Arc` bump.
#[derive(Debug, PartialEq)]
pub(crate) struct TreeTransformCoefficients<T> {
    pub(super) singles: Vec<T>,
    pub(super) matrices: Vec<Arc<[T]>>,
    /// Indexed by `dst_layout_start / 2`. Why halved: every spec owns at least
    /// one destination and one source layout, so two specs never share a slot.
    matrix_slots: Vec<u32>,
    pub(super) len: usize,
}

impl<T: Copy> TreeTransformCoefficients<T> {
    /// Builds the shared payload from `(dst_count, src_count, coefficients,
    /// shared matrix)` per spec in spec order. Malformed specs are stored
    /// as given; compilation rejects them before any read.
    pub(crate) fn from_specs<'a, I>(specs: I) -> Result<Self, OperationError>
    where
        I: Iterator<Item = (usize, usize, &'a [T], Option<&'a Arc<[T]>>)> + Clone,
        T: 'a,
    {
        let (mut single_count, mut matrix_count, mut layout_count) = (0usize, 0usize, 0usize);
        for (dst_count, src_count, _, _) in specs.clone() {
            if dst_count == 1 && src_count == 1 {
                single_count += 1;
            } else {
                matrix_count += 1;
            }
            layout_count = layout_count
                .checked_add(dst_count)
                .and_then(|count| count.checked_add(src_count))
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        let mut coefficients = Self {
            singles: Vec::with_capacity(single_count),
            matrices: Vec::with_capacity(matrix_count),
            matrix_slots: Vec::new(),
            len: single_count,
        };
        if matrix_count != 0 {
            coefficients.matrix_slots = vec![u32::MAX; layout_count.div_ceil(2)];
        }
        let mut layout_start = 0usize;
        for (dst_count, src_count, values, shared) in specs {
            if dst_count == 1 && src_count == 1 {
                if let Some(&value) = values.first() {
                    coefficients.singles.push(value);
                }
            } else {
                let slot = u32::try_from(coefficients.matrices.len())
                    .map_err(|_| OperationError::ElementCountOverflow)?;
                if let Some(entry) = coefficients.matrix_slots.get_mut(layout_start / 2) {
                    *entry = slot;
                }
                coefficients.len = coefficients
                    .len
                    .checked_add(values.len())
                    .ok_or(OperationError::ElementCountOverflow)?;
                coefficients
                    .matrices
                    .push(shared.cloned().unwrap_or_else(|| Arc::from(values)));
            }
            layout_start += dst_count + src_count;
        }
        Ok(coefficients)
    }
}

impl<T> TreeTransformCoefficients<T> {
    #[inline]
    pub(super) fn matrix(&self, dst_layout_start: usize) -> Option<&[T]> {
        let slot = *self.matrix_slots.get(dst_layout_start / 2)?;
        self.matrices.get(slot as usize).map(|matrix| &**matrix)
    }

    /// Conservative charge: the shared matrices are charged in full for every
    /// bound structure, as the flattened payload was before.
    pub(super) fn charged_bytes(&self) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
        let element = core::mem::size_of::<T>();
        self.matrices.iter().fold(
            ARC_CONTROL_BYTES
                .saturating_add(core::mem::size_of::<Self>())
                .saturating_add(self.singles.capacity().saturating_mul(element))
                .saturating_add(
                    self.matrices
                        .capacity()
                        .saturating_mul(core::mem::size_of::<Arc<[T]>>()),
                )
                .saturating_add(
                    self.matrix_slots
                        .capacity()
                        .saturating_mul(core::mem::size_of::<u32>()),
                ),
            |bytes, matrix| {
                bytes
                    .saturating_add(ARC_CONTROL_BYTES)
                    .saturating_add(matrix.len().saturating_mul(element))
            },
        )
    }
}

/// Whether a binding's spec has no same-length entry in the shared payload it
/// was given; the payload is built from the same specs, so this is a
/// structural check, not a value comparison.
pub(super) fn shared_coefficient_mismatch<T>(shared: Option<&[T]>, spec: &[T]) -> bool {
    shared.is_none_or(|shared| shared.len() != spec.len())
}

pub(super) fn shared_payload_mismatch() -> OperationError {
    OperationError::InvalidArgument {
        message: "tree transform plan coefficients disagree with its specs",
    }
}

/// Upper bound of [`TreeTransformCoefficients`]' own heap bytes for `specs`
/// (matrices excluded: their `Arc`s belong to the specs), for a cache that
/// charges a plan before its first binding builds the payload.
pub(crate) fn charged_shared_coefficient_bytes<T>(
    spec_shapes: impl Iterator<Item = (usize, usize)>,
) -> usize {
    const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
    spec_shapes.fold(
        ARC_CONTROL_BYTES.saturating_add(core::mem::size_of::<TreeTransformCoefficients<T>>()),
        |bytes, (dst_count, src_count)| {
            bytes
                .saturating_add(core::mem::size_of::<T>().max(core::mem::size_of::<Arc<[T]>>()))
                .saturating_add(
                    dst_count
                        .saturating_add(src_count)
                        .div_ceil(2)
                        .saturating_add(1)
                        .saturating_mul(core::mem::size_of::<u32>()),
                )
        },
    )
}
