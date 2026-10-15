use super::*;
use crate::transform_plan::TreeTransformSpecGroup;

/// One plan's coefficients, shared by every binding of that plan.
///
/// Holds the Single scalars and the Multi matrices as the specs' own `Arc`s
/// (spec order, which is also `coefficient_start` order). A Multi block
/// names its matrix by ordinal, the number of Multi specs before it, which
/// is the same in every binding, so binding an existing plan is one `Arc`
/// bump.
#[derive(Debug)]
pub(crate) struct TreeTransformCoefficients<T> {
    pub(super) singles: Vec<T>,
    pub(super) matrices: Vec<Arc<[T]>>,
    pub(super) len: usize,
    /// Type-erased identity of this payload: a device upload of it is keyed
    /// on a `Weak` of this marker, which pins the address while the key
    /// lives, so a freed payload's address can never alias a new one. Only a
    /// device keys on it, so a host-only build allocates none.
    #[cfg(feature = "cuda")]
    identity: Arc<()>,
    /// The ids of the cached groups a shared group plan built this payload
    /// from, in order; empty for any other producer. Ids, not `Weak`s: a
    /// `Weak` would keep an evicted group's allocation alive (#2251), while
    /// an id is never reused, so it identifies the group as exactly as a
    /// pinned address did (#1753) and retains nothing.
    sources: Vec<u64>,
}

// The identity and provenance are bookkeeping, never part of the value.
impl<T: PartialEq> PartialEq for TreeTransformCoefficients<T> {
    fn eq(&self, other: &Self) -> bool {
        self.singles == other.singles && self.matrices == other.matrices && self.len == other.len
    }
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
        let (mut single_count, mut matrix_count) = (0usize, 0usize);
        for (dst_count, src_count, _, _) in specs.clone() {
            if dst_count == 1 && src_count == 1 {
                single_count += 1;
            } else {
                matrix_count += 1;
            }
        }
        // The ordinal a Multi block stores is a `u32`.
        u32::try_from(matrix_count).map_err(|_| OperationError::ElementCountOverflow)?;
        let mut coefficients = Self {
            singles: Vec::with_capacity(single_count),
            matrices: Vec::with_capacity(matrix_count),
            len: single_count,
            #[cfg(feature = "cuda")]
            identity: Arc::new(()),
            sources: Vec::new(),
        };
        for (dst_count, src_count, values, shared) in specs {
            if dst_count == 1 && src_count == 1 {
                if let Some(&value) = values.first() {
                    coefficients.singles.push(value);
                }
            } else {
                coefficients.len = coefficients
                    .len
                    .checked_add(values.len())
                    .ok_or(OperationError::ElementCountOverflow)?;
                coefficients
                    .matrices
                    .push(shared.cloned().unwrap_or_else(|| Arc::from(values)));
            }
        }
        Ok(coefficients)
    }
}

impl<T> TreeTransformCoefficients<T> {
    /// Records the cached groups this payload was built from, in order.
    pub(crate) fn built_from(mut self, groups: &[Arc<TreeTransformSpecGroup<T>>]) -> Self {
        self.sources = groups.iter().map(|group| group.id()).collect();
        self
    }

    /// Whether this payload was built from exactly `groups`, in order. Exact:
    /// group ids are never reused, so equal ids are the same group.
    pub(crate) fn is_built_from(&self, groups: &[Arc<TreeTransformSpecGroup<T>>]) -> bool {
        self.sources.len() == groups.len()
            && self
                .sources
                .iter()
                .zip(groups)
                .all(|(&source, group)| source == group.id())
    }

    #[inline]
    #[cfg(feature = "cuda")]
    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.identity
    }

    #[inline]
    pub(super) fn matrix(&self, ordinal: u32) -> Option<&[T]> {
        self.matrices.get(ordinal as usize).map(|matrix| &**matrix)
    }

    /// Conservative charge: the shared matrices are charged in full for every
    /// bound structure, as the flattened payload was before.
    pub(super) fn charged_bytes(&self) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
        let element = core::mem::size_of::<T>();
        self.matrices.iter().fold(
            ARC_CONTROL_BYTES
                .saturating_add(core::mem::size_of::<Self>())
                // The identity marker's own allocation.
                .saturating_add(if cfg!(feature = "cuda") {
                    ARC_CONTROL_BYTES
                } else {
                    0
                })
                .saturating_add(
                    self.sources
                        .capacity()
                        .saturating_mul(core::mem::size_of::<u64>()),
                )
                .saturating_add(self.singles.capacity().saturating_mul(element))
                .saturating_add(
                    self.matrices
                        .capacity()
                        .saturating_mul(core::mem::size_of::<Arc<[T]>>()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_payload_never_matches_a_group_created_after_its_source_dropped() {
        type Specs<'a> = std::iter::Empty<(usize, usize, &'a [f64], Option<&'a Arc<[f64]>>)>;
        let source = Arc::new(TreeTransformSpecGroup::<f64>::new(Vec::new()));
        let payload = TreeTransformCoefficients::from_specs(Specs::default())
            .unwrap()
            .built_from(std::slice::from_ref(&source));
        assert!(payload.is_built_from(std::slice::from_ref(&source)));
        drop(source);

        // What: the payload pins nothing, yet a group allocated afterwards,
        // possibly at the freed address, is not taken for its source.
        let later = Arc::new(TreeTransformSpecGroup::<f64>::new(Vec::new()));
        assert!(!payload.is_built_from(std::slice::from_ref(&later)));
        assert!(!payload.is_built_from(&[]));
    }

    #[test]
    fn multi_matrices_are_addressed_by_ordinal_with_no_per_layout_table() {
        // Two Multi specs around a Single, with wide layouts that a
        // per-layout slot table would have charged for.
        let wide = vec![1.0_f64; 64 * 32];
        let small = [2.0_f64; 4];
        let specs = [
            (64usize, 32usize, wide.as_slice(), None),
            (1, 1, &[3.0][..], None),
            (2, 2, &small[..], None),
        ];
        let coefficients = TreeTransformCoefficients::from_specs(specs.iter().copied()).unwrap();

        // What: ordinal `k` is the k-th Multi spec's matrix; the charge is the
        // scalars, the matrix handles and the matrices, nothing per layout.
        assert_eq!(coefficients.matrix(0), Some(wide.as_slice()));
        assert_eq!(coefficients.matrix(1), Some(&small[..]));
        assert_eq!(coefficients.matrix(2), None);
        let arc_control = 2 * core::mem::size_of::<usize>();
        let element = core::mem::size_of::<f64>();
        assert_eq!(
            coefficients.charged_bytes(),
            arc_control
                + core::mem::size_of::<TreeTransformCoefficients<f64>>()
                + if cfg!(feature = "cuda") {
                    arc_control
                } else {
                    0
                }
                + coefficients.singles.capacity() * element
                + coefficients.matrices.capacity() * core::mem::size_of::<Arc<[f64]>>()
                + (arc_control + wide.len() * element)
                + (arc_control + small.len() * element)
        );
    }
}
