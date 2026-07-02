use tenet_core::SimilarStorage;

/// Crate-internal same-placement scratch allocator.
///
/// This is the operations-layer adapter around `SimilarStorage`; it allocates
/// storage from an existing storage reference but does not expose host slices or
/// choose an execution backend.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SamePlacementScratchAllocator;

impl SamePlacementScratchAllocator {
    #[inline]
    pub(crate) fn filled_like<T, S>(&self, storage: &S, len: usize, value: T) -> S::Similar
    where
        T: Clone,
        S: SimilarStorage<T>,
    {
        storage.similar_filled(len, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::{Placement, TensorStorage};

    #[derive(Debug)]
    struct PlacementOnlyStorage<T> {
        len: usize,
        _marker: std::marker::PhantomData<T>,
    }

    #[derive(Debug, PartialEq)]
    struct PlacementOnlyScratch<T> {
        data: Vec<T>,
    }

    impl<T> PlacementOnlyStorage<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<T> TensorStorage<T> for PlacementOnlyStorage<T> {
        fn len(&self) -> usize {
            self.len
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    impl<T: Clone> SimilarStorage<T> for PlacementOnlyStorage<T> {
        type Similar = PlacementOnlyScratch<T>;

        fn similar_filled(&self, len: usize, value: T) -> Self::Similar
        where
            T: Clone,
        {
            PlacementOnlyScratch {
                data: vec![value; len],
            }
        }
    }

    impl<T> TensorStorage<T> for PlacementOnlyScratch<T> {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    #[test]
    fn same_placement_allocator_uses_similar_storage_type_without_host_slices() {
        let storage = PlacementOnlyStorage::<i32>::new(2);
        let allocator = SamePlacementScratchAllocator;

        let scratch = allocator.filled_like(&storage, 3, 7);

        assert_eq!(scratch.data, vec![7, 7, 7]);
        assert_eq!(scratch.len(), 3);
        assert_eq!(scratch.placement(), storage.placement());
    }
}
