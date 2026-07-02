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

/// Source/destination scratch shape for tree-transform replay.
///
/// The cached tree-transform structure remains placement-neutral; these slots
/// name the storage-local scratch buffers needed to replay source packs into
/// destination packs. Current host replay instantiates both slots with
/// `HostScratchBuffer<T>`.
#[derive(Clone, Debug)]
pub(crate) struct TreeTransformScratchBuffers<Source, Destination> {
    source: Source,
    destination: Destination,
}

impl<Source, Destination> Default for TreeTransformScratchBuffers<Source, Destination>
where
    Source: Default,
    Destination: Default,
{
    fn default() -> Self {
        Self {
            source: Source::default(),
            destination: Destination::default(),
        }
    }
}

impl<Source, Destination> TreeTransformScratchBuffers<Source, Destination> {
    #[inline]
    pub(crate) fn from_parts(source: Source, destination: Destination) -> Self {
        Self {
            source,
            destination,
        }
    }

    #[inline]
    pub(crate) fn source(&self) -> &Source {
        &self.source
    }

    #[inline]
    pub(crate) fn source_mut(&mut self) -> &mut Source {
        &mut self.source
    }

    #[inline]
    pub(crate) fn destination(&self) -> &Destination {
        &self.destination
    }

    #[inline]
    pub(crate) fn destination_mut(&mut self) -> &mut Destination {
        &mut self.destination
    }

    #[inline]
    pub(crate) fn source_and_destination_mut(&mut self) -> (&Source, &mut Destination) {
        (&self.source, &mut self.destination)
    }
}

/// Storage-aware tree-transform replay workspace.
///
/// This is the crate-private production boundary for TensorKit-style
/// `similar(src.data, ...)` / `similar(dst.data, ...)` scratch allocation. It
/// still feeds the host-slice replay kernels; it does not imply device replay.
#[derive(Clone, Debug)]
pub(crate) struct StorageTreeTransformWorkspace<SourceScratch, DestinationScratch> {
    zero_strides: Vec<isize>,
    packed: Option<TreeTransformScratchBuffers<SourceScratch, DestinationScratch>>,
}

impl<SourceScratch, DestinationScratch> Default
    for StorageTreeTransformWorkspace<SourceScratch, DestinationScratch>
{
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            packed: None,
        }
    }
}

impl<SourceScratch, DestinationScratch>
    StorageTreeTransformWorkspace<SourceScratch, DestinationScratch>
{
    pub(crate) fn prepare_from_storages<T, DSrc, DDst>(
        &mut self,
        src_storage: &DSrc,
        dst_storage: &DDst,
        source_len: usize,
        destination_len: usize,
        zero: T,
    ) where
        T: Clone,
        DSrc: SimilarStorage<T, Similar = SourceScratch>,
        DDst: SimilarStorage<T, Similar = DestinationScratch>,
    {
        self.packed = Some(TreeTransformScratchBuffers::from_parts(
            src_storage.similar_filled(source_len, zero.clone()),
            dst_storage.similar_filled(destination_len, zero),
        ));
    }

    #[inline]
    pub(crate) fn zero_strides_mut(&mut self) -> &mut Vec<isize> {
        &mut self.zero_strides
    }

    #[inline]
    pub(crate) fn replay_parts_mut(
        &mut self,
    ) -> (
        &mut Vec<isize>,
        &mut TreeTransformScratchBuffers<SourceScratch, DestinationScratch>,
    ) {
        (
            &mut self.zero_strides,
            self.packed
                .as_mut()
                .expect("storage tree-transform scratch prepared before replay"),
        )
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

    #[test]
    fn tree_transform_scratch_buffers_keep_source_and_destination_slots() {
        let buffers = TreeTransformScratchBuffers::from_parts(
            PlacementOnlyScratch { data: vec![1, 2] },
            PlacementOnlyScratch {
                data: vec![3, 4, 5],
            },
        );

        assert_eq!(buffers.source().data, vec![1, 2]);
        assert_eq!(buffers.destination().data, vec![3, 4, 5]);
        assert_eq!(buffers.source().len(), 2);
        assert_eq!(buffers.destination().len(), 3);
    }
}
