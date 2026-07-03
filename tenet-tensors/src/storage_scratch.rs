use tenet_core::{ScratchStorage, SimilarStorage};

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
        SourceScratch: ScratchStorage<T>,
        DestinationScratch: ScratchStorage<T>,
    {
        match &mut self.packed {
            Some(buffers)
                if buffers.source().placement() == src_storage.placement()
                    && buffers.destination().placement() == dst_storage.placement() =>
            {
                buffers.source_mut().reset_filled(source_len, zero.clone());
                buffers
                    .destination_mut()
                    .reset_filled(destination_len, zero);
            }
            _ => {
                self.packed = Some(TreeTransformScratchBuffers::from_parts(
                    src_storage.similar_filled(source_len, zero.clone()),
                    dst_storage.similar_filled(destination_len, zero),
                ));
            }
        }
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

/// Storage-aware dense-contraction replay workspace.
///
/// The output scratch is allocated from destination storage, because it is the
/// dense contraction result scattered into the destination tensor layout.
#[derive(Clone, Debug)]
pub(crate) struct StorageTensorContractWorkspace<OutputScratch> {
    output: Option<OutputScratch>,
    zero_strides: Vec<isize>,
}

impl<OutputScratch> Default for StorageTensorContractWorkspace<OutputScratch> {
    fn default() -> Self {
        Self {
            output: None,
            zero_strides: Vec::new(),
        }
    }
}

impl<OutputScratch> StorageTensorContractWorkspace<OutputScratch> {
    pub(crate) fn prepare_from_dst_storage<T, DDst>(
        &mut self,
        dst_storage: &DDst,
        len: usize,
        zero: T,
    ) where
        T: Clone,
        DDst: SimilarStorage<T, Similar = OutputScratch>,
        OutputScratch: ScratchStorage<T>,
    {
        match &mut self.output {
            Some(output) if output.placement() == dst_storage.placement() => {
                output.reset_filled(len, zero)
            }
            _ => self.output = Some(dst_storage.similar_filled(len, zero)),
        }
    }

    #[inline]
    pub(crate) fn replay_parts_mut(&mut self) -> (&mut Vec<isize>, &mut OutputScratch) {
        (
            &mut self.zero_strides,
            self.output
                .as_mut()
                .expect("storage tensor-contract output scratch prepared before replay"),
        )
    }
}

/// LHS/RHS/destination scratch shape for canonical fusion-block contraction.
///
/// This mirrors the TensorKit pack-GEMM-scatter slots: `lhs` and `rhs` hold
/// packed source blocks, while `destination` holds the dense matmul result
/// before scatter.
#[derive(Clone, Debug)]
pub(crate) struct FusionBlockContractScratchBuffers<Lhs, Rhs, Destination> {
    lhs: Lhs,
    rhs: Rhs,
    destination: Destination,
}

impl<Lhs, Rhs, Destination> Default for FusionBlockContractScratchBuffers<Lhs, Rhs, Destination>
where
    Lhs: Default,
    Rhs: Default,
    Destination: Default,
{
    fn default() -> Self {
        Self {
            lhs: Lhs::default(),
            rhs: Rhs::default(),
            destination: Destination::default(),
        }
    }
}

impl<Lhs, Rhs, Destination> FusionBlockContractScratchBuffers<Lhs, Rhs, Destination> {
    #[inline]
    pub(crate) fn from_parts(lhs: Lhs, rhs: Rhs, destination: Destination) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    #[inline]
    pub(crate) fn lhs(&self) -> &Lhs {
        &self.lhs
    }

    #[inline]
    pub(crate) fn lhs_mut(&mut self) -> &mut Lhs {
        &mut self.lhs
    }

    #[inline]
    pub(crate) fn rhs(&self) -> &Rhs {
        &self.rhs
    }

    #[inline]
    pub(crate) fn rhs_mut(&mut self) -> &mut Rhs {
        &mut self.rhs
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
    pub(crate) fn inputs_and_destination_mut(&mut self) -> (&Lhs, &Rhs, &mut Destination) {
        (&self.lhs, &self.rhs, &mut self.destination)
    }
}

/// Storage-aware canonical fusion-block contraction workspace.
///
/// Allocation origins are explicit: LHS pack scratch from LHS storage, RHS pack
/// scratch from RHS storage, and matmul output scratch from destination storage.
#[derive(Clone, Debug)]
pub(crate) struct StorageFusionBlockContractWorkspace<LhsScratch, RhsScratch, DestinationScratch> {
    buffers: Option<FusionBlockContractScratchBuffers<LhsScratch, RhsScratch, DestinationScratch>>,
}

impl<LhsScratch, RhsScratch, DestinationScratch> Default
    for StorageFusionBlockContractWorkspace<LhsScratch, RhsScratch, DestinationScratch>
{
    fn default() -> Self {
        Self { buffers: None }
    }
}

impl<LhsScratch, RhsScratch, DestinationScratch>
    StorageFusionBlockContractWorkspace<LhsScratch, RhsScratch, DestinationScratch>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_from_storages<T, DLhs, DRhs, DDst>(
        &mut self,
        lhs_storage: &DLhs,
        rhs_storage: &DRhs,
        dst_storage: &DDst,
        lhs_len: usize,
        rhs_len: usize,
        destination_len: usize,
        zero: T,
    ) where
        T: Clone,
        DLhs: SimilarStorage<T, Similar = LhsScratch>,
        DRhs: SimilarStorage<T, Similar = RhsScratch>,
        DDst: SimilarStorage<T, Similar = DestinationScratch>,
        LhsScratch: ScratchStorage<T>,
        RhsScratch: ScratchStorage<T>,
        DestinationScratch: ScratchStorage<T>,
    {
        match &mut self.buffers {
            Some(buffers)
                if buffers.lhs().placement() == lhs_storage.placement()
                    && buffers.rhs().placement() == rhs_storage.placement()
                    && buffers.destination().placement() == dst_storage.placement() =>
            {
                buffers.lhs_mut().reset_filled(lhs_len, zero.clone());
                buffers.rhs_mut().reset_filled(rhs_len, zero.clone());
                buffers
                    .destination_mut()
                    .reset_filled(destination_len, zero);
            }
            _ => {
                self.buffers = Some(FusionBlockContractScratchBuffers::from_parts(
                    lhs_storage.similar_filled(lhs_len, zero.clone()),
                    rhs_storage.similar_filled(rhs_len, zero.clone()),
                    dst_storage.similar_filled(destination_len, zero),
                ));
            }
        }
    }

    #[inline]
    pub(crate) fn buffers_mut(
        &mut self,
    ) -> &mut FusionBlockContractScratchBuffers<LhsScratch, RhsScratch, DestinationScratch> {
        self.buffers
            .as_mut()
            .expect("storage fusion-block scratch prepared before replay")
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
