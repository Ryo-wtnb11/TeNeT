use num_traits::Zero;
use std::sync::Arc;
use tenet_core::{Placement, ScratchStorage, SimilarStorage};

use crate::{host_scratch::HostScratchBuffer, OperationError, ReportsPlacement};

use super::dynamic_space::DynamicFusionMapSpace;

/// Host scratch tensor for dynamic fusion-space lowering.
///
/// The buffer is host-owned scratch storage and exposes host slices.
#[derive(Clone, Debug)]
pub(crate) struct HostDynamicFusionScratch<T> {
    space: Arc<DynamicFusionMapSpace>,
    data: HostScratchBuffer<T>,
}

pub(crate) type DynamicFusionScratch<T> = HostDynamicFusionScratch<T>;

impl<T> HostDynamicFusionScratch<T>
where
    T: Clone + Zero,
{
    pub(crate) fn zeroed(space: Arc<DynamicFusionMapSpace>) -> Result<Self, OperationError> {
        let len = space.required_len()?;
        Ok(Self {
            space,
            data: HostScratchBuffer::filled(len, T::zero()),
        })
    }

    pub(crate) fn fill_zero(&mut self) {
        self.data.fill(T::zero());
    }

    /// Re-points this scratch at a different space, reusing the existing
    /// buffer's capacity instead of allocating a fresh one. `resize_filled`
    /// only reallocates when the new length exceeds the current capacity, so a
    /// slot cycled through many contraction shapes grows once to the largest
    /// size seen and is then reused — collapsing the per-eval realloc / page
    /// churn (madvise/bzero) that a fresh allocation on every space change
    /// caused. The whole buffer is zeroed since `resize_filled` only fills the
    /// grown tail.
    pub(crate) fn reset(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
    ) -> Result<(), OperationError> {
        let len = space.required_len()?;
        self.space = space;
        self.data.resize_filled(len, T::zero());
        self.data.fill(T::zero());
        Ok(())
    }
}

impl<T> HostDynamicFusionScratch<T> {
    #[inline]
    pub(crate) fn space(&self) -> &DynamicFusionMapSpace {
        self.space.as_ref()
    }

    #[inline]
    pub(crate) fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

impl<T> ReportsPlacement for HostDynamicFusionScratch<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

/// Host scratch workspace for dynamic fusion-space lowering.
///
/// Device lowering should use a separate device scratch workspace.
#[derive(Clone, Debug)]
pub(crate) struct HostDynamicFusionScratchWorkspace<T> {
    lhs: Option<DynamicFusionScratch<T>>,
    rhs: Option<DynamicFusionScratch<T>>,
    dst: Option<DynamicFusionScratch<T>>,
}

pub(crate) type DynamicFusionScratchWorkspace<T> = HostDynamicFusionScratchWorkspace<T>;

impl<T> Default for HostDynamicFusionScratchWorkspace<T> {
    fn default() -> Self {
        Self {
            lhs: None,
            rhs: None,
            dst: None,
        }
    }
}

impl<T> HostDynamicFusionScratchWorkspace<T>
where
    T: Clone + Zero,
{
    pub(crate) fn prepare_lhs(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.lhs, space)
    }

    pub(crate) fn prepare_rhs(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.rhs, space)
    }

    pub(crate) fn prepare_dst(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.dst, space)
    }

    pub(crate) fn lhs(&self) -> &DynamicFusionScratch<T> {
        self.lhs
            .as_ref()
            .expect("lhs dynamic scratch prepared before replay")
    }

    pub(crate) fn rhs(&self) -> &DynamicFusionScratch<T> {
        self.rhs
            .as_ref()
            .expect("rhs dynamic scratch prepared before replay")
    }

    pub(crate) fn dst(&self) -> &DynamicFusionScratch<T> {
        self.dst
            .as_ref()
            .expect("dst dynamic scratch prepared before replay")
    }

    pub(crate) fn lhs_rhs(&self) -> (&DynamicFusionScratch<T>, &DynamicFusionScratch<T>) {
        (self.lhs(), self.rhs())
    }

    pub(crate) fn lhs_rhs_dst_mut(
        &mut self,
    ) -> (
        &DynamicFusionScratch<T>,
        &DynamicFusionScratch<T>,
        &mut DynamicFusionScratch<T>,
    ) {
        let Self { lhs, rhs, dst } = self;
        (
            lhs.as_ref()
                .expect("lhs dynamic scratch prepared before replay"),
            rhs.as_ref()
                .expect("rhs dynamic scratch prepared before replay"),
            dst.as_mut()
                .expect("dst dynamic scratch prepared before replay"),
        )
    }
}

impl<T> ReportsPlacement for HostDynamicFusionScratchWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

/// Storage-origin scratch tensor for dynamic fusion-space lowering.
///
/// The buffer type is the `SimilarStorage::Similar` of the storage the scratch
/// was allocated from, so the allocation origin (and therefore placement) is
/// carried in the type instead of being fixed to a host buffer.
#[derive(Clone, Debug)]
pub(crate) struct StorageDynamicFusionScratch<Buf> {
    space: Arc<DynamicFusionMapSpace>,
    data: Buf,
}

impl<Buf> StorageDynamicFusionScratch<Buf> {
    pub(crate) fn from_storage<T, S>(
        space: Arc<DynamicFusionMapSpace>,
        storage: &S,
        zero: T,
    ) -> Result<Self, OperationError>
    where
        T: Clone,
        S: SimilarStorage<T, Similar = Buf>,
    {
        let len = space.required_len()?;
        Ok(Self {
            space,
            data: storage.similar_filled(len, zero),
        })
    }

    pub(crate) fn reset_from_storage<T, S>(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
        storage: &S,
        zero: T,
    ) -> Result<(), OperationError>
    where
        T: Clone,
        S: SimilarStorage<T, Similar = Buf>,
        Buf: ScratchStorage<T>,
    {
        let len = space.required_len()?;
        self.space = space;
        self.data.reset_filled(len, zero);
        debug_assert_eq!(self.data.placement(), storage.placement());
        Ok(())
    }

    #[inline]
    pub(crate) fn space(&self) -> &DynamicFusionMapSpace {
        self.space.as_ref()
    }

    #[inline]
    pub(crate) fn buffer(&self) -> &Buf {
        &self.data
    }

    #[inline]
    pub(crate) fn buffer_mut(&mut self) -> &mut Buf {
        &mut self.data
    }
}

/// Storage-origin scratch workspace for dynamic fusion-space lowering.
///
/// Each slot is allocated from the storage of the operand it lowers: LHS
/// core scratch from LHS storage, RHS core scratch from RHS storage,
/// and core destination scratch from destination storage. The dynamic
/// fusion-space cache stays placement-neutral; these buffers are execution-time
/// allocations.
#[derive(Clone, Debug)]
pub(crate) struct StorageDynamicFusionScratchWorkspace<LhsScratch, RhsScratch, DstScratch> {
    lhs: Option<StorageDynamicFusionScratch<LhsScratch>>,
    rhs: Option<StorageDynamicFusionScratch<RhsScratch>>,
    dst: Option<StorageDynamicFusionScratch<DstScratch>>,
}

impl<LhsScratch, RhsScratch, DstScratch> Default
    for StorageDynamicFusionScratchWorkspace<LhsScratch, RhsScratch, DstScratch>
{
    fn default() -> Self {
        Self {
            lhs: None,
            rhs: None,
            dst: None,
        }
    }
}

impl<LhsScratch, RhsScratch, DstScratch>
    StorageDynamicFusionScratchWorkspace<LhsScratch, RhsScratch, DstScratch>
{
    pub(crate) fn prepare_lhs_from_storage<T, S>(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
        storage: &S,
        zero: T,
    ) -> Result<&mut StorageDynamicFusionScratch<LhsScratch>, OperationError>
    where
        T: Clone,
        S: SimilarStorage<T, Similar = LhsScratch>,
        LhsScratch: ScratchStorage<T>,
    {
        match &mut self.lhs {
            Some(scratch)
                if scratch.buffer().placement() == storage.placement()
                    && (Arc::ptr_eq(&scratch.space, &space)
                        || scratch.space.as_ref() == space.as_ref()) =>
            {
                scratch.reset_from_storage(space, storage, zero)?;
            }
            _ => {
                self.lhs = Some(StorageDynamicFusionScratch::from_storage(
                    space, storage, zero,
                )?);
            }
        }
        Ok(self
            .lhs
            .as_mut()
            .expect("lhs storage dynamic scratch prepared before return"))
    }

    pub(crate) fn prepare_rhs_from_storage<T, S>(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
        storage: &S,
        zero: T,
    ) -> Result<&mut StorageDynamicFusionScratch<RhsScratch>, OperationError>
    where
        T: Clone,
        S: SimilarStorage<T, Similar = RhsScratch>,
        RhsScratch: ScratchStorage<T>,
    {
        match &mut self.rhs {
            Some(scratch)
                if scratch.buffer().placement() == storage.placement()
                    && (Arc::ptr_eq(&scratch.space, &space)
                        || scratch.space.as_ref() == space.as_ref()) =>
            {
                scratch.reset_from_storage(space, storage, zero)?;
            }
            _ => {
                self.rhs = Some(StorageDynamicFusionScratch::from_storage(
                    space, storage, zero,
                )?);
            }
        }
        Ok(self
            .rhs
            .as_mut()
            .expect("rhs storage dynamic scratch prepared before return"))
    }

    pub(crate) fn prepare_dst_from_storage<T, S>(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
        storage: &S,
        zero: T,
    ) -> Result<&mut StorageDynamicFusionScratch<DstScratch>, OperationError>
    where
        T: Clone,
        S: SimilarStorage<T, Similar = DstScratch>,
        DstScratch: ScratchStorage<T>,
    {
        match &mut self.dst {
            Some(scratch)
                if scratch.buffer().placement() == storage.placement()
                    && (Arc::ptr_eq(&scratch.space, &space)
                        || scratch.space.as_ref() == space.as_ref()) =>
            {
                scratch.reset_from_storage(space, storage, zero)?;
            }
            _ => {
                self.dst = Some(StorageDynamicFusionScratch::from_storage(
                    space, storage, zero,
                )?);
            }
        }
        Ok(self
            .dst
            .as_mut()
            .expect("dst storage dynamic scratch prepared before return"))
    }

    pub(crate) fn dst(&self) -> &StorageDynamicFusionScratch<DstScratch> {
        self.dst
            .as_ref()
            .expect("dst storage dynamic scratch prepared before replay")
    }

    pub(crate) fn lhs_rhs(
        &self,
    ) -> (
        &StorageDynamicFusionScratch<LhsScratch>,
        &StorageDynamicFusionScratch<RhsScratch>,
    ) {
        (
            self.lhs
                .as_ref()
                .expect("lhs storage dynamic scratch prepared before replay"),
            self.rhs
                .as_ref()
                .expect("rhs storage dynamic scratch prepared before replay"),
        )
    }

    pub(crate) fn lhs_rhs_dst_mut(
        &mut self,
    ) -> (
        &StorageDynamicFusionScratch<LhsScratch>,
        &StorageDynamicFusionScratch<RhsScratch>,
        &mut StorageDynamicFusionScratch<DstScratch>,
    ) {
        let Self { lhs, rhs, dst } = self;
        (
            lhs.as_ref()
                .expect("lhs storage dynamic scratch prepared before replay"),
            rhs.as_ref()
                .expect("rhs storage dynamic scratch prepared before replay"),
            dst.as_mut()
                .expect("dst storage dynamic scratch prepared before replay"),
        )
    }
}

fn prepare_scratch_slot<T>(
    slot: &mut Option<DynamicFusionScratch<T>>,
    space: Arc<DynamicFusionMapSpace>,
) -> Result<&mut DynamicFusionScratch<T>, OperationError>
where
    T: Clone + Zero,
{
    match slot {
        Some(scratch)
            if Arc::ptr_eq(&scratch.space, &space) || scratch.space.as_ref() == space.as_ref() =>
        {
            scratch.fill_zero();
        }
        Some(scratch) => {
            scratch.reset(space)?;
        }
        None => {
            *slot = Some(DynamicFusionScratch::zeroed(space)?);
        }
    }
    Ok(slot
        .as_mut()
        .expect("dynamic scratch slot prepared before return"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::{
        BlockStructure, FusionTensorMapSpace, FusionTreeHomSpace, SectorId, TensorMapSpace,
    };

    fn scratch_space(len: usize) -> Arc<DynamicFusionMapSpace> {
        let dense_space = TensorMapSpace::<1, 0>::from_dims([len], []).unwrap();
        let homspace = FusionTreeHomSpace::from_sectors(
            [(SectorId::new(0), len)],
            std::iter::empty::<(SectorId, usize)>(),
        );
        let structure = BlockStructure::packed_column_major(1, [vec![len]]).unwrap();
        let fusion_space = FusionTensorMapSpace::new(dense_space, homspace, structure).unwrap();
        Arc::new(DynamicFusionMapSpace::from_typed(&fusion_space))
    }

    #[test]
    fn dynamic_fusion_scratch_workspace_is_explicit_host_workspace() {
        let workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        let alias = DynamicFusionScratchWorkspace::<f64>::default();

        assert_eq!(workspace.placement(), Placement::Host);
        assert!(workspace.is_host_placement());
        assert_eq!(alias.placement(), Placement::Host);
    }

    #[test]
    fn dynamic_fusion_scratch_reuse_zeros_existing_buffer() {
        let space = scratch_space(3);
        let mut workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        {
            let scratch = workspace.prepare_lhs(space.clone()).unwrap();
            scratch.data_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        }

        let scratch = workspace.prepare_lhs(space).unwrap();

        assert_eq!(scratch.data(), &[0.0, 0.0, 0.0]);
    }
}
