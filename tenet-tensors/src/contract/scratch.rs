use num_traits::Zero;
use std::sync::Arc;
use tenet_core::Placement;

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

    /// Re-points an overwrite-only source scratch at a different space while
    /// preserving initialized storage and filling only a newly grown tail.
    pub(crate) fn reset_for_overwrite(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
    ) -> Result<(), OperationError> {
        let len = space.required_len()?;
        self.space = space;
        self.data.resize_filled(len, T::zero());
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

    #[inline]
    fn retained_bytes(&self) -> usize {
        self.data
            .capacity()
            .saturating_mul(std::mem::size_of::<T>())
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
        prepare_overwrite_scratch_slot(&mut self.lhs, space)
    }

    pub(crate) fn prepare_rhs(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_overwrite_scratch_slot(&mut self.rhs, space)
    }

    pub(crate) fn prepare_dst(
        &mut self,
        space: Arc<DynamicFusionMapSpace>,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_overwrite_scratch_slot(&mut self.dst, space)
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

    pub(crate) fn optional_sources_dst_mut(
        &mut self,
    ) -> (
        Option<&DynamicFusionScratch<T>>,
        Option<&DynamicFusionScratch<T>>,
        &mut DynamicFusionScratch<T>,
    ) {
        // Why not branch over four borrow combinations: optional immutable
        // source slots let the caller select both views around one mutable dst.
        let Self { lhs, rhs, dst } = self;
        (
            lhs.as_ref(),
            rhs.as_ref(),
            dst.as_mut()
                .expect("dst dynamic scratch prepared before replay"),
        )
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        [&self.lhs, &self.rhs, &self.dst]
            .into_iter()
            .flatten()
            .fold(0usize, |bytes, scratch| {
                bytes.saturating_add(scratch.retained_bytes())
            })
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    #[cfg(test)]
    pub(crate) fn dst_data_mut(&mut self) -> Option<&mut [T]> {
        self.dst.as_mut().map(DynamicFusionScratch::data_mut)
    }
}

impl<T> ReportsPlacement for HostDynamicFusionScratchWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

fn prepare_overwrite_scratch_slot<T>(
    slot: &mut Option<DynamicFusionScratch<T>>,
    space: Arc<DynamicFusionMapSpace>,
) -> Result<&mut DynamicFusionScratch<T>, OperationError>
where
    T: Clone + Zero,
{
    // Why not clear reused storage: every caller immediately runs the explicit
    // overwrite replay, which writes every logical destination itself.
    match slot {
        Some(scratch)
            if Arc::ptr_eq(&scratch.space, &space) || scratch.space.as_ref() == space.as_ref() => {}
        Some(scratch) => {
            scratch.reset_for_overwrite(space)?;
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
        let fusion_space =
            FusionTensorMapSpace::new_unbound(dense_space, homspace, structure).unwrap();
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
    fn dynamic_fusion_scratch_same_shape_reuse_keeps_initialized_contents() {
        // What: overwrite-only source scratch preserves initialized warm storage.
        let space = scratch_space(3);
        let mut workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        {
            let scratch = workspace.prepare_lhs(space.clone()).unwrap();
            scratch.data_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        }

        let scratch = workspace.prepare_lhs(space).unwrap();

        assert_eq!(scratch.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn dynamic_fusion_scratch_growth_initializes_only_the_new_tail() {
        // What: growing overwrite scratch preserves its prefix and initializes its new tail.
        let mut workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        workspace
            .prepare_lhs(scratch_space(3))
            .unwrap()
            .data_mut()
            .copy_from_slice(&[1.0, 2.0, 3.0]);

        let scratch = workspace.prepare_lhs(scratch_space(5)).unwrap();

        assert_eq!(scratch.data(), &[1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn dynamic_fusion_destination_scratch_reuse_preserves_initialized_contents() {
        // What: preparation skips the whole-buffer clear; the core replay owns
        // strong-zero initialization of active and inactive blocks.
        let space = scratch_space(3);
        let mut workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        workspace
            .prepare_dst(space.clone())
            .unwrap()
            .data_mut()
            .fill(f64::NAN);

        let scratch = workspace.prepare_dst(space).unwrap();

        assert!(scratch.data().iter().all(|value| value.is_nan()));
    }

    #[test]
    fn dynamic_fusion_scratch_retained_bytes_use_capacity_and_clear_releases_it() {
        let mut workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        workspace.prepare_lhs(scratch_space(3)).unwrap();
        workspace.prepare_rhs(scratch_space(5)).unwrap();
        workspace.prepare_dst(scratch_space(7)).unwrap();
        let expected: usize = [&workspace.lhs, &workspace.rhs, &workspace.dst]
            .into_iter()
            .flatten()
            .map(HostDynamicFusionScratch::retained_bytes)
            .sum();

        assert_eq!(workspace.retained_bytes(), expected);
        assert!(expected >= (3 + 5 + 7) * std::mem::size_of::<f64>());
        workspace.clear();
        assert_eq!(workspace.retained_bytes(), 0);
    }
}
