use num_traits::Zero;
use std::sync::Arc;
use tenet_core::Placement;

use crate::{OperationError, ReportsPlacement};

use super::dynamic_space::DynamicFusionMapSpace;

/// Host scratch tensor for dynamic fusion-space lowering.
///
/// The buffer is host-owned `Vec<T>` storage and exposes host slices.
#[derive(Clone, Debug)]
pub(crate) struct HostDynamicFusionScratch<T> {
    space: Arc<DynamicFusionMapSpace>,
    data: Vec<T>,
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
            data: vec![T::zero(); len],
        })
    }

    pub(crate) fn fill_zero(&mut self) {
        self.data.fill(T::zero());
    }
}

impl<T> HostDynamicFusionScratch<T> {
    #[inline]
    pub(crate) fn space(&self) -> &DynamicFusionMapSpace {
        self.space.as_ref()
    }

    #[inline]
    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
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
        _ => {
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

    #[test]
    fn dynamic_fusion_scratch_workspace_is_explicit_host_workspace() {
        let workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        let alias = DynamicFusionScratchWorkspace::<f64>::default();

        assert_eq!(workspace.placement(), Placement::Host);
        assert!(workspace.is_host_placement());
        assert_eq!(alias.placement(), Placement::Host);
    }
}
