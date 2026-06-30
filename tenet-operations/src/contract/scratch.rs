use num_traits::Zero;

use crate::OperationError;

use super::dynamic_space::DynamicFusionMapSpace;

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionScratch<T> {
    space: DynamicFusionMapSpace,
    data: Vec<T>,
}

impl<T> DynamicFusionScratch<T>
where
    T: Clone + Zero,
{
    pub(crate) fn zeroed(space: DynamicFusionMapSpace) -> Result<Self, OperationError> {
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

impl<T> DynamicFusionScratch<T> {
    #[inline]
    pub(crate) fn space(&self) -> &DynamicFusionMapSpace {
        &self.space
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

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionScratchWorkspace<T> {
    lhs: Option<DynamicFusionScratch<T>>,
    rhs: Option<DynamicFusionScratch<T>>,
    dst: Option<DynamicFusionScratch<T>>,
}

impl<T> Default for DynamicFusionScratchWorkspace<T> {
    fn default() -> Self {
        Self {
            lhs: None,
            rhs: None,
            dst: None,
        }
    }
}

impl<T> DynamicFusionScratchWorkspace<T>
where
    T: Clone + Zero,
{
    pub(crate) fn prepare_lhs(
        &mut self,
        space: DynamicFusionMapSpace,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.lhs, space)
    }

    pub(crate) fn prepare_rhs(
        &mut self,
        space: DynamicFusionMapSpace,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.rhs, space)
    }

    pub(crate) fn prepare_dst(
        &mut self,
        space: DynamicFusionMapSpace,
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

fn prepare_scratch_slot<T>(
    slot: &mut Option<DynamicFusionScratch<T>>,
    space: DynamicFusionMapSpace,
) -> Result<&mut DynamicFusionScratch<T>, OperationError>
where
    T: Clone + Zero,
{
    match slot {
        Some(scratch) if scratch.space == space => {
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
