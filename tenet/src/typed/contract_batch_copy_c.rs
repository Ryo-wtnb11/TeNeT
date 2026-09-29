//! CopyC: one direct temporary contraction, then one completed output transform.

use super::*;
use tenet_operations::{
    admit_tree_transform_members_overwrite_raw, tree_transform_members_overwrite_raw,
    StridedHostKernelAdapter, TreeTransformStructure, TreeTransformWorkspace,
};

pub(super) struct CopyCGeometryBinding<R> {
    pub(super) geometry: crate::typed::checked_generic_contract::CopyCGeometry,
    pub(super) temporary_space: BoundDynamicFusionMapSpace<R>,
}

impl<R> CopyCGeometryBinding<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    pub(super) fn new<D: TensorScalar>(
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        spec: &crate::typed::ContractSpec<'_>,
        output_axes: &[usize],
        orientation: tenet_tensors::FusionContractOrientation,
    ) -> Result<Self, Error> {
        let geometry = crate::typed::checked_generic_contract::CopyCGeometry::new(
            orientation,
            lhs.space.space().rank(),
            rhs.space.space().rank(),
            spec.lhs,
            spec.rhs,
            output_axes,
            spec.codomain.len(),
        );
        let (first, second, first_axes, second_axes) =
            geometry.oriented(lhs, rhs, spec.lhs, spec.rhs);
        let temporary_space = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &first.space,
            &second.space,
            first_axes,
            second_axes,
            OutputAxisOrder::identity(),
        )?;
        Ok(Self {
            geometry,
            temporary_space,
        })
    }
}

pub(super) struct CopyCPlan<R> {
    pub(super) temporary_space: BoundDynamicFusionMapSpace<R>,
    pub(super) transform: Arc<TreeTransformStructure<f64>>,
    pub(super) input_swapped: bool,
}

pub(super) struct CopyCWorkspace<D> {
    temporary: Vec<D>,
    transform: TreeTransformWorkspace<D>,
}

impl<D> Default for CopyCWorkspace<D> {
    fn default() -> Self {
        Self {
            temporary: Vec::new(),
            transform: TreeTransformWorkspace::default(),
        }
    }
}

impl<D> CopyCWorkspace<D> {
    pub(super) fn retained_bytes(&self) -> usize {
        self.temporary
            .capacity()
            .saturating_mul(std::mem::size_of::<D>())
            .saturating_add(self.transform.retained_bytes())
    }
}

impl<R> CopyCPlan<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    pub(super) fn run<D: TensorScalar>(
        &self,
        plan: &ContractPlan<R, D>,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut [D],
        members: usize,
        workspace: &mut ContractWorkspace<R, D>,
    ) -> Result<(), Error> {
        let copy = workspace.copy_c.as_mut().ok_or_else(|| {
            Error::InvalidArgument("copyC workspace belongs to another plan".into())
        })?;
        let temporary_len = self.temporary_space.space().required_len()?;
        let total = plan.total_len(temporary_len, members)?;
        // Admission includes all transform views, coefficients and packed jobs.
        // It precedes even the temporary core submission.
        admit_tree_transform_members_overwrite_raw(
            &mut copy.transform,
            &self.transform,
            plan.space.space().structure(),
            self.temporary_space.space().structure(),
            dst.len(),
            total,
            members,
        )?;
        if workspace
            .replay
            .as_ref()
            .is_none_or(|(replay, _)| replay.members() != members)
        {
            workspace.replay = plan.resolution.stacked_direct_host_replay(members)?;
        }
        let (replay, swapped) = workspace
            .replay
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("copyC temporary is not a direct core".into()))?;
        let (left, right) = if self.input_swapped ^ *swapped {
            (rhs, lhs)
        } else {
            (lhs, rhs)
        };
        let left =
            StackedStorageView::new::<D>(&left.storage, left.member_len, members, left.member_len)?;
        let right = StackedStorageView::new::<D>(
            &right.storage,
            right.member_len,
            members,
            right.member_len,
        )?;
        copy.temporary.resize(total, D::from_real(0.0));
        let mut temporary = StackedStorageViewMut::new::<D>(
            &mut copy.temporary,
            temporary_len,
            members,
            temporary_len,
        )?;
        let mut lease = plan.runtime.lease_context()?;
        let lane = lease.context().multiplicity_free_lane::<D>()?;
        lane.execute_stacked_direct_host(replay, &mut temporary, &left, &right, true)?;
        let backend = lane.tree_context_mut().backend_mut();
        let threads = backend.recoupling_threads();
        tree_transform_members_overwrite_raw(
            &mut StridedHostKernelAdapter::default(),
            backend.dense_mut(),
            &mut copy.transform,
            &self.transform,
            plan.space.space().structure(),
            self.temporary_space.space().structure(),
            dst,
            &copy.temporary,
            members,
            threads,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::{U1FusionRule, U1Irrep};
    use crate::typed::{ContractSpec, GradedSpace, TensorMap};
    use std::sync::Arc;

    #[test]
    fn poisoned_temporary_and_output_are_overwritten_on_warm_replay() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 2),
                (U1Irrep::new(2), 1),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 55).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 56).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1, 0],
            domain: &[2],
        };
        let expected = a.contract(&b, &spec).unwrap();
        let left = StackedTensorMap::pack(&[&a, &a]).unwrap();
        let right = StackedTensorMap::pack(&[&b, &b]).unwrap();
        let plan = ContractPlan::new(&left, &right, &spec).unwrap();
        assert!(plan.copy_c.is_some());
        let mut workspace = plan.workspace();
        plan.execute(&left, &right, &mut workspace).unwrap();
        let copy = workspace.copy_c.as_mut().unwrap();
        assert!(copy.temporary.contains(&0.0));
        copy.temporary.fill(f64::NAN);
        let poisoned = expected.scale(f64::NAN);
        let mut dst = StackedTensorMap::pack(&[&poisoned, &poisoned]).unwrap();
        plan.execute_into(&left, &right, &mut dst, &mut workspace)
            .unwrap();
        for i in 0..2 {
            for (&actual, &reference) in dst
                .member(i)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
            {
                let tolerance = 128.0 * 64.0_f64.sqrt() * f64::EPSILON * reference.abs().max(1.0);
                assert!((actual - reference).abs() <= tolerance);
            }
        }
        assert!(workspace
            .copy_c
            .as_ref()
            .unwrap()
            .temporary
            .iter()
            .all(|x| x.is_finite()));
    }
}
