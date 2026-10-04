//! Member-batched device replay of a Host-compiled `DynamicTree` route whose
//! transforms are unconjugated nonzero `Single` moves.
//!
//! Each member follows the eager device sequence of
//! [`execute_dynamic_tree_on_cuda`](super::cuda): source transforms (the
//! fermionic twist folded into the move writing each twisted block), the
//! fully direct core GEMMs, then the output transform. The batch appends a
//! member axis to every move region and GEMM, so the submission count does not
//! depend on `B`. The Host member authority is
//! `execute_dynamic_tree_execution_artifact_members_host`.

use std::sync::Arc;

use tenet_core::BlockStructure;
use tenet_dense::{cuda_region_zero, CudaDenseContext, CudaDenseStorage, CudaRegion, CudaScalar};
use tenet_operations::cuda::{CudaStackedStorageGemm, CudaStorage};
use tenet_operations::cuda_transform::{CudaMemberZeroRegions, CudaSingleMemberRegions};
use tenet_operations::stacked::{StackedStorageView, StackedStorageViewMut};

use super::DynamicTreeExecutionArtifact;
use crate::contract::resolution::{StorageContractResolution, StorageContractRoute};
use crate::{OperationError, RecouplingCoefficientAction};

/// Caller-owned device state of one member-batched `DynamicTree` replay at
/// one `B`: validated member regions, scaled-move coefficients, and the
/// transformed-source and core-destination stacks. `S` is the plan's device
/// storage, so the holder needs no payload bound.
#[doc(hidden)]
pub struct CudaDynamicTreeMembersWorkspace<S> {
    /// The artifact and `B` everything below was prepared for.
    artifact: Option<Arc<DynamicTreeExecutionArtifact<f64>>>,
    members: usize,
    /// lhs source, rhs source and output transform regions.
    stages: [Option<CudaSingleMemberRegions>; 3],
    /// Scaled-move coefficients per stage; structure-only, kept across `B`.
    coefficients: [Option<CudaDenseStorage>; 3],
    /// Transformed lhs, transformed rhs and core destination stacks.
    buffers: [Option<S>; 3],
    /// Core inactive blocks of a directly written destination, per member.
    core_zeros: Vec<CudaRegion>,
}

impl<S> Default for CudaDynamicTreeMembersWorkspace<S> {
    fn default() -> Self {
        Self {
            artifact: None,
            members: 0,
            stages: [None, None, None],
            coefficients: [None, None, None],
            buffers: [None, None, None],
            core_zeros: Vec::new(),
        }
    }
}

fn artifact(
    resolution: &StorageContractResolution<f64>,
) -> Result<&Arc<DynamicTreeExecutionArtifact<f64>>, OperationError> {
    match &resolution.route {
        StorageContractRoute::DynamicTree(artifact) => Ok(artifact),
        _ => Err(OperationError::UnsupportedTensorContractScope {
            message: "CUDA member contraction requires a transformed-tree route",
        }),
    }
}

/// `(borrowed, scales)` of the lhs and rhs source stages.
fn source_scales(artifact: &DynamicTreeExecutionArtifact<f64>) -> [(bool, &[(usize, f64)]); 2] {
    let [lhs, rhs] = artifact.stage_scales();
    [(artifact.lhs_borrowed, lhs), (artifact.rhs_borrowed, rhs)]
}

impl StorageContractResolution<f64> {
    /// Checks, without a device, that this route has a CUDA member replay,
    /// and returns the device plan entries one workspace holds: distinct core
    /// GEMM shapes, core zero fills, and each stage's zero fills and scaled
    /// moves.
    #[doc(hidden)]
    pub fn admit_cuda_dynamic_tree_members(&self) -> Result<usize, OperationError> {
        let artifact = artifact(self)?;
        self.admit_cuda_inactive_regions()?;
        artifact.block_plan.require_identity_direct_replay()?;
        let mut entries = artifact.block_plan.cuda_direct_plan_entries();
        let transforms = [&artifact.lhs_transform, &artifact.rhs_transform];
        for ((borrowed, scales), transform) in source_scales(artifact).into_iter().zip(transforms) {
            if borrowed {
                continue;
            }
            entries +=
                CudaSingleMemberRegions::admit_scaled(&transform.transform_structure, scales)?;
        }
        if let Some(output) = &artifact.core_dst {
            entries += CudaSingleMemberRegions::admit(&output.output_transform_structure)?;
        }
        Ok(entries)
    }
}

fn stack_len(member_len: usize, members: usize) -> Result<usize, OperationError> {
    member_len
        .checked_mul(members)
        .ok_or(OperationError::ElementCountOverflow)
}

impl<D> CudaDynamicTreeMembersWorkspace<CudaStorage<D>>
where
    D: CudaScalar + RecouplingCoefficientAction<f64>,
{
    /// Retained device payload bytes plus Host region metadata.
    pub fn retained_bytes(&self) -> usize {
        let elements = self
            .buffers
            .iter()
            .flatten()
            .map(|buffer| buffer.0.len())
            .chain(
                self.coefficients
                    .iter()
                    .flatten()
                    .map(CudaDenseStorage::len),
            )
            .fold(0usize, usize::saturating_add);
        let regions = self
            .stages
            .iter()
            .flatten()
            .map(CudaSingleMemberRegions::retained_bytes)
            .chain(self.core_zeros.iter().map(CudaRegion::retained_heap_bytes))
            .fold(0usize, usize::saturating_add);
        elements
            .saturating_mul(std::mem::size_of::<D>())
            .saturating_add(regions)
    }

    /// Overwrites `dst` with each member's contraction. `dst_zeroed` means
    /// `dst` is a fresh zero stack, so core inactive blocks need no fill.
    /// Every structural and region check finishes before the first device
    /// submission; a warm call at an unchanged `B` transfers nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        ctx: &mut CudaDenseContext,
        resolution: &StorageContractResolution<f64>,
        dst_structure: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
        members: usize,
        dst_zeroed: bool,
    ) -> Result<(), OperationError> {
        let artifact = artifact(resolution)?;
        for (storage, structure) in [
            (&*dst, dst_structure),
            (lhs, &artifact.lhs_transform.replay_structure),
            (rhs, &artifact.rhs_transform.replay_structure),
        ] {
            let expected = stack_len(structure.required_len()?, members)?;
            if storage.0.len() != expected {
                return Err(OperationError::ElementCountMismatch {
                    expected,
                    actual: storage.0.len(),
                });
            }
        }
        if self.members != members
            || !self
                .artifact
                .as_ref()
                .is_some_and(|prepared| Arc::ptr_eq(prepared, artifact))
        {
            self.prepare(ctx, artifact, dst_structure, members)?;
        }
        let [lhs_stage, rhs_stage, output_stage] = &self.stages;
        let [lhs_buffer, rhs_buffer, core_buffer] = &mut self.buffers;
        for (stage, coefficients, source, buffer) in [
            (lhs_stage, &self.coefficients[0], lhs, &mut *lhs_buffer),
            (rhs_stage, &self.coefficients[1], rhs, &mut *rhs_buffer),
        ] {
            if let (Some(stage), Some(buffer)) = (stage, buffer) {
                stage.execute_overwrite(ctx, source, coefficients.as_ref(), buffer)?;
            }
        }
        let physical_lhs = lhs_buffer.as_ref().unwrap_or(lhs);
        let physical_rhs = rhs_buffer.as_ref().unwrap_or(rhs);
        let lens = |transform: &super::DynamicFusionTransformedSourceEntry<f64>| {
            transform.space.required_len()
        };
        let (lhs_len, rhs_len) = (
            lens(&artifact.lhs_transform)?,
            lens(&artifact.rhs_transform)?,
        );
        let ((left, left_len), (right, right_len)) =
            artifact.core_order((physical_lhs, lhs_len), (physical_rhs, rhs_len));
        let core_buffer_len = artifact
            .core_dst
            .as_ref()
            .map_or(Ok(0), |output| output.space.required_len())?;
        let left = StackedStorageView::new::<D>(left, left_len, members, left_len)?;
        let right = StackedStorageView::new::<D>(right, right_len, members, right_len)?;
        let (core, core_len) = match core_buffer {
            // A workspace core stack is born zero and only the core GEMMs
            // write it, so its inactive blocks stay zero without a fill.
            Some(buffer) => (buffer, core_buffer_len),
            None => {
                if !dst_zeroed {
                    for region in &self.core_zeros {
                        cuda_region_zero::<D>(ctx, &mut dst.0, region)
                            .map_err(OperationError::Dense)?;
                    }
                }
                (&mut *dst, dst_structure.required_len()?)
            }
        };
        let mut core = StackedStorageViewMut::new::<D>(core, core_len, members, core_len)?;
        artifact.block_plan.execute_direct_on_storage_prezeroed(
            &mut CudaStackedStorageGemm::new(ctx),
            &mut core,
            &left,
            &right,
        )?;
        if let (Some(stage), Some(buffer)) = (output_stage, core_buffer.as_ref()) {
            stage.execute_overwrite(ctx, buffer, self.coefficients[2].as_ref(), dst)?;
        }
        Ok(())
    }

    fn prepare(
        &mut self,
        ctx: &mut CudaDenseContext,
        artifact: &Arc<DynamicTreeExecutionArtifact<f64>>,
        dst_structure: &Arc<BlockStructure>,
        members: usize,
    ) -> Result<(), OperationError> {
        if !self
            .artifact
            .as_ref()
            .is_some_and(|prepared| Arc::ptr_eq(prepared, artifact))
        {
            self.coefficients = [None, None, None];
        }
        if members == 0 {
            return Err(OperationError::InvalidArgument {
                message: "a CUDA member contraction needs at least one member",
            });
        }
        let dst_len = dst_structure.required_len()?;
        let core_dst_structure = artifact.core_dst_structure(dst_structure);
        let core_len = core_dst_structure.required_len()?;
        let (left, right) = artifact.core_order(&artifact.lhs_transform, &artifact.rhs_transform);
        artifact.block_plan.validate_replay_structures(
            core_dst_structure,
            left.space.structure(),
            right.space.structure(),
        )?;
        let mut stages = [None, None, None];
        let mut zero_len = 0;
        let transforms = [&artifact.lhs_transform, &artifact.rhs_transform];
        for (index, ((borrowed, scales), transform)) in source_scales(artifact)
            .into_iter()
            .zip(transforms)
            .enumerate()
        {
            if borrowed {
                continue;
            }
            let core = transform.space.required_len()?;
            let source = transform.replay_structure.required_len()?;
            stages[index] = Some(CudaSingleMemberRegions::prepare_scaled(
                &transform.transform_structure,
                transform.space.structure(),
                &transform.replay_structure,
                core,
                source,
                stack_len(core, members)?,
                stack_len(source, members)?,
                members,
                scales,
            )?);
        }
        if let Some(output) = &artifact.core_dst {
            stages[2] = Some(CudaSingleMemberRegions::prepare(
                &output.output_transform_structure,
                dst_structure,
                core_dst_structure,
                dst_len,
                core_len,
                stack_len(dst_len, members)?,
                stack_len(core_len, members)?,
                members,
            )?);
        }
        let mut core_zeros = Vec::new();
        if artifact.core_dst.is_none() {
            let zeros = CudaMemberZeroRegions::prepare(
                artifact.block_plan.inactive_destination_regions(),
                dst_len,
                members,
            )?;
            zero_len = zero_len.max(zeros.max_zero_len());
            core_zeros = zeros.into_regions();
        }
        let mut any_zero = !core_zeros.is_empty();
        for stage in stages.iter().flatten() {
            zero_len = zero_len.max(stage.max_zero_len());
            any_zero |= !stage.zeros().is_empty();
        }
        ctx.reserve_zero_template::<D>(stack_len(zero_len, members)?)
            .map_err(OperationError::Dense)?;
        if any_zero {
            ctx.reserve_ones_template::<D>(1)
                .map_err(OperationError::Dense)?;
        }
        for (slot, stage) in self.coefficients.iter_mut().zip(&stages) {
            if let Some(stage) = stage.as_ref().filter(|stage| stage.has_scaled_moves()) {
                if slot.is_none() {
                    *slot = Some(
                        CudaStorage::<D>::upload_owned(ctx, stage.scaled_coefficients::<D>())?.0,
                    );
                }
            }
        }
        // Fresh device stacks cost one zero upload each (#740); their
        // contents are irrelevant except a core stack's inactive blocks.
        let zeros = |len: usize| -> Result<CudaStorage<D>, OperationError> {
            CudaStorage::upload_members(ctx, vec![D::ZERO; stack_len(len, members)?], len, members)
        };
        let buffers = [
            (!artifact.lhs_borrowed)
                .then(|| zeros(artifact.lhs_transform.space.required_len()?))
                .transpose()?,
            (!artifact.rhs_borrowed)
                .then(|| zeros(artifact.rhs_transform.space.required_len()?))
                .transpose()?,
            artifact
                .core_dst
                .is_some()
                .then(|| zeros(core_len))
                .transpose()?,
        ];
        self.stages = stages;
        self.buffers = buffers;
        self.core_zeros = core_zeros;
        self.members = members;
        self.artifact = Some(Arc::clone(artifact));
        Ok(())
    }
}
