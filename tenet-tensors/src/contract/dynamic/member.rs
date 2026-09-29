use super::*;
use tenet_core::{Placement, TensorStorage};
use tenet_dense::DenseExecutor;
use tenet_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use tenet_operations::{
    admit_tree_transform_members_overwrite_raw, tree_transform_members_overwrite_raw,
    TreeTransformWorkspace,
};

/// Mutable Host payload and replay scratch for one artifact at varying B.
#[derive(Default)]
pub(crate) struct DynamicTreeMembersWorkspace<D, C = f64> {
    lhs: Vec<D>,
    rhs: Vec<D>,
    dst: Vec<D>,
    lhs_transform: TreeTransformWorkspace<D>,
    rhs_transform: TreeTransformWorkspace<D>,
    output_transform: TreeTransformWorkspace<D>,
    core: Option<StackedDirectReplay<C>>,
}

impl<D, C: Copy + PartialEq + num_traits::One> DynamicTreeMembersWorkspace<D, C> {
    pub(crate) fn retained_bytes(&self) -> usize {
        (self.lhs.capacity() + self.rhs.capacity() + self.dst.capacity())
            .saturating_mul(std::mem::size_of::<D>())
            + self
                .core
                .as_ref()
                .map_or(0, StackedDirectReplay::retained_bytes)
            + self.lhs_transform.retained_bytes()
            + self.rhs_transform.retained_bytes()
            + self.output_transform.retained_bytes()
    }
}

struct ReadSlice<'a, D>(&'a [D]);
struct WriteSlice<'a, D>(&'a mut [D]);

impl<D> TensorStorage<D> for ReadSlice<'_, D> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn placement(&self) -> Placement {
        Placement::Host
    }
}
impl<D> tenet_core::HostReadableStorage<D> for ReadSlice<'_, D> {
    fn as_slice(&self) -> &[D] {
        self.0
    }
}
impl<D> TensorStorage<D> for WriteSlice<'_, D> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn placement(&self) -> Placement {
        Placement::Host
    }
}
impl<D> tenet_core::HostReadableStorage<D> for WriteSlice<'_, D> {
    fn as_slice(&self) -> &[D] {
        self.0
    }
}
impl<D> tenet_core::HostWritableStorage<D> for WriteSlice<'_, D> {
    fn as_mut_slice(&mut self) -> &mut [D] {
        self.0
    }
}

fn member_len<D>(structure: &Arc<BlockStructure>, members: usize) -> Result<usize, OperationError> {
    let len = structure.required_len()?;
    let total = len
        .checked_mul(members)
        .ok_or(OperationError::ElementCountOverflow)?;
    core::alloc::Layout::array::<D>(total).map_err(|_| OperationError::ElementCountOverflow)?;
    Ok(len)
}

fn exact_len(actual: usize, member: usize, members: usize) -> Result<(), OperationError> {
    let expected = member
        .checked_mul(members)
        .ok_or(OperationError::ElementCountOverflow)?;
    if actual == expected {
        Ok(())
    } else {
        Err(OperationError::ElementCountMismatch { expected, actual })
    }
}

/// Execute one immutable, twist-free artifact over uniform owned-dense Host
/// payloads. The caller binds semantic tensor metadata; this raw seam admits
/// only the artifact's structures and checked, disjoint member spans.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact_members_host<E, BC, D, C>(
    dense: &mut E,
    backend: &mut BC,
    backend_workspace: &mut BC::Workspace,
    artifact: &DynamicTreeExecutionArtifact<C>,
    dst_structure: &Arc<BlockStructure>,
    workspace: &mut DynamicTreeMembersWorkspace<D, C>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    members: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    BC: TensorContractBackend<D, C>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    if artifact.requires_source_twist() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "member artifact replay does not support source twist",
        });
    }
    let lhs_src = &artifact.lhs_transform.replay_structure;
    let rhs_src = &artifact.rhs_transform.replay_structure;
    let lhs_core = artifact.lhs_transform.space.structure();
    let rhs_core = artifact.rhs_transform.space.structure();
    let core_dst = artifact.core_dst.as_ref();
    let core_dst_structure = core_dst.map_or(dst_structure, |entry| entry.space.structure());
    let lhs_src_len = member_len::<D>(lhs_src, members)?;
    let rhs_src_len = member_len::<D>(rhs_src, members)?;
    let dst_len = member_len::<D>(dst_structure, members)?;
    let lhs_core_len = member_len::<D>(lhs_core, members)?;
    let rhs_core_len = member_len::<D>(rhs_core, members)?;
    let core_dst_len = member_len::<D>(core_dst_structure, members)?;
    exact_len(lhs_data.len(), lhs_src_len, members)?;
    exact_len(rhs_data.len(), rhs_src_len, members)?;
    exact_len(dst_data.len(), dst_len, members)?;

    let replay = StackedDirectReplay::new(Arc::clone(&artifact.block_plan), members)?;
    let (left_len, right_len) = if artifact.orientation == FusionContractOrientation::RhsLhs {
        (rhs_core_len, lhs_core_len)
    } else {
        (lhs_core_len, rhs_core_len)
    };
    let (left_structure, right_structure) =
        if artifact.orientation == FusionContractOrientation::RhsLhs {
            (rhs_core, lhs_core)
        } else {
            (lhs_core, rhs_core)
        };
    replay.plan().validate_replay_structures(
        core_dst_structure,
        left_structure,
        right_structure,
    )?;
    if replay.member_lens() != [core_dst_len, left_len, right_len] {
        return Err(OperationError::InvalidArgument {
            message: "member artifact core structure does not match its replay",
        });
    }
    if !artifact.lhs_borrowed {
        admit_tree_transform_members_overwrite_raw::<D, C>(
            &mut workspace.lhs_transform,
            &artifact.lhs_transform.transform_structure,
            lhs_core,
            lhs_src,
            lhs_core_len * members,
            lhs_data.len(),
            members,
        )?;
    }
    if !artifact.rhs_borrowed {
        admit_tree_transform_members_overwrite_raw::<D, C>(
            &mut workspace.rhs_transform,
            &artifact.rhs_transform.transform_structure,
            rhs_core,
            rhs_src,
            rhs_core_len * members,
            rhs_data.len(),
            members,
        )?;
    }
    if let Some(output) = core_dst {
        admit_tree_transform_members_overwrite_raw::<D, C>(
            &mut workspace.output_transform,
            &output.output_transform_structure,
            dst_structure,
            core_dst_structure,
            dst_data.len(),
            core_dst_len * members,
            members,
        )?;
    }

    // All fallible structural and span checks finish before scratch growth or
    // dense work. The raw binder cannot inspect per-member TensorMap metadata.
    if !artifact.lhs_borrowed {
        workspace.lhs.resize(lhs_core_len * members, D::zero());
    }
    if !artifact.rhs_borrowed {
        workspace.rhs.resize(rhs_core_len * members, D::zero());
    }
    if core_dst.is_some() {
        workspace.dst.resize(core_dst_len * members, D::zero());
    }
    workspace.core = Some(replay);

    let mut kernels = crate::StridedHostKernelAdapter::default();
    if !artifact.lhs_borrowed {
        tree_transform_members_overwrite_raw(
            &mut kernels,
            dense,
            &mut workspace.lhs_transform,
            &artifact.lhs_transform.transform_structure,
            lhs_core,
            lhs_src,
            &mut workspace.lhs,
            lhs_data,
            members,
            1,
        )?;
    }
    if !artifact.rhs_borrowed {
        tree_transform_members_overwrite_raw(
            &mut kernels,
            dense,
            &mut workspace.rhs_transform,
            &artifact.rhs_transform.transform_structure,
            rhs_core,
            rhs_src,
            &mut workspace.rhs,
            rhs_data,
            members,
            1,
        )?;
    }
    let physical_lhs = if artifact.lhs_borrowed {
        lhs_data
    } else {
        &workspace.lhs
    };
    let physical_rhs = if artifact.rhs_borrowed {
        rhs_data
    } else {
        &workspace.rhs
    };
    let (core_left, core_right) = if artifact.orientation == FusionContractOrientation::RhsLhs {
        (physical_rhs, physical_lhs)
    } else {
        (physical_lhs, physical_rhs)
    };
    let left_storage = ReadSlice(core_left);
    let right_storage = ReadSlice(core_right);
    let left_view = StackedStorageView::new::<D>(&left_storage, left_len, members, left_len)?;
    let right_view = StackedStorageView::new::<D>(&right_storage, right_len, members, right_len)?;
    let output = if core_dst.is_some() {
        &mut workspace.dst[..]
    } else {
        &mut *dst_data
    };
    let mut output_storage = WriteSlice(output);
    let mut output_view =
        StackedStorageViewMut::new::<D>(&mut output_storage, core_dst_len, members, core_dst_len)?;
    let mut gemm = super::super::fusion_block::BackendRank2Gemm::new(backend, backend_workspace);
    workspace.core.as_ref().unwrap().execute_host(
        &mut kernels,
        &mut gemm,
        &mut output_view,
        &left_view,
        &right_view,
        true,
    )?;
    if let Some(output) = core_dst {
        tree_transform_members_overwrite_raw(
            &mut kernels,
            dense,
            &mut workspace.output_transform,
            &output.output_transform_structure,
            dst_structure,
            core_dst_structure,
            dst_data,
            &workspace.dst,
            members,
            1,
        )?;
    }
    Ok(())
}
