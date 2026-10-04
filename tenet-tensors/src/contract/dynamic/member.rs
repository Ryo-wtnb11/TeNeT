use super::*;
use tenet_core::{validate_block_storage_injective, BlockStructure, Placement, TensorStorage};
use tenet_dense::DenseExecutor;
use tenet_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use tenet_operations::{
    admit_tree_transform_members_overwrite_raw, tree_transform_members_overwrite_raw,
    HostKernelAdapter, TreeTransformWorkspace,
};

/// Mutable Host payload and replay scratch for one artifact at varying B.
#[doc(hidden)]
pub struct DynamicTreeMembersWorkspace<D, C = f64> {
    lhs: Vec<D>,
    rhs: Vec<D>,
    dst: Vec<D>,
    lhs_transform: TreeTransformWorkspace<D>,
    rhs_transform: TreeTransformWorkspace<D>,
    output_transform: TreeTransformWorkspace<D>,
    core: Option<StackedDirectReplay<C>>,
    twist_shape: Vec<usize>,
    twist_strides: Vec<isize>,
    #[cfg(test)]
    core_replay_builds: usize,
    #[cfg(test)]
    twist_actions_applied: usize,
}

impl<D, C> Default for DynamicTreeMembersWorkspace<D, C> {
    fn default() -> Self {
        Self {
            lhs: Vec::new(),
            rhs: Vec::new(),
            dst: Vec::new(),
            lhs_transform: TreeTransformWorkspace::default(),
            rhs_transform: TreeTransformWorkspace::default(),
            output_transform: TreeTransformWorkspace::default(),
            core: None,
            twist_shape: Vec::new(),
            twist_strides: Vec::new(),
            #[cfg(test)]
            core_replay_builds: 0,
            #[cfg(test)]
            twist_actions_applied: 0,
        }
    }
}

impl<D, C: Copy + PartialEq + num_traits::One> DynamicTreeMembersWorkspace<D, C> {
    pub fn retained_bytes(&self) -> usize {
        (self.lhs.capacity() + self.rhs.capacity() + self.dst.capacity())
            .saturating_mul(std::mem::size_of::<D>())
            + self
                .core
                .as_ref()
                .map_or(0, StackedDirectReplay::retained_bytes)
            + self.lhs_transform.retained_bytes()
            + self.rhs_transform.retained_bytes()
            + self.output_transform.retained_bytes()
            + self.twist_shape.capacity() * std::mem::size_of::<usize>()
            + self.twist_strides.capacity() * std::mem::size_of::<isize>()
    }

    #[cfg(test)]
    pub(crate) fn core_replay_builds(&self) -> usize {
        self.core_replay_builds
    }

    #[cfg(test)]
    pub(crate) fn twist_actions_applied(&self) -> usize {
        self.twist_actions_applied
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

fn admit_member_twist<C>(
    structure: &BlockStructure,
    actions: &[RhsTwistAction<C>],
    member_len: usize,
    members: usize,
    shape: &mut Vec<usize>,
    strides: &mut Vec<isize>,
) -> Result<(), OperationError> {
    if actions.is_empty() {
        return Ok(());
    }
    if members == 0 {
        return Err(OperationError::InvalidArgument {
            message: "member source twist requires at least one member",
        });
    }
    validate_block_storage_injective(structure)
        .map_err(OperationError::from_core_preserving_context)?;
    let member_stride =
        isize::try_from(member_len).map_err(|_| OperationError::ElementCountOverflow)?;
    let total = member_len
        .checked_mul(members)
        .ok_or(OperationError::ElementCountOverflow)?;
    let mut block_index = 0;
    for action in actions {
        let mut admitted = None;
        while block_index < structure.block_count() {
            let block = structure
                .block(block_index)
                .map_err(OperationError::from_core_preserving_context)?;
            block_index += 1;
            if usize::try_from(action.offset).ok() == Some(block.offset())
                && action.shape == block.shape()
                && action.strides.len() == block.strides().len()
                && action
                    .strides
                    .iter()
                    .zip(block.strides())
                    .all(|(&actual, &expected)| isize::try_from(expected) == Ok(actual))
            {
                admitted = Some(block);
                break;
            }
        }
        let block = admitted.ok_or(OperationError::InvalidArgument {
            message: "member source twist does not match transformed block structure",
        })?;
        if !action.shape.contains(&0) {
            let end = block
                .storage_end_exclusive()
                .map_err(OperationError::from_core_preserving_context)?;
            if end > member_len {
                return Err(OperationError::InvalidArgument {
                    message: "member source twist exceeds one transformed source",
                });
            }
            let expanded_end = (members - 1)
                .checked_mul(member_len)
                .and_then(|start| start.checked_add(end))
                .ok_or(OperationError::ElementCountOverflow)?;
            if expanded_end > total || isize::try_from(expanded_end - 1).is_err() {
                return Err(OperationError::ElementCountOverflow);
            }
        }
        shape.clear();
        shape.extend_from_slice(&action.shape);
        shape.push(members);
        strides.clear();
        strides.extend_from_slice(&action.strides);
        strides.push(member_stride);
    }
    Ok(())
}

fn replay_member_twist<A, D, C>(
    kernels: &mut A,
    scratch: &mut [D],
    actions: &[RhsTwistAction<C>],
    members: usize,
    member_len: usize,
    shape: &mut Vec<usize>,
    strides: &mut Vec<isize>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let member_stride =
        isize::try_from(member_len).map_err(|_| OperationError::ElementCountOverflow)?;
    for action in actions {
        shape.clear();
        shape.extend_from_slice(&action.shape);
        shape.push(members);
        strides.clear();
        strides.extend_from_slice(&action.strides);
        strides.push(member_stride);
        kernels.scale_strided(
            scratch,
            shape,
            strides,
            action.offset,
            D::coefficient_as_data(action.factor),
        )?;
    }
    Ok(())
}

/// Execute one immutable artifact over uniform owned-dense Host
/// payloads. The caller binds semantic tensor metadata; this raw seam admits
/// only the artifact's structures and checked, disjoint member spans. TeNeT
/// submits grouped batches to the dense executors; their provider kernel
/// launch count is outside this seam. `threads` controls the ordinary B=1
/// transform schedule.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact_members_host<E, EC, D, C>(
    dense: &mut E,
    backend: &mut tenet_operations::DenseTreeTransformOperations<EC>,
    backend_workspace: &mut super::super::backend::TensorContractWorkspace<D>,
    artifact: &DynamicTreeExecutionArtifact<C>,
    dst_structure: &Arc<BlockStructure>,
    workspace: &mut DynamicTreeMembersWorkspace<D, C>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    members: usize,
    threads: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    EC: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let lhs_src = &artifact.lhs_transform.replay_structure;
    let rhs_src = &artifact.rhs_transform.replay_structure;
    let lhs_core = artifact.lhs_transform.space.structure();
    let rhs_core = artifact.rhs_transform.space.structure();
    let core_dst = artifact.core_dst.as_ref();
    let core_dst_structure = artifact.core_dst_structure(dst_structure);
    let lhs_src_len = member_len::<D>(lhs_src, members)?;
    let rhs_src_len = member_len::<D>(rhs_src, members)?;
    let dst_len = member_len::<D>(dst_structure, members)?;
    let lhs_core_len = member_len::<D>(lhs_core, members)?;
    let rhs_core_len = member_len::<D>(rhs_core, members)?;
    let core_dst_len = member_len::<D>(core_dst_structure, members)?;
    exact_len(lhs_data.len(), lhs_src_len, members)?;
    exact_len(rhs_data.len(), rhs_src_len, members)?;
    exact_len(dst_data.len(), dst_len, members)?;

    let cached = workspace.core.as_ref().is_some_and(|replay| {
        replay.members() == members && Arc::ptr_eq(replay.plan(), &artifact.block_plan)
    });
    let new_replay = if cached {
        None
    } else {
        Some(StackedDirectReplay::new(
            Arc::clone(&artifact.block_plan),
            members,
        )?)
    };
    let replay = new_replay
        .as_ref()
        .or(workspace.core.as_ref())
        .expect("cached or freshly built replay");
    let (left_len, right_len) = artifact.core_order(lhs_core_len, rhs_core_len);
    let (left_structure, right_structure) = artifact.core_order(lhs_core, rhs_core);
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
    if artifact.requires_source_twist() {
        let (structure, len) =
            artifact.on_twisted((lhs_core, lhs_core_len), (rhs_core, rhs_core_len));
        admit_member_twist(
            structure,
            &artifact.source_twist,
            len,
            members,
            &mut workspace.twist_shape,
            &mut workspace.twist_strides,
        )?;
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
    if let Some(replay) = new_replay {
        workspace.core = Some(replay);
        #[cfg(test)]
        {
            workspace.core_replay_builds += 1;
        }
    }

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
            threads,
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
            threads,
        )?;
    }
    if artifact.requires_source_twist() {
        let (scratch, len) = artifact.on_twisted(
            (&mut workspace.lhs[..], lhs_core_len),
            (&mut workspace.rhs[..], rhs_core_len),
        );
        replay_member_twist(
            &mut kernels,
            scratch,
            &artifact.source_twist,
            members,
            len,
            &mut workspace.twist_shape,
            &mut workspace.twist_strides,
        )?;
        #[cfg(test)]
        {
            workspace.twist_actions_applied += artifact.source_twist.len();
        }
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
    let (core_left, core_right) = artifact.core_order(physical_lhs, physical_rhs);
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
    let mut gemm =
        super::super::fusion_block::BackendRank2Gemm::<_, _, C>::new(backend, backend_workspace);
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
            threads,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::BlockSpec;

    #[test]
    fn twist_preflight_rejects_a_view_spilling_into_the_next_member() {
        let structure =
            BlockStructure::from_blocks(vec![BlockSpec::new(vec![2], vec![1], 1).unwrap()])
                .unwrap();
        let actions = [RhsTwistAction {
            shape: vec![2],
            strides: vec![1],
            offset: 1,
            factor: -1.0_f64,
        }];
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        assert!(admit_member_twist(&structure, &actions, 2, 2, &mut shape, &mut strides,).is_err());
        admit_member_twist(&structure, &actions, 3, 2, &mut shape, &mut strides).unwrap();
        assert_eq!(shape, [2, 2]);
        assert_eq!(strides, [1, 3]);
        assert!(admit_member_twist(&structure, &actions, 3, 0, &mut shape, &mut strides).is_err());
    }
}
