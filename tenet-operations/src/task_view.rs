use std::sync::Arc;

use tenet_core::BlockStructure;

use crate::{OperationError, TreeTransformStructure};

/// Backend-neutral scratch sizes required by a completed tree transform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TreeTransformWorkspaceRequirements {
    pub packed_source_len: usize,
    pub packed_destination_len: usize,
    pub converted_coefficient_len: usize,
    pub fused_index_len_per_worker: usize,
}

/// Read-only synchronous replay view over a completed tree-transform structure.
///
/// This adapter contains no provider, fusion-tree, backend schedule, storage,
/// pointer, or mutable workspace state. Concrete executors remain responsible
/// for placement/capability checks, allocation-range disjointness, and ensuring
/// their workspace satisfies [`Self::workspace_requirements`] before dispatch.
/// Descriptor replay through this admission prototype remains the follow-up
/// Host-bridge work.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TreeTransformTaskView<'a, C> {
    structure: &'a TreeTransformStructure<C>,
    source_len: usize,
    destination_len: usize,
}

impl<C: Copy> TreeTransformStructure<C> {
    pub(crate) fn task_view(&self) -> Result<TreeTransformTaskView<'_, C>, OperationError> {
        let (source_len, destination_len) = self.replay_storage_lens()?;
        Ok(TreeTransformTaskView {
            structure: self,
            source_len,
            destination_len,
        })
    }
}

impl<'a, C: Copy> TreeTransformTaskView<'a, C> {
    pub(crate) fn workspace_requirements(self) -> TreeTransformWorkspaceRequirements {
        let plan = self.structure.recoupling_plan();
        TreeTransformWorkspaceRequirements {
            packed_source_len: plan.source_len(),
            packed_destination_len: plan.destination_len(),
            converted_coefficient_len: plan.coefficient_len(),
            fused_index_len_per_worker: self.structure.layouts().max_fused_rank(),
        }
    }

    pub(crate) fn validate_workspace_requirements<D>(self) -> Result<(), OperationError> {
        let requirements = self.workspace_requirements();
        for len in [
            requirements.packed_source_len,
            requirements.packed_destination_len,
            requirements.converted_coefficient_len,
        ] {
            core::alloc::Layout::array::<D>(len)
                .map_err(|_| OperationError::ElementCountOverflow)?;
        }
        core::alloc::Layout::array::<usize>(requirements.fused_index_len_per_worker)
            .map_err(|_| OperationError::ElementCountOverflow)?;
        Ok(())
    }

    pub(crate) fn validate_structures_and_lengths(
        self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_len: usize,
        src_len: usize,
    ) -> Result<(), OperationError> {
        self.structure
            .validate_replay_structures(dst_structure, src_structure)?;
        validate_exact_len(self.destination_len, dst_len)?;
        validate_exact_len(self.source_len, src_len)
    }
}

fn validate_exact_len(expected: usize, actual: usize) -> Result<(), OperationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(OperationError::ElementCountMismatch { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TreeTransformBlockSpec;

    #[test]
    fn completed_view_exposes_lengths_and_workspace_requirements() {
        let structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![2], vec![2], vec![1], vec![1], vec![1]])
                .unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                TreeTransformBlockSpec::single(2, 2, -1.0_f64),
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 2.0, 3.0, 4.0]),
                TreeTransformBlockSpec::multi(vec![3, 4], vec![3, 4], vec![5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();

        let view = transform.task_view().unwrap();
        view.validate_structures_and_lengths(&structure, &structure, 7, 7)
            .unwrap();
        assert_eq!(
            view.workspace_requirements(),
            TreeTransformWorkspaceRequirements {
                packed_source_len: 6,
                packed_destination_len: 6,
                converted_coefficient_len: 8,
                fused_index_len_per_worker: 1,
            }
        );
    }

    #[test]
    fn admission_rejects_structure_or_nonexact_length() {
        let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2]]).unwrap());
        let other = Arc::new(BlockStructure::packed_column_major(1, [vec![3]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
        )
        .unwrap();
        let view = transform.task_view().unwrap();

        assert_eq!(
            view.validate_structures_and_lengths(&structure, &structure, 3, 2),
            Err(OperationError::ElementCountMismatch {
                expected: 2,
                actual: 3,
            })
        );
        assert_eq!(
            view.validate_structures_and_lengths(&other, &structure, 3, 2),
            Err(OperationError::StructureMismatch { tensor: "dst" })
        );
    }
}
