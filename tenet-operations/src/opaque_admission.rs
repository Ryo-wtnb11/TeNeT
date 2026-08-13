use core::{alloc::Layout, any::TypeId};
use std::sync::{Arc, Weak};

use tenet_core::{BlockStructure, Placement};

use crate::{task_view::TreeTransformTaskView, OperationError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContextIdentity(pub(crate) u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageDomain(pub(crate) u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AllocationIdentity(pub(crate) u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageRegion {
    Empty,
    Bytes {
        domain: StorageDomain,
        allocation: AllocationIdentity,
        start: usize,
        len: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageSnapshot {
    pub(crate) active_len: usize,
    pub(crate) usable_capacity: usize,
    pub(crate) placement: Placement,
    pub(crate) context: ContextIdentity,
    pub(crate) region: StorageRegion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExecutorSnapshot {
    pub(crate) placement: Placement,
    pub(crate) context: ContextIdentity,
    pub(crate) supports_strided: bool,
    pub(crate) supports_matrix: bool,
    pub(crate) scalar: TypeId,
}

#[derive(Clone, Debug)]
pub(crate) struct CoefficientReadiness {
    pub(crate) structure_and_layout: Weak<()>,
    pub(crate) scalar: TypeId,
    pub(crate) context: ContextIdentity,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceSnapshot {
    pub(crate) packed_source: StorageSnapshot,
    pub(crate) packed_destination: StorageSnapshot,
    pub(crate) converted_coefficients: StorageSnapshot,
    pub(crate) fused_index_capacity: usize,
    pub(crate) fused_index_placement: Placement,
    pub(crate) coefficient_readiness: Option<CoefficientReadiness>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TreeTransformAdmissionError {
    Structure(&'static str),
    Length { expected: usize, actual: usize },
    ArithmeticOverflow,
    Placement(&'static str),
    Context(&'static str),
    Capability(&'static str),
    InvalidRegion(&'static str),
    Aliasing,
    WorkspaceCapacity(&'static str),
    CoefficientReadiness,
}

impl From<TreeTransformAdmissionError> for OperationError {
    fn from(error: TreeTransformAdmissionError) -> Self {
        let message = match error {
            TreeTransformAdmissionError::Structure(_) => {
                "tree transform structure admission failed"
            }
            TreeTransformAdmissionError::Length { .. } => {
                "tree transform storage length admission failed"
            }
            TreeTransformAdmissionError::ArithmeticOverflow => {
                "tree transform admission arithmetic overflow"
            }
            TreeTransformAdmissionError::Placement(_) => {
                "tree transform placement admission failed"
            }
            TreeTransformAdmissionError::Context(_) => "tree transform context admission failed",
            TreeTransformAdmissionError::Capability(_) => {
                "tree transform capability admission failed"
            }
            TreeTransformAdmissionError::InvalidRegion(_) => {
                "tree transform storage region admission failed"
            }
            TreeTransformAdmissionError::Aliasing => "tree transform storage regions overlap",
            TreeTransformAdmissionError::WorkspaceCapacity(_) => {
                "tree transform workspace capacity admission failed"
            }
            TreeTransformAdmissionError::CoefficientReadiness => {
                "tree transform coefficient readiness admission failed"
            }
        };
        OperationError::InvalidArgument { message }
    }
}

pub(crate) fn validate_stage_a<D: 'static, C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst: StorageSnapshot,
    src: StorageSnapshot,
    executor: ExecutorSnapshot,
    workers: usize,
) -> Result<usize, TreeTransformAdmissionError> {
    task.validate_structures_and_lengths(
        dst_structure,
        src_structure,
        dst.active_len,
        src.active_len,
    )
    .map_err(map_task_error)?;
    for (name, storage) in [("destination", dst), ("source", src)] {
        if storage.placement != executor.placement {
            return Err(TreeTransformAdmissionError::Placement(name));
        }
        if storage.context != executor.context {
            return Err(TreeTransformAdmissionError::Context(name));
        }
    }
    if !executor.supports_strided {
        return Err(TreeTransformAdmissionError::Capability("strided"));
    }
    if !task.recoupling_plan().is_empty() && !executor.supports_matrix {
        return Err(TreeTransformAdmissionError::Capability("matrix"));
    }
    if executor.scalar != TypeId::of::<D>() {
        return Err(TreeTransformAdmissionError::Capability("scalar"));
    }
    let requirements = task.workspace_requirements();
    for len in [
        requirements.packed_source_len,
        requirements.packed_destination_len,
        requirements.converted_coefficient_len,
    ] {
        Layout::array::<D>(len).map_err(|_| TreeTransformAdmissionError::ArithmeticOverflow)?;
    }
    let fused = workers
        .max(1)
        .checked_mul(requirements.fused_index_len_per_worker)
        .ok_or(TreeTransformAdmissionError::ArithmeticOverflow)?;
    Layout::array::<usize>(fused).map_err(|_| TreeTransformAdmissionError::ArithmeticOverflow)?;
    Ok(fused)
}

pub(crate) fn validate_stage_c<D: 'static, C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    dst: StorageSnapshot,
    src: StorageSnapshot,
    workspace: &WorkspaceSnapshot,
    executor: ExecutorSnapshot,
    fused_index_len: usize,
) -> Result<(), TreeTransformAdmissionError> {
    let requirements = task.workspace_requirements();
    validate_region::<D>("destination", dst)?;
    validate_region::<D>("source", src)?;
    for (name, storage, required) in [
        (
            "packed source",
            workspace.packed_source,
            requirements.packed_source_len,
        ),
        (
            "packed destination",
            workspace.packed_destination,
            requirements.packed_destination_len,
        ),
        (
            "converted coefficients",
            workspace.converted_coefficients,
            requirements.converted_coefficient_len,
        ),
    ] {
        if storage.usable_capacity < required {
            return Err(TreeTransformAdmissionError::WorkspaceCapacity(name));
        }
        if storage.placement != executor.placement {
            return Err(TreeTransformAdmissionError::Placement(name));
        }
        if storage.context != executor.context {
            return Err(TreeTransformAdmissionError::Context(name));
        }
        validate_region::<D>(name, storage)?;
    }
    if workspace.fused_index_capacity < fused_index_len {
        return Err(TreeTransformAdmissionError::WorkspaceCapacity(
            "fused indices",
        ));
    }
    if workspace.fused_index_placement != Placement::Host {
        return Err(TreeTransformAdmissionError::Placement("fused indices"));
    }
    let regions = [
        (dst.region, true),
        (src.region, false),
        (workspace.packed_source.region, true),
        (workspace.packed_destination.region, true),
        (workspace.converted_coefficients.region, false),
    ];
    for left in 0..regions.len() {
        for right in left + 1..regions.len() {
            if (regions[left].1 || regions[right].1) && overlaps(regions[left].0, regions[right].0)?
            {
                return Err(TreeTransformAdmissionError::Aliasing);
            }
        }
    }
    if requirements.converted_coefficient_len != 0 {
        let ready = workspace.coefficient_readiness.as_ref();
        if !ready.is_some_and(|ready| {
            Weak::ptr_eq(&ready.structure_and_layout, &task.admission_identity())
                && ready.scalar == TypeId::of::<D>()
                && ready.context == executor.context
        }) {
            return Err(TreeTransformAdmissionError::CoefficientReadiness);
        }
    }
    Ok(())
}

fn validate_region<D>(
    name: &'static str,
    storage: StorageSnapshot,
) -> Result<(), TreeTransformAdmissionError> {
    let bytes = Layout::array::<D>(storage.usable_capacity)
        .map_err(|_| TreeTransformAdmissionError::ArithmeticOverflow)?
        .size();
    match storage.region {
        StorageRegion::Empty if storage.usable_capacity == 0 => Ok(()),
        StorageRegion::Empty => Err(TreeTransformAdmissionError::InvalidRegion(name)),
        StorageRegion::Bytes { start, len, .. } => {
            start
                .checked_add(len)
                .ok_or(TreeTransformAdmissionError::InvalidRegion(name))?;
            if len < bytes {
                Err(TreeTransformAdmissionError::InvalidRegion(name))
            } else {
                Ok(())
            }
        }
    }
}

fn overlaps(
    left: StorageRegion,
    right: StorageRegion,
) -> Result<bool, TreeTransformAdmissionError> {
    let (
        StorageRegion::Bytes {
            domain: ld,
            allocation: la,
            start: ls,
            len: ll,
        },
        StorageRegion::Bytes {
            domain: rd,
            allocation: ra,
            start: rs,
            len: rl,
        },
    ) = (left, right)
    else {
        return Ok(false);
    };
    let le = ls
        .checked_add(ll)
        .ok_or(TreeTransformAdmissionError::InvalidRegion("storage"))?;
    let re = rs
        .checked_add(rl)
        .ok_or(TreeTransformAdmissionError::InvalidRegion("storage"))?;
    Ok(ld == rd && la == ra && ls < re && rs < le)
}

fn map_task_error(error: OperationError) -> TreeTransformAdmissionError {
    match error {
        OperationError::StructureMismatch { tensor } => {
            TreeTransformAdmissionError::Structure(tensor)
        }
        OperationError::ElementCountMismatch { expected, actual } => {
            TreeTransformAdmissionError::Length { expected, actual }
        }
        _ => TreeTransformAdmissionError::ArithmeticOverflow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TreeTransformBlockSpec;

    fn region(domain: u64, allocation: u64, start: usize, len: usize) -> StorageRegion {
        StorageRegion::Bytes {
            domain: StorageDomain(domain),
            allocation: AllocationIdentity(allocation),
            start,
            len,
        }
    }

    #[test]
    fn half_open_regions_obey_domain_and_empty_rules() {
        assert!(!overlaps(region(1, 1, 0, 8), region(1, 1, 8, 8)).unwrap());
        assert!(overlaps(region(1, 1, 0, 9), region(1, 1, 8, 8)).unwrap());
        assert!(!overlaps(region(1, 1, 0, 9), region(2, 1, 0, 9)).unwrap());
        assert!(!overlaps(StorageRegion::Empty, region(1, 1, 0, 9)).unwrap());
        assert_eq!(
            overlaps(region(1, 1, usize::MAX, 2), region(1, 1, 0, 1)),
            Err(TreeTransformAdmissionError::InvalidRegion("storage"))
        );
    }

    #[test]
    fn nonempty_capacity_requires_a_large_enough_checked_region() {
        let snapshot = StorageSnapshot {
            active_len: 2,
            usable_capacity: 2,
            placement: Placement::Host,
            context: ContextIdentity(1),
            region: StorageRegion::Empty,
        };
        assert_eq!(
            validate_region::<u64>("source", snapshot),
            Err(TreeTransformAdmissionError::InvalidRegion("source"))
        );
        assert_eq!(
            validate_region::<u64>(
                "source",
                StorageSnapshot {
                    region: region(1, 1, 0, 15),
                    ..snapshot
                }
            ),
            Err(TreeTransformAdmissionError::InvalidRegion("source"))
        );
        validate_region::<u64>(
            "source",
            StorageSnapshot {
                region: region(1, 1, 0, 16),
                ..snapshot
            },
        )
        .unwrap();
    }

    #[test]
    fn stage_a_and_c_validate_completed_multi_task() {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
        let transform = crate::TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0_f64, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        let task = transform.task_view().unwrap();
        let context = ContextIdentity(7);
        let executor = ExecutorSnapshot {
            placement: Placement::Host,
            context,
            supports_strided: true,
            supports_matrix: true,
            scalar: TypeId::of::<f64>(),
        };
        let storage = |allocation, capacity| StorageSnapshot {
            active_len: capacity,
            usable_capacity: capacity,
            placement: Placement::Host,
            context,
            region: region(1, allocation, 0, capacity * size_of::<f64>()),
        };
        let dst = storage(1, 4);
        let src = storage(2, 4);
        let fused = validate_stage_a::<f64, _>(task, &structure, &structure, dst, src, executor, 3)
            .unwrap();
        assert_eq!(fused, 3);
        let structure_and_layout = task.admission_identity();
        let workspace = WorkspaceSnapshot {
            packed_source: storage(3, 4),
            packed_destination: storage(4, 4),
            converted_coefficients: storage(5, 4),
            fused_index_capacity: fused,
            fused_index_placement: Placement::Host,
            coefficient_readiness: Some(CoefficientReadiness {
                structure_and_layout,
                scalar: TypeId::of::<f64>(),
                context,
            }),
        };
        validate_stage_c::<f64, _>(task, dst, src, &workspace, executor, fused).unwrap();
        assert_eq!(
            validate_stage_c::<f64, _>(
                task,
                dst,
                src,
                &WorkspaceSnapshot {
                    packed_destination: StorageSnapshot {
                        region: region(1, 1, 8, 32),
                        ..workspace.packed_destination
                    },
                    ..workspace
                },
                executor,
                fused,
            ),
            Err(TreeTransformAdmissionError::Aliasing)
        );
        assert_eq!(
            OperationError::from(TreeTransformAdmissionError::Capability("matrix")),
            OperationError::InvalidArgument {
                message: "tree transform capability admission failed"
            }
        );
    }
}
