use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{
    BlockStructure, CategoricalScalar, FusionTreeHomSpace, FusionTreePairOrientation,
    MultiplicityFreeRigidSymbols,
};

use crate::cache::{
    touch_lru_key, BlockStructureCacheKey, OperationCachePolicy, DEFAULT_OPERATION_CACHE_ENTRIES,
};
use crate::mode::TreeStructureSource;
use crate::tree_context::TreeTransformExecutionContext;
#[cfg(test)]
use crate::DenseTreeTransformOperations;
use crate::{
    DenseBlockScalar, DenseRecouplingScalar, OperationError, RecouplingCoefficientAction,
    TreeTransformBackend, TreeTransformOperation, TreeTransformOperationKind,
    TreeTransformRuleCacheKey, TreeTransformStructure,
};
use tenet_operations::fusion_replay::FusionBlockContractPlan;
use tenet_operations::TensorContractSpecOwned;

use super::backend::TensorContractBackend;
#[cfg(test)]
use super::dynamic_space::encoded_layout_primer;
use super::dynamic_space::{DynamicFusionMapSpace, FusionOperandLayout, LayoutKeyBuilder};
use super::fusion::{
    contract_twist_on_physical_lhs, FusionContractOrientation, FusionContractPlan,
};
use super::fusion_block::FusionBlockContractWorkspace;
use super::resolution::rhs_contract_requires_twist;
use super::scratch::{DynamicFusionScratch, DynamicFusionScratchWorkspace};
use tenet_operations::TensorContractFusionProfile;

#[cfg(test)]
const PROFILED_ARTIFACT_SOURCE_PHASE: u8 = 1 << 0;
#[cfg(test)]
const PROFILED_ARTIFACT_CORE_DST_PHASE: u8 = 1 << 1;
#[cfg(test)]
const PROFILED_ARTIFACT_BLOCK_PLAN_PHASE: u8 = 1 << 2;

#[cfg(test)]
std::thread_local! {
    static PROFILED_ARTIFACT_COMPILE_PHASES: std::cell::Cell<u8> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_profiled_artifact_compile_phases() {
    PROFILED_ARTIFACT_COMPILE_PHASES.set(0);
}

#[cfg(test)]
pub(crate) fn profiled_artifact_compile_phases() -> (bool, bool, bool) {
    let phases = PROFILED_ARTIFACT_COMPILE_PHASES.get();
    (
        phases & PROFILED_ARTIFACT_SOURCE_PHASE != 0,
        phases & PROFILED_ARTIFACT_CORE_DST_PHASE != 0,
        phases & PROFILED_ARTIFACT_BLOCK_PLAN_PHASE != 0,
    )
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_member;
mod member;
pub(crate) use member::execute_dynamic_tree_execution_artifact_members_host;
#[doc(hidden)]
pub use member::DynamicTreeMembersWorkspace;

#[cfg(test)]
use super::{backend, dynamic_space};
use super::{fusion, fusion_block, resolution};

mod artifact;
mod space_cache;
#[cfg(test)]
mod test_entry;
#[cfg(test)]
mod tests;
mod twist;
// Why test-only: the typed plan-level executor is a reference oracle for the
// planner's DynamicTree artifact; production routes through `plan_contract`.
#[cfg(test)]
mod typed_eager;
pub(crate) use artifact::*;
pub(crate) use space_cache::*;
#[cfg(test)]
pub(crate) use test_entry::*;
pub(super) use twist::*;
#[cfg(test)]
pub(crate) use typed_eager::*;

#[derive(Clone, Copy)]
struct CoreSource<'a, D> {
    space: &'a DynamicFusionMapSpace,
    data: &'a [D],
}

impl<'a, D> CoreSource<'a, D> {
    fn borrowed(space: &'a DynamicFusionMapSpace, data: &'a [D]) -> Self {
        Self { space, data }
    }

    fn materialized(space: &'a DynamicFusionMapSpace, data: &'a [D]) -> Self {
        Self { space, data }
    }

    fn from_host_scratch(scratch: &'a DynamicFusionScratch<D>) -> Self {
        Self::materialized(scratch.space(), scratch.data())
    }

    fn space(self) -> &'a DynamicFusionMapSpace {
        self.space
    }

    fn structure(self) -> &'a Arc<BlockStructure> {
        // Why not retain the input structure separately: borrowability proves
        // identical core layout, so the core space remains the single authority.
        self.space().structure()
    }

    fn data(self) -> &'a [D] {
        self.data
    }
}

fn select_core_source<'a, D>(
    borrow: bool,
    borrowed_space: &'a DynamicFusionMapSpace,
    borrowed_data: &'a [D],
    materialize: impl FnOnce() -> CoreSource<'a, D>,
) -> CoreSource<'a, D> {
    if borrow {
        CoreSource::borrowed(borrowed_space, borrowed_data)
    } else {
        materialize()
    }
}

pub(super) fn source_layout_metadata_is_borrowable<HomSpaceMatches>(
    source_space: &DynamicFusionMapSpace,
    core_nout: usize,
    core_rank: usize,
    homspace_matches: HomSpaceMatches,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
) -> bool
where
    HomSpaceMatches: FnOnce() -> bool,
{
    operation.kind() == TreeTransformOperationKind::Permute
        && source_layout_permutation_is_borrowable(
            source_space,
            core_nout,
            core_rank,
            homspace_matches,
            operation.codomain_permutation(),
            operation.domain_permutation(),
            source_conjugate,
        )
}

/// [`source_layout_metadata_is_borrowable`] for a permutation given as its
/// two axis lists, before any `TreeTransformOperation` value exists.
pub(super) fn source_layout_permutation_is_borrowable<HomSpaceMatches>(
    source_space: &DynamicFusionMapSpace,
    core_nout: usize,
    core_rank: usize,
    homspace_matches: HomSpaceMatches,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    source_conjugate: bool,
) -> bool
where
    HomSpaceMatches: FnOnce() -> bool,
{
    if source_conjugate {
        return false;
    }
    if !codomain_permutation
        .iter()
        .copied()
        .eq(0..source_space.nout())
        || !domain_permutation
            .iter()
            .copied()
            .eq(source_space.nout()..source_space.rank())
    {
        return false;
    }
    if core_nout != source_space.nout() || core_rank != source_space.rank() || !homspace_matches() {
        return false;
    }
    true
}

#[cfg(test)]
std::thread_local! {
    static SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_source_layout_homspace_id_comparisons() {
    SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.set(0);
}

#[cfg(test)]
pub(crate) fn source_layout_homspace_id_comparisons() -> usize {
    SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.get()
}

fn source_layout_homspaces_match_by_id(
    source_space: &DynamicFusionMapSpace,
    core_space: &DynamicFusionMapSpace,
) -> bool {
    #[cfg(test)]
    SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.set(SOURCE_LAYOUT_HOMSPACE_ID_COMPARISONS.get() + 1);
    core_space.homspace().id() == source_space.homspace().id()
}

fn source_is_borrowable_core_layout(
    source_space: &DynamicFusionMapSpace,
    source_structure: &Arc<BlockStructure>,
    core_space: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
) -> bool {
    if !source_layout_metadata_is_borrowable(
        source_space,
        core_space.nout(),
        core_space.rank(),
        || source_layout_homspaces_match_by_id(source_space, core_space),
        operation,
        source_conjugate,
    ) {
        return false;
    }
    let core_structure = core_space.structure();
    // Why not compare only the source's declared structure: even identity axes
    // can complete a sparse fusion-tree grid with structural-zero core blocks.
    Arc::ptr_eq(core_structure, source_structure)
        || core_structure.content_id() == source_structure.content_id()
        // Why not rely on content ids alone: an intern reset can assign a new
        // monotonic id to equal live content while an operation cache pins both.
        || core_structure.as_ref() == source_structure.as_ref()
}

/// Which physical operands are read in place, and which one carries the
/// fermionic contraction twist when there is one.
#[derive(Clone, Copy, Debug)]
struct SourceBorrowing {
    lhs_borrowed: bool,
    rhs_borrowed: bool,
    twist_lhs: bool,
}

/// Borrowing once the fermionic contraction twist, if any, claims one
/// operand: the one the layout already copies when exactly one is copied
/// (so no extra materialization), otherwise the smaller
/// ([`contract_twist_on_physical_lhs`]). A twisted operand is never borrowed.
fn resolve_source_borrowing<R>(
    rule: &R,
    plan: &FusionContractPlan,
    lhs_core: &DynamicFusionMapSpace,
    rhs_core: &DynamicFusionMapSpace,
    lhs_layout_borrowable: bool,
    rhs_layout_borrowable: bool,
) -> Result<SourceBorrowing, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let core_right = if reverse { lhs_core } else { rhs_core };
    if !rhs_contract_requires_twist(rule, core_right, plan.core_axes().as_spec())? {
        return Ok(SourceBorrowing {
            lhs_borrowed: lhs_layout_borrowable,
            rhs_borrowed: rhs_layout_borrowable,
            twist_lhs: false,
        });
    }
    let required_len = |space: &DynamicFusionMapSpace| {
        space
            .required_len()
            .map_err(OperationError::from_core_preserving_context)
    };
    let twist_lhs = contract_twist_on_physical_lhs(
        reverse,
        !lhs_layout_borrowable,
        !rhs_layout_borrowable,
        || Ok((required_len(lhs_core)?, required_len(rhs_core)?)),
    )?;
    Ok(SourceBorrowing {
        lhs_borrowed: lhs_layout_borrowable && !twist_lhs,
        rhs_borrowed: rhs_layout_borrowable && twist_lhs,
        twist_lhs,
    })
}
