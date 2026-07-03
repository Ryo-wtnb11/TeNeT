//! Symmetry-free replay half of the canonical fusion-block contraction:
//! plan data (offsets, strides, coefficients), pack/GEMM/scatter execution,
//! workspaces, and the storage-direct device seam. The symmetric compile
//! layer builds these plans; nothing here consumes fusion rules.

use std::collections::HashSet;
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockStructure, HostReadableStorage, HostWritableStorage, Placement, ScratchStorage, SectorId,
    SimilarStorage, TensorStorage,
};

use crate::host_scratch::HostScratchBuffer;
use crate::placement::ReportsPlacement;
use crate::profile::TensorContractFusionProfile;
use crate::storage_scratch::{
    FusionBlockContractScratchBuffers, StorageFusionBlockContractWorkspace,
};
use crate::strided::{offset_to_isize, strides_to_isize};
use crate::structure_identity::validate_structure_identity;
use crate::{DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction};

/// Rank-2 column-major GEMM over host slices: the only capability the replay
/// half needs from a contraction backend. The symmetric layer adapts its
/// contraction backends onto this.
pub trait Rank2Gemm<D> {
    fn matmul_rank2(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError>;
}

pub struct HostCanonicalFusionBlockContractWorkspace<T> {
    buffers: HostFusionBlockContractBuffers<T>,
}

pub type CanonicalFusionBlockContractWorkspace<T> = HostCanonicalFusionBlockContractWorkspace<T>;

impl<T> Default for HostCanonicalFusionBlockContractWorkspace<T> {
    fn default() -> Self {
        Self {
            buffers: HostFusionBlockContractBuffers::default(),
        }
    }
}

impl<T> ReportsPlacement for HostCanonicalFusionBlockContractWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

#[derive(Clone, Debug)]
struct HostFusionBlockContractBuffers<T> {
    packed: FusionBlockContractScratchBuffers<
        HostScratchBuffer<T>,
        HostScratchBuffer<T>,
        HostScratchBuffer<T>,
    >,
}

impl<T> Default for HostFusionBlockContractBuffers<T> {
    fn default() -> Self {
        Self {
            packed: FusionBlockContractScratchBuffers::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CanonicalFusionBlockContractPlan {
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
    inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
    groups: Vec<CanonicalFusionBlockContractGroupPlan>,
}

impl CanonicalFusionBlockContractPlan {
    /// Assembles a compiled plan; called by the symmetric compile layer.
    pub fn from_parts(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<CanonicalFusionBlockContractGroupPlan>,
    ) -> Self {
        Self {
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_raw<A, G, D>(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;

        let trivial_scale = alpha.is_one() && beta.is_zero();
        for group in &self.groups {
            if !group.is_fully_direct(trivial_scale) {
                fusion_workspace
                    .buffers
                    .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
                fusion_workspace
                    .buffers
                    .clear_inputs(group.lhs.needs_clear, group.rhs.needs_clear);
            }
            execute_group_with_scratch_buffers(
                kernels,
                gemm,
                group,
                &mut fusion_workspace.buffers.packed,
                dst_data,
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_raw_profiled<A, G, D>(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TensorContractFusionProfile,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        let total_start = std::time::Instant::now();

        let start = std::time::Instant::now();
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        profile.canonical_validate += start.elapsed();

        let start = std::time::Instant::now();
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;
        profile.canonical_scale += start.elapsed();

        let trivial_scale = alpha.is_one() && beta.is_zero();
        for group in &self.groups {
            profile.canonical_contract_groups += 1;

            if !group.is_fully_direct(trivial_scale) {
                let start = std::time::Instant::now();
                fusion_workspace
                    .buffers
                    .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
                fusion_workspace
                    .buffers
                    .clear_inputs(group.lhs.needs_clear, group.rhs.needs_clear);
                profile.canonical_workspace_prepare += start.elapsed();
            }

            if group.lhs.direct_offset.is_none() {
                let start = std::time::Instant::now();
                pack_group(
                    kernels,
                    &group.lhs,
                    lhs_data,
                    fusion_workspace.buffers.packed.lhs_mut().as_mut_slice(),
                )?;
                profile.canonical_pack_lhs += start.elapsed();
            } else {
                profile.canonical_direct_pack_skips += 1;
            }

            if group.rhs.direct_offset.is_none() {
                let start = std::time::Instant::now();
                pack_group(
                    kernels,
                    &group.rhs,
                    rhs_data,
                    fusion_workspace.buffers.packed.rhs_mut().as_mut_slice(),
                )?;
                profile.canonical_pack_rhs += start.elapsed();
            } else {
                profile.canonical_direct_pack_skips += 1;
            }

            let dst_direct = if trivial_scale {
                group.dst.direct_offset
            } else {
                None
            };
            let start = std::time::Instant::now();
            {
                let (lhs, rhs, dst) = fusion_workspace.buffers.packed.inputs_and_destination_mut();
                let lhs_slice = direct_or_scratch_slice(
                    lhs_data,
                    group.lhs.direct_offset,
                    group.lhs.rows,
                    group.lhs.cols,
                    lhs.as_slice(),
                )?;
                let rhs_slice = direct_or_scratch_slice(
                    rhs_data,
                    group.rhs.direct_offset,
                    group.rhs.rows,
                    group.rhs.cols,
                    rhs.as_slice(),
                )?;
                match dst_direct {
                    Some(base) => {
                        let dst_slice =
                            direct_slice_mut(dst_data, base, group.dst.rows, group.dst.cols)?;
                        matmul_group_plan(gemm, group, lhs_slice, rhs_slice, dst_slice)?;
                        profile.canonical_direct_gemm_groups += 1;
                    }
                    None => {
                        matmul_group_plan(gemm, group, lhs_slice, rhs_slice, dst.as_mut_slice())?;
                    }
                }
            }
            profile.canonical_matmul += start.elapsed();

            if dst_direct.is_none() {
                let start = std::time::Instant::now();
                scatter_group(
                    kernels,
                    &group.dst,
                    dst_data,
                    fusion_workspace.buffers.packed.destination().as_slice(),
                    alpha,
                    beta,
                )?;
                profile.canonical_scatter += start.elapsed();
            }
        }

        profile.canonical_contract_total += total_start.elapsed();
        Ok(())
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_storage_workspace<
        A,
        G,
        D,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
        DDst,
        DLhs,
        DRhs,
    >(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut StorageFusionBlockContractWorkspace<
            DLhs::Similar,
            DRhs::Similar,
            DDst::Similar,
        >,
        dst: &mut tenet_core::TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &tenet_core::TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &tenet_core::TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
        DDst: HostWritableStorage<D> + SimilarStorage<D>,
        DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DLhs: HostReadableStorage<D> + SimilarStorage<D>,
        DLhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DRhs: HostReadableStorage<D> + SimilarStorage<D>,
        DRhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    {
        let dst_structure = Arc::clone(dst.structure());
        let lhs_structure = Arc::clone(lhs.structure());
        let rhs_structure = Arc::clone(rhs.structure());
        self.validate_replay_inputs(
            &dst_structure,
            dst.storage().len(),
            &lhs_structure,
            lhs.storage().len(),
            &rhs_structure,
            rhs.storage().len(),
        )?;
        scale_all_blocks(
            kernels,
            &self.inactive_dst_scale_blocks,
            dst.data_mut(),
            beta,
        )?;

        let lhs_data = lhs.data();
        let rhs_data = rhs.data();
        for group in &self.groups {
            let lens =
                fusion_block_group_scratch_lens(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace.prepare_from_storages(
                lhs.storage(),
                rhs.storage(),
                dst.storage(),
                lens.lhs,
                lens.rhs,
                lens.destination,
                D::zero(),
            );
            execute_group_with_scratch_buffers(
                kernels,
                gemm,
                group,
                fusion_workspace.buffers_mut(),
                dst.data_mut(),
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    /// Storage-aware raw replay for callers whose operands are scratch buffers
    /// rather than `TensorMap`s (the dynamic canonical route).
    ///
    /// Pack scratch allocation origins are passed explicitly: LHS pack scratch
    /// from `lhs_alloc`, RHS pack scratch from `rhs_alloc`, and matmul output
    /// scratch from `dst_alloc`, while replay itself consumes the raw slices.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_storage_raw<A, G, D, SLhs, SRhs, SDst>(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut StorageFusionBlockContractWorkspace<
            SLhs::Similar,
            SRhs::Similar,
            SDst::Similar,
        >,
        lhs_alloc: &SLhs,
        rhs_alloc: &SRhs,
        dst_alloc: &SDst,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
        SLhs: SimilarStorage<D>,
        SLhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        SRhs: SimilarStorage<D>,
        SRhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        SDst: SimilarStorage<D>,
        SDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    {
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;

        for group in &self.groups {
            let lens =
                fusion_block_group_scratch_lens(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace.prepare_from_storages(
                lhs_alloc,
                rhs_alloc,
                dst_alloc,
                lens.lhs,
                lens.rhs,
                lens.destination,
                D::zero(),
            );
            execute_group_with_scratch_buffers(
                kernels,
                gemm,
                group,
                fusion_workspace.buffers_mut(),
                dst_data,
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    /// Storage-aware replay writing into a destination `TensorMap` while the
    /// LHS/RHS operands are raw canonical scratch slices (the dynamic route
    /// with an identity output transform).
    ///
    /// Pack scratch allocation origins: LHS pack from `lhs_alloc`, RHS pack
    /// from `rhs_alloc`, and matmul output scratch from the destination
    /// tensor's own storage.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_storage_raw_sources<
        A,
        G,
        D,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        SDst,
        SLhs,
        SRhs,
        DDst,
    >(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut StorageFusionBlockContractWorkspace<
            SLhs::Similar,
            SRhs::Similar,
            DDst::Similar,
        >,
        lhs_alloc: &SLhs,
        rhs_alloc: &SRhs,
        dst: &mut tenet_core::TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
        SLhs: SimilarStorage<D>,
        SLhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        SRhs: SimilarStorage<D>,
        SRhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DDst: HostWritableStorage<D> + SimilarStorage<D>,
        DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    {
        let dst_structure = Arc::clone(dst.structure());
        self.validate_replay_inputs(
            &dst_structure,
            dst.storage().len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        scale_all_blocks(
            kernels,
            &self.inactive_dst_scale_blocks,
            dst.data_mut(),
            beta,
        )?;

        for group in &self.groups {
            let lens =
                fusion_block_group_scratch_lens(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace.prepare_from_storages(
                lhs_alloc,
                rhs_alloc,
                dst.storage(),
                lens.lhs,
                lens.rhs,
                lens.destination,
                D::zero(),
            );
            execute_group_with_scratch_buffers(
                kernels,
                gemm,
                group,
                fusion_workspace.buffers_mut(),
                dst.data_mut(),
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    /// Executes the contraction purely over storage handles.
    ///
    /// This is the device-side replay seam: the bounds require only
    /// [`TensorStorage`], so no host-slice contract leaks into the path. It
    /// supports exactly the fully-direct coupled-layout case with `alpha = 1`,
    /// `beta = 0` and no inactive destination blocks; every other case must
    /// use the host replay paths until the corresponding device kernels
    /// (pack/scatter, scale, tree transforms) exist behind their own seams.
    #[allow(dead_code)]
    pub fn execute_direct_on_storage<G, D, DDst, DLhs, DRhs>(
        &self,
        gemm: &mut G,
        dst: &mut DDst,
        lhs: &DLhs,
        rhs: &DRhs,
    ) -> Result<(), OperationError>
    where
        G: StorageGemm<D, DDst, DLhs, DRhs>,
        DDst: TensorStorage<D>,
        DLhs: TensorStorage<D>,
        DRhs: TensorStorage<D>,
    {
        if !self.inactive_dst_scale_blocks.is_empty() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "storage-direct replay requires full destination coverage",
            });
        }
        for group in &self.groups {
            let (Some(lhs_base), Some(rhs_base), Some(dst_base)) = (
                group.lhs.direct_offset,
                group.rhs.direct_offset,
                group.dst.direct_offset,
            ) else {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "storage-direct replay requires the coupled-sector matrix layout",
                });
            };
            validate_storage_range(lhs.len(), lhs_base, group.lhs.rows, group.lhs.cols)?;
            validate_storage_range(rhs.len(), rhs_base, group.rhs.rows, group.rhs.cols)?;
            validate_storage_range(dst.len(), dst_base, group.dst.rows, group.dst.cols)?;
            gemm.matmul_range_into(
                dst,
                dst_base,
                lhs,
                lhs_base,
                rhs,
                rhs_base,
                group.lhs.rows,
                group.lhs.cols,
                group.rhs.cols,
            )?;
        }
        Ok(())
    }

    fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("lhs", &self.lhs_structure, lhs_structure)?;
        validate_structure_identity("rhs", &self.rhs_structure, rhs_structure)
    }

    fn validate_replay_inputs(
        &self,
        dst_structure: &Arc<BlockStructure>,
        dst_len: usize,
        lhs_structure: &Arc<BlockStructure>,
        lhs_len: usize,
        rhs_structure: &Arc<BlockStructure>,
        rhs_len: usize,
    ) -> Result<(), OperationError> {
        self.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
        validate_storage_len(dst_structure, dst_len)?;
        validate_storage_len(lhs_structure, lhs_len)?;
        validate_storage_len(rhs_structure, rhs_len)
    }
}

/// Placement-aware block GEMM over storage ranges.
///
/// The device-side replay seam for canonical fusion-block contraction:
/// `dst[dst_offset..][rows x cols] = lhs[lhs_offset..][rows x contracted] *
/// rhs[rhs_offset..][contracted x cols]` as column-major matrices, with no
/// host-slice contract in the trait. The host implementation wraps a
/// a rank-2 GEMM backend; device implementations submit kernels against
/// device storage handles.
pub trait StorageGemm<D, DDst, DLhs, DRhs> {
    #[allow(clippy::too_many_arguments)]
    fn matmul_range_into(
        &mut self,
        dst: &mut DDst,
        dst_offset: usize,
        lhs: &DLhs,
        lhs_offset: usize,
        rhs: &DRhs,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError>;
}

fn validate_storage_range(
    storage_len: usize,
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<(), OperationError> {
    let len = rows
        .checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    let end = base
        .checked_add(len)
        .ok_or(OperationError::ElementCountOverflow)?;
    if end > storage_len {
        return Err(OperationError::ElementCountMismatch {
            expected: end,
            actual: storage_len,
        });
    }
    Ok(())
}

fn validate_storage_len(
    structure: &BlockStructure,
    actual_len: usize,
) -> Result<(), OperationError> {
    let expected = structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if actual_len != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: actual_len,
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct CanonicalFusionBlockContractGroupPlan {
    pub lhs: FusionBlockMatrixGroup,
    pub rhs: FusionBlockMatrixGroup,
    pub dst: FusionBlockMatrixGroup,
}

impl CanonicalFusionBlockContractGroupPlan {
    /// True when GEMM can read both operands from storage and write the
    /// destination group matrix in place (no pack, no scatter).
    fn is_fully_direct(&self, trivial_scale: bool) -> bool {
        trivial_scale
            && self.lhs.direct_offset.is_some()
            && self.rhs.direct_offset.is_some()
            && self.dst.direct_offset.is_some()
    }

    /// Validates and packages a group triple; called by the compile layer.
    pub fn new(
        lhs: FusionBlockMatrixGroup,
        rhs: FusionBlockMatrixGroup,
        dst: FusionBlockMatrixGroup,
    ) -> Result<Self, OperationError> {
        if lhs.cols != rhs.rows {
            return Err(OperationError::ShapeMismatch {
                dst: vec![lhs.cols],
                src: vec![rhs.rows],
            });
        }
        if dst.rows != lhs.rows || dst.cols != rhs.cols {
            return Err(OperationError::ShapeMismatch {
                dst: vec![dst.rows, dst.cols],
                src: vec![lhs.rows, rhs.cols],
            });
        }

        Ok(Self { lhs, rhs, dst })
    }
}

impl<T> HostFusionBlockContractBuffers<T>
where
    T: Clone + Zero,
{
    fn prepare(
        &mut self,
        lhs_rows: usize,
        contracted: usize,
        rhs_cols: usize,
    ) -> Result<(), OperationError> {
        let lens = fusion_block_group_scratch_lens(lhs_rows, contracted, rhs_cols)?;
        self.packed.lhs_mut().resize_filled(lens.lhs, T::zero());
        self.packed.rhs_mut().resize_filled(lens.rhs, T::zero());
        self.packed
            .destination_mut()
            .resize_filled(lens.destination, T::zero());
        Ok(())
    }

    fn clear_inputs(&mut self, clear_lhs: bool, clear_rhs: bool) {
        if clear_lhs {
            self.packed.lhs_mut().fill(T::zero());
        }
        if clear_rhs {
            self.packed.rhs_mut().fill(T::zero());
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FusionBlockContractScratchLens {
    lhs: usize,
    rhs: usize,
    destination: usize,
}

pub fn fusion_block_group_scratch_lens(
    lhs_rows: usize,
    contracted: usize,
    rhs_cols: usize,
) -> Result<FusionBlockContractScratchLens, OperationError> {
    let lhs = lhs_rows
        .checked_mul(contracted)
        .ok_or(OperationError::ElementCountOverflow)?;
    let rhs = contracted
        .checked_mul(rhs_cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination = lhs_rows
        .checked_mul(rhs_cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    Ok(FusionBlockContractScratchLens {
        lhs,
        rhs,
        destination,
    })
}

#[derive(Clone, Debug)]
pub struct FusionBlockMatrixGroup {
    pub coupled: SectorId,
    pub rows: usize,
    pub cols: usize,
    // False only when the group's subblocks cover the packed matrix exactly.
    // Sparse fusion layouts keep this true so stale workspace cannot leak into GEMM.
    pub needs_clear: bool,
    // Storage offset of the group matrix when the operand's subblocks already
    // form it in place (coupled-sector matrix layout, unit coefficients):
    // packing is the identity copy and replay can hand storage to GEMM
    // directly.
    pub direct_offset: Option<usize>,
    pub block_indices: Vec<usize>,
    pub subblocks: Vec<FusionSubblockMatrixLayout>,
}

pub fn direct_group_matrix_offset(
    subblocks: &[FusionSubblockMatrixLayout],
    covers_matrix: bool,
) -> Option<usize> {
    if !covers_matrix {
        return None;
    }
    let mut base: Option<isize> = None;
    for subblock in subblocks {
        if subblock.coefficient != 1.0 {
            return None;
        }
        let strides_match = subblock
            .block
            .shape
            .iter()
            .zip(subblock.block.strides.iter().zip(&subblock.matrix_strides))
            .all(|(&dim, (&stride, &matrix_stride))| dim <= 1 || stride == matrix_stride);
        if !strides_match {
            return None;
        }
        let offset = subblock.block.offset - subblock.matrix_offset;
        if offset < 0 {
            return None;
        }
        match base {
            None => base = Some(offset),
            Some(existing) if existing != offset => return None,
            Some(_) => {}
        }
    }
    base.and_then(|offset| usize::try_from(offset).ok())
}

#[derive(Clone, Debug)]
pub struct FusionSubblockMatrixLayout {
    pub block: FusionStridedBlockLayout,
    pub matrix_offset: isize,
    pub matrix_strides: Vec<isize>,
    pub coefficient: f64,
}

#[derive(Clone, Debug)]
pub struct FusionStridedBlockLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
}

#[derive(Clone, Debug)]
pub struct FusionScaleBlockLayout {
    pub block: FusionStridedBlockLayout,
}

pub fn fusion_scale_block_layouts_excluding(
    structure: &BlockStructure,
    excluded_blocks: &HashSet<usize>,
) -> Result<Vec<FusionScaleBlockLayout>, OperationError> {
    let mut layouts = Vec::with_capacity(structure.block_count());
    for block_index in 0..structure.block_count() {
        if excluded_blocks.contains(&block_index) {
            continue;
        }
        let block = structure.block(block_index)?;
        layouts.push(FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: block.shape().to_vec(),
                strides: strides_to_isize(block.strides())?,
                offset: offset_to_isize(block.offset())?,
            },
        });
    }
    Ok(layouts)
}

fn pack_group<A, T>(
    kernels: &mut A,
    group: &FusionBlockMatrixGroup,
    data: &[T],
    packed: &mut [T],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + RecouplingCoefficientAction<f64>,
{
    for layout in &group.subblocks {
        kernels.copy_scale_strided(
            packed,
            data,
            &layout.block.shape,
            &layout.matrix_strides,
            &layout.block.strides,
            layout.matrix_offset,
            layout.block.offset,
            false,
            T::coefficient_as_data(layout.coefficient),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_group_with_scratch_buffers<A, G, D, LhsScratch, RhsScratch, DestinationScratch>(
    kernels: &mut A,
    gemm: &mut G,
    group: &CanonicalFusionBlockContractGroupPlan,
    scratch: &mut FusionBlockContractScratchBuffers<LhsScratch, RhsScratch, DestinationScratch>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    G: Rank2Gemm<D>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    LhsScratch: HostWritableStorage<D>,
    RhsScratch: HostWritableStorage<D>,
    DestinationScratch: HostWritableStorage<D>,
{
    if group.lhs.direct_offset.is_none() {
        pack_group(
            kernels,
            &group.lhs,
            lhs_data,
            scratch.lhs_mut().as_mut_slice(),
        )?;
    }
    if group.rhs.direct_offset.is_none() {
        pack_group(
            kernels,
            &group.rhs,
            rhs_data,
            scratch.rhs_mut().as_mut_slice(),
        )?;
    }
    let dst_direct = if alpha.is_one() && beta.is_zero() {
        group.dst.direct_offset
    } else {
        None
    };
    let (lhs_scratch, rhs_scratch, dst_scratch) = scratch.inputs_and_destination_mut();
    let lhs_slice = direct_or_scratch_slice(
        lhs_data,
        group.lhs.direct_offset,
        group.lhs.rows,
        group.lhs.cols,
        lhs_scratch.as_slice(),
    )?;
    let rhs_slice = direct_or_scratch_slice(
        rhs_data,
        group.rhs.direct_offset,
        group.rhs.rows,
        group.rhs.cols,
        rhs_scratch.as_slice(),
    )?;
    match dst_direct {
        Some(base) => {
            let dst_slice = direct_slice_mut(dst_data, base, group.dst.rows, group.dst.cols)?;
            matmul_group_plan(gemm, group, lhs_slice, rhs_slice, dst_slice)
        }
        None => {
            matmul_group_plan(
                gemm,
                group,
                lhs_slice,
                rhs_slice,
                dst_scratch.as_mut_slice(),
            )?;
            scatter_group(
                kernels,
                &group.dst,
                dst_data,
                dst_scratch.as_slice(),
                alpha,
                beta,
            )
        }
    }
}

fn direct_matrix_len(rows: usize, cols: usize) -> Result<usize, OperationError> {
    rows.checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)
}

fn direct_or_scratch_slice<'a, T>(
    data: &'a [T],
    direct_offset: Option<usize>,
    rows: usize,
    cols: usize,
    scratch: &'a [T],
) -> Result<&'a [T], OperationError> {
    match direct_offset {
        Some(base) => {
            let len = direct_matrix_len(rows, cols)?;
            let end = base
                .checked_add(len)
                .ok_or(OperationError::ElementCountOverflow)?;
            data.get(base..end)
                .ok_or(OperationError::ElementCountMismatch {
                    expected: end,
                    actual: data.len(),
                })
        }
        None => Ok(scratch),
    }
}

fn direct_slice_mut<T>(
    data: &mut [T],
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<&mut [T], OperationError> {
    let len = direct_matrix_len(rows, cols)?;
    let end = base
        .checked_add(len)
        .ok_or(OperationError::ElementCountOverflow)?;
    let actual = data.len();
    data.get_mut(base..end)
        .ok_or(OperationError::ElementCountMismatch {
            expected: end,
            actual,
        })
}

fn scatter_group<A, T>(
    kernels: &mut A,
    group: &FusionBlockMatrixGroup,
    data: &mut [T],
    packed: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    for layout in &group.subblocks {
        kernels.axpby_strided(
            data,
            packed,
            &layout.block.shape,
            &layout.block.strides,
            &layout.matrix_strides,
            layout.block.offset,
            layout.matrix_offset,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

fn scale_all_blocks<A, T>(
    kernels: &mut A,
    blocks: &[FusionScaleBlockLayout],
    data: &mut [T],
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One + PartialEq,
{
    if beta.is_one() {
        return Ok(());
    }
    for layout in blocks {
        kernels.scale_strided(
            data,
            &layout.block.shape,
            &layout.block.strides,
            layout.block.offset,
            beta,
        )?;
    }
    Ok(())
}

fn matmul_group_plan<G, D>(
    gemm: &mut G,
    group: &CanonicalFusionBlockContractGroupPlan,
    lhs: &[D],
    rhs: &[D],
    dst: &mut [D],
) -> Result<(), OperationError>
where
    G: Rank2Gemm<D>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    gemm.matmul_rank2(
        dst,
        lhs,
        rhs,
        group.lhs.rows,
        group.lhs.cols,
        group.rhs.cols,
    )
}
