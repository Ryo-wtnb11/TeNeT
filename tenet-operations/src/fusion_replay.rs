//! Symmetry-free replay half of the core fusion-block contraction:
//! plan data (offsets, strides, coefficients), direct batched-GEMM execution,
//! workspaces, and the storage-direct device seam. The symmetric compile layer
//! builds these plans; nothing here consumes fusion rules.

use std::collections::HashSet;
use std::sync::Arc;

use num_traits::One;
use tenet_core::{
    BlockStructure, HostReadableStorage, HostWritableStorage, Placement, ScratchStorage, SectorId,
    SimilarStorage, TensorStorage,
};
use tenet_dense::{strided_batch_runs, DenseGemmBatchJob};

use crate::placement::ReportsPlacement;
use crate::profile::TensorContractFusionProfile;
use crate::storage_scratch::StorageFusionBlockContractWorkspace;
use crate::strided::{offset_to_isize, strides_to_isize};
use crate::structure_identity::validate_structure_identity;
use crate::{DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction};

/// Core contraction replay accepts only operands whose coupled-sector
/// matrices sit directly in storage (the one product layout). Anything else
/// is a plan-construction bug, not a runtime fallback.
const NON_COUPLED_OPERAND_MESSAGE: &str =
    "core fusion-block replay requires the coupled sector matrix layout";

/// Rank-2 column-major GEMM over host slices: the only capability the replay
/// half needs from a contraction backend. The symmetric layer adapts its
/// contraction backends onto this.
pub trait Rank2Gemm<D> {
    /// `dst = alpha * lhs * rhs + beta * dst` over column-major matrices
    /// (BLAS gemm semantics).
    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;

    /// Executes a batch of independent GEMMs addressed by offsets into shared
    /// buffers: for each job, the `rows x cols` destination matrix at
    /// `dst[job.dst_offset..]` receives `alpha * lhs_part * rhs_part + beta *
    /// dst_part` (column-major). The plan layer guarantees the destination
    /// ranges of a batch are pairwise disjoint, so implementations may run
    /// jobs in any order or concurrently. The default executes them in order.
    fn matmul_rank2_batch(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        for job in jobs {
            let lhs_slice = direct_slice(lhs, job.lhs_offset, job.rows, job.contracted)?;
            let rhs_slice = direct_slice(rhs, job.rhs_offset, job.contracted, job.cols)?;
            let dst_slice = direct_slice_mut(dst, job.dst_offset, job.rows, job.cols)?;
            self.matmul_rank2(
                dst_slice,
                lhs_slice,
                rhs_slice,
                job.rows,
                job.contracted,
                job.cols,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }
}

/// One GEMM of a fully-direct core batch. Offsets address coupled-sector
/// matrices inside the operands' storage buffers; a batch's destination
/// ranges never overlap (validated when the plan is assembled).
pub type Rank2GemmBatchJob = DenseGemmBatchJob;

pub struct HostFusionBlockContractWorkspace<T> {
    _scalar: std::marker::PhantomData<T>,
}

pub type FusionBlockContractWorkspace<T> = HostFusionBlockContractWorkspace<T>;

impl<T> Default for HostFusionBlockContractWorkspace<T> {
    fn default() -> Self {
        Self {
            _scalar: std::marker::PhantomData,
        }
    }
}

impl<T> ReportsPlacement for HostFusionBlockContractWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

#[derive(Clone, Debug)]
pub struct FusionBlockContractPlan {
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
    inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
    groups: Vec<FusionBlockContractGroupPlan>,
    // Precompiled batch over the groups when every operand is a direct
    // coupled-sector matrix; replay hands it to the backend in a single
    // `matmul_rank2_batch` call. `None` marks a non-direct plan — a valid
    // compile output used only for route decisions, never replayed.
    direct_batch: Option<Vec<Rank2GemmBatchJob>>,
    // Plan-time run partition of `direct_batch` (see issue #103): the backend
    // reads it to route each run without recomputing the partition per replay.
    // Empty for a non-direct plan. A backend-agnostic shape fact, so it lives in
    // this operations-layer plan struct while the route choice (strided seam vs
    // grouped) stays at the dense-executor boundary.
    direct_batch_runs: Vec<usize>,
}

impl FusionBlockContractPlan {
    /// True when every group reads and writes coupled-sector matrices
    /// directly in storage — the only layouts the replay executes. The
    /// symmetric route layer sends anything else through dynamic
    /// materialization.
    pub fn is_fully_direct(&self) -> bool {
        self.direct_batch.is_some()
    }

    /// Assembles a compiled plan; called by the symmetric compile layer.
    /// Non-direct groups yield a route-decision-only plan (no batch);
    /// overlapping destination ranges are rejected because that invariant is
    /// what lets backends run the batch jobs concurrently.
    pub fn from_parts(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<FusionBlockContractGroupPlan>,
    ) -> Result<Self, OperationError> {
        let direct_batch = compile_direct_batch(&groups)?;
        let direct_batch_runs = direct_batch
            .as_deref()
            .map(strided_batch_runs)
            .unwrap_or_default();
        Ok(Self {
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
            direct_batch,
            direct_batch_runs,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_raw<A, G, D>(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
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

        let _ = fusion_workspace;
        gemm.matmul_rank2_batch(
            dst_data,
            lhs_data,
            rhs_data,
            self.direct_batch()?,
            alpha,
            beta,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_raw_profiled<A, G, D>(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
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
        profile.core_validate += start.elapsed();

        let start = std::time::Instant::now();
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;
        profile.core_scale += start.elapsed();

        let _ = fusion_workspace;
        let batch = self.direct_batch()?;
        profile.core_contract_groups += batch.len();
        let start = std::time::Instant::now();
        gemm.matmul_rank2_batch(dst_data, lhs_data, rhs_data, batch, alpha, beta)?;
        profile.core_direct_gemm_groups += batch.len();
        profile.core_matmul += start.elapsed();

        profile.core_contract_total += total_start.elapsed();
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
        let _ = fusion_workspace;
        gemm.matmul_rank2_batch(
            dst.data_mut(),
            lhs_data,
            rhs_data,
            self.direct_batch()?,
            alpha,
            beta,
        )?;
        Ok(())
    }

    fn direct_batch(&self) -> Result<&[Rank2GemmBatchJob], OperationError> {
        self.direct_batch
            .as_deref()
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: NON_COUPLED_OPERAND_MESSAGE,
            })
    }

    /// Plan-time run partition of [`Self::direct_batch`]; handed to the backend
    /// alongside the jobs so it routes runs without recomputing the partition.
    fn direct_batch_runs(&self) -> &[usize] {
        &self.direct_batch_runs
    }

    /// Storage-aware raw replay for callers whose operands are scratch buffers
    /// rather than `TensorMap`s (the dynamic core route).
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
        _lhs_alloc: &SLhs,
        _rhs_alloc: &SRhs,
        _dst_alloc: &SDst,
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

        let _ = fusion_workspace;
        gemm.matmul_rank2_batch(
            dst_data,
            lhs_data,
            rhs_data,
            self.direct_batch()?,
            alpha,
            beta,
        )?;
        Ok(())
    }

    /// Storage-aware replay writing into a destination `TensorMap` while the
    /// LHS/RHS operands are raw core scratch slices (the dynamic route
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
        _lhs_alloc: &SLhs,
        _rhs_alloc: &SRhs,
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

        let _ = fusion_workspace;
        gemm.matmul_rank2_batch(
            dst.data_mut(),
            lhs_data,
            rhs_data,
            self.direct_batch()?,
            alpha,
            beta,
        )?;
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
        self.execute_direct_on_storage_prezeroed(gemm, dst, lhs, rhs)
    }

    /// [`Self::execute_direct_on_storage`] for a destination the caller
    /// guarantees is zero-filled: destination blocks with no contributing
    /// GEMM (the ones the host path scales by `beta = 0`) are left
    /// untouched, which is exactly the overwrite semantics on a zeroed
    /// buffer. The active blocks are still fully overwritten.
    pub fn execute_direct_on_storage_prezeroed<G, D, DDst, DLhs, DRhs>(
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
/// The device-side replay seam for core fusion-block contraction:
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
pub struct FusionBlockContractGroupPlan {
    pub lhs: FusionBlockMatrixGroup,
    pub rhs: FusionBlockMatrixGroup,
    pub dst: FusionBlockMatrixGroup,
}

impl FusionBlockContractGroupPlan {
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

fn direct_matrix_len(rows: usize, cols: usize) -> Result<usize, OperationError> {
    rows.checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)
}

/// Checked mutable `rows x cols` column-major matrix slice at `base`; the
/// bounds-checking counterpart of the batch-job offset addressing.
pub fn direct_slice_mut<T>(
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

/// Lowers the group list to batch jobs. A non-direct group yields `None`
/// (route-decision-only plan); a direct batch additionally enforces pairwise
/// disjoint destination ranges so the structural lowering may order jobs into
/// same-shape strided runs before handing them to the dense backend.
fn compile_direct_batch(
    groups: &[FusionBlockContractGroupPlan],
) -> Result<Option<Vec<Rank2GemmBatchJob>>, OperationError> {
    let mut jobs = Vec::with_capacity(groups.len());
    for group in groups {
        let (Some(lhs_offset), Some(rhs_offset), Some(dst_offset)) = (
            group.lhs.direct_offset,
            group.rhs.direct_offset,
            group.dst.direct_offset,
        ) else {
            return Ok(None);
        };
        jobs.push(Rank2GemmBatchJob {
            dst_offset,
            lhs_offset,
            rhs_offset,
            rows: group.lhs.rows,
            contracted: group.lhs.cols,
            cols: group.rhs.cols,
        });
    }
    let mut dst_ranges: Vec<(usize, usize)> = jobs
        .iter()
        .map(|job| direct_matrix_len(job.rows, job.cols).map(|len| (job.dst_offset, len)))
        .collect::<Result<_, _>>()?;
    dst_ranges.sort_unstable();
    for pair in dst_ranges.windows(2) {
        let (base, len) = pair[0];
        let end = base
            .checked_add(len)
            .ok_or(OperationError::ElementCountOverflow)?;
        if end > pair[1].0 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "core contraction groups must write disjoint destination ranges",
            });
        }
    }
    jobs.sort_by_key(|job| {
        (
            job.rows,
            job.contracted,
            job.cols,
            job.dst_offset,
            job.lhs_offset,
            job.rhs_offset,
        )
    });
    Ok(Some(jobs))
}

/// Checked shared `rows x cols` column-major matrix slice at `base`; the
/// bounds-checking counterpart of the batch-job offset addressing.
pub fn direct_slice<T>(
    data: &[T],
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<&[T], OperationError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_group(rows: usize, cols: usize, offset: Option<usize>) -> FusionBlockMatrixGroup {
        FusionBlockMatrixGroup {
            coupled: SectorId::new(0),
            rows,
            cols,
            needs_clear: false,
            direct_offset: offset,
            block_indices: Vec::new(),
            subblocks: Vec::new(),
        }
    }

    fn group_plan(
        dims: (usize, usize, usize),
        offsets: (usize, usize, usize),
    ) -> FusionBlockContractGroupPlan {
        let (rows, contracted, cols) = dims;
        let (lhs_offset, rhs_offset, dst_offset) = offsets;
        FusionBlockContractGroupPlan::new(
            direct_group(rows, contracted, Some(lhs_offset)),
            direct_group(contracted, cols, Some(rhs_offset)),
            direct_group(rows, cols, Some(dst_offset)),
        )
        .unwrap()
    }

    #[test]
    fn direct_batch_lowers_disjoint_groups() {
        let groups = vec![
            group_plan((2, 3, 4), (0, 0, 0)),
            group_plan((3, 2, 5), (6, 12, 8)),
        ];
        let batch = compile_direct_batch(&groups).unwrap().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].dst_offset, 0);
        assert_eq!(batch[0].rows * batch[0].cols, 8);
        assert_eq!(batch[1].dst_offset, 8);
        assert_eq!(batch[1].contracted, 2);
    }

    #[test]
    fn direct_batch_orders_same_shape_groups_as_strided_runs() {
        let groups = [2usize, 0, 4, 1, 3]
            .into_iter()
            .map(|block| {
                let base = block * 4;
                group_plan((2, 2, 2), (base, base, base))
            })
            .collect::<Vec<_>>();
        let batch = compile_direct_batch(&groups).unwrap().unwrap();
        let offsets = batch
            .iter()
            .map(|job| (job.dst_offset, job.lhs_offset, job.rhs_offset))
            .collect::<Vec<_>>();
        assert_eq!(
            offsets,
            vec![(0, 0, 0), (4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 16)]
        );
    }

    #[test]
    fn direct_batch_plan_bakes_run_partition() {
        // Five same-shape groups fold into one length-5 constant-stride run;
        // the plan stores that partition (issue #103) so the backend routes it
        // without recomputing. Storage matches the shared partition helper.
        let groups = [2usize, 0, 4, 1, 3]
            .into_iter()
            .map(|block| {
                let base = block * 4;
                group_plan((2, 2, 2), (base, base, base))
            })
            .collect::<Vec<_>>();
        let structure = Arc::new(BlockStructure::trivial(&[20]).unwrap());
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            groups,
        )
        .unwrap();
        assert_eq!(plan.direct_batch_runs(), &[5]);
        assert_eq!(
            plan.direct_batch_runs(),
            strided_batch_runs(plan.direct_batch().unwrap())
        );
        assert_eq!(
            plan.direct_batch_runs().iter().sum::<usize>(),
            plan.direct_batch().unwrap().len()
        );
    }

    #[test]
    fn non_direct_plan_has_empty_run_partition() {
        let plan = FusionBlockContractGroupPlan::new(
            direct_group(2, 3, None),
            direct_group(3, 4, Some(0)),
            direct_group(2, 4, Some(0)),
        )
        .unwrap();
        let structure = Arc::new(BlockStructure::trivial(&[12]).unwrap());
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![plan],
        )
        .unwrap();
        assert!(!plan.is_fully_direct());
        assert!(plan.direct_batch_runs().is_empty());
    }

    #[test]
    fn direct_batch_rejects_overlapping_destination_ranges() {
        // First dst covers [0, 8); second starts at 7.
        let groups = vec![
            group_plan((2, 3, 4), (0, 0, 0)),
            group_plan((3, 2, 5), (6, 12, 7)),
        ];
        let error = compile_direct_batch(&groups).unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope { .. }
        ));
    }

    #[test]
    fn non_direct_group_yields_route_decision_only_plan() {
        let plan = FusionBlockContractGroupPlan::new(
            direct_group(2, 3, None),
            direct_group(3, 4, Some(0)),
            direct_group(2, 4, Some(0)),
        )
        .unwrap();
        assert!(compile_direct_batch(&[plan]).unwrap().is_none());
    }

    #[derive(Default)]
    struct CountingBatchGemm {
        rank2_calls: usize,
        batch_calls: usize,
        last_batch_len: usize,
    }

    impl Rank2Gemm<f64> for CountingBatchGemm {
        fn matmul_rank2(
            &mut self,
            _dst: &mut [f64],
            _lhs: &[f64],
            _rhs: &[f64],
            _rows: usize,
            _contracted: usize,
            _cols: usize,
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            self.rank2_calls += 1;
            Ok(())
        }

        fn matmul_rank2_batch(
            &mut self,
            _dst: &mut [f64],
            _lhs: &[f64],
            _rhs: &[f64],
            jobs: &[Rank2GemmBatchJob],
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            self.batch_calls += 1;
            self.last_batch_len = jobs.len();
            Ok(())
        }
    }

    #[test]
    fn profiled_direct_replay_uses_one_batched_gemm_call() {
        let groups = vec![
            group_plan((2, 3, 4), (0, 0, 0)),
            group_plan((3, 2, 5), (6, 12, 8)),
        ];
        let structure = Arc::new(BlockStructure::trivial(&[23]).unwrap());
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            groups,
        )
        .unwrap();
        let mut kernels = crate::StridedHostKernelAdapter;
        let mut gemm = CountingBatchGemm::default();
        let mut workspace = FusionBlockContractWorkspace::<f64>::default();
        let mut profile = TensorContractFusionProfile::default();
        let lhs = vec![0.0; 23];
        let rhs = vec![0.0; 23];
        let mut dst = vec![0.0; 23];

        plan.execute_raw_profiled(
            &mut kernels,
            &mut gemm,
            &mut workspace,
            &structure,
            &mut dst,
            &structure,
            &lhs,
            &structure,
            &rhs,
            1.0,
            0.0,
            &mut profile,
        )
        .unwrap();

        assert_eq!(gemm.batch_calls, 1);
        assert_eq!(gemm.rank2_calls, 0);
        assert_eq!(gemm.last_batch_len, 2);
        assert_eq!(profile.core_contract_groups, 2);
        assert_eq!(profile.core_direct_gemm_groups, 2);
    }
}
