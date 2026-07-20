//! Symmetry-free replay half of the core fusion-block contraction:
//! plan data (offsets, strides, coefficients), group-local host execution,
//! workspaces, and the storage-direct device seam. The symmetric compile layer
//! builds these plans; nothing here consumes fusion rules.

use std::collections::HashSet;
use std::sync::Arc;

use num_traits::One;
use tenet_core::{
    BlockStructure, HostReadableStorage, HostWritableStorage, Placement, ScratchStorage, SectorId,
    SimilarStorage, TensorStorage,
};
pub use tenet_dense::MatrixOp;
use tenet_dense::{strided_batch_runs, DenseGemmBatchJob};

use crate::host_scalar_kernels::validate_raw_strided_bounds;
use crate::host_scratch::HostScratchBuffer;
use crate::placement::ReportsPlacement;
use crate::profile::TensorContractFusionProfile;
use crate::storage_scratch::StorageFusionBlockContractWorkspace;
use crate::strided::{offset_to_isize, strides_to_isize};
use crate::structure_identity::validate_structure_identity;
use crate::transform_structure::validate_destination_layouts_injective;
use crate::{DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction};

/// Storage-handle replay accepts only operands whose coupled-sector matrices
/// sit directly in storage. Host replay also supports group-local pack/scatter,
/// but storage/device kernels do not yet expose that capability.
const NON_COUPLED_OPERAND_MESSAGE: &str =
    "storage-handle core fusion-block replay requires the coupled sector matrix layout";

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
    ///
    /// `runs` is the batch's plan-time run partition (see issue #103); backends
    /// that route runs read it, the serial default ignores it.
    fn matmul_rank2_batch(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        let _ = runs;
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

    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_batch_with_ops(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        if lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity {
            return self.matmul_rank2_batch(dst, lhs, rhs, jobs, runs, alpha, beta);
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "rank-2 GEMM backend does not implement transpose/adjoint batches",
        })
    }
}

/// One GEMM job addressed inside its selected source and destination buffers.
/// Fully-direct jobs use tensor storage; irregular jobs use group-local scratch
/// for exactly the operands selected by their execution class.
pub type Rank2GemmBatchJob = DenseGemmBatchJob;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusionGroupExecutionClass(u8);

impl FusionGroupExecutionClass {
    const PACK_LHS: u8 = 1;
    const PACK_RHS: u8 = 2;
    const SCATTER_DST: u8 = 4;

    fn compile(group: &FusionBlockContractGroupPlan) -> Self {
        let mut bits = 0;
        if group.lhs.direct_offset.is_none() {
            bits |= Self::PACK_LHS;
        }
        if group.rhs.direct_offset.is_none() {
            bits |= Self::PACK_RHS;
        }
        if group.dst.direct_offset.is_none() {
            bits |= Self::SCATTER_DST;
        }
        Self(bits)
    }

    #[inline]
    fn is_direct(self) -> bool {
        self.0 == 0
    }

    #[inline]
    fn packs_lhs(self) -> bool {
        self.0 & Self::PACK_LHS != 0
    }

    #[inline]
    fn packs_rhs(self) -> bool {
        self.0 & Self::PACK_RHS != 0
    }

    #[inline]
    fn scatters_dst(self) -> bool {
        self.0 & Self::SCATTER_DST != 0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FusionGroupScratchLayout {
    lhs_len: usize,
    rhs_offset: usize,
    rhs_len: usize,
    dst_offset: usize,
    dst_len: usize,
    total_len: usize,
}

#[derive(Clone, Debug)]
struct FusionIrregularGroupExecution {
    group_index: usize,
    class: FusionGroupExecutionClass,
    job: Rank2GemmBatchJob,
    scratch: FusionGroupScratchLayout,
}

type CompiledGroupExecution = (
    Vec<Rank2GemmBatchJob>,
    Vec<FusionIrregularGroupExecution>,
    usize,
);

pub struct HostFusionBlockContractWorkspace<T> {
    scratch: HostScratchBuffer<T>,
}

pub type FusionBlockContractWorkspace<T> = HostFusionBlockContractWorkspace<T>;

impl<T> Default for HostFusionBlockContractWorkspace<T> {
    fn default() -> Self {
        Self {
            scratch: HostScratchBuffer::default(),
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
    direct_batch: Vec<Rank2GemmBatchJob>,
    // Plan-time run partition of `direct_batch` (see issue #103): the backend
    // reads it to route each run without recomputing the partition per replay.
    // A backend-agnostic shape fact, so it lives in this operations-layer plan
    // while the route choice (strided seam vs grouped) stays at the
    // dense-executor boundary.
    direct_batch_runs: Vec<usize>,
    irregular: Vec<FusionIrregularGroupExecution>,
    max_irregular_scratch_len: usize,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
}

impl FusionBlockContractPlan {
    /// True when every group reads and writes coupled-sector matrices directly
    /// in storage.
    pub fn is_fully_direct(&self) -> bool {
        self.irregular.is_empty()
    }

    /// Assembles a compiled plan; called by the symmetric compile layer.
    /// Direct groups form one backend batch, while each irregular group owns a
    /// fixed group-local pack/scatter descriptor in coupled-sector order.
    /// Overlapping direct destination ranges are rejected because backends may
    /// run the direct batch concurrently.
    pub fn from_parts(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<FusionBlockContractGroupPlan>,
    ) -> Result<Self, OperationError> {
        Self::from_parts_with_ops(
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
            MatrixOp::Identity,
            MatrixOp::Identity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts_with_ops(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<FusionBlockContractGroupPlan>,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
    ) -> Result<Self, OperationError> {
        validate_compiled_plan_layouts(
            &dst_structure,
            &lhs_structure,
            &rhs_structure,
            &inactive_dst_scale_blocks,
            &groups,
            lhs_op,
            rhs_op,
        )?;
        let (direct_batch, irregular, max_irregular_scratch_len) =
            compile_group_execution(&groups)?;
        let direct_batch_runs = strided_batch_runs(&direct_batch);
        Ok(Self {
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
            direct_batch,
            direct_batch_runs,
            irregular,
            max_irregular_scratch_len,
            lhs_op,
            rhs_op,
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
        self.execute_host::<_, _, _, false>(
            kernels,
            gemm,
            fusion_workspace,
            dst_structure,
            dst_data,
            lhs_structure,
            lhs_data,
            rhs_structure,
            rhs_data,
            alpha,
            beta,
            None,
        )
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
        self.execute_host::<_, _, _, true>(
            kernels,
            gemm,
            fusion_workspace,
            dst_structure,
            dst_data,
            lhs_structure,
            lhs_data,
            rhs_structure,
            rhs_data,
            alpha,
            beta,
            Some(profile),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_host<A, G, D, const PROFILED: bool>(
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
        mut profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        let total_start = PROFILED.then(std::time::Instant::now);
        let start = PROFILED.then(std::time::Instant::now);
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
            profile.core_validate += start.elapsed();
        }

        if self.max_irregular_scratch_len != 0 {
            let start = PROFILED.then(std::time::Instant::now);
            fusion_workspace
                .scratch
                .resize_filled(self.max_irregular_scratch_len, D::zero());
            if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
                profile.core_workspace_prepare += start.elapsed();
            }
        } else {
            fusion_workspace.scratch.resize_filled(0, D::zero());
        }

        let start = PROFILED.then(std::time::Instant::now);
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;
        if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
            profile.core_scale += start.elapsed();
        }

        if PROFILED {
            if let Some(profile) = profile.as_deref_mut() {
                profile.core_contract_groups += self.groups.len();
            }
        }
        if !self.direct_batch.is_empty() {
            let start = PROFILED.then(std::time::Instant::now);
            self.execute_batch(
                gemm,
                dst_data,
                lhs_data,
                rhs_data,
                &self.direct_batch,
                &self.direct_batch_runs,
                alpha,
                beta,
            )?;
            if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
                profile.core_matmul += start.elapsed();
            }
            if PROFILED {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.core_direct_gemm_groups += self.direct_batch.len();
                }
            }
        }

        for execution in &self.irregular {
            let group_profile = if PROFILED {
                profile.as_deref_mut()
            } else {
                None
            };
            self.execute_irregular_group::<_, _, _, PROFILED>(
                kernels,
                gemm,
                fusion_workspace,
                execution,
                dst_data,
                lhs_data,
                rhs_data,
                alpha,
                beta,
                group_profile,
            )?;
        }

        if let (Some(start), Some(profile)) = (total_start, profile) {
            profile.core_contract_total += start.elapsed();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_irregular_group<A, G, D, const PROFILED: bool>(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
        execution: &FusionIrregularGroupExecution,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        alpha: D,
        beta: D,
        mut profile: Option<&mut TensorContractFusionProfile>,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        let group = &self.groups[execution.group_index];
        let scratch = &mut fusion_workspace.scratch.as_mut_slice()[..execution.scratch.total_len];
        if execution.class.packs_lhs() {
            let lhs = &mut scratch[..execution.scratch.lhs_len];
            if group.lhs.needs_clear {
                lhs.fill(D::zero());
            }
            let start = PROFILED.then(std::time::Instant::now);
            pack_group(kernels, &group.lhs, lhs_data, lhs)?;
            if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
                profile.core_pack_lhs += start.elapsed();
            }
        } else if PROFILED {
            if let Some(profile) = profile.as_deref_mut() {
                profile.core_direct_pack_skips += 1;
            }
        }
        if execution.class.packs_rhs() {
            let start_index = execution.scratch.rhs_offset;
            let end = start_index + execution.scratch.rhs_len;
            let rhs = &mut scratch[start_index..end];
            if group.rhs.needs_clear {
                rhs.fill(D::zero());
            }
            let start = PROFILED.then(std::time::Instant::now);
            pack_group(kernels, &group.rhs, rhs_data, rhs)?;
            if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
                profile.core_pack_rhs += start.elapsed();
            }
        } else if PROFILED {
            if let Some(profile) = profile.as_deref_mut() {
                profile.core_direct_pack_skips += 1;
            }
        }

        let start = PROFILED.then(std::time::Instant::now);
        if execution.class.scatters_dst() {
            let (inputs, destination) = scratch.split_at_mut(execution.scratch.dst_offset);
            let destination = &mut destination[..execution.scratch.dst_len];
            let lhs = if execution.class.packs_lhs() {
                &inputs[..execution.scratch.lhs_len]
            } else {
                lhs_data
            };
            let rhs = if execution.class.packs_rhs() {
                let start = execution.scratch.rhs_offset;
                &inputs[start..start + execution.scratch.rhs_len]
            } else {
                rhs_data
            };
            self.execute_batch(
                gemm,
                destination,
                lhs,
                rhs,
                std::slice::from_ref(&execution.job),
                &[1],
                alpha,
                D::zero(),
            )?;
            if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
                profile.core_matmul += start.elapsed();
            }
            let start = PROFILED.then(std::time::Instant::now);
            scatter_group(kernels, &group.dst, dst_data, destination, beta)?;
            if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
                profile.core_scatter += start.elapsed();
            }
        } else {
            let lhs = if execution.class.packs_lhs() {
                &scratch[..execution.scratch.lhs_len]
            } else {
                lhs_data
            };
            let rhs = if execution.class.packs_rhs() {
                let start = execution.scratch.rhs_offset;
                &scratch[start..start + execution.scratch.rhs_len]
            } else {
                rhs_data
            };
            self.execute_batch(
                gemm,
                dst_data,
                lhs,
                rhs,
                std::slice::from_ref(&execution.job),
                &[1],
                alpha,
                beta,
            )?;
            if let (Some(start), Some(profile)) = (start, profile.as_deref_mut()) {
                profile.core_matmul += start.elapsed();
            }
            if PROFILED {
                if let Some(profile) = profile {
                    profile.core_direct_gemm_groups += 1;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_batch<G, D>(
        &self,
        gemm: &mut G,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        G: Rank2Gemm<D>,
        D: Copy,
    {
        if self.lhs_op == MatrixOp::Identity && self.rhs_op == MatrixOp::Identity {
            gemm.matmul_rank2_batch(dst, lhs, rhs, jobs, runs, alpha, beta)
        } else {
            gemm.matmul_rank2_batch_with_ops(
                dst,
                lhs,
                rhs,
                jobs,
                runs,
                self.lhs_op,
                self.rhs_op,
                alpha,
                beta,
            )
        }
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
        self.require_fully_direct_storage()?;
        self.require_identity_storage_ops()?;
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
            &self.direct_batch,
            self.direct_batch_runs(),
            alpha,
            beta,
        )?;
        Ok(())
    }

    #[cfg(test)]
    fn direct_batch(&self) -> &[Rank2GemmBatchJob] {
        &self.direct_batch
    }

    fn require_fully_direct_storage(&self) -> Result<(), OperationError> {
        if self.is_fully_direct() {
            Ok(())
        } else {
            Err(OperationError::UnsupportedTensorContractScope {
                message: NON_COUPLED_OPERAND_MESSAGE,
            })
        }
    }

    fn require_identity_storage_ops(&self) -> Result<(), OperationError> {
        if self.lhs_op == MatrixOp::Identity && self.rhs_op == MatrixOp::Identity {
            return Ok(());
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "storage-handle core replay does not expose transpose/adjoint matrix views",
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
        self.require_fully_direct_storage()?;
        self.require_identity_storage_ops()?;
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;

        let _ = fusion_workspace;
        gemm.matmul_rank2_batch(
            dst_data,
            lhs_data,
            rhs_data,
            &self.direct_batch,
            self.direct_batch_runs(),
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
        self.require_fully_direct_storage()?;
        self.require_identity_storage_ops()?;
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
            &self.direct_batch,
            self.direct_batch_runs(),
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
        self.require_fully_direct_storage()?;
        self.require_identity_storage_ops()?;
        for group in &self.groups {
            let (Some(lhs_base), Some(rhs_base), Some(dst_base)) = (
                group.lhs.direct_offset,
                group.rhs.direct_offset,
                group.dst.direct_offset,
            ) else {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: NON_COUPLED_OPERAND_MESSAGE,
                });
            };
            validate_storage_range(lhs.len(), lhs_base, group.lhs.rows, group.lhs.cols)?;
            validate_storage_range(rhs.len(), rhs_base, group.rhs.rows, group.rhs.cols)?;
            validate_storage_range(dst.len(), dst_base, group.dst.rows, group.dst.cols)?;
        }
        for group in &self.groups {
            let (Some(lhs_base), Some(rhs_base), Some(dst_base)) = (
                group.lhs.direct_offset,
                group.rhs.direct_offset,
                group.dst.direct_offset,
            ) else {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: NON_COUPLED_OPERAND_MESSAGE,
                });
            };
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

#[derive(Clone, Debug, PartialEq)]
pub struct FusionStridedBlockLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
}

#[derive(Clone, Debug, PartialEq)]
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

fn group_scratch_layout(
    class: FusionGroupExecutionClass,
    group: &FusionBlockContractGroupPlan,
) -> Result<FusionGroupScratchLayout, OperationError> {
    let lhs_len = if class.packs_lhs() {
        direct_matrix_len(group.lhs.rows, group.lhs.cols)?
    } else {
        0
    };
    let rhs_len = if class.packs_rhs() {
        direct_matrix_len(group.rhs.rows, group.rhs.cols)?
    } else {
        0
    };
    let dst_len = if class.scatters_dst() {
        direct_matrix_len(group.dst.rows, group.dst.cols)?
    } else {
        0
    };
    let rhs_offset = lhs_len;
    let dst_offset = rhs_offset
        .checked_add(rhs_len)
        .ok_or(OperationError::ElementCountOverflow)?;
    let total_len = dst_offset
        .checked_add(dst_len)
        .ok_or(OperationError::ElementCountOverflow)?;
    Ok(FusionGroupScratchLayout {
        lhs_len,
        rhs_offset,
        rhs_len,
        dst_offset,
        dst_len,
        total_len,
    })
}

fn validate_group_replay_bounds(
    group: &FusionBlockMatrixGroup,
    storage_len: usize,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Result<(), OperationError> {
    let matrix_len = direct_matrix_len(matrix_rows, matrix_cols)?;
    if let Some(base) = group.direct_offset {
        validate_storage_range(storage_len, base, matrix_rows, matrix_cols)?;
    }
    if group.block_indices.len() != group.subblocks.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "fusion block group",
        });
    }
    for layout in &group.subblocks {
        validate_raw_strided_bounds(
            storage_len,
            &layout.block.shape,
            &layout.block.strides,
            layout.block.offset,
        )?;
        validate_raw_strided_bounds(
            matrix_len,
            &layout.block.shape,
            &layout.matrix_strides,
            layout.matrix_offset,
        )?;
    }
    Ok(())
}

fn validate_compiled_plan_layouts(
    dst_structure: &BlockStructure,
    lhs_structure: &BlockStructure,
    rhs_structure: &BlockStructure,
    inactive_dst_scale_blocks: &[FusionScaleBlockLayout],
    groups: &[FusionBlockContractGroupPlan],
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
) -> Result<(), OperationError> {
    validate_destination_layouts_injective(
        dst_structure,
        "fusion block contraction destination layouts overlap",
    )?;
    let dst_len = dst_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let lhs_len = lhs_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let rhs_len = rhs_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut active_dst_blocks = vec![false; dst_structure.block_count()];
    for group in groups {
        validate_group_shape(group)?;
        validate_group_against_structure("lhs", &group.lhs, lhs_structure, lhs_len, lhs_op)?;
        validate_group_against_structure("rhs", &group.rhs, rhs_structure, rhs_len, rhs_op)?;
        validate_group_against_structure(
            "dst",
            &group.dst,
            dst_structure,
            dst_len,
            MatrixOp::Identity,
        )?;
        for &block_index in &group.dst.block_indices {
            let owned = active_dst_blocks.get_mut(block_index).ok_or(
                OperationError::BlockIndexOutOfBounds {
                    tensor: "dst",
                    index: block_index,
                    count: dst_structure.block_count(),
                },
            )?;
            if *owned {
                return Err(OperationError::DuplicateTransformDestination {
                    dst_block: block_index,
                });
            }
            *owned = true;
        }
    }
    let mut inactive = inactive_dst_scale_blocks.iter();
    for (block_index, active) in active_dst_blocks.into_iter().enumerate() {
        if active {
            continue;
        }
        let layout = inactive.next().ok_or(OperationError::StructureMismatch {
            tensor: "inactive dst blocks",
        })?;
        let block = dst_structure.block(block_index)?;
        if block.shape() != layout.block.shape
            || strides_to_isize(block.strides())? != layout.block.strides
            || offset_to_isize(block.offset())? != layout.block.offset
        {
            return Err(OperationError::StructureMismatch {
                tensor: "inactive dst blocks",
            });
        }
    }
    if inactive.next().is_some() {
        return Err(OperationError::StructureMismatch {
            tensor: "inactive dst blocks",
        });
    }
    Ok(())
}

fn validate_group_shape(group: &FusionBlockContractGroupPlan) -> Result<(), OperationError> {
    if group.lhs.cols != group.rhs.rows {
        return Err(OperationError::ShapeMismatch {
            dst: vec![group.lhs.cols],
            src: vec![group.rhs.rows],
        });
    }
    if group.dst.rows != group.lhs.rows || group.dst.cols != group.rhs.cols {
        return Err(OperationError::ShapeMismatch {
            dst: vec![group.dst.rows, group.dst.cols],
            src: vec![group.lhs.rows, group.rhs.cols],
        });
    }
    Ok(())
}

fn validate_group_against_structure(
    tensor: &'static str,
    group: &FusionBlockMatrixGroup,
    structure: &BlockStructure,
    storage_len: usize,
    op: MatrixOp,
) -> Result<(), OperationError> {
    let (matrix_rows, matrix_cols) = match op {
        MatrixOp::Identity => (group.rows, group.cols),
        MatrixOp::Transpose | MatrixOp::Adjoint => (group.cols, group.rows),
    };
    validate_group_replay_bounds(group, storage_len, matrix_rows, matrix_cols)?;
    if group.subblocks.is_empty() {
        return Err(OperationError::StructureMismatch { tensor });
    }
    for (&block_index, layout) in group.block_indices.iter().zip(&group.subblocks) {
        let block =
            structure
                .block(block_index)
                .map_err(|_| OperationError::BlockIndexOutOfBounds {
                    tensor,
                    index: block_index,
                    count: structure.block_count(),
                })?;
        if block.shape() != layout.block.shape
            || strides_to_isize(block.strides())? != layout.block.strides
            || offset_to_isize(block.offset())? != layout.block.offset
        {
            return Err(OperationError::StructureMismatch { tensor });
        }
    }
    let covers_matrix = matrix_layouts_cover_exactly(group, matrix_rows, matrix_cols)
        .map_err(|_| OperationError::StructureMismatch { tensor })?;
    if group.needs_clear == covers_matrix {
        return Err(OperationError::StructureMismatch { tensor });
    }
    if let Some(offset) = group.direct_offset {
        if direct_group_matrix_offset(&group.subblocks, covers_matrix) != Some(offset) {
            return Err(OperationError::StructureMismatch { tensor });
        }
    }
    Ok(())
}

fn matrix_layouts_cover_exactly(
    group: &FusionBlockMatrixGroup,
    matrix_rows: usize,
    matrix_cols: usize,
) -> Result<bool, OperationError> {
    let matrix_len = direct_matrix_len(matrix_rows, matrix_cols)?;
    if matrix_len == 0 {
        return Ok(group
            .subblocks
            .iter()
            .all(|layout| layout.block.shape.contains(&0)));
    }
    let mut rectangles = Vec::with_capacity(group.subblocks.len());
    for layout in &group.subblocks {
        let rectangle = canonical_matrix_rectangle(matrix_rows, matrix_cols, layout)?;
        if rectangle[0].0 != rectangle[0].1 && rectangle[1].0 != rectangle[1].1 {
            rectangles.push(rectangle);
        }
    }

    for axis in 0..2 {
        let mut intervals: Vec<_> = rectangles.iter().map(|rectangle| rectangle[axis]).collect();
        intervals.sort_unstable();
        intervals.dedup();
        if intervals.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion block matrix",
            });
        }
    }

    rectangles.sort_unstable();
    if rectangles.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(OperationError::StructureMismatch {
            tensor: "fusion block matrix",
        });
    }
    let occupied = rectangles.iter().try_fold(0usize, |total, rectangle| {
        let rows = rectangle[0].1 - rectangle[0].0;
        let cols = rectangle[1].1 - rectangle[1].0;
        total
            .checked_add(
                rows.checked_mul(cols)
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)
    })?;
    Ok(occupied == matrix_len)
}

fn canonical_matrix_rectangle(
    matrix_rows: usize,
    matrix_cols: usize,
    layout: &FusionSubblockMatrixLayout,
) -> Result<[(usize, usize); 2], OperationError> {
    let offset = usize::try_from(layout.matrix_offset)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    for split in 0..=layout.block.shape.len() {
        let mut row_dim = 1usize;
        let mut expected_stride = 1usize;
        let mut canonical = true;
        for (&dim, &stride) in layout.block.shape[..split]
            .iter()
            .zip(&layout.matrix_strides[..split])
        {
            if dim > 1 && usize::try_from(stride).ok() != Some(expected_stride) {
                canonical = false;
                break;
            }
            row_dim = row_dim
                .checked_mul(dim)
                .ok_or(OperationError::ElementCountOverflow)?;
            expected_stride = row_dim;
        }
        if !canonical || row_dim > matrix_rows {
            continue;
        }

        let mut col_dim = 1usize;
        expected_stride = matrix_rows;
        for (&dim, &stride) in layout.block.shape[split..]
            .iter()
            .zip(&layout.matrix_strides[split..])
        {
            if dim > 1 && usize::try_from(stride).ok() != Some(expected_stride) {
                canonical = false;
                break;
            }
            col_dim = col_dim
                .checked_mul(dim)
                .ok_or(OperationError::ElementCountOverflow)?;
            expected_stride = expected_stride
                .checked_mul(dim)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        if !canonical || col_dim > matrix_cols {
            continue;
        }

        let row = offset % matrix_rows;
        let col = offset / matrix_rows;
        let row_end = row
            .checked_add(row_dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        let col_end = col
            .checked_add(col_dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        if row_end <= matrix_rows && col_end <= matrix_cols {
            return Ok([(row, row_end), (col, col_end)]);
        }
    }
    Err(OperationError::StructureMismatch {
        tensor: "fusion block matrix",
    })
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

fn scatter_group<A, T>(
    kernels: &mut A,
    group: &FusionBlockMatrixGroup,
    data: &mut [T],
    packed: &[T],
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + RecouplingCoefficientAction<f64>,
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
            T::coefficient_as_data(layout.coefficient),
            beta,
        )?;
    }
    Ok(())
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

fn compile_group_execution(
    groups: &[FusionBlockContractGroupPlan],
) -> Result<CompiledGroupExecution, OperationError> {
    let mut direct_batch = Vec::new();
    let mut direct_destinations = Vec::new();
    let mut irregular = Vec::with_capacity(groups.len());
    let mut max_scratch_len = 0;
    for (group_index, group) in groups.iter().enumerate() {
        let class = FusionGroupExecutionClass::compile(group);
        if let Some(dst_offset) = group.dst.direct_offset {
            direct_destinations.push(Rank2GemmBatchJob {
                dst_offset,
                lhs_offset: 0,
                rhs_offset: 0,
                rows: group.dst.rows,
                contracted: group.lhs.cols,
                cols: group.dst.cols,
            });
        }
        if class.is_direct() {
            let (Some(lhs_offset), Some(rhs_offset), Some(dst_offset)) = (
                group.lhs.direct_offset,
                group.rhs.direct_offset,
                group.dst.direct_offset,
            ) else {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "direct fusion contraction class requires direct matrix offsets",
                });
            };
            direct_batch.push(Rank2GemmBatchJob {
                dst_offset,
                lhs_offset,
                rhs_offset,
                rows: group.lhs.rows,
                contracted: group.lhs.cols,
                cols: group.rhs.cols,
            });
            continue;
        }
        let scratch = group_scratch_layout(class, group)?;
        max_scratch_len = max_scratch_len.max(scratch.total_len);
        irregular.push(FusionIrregularGroupExecution {
            group_index,
            class,
            job: Rank2GemmBatchJob {
                dst_offset: group.dst.direct_offset.unwrap_or(0),
                lhs_offset: group.lhs.direct_offset.unwrap_or(0),
                rhs_offset: group.rhs.direct_offset.unwrap_or(0),
                rows: group.lhs.rows,
                contracted: group.lhs.cols,
                cols: group.rhs.cols,
            },
            scratch,
        });
    }
    validate_disjoint_direct_destinations(&direct_destinations)?;
    direct_batch.sort_by_key(|job| {
        (
            job.rows,
            job.contracted,
            job.cols,
            job.dst_offset,
            job.lhs_offset,
            job.rhs_offset,
        )
    });

    Ok((direct_batch, irregular, max_scratch_len))
}

fn validate_disjoint_direct_destinations(jobs: &[Rank2GemmBatchJob]) -> Result<(), OperationError> {
    let mut dst_ranges: Vec<(usize, usize)> = jobs
        .iter()
        .filter_map(|job| match direct_matrix_len(job.rows, job.cols) {
            Ok(0) => None,
            Ok(len) => Some(Ok((job.dst_offset, len))),
            Err(error) => Some(Err(error)),
        })
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
    Ok(())
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
    use std::ops::{Add, Mul};

    use num_complex::Complex64;
    use num_traits::Zero;

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

    fn matrix_group(rows: usize, cols: usize, direct: bool) -> FusionBlockMatrixGroup {
        FusionBlockMatrixGroup {
            coupled: SectorId::new(0),
            rows,
            cols,
            needs_clear: false,
            direct_offset: direct.then_some(0),
            block_indices: vec![0],
            subblocks: vec![FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: vec![rows, cols],
                    strides: vec![1, rows as isize],
                    offset: 0,
                },
                matrix_offset: 0,
                matrix_strides: vec![1, rows as isize],
                coefficient: 1.0,
            }],
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

    fn scalar_group(block_index: usize, direct: bool, coefficient: f64) -> FusionBlockMatrixGroup {
        FusionBlockMatrixGroup {
            coupled: SectorId::new(block_index),
            rows: 1,
            cols: 1,
            needs_clear: false,
            direct_offset: direct.then_some(block_index),
            block_indices: vec![block_index],
            subblocks: vec![FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: vec![1],
                    strides: vec![1],
                    offset: block_index as isize,
                },
                matrix_offset: 0,
                matrix_strides: vec![1],
                coefficient,
            }],
        }
    }

    #[derive(Default)]
    struct NaiveGemm;

    impl<T> Rank2Gemm<T> for NaiveGemm
    where
        T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
    {
        fn matmul_rank2(
            &mut self,
            dst: &mut [T],
            lhs: &[T],
            rhs: &[T],
            rows: usize,
            contracted: usize,
            cols: usize,
            alpha: T,
            beta: T,
        ) -> Result<(), OperationError> {
            for col in 0..cols {
                for row in 0..rows {
                    let mut value = T::zero();
                    for inner in 0..contracted {
                        value = value + lhs[row + inner * rows] * rhs[inner + col * contracted];
                    }
                    let index = row + col * rows;
                    dst[index] = alpha * value + beta * dst[index];
                }
            }
            Ok(())
        }
    }

    fn all_classes_apply_alpha_beta_and_coefficients<T>()
    where
        T: DenseBlockScalar
            + RecouplingCoefficientAction<f64>
            + From<f64>
            + Add<Output = T>
            + Mul<Output = T>
            + std::fmt::Debug
            + PartialEq,
    {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, (0..8).map(|_| vec![1])).unwrap());
        let mut groups = Vec::new();
        for class in 0usize..8 {
            let pack_lhs = class & usize::from(FusionGroupExecutionClass::PACK_LHS) != 0;
            let pack_rhs = class & usize::from(FusionGroupExecutionClass::PACK_RHS) != 0;
            let scatter_dst = class & usize::from(FusionGroupExecutionClass::SCATTER_DST) != 0;
            groups.push(
                FusionBlockContractGroupPlan::new(
                    scalar_group(class, !pack_lhs, if pack_lhs { 2.0 } else { 1.0 }),
                    scalar_group(class, !pack_rhs, if pack_rhs { 3.0 } else { 1.0 }),
                    scalar_group(class, !scatter_dst, if scatter_dst { 5.0 } else { 1.0 }),
                )
                .unwrap(),
            );
        }
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            groups,
        )
        .unwrap();
        assert!(!plan.is_fully_direct());
        assert_eq!(plan.direct_batch.len(), 1);
        assert_eq!(plan.irregular.len(), 7);
        assert_eq!(plan.max_irregular_scratch_len, 3);

        let lhs = (0..8)
            .map(|index| T::from(index as f64 + 1.0))
            .collect::<Vec<_>>();
        let rhs = (0..8)
            .map(|index| T::from(index as f64 + 2.0))
            .collect::<Vec<_>>();
        let initial = (0..8)
            .map(|index| T::from(index as f64 + 7.0))
            .collect::<Vec<_>>();
        let alpha = T::from(1.5);
        let beta = T::from(-0.25);
        let mut expected = Vec::new();
        for class in 0usize..8 {
            let lhs_coefficient = if class & usize::from(FusionGroupExecutionClass::PACK_LHS) != 0 {
                T::from(2.0)
            } else {
                T::from(1.0)
            };
            let rhs_coefficient = if class & usize::from(FusionGroupExecutionClass::PACK_RHS) != 0 {
                T::from(3.0)
            } else {
                T::from(1.0)
            };
            let dst_coefficient =
                if class & usize::from(FusionGroupExecutionClass::SCATTER_DST) != 0 {
                    T::from(5.0)
                } else {
                    T::from(1.0)
                };
            expected.push(
                dst_coefficient
                    * alpha
                    * lhs_coefficient
                    * lhs[class]
                    * rhs_coefficient
                    * rhs[class]
                    + beta * initial[class],
            );
        }

        let mut kernels = crate::StridedHostKernelAdapter::default();
        let mut gemm = NaiveGemm;
        let mut workspace = FusionBlockContractWorkspace::<T>::default();
        let mut dst = initial.clone();
        plan.execute_raw(
            &mut kernels,
            &mut gemm,
            &mut workspace,
            &structure,
            &mut dst,
            &structure,
            &lhs,
            &structure,
            &rhs,
            alpha,
            beta,
        )
        .unwrap();
        assert_eq!(dst, expected);
        assert_eq!(workspace.scratch.len(), 3);

        workspace.scratch.fill(T::from(99.0));
        dst.clone_from(&initial);
        let mut profile = TensorContractFusionProfile::default();
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
            alpha,
            beta,
            &mut profile,
        )
        .unwrap();
        assert_eq!(dst, expected);
        assert_eq!(profile.core_contract_groups, 8);
        assert_eq!(profile.core_direct_gemm_groups, 4);
    }

    #[test]
    fn all_eight_group_classes_replay_f64() {
        all_classes_apply_alpha_beta_and_coefficients::<f64>();
    }

    #[test]
    fn all_eight_group_classes_replay_c64() {
        all_classes_apply_alpha_beta_and_coefficients::<Complex64>();
    }

    #[test]
    fn sparse_pack_clears_stale_group_scratch() {
        let lhs_structure = Arc::new(BlockStructure::trivial(&[1, 1]).unwrap());
        let rhs_structure = Arc::new(BlockStructure::trivial(&[2, 1]).unwrap());
        let dst_structure = Arc::new(BlockStructure::trivial(&[2, 1]).unwrap());
        let lhs = FusionBlockMatrixGroup {
            coupled: SectorId::new(0),
            rows: 2,
            cols: 2,
            needs_clear: true,
            direct_offset: None,
            block_indices: vec![0],
            subblocks: vec![FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: vec![1, 1],
                    strides: vec![1, 1],
                    offset: 0,
                },
                matrix_offset: 0,
                matrix_strides: vec![1, 2],
                coefficient: 1.0,
            }],
        };
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&dst_structure),
            Arc::clone(&lhs_structure),
            Arc::clone(&rhs_structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                lhs,
                matrix_group(2, 1, true),
                matrix_group(2, 1, true),
            )
            .unwrap()],
        )
        .unwrap();
        let mut workspace = FusionBlockContractWorkspace::<f64>::default();
        workspace.scratch.resize_filled(4, 41.0);
        let mut dst = vec![7.0, 11.0];
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
            &mut NaiveGemm,
            &mut workspace,
            &dst_structure,
            &mut dst,
            &lhs_structure,
            &[3.0],
            &rhs_structure,
            &[5.0, 13.0],
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(dst, [15.0, 0.0]);
        assert_eq!(workspace.scratch.len(), 4);
    }

    struct ComplexOpGemm;

    impl Rank2Gemm<Complex64> for ComplexOpGemm {
        fn matmul_rank2(
            &mut self,
            dst: &mut [Complex64],
            lhs: &[Complex64],
            rhs: &[Complex64],
            _rows: usize,
            _contracted: usize,
            _cols: usize,
            alpha: Complex64,
            beta: Complex64,
        ) -> Result<(), OperationError> {
            dst[0] = alpha * lhs[0] * rhs[0] + beta * dst[0];
            Ok(())
        }

        fn matmul_rank2_batch_with_ops(
            &mut self,
            dst: &mut [Complex64],
            lhs: &[Complex64],
            rhs: &[Complex64],
            jobs: &[Rank2GemmBatchJob],
            _runs: &[usize],
            lhs_op: MatrixOp,
            rhs_op: MatrixOp,
            alpha: Complex64,
            beta: Complex64,
        ) -> Result<(), OperationError> {
            for job in jobs {
                let lhs_value = match lhs_op {
                    MatrixOp::Adjoint => lhs[job.lhs_offset].conj(),
                    MatrixOp::Identity | MatrixOp::Transpose => lhs[job.lhs_offset],
                };
                let rhs_value = match rhs_op {
                    MatrixOp::Adjoint => rhs[job.rhs_offset].conj(),
                    MatrixOp::Identity | MatrixOp::Transpose => rhs[job.rhs_offset],
                };
                let dst_value = &mut dst[job.dst_offset];
                *dst_value = alpha * lhs_value * rhs_value + beta * *dst_value;
            }
            Ok(())
        }
    }

    #[test]
    fn packed_adjoint_is_conjugated_only_by_gemm() {
        let structure = Arc::new(BlockStructure::trivial(&[1, 1]).unwrap());
        let plan = FusionBlockContractPlan::from_parts_with_ops(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                matrix_group(1, 1, false),
                matrix_group(1, 1, true),
                matrix_group(1, 1, true),
            )
            .unwrap()],
            MatrixOp::Adjoint,
            MatrixOp::Identity,
        )
        .unwrap();
        let lhs = [Complex64::new(2.0, 3.0)];
        let rhs = [Complex64::new(5.0, -7.0)];
        let mut dst = [Complex64::new(11.0, 13.0)];
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
            &mut ComplexOpGemm,
            &mut FusionBlockContractWorkspace::default(),
            &structure,
            &mut dst,
            &structure,
            &lhs,
            &structure,
            &rhs,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
        assert_eq!(dst[0], lhs[0].conj() * rhs[0]);
    }

    #[test]
    fn asymmetric_packed_adjoint_validates_physical_matrix_shape() {
        let dst_structure = Arc::new(BlockStructure::trivial(&[2, 1]).unwrap());
        let lhs_structure = Arc::new(BlockStructure::trivial(&[3, 2]).unwrap());
        let rhs_structure = Arc::new(BlockStructure::trivial(&[3, 1]).unwrap());
        let mut physical_lhs = matrix_group(3, 2, false);
        physical_lhs.rows = 2;
        physical_lhs.cols = 3;
        let plan = FusionBlockContractPlan::from_parts_with_ops(
            dst_structure,
            lhs_structure,
            rhs_structure,
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                physical_lhs,
                matrix_group(3, 1, false),
                matrix_group(2, 1, false),
            )
            .unwrap()],
            MatrixOp::Adjoint,
            MatrixOp::Identity,
        )
        .unwrap();
        assert!(!plan.is_fully_direct());
    }

    #[test]
    fn mixed_active_and_inactive_blocks_apply_beta_once() {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let inactive = vec![FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: vec![1],
                strides: vec![1],
                offset: 1,
            },
        }];
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            inactive,
            vec![FusionBlockContractGroupPlan::new(
                scalar_group(0, false, 2.0),
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
            )
            .unwrap()],
        )
        .unwrap();
        let mut dst = vec![7.0, 11.0];
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
            &mut NaiveGemm,
            &mut FusionBlockContractWorkspace::default(),
            &structure,
            &mut dst,
            &structure,
            &[3.0, 0.0],
            &structure,
            &[5.0, 0.0],
            2.0,
            -3.0,
        )
        .unwrap();
        assert_eq!(dst, [39.0, -33.0]);
    }

    #[test]
    fn malformed_subblock_bounds_are_rejected_during_compile() {
        let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
        let mut lhs = scalar_group(0, false, 1.0);
        lhs.subblocks[0].matrix_offset = 1;
        let error = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                lhs,
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
            )
            .unwrap()],
        )
        .unwrap_err();
        assert!(matches!(error, OperationError::OffsetOverflow { .. }));
    }

    #[test]
    fn direct_and_clear_flags_must_follow_subblock_layouts() {
        let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
        let mut invalid_direct = scalar_group(0, true, 2.0);
        invalid_direct.direct_offset = Some(0);
        let direct_error = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                invalid_direct,
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
            )
            .unwrap()],
        )
        .unwrap_err();
        assert!(matches!(
            direct_error,
            OperationError::StructureMismatch { tensor: "lhs" }
        ));

        let mut invalid_clear = scalar_group(0, false, 1.0);
        invalid_clear.needs_clear = true;
        let clear_error = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                invalid_clear,
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
            )
            .unwrap()],
        )
        .unwrap_err();
        assert!(matches!(
            clear_error,
            OperationError::StructureMismatch { tensor: "lhs" }
        ));
    }

    #[test]
    fn packed_matrix_layouts_must_be_injective() {
        let lhs_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2]]).unwrap());
        let scalar_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
        let mut lhs = matrix_group(2, 1, false);
        lhs.subblocks[0].block.shape = vec![2];
        lhs.subblocks[0].block.strides = vec![1];
        lhs.subblocks[0].matrix_strides[0] = 0;
        lhs.subblocks[0].matrix_strides.truncate(1);
        let mut dst = matrix_group(2, 1, false);
        dst.subblocks[0].block.shape = vec![2];
        dst.subblocks[0].block.strides = vec![1];
        dst.subblocks[0].matrix_strides = vec![1];
        let error = FusionBlockContractPlan::from_parts(
            Arc::clone(&lhs_structure),
            Arc::clone(&lhs_structure),
            Arc::clone(&scalar_structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(lhs, scalar_group(0, false, 1.0), dst).unwrap()],
        )
        .unwrap_err();
        assert!(
            matches!(error, OperationError::StructureMismatch { tensor: "lhs" }),
            "{error:?}"
        );
    }

    #[test]
    fn packed_matrix_geometry_distinguishes_overlap_gap_and_empty_blocks() {
        let subblock = |matrix_offset, extent| FusionSubblockMatrixLayout {
            block: FusionStridedBlockLayout {
                shape: vec![extent],
                strides: vec![1],
                offset: 0,
            },
            matrix_offset,
            matrix_strides: vec![1],
            coefficient: 1.0,
        };
        let group = |rows, subblocks: Vec<FusionSubblockMatrixLayout>| FusionBlockMatrixGroup {
            coupled: SectorId::new(0),
            rows,
            cols: 1,
            needs_clear: true,
            direct_offset: None,
            block_indices: (0..subblocks.len()).collect(),
            subblocks,
        };

        // What: partial rectangle overlap is invalid even when element counts
        // equal the matrix size.
        assert!(matrix_layouts_cover_exactly(
            &group(4, vec![subblock(0, 2), subblock(1, 2)]),
            4,
            1
        )
        .is_err());
        // What: a disjoint incomplete grid is valid but requires scratch clear.
        assert_eq!(
            matrix_layouts_cover_exactly(&group(5, vec![subblock(0, 2), subblock(3, 2)]), 5, 1),
            Ok(false)
        );
        // What: distinct empty tree blocks do not alias any matrix element.
        assert_eq!(
            matrix_layouts_cover_exactly(
                &group(2, vec![subblock(0, 0), subblock(0, 0), subblock(0, 2)]),
                2,
                1
            ),
            Ok(true)
        );
    }

    #[test]
    fn destination_storage_alias_is_rejected_during_compile() {
        use tenet_core::{BlockKey, BlockSpec};

        let block =
            |sector| BlockSpec::with_key(BlockKey::ordinal(sector), vec![1], vec![1], 0).unwrap();
        let aliased =
            Arc::new(BlockStructure::from_blocks_with_rank(1, vec![block(0), block(1)]).unwrap());
        let error = FusionBlockContractPlan::from_parts(
            Arc::clone(&aliased),
            Arc::clone(&aliased),
            Arc::clone(&aliased),
            Vec::new(),
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::InvalidArgument {
                message: "fusion block contraction destination layouts overlap"
            }
        ));
    }

    #[test]
    fn inactive_destination_layouts_are_the_exact_active_complement() {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let active = FusionBlockContractGroupPlan::new(
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
        )
        .unwrap();
        let missing = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![active.clone()],
        )
        .unwrap_err();
        assert!(matches!(
            missing,
            OperationError::StructureMismatch {
                tensor: "inactive dst blocks"
            }
        ));

        let active_layout = FusionScaleBlockLayout {
            block: scalar_group(0, true, 1.0).subblocks[0].block.clone(),
        };
        let overlap = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            vec![active_layout],
            vec![active],
        )
        .unwrap_err();
        assert!(matches!(
            overlap,
            OperationError::StructureMismatch {
                tensor: "inactive dst blocks"
            }
        ));
    }

    #[test]
    fn duplicate_destination_group_is_rejected() {
        let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
        let group = FusionBlockContractGroupPlan::new(
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
        )
        .unwrap();
        let error = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![group.clone(), group],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::DuplicateTransformDestination { dst_block: 0 }
        ));
    }

    #[test]
    fn mixed_plan_rejects_storage_direct_before_mutation() {
        let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                scalar_group(0, false, 1.0),
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
            )
            .unwrap()],
        )
        .unwrap();
        let lhs = vec![2.0];
        let rhs = vec![3.0];
        let mut dst = vec![5.0];
        let error = plan
            .execute_direct_on_storage_prezeroed(&mut RejectingStorageGemm, &mut dst, &lhs, &rhs)
            .unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: NON_COUPLED_OPERAND_MESSAGE
            }
        ));
        assert_eq!(dst, [5.0]);
    }

    #[test]
    fn direct_batch_lowers_disjoint_groups() {
        let groups = vec![
            group_plan((2, 3, 4), (0, 0, 0)),
            group_plan((3, 2, 5), (6, 12, 8)),
        ];
        let (batch, irregular, _) = compile_group_execution(&groups).unwrap();
        assert!(irregular.is_empty());
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
        let (batch, irregular, _) = compile_group_execution(&groups).unwrap();
        assert!(irregular.is_empty());
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
        let groups = (0usize..5)
            .map(|block| {
                FusionBlockContractGroupPlan::new(
                    scalar_group(block, true, 1.0),
                    scalar_group(block, true, 1.0),
                    scalar_group(block, true, 1.0),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, (0..5).map(|_| vec![1])).unwrap());
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
            strided_batch_runs(plan.direct_batch())
        );
        assert_eq!(
            plan.direct_batch_runs().iter().sum::<usize>(),
            plan.direct_batch().len()
        );
    }

    #[test]
    fn non_direct_plan_has_empty_run_partition() {
        let plan = FusionBlockContractGroupPlan::new(
            matrix_group(2, 3, false),
            matrix_group(3, 4, true),
            matrix_group(2, 4, true),
        )
        .unwrap();
        let dst_structure = Arc::new(BlockStructure::trivial(&[2, 4]).unwrap());
        let lhs_structure = Arc::new(BlockStructure::trivial(&[2, 3]).unwrap());
        let rhs_structure = Arc::new(BlockStructure::trivial(&[3, 4]).unwrap());
        let plan = FusionBlockContractPlan::from_parts(
            dst_structure,
            lhs_structure,
            rhs_structure,
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
        let error = compile_group_execution(&groups).unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope { .. }
        ));
    }

    #[test]
    fn direct_batch_ignores_empty_destination_ranges() {
        let groups = vec![
            group_plan((2, 1, 1), (0, 0, 0)),
            group_plan((0, 1, 1), (0, 0, 1)),
        ];
        let (direct, irregular, _) = compile_group_execution(&groups).unwrap();
        assert_eq!(direct.len(), 2);
        assert!(irregular.is_empty());
    }

    #[test]
    fn non_direct_group_compiles_one_irregular_execution() {
        let plan = FusionBlockContractGroupPlan::new(
            direct_group(2, 3, None),
            direct_group(3, 4, Some(0)),
            direct_group(2, 4, Some(0)),
        )
        .unwrap();
        let (direct, irregular, _) = compile_group_execution(&[plan]).unwrap();
        assert!(direct.is_empty());
        assert_eq!(irregular.len(), 1);
    }

    struct RejectingStorageGemm;

    impl StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>> for RejectingStorageGemm {
        fn matmul_range_into(
            &mut self,
            _dst: &mut Vec<f64>,
            _dst_offset: usize,
            _lhs: &Vec<f64>,
            _lhs_offset: usize,
            _rhs: &Vec<f64>,
            _rhs_offset: usize,
            _rows: usize,
            _contracted: usize,
            _cols: usize,
        ) -> Result<(), OperationError> {
            panic!("op-bearing storage replay must reject before GEMM")
        }
    }

    #[test]
    fn op_bearing_plan_rejects_prezeroed_storage_replay() {
        let structure = Arc::new(BlockStructure::trivial(&[2, 2]).unwrap());
        let plan = FusionBlockContractPlan::from_parts_with_ops(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            vec![FusionBlockContractGroupPlan::new(
                matrix_group(2, 2, true),
                matrix_group(2, 2, true),
                matrix_group(2, 2, true),
            )
            .unwrap()],
            MatrixOp::Adjoint,
            MatrixOp::Identity,
        )
        .unwrap();
        let lhs = vec![0.0; 4];
        let rhs = vec![0.0; 4];
        let mut dst = vec![0.0; 4];

        let error = plan
            .execute_direct_on_storage_prezeroed(&mut RejectingStorageGemm, &mut dst, &lhs, &rhs)
            .unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message:
                    "storage-handle core replay does not expose transpose/adjoint matrix views"
            }
        ));
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
            runs: &[usize],
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            debug_assert_eq!(runs.iter().sum::<usize>(), jobs.len());
            self.batch_calls += 1;
            self.last_batch_len = jobs.len();
            Ok(())
        }
    }

    #[test]
    fn profiled_direct_replay_uses_one_batched_gemm_call() {
        let groups = (0usize..2)
            .map(|block| {
                FusionBlockContractGroupPlan::new(
                    scalar_group(block, true, 1.0),
                    scalar_group(block, true, 1.0),
                    scalar_group(block, true, 1.0),
                )
                .unwrap()
            })
            .collect();
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let plan = FusionBlockContractPlan::from_parts(
            Arc::clone(&structure),
            Arc::clone(&structure),
            Arc::clone(&structure),
            Vec::new(),
            groups,
        )
        .unwrap();
        let mut kernels = crate::StridedHostKernelAdapter::default();
        let mut gemm = CountingBatchGemm::default();
        let mut workspace = FusionBlockContractWorkspace::<f64>::default();
        let mut profile = TensorContractFusionProfile::default();
        let lhs = vec![0.0; 2];
        let rhs = vec![0.0; 2];
        let mut dst = vec![0.0; 2];

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
        assert_eq!(profile.core_workspace_prepare, std::time::Duration::ZERO);
        assert_eq!(workspace.scratch.len(), 0);
    }
}
