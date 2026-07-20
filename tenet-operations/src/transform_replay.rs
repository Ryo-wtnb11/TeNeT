use core::mem::MaybeUninit;
use core::ops::{Add, Mul};
use std::sync::{Arc, Weak};

use num_traits::{One, Zero};
use tenet_core::{
    BlockStructure, BlockView, BlockViewMut, HostReadableStorage, HostWritableStorage, Placement,
    ScratchStorage, SimilarStorage, TensorMap,
};
use tenet_dense::{
    strided_batch_runs_into, DefaultDenseExecutor, DenseExecutor, DenseGemmBatchJob,
};

use crate::host_scratch::HostScratchBuffer;
use crate::owned_overwrite_buffer::initialize_owned;
use crate::storage_scratch::{StorageTreeTransformWorkspace, TreeTransformScratchBuffers};
use crate::strided::offset_to_isize;
use crate::tensoradd::{TensorAddDescriptor, TensorAddDescriptorTerm};
use crate::transform_structure::{
    TreeTransformPackReplay, TreeTransformScatterGroupReplay, TreeTransformScatterReplay,
    TreeTransformSingleReplay,
};
use crate::{
    tensoradd_raw_strided_kernel, tensoradd_raw_strided_kernel_trusted, BakedFusedLayout,
    ConjugateValue, DenseRecouplingScalar, HostAllocator, HostKernelAdapter, OperationError,
    RecouplingCoefficientAction, ReportsPlacement, TensorAddStructure, TreeTransformBlock,
    TreeTransformLayout, TreeTransformLayoutTable, TreeTransformReplayProfile,
    TreeTransformStructure,
};

#[derive(Clone, Copy)]
enum DestinationMode<D> {
    Axpby(D),
    // Why not use Axpby(D::zero()): IEEE arithmetic still reads NaN destination
    // values, whereas assignment APIs promise destination-independent output.
    Overwrite,
}

struct PhysicalOverwriteProof<'a, C> {
    structure: &'a TreeTransformStructure<C>,
    dst_structure: &'a Arc<BlockStructure>,
    required_len: usize,
    nout: usize,
}

impl<'a, C: Copy> PhysicalOverwriteProof<'a, C> {
    fn new(
        structure: &'a TreeTransformStructure<C>,
        dst_structure: &'a Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        src_len: usize,
        nout: usize,
    ) -> Result<Option<Self>, OperationError> {
        structure.validate_replay_structures(dst_structure, src_structure)?;
        validate_replay_storage_len(src_structure, src_len)?;
        let required_len = dst_structure.required_len()?;
        if structure.physical_overwrite_len() != Some(required_len) || nout > dst_structure.rank() {
            return Ok(None);
        }
        let Some(regions) = dst_structure.coupled_sector_regions(nout)? else {
            return Ok(None);
        };
        let mut next = 0usize;
        for region in regions.iter() {
            let range = region.range();
            if range.start != next || range.end > required_len {
                return Ok(None);
            }
            next = range.end;
        }
        if next != required_len {
            return Ok(None);
        }
        Ok(Some(Self {
            structure,
            dst_structure,
            required_len,
            nout,
        }))
    }
}

/// Host scratch/replay workspace backed by `Vec<T>`.
///
/// Raw replay methods using this workspace operate on host slices. Device
/// execution should use a separate device workspace instead of hiding device
/// storage behind this type.
#[derive(Clone, Debug)]
pub struct HostTreeTransformWorkspace<T> {
    zero_strides: Vec<isize>,
    packed: TreeTransformScratchBuffers<HostScratchBuffer<T>, HostScratchBuffer<T>>,
    // Recoupling matrices converted into the data scalar type for the GEMM
    // application (TensorKit's basistransform buffer).
    coefficient_scratch: Vec<T>,
    coefficient_structure_identity: Option<Weak<()>>,
    chunk_jobs: Vec<DenseGemmBatchJob>,
    chunk_runs: Vec<usize>,
    chunk_scatter_groups: Vec<usize>,
}

pub type TreeTransformWorkspace<T> = HostTreeTransformWorkspace<T>;

impl<T> Default for HostTreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            packed: TreeTransformScratchBuffers::default(),
            coefficient_scratch: Vec::new(),
            coefficient_structure_identity: None,
            chunk_jobs: Vec::new(),
            chunk_runs: Vec::new(),
            chunk_scatter_groups: Vec::new(),
        }
    }
}

impl<T> HostTreeTransformWorkspace<T> {
    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    #[inline]
    pub fn is_host_workspace(&self) -> bool {
        self.placement() == Placement::Host
    }

    pub fn source_len(&self) -> usize {
        self.packed.source().len()
    }

    pub fn destination_len(&self) -> usize {
        self.packed.destination().len()
    }

    #[cfg(test)]
    fn packed_capacities(&self) -> (usize, usize) {
        (
            self.packed.source().capacity(),
            self.packed.destination().capacity(),
        )
    }

    fn prepare_packed_buffers(&mut self, source_len: usize, destination_len: usize, zero: T)
    where
        T: Clone,
    {
        self.packed
            .source_mut()
            .resize_filled(source_len, zero.clone());
        self.packed
            .destination_mut()
            .resize_filled(destination_len, zero);
    }
}

impl<T> ReportsPlacement for HostTreeTransformWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

fn ensure_recoupling_coefficients<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
) -> Result<bool, OperationError>
where
    D: RecouplingCoefficientAction<C>,
    C: Copy,
{
    let plan = structure.recoupling_plan();
    let same_structure = workspace
        .coefficient_structure_identity
        .as_ref()
        .and_then(Weak::upgrade)
        .is_some_and(|identity| Arc::ptr_eq(&identity, structure.identity_marker()));
    if same_structure && workspace.coefficient_scratch.len() == plan.coefficient_len() {
        return Ok(false);
    }

    workspace.coefficient_scratch.clear();
    workspace
        .coefficient_scratch
        .reserve(plan.coefficient_len());
    for (block_index, _) in plan.entries() {
        let block = recoupling_multi_block(structure, block_index)?;
        let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            coefficient_start,
            ..
        } = *block
        else {
            continue;
        };
        let coefficient_len = src_count
            .checked_mul(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let coefficient_end = coefficient_start
            .checked_add(coefficient_len)
            .ok_or(OperationError::ElementCountOverflow)?;
        let coefficients = structure
            .recoupling_coefficients_dst_src()
            .get(coefficient_start..coefficient_end)
            .ok_or(OperationError::CoefficientCountMismatch {
                expected: coefficient_end,
                actual: structure.recoupling_coefficients_dst_src().len(),
            })?;
        workspace.coefficient_scratch.extend(
            coefficients
                .iter()
                .map(|&coefficient| D::coefficient_as_data(coefficient)),
        );
    }
    if workspace.coefficient_scratch.len() != plan.coefficient_len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: plan.coefficient_len(),
            actual: workspace.coefficient_scratch.len(),
        });
    }
    workspace.coefficient_structure_identity = Some(Arc::downgrade(structure.identity_marker()));
    Ok(true)
}

fn recoupling_multi_block<C: Copy>(
    structure: &TreeTransformStructure<C>,
    block_index: usize,
) -> Result<&TreeTransformBlock, OperationError> {
    // Lazy error construction: recoupling_multi_block is called per block on the
    // hot replay path (pack/recouple/scatter). Eager .ok_or built the
    // BlockIndexOutOfBounds struct on every success too, which the d=4 bisect
    // (see issue #103) attributed to the compose regression. .ok_or_else only
    // builds it on the never-taken out-of-bounds path.
    let block = structure.blocks().get(block_index).ok_or_else(|| {
        OperationError::BlockIndexOutOfBounds {
            tensor: "recoupling block",
            index: block_index,
            count: structure.blocks().len(),
        }
    })?;
    match block {
        TreeTransformBlock::Multi { .. } => Ok(block),
        TreeTransformBlock::Single { .. } => Err(OperationError::BlockIndexOutOfBounds {
            tensor: "recoupling block",
            index: block_index,
            count: structure.blocks().len(),
        }),
    }
}

fn scale_inactive_destinations<A, D, C>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    structure: &TreeTransformStructure<C>,
    dst_data: &mut [D],
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + PartialEq + Zero + One,
    C: Copy,
{
    match mode {
        DestinationMode::Axpby(beta) => {
            if beta == D::one() {
                return Ok(());
            }
            // Scaling the complete storage would also mutate padding not owned by any
            // block, so compile only the destination layouts with no active replay.
            for &layout_index in structure.inactive_destination_layouts() {
                let layout = structure.layouts().entry(layout_index);
                kernels.scale_strided(
                    dst_data,
                    structure.layouts().shape(layout),
                    structure.layouts().strides(layout),
                    layout.offset,
                    beta,
                )?;
            }
        }
        DestinationMode::Overwrite => {
            let zero = [D::zero()];
            for &layout_index in structure.inactive_destination_layouts() {
                let layout = structure.layouts().entry(layout_index);
                zero_strides.clear();
                zero_strides.resize(structure.layouts().shape(layout).len(), 0);
                kernels.copy_scale_strided(
                    dst_data,
                    &zero,
                    structure.layouts().shape(layout),
                    structure.layouts().strides(layout),
                    zero_strides,
                    layout.offset,
                    0,
                    false,
                    D::one(),
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod coefficient_cache_tests {
    use super::*;
    use crate::TreeTransformBlockSpec;

    fn multi_recoupling_structure(coefficients: [f64; 4]) -> TreeTransformStructure<f64> {
        let block_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                coefficients.to_vec(),
            )],
        )
        .unwrap()
    }

    #[test]
    fn recoupling_coefficients_cache_uses_live_structure_identity() {
        let structure = multi_recoupling_structure([1.0, 2.0, 3.0, 4.0]);
        let mut workspace = TreeTransformWorkspace::<f64>::default();

        assert!(ensure_recoupling_coefficients(&mut workspace, &structure).unwrap());
        assert_eq!(workspace.coefficient_scratch, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!ensure_recoupling_coefficients(&mut workspace, &structure).unwrap());

        let structure_clone = structure.clone();
        assert!(!ensure_recoupling_coefficients(&mut workspace, &structure_clone).unwrap());

        let equal_but_distinct = multi_recoupling_structure([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(structure, equal_but_distinct);
        workspace.coefficient_scratch.fill(-1.0);
        assert!(ensure_recoupling_coefficients(&mut workspace, &equal_but_distinct).unwrap());
        assert_eq!(workspace.coefficient_scratch, vec![1.0, 2.0, 3.0, 4.0]);
    }
}

#[cfg(test)]
mod inactive_destination_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use std::time::Duration;
    use tenet_core::{BlockKey, BlockSpec, TensorMapSpace, Trivial};
    use tenet_dense::{
        DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
        DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite,
    };

    type TestTensor = TensorMap<f64, 1, 0, Trivial, Vec<f64>>;

    fn fixture() -> (TestTensor, TestTensor, TreeTransformStructure<f64>) {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        let src: TestTensor = TensorMap::from_vec_with_structure(
            vec![3.0],
            TensorMapSpace::from_dims([1], []).unwrap(),
            src_structure,
        )
        .unwrap();
        let dst = TensorMap::from_vec_with_structure(
            vec![10.0, 20.0],
            TensorMapSpace::from_dims([2], []).unwrap(),
            dst_structure,
        )
        .unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 2.0)],
        )
        .unwrap();
        (dst, src, structure)
    }

    fn expected(beta: f64) -> [f64; 2] {
        [6.0 + beta * 10.0, beta * 20.0]
    }

    fn custom_structure(blocks: Vec<BlockSpec>) -> BlockStructure {
        BlockStructure::from_blocks_with_rank(1, blocks).unwrap()
    }

    fn block(sector: usize, shape: usize, stride: usize, offset: usize) -> BlockSpec {
        BlockSpec::with_key(BlockKey::ordinal(sector), vec![shape], vec![stride], offset).unwrap()
    }

    fn identity_multi_fixture() -> (
        Arc<BlockStructure>,
        Arc<BlockStructure>,
        TreeTransformStructure<f64>,
    ) {
        let src = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let dst =
            Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1]]).unwrap());
        let replay = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        (dst, src, replay)
    }

    struct FailFirstBatchExecutor {
        inner: DefaultDenseExecutor,
        fail_next: bool,
    }

    impl FailFirstBatchExecutor {
        fn new() -> Self {
            Self {
                inner: DefaultDenseExecutor::new(),
                fail_next: true,
            }
        }
    }

    impl DenseExecutor for FailFirstBatchExecutor {
        fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.svd(input)
        }

        fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.qr(input)
        }

        fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.eigh(input)
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.inner.dot_general_into(output, lhs, rhs, config)
        }

        fn matmul_batch_axpby_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            jobs: &[DenseGemmBatchJob],
            runs: &[usize],
            alpha: DenseScalar,
            beta: DenseScalar,
        ) -> Result<(), DenseError> {
            if self.fail_next {
                self.fail_next = false;
                match output {
                    DenseWrite::F64(mut output) => output.data_mut().fill(f64::NAN),
                    _ => unreachable!("retry oracle uses f64"),
                }
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: "matmul_batch_axpby_into",
                    message: "injected first-call failure".to_string(),
                });
            }
            self.inner
                .matmul_batch_axpby_into(output, lhs, rhs, jobs, runs, alpha, beta)
        }
    }

    #[test]
    fn compile_rejects_inactive_destination_aliases() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
        assert_eq!(
            TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[],)
                .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_active_inactive_destination_aliases() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_active_destination_aliases() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[
                    TreeTransformBlockSpec::single(0, 0, 1.0),
                    TreeTransformBlockSpec::single(1, 1, 1.0),
                ],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_self_overlapping_destination_layout() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 0, 0)]);
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap"
            }
        );
    }

    #[test]
    fn compile_rejects_nonzero_stride_self_overlap() {
        let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
        let dst_structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(BlockKey::opaque([0]), vec![2, 2], vec![1, 1], 0).unwrap()],
        )
        .unwrap();
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap_err(),
            OperationError::InvalidArgument {
                message: "tree transform destination layouts overlap",
            }
        );
    }

    #[test]
    fn interleaved_disjoint_destinations_with_overlapping_ranges_are_valid() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]);
        let structure =
            TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
                .unwrap();
        let mut dst = vec![10.0, 20.0, 30.0, 40.0];
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::new(dst_structure),
            &Arc::new(src_structure),
            &mut dst,
            &[3.0],
            1.0,
            0.5,
            4,
        )
        .unwrap();
        assert_eq!(dst, [5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn threaded_active_interleaved_layout_uses_the_serial_fallback() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]);
        let structure = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, -1.0),
            ],
        )
        .unwrap();
        assert!(!structure.parallel_schedule().singles_slice_disjoint);
        let src = [1.0, 2.0, 3.0, 4.0];
        let mut serial = [10.0, 20.0, 30.0, 40.0];
        let mut threaded = serial;
        let dst_structure = Arc::new(dst_structure);
        let src_structure = Arc::new(src_structure);
        for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                dst,
                &src,
                1.0,
                0.5,
                threads,
            )
            .unwrap();
        }

        assert_eq!(threaded, serial);
        assert_eq!(threaded, [7.0, 7.0, 19.0, 16.0]);
    }

    #[test]
    fn threaded_replay_handles_rank_zero_blocks() {
        let structure = Arc::new(
            BlockStructure::packed_column_major(0, [Vec::<usize>::new(), Vec::new()]).unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, -1.0),
            ],
        )
        .unwrap();
        let src = [3.0, 5.0];
        let mut serial = [10.0, 20.0];
        let mut threaded = serial;
        for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                dst,
                &src,
                1.0,
                0.5,
                threads,
            )
            .unwrap();
        }
        assert_eq!(threaded, serial);
        assert_eq!(threaded, [11.0, 5.0]);
    }

    #[test]
    fn threaded_replay_ignores_zero_extent_work() {
        let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        assert!(transform.parallel_schedule().singles.is_empty());
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut [],
            &[],
            1.0,
            f64::NAN,
            4,
        )
        .unwrap();
    }

    #[test]
    fn zero_extent_profile_counts_match_serial_replay() {
        let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let mut serial = TreeTransformReplayProfile::default();
        let mut threaded = TreeTransformReplayProfile::default();
        for (profile, threads) in [(&mut serial, 1), (&mut threaded, 4)] {
            tree_transform_structure_with_structural_recoupling_raw_profiled(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut [],
                &[],
                1.0,
                0.0,
                threads,
                profile,
            )
            .unwrap();
        }

        assert_eq!(threaded.single_blocks, serial.single_blocks);
        assert_eq!(threaded.multi_blocks, serial.multi_blocks);
        assert_eq!(threaded.packed_columns, serial.packed_columns);
        assert_eq!(threaded.scattered_columns, serial.scattered_columns);
    }

    #[test]
    fn zero_extent_multi_profile_counts_match_serial_replay() {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![0], vec![0]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        let mut serial = TreeTransformReplayProfile::default();
        let mut threaded = TreeTransformReplayProfile::default();
        for (profile, threads) in [(&mut serial, 1), (&mut threaded, 4)] {
            tree_transform_structure_with_structural_recoupling_raw_profiled(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut [],
                &[],
                1.0,
                0.0,
                threads,
                profile,
            )
            .unwrap();
        }

        assert_eq!(threaded.multi_blocks, serial.multi_blocks);
        assert_eq!(threaded.packed_columns, serial.packed_columns);
        assert_eq!(threaded.scattered_columns, serial.scattered_columns);
        assert_eq!(threaded.packed_columns, 2);
        assert_eq!(threaded.scattered_columns, 2);
    }

    #[test]
    fn threaded_multi_scatter_falls_back_for_interleaved_destinations() {
        let src_structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
        let dst_structure = Arc::new(custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]));
        let transform = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        assert_eq!(transform.parallel_schedule().scatter_groups.len(), 1);
        assert!(!transform.parallel_schedule().scatter_groups[0].slice_disjoint);
        let src = [1.0, 2.0, 3.0, 4.0];
        let mut serial = [10.0, 20.0, 30.0, 40.0];
        let mut threaded = serial;
        for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst_structure,
                &src_structure,
                dst,
                &src,
                1.0,
                0.5,
                threads,
            )
            .unwrap();
        }

        assert_eq!(threaded, serial);
        assert_eq!(threaded, [6.0, 13.0, 17.0, 24.0]);
    }

    #[test]
    fn many_range_connected_interleaved_destinations_are_valid() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure =
            custom_structure((0..64).map(|offset| block(offset, 2, 64, offset)).collect());
        TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
            .unwrap();
    }

    #[test]
    fn inactive_destination_scaling_preserves_storage_padding() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 2, 2, 0)]);
        let structure =
            TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
                .unwrap();
        let mut dst = vec![10.0, 99.0, 30.0];
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::new(dst_structure),
            &Arc::new(src_structure),
            &mut dst,
            &[3.0],
            1.0,
            0.5,
            4,
        )
        .unwrap();
        assert_eq!(dst, [5.0, 99.0, 15.0]);
    }

    #[test]
    fn nan_beta_reaches_active_and_inactive_destinations() {
        let (mut dst, src, structure) = fixture();
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            f64::NAN,
            4,
        )
        .unwrap();
        assert!(dst.data().iter().all(|value| value.is_nan()));
    }

    #[test]
    fn generic_beta_zero_keeps_existing_ieee_inactive_behavior() {
        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);

        tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(dst.data()[0], 6.0);
        assert!(dst.data()[1].is_nan());
    }

    #[test]
    fn overwrite_single_does_not_read_nan_destinations_in_any_driver() {
        for threads in [1, 2] {
            let (mut dst, src, structure) = fixture();
            dst.data_mut().fill(f64::NAN);
            tree_transform_structure_overwrite_with_structural_recoupling(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                threads,
            )
            .unwrap();
            assert_eq!(dst.data(), &[6.0, 0.0]);
        }

        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        tree_transform_structure_overwrite_with_storage_workspace_strided_kernel(
            &mut StridedHostKernelAdapter::default(),
            &mut StorageTreeTransformWorkspace::<Vec<f64>, Vec<f64>>::default(),
            &structure,
            &mut dst,
            &src,
            1.0,
        )
        .unwrap();
        assert_eq!(dst.data(), &[6.0, 0.0]);

        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
        )
        .unwrap();
        assert_eq!(dst.data(), &[6.0, 0.0]);

        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            1,
            &mut TreeTransformReplayProfile::default(),
        )
        .unwrap();
        assert_eq!(dst.data(), &[6.0, 0.0]);
    }

    #[test]
    fn overwrite_multi_does_not_read_nan_active_or_inactive_destinations() {
        let (dst_structure, src_structure, structure) = identity_multi_fixture();
        let mut dst = vec![f64::NAN; 3];

        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut dst,
            &[3.0, 4.0],
            2.0,
        )
        .unwrap();

        assert_eq!(dst, [6.0, 8.0, 0.0]);
    }

    #[test]
    fn overwrite_multi_c64_does_not_read_nan_destinations() {
        let (dst_structure, src_structure, structure) = identity_multi_fixture();
        for threads in [1, 4] {
            let nan = num_complex::Complex64::new(f64::NAN, f64::NAN);
            let mut dst = vec![nan; 3];
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &[
                    num_complex::Complex64::new(3.0, 1.0),
                    num_complex::Complex64::new(4.0, -1.0),
                ],
                num_complex::Complex64::new(2.0, 0.0),
                threads,
            )
            .unwrap();
            assert_eq!(
                dst,
                [
                    num_complex::Complex64::new(6.0, 2.0),
                    num_complex::Complex64::new(8.0, -2.0),
                    num_complex::Complex64::new(0.0, 0.0),
                ]
            );
        }
    }

    #[test]
    fn overwrite_threaded_single_multi_and_storage_multi_ignore_destination_bits() {
        let src_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1]]).unwrap(),
        );
        let dst_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1], vec![1]])
                .unwrap(),
        );
        let structure = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, -1.0),
                TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![1.0, 0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
        let src = [3.0, 4.0, 5.0, 6.0];
        let expected = [6.0, -4.0, 5.0, 6.0, 0.0];

        for threads in [1, 4] {
            let mut dst = [f64::NAN; 5];
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &src,
                1.0,
                threads,
            )
            .unwrap();
            assert_eq!(dst, expected);
        }

        let src: TestTensor = TensorMap::from_vec_with_structure(
            src.to_vec(),
            TensorMapSpace::from_dims([4], []).unwrap(),
            Arc::unwrap_or_clone(src_structure),
        )
        .unwrap();
        let mut dst: TestTensor = TensorMap::from_vec_with_structure(
            vec![f64::NAN; 5],
            TensorMapSpace::from_dims([5], []).unwrap(),
            Arc::unwrap_or_clone(dst_structure),
        )
        .unwrap();
        tree_transform_structure_overwrite_with_storage_workspace_strided_kernel(
            &mut StridedHostKernelAdapter::default(),
            &mut StorageTreeTransformWorkspace::<Vec<f64>, Vec<f64>>::default(),
            &structure,
            &mut dst,
            &src,
            1.0,
        )
        .unwrap();
        assert_eq!(dst.data(), &expected);
    }

    #[test]
    fn overwrite_multi_recovers_from_dirty_packed_scratch_after_failure() {
        let (dst_structure, src_structure, replay) = identity_multi_fixture();
        let mut workspace = TreeTransformWorkspace::default();
        let mut dense = FailFirstBatchExecutor::new();
        let mut dst = [f64::NAN; 3];

        assert!(
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut workspace,
                &replay,
                &dst_structure,
                &src_structure,
                &mut dst,
                &[3.0, 4.0],
                2.0,
                1,
            )
            .is_err()
        );
        assert!(workspace
            .packed
            .destination()
            .as_slice()
            .iter()
            .all(|value| value.is_nan()));

        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut workspace,
            &replay,
            &dst_structure,
            &src_structure,
            &mut dst,
            &[3.0, 4.0],
            2.0,
            1,
        )
        .unwrap();
        assert_eq!(dst, [6.0, 8.0, 0.0]);
    }

    #[test]
    fn overwrite_validates_before_mutation_and_accepts_rank_boundaries() {
        let (mut dst, src, structure) = fixture();
        dst.data_mut().fill(f64::NAN);
        let before = dst.data().to_vec();
        let result = tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            &[],
            1.0,
        );
        assert!(result.is_err());
        assert_eq!(
            dst.data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let scalar_structure =
            Arc::new(BlockStructure::packed_column_major(0, [Vec::<usize>::new()]).unwrap());
        let scalar_replay = TreeTransformStructure::compile_structures(
            &scalar_structure,
            &scalar_structure,
            &[TreeTransformBlockSpec::single(0, 0, 2.0)],
        )
        .unwrap();
        let mut scalar_dst = [f64::NAN];
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &scalar_replay,
            &scalar_structure,
            &scalar_structure,
            &mut scalar_dst,
            &[3.0],
            1.0,
        )
        .unwrap();
        assert_eq!(scalar_dst, [6.0]);

        let empty_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
        let empty_replay = TreeTransformStructure::compile_structures(
            &empty_structure,
            &empty_structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &empty_replay,
            &empty_structure,
            &empty_structure,
            &mut [],
            &[],
            1.0,
        )
        .unwrap();
    }

    #[test]
    fn profiled_overwrite_zeros_inactive_layout_without_touching_padding() {
        let src_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
        let dst_structure = Arc::new(custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 2)]));
        let structure = TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[TreeTransformBlockSpec::single(0, 0, 2.0)],
        )
        .unwrap();
        let mut dst = [f64::NAN, 99.0, f64::NAN];
        let mut profile = TreeTransformReplayProfile::default();

        tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut dst,
            &[3.0],
            1.0,
            2,
            &mut profile,
        )
        .unwrap();

        assert_eq!(dst, [6.0, 99.0, 0.0]);
        assert_eq!(profile.single_blocks, 1);
    }

    #[derive(Clone, Default)]
    struct SlowScaleAdapter(StridedHostKernelAdapter);

    impl HostKernelAdapter<f64> for SlowScaleAdapter {
        fn add_strided(
            &mut self,
            zero_strides: &mut Vec<isize>,
            dst_data: &mut [f64],
            src_data: &[f64],
            shape: &[usize],
            dst_strides: &[isize],
            src_strides: &[isize],
            dst_offset: isize,
            src_offset: isize,
            source_conjugate: bool,
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            self.0.add_strided(
                zero_strides,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                source_conjugate,
                alpha,
                beta,
            )
        }

        fn axpby_strided(
            &mut self,
            dst_data: &mut [f64],
            src_data: &[f64],
            shape: &[usize],
            dst_strides: &[isize],
            src_strides: &[isize],
            dst_offset: isize,
            src_offset: isize,
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            self.0.axpby_strided(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                alpha,
                beta,
            )
        }

        fn copy_scale_strided(
            &mut self,
            dst_data: &mut [f64],
            src_data: &[f64],
            shape: &[usize],
            dst_strides: &[isize],
            src_strides: &[isize],
            dst_offset: isize,
            src_offset: isize,
            source_conjugate: bool,
            alpha: f64,
        ) -> Result<(), OperationError> {
            self.0.copy_scale_strided(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                source_conjugate,
                alpha,
            )
        }

        fn scale_strided(
            &mut self,
            dst_data: &mut [f64],
            shape: &[usize],
            dst_strides: &[isize],
            dst_offset: isize,
            beta: f64,
        ) -> Result<(), OperationError> {
            std::thread::sleep(Duration::from_millis(40));
            self.0
                .scale_strided(dst_data, shape, dst_strides, dst_offset, beta)
        }

        fn recoupling_src_times_u_transpose<C>(
            &mut self,
            destination: &mut [f64],
            source: &[f64],
            coefficients: &[C],
            coefficient_start: usize,
            element_count: usize,
            src_count: usize,
            dst_count: usize,
        ) -> Result<(), OperationError>
        where
            C: Copy,
            f64: RecouplingCoefficientAction<C>,
        {
            self.0.recoupling_src_times_u_transpose(
                destination,
                source,
                coefficients,
                coefficient_start,
                element_count,
                src_count,
                dst_count,
            )
        }
    }

    #[test]
    fn profiled_replay_attributes_inactive_destination_scaling() {
        let (mut dst, src, structure) = fixture();
        let mut profile = TreeTransformReplayProfile::default();
        tree_transform_structure_with_structural_recoupling_raw_profiled(
            &mut SlowScaleAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            0.5,
            1,
            &mut profile,
        )
        .unwrap();
        let attributed = profile.validate
            + profile.single_total
            + profile.strided_kernel.saturating_sub(profile.single_total)
            + profile.multi_workspace_prepare
            + profile.multi_pack
            + profile.multi_coefficient_prepare
            + profile.multi_matmul_total
            + profile.multi_scatter;
        assert!(profile.total.saturating_sub(attributed) < Duration::from_millis(20));
    }

    #[test]
    fn structural_serial_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            tree_transform_structure_with_structural_recoupling(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                beta,
                1,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }

    #[test]
    fn structural_threaded_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            tree_transform_structure_with_structural_recoupling(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                beta,
                2,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }

    #[test]
    fn storage_workspace_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            tree_transform_structure_with_storage_workspace_strided_kernel(
                &mut StridedHostKernelAdapter::default(),
                &mut StorageTreeTransformWorkspace::<Vec<f64>, Vec<f64>>::default(),
                &structure,
                &mut dst,
                &src,
                1.0,
                beta,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }

    #[test]
    fn strided_replay_scales_inactive_destinations() {
        for beta in [0.0, 0.5, 1.0] {
            let (mut dst, src, structure) = fixture();
            let dst_structure = Arc::clone(dst.structure());
            let src_structure = Arc::clone(src.structure());
            tree_transform_structure_with_strided_kernel_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                dst.data_mut(),
                src.data(),
                1.0,
                beta,
            )
            .unwrap();
            assert_eq!(dst.data(), &expected(beta));
        }
    }
}

pub fn tensoradd_structure_with_strided_kernel<
    T,
    const NOUT: usize,
    const NIN: usize,
    S,
    DDst,
    DSrc,
>(
    allocator: &mut HostAllocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    let descriptor = structure.descriptor();
    structure.validate_replay_structures(dst.structure(), src.structure())?;
    if dst.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: dst.structure().block_count(),
            src: descriptor.terms().len(),
        });
    }
    if src.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: descriptor.terms().len(),
            src: src.structure().block_count(),
        });
    }

    let zero_strides = &mut allocator.zero_strides;
    let dst_data = dst.data_mut();
    let src_data = src.data();
    for term in descriptor.terms() {
        tensoradd_prepared_block_with_strided_kernel(
            zero_strides,
            descriptor,
            term,
            dst_data,
            src_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

pub fn tree_transform_structure_with_strided_kernel<
    A,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_with_strided_kernel_raw(
        kernels,
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

pub fn tree_transform_structure_with_storage_workspace_strided_kernel<
    A,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    workspace: &mut StorageTreeTransformWorkspace<DSrc::Similar, DDst::Similar>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
    DDst: HostWritableStorage<D> + SimilarStorage<D>,
    DSrc: HostReadableStorage<D> + SimilarStorage<D>,
    DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    DSrc::Similar: HostWritableStorage<D> + ScratchStorage<D>,
{
    tree_transform_structure_with_storage_workspace_strided_kernel_mode(
        kernels,
        workspace,
        structure,
        dst,
        src,
        alpha,
        DestinationMode::Axpby(beta),
    )
}

pub fn tree_transform_structure_overwrite_with_storage_workspace_strided_kernel<
    A,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    workspace: &mut StorageTreeTransformWorkspace<DSrc::Similar, DDst::Similar>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
    DDst: HostWritableStorage<D> + SimilarStorage<D>,
    DSrc: HostReadableStorage<D> + SimilarStorage<D>,
    DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    DSrc::Similar: HostWritableStorage<D> + ScratchStorage<D>,
{
    tree_transform_structure_with_storage_workspace_strided_kernel_mode(
        kernels,
        workspace,
        structure,
        dst,
        src,
        alpha,
        DestinationMode::Overwrite,
    )
}

fn tree_transform_structure_with_storage_workspace_strided_kernel_mode<
    A,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    workspace: &mut StorageTreeTransformWorkspace<DSrc::Similar, DDst::Similar>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
    DDst: HostWritableStorage<D> + SimilarStorage<D>,
    DSrc: HostReadableStorage<D> + SimilarStorage<D>,
    DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    DSrc::Similar: HostWritableStorage<D> + ScratchStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    structure.validate_replay_structures(&dst_structure, &src_structure)?;
    validate_replay_storage_len(&dst_structure, dst.storage().len())?;
    validate_replay_storage_len(&src_structure, src.storage().len())?;

    scale_inactive_destinations(
        kernels,
        workspace.zero_strides_mut(),
        structure,
        dst.data_mut(),
        mode,
    )?;

    let src_data = src.data();
    for block in structure.blocks() {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                kernels,
                workspace.zero_strides_mut(),
                structure.layouts(),
                dst_layout,
                src_layout,
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst.data_mut(),
                src_data,
                alpha,
                mode,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => {
                let source_len = element_count
                    .checked_mul(src_count)
                    .ok_or(OperationError::ElementCountOverflow)?;
                let destination_len = element_count
                    .checked_mul(dst_count)
                    .ok_or(OperationError::ElementCountOverflow)?;
                workspace.prepare_from_storages(
                    src.storage(),
                    dst.storage(),
                    source_len,
                    destination_len,
                    D::zero(),
                );
                let (zero_strides, scratch) = workspace.replay_parts_mut();
                tree_transform_multi_with_scratch_buffers(
                    kernels,
                    zero_strides,
                    scratch,
                    structure.layouts(),
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                    structure.recoupling_coefficients_dst_src(),
                    structure.storage_conjugate(),
                    dst.data_mut(),
                    src_data,
                    alpha,
                    mode,
                )?;
            }
        }
    }
    Ok(())
}

/// Replays a prepared tree-transform structure on host slices.
pub fn tree_transform_structure_with_strided_kernel_raw<A, D, C>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    tree_transform_structure_with_strided_kernel_raw_mode(
        kernels,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Axpby(beta),
    )
}

pub fn tree_transform_structure_overwrite_with_strided_kernel_raw<A, D, C>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    tree_transform_structure_with_strided_kernel_raw_mode(
        kernels,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Overwrite,
    )
}

fn tree_transform_structure_with_strided_kernel_raw_mode<A, D, C>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    scale_inactive_destinations(
        kernels,
        &mut workspace.zero_strides,
        structure,
        dst_data,
        mode,
    )?;
    for block in structure.blocks() {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                kernels,
                &mut workspace.zero_strides,
                structure.layouts(),
                dst_layout,
                src_layout,
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                mode,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => tree_transform_multi_with_pack_gemm_scatter(
                kernels,
                workspace,
                structure.layouts(),
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                structure.recoupling_coefficients_dst_src(),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                mode,
            )?,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_with_structural_recoupling<
    A,
    E,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_with_structural_recoupling_raw(
        kernels,
        dense,
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
        threads,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_overwrite_with_structural_recoupling<
    A,
    E,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        kernels,
        dense,
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        threads,
    )
}

/// Replays a prepared structural-recoupling tree transform on host slices.
///
/// `threads` selects the replay parallelism (a property of the executing
/// backend, not of the cached structure): `<= 1` reuses one pack buffer and
/// submits each Multi group independently; `> 1` runs Single applies, Multi
/// pack columns and Multi scatter columns as independent work items over up to
/// `threads` work-stealing workers. Multi blocks submit bounded grouped
/// recoupling batches between their pack and scatter phases.
#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_with_structural_recoupling_raw<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_mode(
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Axpby(beta),
        threads,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_overwrite_with_structural_recoupling_raw<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_mode(
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Overwrite,
        threads,
    )
}

/// Internal, unstable owned-output path for the serial built-in host executor.
///
/// Returns `Ok(None)` without allocating output when the destination does not
/// have a proof of exact physical overwrite coverage.
///
/// Why public: `tenet-tensors` is a separate crate. This concrete-executor seam
/// is not a general backend API; downstream callers must not rely on it.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn try_tree_transform_structure_overwrite_owned_raw<D, C>(
    dense: &mut DefaultDenseExecutor,
    workspace: &mut TreeTransformWorkspace<D>,
    transpose_backend: crate::TransposeBackend,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    nout: usize,
    src_data: &[D],
    alpha: D,
) -> Result<Option<Vec<D>>, OperationError>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let Some(proof) = PhysicalOverwriteProof::new(
        structure,
        dst_structure,
        src_structure,
        src_data.len(),
        nout,
    )?
    else {
        return Ok(None);
    };
    debug_assert_eq!(proof.required_len, dst_structure.required_len()?);
    debug_assert_eq!(proof.nout, nout);
    debug_assert!(Arc::ptr_eq(proof.dst_structure, dst_structure));
    debug_assert!(core::ptr::eq(proof.structure, structure));

    initialize_owned(proof.required_len, |dst_data| {
        let layouts = structure.layouts();
        let recoupling_plan = structure.recoupling_plan();
        let mut kernels =
            crate::StridedHostKernelAdapter::with_transpose_backend(transpose_backend);

        for &layout_index in structure.inactive_destination_layouts() {
            write_uninit_layout_zero(layouts, layouts.entry(layout_index), dst_data)?;
        }
        for block in structure.blocks() {
            let TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } = *block
            else {
                continue;
            };
            write_uninit_layout_from_source(
                layouts,
                dst_layout,
                src_layout,
                dst_data,
                src_data,
                structure.storage_conjugate(),
                alpha.scale_by_coefficient(structure.coefficient(coefficient)),
            )?;
        }

        if recoupling_plan.is_empty() {
            return Ok(());
        }
        ensure_recoupling_coefficients(workspace, structure)?;
        for (block_index, job) in recoupling_plan.entries() {
            let TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                element_count,
                ..
            } = *recoupling_multi_block(structure, block_index)?
            else {
                unreachable!("recoupling_multi_block only returns Multi blocks");
            };
            let source_len = element_count
                .checked_mul(src_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            let destination_len = element_count
                .checked_mul(dst_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
            for src_index in 0..src_count {
                pack_layout_into_column(
                    &mut kernels,
                    layouts,
                    src_layout_start + src_index,
                    src_data,
                    workspace.packed.source_mut().as_mut_slice(),
                    src_index * element_count,
                    structure.storage_conjugate(),
                )?;
            }
            {
                let local_job = DenseGemmBatchJob {
                    dst_offset: 0,
                    lhs_offset: 0,
                    ..*job
                };
                let (source, destination) = workspace.packed.source_and_destination_mut();
                recoupling_gemm_batch(
                    dense,
                    destination.as_mut_slice(),
                    source.as_slice(),
                    &workspace.coefficient_scratch,
                    core::slice::from_ref(&local_job),
                    &[1],
                )?;
            }
            for dst_index in 0..dst_count {
                write_uninit_layout_from_packed(
                    layouts,
                    dst_layout_start + dst_index,
                    dst_data,
                    workspace.packed.destination().as_slice(),
                    dst_index * element_count,
                    alpha,
                )?;
            }
        }
        Ok(())
    })
    .map(Some)
}

#[cfg(test)]
mod bounded_workspace_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;

    fn many_group_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>, Vec<f64>) {
        const GROUPS: usize = 32;
        let mut shapes = Vec::with_capacity(2 * GROUPS);
        let mut specs = Vec::with_capacity(GROUPS);
        let mut source = Vec::new();
        for group in 0..GROUPS {
            let elements = group % 8 + 1;
            shapes.push(vec![elements]);
            shapes.push(vec![elements]);
            let first = 2 * group;
            specs.push(TreeTransformBlockSpec::multi(
                vec![first, first + 1],
                vec![first, first + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            ));
            let base = source.len();
            source.extend((0..2 * elements).map(|index| (base + index + 1) as f64));
        }
        let structure = Arc::new(BlockStructure::packed_column_major(1, shapes).unwrap());
        let transform =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        (structure, transform, source)
    }

    #[test]
    fn many_groups_bound_serial_and_parallel_pack_scratch_by_concurrency() {
        let (structure, transform, source) = many_group_fixture();
        let total_source = transform.recoupling_plan().source_len();
        let total_destination = transform.recoupling_plan().destination_len();

        for threads in [1, 3] {
            let mut destination = vec![0.0; source.len()];
            let mut workspace = TreeTransformWorkspace::default();
            tree_transform_structure_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut workspace,
                &transform,
                &structure,
                &structure,
                &mut destination,
                &source,
                1.0,
                0.0,
                threads,
            )
            .unwrap();

            // What: 32 independent groups replay exactly, while retained pack
            // capacity follows at most `threads` largest groups rather than all.
            assert_eq!(destination, source);
            let (source_capacity, destination_capacity) = workspace.packed_capacities();
            let chunk_source_bound = (threads * 2 * 8).next_power_of_two();
            let chunk_destination_bound = (threads * 2 * 8).next_power_of_two();
            assert!(source_capacity <= chunk_source_bound);
            assert!(destination_capacity <= chunk_destination_bound);
            assert!(source_capacity < total_source);
            assert!(destination_capacity < total_destination);
        }
    }

    #[test]
    fn bounded_group_replay_preserves_complex_alpha_beta() {
        let (structure, transform, source) = many_group_fixture();
        let source = source
            .into_iter()
            .map(|value| Complex64::new(value, -value))
            .collect::<Vec<_>>();
        let initial = Complex64::new(2.0, -3.0);
        let alpha = Complex64::new(0.5, 0.25);
        let beta = Complex64::new(-0.5, 0.75);
        let mut destination = vec![initial; source.len()];

        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut destination,
            &source,
            alpha,
            beta,
            3,
        )
        .unwrap();

        // What: chunk boundaries do not change complex recoupling or axpby.
        let expected = source
            .iter()
            .map(|&value| alpha * value + beta * initial)
            .collect::<Vec<_>>();
        assert_eq!(destination, expected);
    }
}

#[cfg(test)]
mod owned_overwrite_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;
    use tenet_core::{
        BlockKey, BlockSpec, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
        FusionTreePairKey, SectorLeg, TensorMapSpace, Z2FusionRule, Z2Irrep,
    };

    fn canonical_structure(offset: usize) -> Arc<BlockStructure> {
        let key = BlockKey::from(
            FusionTreePairKey::try_pair_from_sector_ids(
                [1],
                [1],
                1,
                [false],
                [false],
                [],
                [],
                [],
                [],
            )
            .unwrap(),
        );
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                2,
                vec![BlockSpec::with_key(key, vec![2, 3], vec![1, 2], offset).unwrap()],
            )
            .unwrap(),
        )
    }

    #[test]
    fn owned_writer_matches_initialized_oracle_for_real_and_complex() {
        // What: canonical serial owned replay writes every physical value and
        // is byte-for-byte equal to the initialized overwrite oracle.
        let structure = canonical_structure(0);
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, -2.0)],
        )
        .unwrap();

        let mut expected = vec![f64::NAN; 6];
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut expected,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3.0,
            1,
        )
        .unwrap();
        let actual = try_tree_transform_structure_overwrite_owned_raw(
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            crate::TransposeBackend::FusedLoops,
            &transform,
            &structure,
            &structure,
            1,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3.0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(actual, expected);

        let complex_src = (1..=6)
            .map(|value| Complex64::new(value as f64, -(value as f64)))
            .collect::<Vec<_>>();
        let complex = try_tree_transform_structure_overwrite_owned_raw(
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            crate::TransposeBackend::FusedLoops,
            &transform,
            &structure,
            &structure,
            1,
            &complex_src,
            Complex64::new(3.0, 1.0),
        )
        .unwrap()
        .unwrap();
        let scale = Complex64::new(3.0, 1.0) * -2.0;
        assert_eq!(
            complex,
            complex_src.iter().map(|&v| scale * v).collect::<Vec<_>>()
        );
    }

    #[test]
    fn owned_writer_multi_matches_direct_matrix_oracle() {
        // What: the uninitialized Multi writer applies every destination-by-source
        // recoupling coefficient and writes the final owned payload exactly once.
        let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new(
                    [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
                    false,
                )]),
                FusionProductSpace::new([SectorLeg::new(
                    [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
                    false,
                )]),
            ),
            &Z2FusionRule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap();
        let structure = Arc::clone(space.subblock_structure());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![2.0, 3.0, 5.0, 7.0],
            )],
        )
        .unwrap();

        let actual = try_tree_transform_structure_overwrite_owned_raw(
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            crate::TransposeBackend::FusedLoops,
            &transform,
            &structure,
            &structure,
            1,
            &[11.0, 13.0],
            2.0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            actual,
            [
                2.0 * (2.0 * 11.0 + 3.0 * 13.0),
                2.0 * (5.0 * 11.0 + 7.0 * 13.0)
            ]
        );
    }

    #[test]
    fn owned_writer_rejects_padding_and_out_of_range_split_before_allocation() {
        // What: a holey destination and an out-of-range split receive no owned
        // result, leaving the caller on the initialized fallback path.
        let canonical = canonical_structure(0);
        let padded = canonical_structure(1);
        let padded_transform = TreeTransformStructure::compile_structures(
            &padded,
            &canonical,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let mut dense = DefaultDenseExecutor::new();
        let mut workspace = TreeTransformWorkspace::default();
        assert!(try_tree_transform_structure_overwrite_owned_raw(
            &mut dense,
            &mut workspace,
            crate::TransposeBackend::FusedLoops,
            &padded_transform,
            &padded,
            &canonical,
            1,
            &[1.0; 6],
            1.0,
        )
        .unwrap()
        .is_none());

        let canonical_transform = TreeTransformStructure::compile_structures(
            &canonical,
            &canonical,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        assert!(try_tree_transform_structure_overwrite_owned_raw(
            &mut dense,
            &mut workspace,
            crate::TransposeBackend::FusedLoops,
            &canonical_transform,
            &canonical,
            &canonical,
            3,
            &[1.0; 6],
            1.0,
        )
        .unwrap()
        .is_none());
    }
}

fn layout_linear_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[isize],
    base: isize,
) -> Result<usize, OperationError> {
    let mut offset = base;
    for (&dim, &stride) in shape.iter().zip(strides) {
        let coordinate = if dim == 0 { 0 } else { linear % dim };
        if dim != 0 {
            linear /= dim;
        }
        let coordinate =
            isize::try_from(coordinate).map_err(|_| OperationError::ElementCountOverflow)?;
        offset = offset
            .checked_add(
                coordinate
                    .checked_mul(stride)
                    .ok_or(OperationError::ElementCountOverflow)?,
            )
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    usize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })
}

/// Fused loop nest over a prebaked layout writing into uninitialized memory
/// (issue #232, condition 2), mirroring `apply_fused_pair_slices` but with
/// `MaybeUninit::write` for the destination. Each destination offset is visited
/// exactly once, identical to `layout_linear_offset`'s odometer, so the
/// write-once-then-`assume_init` invariant of `initialize_owned` (#226/#233) is
/// preserved: the normalization only drops extent-1 axes, reorders, and fuses
/// contiguous runs — the *set* of visited (dst, src) offsets is unchanged, and
/// there is no read-after-write within a single writer (`src` is a disjoint,
/// fully-initialized slice). Rank is bounded by the fusion limit.
fn write_fused_uninit<D, F>(
    baked: BakedFusedLayout<'_>,
    dst: &mut [MaybeUninit<D>],
    src: &[D],
    dst_offset: isize,
    src_offset: isize,
    map: F,
) where
    D: Copy,
    F: Fn(D) -> D,
{
    let dims = baked.dims();
    let dst_strides = baked.dst_strides();
    let src_strides = baked.src_strides();
    let rank = dims.len();
    if rank == 0 || dims.iter().any(|&dim| dim == 0) {
        return;
    }
    let inner_len = dims[0];
    let inner_dst = dst_strides[0];
    let inner_src = src_strides[0];
    let mut index = [0usize; 8];
    let mut dst_base = dst_offset;
    let mut src_base = src_offset;
    loop {
        for position in 0..inner_len {
            let dst_position = (dst_base + position as isize * inner_dst) as usize;
            let src_position = (src_base + position as isize * inner_src) as usize;
            dst[dst_position].write(map(src[src_position]));
        }
        let mut axis = 1;
        loop {
            if axis >= rank {
                return;
            }
            index[axis] += 1;
            dst_base += dst_strides[axis];
            src_base += src_strides[axis];
            if index[axis] < dims[axis] {
                break;
            }
            dst_base -= dims[axis] as isize * dst_strides[axis];
            src_base -= dims[axis] as isize * src_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

// Why-not fuse the zero writer: it has no paired source view, and a pure
// permute — the deg=1 U(1) owned-path regime this optimization targets — always
// touches every destination block, so `inactive_destination_layouts` is empty
// and this writer never runs on the hot path. A dedicated single-side fused
// walk would add a fourth baked role for no measured win, so it stays on the
// per-element odometer (issue #232, condition 2).
fn write_uninit_layout_zero<D: Zero + Copy>(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    dst: &mut [MaybeUninit<D>],
) -> Result<(), OperationError> {
    for linear in 0..layout.element_count {
        let index = layout_linear_offset(
            linear,
            layouts.shape(layout),
            layouts.strides(layout),
            layout.offset,
        )?;
        dst[index].write(D::zero());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_uninit_layout_from_source<D>(
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    src_index: usize,
    dst: &mut [MaybeUninit<D>],
    src: &[D],
    conjugate: bool,
    scale: D,
) -> Result<(), OperationError>
where
    D: Copy + Mul<D, Output = D> + ConjugateValue,
{
    let dst_layout = layouts.entry(dst_index);
    let src_layout = layouts.entry(src_index);
    if let Some(baked) = layouts.fused_baked(dst_index) {
        write_fused_uninit(
            baked,
            dst,
            src,
            dst_layout.offset,
            src_layout.offset,
            move |value| scale * value.maybe_conj(conjugate),
        );
        return Ok(());
    }
    for linear in 0..dst_layout.element_count {
        let dst_index = layout_linear_offset(
            linear,
            layouts.shape(dst_layout),
            layouts.strides(dst_layout),
            dst_layout.offset,
        )?;
        let src_index = layout_linear_offset(
            linear,
            layouts.shape(src_layout),
            layouts.strides(src_layout),
            src_layout.offset,
        )?;
        dst[dst_index].write(scale * src[src_index].maybe_conj(conjugate));
    }
    Ok(())
}

fn write_uninit_layout_from_packed<D>(
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    dst: &mut [MaybeUninit<D>],
    packed: &[D],
    packed_offset: usize,
    alpha: D,
) -> Result<(), OperationError>
where
    D: Copy + Mul<D, Output = D>,
{
    let layout = layouts.entry(dst_index);
    if let Some(baked) = layouts.fused_baked(dst_index) {
        // The scatter role bakes src = packed (column-major) strides, so the
        // fused walk over `packed` starting at `packed_offset` reproduces the
        // odometer's `packed[packed_offset + linear]` column-major gather.
        write_fused_uninit(
            baked,
            dst,
            packed,
            layout.offset,
            offset_to_isize(packed_offset)?,
            move |value| alpha * value,
        );
        return Ok(());
    }
    for linear in 0..layout.element_count {
        let dst_index = layout_linear_offset(
            linear,
            layouts.shape(layout),
            layouts.strides(layout),
            layout.offset,
        )?;
        dst[dst_index].write(alpha * packed[packed_offset + linear]);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_structure_with_structural_recoupling_raw_mode<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    scale_inactive_destinations(
        kernels,
        &mut workspace.zero_strides,
        structure,
        dst_data,
        mode,
    )?;
    if threads > 1 {
        return tree_transform_blocks_with_batched_recoupling_parallel(
            kernels, dense, workspace, structure, dst_data, src_data, alpha, mode, threads, None,
        );
    }
    tree_transform_blocks_with_batched_recoupling(
        kernels, dense, workspace, structure, dst_data, src_data, alpha, mode, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_with_structural_recoupling_raw_profiled<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_profiled_mode(
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Axpby(beta),
        threads,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    threads: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_profiled_mode(
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Overwrite,
        threads,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_structure_with_structural_recoupling_raw_profiled_mode<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
    threads: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let total_start = std::time::Instant::now();

    let start = std::time::Instant::now();
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    profile.validate += start.elapsed();

    let start = std::time::Instant::now();
    scale_inactive_destinations(
        kernels,
        &mut workspace.zero_strides,
        structure,
        dst_data,
        mode,
    )?;
    profile.strided_kernel += start.elapsed();

    if threads > 1 {
        tree_transform_blocks_with_batched_recoupling_parallel(
            kernels,
            dense,
            workspace,
            structure,
            dst_data,
            src_data,
            alpha,
            mode,
            threads,
            Some(profile),
        )?;
    } else {
        tree_transform_blocks_with_batched_recoupling(
            kernels,
            dense,
            workspace,
            structure,
            dst_data,
            src_data,
            alpha,
            mode,
            Some(profile),
        )?;
    }

    profile.total += total_start.elapsed();
    Ok(())
}

/// Executes a validated tree-transform block list against a dense executor.
/// Single blocks apply directly; each Multi block reuses one source/destination
/// pack pair for `destination = source * U^T`.
///
/// Inlined into both the plain and profiled entry points so the
/// `Option<&mut profile>` checks constant-fold away in the unprofiled copy.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tree_transform_blocks_with_batched_recoupling<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let layouts = structure.layouts();
    let recoupling_plan = structure.recoupling_plan();

    // All-Single structures (abelian recoupling is diagonal) skip the batch
    // machinery entirely: no pack scratch, no job list, no scatter pass.
    if recoupling_plan.is_empty() {
        for block in structure.blocks() {
            let TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } = *block
            else {
                unreachable!("checked above: no Multi blocks");
            };
            let start = profile.as_ref().map(|_| std::time::Instant::now());
            tree_transform_single_with_strided_kernel(
                kernels,
                &mut workspace.zero_strides,
                layouts,
                dst_layout,
                src_layout,
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                mode,
            )?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                let elapsed = start.elapsed();
                profile.single_blocks += 1;
                profile.single_total += elapsed;
                profile.strided_kernel += elapsed;
            }
        }
        return Ok(());
    }

    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let converted = ensure_recoupling_coefficients(workspace, structure)?;
    if converted {
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_coefficient_prepare += start.elapsed();
        }
    }

    // Singles apply directly in replay order. Multi blocks are packed through
    // the compile-time recoupling entries, whose order is chosen to form
    // same-shape strided GEMM runs.
    for block in structure.blocks() {
        let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = *block
        else {
            continue;
        };
        // Timestamps only under profiling: the per-block clock reads are
        // measurable against microsecond replays.
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        tree_transform_single_with_strided_kernel(
            kernels,
            &mut workspace.zero_strides,
            layouts,
            dst_layout,
            src_layout,
            structure.coefficient(coefficient),
            structure.storage_conjugate(),
            dst_data,
            src_data,
            alpha,
            mode,
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.single_blocks += 1;
            profile.single_total += elapsed;
            profile.strided_kernel += elapsed;
        }
    }

    for (block_index, job) in recoupling_plan.entries() {
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            element_count,
            ..
        } = *recoupling_multi_block(structure, block_index)?
        else {
            unreachable!("recoupling_multi_block only returns Multi blocks");
        };
        debug_assert_eq!(job.rows, element_count);
        debug_assert_eq!(job.contracted, src_count);
        debug_assert_eq!(job.cols, dst_count);
        let source_len = element_count
            .checked_mul(src_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let destination_len = element_count
            .checked_mul(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_workspace_prepare += start.elapsed();
        }
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        for src_index in 0..src_count {
            pack_layout_into_column(
                kernels,
                layouts,
                src_layout_start + src_index,
                src_data,
                workspace.packed.source_mut().as_mut_slice(),
                src_index * element_count,
                structure.storage_conjugate(),
            )?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_blocks += 1;
            profile.packed_columns += src_count;
            profile.multi_pack += start.elapsed();
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        {
            let local_job = DenseGemmBatchJob {
                dst_offset: 0,
                lhs_offset: 0,
                ..*job
            };
            let (source, destination) = workspace.packed.source_and_destination_mut();
            recoupling_gemm_batch(
                dense,
                destination.as_mut_slice(),
                source.as_slice(),
                &workspace.coefficient_scratch,
                core::slice::from_ref(&local_job),
                &[1],
            )?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_scalar_recoupling += elapsed;
            profile.multi_matmul_total += elapsed;
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        for dst_index in 0..dst_count {
            scatter_column_into_layout(
                kernels,
                &mut workspace.zero_strides,
                layouts,
                dst_layout_start + dst_index,
                workspace.packed.destination().as_slice(),
                dst_index * element_count,
                dst_data,
                alpha,
                mode,
            )?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.scattered_columns += dst_count;
            profile.multi_scatter += start.elapsed();
        }
    }
    Ok(())
}

fn parallel_split(items: usize, threads: usize) -> usize {
    let left_threads = threads / 2;
    items
        .saturating_mul(left_threads)
        .div_ceil(threads)
        .clamp(1, items - 1)
}

#[allow(clippy::too_many_arguments)]
fn replay_pack_columns<A, D>(
    mut kernels: A,
    layouts: &TreeTransformLayoutTable,
    items: &[TreeTransformPackReplay],
    packed_source: &mut [D],
    packed_start: usize,
    src_data: &[D],
    storage_conjugate: bool,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + ConjugateValue,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        for item in items {
            pack_layout_into_column(
                &mut kernels,
                layouts,
                item.src_layout,
                src_data,
                packed_source,
                item.packed_offset - packed_start,
                storage_conjugate,
            )?;
        }
        return Ok(());
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = items[middle].packed_offset;
    let (left_data, right_data) = packed_source.split_at_mut(boundary - packed_start);
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let right_kernels = kernels.clone();
    let (left, right) = rayon::join(
        || {
            replay_pack_columns(
                kernels,
                layouts,
                left_items,
                left_data,
                packed_start,
                src_data,
                storage_conjugate,
                left_threads,
            )
        },
        || {
            replay_pack_columns(
                right_kernels,
                layouts,
                right_items,
                right_data,
                boundary,
                src_data,
                storage_conjugate,
                right_threads,
            )
        },
    );
    left?;
    right
}

#[allow(clippy::too_many_arguments)]
fn replay_single_blocks<A, D, C>(
    mut kernels: A,
    structure: &TreeTransformStructure<C>,
    items: &[TreeTransformSingleReplay],
    dst_data: &mut [D],
    dst_start: isize,
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        let mut zero_strides = Vec::new();
        for item in items {
            let dst_layout = structure.layouts().entry(item.dst_layout);
            let src_layout = structure.layouts().entry(item.src_layout);
            let baked = structure.layouts().fused_baked(item.dst_layout);
            let scale = alpha.scale_by_coefficient(structure.coefficient(item.coefficient));
            match mode {
                DestinationMode::Axpby(beta) => kernels.add_strided_baked(
                    &mut zero_strides,
                    dst_data,
                    src_data,
                    structure.layouts().shape(dst_layout),
                    structure.layouts().strides(dst_layout),
                    structure.layouts().strides(src_layout),
                    dst_layout.offset - dst_start,
                    src_layout.offset,
                    structure.storage_conjugate(),
                    scale,
                    beta,
                    baked,
                )?,
                DestinationMode::Overwrite => kernels.copy_scale_strided_baked(
                    dst_data,
                    src_data,
                    structure.layouts().shape(dst_layout),
                    structure.layouts().strides(dst_layout),
                    structure.layouts().strides(src_layout),
                    dst_layout.offset - dst_start,
                    src_layout.offset,
                    structure.storage_conjugate(),
                    scale,
                    baked,
                )?,
            }
        }
        return Ok(());
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = items[middle].dst_lo;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst_data.split_at_mut(split);
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let right_kernels = kernels.clone();
    let (left, right) = rayon::join(
        || {
            replay_single_blocks(
                kernels,
                structure,
                left_items,
                left_data,
                dst_start,
                src_data,
                alpha,
                mode,
                left_threads,
            )
        },
        || {
            replay_single_blocks(
                right_kernels,
                structure,
                right_items,
                right_data,
                boundary,
                src_data,
                alpha,
                mode,
                right_threads,
            )
        },
    );
    left?;
    right
}

#[allow(clippy::too_many_arguments)]
fn replay_scatter_columns<A, D>(
    mut kernels: A,
    layouts: &TreeTransformLayoutTable,
    items: &[TreeTransformScatterReplay],
    dst_data: &mut [D],
    dst_start: isize,
    packed_destination: &[D],
    packed_start: usize,
    alpha: D,
    mode: DestinationMode<D>,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + ConjugateValue,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        for item in items {
            let layout = layouts.entry(item.dst_layout);
            let baked = layouts.fused_baked(item.dst_layout);
            match mode {
                DestinationMode::Axpby(beta) => kernels.axpby_strided_baked(
                    dst_data,
                    packed_destination,
                    layouts.shape(layout),
                    layouts.strides(layout),
                    layouts.packed_strides(layout),
                    layout.offset - dst_start,
                    offset_to_isize(item.packed_offset - packed_start)?,
                    alpha,
                    beta,
                    baked,
                )?,
                DestinationMode::Overwrite => kernels.copy_scale_strided_baked(
                    dst_data,
                    packed_destination,
                    layouts.shape(layout),
                    layouts.strides(layout),
                    layouts.packed_strides(layout),
                    layout.offset - dst_start,
                    offset_to_isize(item.packed_offset - packed_start)?,
                    false,
                    alpha,
                    baked,
                )?,
            }
        }
        return Ok(());
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = items[middle].dst_lo;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst_data.split_at_mut(split);
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let right_kernels = kernels.clone();
    let (left, right) = rayon::join(
        || {
            replay_scatter_columns(
                kernels,
                layouts,
                left_items,
                left_data,
                dst_start,
                packed_destination,
                packed_start,
                alpha,
                mode,
                left_threads,
            )
        },
        || {
            replay_scatter_columns(
                right_kernels,
                layouts,
                right_items,
                right_data,
                boundary,
                packed_destination,
                packed_start,
                alpha,
                mode,
                right_threads,
            )
        },
    );
    left?;
    right
}

#[allow(clippy::too_many_arguments)]
fn replay_scatter_groups<A, D>(
    kernels: A,
    layouts: &TreeTransformLayoutTable,
    scatter_columns: &[TreeTransformScatterReplay],
    scatter_groups: &[TreeTransformScatterGroupReplay],
    groups: &[usize],
    dst_data: &mut [D],
    dst_start: isize,
    packed_destination: &[D],
    packed_start: usize,
    alpha: D,
    mode: DestinationMode<D>,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + ConjugateValue,
{
    if groups.is_empty() {
        return Ok(());
    }
    if threads <= 1 || groups.len() == 1 {
        for &group in groups {
            let columns = scatter_groups[group].columns.clone();
            replay_scatter_columns(
                kernels.clone(),
                layouts,
                &scatter_columns[columns],
                dst_data,
                dst_start,
                packed_destination,
                packed_start,
                alpha,
                mode,
                1,
            )?;
        }
        return Ok(());
    }

    let middle = parallel_split(groups.len(), threads);
    let boundary = scatter_columns[scatter_groups[groups[middle]].columns.start].dst_lo;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst_data.split_at_mut(split);
    let (left_groups, right_groups) = groups.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let right_kernels = kernels.clone();
    let (left, right) = rayon::join(
        || {
            replay_scatter_groups(
                kernels,
                layouts,
                scatter_columns,
                scatter_groups,
                left_groups,
                left_data,
                dst_start,
                packed_destination,
                packed_start,
                alpha,
                mode,
                left_threads,
            )
        },
        || {
            replay_scatter_groups(
                right_kernels,
                layouts,
                scatter_columns,
                scatter_groups,
                right_groups,
                right_data,
                boundary,
                packed_destination,
                packed_start,
                alpha,
                mode,
                right_threads,
            )
        },
    );
    left?;
    right
}

/// Threaded variant of [`tree_transform_blocks_with_batched_recoupling`]
/// (TensorKit `_add_abelian_kernel_threaded!` / `_add_general_kernel_threaded!`
/// precedent, indexmanipulations.jl:520-738):
///
/// - Singles apply in parallel when their compiled slices are disjoint. Multi
///   blocks replay in concurrency-bounded chunks whose pack and scatter phases
///   each enter Rayon once. Work items are independent because the compile step
///   rejects duplicate destination blocks
///   (`OperationError::DuplicateTransformDestination`) and pack columns are
///   disjoint scratch ranges by construction; the workspace forbids `unsafe`,
///   so disjointness is realized structurally by recursively splitting the
///   buffers at compiled boundaries (`split_at_mut`) and rebasing offsets,
///   instead of TensorKit-style shared writes. Interleaved layouts whose
///   bounding slices overlap stay serial: exact element disjointness is not
///   enough to create independent Rust slices without unsafe code.
/// - Each chunk is one serial grouped GEMM call between its pack and scatter
///   phases. The dense executor owns its own parallelism, so no nesting arises.
///
/// Parallel copy scheduling uses recursive `rayon::join` on the global pool,
/// capped by the configured worker count. Replay descriptors and safe split
/// boundaries are compiled into the structure.
///
/// Per-task state is one cloned kernel adapter (a ZST for the strided adapter)
/// and one `Vec::new()` zero-strides scratch. Numerical and chunk-descriptor
/// storage stays in the reused workspace.
///
/// Profiling attribution is phase-level; per-item clocks across workers would
/// measure contention, not work.
#[allow(clippy::too_many_arguments)]
fn tree_transform_blocks_with_batched_recoupling_parallel<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = structure.layouts();
    let recoupling_plan = structure.recoupling_plan();
    let schedule = structure.parallel_schedule();

    let single_count = schedule.single_block_count;
    let multi_count = recoupling_plan.jobs().len();
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let converted = ensure_recoupling_coefficients(workspace, structure)?;
    if converted {
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_coefficient_prepare += start.elapsed();
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.single_blocks += single_count;
        profile.multi_blocks += multi_count;
    }

    let storage_conjugate = structure.storage_conjugate();

    // Single destinations are independent of every Multi destination.
    {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        if schedule.singles_slice_disjoint {
            replay_single_blocks(
                kernels.clone(),
                structure,
                &schedule.singles,
                dst_data,
                0,
                src_data,
                alpha,
                mode,
                threads,
            )?;
        } else {
            let mut zero_strides = Vec::new();
            for item in &schedule.singles {
                tree_transform_single_with_strided_kernel(
                    kernels,
                    &mut zero_strides,
                    layouts,
                    item.dst_layout,
                    item.src_layout,
                    structure.coefficient(item.coefficient),
                    storage_conjugate,
                    dst_data,
                    src_data,
                    alpha,
                    mode,
                )?;
            }
        }

        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.single_total += elapsed;
            profile.strided_kernel += elapsed;
        }
    }

    // Why not retain one pack arena for the whole plan: TensorKit owns scratch
    // per concurrently executing fusion block. The existing replay-thread
    // budget is also the natural bound for a grouped dense submission.
    let chunk_size = threads.max(1);
    let mut pack_cursor = 0;
    for (chunk_index, chunk) in recoupling_plan.jobs().chunks(chunk_size).enumerate() {
        let packed_column_count = chunk.iter().map(|job| job.contracted).sum::<usize>();
        let scattered_column_count = chunk.iter().map(|job| job.cols).sum::<usize>();
        let source_start = chunk[0].lhs_offset;
        let destination_start = chunk[0].dst_offset;
        let source_end = chunk
            .last()
            .and_then(|job| {
                job.rows
                    .checked_mul(job.contracted)
                    .and_then(|len| job.lhs_offset.checked_add(len))
            })
            .ok_or(OperationError::ElementCountOverflow)?;
        let destination_end = chunk
            .last()
            .and_then(|job| {
                job.rows
                    .checked_mul(job.cols)
                    .and_then(|len| job.dst_offset.checked_add(len))
            })
            .ok_or(OperationError::ElementCountOverflow)?;

        workspace.chunk_jobs.clear();
        workspace
            .chunk_jobs
            .extend(chunk.iter().map(|job| DenseGemmBatchJob {
                dst_offset: job.dst_offset - destination_start,
                lhs_offset: job.lhs_offset - source_start,
                ..*job
            }));
        strided_batch_runs_into(&workspace.chunk_jobs, &mut workspace.chunk_runs);

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        workspace.prepare_packed_buffers(
            source_end - source_start,
            destination_end - destination_start,
            D::zero(),
        );
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_workspace_prepare += start.elapsed();
        }

        let pack_start = pack_cursor;
        while pack_cursor < schedule.pack_columns.len()
            && schedule.pack_columns[pack_cursor].packed_offset < source_end
        {
            pack_cursor += 1;
        }
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        replay_pack_columns(
            kernels.clone(),
            layouts,
            &schedule.pack_columns[pack_start..pack_cursor],
            workspace.packed.source_mut().as_mut_slice(),
            source_start,
            src_data,
            storage_conjugate,
            threads,
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.packed_columns += packed_column_count;
            profile.multi_pack += start.elapsed();
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let (source, destination) = workspace.packed.source_and_destination_mut();
        recoupling_gemm_batch(
            dense,
            destination.as_mut_slice(),
            source.as_slice(),
            &workspace.coefficient_scratch,
            &workspace.chunk_jobs,
            &workspace.chunk_runs,
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_scalar_recoupling += elapsed;
            profile.multi_matmul_total += elapsed;
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let packed_destination = workspace.packed.destination().as_slice();
        let first_group = chunk_index * chunk_size;
        workspace.chunk_scatter_groups.clear();
        workspace.chunk_scatter_groups.extend(
            (first_group..first_group + chunk.len())
                .filter(|&group| !schedule.scatter_groups[group].columns.is_empty()),
        );
        // Why not sort/copy every scatter descriptor: only the at-most-T group
        // indices need destination order for one safe Rayon split tree.
        workspace
            .chunk_scatter_groups
            .sort_unstable_by_key(|&group| {
                schedule.scatter_columns[schedule.scatter_groups[group].columns.start].dst_lo
            });
        let scatter_slice_disjoint = workspace
            .chunk_scatter_groups
            .iter()
            .all(|&group| schedule.scatter_groups[group].slice_disjoint)
            && workspace.chunk_scatter_groups.windows(2).all(|groups| {
                let left_end = schedule.scatter_groups[groups[0]].columns.end;
                let right_start = schedule.scatter_groups[groups[1]].columns.start;
                schedule.scatter_columns[left_end - 1].dst_hi
                    < schedule.scatter_columns[right_start].dst_lo
            });
        if scatter_slice_disjoint {
            replay_scatter_groups(
                kernels.clone(),
                layouts,
                &schedule.scatter_columns,
                &schedule.scatter_groups,
                &workspace.chunk_scatter_groups,
                dst_data,
                0,
                packed_destination,
                destination_start,
                alpha,
                mode,
                threads,
            )?;
        } else {
            let mut zero_strides = Vec::new();
            for &group in &workspace.chunk_scatter_groups {
                for item in
                    &schedule.scatter_columns[schedule.scatter_groups[group].columns.clone()]
                {
                    scatter_column_into_layout(
                        kernels,
                        &mut zero_strides,
                        layouts,
                        item.dst_layout,
                        packed_destination,
                        item.packed_offset - destination_start,
                        dst_data,
                        alpha,
                        mode,
                    )?;
                }
            }
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.scattered_columns += scattered_column_count;
            profile.multi_scatter += start.elapsed();
        }
    }
    debug_assert_eq!(pack_cursor, schedule.pack_columns.len());
    Ok(())
}

pub fn tensoradd_block_with_strided_kernel<T>(
    allocator: &mut HostAllocator,
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let dst_shape = dst.shape().to_vec();
    let dst_strides = crate::strided::strides_to_isize(dst.strides())?;
    let dst_offset = offset_to_isize(dst.offset())?;
    let (dst_data, _) = dst.into_parts();
    let src_shape = src.shape().to_vec();
    let src_strides = crate::strided::strides_to_isize(src.strides())?;
    let src_offset = offset_to_isize(src.offset())?;
    let src_data = src.data();

    if dst_shape != src_shape {
        return Err(OperationError::ShapeMismatch {
            dst: dst_shape,
            src: src_shape,
        });
    }

    tensoradd_raw_strided_kernel(
        &mut allocator.zero_strides,
        dst_data,
        src_data,
        &dst_shape,
        &dst_strides,
        &src_strides,
        dst_offset,
        src_offset,
        false,
        alpha,
        beta,
    )
}

fn tensoradd_prepared_block_with_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    descriptor: &TensorAddDescriptor,
    term: &TensorAddDescriptorTerm,
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    tensoradd_raw_strided_kernel_trusted(
        zero_strides,
        dst_data,
        src_data,
        descriptor.shape(term),
        descriptor.dst_strides(term),
        descriptor.src_strides(term),
        term.dst_offset,
        term.src_offset,
        descriptor.source_conjugate(),
        alpha,
        beta,
    )
}

pub(crate) fn validate_replay_storage_len(
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

pub(crate) fn zero_tree_transform_destination<A, D>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    dst_structure: &BlockStructure,
    dst_data: &mut [D],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + Zero + One,
{
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    let zero = [D::zero()];
    let mut dst_strides = Vec::new();
    for block_index in 0..dst_structure.block_count() {
        let block = dst_structure.block(block_index)?;
        zero_strides.clear();
        zero_strides.resize(block.shape().len(), 0);
        dst_strides.clear();
        for &stride in block.strides() {
            dst_strides
                .push(isize::try_from(stride).map_err(|_| OperationError::ElementCountOverflow)?);
        }
        kernels.copy_scale_strided(
            dst_data,
            &zero,
            block.shape(),
            &dst_strides,
            zero_strides,
            offset_to_isize(block.offset())?,
            0,
            false,
            D::one(),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn tree_transform_single_with_strided_kernel<A, D, C>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    src_index: usize,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let dst_layout = layouts.entry(dst_index);
    let src_layout = layouts.entry(src_index);
    let shape = layouts.shape(dst_layout);
    let baked = layouts.fused_baked(dst_index);
    let scale = alpha.scale_by_coefficient(coefficient);
    match mode {
        DestinationMode::Axpby(beta) => kernels.add_strided_baked(
            zero_strides,
            dst_data,
            src_data,
            shape,
            layouts.strides(dst_layout),
            layouts.strides(src_layout),
            dst_layout.offset,
            src_layout.offset,
            source_conjugate,
            scale,
            beta,
            baked,
        ),
        DestinationMode::Overwrite => kernels.copy_scale_strided_baked(
            dst_data,
            src_data,
            shape,
            layouts.strides(dst_layout),
            layouts.strides(src_layout),
            dst_layout.offset,
            src_layout.offset,
            source_conjugate,
            scale,
            baked,
        ),
    }
}

/// Applies a batch of Multi-block recoupling matrices over shared flat scratch
/// buffers: per job, the column-major
/// (element_count x dst_count) destination block receives `source_block *
/// U^T`, with `recoupling_coefficients_dst_src` (row-major `U[dst, src]`)
/// reinterpreted as the column-major (src_count x dst_count) matrix `U^T`.
/// This is TensorKit's `_add_transform_multi!` `mul!` step submitted as one
/// grouped call; the naive per-element loop in the kernel adapter remains
/// only for adapters without a dense executor. Job offsets are relative to the
/// supplied chunk scratch, matching the trusted-view validation contract.
fn recoupling_gemm_batch<E, D>(
    dense: &mut E,
    destination: &mut [D],
    source: &[D],
    coefficients: &[D],
    jobs: &[DenseGemmBatchJob],
    runs: &[usize],
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar,
{
    let dst_shape = [destination.len()];
    let lhs_shape = [source.len()];
    let rhs_shape = [coefficients.len()];
    let flat_strides = [1];
    let lhs = D::dense_read(tenet_dense::DenseView::new_trusted(
        source,
        &lhs_shape,
        &flat_strides,
        0,
    ));
    let rhs = D::dense_read(tenet_dense::DenseView::new_trusted(
        coefficients,
        &rhs_shape,
        &flat_strides,
        0,
    ));
    let output = D::dense_write(tenet_dense::DenseViewMut::new_trusted(
        destination,
        &dst_shape,
        &flat_strides,
        0,
    ));
    dense
        .matmul_batch_axpby_into(
            output,
            lhs,
            rhs,
            jobs,
            runs,
            D::one().dense_scalar(),
            D::zero().dense_scalar(),
        )
        .map_err(OperationError::Dense)
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_pack_gemm_scatter<A, D, C>(
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + Zero + One + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
    tree_transform_multi_with_scratch_buffers(
        kernels,
        &mut workspace.zero_strides,
        &mut workspace.packed,
        layouts,
        dst_layout_start,
        dst_count,
        src_layout_start,
        src_count,
        coefficient_start,
        element_count,
        recoupling_coefficients_dst_src,
        source_conjugate,
        dst_data,
        src_data,
        alpha,
        mode,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_scratch_buffers<A, D, C, SourceScratch, DestinationScratch>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    scratch: &mut TreeTransformScratchBuffers<SourceScratch, DestinationScratch>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + One + RecouplingCoefficientAction<C>,
    C: Copy,
    SourceScratch: HostWritableStorage<D>,
    DestinationScratch: HostWritableStorage<D>,
{
    for src_index in 0..src_count {
        pack_layout_into_column(
            kernels,
            layouts,
            src_layout_start + src_index,
            src_data,
            scratch.source_mut().as_mut_slice(),
            src_index * element_count,
            source_conjugate,
        )?;
    }

    {
        let (source, destination) = scratch.source_and_destination_mut();
        kernels.recoupling_src_times_u_transpose(
            destination.as_mut_slice(),
            source.as_slice(),
            recoupling_coefficients_dst_src,
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
    }

    for dst_index in 0..dst_count {
        scatter_column_into_layout(
            kernels,
            zero_strides,
            layouts,
            dst_layout_start + dst_index,
            scratch.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            mode,
        )?;
    }
    Ok(())
}

fn pack_layout_into_column<A, T>(
    kernels: &mut A,
    layouts: &TreeTransformLayoutTable,
    entry_index: usize,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One,
{
    let layout = layouts.entry(entry_index);
    let shape = layouts.shape(layout);
    let baked = layouts.fused_baked(entry_index);
    let packed_offset = offset_to_isize(packed_offset)?;
    kernels.copy_scale_strided_baked(
        packed,
        src_data,
        shape,
        layouts.packed_strides(layout),
        layouts.strides(layout),
        packed_offset,
        layout.offset,
        source_conjugate,
        T::one(),
        baked,
    )
}

#[allow(clippy::too_many_arguments)]
fn scatter_column_into_layout<A, T>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    entry_index: usize,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    mode: DestinationMode<T>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    let layout = layouts.entry(entry_index);
    let shape = layouts.shape(layout);
    let baked = layouts.fused_baked(entry_index);
    match mode {
        DestinationMode::Axpby(beta) => {
            zero_strides.clear();
            kernels.axpby_strided_baked(
                dst_data,
                packed,
                shape,
                layouts.strides(layout),
                layouts.packed_strides(layout),
                layout.offset,
                offset_to_isize(packed_offset)?,
                alpha,
                beta,
                baked,
            )
        }
        DestinationMode::Overwrite => kernels.copy_scale_strided_baked(
            dst_data,
            packed,
            shape,
            layouts.strides(layout),
            layouts.packed_strides(layout),
            layout.offset,
            offset_to_isize(packed_offset)?,
            false,
            alpha,
            baked,
        ),
    }
}
