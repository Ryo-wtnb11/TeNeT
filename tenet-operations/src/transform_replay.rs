use core::ops::{Add, Mul};
use std::sync::{Arc, Weak};

use num_traits::{One, Zero};
use tenet_core::{
    BlockStructure, BlockView, BlockViewMut, HostReadableStorage, HostWritableStorage, Placement,
    ScratchStorage, SimilarStorage, TensorMap,
};
use tenet_dense::{DenseExecutor, DenseGemmBatchJob};

use crate::host_scratch::HostScratchBuffer;
use crate::storage_scratch::{StorageTreeTransformWorkspace, TreeTransformScratchBuffers};
use crate::strided::offset_to_isize;
use crate::tensoradd::{TensorAddDescriptor, TensorAddDescriptorTerm};
use crate::{
    tensoradd_raw_strided_kernel, tensoradd_raw_strided_kernel_trusted, ConjugateValue,
    DenseRecouplingScalar, HostAllocator, HostKernelAdapter, OperationError,
    RecouplingCoefficientAction, ReportsPlacement, TensorAddStructure, TreeTransformBlock,
    TreeTransformLayout, TreeTransformLayoutTable, TreeTransformReplayProfile,
    TreeTransformStructure,
};

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
    // application (TensorKit's basistransform buffer); replay packs every
    // Multi block's matrix into this one buffer so the recoupling GEMMs
    // submit as a single batch.
    coefficient_scratch: Vec<T>,
    coefficient_structure_identity: Option<Weak<()>>,
}

pub type TreeTransformWorkspace<T> = HostTreeTransformWorkspace<T>;

impl<T> Default for HostTreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            packed: TreeTransformScratchBuffers::default(),
            coefficient_scratch: Vec::new(),
            coefficient_structure_identity: None,
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
            .recoupling_coefficients_dst_src
            .get(coefficient_start..coefficient_end)
            .ok_or(OperationError::CoefficientCountMismatch {
                expected: coefficient_end,
                actual: structure.recoupling_coefficients_dst_src.len(),
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

fn recoupling_multi_block<C>(
    structure: &TreeTransformStructure<C>,
    block_index: usize,
) -> Result<&TreeTransformBlock, OperationError> {
    // Lazy error construction: recoupling_multi_block is called per block on the
    // hot replay path (pack/recouple/scatter). Eager .ok_or built the
    // BlockIndexOutOfBounds struct on every success too, which the d=4 bisect
    // (see issue #103) attributed to the compose regression. .ok_or_else only
    // builds it on the never-taken out-of-bounds path.
    let block =
        structure
            .blocks
            .get(block_index)
            .ok_or_else(|| OperationError::BlockIndexOutOfBounds {
                tensor: "recoupling block",
                index: block_index,
                count: structure.blocks.len(),
            })?;
    match block {
        TreeTransformBlock::Multi { .. } => Ok(block),
        TreeTransformBlock::Single { .. } => Err(OperationError::BlockIndexOutOfBounds {
            tensor: "recoupling block",
            index: block_index,
            count: structure.blocks.len(),
        }),
    }
}

fn scale_inactive_destinations<A, D, C>(
    kernels: &mut A,
    structure: &TreeTransformStructure<C>,
    dst_data: &mut [D],
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + PartialEq + One,
    C: Copy,
{
    if beta == D::one() {
        return Ok(());
    }
    // Scaling the complete storage would also mutate padding not owned by any
    // block, so compile only the destination layouts with no active replay.
    for &layout_index in structure.inactive_destination_layouts() {
        let layout = structure.layouts.entry(layout_index);
        kernels.scale_strided(
            dst_data,
            structure.layouts.shape(layout),
            structure.layouts.strides(layout),
            layout.offset,
            beta,
        )?;
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
    use tenet_dense::DefaultDenseExecutor;

    type TestTensor = TensorMap<f64, 1, 0, Trivial, Vec<f64>>;

    fn fixture() -> (TestTensor, TestTensor, TreeTransformStructure<f64>) {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        let src = TensorMap::from_vec_with_structure(
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
        BlockSpec::with_key(
            BlockKey::sector_ids([sector]),
            vec![shape],
            vec![stride],
            offset,
        )
        .unwrap()
    }

    #[test]
    fn compile_rejects_inactive_destination_aliases() {
        let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
        let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
        assert_eq!(
            TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[],)
                .unwrap_err(),
            OperationError::OverlappingTransformDestination {
                first_dst_block: 0,
                second_dst_block: 1,
                offset: 0
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
            OperationError::OverlappingTransformDestination {
                first_dst_block: 0,
                second_dst_block: 1,
                offset: 0
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
            OperationError::OverlappingTransformDestination {
                first_dst_block: 0,
                second_dst_block: 1,
                offset: 0
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
            OperationError::OverlappingTransformDestination {
                first_dst_block: 0,
                second_dst_block: 0,
                offset: 0
            }
        );
    }

    #[test]
    fn compile_rejects_nonzero_stride_self_overlap() {
        let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
        let dst_structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::with_key(BlockKey::sector_ids([0]), vec![2, 2], vec![1, 1], 0).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            TreeTransformStructure::compile_structures(
                &dst_structure,
                &src_structure,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap_err(),
            OperationError::OverlappingTransformDestination {
                first_dst_block: 0,
                second_dst_block: 0,
                offset: 1,
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
        tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::new(dst_structure),
            &Arc::new(src_structure),
            &mut dst,
            &[3.0],
            1.0,
            0.5,
        )
        .unwrap();
        assert_eq!(dst, [5.0, 10.0, 15.0, 20.0]);
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
        tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::new(dst_structure),
            &Arc::new(src_structure),
            &mut dst,
            &[3.0],
            1.0,
            0.5,
        )
        .unwrap();
        assert_eq!(dst, [5.0, 99.0, 15.0]);
    }

    #[test]
    fn nan_beta_reaches_active_and_inactive_destinations() {
        let (mut dst, src, structure) = fixture();
        tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            f64::NAN,
        )
        .unwrap();
        assert!(dst.data().iter().all(|value| value.is_nan()));
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
            + profile.inactive_scale
            + profile.single_total
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
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    structure.validate_replay_structures(&dst_structure, &src_structure)?;
    validate_replay_storage_len(&dst_structure, dst.storage().len())?;
    validate_replay_storage_len(&src_structure, src.storage().len())?;

    scale_inactive_destinations(kernels, structure, dst.data_mut(), beta)?;

    let src_data = src.data();
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                kernels,
                workspace.zero_strides_mut(),
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst.data_mut(),
                src_data,
                alpha,
                beta,
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
                    &structure.layouts,
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                    &structure.recoupling_coefficients_dst_src,
                    structure.storage_conjugate(),
                    dst.data_mut(),
                    src_data,
                    alpha,
                    beta,
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
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    scale_inactive_destinations(kernels, structure, dst_data, beta)?;
    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                kernels,
                &mut workspace.zero_strides,
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
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
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.recoupling_coefficients_dst_src,
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
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

/// Replays a prepared structural-recoupling tree transform on host slices.
///
/// `threads` selects the replay parallelism (a property of the executing
/// backend, not of the cached structure): `<= 1` runs the existing serial
/// path unchanged; `> 1` runs Single applies, Multi pack columns and Multi
/// scatter columns as independent work items over up to `threads`
/// work-stealing workers, with the batched recoupling GEMM staying a single
/// serial call between the two parallel phases.
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
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    scale_inactive_destinations(kernels, structure, dst_data, beta)?;
    if threads > 1 {
        return tree_transform_blocks_with_batched_recoupling_parallel(
            kernels, dense, workspace, structure, dst_data, src_data, alpha, beta, threads, None,
        );
    }
    tree_transform_blocks_with_batched_recoupling(
        kernels, dense, workspace, structure, dst_data, src_data, alpha, beta, None,
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
    let total_start = std::time::Instant::now();

    let start = std::time::Instant::now();
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    profile.validate += start.elapsed();

    let start = std::time::Instant::now();
    scale_inactive_destinations(kernels, structure, dst_data, beta)?;
    profile.inactive_scale += start.elapsed();

    if threads > 1 {
        tree_transform_blocks_with_batched_recoupling_parallel(
            kernels,
            dense,
            workspace,
            structure,
            dst_data,
            src_data,
            alpha,
            beta,
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
            beta,
            Some(profile),
        )?;
    }

    profile.total += total_start.elapsed();
    Ok(())
}

/// Executes a validated tree-transform block list against a dense executor:
/// Single blocks apply directly through the strided kernel, and every Multi
/// block packs into one shared source/destination scratch pair so all the
/// recoupling GEMMs (`destination = source * U^T` per block) submit as a
/// single batched call — small transform groups then pay the dense executor's
/// per-call dispatch cost once per replay instead of once per block.
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
    beta: D,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let layouts = &structure.layouts;
    let recoupling_plan = structure.recoupling_plan();

    // All-Single structures (abelian recoupling is diagonal) skip the batch
    // machinery entirely: no pack scratch, no job list, no scatter pass.
    if recoupling_plan.is_empty() {
        for block in &structure.blocks {
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
                layouts.entry(dst_layout),
                layouts.entry(src_layout),
                structure.coefficient(coefficient),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                beta,
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

    // Size the shared pack scratch from the compile-time recoupling plan.
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    workspace.prepare_packed_buffers(
        recoupling_plan.source_len(),
        recoupling_plan.destination_len(),
        D::zero(),
    );
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.multi_workspace_prepare += start.elapsed();
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
    for block in &structure.blocks {
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
            layouts.entry(dst_layout),
            layouts.entry(src_layout),
            structure.coefficient(coefficient),
            structure.storage_conjugate(),
            dst_data,
            src_data,
            alpha,
            beta,
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
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        for src_index in 0..src_count {
            let layout = layouts.entry(src_layout_start + src_index);
            pack_layout_into_column(
                kernels,
                layouts,
                layout,
                src_data,
                workspace.packed.source_mut().as_mut_slice(),
                job.lhs_offset + src_index * element_count,
                structure.storage_conjugate(),
            )?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_blocks += 1;
            profile.packed_columns += src_count;
            profile.multi_pack += start.elapsed();
        }
    }

    // One batched recoupling GEMM across all Multi blocks (TensorKit's
    // `_add_transform_multi!` `mul!` step, grouped).
    if !recoupling_plan.jobs().is_empty() {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let (source, destination) = workspace.packed.source_and_destination_mut();
        recoupling_gemm_batch(
            dense,
            destination.as_mut_slice(),
            source.as_slice(),
            &workspace.coefficient_scratch,
            recoupling_plan.jobs(),
            recoupling_plan.runs(),
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_scalar_recoupling += elapsed;
            profile.multi_matmul_total += elapsed;
        }
    }

    // Scatter each Multi block's destination columns back out.
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let mut scattered_columns = 0usize;
    for (block_index, job) in recoupling_plan.entries() {
        if let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            element_count,
            ..
        } = *recoupling_multi_block(structure, block_index)?
        {
            debug_assert_eq!(job.rows, element_count);
            debug_assert_eq!(job.cols, dst_count);
            for dst_index in 0..dst_count {
                let layout = layouts.entry(dst_layout_start + dst_index);
                scatter_column_into_layout(
                    kernels,
                    &mut workspace.zero_strides,
                    layouts,
                    layout,
                    workspace.packed.destination().as_slice(),
                    job.dst_offset + dst_index * element_count,
                    dst_data,
                    alpha,
                    beta,
                )?;
            }
            scattered_columns += dst_count;
        }
    }
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.scattered_columns += scattered_columns;
        profile.multi_scatter += start.elapsed();
    }
    Ok(())
}

/// Inclusive index range `[lo, hi]` touched by a layout's strided walk over
/// `shape` from `offset` (negative strides walk downward).
fn layout_index_range(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
) -> (isize, isize) {
    let mut lo = layout.offset;
    let mut hi = layout.offset;
    for (&extent, &stride) in layouts
        .shape(layout)
        .iter()
        .zip(layouts.strides(layout).iter())
    {
        let span = (extent as isize - 1) * stride;
        if span < 0 {
            lo += span;
        } else {
            hi += span;
        }
    }
    (lo, hi)
}

/// Splits `data` into one disjoint `&mut` region per item (items sorted by
/// `lo`, each `(payload, lo, hi)` an inclusive touched range); each result
/// carries the region and its absolute start index so layout offsets can be
/// rebased. Returns `None` when regions overlap or run out of bounds — valid
/// packed transform structures never do (compile rejects duplicate
/// destination blocks), so `None` only guards degenerate stride patterns and
/// sends the caller down the serial path.
#[allow(clippy::type_complexity)]
fn split_regions<'a, T, P: Copy>(
    data: &'a mut [T],
    items: &[(P, isize, isize)],
) -> Option<Vec<(P, &'a mut [T], isize)>> {
    let mut regions = Vec::with_capacity(items.len());
    let mut rest = data;
    // Absolute index where `rest` begins.
    let mut consumed = 0isize;
    for &(payload, lo, hi) in items {
        if lo < consumed || hi < lo {
            return None;
        }
        let skip = (lo - consumed) as usize;
        let len = (hi - lo + 1) as usize;
        if skip.checked_add(len)? > rest.len() {
            return None;
        }
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(skip);
        let (region, tail) = tail.split_at_mut(len);
        regions.push((payload, region, lo));
        rest = tail;
        consumed = hi + 1;
    }
    Some(regions)
}

/// Threaded variant of [`tree_transform_blocks_with_batched_recoupling`]
/// (TensorKit `_add_abelian_kernel_threaded!` / `_add_general_kernel_threaded!`
/// precedent, indexmanipulations.jl:520-738):
///
/// - Phase A packs every Multi source column and applies every Single block
///   in parallel across tree pairs; phase B scatters every Multi destination
///   column in parallel. Work items are independent because the compile step
///   rejects duplicate destination blocks
///   (`OperationError::DuplicateTransformDestination`) and pack columns are
///   disjoint scratch ranges by construction; the workspace forbids `unsafe`,
///   so disjointness is realized structurally by pre-splitting the buffers
///   into per-item `&mut` regions (`split_at_mut`) and rebasing offsets,
///   instead of TensorKit-style shared writes.
/// - The batched recoupling GEMM stays ONE serial grouped call between the
///   two parallel phases — the dense executor owns its own parallelism and
///   no nesting arises because the batch submits outside both regions.
///
/// Scheduling: rayon parallel iterators on the global pool (the same pool
/// strided-kernel's threaded kernels use), with `with_min_len` bounding the
/// split count to the configured `threads` — the moral equivalent of
/// TensorKit's `min(ntasks, nblocks)` spawned workers.
///
/// Per-task state is one cloned kernel adapter (a ZST for the strided
/// adapter) and one `Vec::new()` zero-strides scratch (no allocation until a
/// kernel actually needs it); the pack/destination scratch itself stays the
/// reused workspace buffer, so the `ScratchStorage` reuse contract is
/// untouched — a deliberate deviation from TensorKit, which allocates pack
/// buffers inside every spawned task.
///
/// Profiling attribution is coarser than the serial path: phase A lands in
/// `multi_pack` (Singles included) and phase B in `multi_scatter`; per-item
/// clocks across workers would measure contention, not work.
#[allow(clippy::too_many_arguments)]
fn tree_transform_blocks_with_batched_recoupling_parallel<A, E, D, C>(
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    use rayon::prelude::*;

    let layouts = &structure.layouts;
    let recoupling_plan = structure.recoupling_plan();

    // Build phase (serial): size the pack scratch from the compile-time plan,
    // ensure converted coefficients, and collect the parallel work items.
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    workspace.prepare_packed_buffers(
        recoupling_plan.source_len(),
        recoupling_plan.destination_len(),
        D::zero(),
    );
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.multi_workspace_prepare += start.elapsed();
    }

    // (dst_layout, src_layout, coefficient index) per Single block.
    let mut singles: Vec<(usize, usize, usize)> = Vec::new();
    // (source layout, packed source offset, column length) per Multi pack column.
    let mut pack_columns: Vec<(usize, usize, usize)> = Vec::new();
    // (dst layout, packed destination offset) per Multi scatter column.
    let mut scatter_columns: Vec<(usize, usize)> = Vec::new();

    for block in &structure.blocks {
        let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = *block
        else {
            continue;
        };
        singles.push((dst_layout, src_layout, coefficient));
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
        for src_index in 0..src_count {
            pack_columns.push((
                src_layout_start + src_index,
                job.lhs_offset + src_index * element_count,
                element_count,
            ));
        }
        for dst_index in 0..dst_count {
            scatter_columns.push((
                dst_layout_start + dst_index,
                job.dst_offset + dst_index * element_count,
            ));
        }
    }
    let single_count = singles.len();
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

    // At most `threads` parallel chunks per phase (TensorKit's
    // `min(ntasks, nblocks)` worker bound) on rayon's global pool.
    let min_len = |items: usize| items.div_ceil(threads).max(1);
    let storage_conjugate = structure.storage_conjugate();

    // Phase A: pack columns and Single applies in parallel.
    {
        let start = profile.as_ref().map(|_| std::time::Instant::now());

        let mut source_items: Vec<(usize, isize, isize)> = Vec::with_capacity(pack_columns.len());
        for &(layout, offset, len) in &pack_columns {
            if len == 0 {
                return Err(OperationError::ElementCountOverflow);
            }
            let hi_offset = offset
                .checked_add(len)
                .and_then(|end| end.checked_sub(1))
                .ok_or(OperationError::ElementCountOverflow)?;
            source_items.push((
                layout,
                offset_to_isize(offset)?,
                offset_to_isize(hi_offset)?,
            ));
        }
        source_items.sort_unstable_by_key(|&(_, lo, _)| lo);
        let column_regions =
            split_regions(workspace.packed.source_mut().as_mut_slice(), &source_items).ok_or_else(
                || OperationError::StridedKernel {
                    message: "recoupling source scratch ranges are not disjoint".to_string(),
                },
            )?;
        let pack_chunk = min_len(column_regions.len());
        column_regions
            .into_par_iter()
            .with_min_len(pack_chunk)
            .try_for_each_init(
                || kernels.clone(),
                |kernels, (layout, column, _)| {
                    pack_layout_into_column(
                        kernels,
                        layouts,
                        layouts.entry(layout),
                        src_data,
                        column,
                        0,
                        storage_conjugate,
                    )
                },
            )?;

        // Single blocks write disjoint destination subblocks: split dst_data
        // into per-item regions and rebase the destination offsets.
        let mut items: Vec<((usize, usize, usize), isize, isize)> = singles
            .iter()
            .map(|&item| {
                let (lo, hi) = layout_index_range(layouts, layouts.entry(item.0));
                (item, lo, hi)
            })
            .collect();
        items.sort_unstable_by_key(|&(_, lo, _)| lo);
        match split_regions(dst_data, &items) {
            Some(regions) => {
                let single_chunk = min_len(regions.len());
                regions
                    .into_par_iter()
                    .with_min_len(single_chunk)
                    .try_for_each_init(
                        || (kernels.clone(), Vec::new()),
                        |(kernels, zero_strides),
                         ((dst_layout, src_layout, coefficient), region, region_start)| {
                            let dst_layout = layouts.entry(dst_layout);
                            let src_layout = layouts.entry(src_layout);
                            let scale =
                                alpha.scale_by_coefficient(structure.coefficient(coefficient));
                            kernels.add_strided(
                                zero_strides,
                                region,
                                src_data,
                                layouts.shape(dst_layout),
                                layouts.strides(dst_layout),
                                layouts.strides(src_layout),
                                dst_layout.offset - region_start,
                                src_layout.offset,
                                storage_conjugate,
                                scale,
                                beta,
                            )
                        },
                    )?;
            }
            // Degenerate (overlapping) regions: fall back to the serial
            // Single loop; valid packed structures never reach this.
            None => {
                let mut zero_strides = Vec::new();
                for &(dst_layout, src_layout, coefficient) in &singles {
                    tree_transform_single_with_strided_kernel(
                        kernels,
                        &mut zero_strides,
                        layouts,
                        layouts.entry(dst_layout),
                        layouts.entry(src_layout),
                        structure.coefficient(coefficient),
                        storage_conjugate,
                        dst_data,
                        src_data,
                        alpha,
                        beta,
                    )?;
                }
            }
        }

        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.packed_columns += pack_columns.len();
            profile.multi_pack += start.elapsed();
        }
    }

    // One batched recoupling GEMM across all Multi blocks, outside both
    // parallel regions (the dense executor owns its parallelism).
    if !recoupling_plan.jobs().is_empty() {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let (source, destination) = workspace.packed.source_and_destination_mut();
        recoupling_gemm_batch(
            dense,
            destination.as_mut_slice(),
            source.as_slice(),
            &workspace.coefficient_scratch,
            recoupling_plan.jobs(),
            recoupling_plan.runs(),
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_scalar_recoupling += elapsed;
            profile.multi_matmul_total += elapsed;
        }
    }

    // Phase B: scatter destination columns in parallel (disjoint destination
    // subblocks, same compile guarantee as the Singles).
    {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let packed_destination = workspace.packed.destination().as_slice();
        let mut items: Vec<((usize, usize), isize, isize)> = scatter_columns
            .iter()
            .map(|&item| {
                let (lo, hi) = layout_index_range(layouts, layouts.entry(item.0));
                (item, lo, hi)
            })
            .collect();
        items.sort_unstable_by_key(|&(_, lo, _)| lo);
        match split_regions(dst_data, &items) {
            Some(regions) => {
                let scatter_chunk = min_len(regions.len());
                regions
                    .into_par_iter()
                    .with_min_len(scatter_chunk)
                    .try_for_each_init(
                        || kernels.clone(),
                        |kernels, ((layout, packed_offset), region, region_start)| {
                            let layout = layouts.entry(layout);
                            kernels.axpby_strided(
                                region,
                                packed_destination,
                                layouts.shape(layout),
                                layouts.strides(layout),
                                layouts.packed_strides(layout),
                                layout.offset - region_start,
                                offset_to_isize(packed_offset)?,
                                alpha,
                                beta,
                            )
                        },
                    )?;
            }
            None => {
                let mut zero_strides = Vec::new();
                for &(layout, packed_offset) in &scatter_columns {
                    scatter_column_into_layout(
                        kernels,
                        &mut zero_strides,
                        layouts,
                        layouts.entry(layout),
                        packed_destination,
                        packed_offset,
                        dst_data,
                        alpha,
                        beta,
                    )?;
                }
            }
        }
        if let (Some(profile), Some(start)) = (profile, start) {
            profile.scattered_columns += scatter_columns.len();
            profile.multi_scatter += start.elapsed();
        }
    }
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

fn validate_replay_storage_len(
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

#[allow(clippy::too_many_arguments)]
fn tree_transform_single_with_strided_kernel<A, D, C>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    dst_layout: &TreeTransformLayout,
    src_layout: &TreeTransformLayout,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let shape = layouts.shape(dst_layout);
    let scale = alpha.scale_by_coefficient(coefficient);
    kernels.add_strided(
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
    )
}

/// Applies every Multi block's recoupling matrix in one batched GEMM over
/// shared flat scratch buffers: per job, the column-major
/// (element_count x dst_count) destination block receives `source_block *
/// U^T`, with `recoupling_coefficients_dst_src` (row-major `U[dst, src]`)
/// reinterpreted as the column-major (src_count x dst_count) matrix `U^T`.
/// This is TensorKit's `_add_transform_multi!` `mul!` step submitted as one
/// grouped call; the naive per-element loop in the kernel adapter remains
/// only for adapters without a dense executor. Job offsets are constructed by
/// the replay against scratch sized to their exact totals, matching the
/// plan-compile validation contract of the trusted views.
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
    beta: D,
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
        beta,
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
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + One + RecouplingCoefficientAction<C>,
    C: Copy,
    SourceScratch: HostWritableStorage<D>,
    DestinationScratch: HostWritableStorage<D>,
{
    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            kernels,
            layouts,
            layout,
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
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            kernels,
            zero_strides,
            layouts,
            layout,
            scratch.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

fn pack_layout_into_column<A, T>(
    kernels: &mut A,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One,
{
    let shape = layouts.shape(layout);
    let packed_offset = offset_to_isize(packed_offset)?;
    kernels.copy_scale_strided(
        packed,
        src_data,
        shape,
        layouts.packed_strides(layout),
        layouts.strides(layout),
        packed_offset,
        layout.offset,
        source_conjugate,
        T::one(),
    )
}

#[allow(clippy::too_many_arguments)]
fn scatter_column_into_layout<A, T>(
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    let shape = layouts.shape(layout);
    zero_strides.clear();
    kernels.axpby_strided(
        dst_data,
        packed,
        shape,
        layouts.strides(layout),
        layouts.packed_strides(layout),
        layout.offset,
        offset_to_isize(packed_offset)?,
        alpha,
        beta,
    )
}
