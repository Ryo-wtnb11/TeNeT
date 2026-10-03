use super::*;
use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
use std::time::Duration;
use tenet_core::{BlockKey, BlockSpec, TensorMapSpace, Trivial};
use tenet_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite,
};

mod compile_validation;
mod overwrite;
mod profile;
mod threaded;

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
    let structure =
        TreeTransformStructure::compile(&dst, &src, &[TreeTransformBlockSpec::single(0, 0, 2.0)])
            .unwrap();
    (dst, src, structure)
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

#[test]
fn tree_transform_replay_dispatches_through_kernel_adapter() {
    use crate::kernel_adapter::observed::{KernelCall, ObservedKernels};

    // Source columns sit in a different block order from the spec, and the
    // destination blocks are reversed, so pack and scatter must follow the
    // spec's block indices rather than storage order.
    let src =
        Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2], vec![2]]).unwrap());
    let dst = Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
    let structure = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![1, 0],
            vec![0, 2, 1],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let mut dst_data = [0.0; 4];
    let mut kernels = ObservedKernels::recording();

    tree_transform_structure_with_strided_kernel_raw(
        &mut kernels,
        &mut TreeTransformWorkspace::default(),
        &structure,
        &dst,
        &src,
        &mut dst_data,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst_data, [7020.0, 9240.0, 3510.0, 4620.0]);
    assert_eq!(kernels.copy_offsets().len(), 3);
    assert_eq!(kernels.count(KernelCall::Recoupling), 1);
    assert_eq!(kernels.count(KernelCall::Axpby), 2);
    assert_eq!(kernels.count(KernelCall::Add), 0);
    assert!(kernels.scale_offsets().is_empty());
}
