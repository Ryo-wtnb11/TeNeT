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
