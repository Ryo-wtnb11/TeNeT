use std::hint::black_box;
use std::time::{Duration, Instant};

use tenet_core::{BlockKey, BlockStructure, TensorMap, TensorMapSpace};
use tenet_tensors::{
    tensoradd_execute_with, tree_transform_execute_with, HostAllocator, HostTensorOperations,
    OutputAxisOrder, TensorAddStructure, TreeTransformBlockSpec, TreeTransformKeyBlockSpec,
    TreeTransformStructure, TreeTransformWorkspace,
};

const BLOCK_COUNTS: &[usize] = &[1, 8, 64, 512, 4096];

fn main() {
    println!("# current TensorAddStructure path");
    println!("block_count,key_order,compile_ns,compiled_replay_ns");
    for &block_count in BLOCK_COUNTS {
        run_case(block_count, KeyOrder::Ordered);
        run_case(block_count, KeyOrder::Reversed);
    }
    println!();
    println!("# lookup strategy candidates");
    println!("block_count,key_order,sorted_merge_ns,direct_id_ns");
    for &block_count in BLOCK_COUNTS {
        run_pairing_case(block_count, KeyOrder::Ordered);
        run_pairing_case(block_count, KeyOrder::Reversed);
    }
    println!();
    println!("# current TreeTransform single-tree path");
    println!("block_count,compile_ns,compiled_replay_ns");
    for &block_count in BLOCK_COUNTS {
        run_tree_single_case(block_count);
    }
    println!();
    println!("# current TreeTransform multi-tree path");
    println!("group_count,compile_ns,compiled_replay_ns");
    for &group_count in BLOCK_COUNTS {
        run_tree_multi_case(group_count);
    }
    println!();
    println!("# keyed TreeTransform multi-tree path");
    println!("group_count,key_order,compile_ns,compiled_replay_ns");
    for &group_count in BLOCK_COUNTS {
        run_tree_multi_keyed_case(group_count, KeyOrder::Ordered);
        run_tree_multi_keyed_case(group_count, KeyOrder::Reversed);
    }
}

#[derive(Clone, Copy, Debug)]
enum KeyOrder {
    Ordered,
    Reversed,
}

impl KeyOrder {
    fn name(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::Reversed => "reversed",
        }
    }
}

fn run_case(block_count: usize, key_order: KeyOrder) {
    let src = tensor(block_count, KeyOrder::Ordered);
    let mut dst = tensor(block_count, key_order);

    let compile_iters = iterations(block_count, 200_000);
    let compile_elapsed = elapsed_per_iter(compile_iters, || {
        let structure =
            TensorAddStructure::compile(&dst, &src, OutputAxisOrder::identity()).unwrap();
        black_box(structure.terms().len());
    });

    let structure = TensorAddStructure::compile(&dst, &src, OutputAxisOrder::identity()).unwrap();
    let replay_iters = iterations(block_count, 50_000);
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    let replay_elapsed = elapsed_per_iter(replay_iters, || {
        tensoradd_execute_with(
            &mut backend,
            &mut allocator,
            &structure,
            &mut dst,
            &src,
            1.0_f64,
            0.0_f64,
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    println!(
        "{},{},{},{}",
        block_count,
        key_order.name(),
        compile_elapsed.as_nanos(),
        replay_elapsed.as_nanos()
    );
}

fn run_pairing_case(block_count: usize, key_order: KeyOrder) {
    let src = keys(block_count, KeyOrder::Ordered);
    let dst = keys(block_count, key_order);
    let iters = iterations(block_count, 200_000);

    let sorted_elapsed = elapsed_per_iter(iters, || {
        black_box(pair_sorted_merge(&dst, &src));
    });
    let direct_elapsed = elapsed_per_iter(iters, || {
        black_box(pair_direct_id(&dst, &src));
    });

    println!(
        "{},{},{},{}",
        block_count,
        key_order.name(),
        sorted_elapsed.as_nanos(),
        direct_elapsed.as_nanos()
    );
}

fn run_tree_single_case(block_count: usize) {
    let src = tensor(block_count, KeyOrder::Ordered);
    let mut dst = tensor(block_count, KeyOrder::Ordered);
    let specs = (0..block_count)
        .map(|block| TreeTransformBlockSpec::single(block, block, 2.0_f64))
        .collect::<Vec<_>>();

    let compile_iters = iterations(block_count, 100_000);
    let compile_elapsed = elapsed_per_iter(compile_iters, || {
        let structure = TreeTransformStructure::compile(&dst, &src, &specs).unwrap();
        black_box(structure.block_count());
    });

    let structure = TreeTransformStructure::compile(&dst, &src, &specs).unwrap();
    let replay_iters = iterations(block_count, 30_000);
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();
    let replay_elapsed = elapsed_per_iter(replay_iters, || {
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            1.0_f64,
            0.0_f64,
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    println!(
        "{},{},{}",
        block_count,
        compile_elapsed.as_nanos(),
        replay_elapsed.as_nanos()
    );
}

fn run_tree_multi_case(group_count: usize) {
    let block_count = group_count * 2;
    let src = tensor(block_count, KeyOrder::Ordered);
    let mut dst = tensor(block_count, KeyOrder::Ordered);
    let specs = (0..group_count)
        .map(|group| {
            let first = group * 2;
            TreeTransformBlockSpec::multi(
                vec![first, first + 1],
                vec![first, first + 1],
                vec![1.0_f64, 2.0, 3.0, 4.0],
            )
        })
        .collect::<Vec<_>>();

    let compile_iters = iterations(block_count, 100_000);
    let compile_elapsed = elapsed_per_iter(compile_iters, || {
        let structure = TreeTransformStructure::compile(&dst, &src, &specs).unwrap();
        black_box(structure.block_count());
    });

    let structure = TreeTransformStructure::compile(&dst, &src, &specs).unwrap();
    let replay_iters = iterations(block_count, 30_000);
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();
    let replay_elapsed = elapsed_per_iter(replay_iters, || {
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            1.0_f64,
            0.0_f64,
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    println!(
        "{},{},{}",
        group_count,
        compile_elapsed.as_nanos(),
        replay_elapsed.as_nanos()
    );
}

fn run_tree_multi_keyed_case(group_count: usize, key_order: KeyOrder) {
    let block_count = group_count * 2;
    let src = tensor(block_count, KeyOrder::Ordered);
    let mut dst = tensor(block_count, key_order);
    let specs = (0..group_count)
        .map(|group| {
            let first = group * 2;
            TreeTransformKeyBlockSpec::multi(
                [
                    BlockKey::sector_ids([first]),
                    BlockKey::sector_ids([first + 1]),
                ],
                [
                    BlockKey::sector_ids([first]),
                    BlockKey::sector_ids([first + 1]),
                ],
                vec![1.0_f64, 2.0, 3.0, 4.0],
            )
        })
        .collect::<Vec<_>>();

    let compile_iters = iterations(block_count, 100_000);
    let compile_elapsed = elapsed_per_iter(compile_iters, || {
        let structure = TreeTransformStructure::compile_keyed(&dst, &src, &specs).unwrap();
        black_box(structure.block_count());
    });

    let structure = TreeTransformStructure::compile_keyed(&dst, &src, &specs).unwrap();
    let replay_iters = iterations(block_count, 30_000);
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();
    let replay_elapsed = elapsed_per_iter(replay_iters, || {
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            1.0_f64,
            0.0_f64,
        )
        .unwrap();
        black_box(dst.data()[0]);
    });

    println!(
        "{},{},{},{}",
        group_count,
        key_order.name(),
        compile_elapsed.as_nanos(),
        replay_elapsed.as_nanos()
    );
}

fn iterations(block_count: usize, budget: usize) -> usize {
    (budget / block_count.max(1)).clamp(10, 20_000)
}

fn elapsed_per_iter(mut iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..iters.min(100) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    if iters == 0 {
        iters = 1;
    }
    Duration::from_nanos((elapsed.as_nanos() / iters as u128) as u64)
}

fn tensor(block_count: usize, key_order: KeyOrder) -> TensorMap<f64, 2, 0> {
    let space = TensorMapSpace::<2, 0>::from_dims([block_count, 1], []).unwrap();
    let structure = packed_fixture_structure(
        2,
        keys(block_count, key_order)
            .into_iter()
            .map(|key| (key, vec![1, 1])),
    )
    .unwrap();
    TensorMap::from_vec_with_structure(vec![0.0_f64; block_count], space, structure).unwrap()
}

fn keys(block_count: usize, key_order: KeyOrder) -> Vec<BlockKey> {
    let ids: Box<dyn Iterator<Item = usize>> = match key_order {
        KeyOrder::Ordered => Box::new(0..block_count),
        KeyOrder::Reversed => Box::new((0..block_count).rev()),
    };
    ids.map(|key| BlockKey::sector_ids([key])).collect()
}

fn pair_sorted_merge(dst: &[BlockKey], src: &[BlockKey]) -> usize {
    let mut dst_sorted = dst
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<Vec<_>>();
    let mut src_sorted = src
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<Vec<_>>();
    dst_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    src_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let mut checksum = 0usize;
    let mut dst_pos = 0usize;
    let mut src_pos = 0usize;
    while dst_pos < dst_sorted.len() && src_pos < src_sorted.len() {
        match dst_sorted[dst_pos].0.cmp(&src_sorted[src_pos].0) {
            std::cmp::Ordering::Less => dst_pos += 1,
            std::cmp::Ordering::Greater => src_pos += 1,
            std::cmp::Ordering::Equal => {
                checksum ^= dst_sorted[dst_pos].1.wrapping_mul(31) ^ src_sorted[src_pos].1;
                dst_pos += 1;
                src_pos += 1;
            }
        }
    }
    checksum
}

fn pair_direct_id(dst: &[BlockKey], src: &[BlockKey]) -> usize {
    let max_id = src
        .iter()
        .chain(dst)
        .map(single_sector_id)
        .max()
        .unwrap_or(0);
    let mut src_by_id = vec![usize::MAX; max_id + 1];
    for (src_index, src_key) in src.iter().enumerate() {
        src_by_id[single_sector_id(src_key)] = src_index;
    }
    let mut checksum = 0usize;
    for (dst_index, dst_key) in dst.iter().enumerate() {
        checksum ^= dst_index.wrapping_mul(31) ^ src_by_id[single_sector_id(dst_key)];
    }
    checksum
}

fn single_sector_id(key: &BlockKey) -> usize {
    match key {
        BlockKey::FusionTree(tree) if tree.uncoupled().len() == 1 => tree.uncoupled()[0].id(),
        BlockKey::Dense => 0,
        _ => panic!("benchmark keys must be single-sector keys"),
    }
}

/// Fixture layout: subblocks packed contiguously in key order (not a product
/// layout; exercises the arbitrary-strided-view contract).
fn packed_fixture_structure<I, K>(
    rank: usize,
    blocks: I,
) -> Result<BlockStructure, tenet_core::CoreError>
where
    I: IntoIterator<Item = (K, Vec<usize>)>,
    K: Into<tenet_core::BlockKey>,
{
    let mut keys = Vec::new();
    let mut shapes = Vec::new();
    for (key, shape) in blocks {
        keys.push(key.into());
        shapes.push(shape);
    }
    BlockStructure::from_parts(
        tenet_core::SectorStructure::from_keys(rank, keys)?,
        tenet_core::DegeneracyStructure::packed_column_major(rank, shapes)?,
    )
}
