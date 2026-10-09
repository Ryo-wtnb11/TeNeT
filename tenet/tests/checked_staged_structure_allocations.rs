//! #2050: a pre-commit plan compiled against a staged checked destination
//! lends the live canonical structure on a complete-cache hit instead of
//! wrapping a cloned preview in a fresh `Arc`. Warm allocation calls of the
//! public checked permute and contraction (non-identity, so a destination is
//! staged) are pinned as upper bounds. #2129: the contraction's core plan
//! compiles from coupled regions, so its warm calls are bounded per output
//! block.

use std::sync::Arc;

use tenet::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[path = "../../tests/support/toy_rules.rs"]
mod toy_rules;

use toy_rules::{GenericLabel, GenericToy};

type Tensor = TensorMap<GenericToy, f64>;

fn tensor(runtime: &Runtime, leg: &GradedSpace<GenericToy>, rank: usize) -> Tensor {
    let legs = vec![leg; rank / 2];
    TensorMap::from_subblock_fn(runtime, legs.clone(), legs, |trees, indices| {
        trees.codomain_vertices().first().map_or(0, |v| v.get()) as f64
            + indices.iter().sum::<usize>() as f64
    })
    .unwrap()
}

/// Warm allocation calls and bytes of one public checked call.
struct Warm {
    permute: (u64, u64),
    contract: (u64, u64),
    /// Output blocks `P` of the contraction.
    out_blocks: usize,
    /// Blocks `P_c` and coupled sectors `S` of its core product.
    core_blocks: usize,
    core_sectors: usize,
    /// Recoupling groups `G`: composed-coefficient lookups of one warm call.
    groups: u64,
}

fn coefficient_lookups() -> u64 {
    tenet::cache::stats()
        .into_iter()
        .find(|info| info.kind() == tenet::cache::StructureCacheKind::TreeTransformCoefficients)
        .map(|info| info.hits() + info.misses())
        .unwrap()
}

fn warm(rank: usize) -> Warm {
    let _serial = counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(GenericToy),
        [(GenericLabel::Vacuum, 1), (GenericLabel::X, 2)],
    )
    .unwrap();
    let lhs = tensor(&runtime, &leg, rank);
    let rhs = tensor(&runtime, &leg, rank);
    let half = rank / 2;
    let reversed: Vec<usize> = (0..rank).rev().collect();
    let (codomain, domain) = reversed.split_at(half);

    let open = 2 * rank - 2;
    let codomain_open: Vec<usize> = (0..open / 2).rev().collect();
    let domain_open: Vec<usize> = (open / 2..open).collect();
    let (lhs_axes, rhs_axes) = ([rank - 1], [0]);
    let spec = ContractSpec {
        lhs: &lhs_axes,
        rhs: &rhs_axes,
        codomain: &codomain_open,
        domain: &domain_open,
    };

    drop(lhs.permute(codomain, domain).unwrap());
    let out_blocks = lhs.contract(&rhs, &spec).unwrap().subblock_count();
    let (out, permute) = counting_alloc::measure(|| lhs.permute(codomain, domain).unwrap());
    drop(out);
    let lookups = coefficient_lookups();
    let (out, contract) = counting_alloc::measure(|| lhs.contract(&rhs, &spec).unwrap());
    let groups = coefficient_lookups() - lookups;
    drop(out);
    let lhs_core: Vec<usize> = (0..rank - 1).collect();
    let rhs_core: Vec<usize> = (1..rank).collect();
    let core = lhs
        .permute(&lhs_core, &[rank - 1])
        .unwrap()
        .compose(&rhs.permute(&[0], &rhs_core).unwrap())
        .unwrap();
    Warm {
        permute: (permute.calls, permute.bytes),
        contract: (contract.calls, contract.bytes),
        out_blocks,
        core_blocks: core.subblock_count(),
        core_sectors: core.blocks().unwrap().len(),
        groups,
    }
}

/// Permute: head calls per rank; base was one `Arc` (1 call, 32 bytes) per
/// staged destination higher (#2050). Contract (#2129 leaf A): the core plan
/// compiles from coupled regions, so warm calls stay below a few per output
/// block; the per-subblock core builder cost about ten per block (base
/// 114 / 2,552 / 389,907 calls at rank 2 / 4 / 6). The remaining O(P) is the
/// uncommitted staged and core previews that #2063 removes, which then
/// bounds calls by `c·(S + G)`. Bounds, not equalities: platform allocators
/// and the toy provider may shift incidental sizes.
#[test]
fn warm_staged_checked_plans_lend_the_canonical_structure() {
    let permute_bound = [(2usize, 7u64), (4, 12), (6, 51)];
    for (rank, permute_calls) in permute_bound {
        let w = warm(rank);
        let what = format!(
            "rank {rank}: permute {:?} contract {:?} P={} P_c={} S={} G={}",
            w.permute, w.contract, w.out_blocks, w.core_blocks, w.core_sectors, w.groups
        );
        assert!(w.permute.0 <= permute_calls, "{what}");
        assert!(w.contract.0 <= 3 * w.out_blocks as u64 + 64, "{what}");
    }
}
