//! #2050: a pre-commit plan compiled against a staged checked destination
//! lends the live canonical structure on a complete-cache hit instead of
//! wrapping a cloned preview in a fresh `Arc`. Warm allocation calls of the
//! public checked permute and contraction (non-identity, so a destination is
//! staged) are pinned as upper bounds.

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

/// Warm (calls, bytes) of the public checked permute and contract.
fn warm(rank: usize) -> ((u64, u64), (u64, u64)) {
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
    drop(lhs.contract(&rhs, &spec).unwrap());
    let (out, permute) = counting_alloc::measure(|| lhs.permute(codomain, domain).unwrap());
    drop(out);
    let (out, contract) = counting_alloc::measure(|| lhs.contract(&rhs, &spec).unwrap());
    drop(out);
    (
        (permute.calls, permute.bytes),
        (contract.calls, contract.bytes),
    )
}

/// Head (calls, bytes) per rank; base was one `Arc` (1 call, 32 bytes) per
/// staged destination higher. Bounds, not equalities: platform allocators and
/// the toy provider may shift incidental sizes.
#[test]
fn warm_staged_checked_plans_lend_the_canonical_structure() {
    let permute_bound = [(2usize, 7u64), (4, 12), (6, 51)];
    let contract_bound = [(2usize, 114u64), (4, 2552), (6, 389_907)];
    for ((rank, permute_calls), (_, contract_calls)) in
        permute_bound.into_iter().zip(contract_bound)
    {
        let (permute, contract) = warm(rank);
        assert!(
            permute.0 <= permute_calls,
            "rank {rank} permute {permute:?}"
        );
        assert!(
            contract.0 <= contract_calls,
            "rank {rank} contract {contract:?}"
        );
    }
}
