//! Measurement only (#1962): allocation calls and bytes of the complete
//! public checked Generic transform (cold: every structure cache cleared, so
//! destination layout and plan are compiled, then executed) across sharing
//! regimes. Provider calls and coefficient arithmetic of the plan build are
//! measured by the unit test `checked_composer_measurement`. Run with
//! `cargo test --release -p tenet-tensors --features racah-generated --test
//! checked_generic_composer_measurement -- --ignored --nocapture --test-threads=1`.
//! Wall-clock values are observations, never a gate.
#![cfg(feature = "racah-generated")]

use std::sync::Arc;

use tenet_core::{
    clear_structure_caches, FusionProductSpace, FusionTreeHomSpace, SUNFusionRule, SectorId,
    SectorLeg,
};
use tenet_tensors::{
    tree_transform_dyn_owned_checked_generic_input_in_context, BoundDynamicFusionMapSpace,
    CheckedTreeTransformInput, RuleIdentity, TreeTransformExecutionContext, TreeTransformOperation,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn homspace(legs: &[Vec<SectorId>], codomain_rank: usize) -> FusionTreeHomSpace {
    let leg =
        |sectors: &Vec<SectorId>| SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new(legs[..codomain_rank].iter().map(leg)),
        FusionProductSpace::new(legs[codomain_rank..].iter().map(leg)),
    )
}

/// The planar transpose rotating the linearized legs (codomain, then
/// reversed domain) by `shift` into a `split`-leg codomain.
fn rotation(
    codomain_rank: usize,
    domain_rank: usize,
    shift: usize,
    split: usize,
) -> TreeTransformOperation {
    let total = codomain_rank + domain_rank;
    let linear = (0..codomain_rank)
        .chain((codomain_rank..total).rev())
        .collect::<Vec<_>>();
    let rotated = (0..total)
        .map(|index| linear[(index + shift) % total])
        .collect::<Vec<_>>();
    TreeTransformOperation::transpose(
        rotated[..split].to_vec(),
        rotated[split..].iter().rev().copied().collect::<Vec<_>>(),
    )
}

fn median(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn public_cold(
    plain: &Arc<SUNFusionRule>,
    space: &FusionTreeHomSpace,
    operation: &TreeTransformOperation,
) -> (counting_alloc::Allocs, u128, usize) {
    clear_structure_caches();
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(plain),
        space.clone(),
    )
    .unwrap();
    let data = vec![1.0f64; source.space().required_len().unwrap()];
    let blocks = source.space().structure().block_count();
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    let started = std::time::Instant::now();
    let (output, allocs) = counting_alloc::measure(|| {
        tree_transform_dyn_owned_checked_generic_input_in_context(
            &mut context,
            operation.clone(),
            CheckedTreeTransformInput::direct(&source, &data),
            1.0,
        )
        .unwrap()
    });
    let ns = started.elapsed().as_nanos();
    std::hint::black_box(output);
    (allocs, ns, blocks)
}

fn measure_case(
    regime: &str,
    case: &str,
    legs: &[Vec<SectorId>],
    codomain_rank: usize,
    operation: TreeTransformOperation,
) {
    let space = homspace(legs, codomain_rank);
    let plain = Arc::new(SUNFusionRule::new(3).unwrap());
    // Warm racah's symbol caches; TeNeT's caches are cleared per sample.
    public_cold(&plain, &space, &operation);
    let samples = (0..3)
        .map(|_| public_cold(&plain, &space, &operation))
        .collect::<Vec<_>>();
    let (allocs, _, blocks) = samples[0];
    assert!(samples.iter().all(|sample| sample.0.calls == allocs.calls));
    let ns = median(samples.iter().map(|sample| sample.1).collect());
    println!(
        "regime={regime} case={case} blocks={blocks} public_cold_alloc_calls={} public_cold_alloc_bytes={} public_cold_peak_live_bytes={} public_cold_median_ns={ns}",
        allocs.calls, allocs.bytes, allocs.peak_live_bytes,
    );
}

#[test]
#[ignore = "measurement: run explicitly with --ignored --nocapture"]
fn measure_checked_generic_composer_sharing_regimes() {
    let _serial = counting_alloc::serial();
    let rule = SUNFusionRule::new(3).unwrap();
    let adjoint = vec![rule.encode_dynkin(&[1, 1]).unwrap()];
    // Pieri: tensoring with a symmetric power (p, 0) or its dual is
    // multiplicity-free, so every vertex over these legs is multiplicity-free
    // and every bend is one-to-one.
    let symmetric = vec![
        rule.encode_dynkin(&[1, 0]).unwrap(),
        rule.encode_dynkin(&[2, 0]).unwrap(),
        rule.encode_dynkin(&[0, 2]).unwrap(),
    ];
    let adj = |rank: usize| vec![adjoint.clone(); rank];
    let sym = |rank: usize| vec![symmetric.clone(); rank];

    for (rank, codomain_rank) in [(2, 1), (4, 2), (6, 3)] {
        let domain_rank = rank - codomain_rank;
        measure_case(
            "high",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_transpose_anticlockwise"),
            &adj(rank),
            codomain_rank,
            rotation(codomain_rank, domain_rank, 1, codomain_rank),
        );
        measure_case(
            "high",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_transpose_clockwise"),
            &adj(rank),
            codomain_rank,
            rotation(codomain_rank, domain_rank, rank - 1, codomain_rank),
        );
        measure_case(
            "high",
            &format!("r{rank}_{rank}|0_permute_reverse"),
            &adj(rank),
            rank,
            TreeTransformOperation::permute((0..rank).rev().collect::<Vec<_>>(), []),
        );
        measure_case(
            "high",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_braid_levels"),
            &adj(rank),
            codomain_rank,
            TreeTransformOperation::braid(
                (0..codomain_rank).rev().collect::<Vec<_>>(),
                (codomain_rank..rank).rev().collect::<Vec<_>>(),
                (0..codomain_rank).collect::<Vec<_>>(),
                (codomain_rank..rank).rev().collect::<Vec<_>>(),
            ),
        );
        // A pure repartition is a chain of bends; over multiplicity-free
        // vertices each bend maps one tree to one tree, so no intermediate
        // tree is shared (c ≡ 1). Cycles are not used here: a fold
        // recouples the coupled line and fans out.
        for split in [codomain_rank + 1, codomain_rank - 1] {
            measure_case(
                "low",
                &format!("r{rank}_{codomain_rank}|{domain_rank}_repartition_to_{split}"),
                &sym(rank),
                codomain_rank,
                rotation(codomain_rank, domain_rank, 0, split),
            );
        }
        measure_case(
            "identity",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_permute_identity"),
            &adj(rank),
            codomain_rank,
            TreeTransformOperation::permute(
                (0..codomain_rank).collect::<Vec<_>>(),
                (codomain_rank..rank).collect::<Vec<_>>(),
            ),
        );
    }
}
