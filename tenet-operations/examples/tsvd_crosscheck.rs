//! Prints per-sector singular values for deterministic fusion tensors so they
//! can be compared against TensorKit (`benchmarks/tensorkit_tsvd_crosscheck.jl`).
//!
//! Both sides fill every fusion-tree pair block with the same integer-hash
//! function of the sector labels and degeneracy indices. Singular values per
//! coupled sector are invariant under fusion-tree ordering and per-tree basis
//! conventions, so equal spectra validate the fusion-space structure and the
//! blockwise SVD end to end.

use tenet_core::{
    BlockKey, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, SU2Irrep, SectorId, SectorLeg, TensorMap, TensorMapSpace,
    U1FusionRule, U1Irrep,
};
use tenet_operations::svd_vals;

const DEGENERACY: usize = 2;

fn main() {
    run_case(
        "U1",
        &U1FusionRule,
        &[-1, 0, 1].map(U1Irrep::new).map(|irrep| irrep.sector_id()),
        |sector| U1Irrep::from_sector_id(sector).expect("U1 sector").charge() as i64,
    );
    run_case(
        "SU2",
        &tenet_core::SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
        |sector| sector.id() as i64,
    );
}

/// Shared deterministic fill, identical to the Julia side:
/// `((3 + 7 l1 + 11 l2 + 13 m1 + 17 m2 + 19 lc + 23 i1 + 29 i2 + 31 j1 + 37 j2) % 41) - 20`
/// where `l*`/`m*`/`lc` are integer sector labels (charge for U(1), twice the
/// spin for SU(2)) and `i*`/`j*` are one-based degeneracy indices.
#[allow(clippy::too_many_arguments)]
fn fill_value(
    l1: i64,
    l2: i64,
    m1: i64,
    m2: i64,
    lc: i64,
    i1: i64,
    i2: i64,
    j1: i64,
    j2: i64,
) -> f64 {
    let hash =
        3 + 7 * l1 + 11 * l2 + 13 * m1 + 17 * m2 + 19 * lc + 23 * i1 + 29 * i2 + 31 * j1 + 37 * j2;
    (hash.rem_euclid(41) - 20) as f64
}

fn run_case<R>(name: &str, rule: &R, sectors: &[SectorId], label_of: impl Fn(SectorId) -> i64)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let leg = || SectorLeg::new(sectors.iter().copied(), false);
    let leg_dim = sectors.len() * DEGENERACY;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        rule,
        vec![vec![DEGENERACY; 4]; key_count],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    let mut tensor =
        TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; len], space).unwrap();

    let structure = std::sync::Arc::clone(tensor.structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let codomain = key.codomain_tree().uncoupled();
        let domain = key.domain_tree().uncoupled();
        let coupled = key
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| rule.vacuum());
        let labels = [
            label_of(codomain[0]),
            label_of(codomain[1]),
            label_of(domain[0]),
            label_of(domain[1]),
            label_of(coupled),
        ];
        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        for j2 in 0..shape[3] {
            for j1 in 0..shape[2] {
                for i2 in 0..shape[1] {
                    for i1 in 0..shape[0] {
                        let position = offset
                            + i1 * strides[0]
                            + i2 * strides[1]
                            + j1 * strides[2]
                            + j2 * strides[3];
                        tensor.data_mut()[position] = fill_value(
                            labels[0],
                            labels[1],
                            labels[2],
                            labels[3],
                            labels[4],
                            (i1 + 1) as i64,
                            (i2 + 1) as i64,
                            (j1 + 1) as i64,
                            (j2 + 1) as i64,
                        );
                    }
                }
            }
        }
    }

    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let spectra = svd_vals(&mut dense, rule, &tensor).unwrap();
    let mut entries: Vec<(i64, Vec<f64>)> = spectra
        .iter()
        .map(|entry| (label_of(entry.sector), entry.values.clone()))
        .collect();
    entries.sort_by_key(|(label, _)| *label);
    for (label, values) in entries {
        let formatted = values
            .iter()
            .map(|value| format!("{value:.10}"))
            .collect::<Vec<_>>()
            .join(",");
        println!("{name}\t{label}\t{formatted}");
    }
}
