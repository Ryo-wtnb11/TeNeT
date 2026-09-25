//! #1517: contraction of expert-layer tilings whose coupled-sector row and
//! column tree stackings differ from each other or from the partner operand.

use std::collections::BTreeMap;

use super::*;
use tenet_core::{
    BlockKey, BlockStructure, FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace,
    FusionTreePairKey, SU2FusionRule, SU2Irrep, SectorLeg, U1FusionRule, U1Irrep, Z2FusionRule,
    Z2Irrep,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stacking {
    /// The ordinary facade layout, in hom-space key order.
    Canonical,
    /// Rows by ascending codomain tree, columns by ascending domain tree.
    Sorted,
    /// Rows ascending, columns descending: row `i` and column `i` of a
    /// multi-tree sector name different trees.
    ColumnsReversed,
    /// Rows and columns both descending: one consistent, non-canonical basis.
    BothReversed,
}

/// `[s…]⊗[s…] ← [s…]⊗[s…]` endomorphism whose entries depend only on the
/// fusion-tree key and the degeneracy index, laid out with `stacking`.
fn endomorphism<R>(
    runtime: &Runtime,
    rule: R,
    sectors: &[(SectorId, usize)],
    stacking: Stacking,
) -> TensorMap<R, f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Clone
        + 'static,
{
    let leg = || SectorLeg::new(sectors.iter().copied(), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let dim = |sector: SectorId| {
        sectors
            .iter()
            .find(|(candidate, _)| *candidate == sector)
            .unwrap()
            .1
    };
    let keys = homspace.fusion_tree_keys(&rule);
    let mut blocks: Vec<(FusionTreePairKey, Vec<usize>)> = keys
        .iter()
        .map(|key| {
            let shape = key
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(key.domain_tree().uncoupled())
                .map(|&sector| dim(sector))
                .collect();
            (key.clone(), shape)
        })
        .collect();
    blocks.sort_by(|(a, _), (b, _)| {
        if stacking == Stacking::Canonical {
            return std::cmp::Ordering::Equal;
        }
        let (codomain, domain) = (
            a.codomain_tree().cmp(b.codomain_tree()),
            a.domain_tree().cmp(b.domain_tree()),
        );
        match stacking {
            Stacking::Canonical => unreachable!(),
            Stacking::Sorted => codomain.then(domain),
            Stacking::ColumnsReversed => codomain.then(domain.reverse()),
            Stacking::BothReversed => codomain.reverse().then(domain.reverse()),
        }
    });
    let structure = BlockStructure::coupled_sector_matrix_with_keys(&rule, 2, 4, blocks).unwrap();
    let dense_dim: usize = sectors.iter().map(|(_, d)| d).sum();
    let space = tenet_core::FusionTensorMapSpace::new_unbound(
        tenet_core::TensorMapSpace::<2, 2>::from_dims([dense_dim; 2], [dense_dim; 2]).unwrap(),
        homspace,
        structure,
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let core = tenet_core::TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(
        space,
        0.0,
        |key, indices| {
            let BlockKey::FusionTree(tree) = key else {
                unreachable!("fusion-tree blocks")
            };
            let position = keys.iter().position(|candidate| candidate == tree).unwrap();
            let index = indices
                .iter()
                .enumerate()
                .map(|(axis, &i)| (axis + 1) * i)
                .sum::<usize>();
            if tree.codomain_tree() == tree.domain_tree() {
                3.0 + 0.25 * index as f64
            } else {
                0.5 + 0.5 * ((7 * position + 3 * index) % 5) as f64 - 1.0
            }
        },
    )
    .unwrap();
    let space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        tenet_tensors::DynamicFusionMapSpace::from_typed(core.fusion_space().unwrap()),
        Arc::new(rule),
    )
    .unwrap();
    TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody {
            space,
            data: Arc::new(TypedData::Dense(core.data().to_vec())),
            dense_cache: std::sync::OnceLock::new(),
        }),
    }
}

/// Entries keyed by fusion tree and degeneracy index, independent of layout.
fn entries<R>(tensor: &TensorMap<R, f64>) -> BTreeMap<(BlockKey, Vec<usize>), f64> {
    let data = tensor.data();
    let mut entries = BTreeMap::new();
    for index in 0..tensor.block_count() {
        let block = tensor.block(index).unwrap();
        let shape = block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        for linear in 0..count {
            let mut rest = linear;
            let coordinates = shape
                .iter()
                .map(|&extent| {
                    let coordinate = rest % extent;
                    rest /= extent;
                    coordinate
                })
                .collect::<Vec<_>>();
            let offset = block.offset()
                + coordinates
                    .iter()
                    .zip(block.strides())
                    .map(|(&c, &s)| c * s)
                    .sum::<usize>();
            entries.insert((block.key().clone(), coordinates), data[offset]);
        }
    }
    entries
}

fn assert_same_operator<R>(actual: &TensorMap<R, f64>, expected: &TensorMap<R, f64>, what: &str) {
    let actual = entries(actual);
    let expected = entries(expected);
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "{what}: different block keys"
    );
    for ((key, value), expected) in actual.iter().zip(expected.values()) {
        assert!(
            (value - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{what}: {key:?}: {value} != {expected}"
        );
    }
}

/// Independent composition oracle over tree identity: `(A B)[X, Z] =
/// Σ_Y A[X, Y] B[Y, Z]` with no recoupling, since composition crosses no leg.
fn compose_oracle(
    lhs: &BTreeMap<(BlockKey, Vec<usize>), f64>,
    rhs: &BTreeMap<(BlockKey, Vec<usize>), f64>,
) -> BTreeMap<(BlockKey, Vec<usize>), f64> {
    let mut product = BTreeMap::new();
    for ((lhs_key, lhs_index), lhs_value) in lhs {
        let BlockKey::FusionTree(lhs_tree) = lhs_key else {
            unreachable!()
        };
        for ((rhs_key, rhs_index), rhs_value) in rhs {
            let BlockKey::FusionTree(rhs_tree) = rhs_key else {
                unreachable!()
            };
            if lhs_tree.domain_tree() != rhs_tree.codomain_tree()
                || lhs_index[2..] != rhs_index[..2]
            {
                continue;
            }
            let key = BlockKey::FusionTree(FusionTreePairKey::pair(
                lhs_tree.codomain_tree().clone(),
                rhs_tree.domain_tree().clone(),
            ));
            let index = [&lhs_index[..2], &rhs_index[2..]].concat();
            *product.entry((key, index)).or_insert(0.0) += lhs_value * rhs_value;
        }
    }
    product
}

fn assert_entries_close(
    actual: &BTreeMap<(BlockKey, Vec<usize>), f64>,
    expected: &BTreeMap<(BlockKey, Vec<usize>), f64>,
    what: &str,
) {
    for (key, expected) in expected {
        let value = actual.get(key).copied().unwrap_or(0.0);
        assert!(
            (value - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{what}: {key:?}: {value} != {expected}"
        );
    }
    for (key, value) in actual {
        assert!(
            expected.contains_key(key) || *value == 0.0,
            "{what}: unexpected entry {key:?} = {value}"
        );
    }
}

/// Every stacking pair either reproduces the canonical result or is refused.
fn assert_stacking_invariant<R>(rule: R, sectors: &[(SectorId, usize)])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Clone
        + 'static,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let build = |stacking| endomorphism(&runtime, rule.clone(), sectors, stacking);
    let good = build(Stacking::Canonical);
    let tilings = [
        Stacking::Canonical,
        Stacking::Sorted,
        Stacking::ColumnsReversed,
        Stacking::BothReversed,
    ]
    .map(|stacking| (stacking, build(stacking)));
    for (stacking, tensor) in &tilings {
        assert_same_operator(tensor, &good, &format!("{stacking:?} fixture"));
    }
    let square = compose_oracle(&entries(&good), &entries(&good));
    let cube = compose_oracle(&square, &entries(&good));
    for (stacking, tensor) in &tilings {
        let what = |op: &str| format!("{stacking:?} {op}");
        assert_entries_close(
            &entries(&tensor.powi(2).unwrap()),
            &square,
            &what("powi(2)"),
        );
        assert_entries_close(&entries(&tensor.powi(3).unwrap()), &cube, &what("powi(3)"));
        // A lazy adjoint admits only the ordinary layout; others are refused.
        if let Ok(adjoint) = tensor.adjoint() {
            let good_adjoint = good.adjoint().unwrap();
            assert_same_operator(
                &adjoint.compose(tensor).unwrap(),
                &good_adjoint.compose(&good).unwrap(),
                &what("adjoint compose"),
            );
            assert_same_operator(
                &tensor.compose(&adjoint).unwrap(),
                &good.compose(&good_adjoint).unwrap(),
                &what("compose adjoint"),
            );
            assert_same_operator(
                &adjoint.compose(&tilings[2].1).unwrap(),
                &good_adjoint.compose(&good).unwrap(),
                &what("adjoint compose mis-stacked"),
            );
        } else {
            assert_ne!(*stacking, Stacking::Canonical);
        }
        assert!(
            (tensor.tr().unwrap() - good.tr().unwrap()).abs() <= 1e-12,
            "{stacking:?} tr"
        );
        assert_same_operator(
            &tensor.trace_pairs(&[(1, 3)]).unwrap(),
            &good.trace_pairs(&[(1, 3)]).unwrap(),
            &what("trace_pairs"),
        );
        assert_same_operator(
            &tensor.permute(&[1, 0], &[3, 2]).unwrap(),
            &good.permute(&[1, 0], &[3, 2]).unwrap(),
            &what("permute"),
        );
        // Addition pairs storage positionally, so it admits only one layout.
        match tensor.add(&good, 1.0, 1.0) {
            Ok(sum) => assert_same_operator(&sum, &good.scale(2.0), &what("add")),
            Err(error) => assert!(
                format!("{error:?}").contains("block layouts"),
                "{stacking:?} add: {error:?}"
            ),
        }
        for (other_stacking, other) in &tilings {
            let what = |op: &str| format!("{stacking:?}·{other_stacking:?} {op}");
            assert_entries_close(
                &entries(&tensor.compose(other).unwrap()),
                &square,
                &what("compose"),
            );
            // Full contraction, identity and swapped output order: the
            // sorted and swapped operand candidates.
            let full = good
                .contract(&good, &[2, 3], &[0, 1], &[0, 1, 2, 3])
                .unwrap();
            assert_same_operator(
                &tensor
                    .contract(other, &[2, 3], &[0, 1], &[0, 1, 2, 3])
                    .unwrap(),
                &full,
                &what("contract"),
            );
            assert_same_operator(
                &tensor
                    .contract(other, &[2, 3], &[0, 1], &[2, 3, 0, 1])
                    .unwrap(),
                &good
                    .contract(&good, &[2, 3], &[0, 1], &[2, 3, 0, 1])
                    .unwrap(),
                &what("contract swapped output"),
            );
            // One contracted leg: recoupled source and output transforms.
            assert_same_operator(
                &tensor
                    .contract(other, &[3], &[1], &[0, 1, 2, 3, 4, 5])
                    .unwrap(),
                &good
                    .contract(&good, &[3], &[1], &[0, 1, 2, 3, 4, 5])
                    .unwrap(),
                &what("partial contract"),
            );
            assert_same_operator(
                &tensor
                    .contract(other, &[2], &[0], &[3, 4, 0, 1, 2, 5])
                    .unwrap(),
                &good
                    .contract(&good, &[2], &[0], &[3, 4, 0, 1, 2, 5])
                    .unwrap(),
                &what("partial contract permuted"),
            );
        }
    }
}

#[test]
fn z2_mis_stacked_contractions_match_tree_identity() {
    assert_stacking_invariant(
        Z2FusionRule,
        &[
            (Z2Irrep::new(0).sector_id(), 2),
            (Z2Irrep::new(1).sector_id(), 1),
        ],
    );
}

#[test]
fn fermion_parity_mis_stacked_contractions_match_tree_identity() {
    assert_stacking_invariant(
        FermionParityFusionRule,
        &[(SectorId::new(0), 2), (SectorId::new(1), 1)],
    );
}

#[test]
fn u1_mis_stacked_contractions_match_tree_identity() {
    assert_stacking_invariant(
        U1FusionRule,
        &[
            (U1Irrep::new(-1).sector_id(), 1),
            (U1Irrep::new(0).sector_id(), 2),
            (U1Irrep::new(1).sector_id(), 1),
        ],
    );
}

#[test]
fn su2_mis_stacked_contractions_match_tree_identity() {
    assert_stacking_invariant(
        SU2FusionRule,
        &[
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
        ],
    );
}
