//! #1517: contraction of expert-layer tilings whose coupled-sector row and
//! column tree stackings differ from each other or from the partner operand.

use std::collections::BTreeMap;

use super::*;
use tenet_core::{
    BlockKey, BlockStructure, FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace,
    FusionTreePairKey, PhysicalFusionBasis, SU2FusionRule, SU2Irrep, SectorLeg, U1FusionRule,
    U1Irrep, Z2FusionRule, Z2Irrep,
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

/// Column-major einsum over physical-basis arrays: contracts `lhs_axes` with
/// `rhs_axes` and orders the open axes (lhs then rhs) by `output_axes`.
fn dense_contract(
    lhs: &PhysicalDense<f64>,
    rhs: &PhysicalDense<f64>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
) -> PhysicalDense<f64> {
    let decode = |shape: &[usize], mut linear: usize| {
        shape
            .iter()
            .map(|&extent| {
                let coordinate = linear % extent;
                linear /= extent;
                coordinate
            })
            .collect::<Vec<_>>()
    };
    let open = |shape: &[usize], axes: &[usize]| {
        (0..shape.len())
            .filter(|axis| !axes.contains(axis))
            .collect::<Vec<_>>()
    };
    let (lhs_open, rhs_open) = (open(&lhs.shape, lhs_axes), open(&rhs.shape, rhs_axes));
    let open_extent = |i: usize| {
        if i < lhs_open.len() {
            lhs.shape[lhs_open[i]]
        } else {
            rhs.shape[rhs_open[i - lhs_open.len()]]
        }
    };
    let shape = output_axes
        .iter()
        .map(|&i| open_extent(i))
        .collect::<Vec<_>>();
    let mut data = vec![0.0; shape.iter().product()];
    for (l, &lhs_value) in lhs.data.iter().enumerate() {
        let a = decode(&lhs.shape, l);
        for (r, &rhs_value) in rhs.data.iter().enumerate() {
            let b = decode(&rhs.shape, r);
            if lhs_axes.iter().zip(rhs_axes).any(|(&i, &j)| a[i] != b[j]) {
                continue;
            }
            let coordinates = lhs_open
                .iter()
                .map(|&axis| a[axis])
                .chain(rhs_open.iter().map(|&axis| b[axis]))
                .collect::<Vec<_>>();
            let linear = output_axes
                .iter()
                .zip(&shape)
                .rev()
                .fold(0, |acc, (&i, &extent)| acc * extent + coordinates[i]);
            data[linear] += lhs_value * rhs_value;
        }
    }
    PhysicalDense { shape, data }
}

fn assert_dense_close(actual: &PhysicalDense<f64>, expected: &PhysicalDense<f64>, what: &str) {
    assert_eq!(actual.shape, expected.shape, "{what}: shape");
    for (index, (value, expected)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (value - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{what}: [{index}]: {value} != {expected}"
        );
    }
}

/// `compose`, `powi` and `contract` of every stacking pair (and the lazy
/// adjoint) match a physical-basis einsum of the canonical operand, which
/// does not depend on any fusion-tree basis or stacking. Returns the number
/// of checks.
fn assert_stacking_matches_dense<R>(rule: R, sectors: &[(SectorId, usize)]) -> usize
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + PhysicalFusionBasis<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Clone
        + 'static,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let build = |stacking| endomorphism(&runtime, rule.clone(), sectors, stacking);
    let good = build(Stacking::Canonical);
    let dense = good.to_physical_dense().unwrap();
    // A† of a real operator: swap the codomain and domain axis pairs.
    let dense_adjoint = dense_contract(
        &dense,
        &PhysicalDense {
            shape: vec![],
            data: vec![1.0],
        },
        &[],
        &[],
        &[2, 3, 0, 1],
    );
    let mut operands = [
        Stacking::Canonical,
        Stacking::Sorted,
        Stacking::ColumnsReversed,
        Stacking::BothReversed,
    ]
    .map(|stacking| (format!("{stacking:?}"), build(stacking), &dense))
    .to_vec();
    operands.push(("Adjoint".into(), good.adjoint().unwrap(), &dense_adjoint));
    let mut checks = 0;
    let mut check = |actual: &TensorMap<R, f64>, expected: &PhysicalDense<f64>, what: String| {
        assert_dense_close(&actual.to_physical_dense().unwrap(), expected, &what);
        checks += 1;
    };
    for (name, tensor, dense) in &operands {
        let square = dense_contract(dense, dense, &[2, 3], &[0, 1], &[0, 1, 2, 3]);
        let cube = dense_contract(&square, dense, &[2, 3], &[0, 1], &[0, 1, 2, 3]);
        check(&tensor.powi(2).unwrap(), &square, format!("{name} powi(2)"));
        check(&tensor.powi(3).unwrap(), &cube, format!("{name} powi(3)"));
        for (other_name, other, other_dense) in &operands {
            let what = |op: &str| format!("{name}·{other_name} {op}");
            check(
                &tensor.compose(other).unwrap(),
                &dense_contract(dense, other_dense, &[2, 3], &[0, 1], &[0, 1, 2, 3]),
                what("compose"),
            );
            for (lhs_axes, rhs_axes, output_axes, op) in [
                (&[2, 3][..], &[0, 1][..], &[0, 1, 2, 3][..], "contract"),
                (&[2, 3], &[0, 1], &[2, 3, 0, 1], "contract swapped"),
                (&[2, 3], &[0, 1], &[1, 0, 2, 3], "contract copyC"),
                (&[3], &[1], &[0, 1, 2, 3, 4, 5], "contract one leg"),
            ] {
                check(
                    &tensor
                        .contract(other, lhs_axes, rhs_axes, output_axes)
                        .unwrap(),
                    &dense_contract(dense, other_dense, lhs_axes, rhs_axes, output_axes),
                    what(op),
                );
            }
        }
    }
    checks
}

#[test]
fn u1_mis_stacked_contractions_match_physical_dense_einsum() {
    let checks = assert_stacking_matches_dense(
        U1FusionRule,
        &[
            (U1Irrep::new(-1).sector_id(), 1),
            (U1Irrep::new(0).sector_id(), 2),
            (U1Irrep::new(1).sector_id(), 1),
        ],
    );
    assert_eq!(checks, 5 * 2 + 5 * 5 * 5);
}

#[test]
fn su2_mis_stacked_contractions_match_physical_dense_einsum() {
    let checks = assert_stacking_matches_dense(
        SU2FusionRule,
        &[
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
        ],
    );
    assert_eq!(checks, 5 * 2 + 5 * 5 * 5);
}

#[test]
fn prepared_compose_rejects_a_non_direct_plan_at_new() {
    // What: an expert tiling whose column trees are stacked opposite to the
    // partner's rows has no canonical fully-direct plan, so the handle refuses
    // it at `new` with the typed error the eager device composition reports,
    // while eager Host `compose` still runs it.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let sectors = [
        (U1Irrep::new(0).sector_id(), 2),
        (U1Irrep::new(1).sector_id(), 1),
    ];
    let reversed = endomorphism(&runtime, U1FusionRule, &sectors, Stacking::ColumnsReversed);
    let canonical = endomorphism(&runtime, U1FusionRule, &sectors, Stacking::Canonical);
    assert!(reversed.compose(&canonical).is_ok());
    let lhs = StackedTensorMap::pack(&[reversed]).unwrap();
    let rhs = StackedTensorMap::pack(&[canonical]).unwrap();
    let error = PreparedCompose::new(&lhs, &rhs)
        .err()
        .expect("non-direct plan");
    assert!(
        matches!(
            &error,
            Error::Operation(operation)
                if matches!(operation.as_ref(), tenet_tensors::OperationError::UnsupportedTensorContractScope { .. })
        ),
        "{error:?}"
    );
}
