//! A tree transform of a lazy adjoint is the lazy adjoint of the transformed
//! parent in both admission modes (#1865).
//!
//! TensorKit `cfaa073e` defines `permute(t'::AdjointTensorMap, p) =
//! adjoint(permute(parent(t'), adjointtensorindices(t', p)))`
//! (`src/tensors/indexmanipulations.jl:260-264`), and `braid` (L343-352),
//! `transpose` (L414-421) and `repartition` (through `transpose`) the same
//! way, for every fusion style. QSpace has no lazy adjoint (`QSpace::Permute`
//! conjugates eagerly), so TensorKit semantics govern.
//!
//! Oracles, each independent of the transform route under test:
//! - (i) reference steps: `copy(t')` (the uncached `materialize`), then the
//!   same transform on the owned tensor;
//! - (ii) cross-mode agreement: SU(2) admitted multiplicity-free
//!   (`SU2FusionRule`) against checked Generic (`SUNFusionRule::new(2)`), two
//!   coefficient engines on one input;
//! - SU(3) (checked only) with outer multiplicity two (`8 ⊗ 8 → 8`) against (i).
//!
//! Host only: checked Generic has no device API (#1756); the multiplicity-free
//! device lazy-adjoint gates live in `representation_gates/cuda.rs`.

#![cfg(feature = "racah-generated")]

use std::collections::BTreeMap;
use std::sync::Arc;

use tenet::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
use tenet::typed::__network::{network_reuse_class, NetworkReuseClass};
use tenet::typed::{BlockFusionTrees, Complex64, GradedSpace, TensorMap};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::host_runtime;

/// `(2j, k)` per sector: nontrivial, distinct degeneracies.
const SECTORS: [(usize, usize); 3] = [(0, 2), (1, 3), (2, 2)];

/// `(name, entries of the materialized result)` per transform.
type Rows = Vec<(&'static str, BTreeMap<String, Vec<Complex64>>)>;

fn twice<S>(trees: &BlockFusionTrees<S>, spin: impl Fn(&S) -> usize) -> String {
    let map = |labels: &[S]| labels.iter().map(&spin).collect::<Vec<_>>();
    format!(
        "{:?}",
        (
            map(trees.codomain_uncoupled()),
            map(trees.codomain_innerlines()),
            map(trees.domain_uncoupled()),
            map(trees.domain_innerlines()),
            spin(trees.coupled()),
        )
    )
}

/// A value fixed by the block key and local index only, so both SU(2)
/// providers fill the same reduced entries.
fn value(key: &str, index: &[usize]) -> f64 {
    let hash = format!("{key}{index:?}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    ((hash % 2001) as f64 - 1000.0) / 257.0
}

fn assert_close(
    what: &str,
    got: &BTreeMap<String, Vec<Complex64>>,
    want: &BTreeMap<String, Vec<Complex64>>,
) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{what}: blocks"
    );
    for (key, want) in want {
        let got = &got[key];
        assert_eq!(got.len(), want.len(), "{what} {key}: length");
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).norm() <= 1e-10 * (1.0 + w.norm()),
                "{what} {key}: {g} vs {w}"
            );
        }
    }
}

/// The lazy adjoint of `$source` through every tree transform: each result
/// stays a lazy adjoint, leaves its input lazy, and equals oracle (i).
macro_rules! transform_rows {
    ($source:expr, $key:expr) => {{
        let key = $key;
        let entries = |t: &TensorMap<_, _>| -> BTreeMap<String, Vec<Complex64>> {
            t.subblocks()
                .unwrap()
                .map(|(trees, view)| {
                    let shape = view.shape().to_vec();
                    let len = shape.iter().product::<usize>();
                    let values = (0..len)
                        .map(|mut linear| {
                            let index: Vec<usize> = shape
                                .iter()
                                .map(|&extent| {
                                    let i = linear % extent;
                                    linear /= extent;
                                    i
                                })
                                .collect();
                            Complex64::from(*view.get(&index).unwrap())
                        })
                        .collect();
                    (key(&trees), values)
                })
                .collect()
        };
        let lazy = $source.adjoint().unwrap();
        let eager = lazy.materialize().unwrap();
        let mut rows: Rows = Vec::new();
        macro_rules! row {
            ($name:expr, |$t:ident| $op:expr) => {{
                assert!(matches!(
                    network_reuse_class(&lazy, false),
                    NetworkReuseClass::LazyAdjoint
                ));
                let actual = {
                    let $t = &lazy;
                    $op.unwrap()
                };
                let expected = {
                    let $t = &eager;
                    $op.unwrap()
                };
                assert!(matches!(
                    network_reuse_class(&lazy, false),
                    NetworkReuseClass::LazyAdjoint
                ));
                assert!(
                    matches!(
                        network_reuse_class(&actual, false),
                        NetworkReuseClass::LazyAdjoint
                    ),
                    "{}",
                    $name
                );
                let actual = entries(&actual.materialize().unwrap());
                assert_close($name, &actual, &entries(&expected));
                rows.push(($name, actual));
            }};
        }
        row!("permute", |t| t.permute(&[2, 0], &[1]));
        row!("braid", |t| t.braid(&[2, 0], &[1], &[17, 3, 11]));
        row!("transpose", |t| t.transpose(&[2, 1], &[0]));
        row!("repartition", |t| t.repartition(2));
        rows
    }};
}

fn su2_spin(sector: &SU2Irrep) -> usize {
    sector.twice_spin()
}

#[expect(clippy::ptr_arg, reason = "the SU(N) sector label is a `Vec<i64>`")]
fn sun_spin(sector: &Vec<i64>) -> usize {
    sector[0] as usize
}

fn su2_sources() -> (TensorMap<SU2FusionRule, f64>, TensorMap<SUNFusionRule, f64>) {
    let runtime = host_runtime();
    let mf = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
    )
    .unwrap();
    let checked = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(2).unwrap()),
        SECTORS.map(|(s, k)| (vec![s as i64], k)),
    )
    .unwrap();
    (
        TensorMap::from_subblock_fn(&runtime, [&mf, &mf], [&mf], |trees, index| {
            value(&twice(trees, su2_spin), index)
        })
        .unwrap(),
        TensorMap::from_subblock_fn(
            &runtime,
            [&checked, &checked],
            [&checked],
            |trees, index| value(&twice(trees, sun_spin), index),
        )
        .unwrap(),
    )
}

fn assert_modes_agree(mf: Rows, checked: Rows) {
    for ((name, mf), (_, checked)) in mf.iter().zip(&checked) {
        assert_close(
            &format!("{name}: checked vs multiplicity-free"),
            checked,
            mf,
        );
    }
}

#[test]
fn su2_adjoint_transforms_stay_lazy_and_agree_across_modes_f64() {
    let (mf, checked) = su2_sources();
    let mf = transform_rows!(mf, |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin));
    let checked = transform_rows!(checked, |t: &BlockFusionTrees<Vec<i64>>| twice(t, sun_spin));
    assert_modes_agree(mf, checked);
}

#[test]
fn su2_adjoint_transforms_stay_lazy_and_agree_across_modes_c64() {
    // `1 + 2i` makes the conjugation visible.
    let phase = Complex64::new(1.0, 2.0);
    let (mf, checked) = su2_sources();
    let (mf, checked) = (
        mf.convert::<Complex64>().scale(phase),
        checked.convert::<Complex64>().scale(phase),
    );
    let mf = transform_rows!(mf, |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin));
    let checked = transform_rows!(checked, |t: &BlockFusionTrees<Vec<i64>>| twice(t, sun_spin));
    assert_modes_agree(mf, checked);
}

#[test]
fn su3_adjoint_transforms_with_outer_multiplicity_stay_lazy() {
    let runtime = host_runtime();
    let v = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(3).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &v], [&v], |trees, index| {
            value(&format!("{trees:?}"), index)
        })
        .unwrap();
    let key = |t: &BlockFusionTrees<Vec<i64>>| format!("{t:?}");
    let f64_rows = transform_rows!(source, key);
    let c64 = source
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 2.0));
    let c64_rows = transform_rows!(c64, key);
    assert_eq!(f64_rows.len(), c64_rows.len());
    assert!(f64_rows.iter().all(|(_, entries)| !entries.is_empty()));
}
