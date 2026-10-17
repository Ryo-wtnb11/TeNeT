//! The lazy adjoint of a complex-symbol rule (#2281): Ising through
//! `TableRule`, whose R-symbols are complex, so a transform of `t^H` read
//! through its parent must use the dagger (inverse, conjugated) operation.
//!
//! Oracles:
//! - materialization: TensorKit `block(t', c) = block(t, c)'` written per
//!   element (codomain and domain trees swap, indices move the domain axes
//!   first, the value is conjugated), from the same per-entry hash that
//!   filled the parent, so no symtn kernel is involved;
//! - every braid, planar transpose and repartition of the lazy adjoint (the
//!   parent's lowered transform, adjointed) equals the same operation on the
//!   materialized adjoint (the forward transform of the hand-checked payload).
//!   The two routes evaluate different coefficient sequences, so their
//!   agreement is the dagger identity `op(t^H) = (op^dagger(t))^H` with the
//!   complex Ising R and F symbols;
//! - `compose` with an adjoint-view operand equals `compose` with the
//!   materialized adjoint.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use num_complex::Complex64;
use symtn::typed::{BlockFusionTrees, GradedSpace, Runtime, TensorMap};
use symtn_sectors::{TableRule, TableSector};

/// At most a few unit-modulus 2x2 recouplings and one dense pass touch any
/// entry of the `O(1)`-magnitude payload.
const BOUND: f64 = 256.0 * f64::EPSILON;

type Ising = TensorMap<TableRule, Complex64>;
type Transform<'a> = &'a dyn Fn(&Ising) -> Result<Ising, symtn::typed::Error>;
type Entries = BTreeMap<(BlockFusionTrees<TableSector>, Vec<usize>), Complex64>;

fn hashed(
    coupled: &TableSector,
    codomain: (&[TableSector], &[TableSector]),
    domain: (&[TableSector], &[TableSector]),
    indices: &[usize],
    salt: u8,
) -> f64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (coupled, codomain, domain, indices, salt).hash(&mut hasher);
    let bits = hasher.finish();
    let magnitude = 0.25 + (bits >> 11) as f64 / (1u64 << 53) as f64;
    if bits & 1 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn entry(trees: &BlockFusionTrees<TableSector>, indices: &[usize], salt: u8) -> Complex64 {
    let side = |salt| {
        hashed(
            trees.coupled(),
            (trees.codomain_uncoupled(), trees.codomain_innerlines()),
            (trees.domain_uncoupled(), trees.domain_innerlines()),
            indices,
            salt,
        )
    };
    Complex64::new(side(2 * salt), side(2 * salt + 1))
}

/// The parent entry the adjoint entry `(trees, indices)` reads, conjugated.
fn adjoint_entry(trees: &BlockFusionTrees<TableSector>, indices: &[usize], salt: u8) -> Complex64 {
    let rank = trees.codomain_uncoupled().len();
    let mut parent = indices[rank..].to_vec();
    parent.extend_from_slice(&indices[..rank]);
    let side = |salt| {
        hashed(
            trees.coupled(),
            (trees.domain_uncoupled(), trees.domain_innerlines()),
            (trees.codomain_uncoupled(), trees.codomain_innerlines()),
            &parent,
            salt,
        )
    };
    Complex64::new(side(2 * salt), side(2 * salt + 1)).conj()
}

fn entries(tensor: &TensorMap<TableRule, Complex64>) -> Entries {
    let mut out = Entries::new();
    for (trees, block) in tensor.subblocks().unwrap() {
        let shape = block.shape().to_vec();
        let mut index = vec![0; shape.len()];
        'walk: loop {
            out.insert((trees.clone(), index.clone()), *block.get(&index).unwrap());
            for axis in 0..shape.len() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    continue 'walk;
                }
                index[axis] = 0;
            }
            break;
        }
    }
    out
}

fn assert_close(what: &str, got: &Entries, want: &Entries) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{what}: entry sets differ"
    );
    assert!(!want.is_empty(), "{what}: empty");
    for (key, value) in want {
        let residual = (got[key] - value).norm();
        assert!(
            residual <= BOUND,
            "{what} at {key:?}: {:?} != {value:?} (residual {residual:e})",
            got[key]
        );
    }
}

struct Fixture {
    runtime: Runtime,
    leg: GradedSpace<TableRule>,
}

fn fixture() -> Fixture {
    // AnyonWiki Ising: 1 = unit, 2 = psi, 3 = sigma; nontrivial degeneracy on
    // sigma so every block is a genuine matrix.
    let leg = GradedSpace::try_new(
        Arc::new(TableRule::ising()),
        [
            (TableSector::new(1), 1),
            (TableSector::new(2), 1),
            (TableSector::new(3), 2),
        ],
    )
    .unwrap();
    Fixture {
        runtime: Runtime::builder().build().unwrap(),
        leg,
    }
}

fn parent(fixture: &Fixture, salt: u8) -> TensorMap<TableRule, Complex64> {
    let v = &fixture.leg;
    TensorMap::from_subblock_fn(&fixture.runtime, [v, v, v], [v], |trees, indices| {
        entry(trees, indices, salt)
    })
    .unwrap()
}

#[test]
fn ising_adjoint_materializes_to_the_blockwise_conjugate_transpose() {
    let fixture = fixture();
    let t = parent(&fixture, 0);
    let adjoint = t.adjoint().unwrap();
    let materialized = adjoint.materialize().unwrap();
    let want: Entries = entries(&materialized)
        .keys()
        .map(|(trees, indices)| {
            (
                (trees.clone(), indices.clone()),
                adjoint_entry(trees, indices, 0),
            )
        })
        .collect();
    // A materialization is an exact copy-with-conjugation.
    assert_eq!(entries(&materialized), want);
    assert_eq!(
        adjoint.adjoint().unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );
}

#[test]
fn ising_lazy_adjoint_transforms_equal_transforms_of_the_materialized_adjoint() {
    let fixture = fixture();
    let t = parent(&fixture, 1);
    // t^H : V <- V V V, axes [0 | 1 2 3], planar order [0, 3, 2, 1].
    let lazy = t.adjoint().unwrap();
    let eager = lazy.materialize().unwrap();
    let check = |what: String, op: Transform<'_>| {
        let want = entries(&op(&eager).unwrap());
        let result = op(&lazy).unwrap();
        // The lazy-adjoint arm returns a lazy view; a materialize-first
        // fallback would return an owned payload.
        assert!(
            result.dense_data().is_err(),
            "{what}: the lazy-adjoint arm was not taken"
        );
        let got = entries(&result.materialize().unwrap());
        assert_close(&what, &got, &want);
    };

    let planar = [0usize, 3, 2, 1];
    for rotation in 0..4 {
        let order: Vec<usize> = (0..4).map(|i| planar[(i + rotation) % 4]).collect();
        for split in 0..=4 {
            let codomain = order[..split].to_vec();
            let mut domain = order[split..].to_vec();
            domain.reverse();
            check(format!("transpose {codomain:?} {domain:?}"), &|x| {
                x.transpose(&codomain, &domain)
            });
        }
    }
    for split in 0..=4 {
        check(format!("repartition {split}"), &|x| x.repartition(split));
    }
    let braids: [(&[usize], &[usize]); 4] = [
        (&[0], &[2, 1, 3]),
        (&[1], &[0, 2, 3]),
        (&[2, 0], &[3, 1]),
        (&[], &[3, 0, 1, 2]),
    ];
    for (codomain, domain) in braids {
        for levels in [[0usize, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]] {
            check(
                format!("braid {codomain:?} {domain:?} levels {levels:?}"),
                &|x| x.braid(codomain, domain, &levels),
            );
        }
    }
}

#[test]
fn ising_compose_with_an_adjoint_view_equals_the_materialized_adjoint() {
    let fixture = fixture();
    let t = parent(&fixture, 2);
    let s = parent(&fixture, 3);
    // (V V V <- V) o (V <- V V V)
    let materialized = s.adjoint().unwrap().materialize().unwrap();
    let want = entries(&t.compose(&materialized).unwrap());
    assert_close(
        "compose(t, s^H view)",
        &entries(&t.compose(s.adjoint_view()).unwrap()),
        &want,
    );
    // (V <- V V V) o (V V V <- V), the lazy operand on the left.
    let want = entries(&materialized.compose(&t).unwrap());
    assert_close(
        "compose(s^H, t)",
        &entries(&s.adjoint().unwrap().compose(&t).unwrap()),
        &want,
    );
}

#[test]
fn ising_axpby_with_lazy_adjoints_equals_the_materialized_combination() {
    let fixture = fixture();
    let (t, s) = (parent(&fixture, 4), parent(&fixture, 5));
    let (alpha, beta) = (Complex64::new(0.75, -0.5), Complex64::new(-1.25, 0.375));
    let (lazy_t, lazy_s) = (t.adjoint().unwrap(), s.adjoint().unwrap());
    let (eager_t, eager_s) = (lazy_t.materialize().unwrap(), lazy_s.materialize().unwrap());
    let want = entries(&eager_t.axpby(alpha, &eager_s, beta).unwrap());
    // Both lazy: the lazy adjoint of the parents' conjugate-coefficient sum.
    let both = lazy_t.axpby(alpha, &lazy_s, beta).unwrap();
    assert_close(
        "axpby(t^H, s^H)",
        &entries(&both.materialize().unwrap()),
        &want,
    );
    // One lazy operand, read in its logical orientation.
    let one = eager_t.axpby(alpha, s.adjoint_view(), beta).unwrap();
    assert_close("axpby(t^H, s^H view)", &entries(&one), &want);
}
