//! `*_into` admits what eager admits, in every admission mode (#1870).
//!
//! TensorKit `cfaa073e`: `permute!`/`braid!`/`transpose!`/`repartition!`
//! (`src/tensors/indexmanipulations.jl:217, 303, 374, 435`) check the
//! destination space and run `add_transform!` (L547) with `α, β` for every
//! fusion style. QSpace (`dd2cc7e1`) has no accumulate-into-existing form
//! (`QSpace::Permute` is in place, `contract` overwrites), so TensorKit
//! semantics govern.
//!
//! Each cell runs one `_into` on an existing destination `dst₀` and asserts:
//! 1. `dst == α·eager + β·dst₀` within tolerance, with `β = 0` seeded by NaN
//!    (never read) and `β ∈ {0.5, 1}` by distinct finite values; for an owned
//!    source the reference-step oracle (`materialize`, then eager) is eager
//!    itself;
//! 2. multiplicity-free SU(2) (`SU2FusionRule`) == checked Generic SU(2)
//!    (`SUNFusionRule::new(2)`), two coefficient engines on one input;
//! 3. the destination keeps its payload allocation and provider.
//!
//! Rows (leaf A1): owned dense source × {permute, braid, transpose,
//! repartition}. Columns: multiplicity-free real symbols (U(1), SU(2),
//! fermion parity, each with dual legs), multiplicity-free complex symbols
//! (Fibonacci), checked Generic (SU(2), SU(3) with outer multiplicity) ×
//! {f64, Complex64 scaled by `1 + 2i`}.
//!
//! Compile-time capability absences, recorded and not counted as passing
//! cells: checked Generic and complex-symbol rules × CUDA (the device
//! transforms are bounded `MultiplicityFreeRigidSymbols<Scalar = f64>`,
//! #1756), and a complex-symbol rule × lazy adjoint (`TypedAdjointSpace`
//! needs `Scalar = f64`). Fibonacci `permute` is anyonic and asserted as the
//! same rejection as eager.

use std::collections::BTreeMap;
use std::sync::Arc;

use tenet::sector::{
    FermionParityFusionRule, FibonacciFusionRule, FibonacciSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::typed::{BlockFusionTrees, Complex64, GradedSpace, TensorMap};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

use fixtures::host_runtime;

const BETAS: [f64; 3] = [0.0, 0.5, 1.0];

/// `(cell, entries of the destination after the call)`.
type Rows = Vec<(String, BTreeMap<String, Vec<Complex64>>)>;

trait Lane: Copy + Into<Complex64> {
    fn lift(x: f64) -> Self;
    /// `alpha`: a non-real phase on the complex lane.
    fn alpha() -> Self;
}

impl Lane for f64 {
    fn lift(x: f64) -> Self {
        x
    }
    fn alpha() -> Self {
        1.5
    }
}

impl Lane for Complex64 {
    fn lift(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn alpha() -> Self {
        Complex64::new(1.5, -0.5)
    }
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

fn close(got: Complex64, want: Complex64) -> bool {
    (got - want).norm() <= 1e-10 * (1.0 + want.norm())
}

fn dense<R, D: Lane>(t: &TensorMap<R, D>) -> Vec<Complex64> {
    t.dense_data().unwrap().iter().map(|&x| x.into()).collect()
}

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

fn su2_spin(sector: &SU2Irrep) -> usize {
    sector.twice_spin()
}

#[cfg(feature = "racah-generated")]
#[expect(clippy::ptr_arg, reason = "the SU(N) sector label is a `Vec<i64>`")]
fn sun_spin(sector: &Vec<i64>) -> usize {
    sector[0] as usize
}

/// The subblock entries of a tensor, keyed by `key`.
macro_rules! entries {
    ($t:expr, $key:expr) => {{
        let key = $key;
        $t.subblocks()
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
                        (*view.get(&index).unwrap()).into()
                    })
                    .collect::<Vec<Complex64>>();
                (key(&trees), values)
            })
            .collect::<BTreeMap<String, Vec<Complex64>>>()
    }};
}

/// One cell per `β`: `$into` on a fresh `dst₀` on eager's legs equals
/// `α·eager + β·dst₀` and keeps the destination's allocation and provider.
macro_rules! cell {
    ($rows:expr, $runtime:expr, $source:expr, $key:expr, $d:ty, $name:expr,
     |$t:ident| $eager:expr, |$dst:ident, $a:ident, $b:ident| $into:expr) => {{
        let key = $key;
        let $t = &$source;
        let eager = $eager.unwrap();
        let expected_eager = dense(&eager);
        for beta in BETAS {
            let mut destination: TensorMap<_, $d> = TensorMap::from_subblock_fn(
                $runtime,
                &eager.codomain(),
                &eager.domain(),
                |trees, index| {
                    if beta == 0.0 {
                        <$d>::lift(f64::NAN)
                    } else {
                        <$d>::lift(value(&format!("dst{}", key(trees)), index))
                    }
                },
            )
            .unwrap();
            let before = dense(&destination);
            let payload = destination.dense_data().unwrap().as_ptr();
            let provider = destination.provider() as *const _;
            {
                let ($dst, $a, $b) = (&mut destination, <$d>::alpha(), <$d>::lift(beta));
                $into.unwrap();
            }
            let what = format!("{} beta {beta}", $name);
            assert_eq!(
                destination.dense_data().unwrap().as_ptr(),
                payload,
                "{what}"
            );
            assert!(std::ptr::eq(destination.provider(), provider), "{what}");
            let alpha: Complex64 = <$d>::alpha().into();
            let got = dense(&destination);
            assert_eq!(got.len(), expected_eager.len(), "{what}");
            for (i, (&got, &e)) in got.iter().zip(&expected_eager).enumerate() {
                let want = alpha * e
                    + if beta == 0.0 {
                        Complex64::new(0.0, 0.0)
                    } else {
                        before[i] * beta
                    };
                assert!(close(got, want), "{what} [{i}]: {got} vs {want}");
            }
            $rows.push((what, entries!(destination, key)));
        }
    }};
}

/// Every tree-transform `_into` of an owned dense rank-(2, 2) `$source`.
macro_rules! transform_cells {
    ($runtime:expr, $source:expr, $key:expr, $d:ty) => {{
        let mut rows: Rows = Vec::new();
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "permute",
            |t| t.permute(&[2, 0], &[1, 3]),
            |d, a, b| $source.permute_into(&[2, 0], &[1, 3], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "braid",
            |t| t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]),
            |d, a, b| $source.braid_into(&[1, 0], &[3, 2], &[0, 1, 2, 3], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "transpose",
            |t| t.transpose(&[1, 3], &[0, 2]),
            |d, a, b| $source.transpose_into(&[1, 3], &[0, 2], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "repartition",
            |t| t.repartition(1),
            |d, a, b| $source.repartition_into(d, a, b)
        );
        rows
    }};
}

/// A rank-(2, 2) source `[v, w'] ← [v, w']` with a dual leg on each side,
/// as f64.
macro_rules! source {
    ($runtime:expr, $v:expr, $w:expr, $key:expr) => {{
        let key = $key;
        let w = $w.try_dual().unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($runtime, [&$v, &w], [&$v, &w], |trees, index| {
                value(&key(trees), index)
            })
            .unwrap();
        source
    }};
}

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let rule = Arc::new(U1FusionRule);
    (
        GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 3),
                (U1Irrep::new(1), 2),
            ],
        )
        .unwrap(),
        GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap(),
    )
}

/// `(2j, k)` per sector: nontrivial, distinct degeneracies.
const SU2_SECTORS: [(usize, usize); 3] = [(0, 2), (1, 3), (2, 2)];

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
    )
    .unwrap()
}

#[cfg(feature = "racah-generated")]
fn sun2_leg() -> GradedSpace<tenet::sector::SUNFusionRule> {
    GradedSpace::try_new(
        Arc::new(tenet::sector::SUNFusionRule::new(2).unwrap()),
        SU2_SECTORS.map(|(s, k)| (vec![s as i64], k)),
    )
    .unwrap()
}

fn fz2_legs() -> (
    GradedSpace<FermionParityFusionRule>,
    GradedSpace<FermionParityFusionRule>,
) {
    let rule = Arc::new(FermionParityFusionRule);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)]).unwrap(),
        GradedSpace::try_new(rule, [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)]).unwrap(),
    )
}

fn debug_key<S: std::fmt::Debug>(trees: &BlockFusionTrees<S>) -> String {
    format!("{trees:?}")
}

macro_rules! both_lanes {
    ($runtime:expr, $v:expr, $w:expr, $key:expr) => {{
        let f64_rows = {
            let source = source!($runtime, $v, $w, $key);
            transform_cells!($runtime, source, $key, f64)
        };
        let c64_rows = {
            // `1 + 2i` makes a lost or doubled conjugation visible.
            let source = source!($runtime, $v, $w, $key)
                .convert::<Complex64>()
                .scale(Complex64::new(1.0, 2.0));
            transform_cells!($runtime, source, $key, Complex64)
        };
        (f64_rows, c64_rows)
    }};
}

#[test]
fn multiplicity_free_real_symbol_transform_into_is_alpha_eager_plus_beta_destination() {
    let runtime = host_runtime();
    let (v, w) = u1_legs();
    both_lanes!(&runtime, v, w, debug_key::<U1Irrep>);
    let (v, w) = fz2_legs();
    both_lanes!(&runtime, v, w, debug_key::<Z2Irrep>);
    let v = su2_leg();
    both_lanes!(&runtime, v, v, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
}

#[test]
fn multiplicity_free_complex_symbol_transform_into_is_alpha_eager_plus_beta_destination() {
    let runtime = host_runtime();
    let leg = GradedSpace::try_new(
        Arc::new(FibonacciFusionRule),
        [(FibonacciSector::Vacuum, 2), (FibonacciSector::Tau, 3)],
    )
    .unwrap();
    let key = debug_key::<FibonacciSector>;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            Complex64::new(
                value(&key(trees), index),
                value(&format!("im{}", key(trees)), index),
            )
        })
        .unwrap();
    let mut rows: Rows = Vec::new();
    cell!(
        rows,
        &runtime,
        source,
        key,
        Complex64,
        "braid",
        |t| t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]),
        |d, a, b| source.braid_into(&[1, 0], &[3, 2], &[0, 1, 2, 3], d, a, b)
    );
    cell!(
        rows,
        &runtime,
        source,
        key,
        Complex64,
        "transpose",
        |t| t.transpose(&[1, 3], &[0, 2]),
        |d, a, b| source.transpose_into(&[1, 3], &[0, 2], d, a, b)
    );
    cell!(
        rows,
        &runtime,
        source,
        key,
        Complex64,
        "repartition",
        |t| t.repartition(1),
        |d, a, b| source.repartition_into(d, a, b)
    );
    assert_eq!(rows.len(), 3 * BETAS.len());

    // Anyonic: `permute` is refused alike, and the destination is untouched.
    let eager = source.permute(&[2, 0], &[1, 3]).unwrap_err();
    let mut destination = source.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]).unwrap();
    let before = dense(&destination);
    let into = source
        .permute_into(
            &[1, 0],
            &[3, 2],
            &mut destination,
            Complex64::alpha(),
            Complex64::lift(0.5),
        )
        .unwrap_err();
    assert_eq!(
        std::mem::discriminant(&into),
        std::mem::discriminant(&eager),
        "{into:?} vs {eager:?}"
    );
    assert_eq!(dense(&destination), before);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_transform_into_is_alpha_eager_plus_beta_destination_and_matches_multiplicity_free(
) {
    let runtime = host_runtime();
    let mf = su2_leg();
    let checked = sun2_leg();
    let (mf_f64, mf_c64) = both_lanes!(&runtime, mf, mf, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
    let (checked_f64, checked_c64) =
        both_lanes!(&runtime, checked, checked, |t: &BlockFusionTrees<
            Vec<i64>,
        >| twice(t, sun_spin));
    for (mf, checked) in [(mf_f64, checked_f64), (mf_c64, checked_c64)] {
        assert_eq!(mf.len(), checked.len());
        for ((name, mf), (_, checked)) in mf.iter().zip(&checked) {
            assert_eq!(
                mf.keys().collect::<Vec<_>>(),
                checked.keys().collect::<Vec<_>>(),
                "{name}"
            );
            for (key, mf) in mf {
                for (&c, &m) in checked[key].iter().zip(mf) {
                    assert!(
                        close(c, m),
                        "{name} {key}: checked {c} vs multiplicity-free {m}"
                    );
                }
            }
        }
    }

    // SU(3): `8 ⊗ 8 → 8` carries outer multiplicity two.
    let v = GradedSpace::try_new(
        Arc::new(tenet::sector::SUNFusionRule::new(3).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    both_lanes!(&runtime, v, v, debug_key::<Vec<i64>>);
}

/// A warm checked `_into` allocates no `n_C` result: it costs the eager
/// op's staging minus the payload, and makes no allocation of the payload's
/// size.
#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_warm_transform_into_allocates_no_result_payload() {
    let _serial = counting_alloc::serial();
    let runtime = host_runtime();
    let v = sun2_leg();
    let source = source!(&runtime, v, v, |t: &BlockFusionTrees<Vec<i64>>| twice(
        t, sun_spin
    ));
    let mut destination = source.permute(&[2, 0], &[1, 3]).unwrap();
    let payload_bytes = std::mem::size_of_val(destination.dense_data().unwrap());
    for _ in 0..2 {
        drop(source.permute(&[2, 0], &[1, 3]).unwrap());
        source
            .permute_into(&[2, 0], &[1, 3], &mut destination, 1.0, 0.5)
            .unwrap();
    }
    let (_, eager) = counting_alloc::measure(|| drop(source.permute(&[2, 0], &[1, 3]).unwrap()));
    let (_, into) = counting_alloc::measure_matching(payload_bytes..=usize::MAX, || {
        source
            .permute_into(&[2, 0], &[1, 3], &mut destination, 1.0, 0.5)
            .unwrap()
    });
    assert_eq!(into.matched_calls, 0, "{into:?}");
    assert!(
        eager.bytes >= into.bytes + payload_bytes as u64,
        "eager {eager:?} vs into {into:?}, payload {payload_bytes}"
    );
}
