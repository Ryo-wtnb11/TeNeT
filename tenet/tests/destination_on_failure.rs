//! The destination of a failing `*_into` is left untouched (#1551).
//!
//! Every malformed-input class of every linear destination operation —
//! runtime, rule, space, leg index, capability, destination representation,
//! aliasing and a shared destination — is called on a destination filled
//! with distinct sentinels, and the destination's bytes are compared bit
//! for bit before and after the failing call. The oracle is those bytes
//! alone, independent of every operation under test; each fixture also runs
//! one valid call on a fresh destination, so a rejection is caused by the one
//! malformed input and not by a broken fixture. Host tensors additionally
//! keep their payload pointer. `scale_assign` gets the same destination rule.
//! The device variants run with `--features cuda -- --ignored`.

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::{U1FusionRule, U1Irrep, ZNFusionRule};
use tenet::typed::{ContractSpec, Error, GradedSpace, Runtime, SectorSpectrum, TensorMap};

/// A fresh tensor on `like`'s legs filled with distinct sentinels,
/// alternating quiet NaNs with distinct finite values. Overwriting a NaN
/// changes its bits; a read-modify-write can keep a NaN's payload (IEEE
/// arithmetic propagates it), so the finite neighbours catch a `beta` pass.
macro_rules! sentinel {
    ($rt:expr, $like:expr) => {{
        let like = &$like;
        let (codomain, domain) = (like.codomain(), like.domain());
        let mut next = 0u64;
        TensorMap::<_, f64>::from_subblock_fn($rt, &codomain, &domain, |_, _| {
            next += 1;
            sentinel_value(next)
        })
        .unwrap()
    }};
}

/// Calls `$call` on `$dst`, expects an error matching `$err`, and checks
/// that the destination's legs and payload bits (and, on Host, its payload
/// pointer) are those from before the call.
macro_rules! rejects {
    ($snap:expr, $what:expr, $dst:expr, $err:pat, $call:expr) => {{
        let mut dst = $dst;
        let (codomain, domain) = (dst.codomain(), dst.domain());
        #[allow(clippy::redundant_closure_call)]
        let before = ($snap)(&dst);
        let result = call(&mut dst, $call);
        // `$err` is `_` where only the rejection, not its kind, is pinned.
        #[allow(clippy::redundant_pattern_matching)]
        let rejected = matches!(result, Err($err));
        assert!(
            rejected,
            "{}: expected {}, got {result:?}",
            $what,
            stringify!($err)
        );
        #[allow(clippy::redundant_closure_call)]
        let after = ($snap)(&dst);
        assert!(before == after, "{}: destination changed", $what);
        assert!(
            dst.codomain() == codomain && dst.domain() == domain,
            "{}: destination legs changed",
            $what
        );
    }};
}

fn sentinel_value(index: u64) -> f64 {
    if index.is_multiple_of(2) {
        f64::from_bits(0x7ff8_0000_0000_0000 | index)
    } else {
        std::f64::consts::PI * index as f64
    }
}

/// Fixes the destination type before the closure body is checked.
fn call<T, E>(destination: &mut T, f: impl FnOnce(&mut T) -> Result<(), E>) -> Result<(), E> {
    f(destination)
}

fn host_snap<R>(t: &TensorMap<R, f64>) -> (Vec<u64>, usize) {
    let data = t.dense_data().unwrap();
    (
        data.iter().map(|x| x.to_bits()).collect(),
        data.as_ptr() as usize,
    )
}

fn legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(-1), 2),
        ],
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

/// Every U(1) malformed-input class of every `*_into`. `$lift` puts a Host
/// tensor at the placement under test; `$snap` reads a destination's bytes.
macro_rules! u1_suite {
    ($rt:expr, $other_rt:expr, $lift:expr, $snap:expr) => {{
        let (rt, other_rt) = ($rt, $other_rt);
        let lift = $lift;
        let (v, w) = legs();
        let host_t = TensorMap::<_, f64>::rand_with_seed(rt, [&v, &w], [&v, &w], 1).unwrap();
        let host_rhs = TensorMap::<_, f64>::rand_with_seed(rt, [&v, &w], [&v], 2).unwrap();
        let (t, rhs) = (lift(&host_t), lift(&host_rhs));
        let lazy = t.adjoint().unwrap();
        let spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[0, 1],
            codomain: &[2, 0],
            domain: &[1],
        };
        let bad_spec = ContractSpec {
            lhs: &[2, 9],
            rhs: &[0, 1],
            codomain: &[2, 0],
            domain: &[1],
        };
        // Pairs `t`'s domain `v` with `rhs`'s codomain `w`: not mutually dual.
        let undual_spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[1, 0],
            codomain: &[2, 0],
            domain: &[1],
        };
        let fresh = |like: &TensorMap<_, f64>| lift(&sentinel!(rt, like));
        let foreign = |like: &TensorMap<_, f64>| lift(&sentinel!(other_rt, like));
        // Wrong space: every leg dualized.
        let wrong = |like: &TensorMap<_, f64>| {
            let dual = |legs: Vec<GradedSpace<_>>| {
                legs.iter()
                    .map(|leg| leg.try_dual().unwrap())
                    .collect::<Vec<_>>()
            };
            let (codomain, domain) = (dual(like.codomain()), dual(like.domain()));
            let mut next = 0u64;
            lift(
                &TensorMap::<_, f64>::from_subblock_fn(rt, &codomain, &domain, |_, _| {
                    next += 1;
                    sentinel_value(next)
                })
                .unwrap(),
            )
        };
        let permuted = host_t.permute(&[2, 0], &[1, 3]).unwrap();
        let braided = host_t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]).unwrap();
        let transposed = host_t.transpose(&[1, 3], &[0, 2]).unwrap();
        let repartitioned = host_t.repartition(1).unwrap();
        let traced = host_t.trace_pairs(&[(0, 2)]).unwrap();
        let contracted = host_t.contract(&host_rhs, &spec).unwrap();

        // The fixtures are valid: the unmodified call succeeds.
        t.permute_into(&[2, 0], &[1, 3], &mut fresh(&permuted), 1.0, 0.5)
            .unwrap();
        t.braid_into(
            &[1, 0],
            &[3, 2],
            &[0, 1, 2, 3],
            &mut fresh(&braided),
            1.0,
            0.5,
        )
        .unwrap();
        t.transpose_into(&[1, 3], &[0, 2], &mut fresh(&transposed), 1.0, 0.5)
            .unwrap();
        t.repartition_into(&mut fresh(&repartitioned), 1.0, 0.5)
            .unwrap();
        t.trace_pairs_into(&[(0, 2)], &mut fresh(&traced), 1.0, 0.5)
            .unwrap();
        t.contract_into(&rhs, &spec, &mut fresh(&contracted), 1.0, 0.5)
            .unwrap();
        t.axpby_into(&mut fresh(&host_t), 1.0, 0.5).unwrap();

        // A second handle on the destination's storage for the duration of
        // the call.
        macro_rules! shared {
            ($f:expr) => {
                |d: &mut _| {
                    let clone = d.clone();
                    #[allow(clippy::redundant_closure_call)]
                    let result = ($f)(d);
                    drop(clone);
                    result
                }
            };
        }

        // permute_into
        let op = "permute_into";
        rejects!(
            $snap,
            format!("{op} runtime"),
            foreign(&permuted),
            Error::RuntimeMismatch,
            |d: &mut _| t.permute_into(&[2, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} space"),
            wrong(&permuted),
            Error::InvalidArgument(_),
            |d: &mut _| t.permute_into(&[2, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} leg out of range"),
            fresh(&permuted),
            _,
            |d: &mut _| t.permute_into(&[4, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} leg repeated"),
            fresh(&permuted),
            _,
            |d: &mut _| t.permute_into(&[0, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} lazy source"),
            fresh(&permuted),
            _,
            |d: &mut _| lazy.permute_into(&[2, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} alias"),
            fresh(&host_t),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let source = d.clone();
                source.permute_into(&[0, 1], &[2, 3], d, 1.0, 0.5)
            }
        );
        rejects!(
            $snap,
            format!("{op} shared"),
            fresh(&permuted),
            Error::DestinationShared,
            shared!(|d: &mut _| t.permute_into(&[2, 0], &[1, 3], d, 1.0, 0.5))
        );

        // braid_into
        let op = "braid_into";
        let levels = [0, 1, 2, 3];
        rejects!(
            $snap,
            format!("{op} runtime"),
            foreign(&braided),
            Error::RuntimeMismatch,
            |d: &mut _| t.braid_into(&[1, 0], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} space"),
            wrong(&braided),
            Error::InvalidArgument(_),
            |d: &mut _| t.braid_into(&[1, 0], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} levels"),
            fresh(&braided),
            Error::InvalidArgument(_),
            |d: &mut _| t.braid_into(&[1, 0], &[3, 2], &levels[..3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} leg out of range"),
            fresh(&braided),
            _,
            |d: &mut _| t.braid_into(&[1, 7], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} lazy source"),
            fresh(&braided),
            _,
            |d: &mut _| lazy.braid_into(&[1, 0], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} shared"),
            fresh(&braided),
            Error::DestinationShared,
            shared!(|d: &mut _| t.braid_into(&[1, 0], &[3, 2], &levels, d, 1.0, 0.5))
        );

        // transpose_into
        let op = "transpose_into";
        rejects!(
            $snap,
            format!("{op} runtime"),
            foreign(&transposed),
            Error::RuntimeMismatch,
            |d: &mut _| t.transpose_into(&[1, 3], &[0, 2], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} space"),
            wrong(&transposed),
            Error::InvalidArgument(_),
            |d: &mut _| t.transpose_into(&[1, 3], &[0, 2], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} non-planar"),
            fresh(&transposed),
            _,
            |d: &mut _| t.transpose_into(&[0, 2], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} leg out of range"),
            fresh(&transposed),
            _,
            |d: &mut _| t.transpose_into(&[1, 8], &[0, 2], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} shared"),
            fresh(&transposed),
            Error::DestinationShared,
            shared!(|d: &mut _| t.transpose_into(&[1, 3], &[0, 2], d, 1.0, 0.5))
        );

        // repartition_into
        let op = "repartition_into";
        rejects!(
            $snap,
            format!("{op} runtime"),
            foreign(&repartitioned),
            Error::RuntimeMismatch,
            |d: &mut _| t.repartition_into(d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} space"),
            wrong(&repartitioned),
            Error::InvalidArgument(_),
            |d: &mut _| t.repartition_into(d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} rank"),
            fresh(&traced),
            Error::InvalidArgument(_),
            |d: &mut _| t.repartition_into(d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} lazy source"),
            fresh(&repartitioned),
            _,
            |d: &mut _| lazy.repartition_into(d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} shared"),
            fresh(&repartitioned),
            Error::DestinationShared,
            shared!(|d: &mut _| t.repartition_into(d, 1.0, 0.5))
        );

        // trace_pairs_into
        let op = "trace_pairs_into";
        rejects!(
            $snap,
            format!("{op} runtime"),
            foreign(&traced),
            Error::RuntimeMismatch,
            |d: &mut _| t.trace_pairs_into(&[(0, 2)], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} space"),
            wrong(&traced),
            Error::InvalidArgument(_),
            |d: &mut _| t.trace_pairs_into(&[(0, 2)], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} leg out of range"),
            fresh(&traced),
            _,
            |d: &mut _| t.trace_pairs_into(&[(0, 9)], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} leg repeated"),
            fresh(&traced),
            _,
            |d: &mut _| t.trace_pairs_into(&[(0, 0)], d, 1.0, 0.5)
        );
        // On the legs the pair would leave, so the duality check itself (in
        // the trace compile, after every destination check) rejects it.
        let undual_traced = TensorMap::<_, f64>::rand_with_seed(rt, [&w], [&v], 6).unwrap();
        rejects!(
            $snap,
            format!("{op} legs not dual"),
            fresh(&undual_traced),
            _,
            |d: &mut _| t.trace_pairs_into(&[(0, 3)], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} alias"),
            fresh(&host_t),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let source = d.clone();
                source.trace_pairs_into(&[], d, 1.0, 0.5)
            }
        );
        rejects!(
            $snap,
            format!("{op} shared"),
            fresh(&traced),
            Error::DestinationShared,
            shared!(|d: &mut _| t.trace_pairs_into(&[(0, 2)], d, 1.0, 0.5))
        );

        // contract_into
        let op = "contract_into";
        rejects!(
            $snap,
            format!("{op} runtime"),
            foreign(&contracted),
            Error::RuntimeMismatch,
            |d: &mut _| t.contract_into(&rhs, &spec, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} space"),
            wrong(&contracted),
            Error::InvalidArgument(_),
            |d: &mut _| t.contract_into(&rhs, &spec, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} leg out of range"),
            fresh(&contracted),
            _,
            |d: &mut _| t.contract_into(&rhs, &bad_spec, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} legs not dual"),
            fresh(&contracted),
            _,
            |d: &mut _| t.contract_into(&rhs, &undual_spec, d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} alias"),
            fresh(&contracted),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let lhs = d.clone();
                lhs.contract_into(&rhs, &spec, d, 1.0, 0.5)
            }
        );
        rejects!(
            $snap,
            format!("{op} shared"),
            fresh(&contracted),
            Error::DestinationShared,
            shared!(|d: &mut _| t.contract_into(&rhs, &spec, d, 1.0, 0.5))
        );

        // axpby_into
        let op = "axpby_into";
        rejects!(
            $snap,
            format!("{op} runtime"),
            foreign(&host_t),
            Error::RuntimeMismatch,
            |d: &mut _| t.axpby_into(d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} space"),
            wrong(&host_t),
            Error::InvalidArgument(_),
            |d: &mut _| t.axpby_into(d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            format!("{op} alias"),
            fresh(&host_t),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let source = d.clone();
                source.axpby_into(d, 1.0, 0.5)
            }
        );
        rejects!(
            $snap,
            format!("{op} shared"),
            fresh(&host_t),
            Error::DestinationShared,
            shared!(|d: &mut _| t.axpby_into(d, 1.0, 0.5))
        );
    }};
}

/// Z2 sources against a Z3 destination: the same Rust type, another rule.
macro_rules! rule_suite {
    ($rt:expr, $lift:expr, $snap:expr) => {{
        let rt = $rt;
        let lift = $lift;
        let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
        let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
        let a =
            GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2), (z2.irrep(1), 1)]).unwrap();
        let b =
            GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 2), (z3.irrep(1), 1)]).unwrap();
        let host_t = TensorMap::<_, f64>::rand_with_seed(rt, [&a, &a], [&a, &a], 3).unwrap();
        let t = lift(&host_t);
        let rank2 = |leg: &GradedSpace<ZNFusionRule>| {
            lift(&sentinel!(
                rt,
                TensorMap::<_, f64>::rand_with_seed(rt, [leg], [leg], 4).unwrap()
            ))
        };
        let rank4 = |leg: &GradedSpace<ZNFusionRule>| {
            lift(&sentinel!(
                rt,
                TensorMap::<_, f64>::rand_with_seed(rt, [leg, leg], [leg, leg], 5).unwrap()
            ))
        };
        let spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[0, 1],
            codomain: &[0, 1],
            domain: &[2, 3],
        };
        rejects!(
            $snap,
            "permute_into rule",
            rank4(&b),
            Error::RuleMismatch,
            |d: &mut _| t.permute_into(&[0, 1], &[2, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            "braid_into rule",
            rank4(&b),
            Error::RuleMismatch,
            |d: &mut _| t.braid_into(&[0, 1], &[2, 3], &[0, 1, 2, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            "transpose_into rule",
            rank4(&b),
            Error::RuleMismatch,
            |d: &mut _| t.transpose_into(&[0, 1], &[2, 3], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            "repartition_into rule",
            rank4(&b),
            Error::RuleMismatch,
            |d: &mut _| t.repartition_into(d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            "trace_pairs_into rule",
            rank2(&b),
            Error::RuleMismatch,
            |d: &mut _| t.trace_pairs_into(&[(0, 2)], d, 1.0, 0.5)
        );
        rejects!(
            $snap,
            "contract_into rule",
            rank4(&b),
            Error::RuleMismatch,
            |d: &mut _| t.contract_into(&t, &spec, d, 1.0, 0.5)
        );
        // `axpby_into` compares the operand spaces, which carry the rule.
        rejects!(
            $snap,
            "axpby_into rule",
            rank4(&b),
            Error::InvalidArgument(_),
            |d: &mut _| t.axpby_into(d, 1.0, 0.5)
        );
        // The fixture is otherwise valid.
        t.contract_into(&t, &spec, &mut rank4(&a), 1.0, 0.5)
            .unwrap();
        t.trace_pairs_into(&[(0, 2)], &mut rank2(&a), 1.0, 0.5)
            .unwrap();
    }};
}

#[test]
fn every_host_into_leaves_a_rejected_destination_bit_identical() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let other = Runtime::builder().dense_threads(1).build().unwrap();
    let lift = |x: &TensorMap<U1FusionRule, f64>| x.materialize().unwrap();
    u1_suite!(&rt, &other, lift, host_snap);
    let lift = |x: &TensorMap<ZNFusionRule, f64>| x.materialize().unwrap();
    rule_suite!(&rt, lift, host_snap);
}

/// Host-only representation and capability classes: a compact or lazy
/// destination and a compact source the operation cannot take. (A
/// non-symmetric braiding provider cannot name `contract_into` at all: the
/// facade's trait bounds exclude it at compile time.)
#[test]
fn host_capability_and_representation_rejections_leave_the_destination_bit_identical() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, _) = legs();
    let t = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&v], 1).unwrap();
    let diag_snap = |d: &TensorMap<U1FusionRule, f64>| {
        d.diagview()
            .unwrap()
            .iter()
            .flat_map(|entry| entry.values.iter().map(|x| x.to_bits()))
            .collect::<Vec<_>>()
    };
    let compact = || {
        TensorMap::<_, f64>::diagonal(
            &rt,
            &v,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![f64::NAN, 2.0],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![3.0],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(-1),
                    values: vec![-0.0, 5.0],
                },
            ],
        )
        .unwrap()
    };
    // A compact diagonal destination on the result's space.
    rejects!(
        diag_snap,
        "permute_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.permute_into(&[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "braid_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.braid_into(&[0], &[1], &[0, 1], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "transpose_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.transpose_into(&[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "repartition_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.repartition_into(d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "trace_pairs_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.trace_pairs_into(&[], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "axpby_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.axpby_into(d, 1.0, 0.5)
    );
    let unit = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    rejects!(
        diag_snap,
        "contract_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.contract_into(&t, &unit, d, 1.0, 0.5)
    );

    // A lazy-adjoint destination: its parent's bytes are the destination's.
    let parent = sentinel!(&rt, t);
    let lazy_snap = |_: &TensorMap<U1FusionRule, f64>| host_snap(&parent);
    rejects!(
        lazy_snap,
        "permute_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.permute_into(&[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "repartition_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.repartition_into(d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "trace_pairs_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.trace_pairs_into(&[], d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "axpby_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.axpby_into(d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "contract_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.contract_into(&t, &unit, d, 1.0, 0.5)
    );

    // A compact source where the operation has no compact arm.
    let s = compact();
    // A rank-0 result has no leg to build a sentinel on; its own value is
    // the reference.
    let scalar = t.trace_pairs(&[(0, 1)]).unwrap().materialize().unwrap();
    rejects!(
        host_snap,
        "trace_pairs_into compact source",
        scalar,
        Error::Unsupported { .. },
        |d: &mut _| s.trace_pairs_into(&[(0, 1)], d, 1.0, 0.5)
    );
    rejects!(
        host_snap,
        "permute_into compact source",
        sentinel!(&rt, t),
        Error::InvalidArgument(_),
        |d: &mut _| s.permute_into(&[0], &[1], d, 1.0, 0.5)
    );
}

/// `scale_assign` takes the destination rule: in place on the receiver's own
/// storage, `DestinationShared` (receiver untouched) when it is shared.
#[test]
fn scale_assign_is_in_place_or_refuses_a_shared_receiver() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = legs();
    let factor = Complex64::new(0.5, -2.0);

    // Dense: every element times the factor, the payload pointer kept.
    let original = TensorMap::<_, Complex64>::rand_with_seed(&rt, [&v], [&v, &w], 1).unwrap();
    let mut dense = original.materialize().unwrap();
    let pointer = dense.dense_data().unwrap().as_ptr();
    dense.scale_assign(factor).unwrap();
    assert_eq!(dense.dense_data().unwrap().as_ptr(), pointer);
    let expected: Vec<_> = original
        .dense_data()
        .unwrap()
        .iter()
        .map(|&x| x * factor)
        .collect();
    assert_eq!(dense.dense_data().unwrap(), expected.as_slice());

    // Shared: refused, both handles unchanged.
    let clone = dense.clone();
    let before = dense.dense_data().unwrap().to_vec();
    assert_eq!(dense.scale_assign(factor), Err(Error::DestinationShared));
    assert_eq!(dense.dense_data().unwrap(), before.as_slice());
    assert_eq!(clone.dense_data().unwrap(), before.as_slice());
    drop(clone);

    // Compact diagonal: scaled in place, still compact.
    let values = [vec![1.0, -2.0], vec![3.0], vec![0.25, 5.0]];
    let sectors = [U1Irrep::new(0), U1Irrep::new(1), U1Irrep::new(-1)];
    let spectra = || {
        sectors
            .iter()
            .zip(&values)
            .map(|(&sector, values)| SectorSpectrum {
                sector,
                values: values.iter().map(|&x| Complex64::new(x, 0.0)).collect(),
            })
    };
    let mut compact = TensorMap::<_, Complex64>::diagonal(&rt, &v, spectra()).unwrap();
    compact.scale_assign(factor).unwrap();
    let got = compact.diagview().unwrap();
    assert_eq!(got.len(), sectors.len());
    for (&sector, values) in sectors.iter().zip(&values) {
        let entry = got.iter().find(|entry| entry.sector == sector).unwrap();
        let want: Vec<_> = values
            .iter()
            .map(|&x| Complex64::new(x, 0.0) * factor)
            .collect();
        assert_eq!(entry.values, want);
    }
    let clone = compact.clone();
    assert_eq!(compact.scale_assign(factor), Err(Error::DestinationShared));
    assert_eq!(compact.diagview().unwrap(), got);
    drop(clone);

    // Lazy adjoint: stays lazy; its logical values are `factor * t^H`.
    let parent = TensorMap::<_, Complex64>::rand_with_seed(&rt, [&v], [&v, &w], 2).unwrap();
    let logical = parent.adjoint().unwrap().materialize().unwrap();
    let mut lazy = parent.adjoint().unwrap();
    // The parent handle still shares the storage.
    let before = parent.dense_data().unwrap().to_vec();
    assert_eq!(lazy.scale_assign(factor), Err(Error::DestinationShared));
    assert_eq!(parent.dense_data().unwrap(), before.as_slice());
    drop(parent);
    lazy.scale_assign(factor).unwrap();
    assert!(lazy.dense_data().is_err(), "a lazy adjoint stays lazy");
    let got = lazy.materialize().unwrap();
    let want: Vec<_> = logical
        .dense_data()
        .unwrap()
        .iter()
        .map(|&x| x * factor)
        .collect();
    for (got, want) in got.dense_data().unwrap().iter().zip(&want) {
        assert!(
            (got - want).norm() <= 1e-14 * want.norm().max(1.0),
            "{got} != {want}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn every_device_into_leaves_a_rejected_destination_bit_identical() {
    use tenet::typed::CudaStorage;

    let rt = Runtime::builder().cuda(0).build().unwrap();
    let other = Runtime::builder().cuda(0).build().unwrap();
    fn device_snap<R>(t: &TensorMap<R, f64, CudaStorage<f64>>) -> Vec<u64> {
        t.to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }
    let lift = |x: &TensorMap<U1FusionRule, f64>| {
        // A tensor built on `other` goes to `other`'s device context.
        x.to_cuda().unwrap()
    };
    u1_suite!(&rt, &other, lift, device_snap);
    let lift = |x: &TensorMap<ZNFusionRule, f64>| x.to_cuda().unwrap();
    rule_suite!(&rt, lift, device_snap);

    // Device capability boundary: a lazy-adjoint `axpby_into` source.
    let (v, w) = legs();
    let t = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&v, &w], 1).unwrap();
    let lazy = t.to_cuda().unwrap().adjoint().unwrap();
    let like = t.adjoint().unwrap().materialize().unwrap();
    rejects!(
        device_snap,
        "device axpby_into lazy source",
        sentinel!(&rt, like).to_cuda().unwrap(),
        Error::UnsupportedOnDevice(_),
        |d: &mut _| lazy.axpby_into(d, 1.0, 0.5)
    );

    // A lazy-adjoint device destination: its parent's bytes are its own.
    let square = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&v], 2).unwrap();
    let device_square = square.to_cuda().unwrap();
    let parent = sentinel!(&rt, square).to_cuda().unwrap();
    let lazy_snap = |_: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>| device_snap(&parent);
    let unit = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    rejects!(
        lazy_snap,
        "device permute_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.permute_into(&[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "device trace_pairs_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.trace_pairs_into(&[], d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "device axpby_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.axpby_into(d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "device contract_into lazy dst",
        parent.adjoint().unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.contract_into(&device_square, &unit, d, 1.0, 0.5)
    );
}
