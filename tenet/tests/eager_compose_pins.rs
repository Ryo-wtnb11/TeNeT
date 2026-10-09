//! Characterization pins of Host eager `compose` (#2131), recorded on the
//! base `65cf0057` before eager compose moved onto the shared `plan_compose`.
//!
//! Over U(1), SU(2), fZ2xU(1) and the non-symmetric braiding probes, f64 and
//! c64, owned and lazy-adjoint operands compose with dyadic entries. Every
//! product and partial sum is exact, so the value digest (±0 folded) does not
//! depend on the dense kernel's summation order. [`DIGESTS`] pins every
//! value; [`ERRORS`] pins the `Debug` form of every rejected composition.
//! The non-canonical (irregular-core) tilings are pinned in the lib test
//! `typed::contract_stacking_tests`, where an expert layout can be built.

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

#[path = "braiding_probe/mod.rs"]
mod braiding_probe;

use std::sync::Arc;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use tenet::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use tenet::typed::{GradedSpace, Runtime, TensorMap};

use braiding_probe::{ProbeSector, RealBraidingProbe};
use common::Payload;
use prepared::{fz2u1_legs, members, su2_legs, u1_legs};

fn digest<R, D>(tensor: &TensorMap<R, D>) -> u64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    let data = tensor.materialize().unwrap();
    for value in data.dense_data().unwrap() {
        let (re, im) = value.parts();
        for part in [re, im] {
            let bits = if part == 0.0 { 0 } else { part.to_bits() };
            for byte in bits.to_le_bytes() {
                state ^= u64::from(byte);
                state = state.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    state
}

fn record<R, D>(
    label: &str,
    (v, w): (GradedSpace<R>, GradedSpace<R>),
    rows: &mut Vec<(String, u64)>,
    errors: &mut Vec<(String, String)>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let key = |case: &str| format!("{label} {} {case}", D::NAME);
    // A: V ⊗ V ← W, B: W ← V, C: V ← V.
    let a = members::<R, D>(&runtime, &[&v, &v], &[&w], 1, 1).remove(0);
    let b = members::<R, D>(&runtime, &[&w], &[&v], 1, 2).remove(0);
    let c = members::<R, D>(&runtime, &[&v], &[&v], 1, 3).remove(0);
    let cases: [(&str, TensorMap<R, D>); 6] = [
        ("A*B", a.compose(&b).unwrap()),
        (
            "B'*A' lazy",
            b.adjoint().unwrap().compose(&a.adjoint().unwrap()).unwrap(),
        ),
        (
            "B'*A' view",
            b.adjoint().unwrap().compose(a.adjoint_view()).unwrap(),
        ),
        (
            "B'*A' mixed",
            b.adjoint()
                .unwrap()
                .compose(&a.adjoint().unwrap().materialize().unwrap())
                .unwrap(),
        ),
        ("C*C'", c.compose(c.adjoint_view()).unwrap()),
        ("(A*B)*C", a.compose(&b).unwrap().compose(&c).unwrap()),
    ];
    for (case, tensor) in cases {
        rows.push((key(case), digest(&tensor)));
    }
    let other = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign = members::<R, D>(&other, &[&w], &[&v], 1, 2).remove(0);
    for (case, result) in [
        ("runtime", a.compose(&foreign).err()),
        ("rank", a.compose(&a).err()),
        ("space", a.compose(&c).err()),
        ("adjoint space", b.compose(a.adjoint_view()).err()),
    ] {
        errors.push((key(case), format!("{:?}", result.expect("rejected"))));
    }
}

fn probe_legs<const ANYONIC: bool>() -> (
    GradedSpace<RealBraidingProbe<ANYONIC>>,
    GradedSpace<RealBraidingProbe<ANYONIC>>,
) {
    let rule = Arc::new(RealBraidingProbe::<ANYONIC>);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(ProbeSector, 2)]).unwrap(),
        GradedSpace::try_new(rule, [(ProbeSector, 3)]).unwrap(),
    )
}

const DIGESTS: &[(&str, u64)] = &[
    ("U1 f64 A*B", 0x102ba294663aa663),
    ("U1 f64 B'*A' lazy", 0x2228495290a47463),
    ("U1 f64 B'*A' view", 0x2228495290a47463),
    ("U1 f64 B'*A' mixed", 0x2228495290a47463),
    ("U1 f64 C*C'", 0x92702bf7fbd4f608),
    ("U1 f64 (A*B)*C", 0xd8e2366830f7bfaf),
    ("U1 c64 A*B", 0x5928f0e97281eb35),
    ("U1 c64 B'*A' lazy", 0x1c5c74590e26c84d),
    ("U1 c64 B'*A' view", 0x1c5c74590e26c84d),
    ("U1 c64 B'*A' mixed", 0x1c5c74590e26c84d),
    ("U1 c64 C*C'", 0x6a7871463397660d),
    ("U1 c64 (A*B)*C", 0xf7183d5b1c96c13a),
    ("SU2 f64 A*B", 0xffdf1e81120b6272),
    ("SU2 f64 B'*A' lazy", 0x664732df0b7f28b2),
    ("SU2 f64 B'*A' view", 0x664732df0b7f28b2),
    ("SU2 f64 B'*A' mixed", 0x664732df0b7f28b2),
    ("SU2 f64 C*C'", 0x6daa48272c4d0bbf),
    ("SU2 f64 (A*B)*C", 0x73363baf8c361ecf),
    ("SU2 c64 A*B", 0xa35d3661edd39453),
    ("SU2 c64 B'*A' lazy", 0x4645d6a9d81b1f0b),
    ("SU2 c64 B'*A' view", 0x4645d6a9d81b1f0b),
    ("SU2 c64 B'*A' mixed", 0x4645d6a9d81b1f0b),
    ("SU2 c64 C*C'", 0x467e907f95d54059),
    ("SU2 c64 (A*B)*C", 0xa0a7d4ed1994b182),
    ("fZ2xU1 f64 A*B", 0x76483395b2bf3e5a),
    ("fZ2xU1 f64 B'*A' lazy", 0x737bd8115230629a),
    ("fZ2xU1 f64 B'*A' view", 0x737bd8115230629a),
    ("fZ2xU1 f64 B'*A' mixed", 0x737bd8115230629a),
    ("fZ2xU1 f64 C*C'", 0x6daa48272c4d0bbf),
    ("fZ2xU1 f64 (A*B)*C", 0xa2ae5f5de6917ea6),
    ("fZ2xU1 c64 A*B", 0xe32aebeead95479d),
    ("fZ2xU1 c64 B'*A' lazy", 0x7b1dddef49d21a4d),
    ("fZ2xU1 c64 B'*A' view", 0x7b1dddef49d21a4d),
    ("fZ2xU1 c64 B'*A' mixed", 0x7b1dddef49d21a4d),
    ("fZ2xU1 c64 C*C'", 0x467e907f95d54059),
    ("fZ2xU1 c64 (A*B)*C", 0x05b3c8a06e21a88f),
    ("anyonic f64 A*B", 0x84e39fb9c8f59e3f),
    ("anyonic f64 B'*A' lazy", 0x7e5d02091e0da3d7),
    ("anyonic f64 B'*A' view", 0x7e5d02091e0da3d7),
    ("anyonic f64 B'*A' mixed", 0x7e5d02091e0da3d7),
    ("anyonic f64 C*C'", 0x624c08cac1a4429d),
    ("anyonic f64 (A*B)*C", 0xc9536cc41cd2bd90),
    ("unbraided f64 A*B", 0x84e39fb9c8f59e3f),
    ("unbraided f64 B'*A' lazy", 0x7e5d02091e0da3d7),
    ("unbraided f64 B'*A' view", 0x7e5d02091e0da3d7),
    ("unbraided f64 B'*A' mixed", 0x7e5d02091e0da3d7),
    ("unbraided f64 C*C'", 0x624c08cac1a4429d),
    ("unbraided f64 (A*B)*C", 0xc9536cc41cd2bd90),
];

const ERRORS: &[(&str, &str)] = &[
    ("U1 f64 runtime", "RuntimeMismatch"),
    (
        "U1 f64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "U1 f64 space",
        "Operation(Core(DimensionMismatch { expected: 2, actual: 3 }))",
    ),
    (
        "U1 f64 adjoint space",
        "Operation(Core(DimensionMismatch { expected: 3, actual: 2 }))",
    ),
    ("U1 c64 runtime", "RuntimeMismatch"),
    (
        "U1 c64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "U1 c64 space",
        "Operation(Core(DimensionMismatch { expected: 2, actual: 3 }))",
    ),
    (
        "U1 c64 adjoint space",
        "Operation(Core(DimensionMismatch { expected: 3, actual: 2 }))",
    ),
    ("SU2 f64 runtime", "RuntimeMismatch"),
    (
        "SU2 f64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "SU2 f64 space",
        "Operation(Core(DimensionMismatch { expected: 2, actual: 3 }))",
    ),
    (
        "SU2 f64 adjoint space",
        "Operation(Core(DimensionMismatch { expected: 3, actual: 2 }))",
    ),
    ("SU2 c64 runtime", "RuntimeMismatch"),
    (
        "SU2 c64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "SU2 c64 space",
        "Operation(Core(DimensionMismatch { expected: 2, actual: 3 }))",
    ),
    (
        "SU2 c64 adjoint space",
        "Operation(Core(DimensionMismatch { expected: 3, actual: 2 }))",
    ),
    ("fZ2xU1 f64 runtime", "RuntimeMismatch"),
    (
        "fZ2xU1 f64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "fZ2xU1 f64 space",
        "Operation(Core(DimensionMismatch { expected: 2, actual: 3 }))",
    ),
    (
        "fZ2xU1 f64 adjoint space",
        "Operation(Core(DimensionMismatch { expected: 3, actual: 2 }))",
    ),
    ("fZ2xU1 c64 runtime", "RuntimeMismatch"),
    (
        "fZ2xU1 c64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "fZ2xU1 c64 space",
        "Operation(Core(DimensionMismatch { expected: 2, actual: 3 }))",
    ),
    (
        "fZ2xU1 c64 adjoint space",
        "Operation(Core(DimensionMismatch { expected: 3, actual: 2 }))",
    ),
    ("anyonic f64 runtime", "RuntimeMismatch"),
    (
        "anyonic f64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "anyonic f64 space",
        "Operation(Core(LegDegeneracyMismatch { sector: SectorId(0), expected: 3, actual: 2 }))",
    ),
    (
        "anyonic f64 adjoint space",
        "Operation(Core(LegDegeneracyMismatch { sector: SectorId(0), expected: 2, actual: 3 }))",
    ),
    ("unbraided f64 runtime", "RuntimeMismatch"),
    (
        "unbraided f64 rank",
        "Operation(ContractAxisCountMismatch { lhs: 1, rhs: 2 })",
    ),
    (
        "unbraided f64 space",
        "Operation(Core(LegDegeneracyMismatch { sector: SectorId(0), expected: 3, actual: 2 }))",
    ),
    (
        "unbraided f64 adjoint space",
        "Operation(Core(LegDegeneracyMismatch { sector: SectorId(0), expected: 2, actual: 3 }))",
    ),
];

#[test]
fn host_eager_compose_values_and_errors_are_pinned() {
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    record::<_, f64>("U1", u1_legs(), &mut rows, &mut errors);
    record::<_, Complex64>("U1", u1_legs(), &mut rows, &mut errors);
    record::<_, f64>("SU2", su2_legs(), &mut rows, &mut errors);
    record::<_, Complex64>("SU2", su2_legs(), &mut rows, &mut errors);
    record::<_, f64>("fZ2xU1", fz2u1_legs(), &mut rows, &mut errors);
    record::<_, Complex64>("fZ2xU1", fz2u1_legs(), &mut rows, &mut errors);
    record::<_, f64>("anyonic", probe_legs::<true>(), &mut rows, &mut errors);
    record::<_, f64>("unbraided", probe_legs::<false>(), &mut rows, &mut errors);
    let rows: Vec<(&str, u64)> = rows.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(rows, DIGESTS);
    let errors: Vec<(&str, &str)> = errors
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    assert_eq!(errors, ERRORS);
}
