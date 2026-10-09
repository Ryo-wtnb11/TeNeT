//! Trace permutation columns through cache 4 (#2072): multiplicity-free
//! output bits pinned to the pre-cache revision, cold and warm.

use super::*;
use tenet_core::{clear_structure_caches, FermionParityFusionRule};

type FpSu2Rule = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

/// FNV-1a over 64-bit words: a bit-exact fingerprint of terms and outputs.
pub(super) fn fingerprint(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// `(terms, f64 output, Complex64 output)` fingerprints of one MF trace.
type Prints = [u64; 3];

fn real(i: usize) -> f64 {
    ((i * 7919 + 13) % 1009) as f64 / 37.0 - 13.0
}

fn complex(i: usize) -> Complex64 {
    Complex64::new(real(i), real(i + 503) / 3.0)
}

fn mf_prints<R>(
    provider: Arc<R>,
    src_hom: FusionTreeHomSpace,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Prints
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + tenet_core::CheckedFusionAlgebra,
{
    let src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&provider),
        src_hom,
    )
    .unwrap();
    let dst_hom = crate::tensortrace_fusion_dyn_preflight_checked(&src, axes, dst_nout)
        .unwrap()
        .into_selected_homspace();
    let dst = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(provider, dst_hom)
        .unwrap();
    let structure =
        TensorTraceFusionStructure::<f64>::compile_fusion_dyn_checked(&dst, &src, axes).unwrap();
    let terms = fingerprint(structure.terms().iter().flat_map(|term| {
        [
            term.dst_block() as u64,
            term.src_block() as u64,
            term.coefficient().to_bits(),
        ]
    }));
    let len = src.space().required_len().unwrap();
    let real_data = (0..len).map(real).collect::<Vec<_>>();
    let complex_data = (0..len).map(complex).collect::<Vec<_>>();
    let real_out =
        crate::tensortrace_fusion_dyn_owned_checked(&dst, &src, &real_data, axes, 1.0).unwrap();
    let complex_out = crate::tensortrace_fusion_dyn_owned_checked(
        &dst,
        &src,
        &complex_data,
        axes,
        Complex64::new(1.0, 0.0),
    )
    .unwrap();
    [
        terms,
        fingerprint(real_out.iter().map(|value| value.to_bits())),
        fingerprint(
            complex_out
                .iter()
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()]),
        ),
    ]
}

/// Runs a case cold (after a reset) and warm; both must equal `expected`.
fn assert_cold_and_warm(name: &str, expected: Prints, prints: impl Fn() -> Prints) {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    clear_structure_caches();
    let cold = prints();
    let warm = prints();
    assert_eq!(cold, warm, "{name}: warm bits differ from cold");
    assert_eq!(
        cold, expected,
        "{name}: bits differ from the pinned revision"
    );
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [(0, 2), (1, 1), (2, 2)]
            .map(|(spin, degeneracy)| (SU2Irrep::from_twice_spin(spin).sector_id(), degeneracy)),
        false,
    )
}

fn su2_hom() -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(
        FusionProductSpace::new([su2_leg(), su2_leg()]),
        FusionProductSpace::new([su2_leg(), su2_leg()]),
    )
}

fn fp_su2_hom() -> FusionTreeHomSpace {
    let rule = FpSu2Rule::default();
    let leg = |dual| {
        SectorLeg::new(
            [(0, 0, 1), (1, 1, 2), (1, 2, 1)].map(|(parity, spin, degeneracy)| {
                (
                    rule.encode_component_ids(
                        SectorId::new(parity),
                        SU2Irrep::from_twice_spin(spin).sector_id(),
                    ),
                    degeneracy,
                )
            }),
            dual,
        )
    };
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    )
}

fn u1_hom() -> FusionTreeHomSpace {
    let leg = || {
        SectorLeg::new(
            [(-1, 1), (0, 2), (1, 1)]
                .map(|(charge, degeneracy)| (U1Irrep::new(charge).sector_id(), degeneracy)),
            false,
        )
    };
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    )
}

/// What: multiplicity-free trace terms and outputs keep the bits of
/// c7b82353/a04e59c2 (one `(p…, q…)` braid, the identity permutation, the
/// lazy adjoint, two pairs) for SU(2), the fermionic fZ2⊠SU(2) product
/// (twist factor) and U(1) (Unique), and a warm call equals a cold one.
#[test]
fn mf_trace_bits_match_pinned_revision_cold_and_warm() {
    let open = TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]);
    let identity = TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]);
    let adjoint = TensorTraceAxisSpec::new_with_conjugation(&[1, 3], &[0], &[2], true);
    let both = TensorTraceAxisSpec::new(&[], &[0, 1], &[2, 3]);
    let su2 = || Arc::new(SU2FusionRule);
    let fp = || Arc::new(FpSu2Rule::default());
    let u1 = || Arc::new(U1FusionRule);
    let cases: [(&str, Prints, Box<dyn Fn() -> Prints>); 10] = [
        (
            "su2 open",
            PIN_SU2_OPEN,
            Box::new(|| mf_prints(su2(), su2_hom(), open, 1)),
        ),
        (
            "su2 identity",
            PIN_SU2_IDENTITY,
            Box::new(|| mf_prints(su2(), su2_hom(), identity, 1)),
        ),
        (
            "su2 adjoint",
            PIN_SU2_ADJOINT,
            Box::new(|| mf_prints(su2(), su2_hom(), adjoint, 1)),
        ),
        (
            "su2 both",
            PIN_SU2_BOTH,
            Box::new(|| mf_prints(su2(), su2_hom(), both, 0)),
        ),
        (
            "fp⊠su2 open",
            PIN_FP_OPEN,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), open, 1)),
        ),
        (
            "fp⊠su2 identity",
            PIN_FP_IDENTITY,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), identity, 1)),
        ),
        (
            "fp⊠su2 adjoint",
            PIN_FP_ADJOINT,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), adjoint, 1)),
        ),
        (
            "fp⊠su2 both",
            PIN_FP_BOTH,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), both, 0)),
        ),
        (
            "u1 open",
            PIN_U1_OPEN,
            Box::new(|| mf_prints(u1(), u1_hom(), open, 1)),
        ),
        (
            "u1 adjoint",
            PIN_U1_ADJOINT,
            Box::new(|| mf_prints(u1(), u1_hom(), adjoint, 1)),
        ),
    ];
    for (name, expected, prints) in cases {
        assert_cold_and_warm(name, expected, prints);
    }
}

/// Recorded at a04e59c2 (before #2072).
const PIN_SU2_OPEN: Prints = [
    6797032591768898186,
    11542341682854181360,
    10205597355325690015,
];
const PIN_SU2_IDENTITY: Prints = [
    5946816043996689180,
    10064097200336381756,
    6544399261609154517,
];
const PIN_SU2_ADJOINT: Prints = [
    12583905454829699337,
    11542341682854181360,
    1152431112794283509,
];
const PIN_SU2_BOTH: Prints = [
    16340709039126675528,
    12194415398254016631,
    12541757064838289309,
];
const PIN_FP_OPEN: Prints = [
    14713718370923663374,
    7120500526032442709,
    14955366930349779627,
];
const PIN_FP_IDENTITY: Prints = [
    9319061474827989388,
    18065772038357404277,
    5050611406896117456,
];
const PIN_FP_ADJOINT: Prints = [
    15885020966606187352,
    5860062657676541310,
    9015904545330042487,
];
const PIN_FP_BOTH: Prints = [33104485184910472, 11925859689077087162, 5933484787948759124];
const PIN_U1_OPEN: Prints = [
    12180223929535698292,
    18178240109427535153,
    18316357646671316850,
];
const PIN_U1_ADJOINT: Prints = [
    5011798078153603172,
    12510737463361251689,
    7339927660487231306,
];
