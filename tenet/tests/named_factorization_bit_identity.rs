//! #1541: the named factorization results carry exactly the factors the tuple
//! API returned. Each result is destructured back into the old tuple order and
//! fingerprinted twice. The structural fingerprint (legs, block order, storage
//! class, payload length) is portable and checked everywhere. The exact
//! fingerprint adds every stored value in round-trip `Debug` form, which
//! identifies each `f64` bit pattern; since the dense kernels pick SIMD paths
//! by CPU, its expected values hold only for the recording host class (macOS
//! aarch64). All expected values were recorded with the tuple API on
//! `origin/main` a2ce6c49, single-threaded dense executor, `cpu-faer`.

use num_complex::Complex64;
use tenet::core::{
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::typed::{
    Eig, Eigh, GradedSpace, LeftPolar, Lq, NetworkReuseClass, Qr, RightPolar, Runtime, Svd,
    TensorMap,
};

// The old tuple order of every result.
fn svd<T>(x: Svd<T>) -> (T, T, T) {
    (x.u, x.s, x.vh)
}
fn qr<T>(x: Qr<T>) -> (T, T) {
    (x.q, x.r)
}
fn lq<T>(x: Lq<T>) -> (T, T) {
    (x.l, x.q)
}
fn eigh<T>(x: Eigh<T>) -> (T, T) {
    (x.d, x.v)
}
fn eig<T>(x: Eig<T>) -> (T, T) {
    (x.d, x.v)
}
fn left_polar<T>(x: LeftPolar<T>) -> (T, T) {
    (x.w, x.p)
}
fn right_polar<T>(x: RightPolar<T>) -> (T, T) {
    (x.p, x.wh)
}

fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

macro_rules! factor {
    ($t:expr) => {{
        let t = $t;
        let legs = |legs: Vec<GradedSpace<_>>| {
            legs.iter()
                .map(|leg| {
                    let sectors = leg.sectors().unwrap();
                    format!("{sectors:?}{:?}{}", leg.degeneracies(), leg.is_dual())
                })
                .collect::<Vec<_>>()
        };
        let blocks = (0..t.block_count())
            .map(|i| format!("{:?}", t.block_fusion_trees(i).unwrap().coupled()))
            .collect::<Vec<_>>();
        let (codomain, domain) = (legs(t.codomain()), legs(t.domain()));
        let compact = matches!(t.network_reuse_class(false), NetworkReuseClass::Compact);
        let data = t.data();
        (
            format!(
                "{codomain:?}<-{domain:?}|{blocks:?}|compact={compact}|len={}",
                data.len()
            ),
            format!("{codomain:?}<-{domain:?}|{blocks:?}|compact={compact}|{data:?}"),
        )
    }};
}

macro_rules! factors {
    ($result:expr, $split:expr, $($field:tt),+) => {
        match $result {
            Ok(value) => {
                let tuple = $split(value);
                let parts = [$(factor!(&tuple.$field)),+];
                (
                    parts.iter().map(|part| part.0.as_str()).collect::<Vec<_>>().join(";"),
                    parts.iter().map(|part| part.1.as_str()).collect::<Vec<_>>().join(";"),
                )
            }
            Err(_) => ("Err".to_string(), "Err".to_string()),
        }
    };
}

/// Fingerprints of every factorization, lazy-adjoint inputs included, of the
/// fixture `tall : [v, v] <- [v]`, `wide = tall^H`-shaped, a square
/// endomorphism, and a Hermitian endomorphism.
macro_rules! fingerprints {
    ($runtime:expr, $v:expr, $d:ty, $seed:expr) => {{
        let (runtime, v, seed) = ($runtime, $v, $seed);
        let tall = TensorMap::<_, $d>::rand_with_seed(runtime, [v, v], [v], seed).unwrap();
        let wide = TensorMap::<_, $d>::rand_with_seed(runtime, [v], [v, v], seed + 1).unwrap();
        let endo = TensorMap::<_, $d>::rand_with_seed(runtime, [v], [v], seed + 2).unwrap();
        // Real symmetric blocks with distinct diagonal: Hermitian for both
        // dtypes, built without a contraction (checked Generic rejects lazy
        // operands there).
        let herm = TensorMap::<_, $d>::from_block_fn(runtime, [v], [v], |_, ij| {
            let (i, j) = (ij[0] as f64, ij[1] as f64);
            <$d>::from(
                1.0 / (1.0 + i + j)
                    + if ij[0] == ij[1] {
                        0.37 * (1.0 + i)
                    } else {
                        0.0
                    },
            )
        })
        .unwrap();
        let (tall_h, wide_h) = (tall.adjoint().unwrap(), wide.adjoint().unwrap());
        let (endo_h, herm_h) = (endo.adjoint().unwrap(), herm.adjoint().unwrap());
        [
            factors!(tall.svd_compact(), svd, 0, 1, 2),
            factors!(tall.svd_full(), svd, 0, 1, 2),
            factors!(tall.qr_compact(), qr, 0, 1),
            factors!(wide.qr_full(), qr, 0, 1),
            factors!(wide.lq_compact(), lq, 0, 1),
            factors!(tall.lq_full(), lq, 0, 1),
            factors!(tall.left_polar(), left_polar, 0, 1),
            factors!(wide.right_polar(), right_polar, 0, 1),
            factors!(herm.eigh_full(), eigh, 0, 1),
            factors!(endo.eig_full(), eig, 0, 1),
            factors!(wide_h.svd_compact(), svd, 0, 1, 2),
            factors!(tall_h.svd_full(), svd, 0, 1, 2),
            factors!(wide_h.qr_compact(), qr, 0, 1),
            factors!(tall_h.qr_full(), qr, 0, 1),
            factors!(tall_h.lq_compact(), lq, 0, 1),
            factors!(wide_h.lq_full(), lq, 0, 1),
            factors!(wide_h.left_polar(), left_polar, 0, 1),
            factors!(tall_h.right_polar(), right_polar, 0, 1),
            factors!(herm_h.eigh_full(), eigh, 0, 1),
            factors!(endo_h.eig_full(), eig, 0, 1),
        ]
        .iter()
        .map(|(structure, exact)| (fnv1a(structure), fnv1a(exact)))
        .collect::<Vec<_>>()
    }};
}

const NAMES: [&str; 20] = [
    "svd_compact",
    "svd_full",
    "qr_compact",
    "qr_full",
    "lq_compact",
    "lq_full",
    "left_polar",
    "right_polar",
    "eigh_full",
    "eig_full",
    "adjoint svd_compact",
    "adjoint svd_full",
    "adjoint qr_compact",
    "adjoint qr_full",
    "adjoint lq_compact",
    "adjoint lq_full",
    "adjoint left_polar",
    "adjoint right_polar",
    "adjoint eigh_full",
    "adjoint eig_full",
];

fn check(label: &str, actual: &[(u64, u64)], structure: &[u64; 20], exact: &[u64; 20]) {
    let report = |pick: fn(&(u64, u64)) -> u64| {
        actual
            .iter()
            .map(|hashes| format!("{:#018x},", pick(hashes)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    eprintln!("{label} structure: [{}]", report(|hashes| hashes.0));
    eprintln!("{label} exact: [{}]", report(|hashes| hashes.1));
    for ((name, actual), expected) in NAMES.iter().zip(actual).zip(structure) {
        assert_eq!(actual.0, *expected, "{label} {name} structure");
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        for ((name, actual), expected) in NAMES.iter().zip(actual).zip(exact) {
            assert_eq!(actual.1, *expected, "{label} {name} exact");
        }
    }
}

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        U1FusionRule,
        [(-1, 1), (0, 2), (1, 2)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        SU2FusionRule,
        [(0, 2), (1, 2), (2, 1)].map(|(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
    )
    .unwrap()
}

fn fz2_u1_leg() -> GradedSpace<ProductFusionRule<FermionParityFusionRule, U1FusionRule>> {
    let rule = FermionParityFusionRule.product(U1FusionRule);
    GradedSpace::try_new(
        rule,
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
        ],
    )
    .unwrap()
}

/// One test, fixtures in a fixed order: sector ids are interned per process,
/// and block order follows them, so the fingerprints assume this sequence.
#[test]
fn named_results_match_the_tuple_results() {
    let runtime = runtime();
    let v = u1_leg();
    check(
        "u1 f64",
        &fingerprints!(&runtime, &v, f64, 1541),
        &U1_F64_STRUCTURE,
        &U1_F64,
    );
    check(
        "u1 c64",
        &fingerprints!(&runtime, &v, Complex64, 1541),
        &U1_C64_STRUCTURE,
        &U1_C64,
    );
    let v = su2_leg();
    check(
        "su2 f64",
        &fingerprints!(&runtime, &v, f64, 1541),
        &SU2_F64_STRUCTURE,
        &SU2_F64,
    );
    check(
        "su2 c64",
        &fingerprints!(&runtime, &v, Complex64, 1541),
        &SU2_C64_STRUCTURE,
        &SU2_C64,
    );
    let v = fz2_u1_leg();
    check(
        "fz2u1 f64",
        &fingerprints!(&runtime, &v, f64, 1541),
        &FZ2U1_F64_STRUCTURE,
        &FZ2U1_F64,
    );
    check(
        "fz2u1 c64",
        &fingerprints!(&runtime, &v, Complex64, 1541),
        &FZ2U1_C64_STRUCTURE,
        &FZ2U1_C64,
    );
    #[cfg(feature = "racah-generated")]
    {
        // Checked Generic rejects lazy-adjoint operands for most
        // factorizations; those entries fingerprint the error.
        use std::sync::Arc;
        use tenet::typed::SUNFusionRule;
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let v = GradedSpace::try_new_with_arc(
            provider,
            [(vec![0i64, 0], 2), (vec![1, 0], 2), (vec![0, 1], 1)],
        )
        .unwrap();
        check(
            "su3 f64",
            &fingerprints!(&runtime, &v, f64, 1541),
            &SU3_F64_STRUCTURE,
            &SU3_F64,
        );
        check(
            "su3 c64",
            &fingerprints!(&runtime, &v, Complex64, 1541),
            &SU3_C64_STRUCTURE,
            &SU3_C64,
        );
    }
}

const U1_F64_STRUCTURE: [u64; 20] = [
    0xd12f8de2c5e19768,
    0xef9895eddc9720ef,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
    0xd12f8de2c5e19768,
    0x0fcd5f2aae0adc2b,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
];
const U1_F64: [u64; 20] = [
    0xc7d4a9386959fedf,
    0x6ccb371813d66465,
    0x58073f2ddde11363,
    0x088764e125943a52,
    0xe0d6fe066fe182ad,
    0x370c9d660175bbe6,
    0xbcbc446d43bf055d,
    0x357c53d9413f5f1d,
    0x0427b0cbce01b22b,
    0x9eacc20a6bb38f23,
    0x0ad00b9efda3ea23,
    0xc831e49eeea7726a,
    0xe21e5f8b0b12cffd,
    0x22549abc7bd67c18,
    0x1195baa19ea488ab,
    0xbf80b20bedb48200,
    0xbb570a6121fe8c4f,
    0x325451e43f927af5,
    0x0427b0cbce01b22b,
    0x36177581ad8caa62,
];
const U1_C64_STRUCTURE: [u64; 20] = [
    0xd12f8de2c5e19768,
    0xef9895eddc9720ef,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
    0xd12f8de2c5e19768,
    0x0fcd5f2aae0adc2b,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
];
const U1_C64: [u64; 20] = [
    0x927458fc26ab6326,
    0x1e4c19fac7bfa9a8,
    0xe7f79c639e365d95,
    0xee1f76ddd74b8532,
    0xa3a2255a0eab759e,
    0xfa0e10a816a652c2,
    0x09149f63ca770168,
    0xb865df90b89b55b9,
    0x64ff1514719c1a0b,
    0xda636e8c4f05356c,
    0x820e16417a86d790,
    0x9a9746665dabc567,
    0x4c72e12a12e3744f,
    0xbe5cbd5d3fd12ca5,
    0xf1f84c7abdf96f5e,
    0xf375144f69a913fb,
    0x78ae7ca5225adc4f,
    0xe257ea167da90730,
    0x64ff1514719c1a0b,
    0x2c9c04927fe7eacf,
];
const SU2_F64_STRUCTURE: [u64; 20] = [
    0xc2735e92fe91606d,
    0x996be5ec295e05dc,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
    0xc2735e92fe91606d,
    0x5daae6758d3203b6,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
];
const SU2_F64: [u64; 20] = [
    0x2c8ead24f3d9d3dc,
    0x058a1f2ccec79691,
    0x4aa8203fdfd9b6e2,
    0xd3d3339297a20af0,
    0xcda57eb3985bbc58,
    0xec6248a2f4a04da5,
    0x7e99abd9853338bf,
    0x36ae910f2d23b6bf,
    0x8d436d0f3f902d9f,
    0x9dc68ade90d0c1aa,
    0x7df281dbbdbeddcc,
    0x57f91b8203aa3551,
    0x9d2cf6c8d254834e,
    0x4400f2902da9b76d,
    0xdf7e90da5b1b6b5c,
    0x0b91da8b7f65cf46,
    0x7ecdc6b883224f99,
    0x6be9159b247b41d7,
    0x8d436d0f3f902d9f,
    0xcf22370816fae1bc,
];
const SU2_C64_STRUCTURE: [u64; 20] = [
    0xc2735e92fe91606d,
    0x996be5ec295e05dc,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
    0xc2735e92fe91606d,
    0x5daae6758d3203b6,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
];
const SU2_C64: [u64; 20] = [
    0x39b2e52328588e9d,
    0x289db2d5e4627c4e,
    0x0e1be73ca1cb1367,
    0x2add883d2170dd50,
    0x465743ea98509911,
    0xc9ba0ed39a9500f3,
    0xa3427a38a4f43d93,
    0x39ff2248e18fd31b,
    0x94e02e3ffbeefe7f,
    0x9e3d81568fb83814,
    0x2e96bbde6011518b,
    0x0bb7cdb54edce565,
    0x4b711d0de531435d,
    0x5299717ad9ada771,
    0x5e4c714d4be4874d,
    0xb18b974b92344566,
    0xdef5e464d4e507b0,
    0x87278118a3602fdc,
    0x94e02e3ffbeefe7f,
    0x81607c6f30ddb487,
];
const FZ2U1_F64_STRUCTURE: [u64; 20] = [
    0x0f9538ca0ba4c87a,
    0x04c6b7b5df099c9b,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
    0x0f9538ca0ba4c87a,
    0x5179d2ca9d9ee5cd,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
];
const FZ2U1_F64: [u64; 20] = [
    0x0caf7931bfd3c4cb,
    0xdca9c51a102f39db,
    0x31af1f29f74bd6ec,
    0x8c2abb9739eb28f5,
    0x5cdd525ac34f4390,
    0xcb5c92645fbc3187,
    0xadee5d70a7b31992,
    0xc5002eb3db4cd4f8,
    0x57ef2a6423428527,
    0xac60ab563f375af7,
    0xde7d102cdbc2a14f,
    0x8bc4f55f46d70698,
    0xb6ac73892efb1c56,
    0xf6d4523afd2f09f7,
    0xe31b1faba2c73af2,
    0xa366fac7915221b9,
    0xed4e94194de1464c,
    0xcc4603772aa2b424,
    0x57ef2a6423428527,
    0x7edde4952a2d8c96,
];
const FZ2U1_C64_STRUCTURE: [u64; 20] = [
    0x0f9538ca0ba4c87a,
    0x04c6b7b5df099c9b,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
    0x0f9538ca0ba4c87a,
    0x5179d2ca9d9ee5cd,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
];
const FZ2U1_C64: [u64; 20] = [
    0x35b2149eb72b3b96,
    0x3d449c42d4f9e96c,
    0xb8ca7f09b7ec7296,
    0x68b0c48f92ae94a7,
    0x7e714b9c7e42f52b,
    0x597c31fbb4171f8d,
    0x608f420aaa167cb9,
    0xf5ce8f1ca2e4e03a,
    0x57d059e9b7890eb1,
    0x923f618b3d8c920e,
    0x8d4987d64c522618,
    0x0752768af2b6c871,
    0xd431d41996f08c78,
    0xdb92a7f694a66f0e,
    0xc549648841791991,
    0xc87f5382ec5a540a,
    0x666d3bc521985036,
    0x21f30966c011fa51,
    0x57d059e9b7890eb1,
    0xb86d7d84ea9dab2d,
];
#[cfg(feature = "racah-generated")]
const SU3_F64_STRUCTURE: [u64; 20] = [
    0x12cb72c30df11d0f,
    0x8cc387613de843c4,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0xf065848bb8da7bd1,
    0x66fd3d88cf5e7459,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
];
#[cfg(feature = "racah-generated")]
const SU3_F64: [u64; 20] = [
    0xaa8c6dee3e7d5d6c,
    0x2dd2146f4d494ed3,
    0x941829a514907395,
    0x07ec7dd204814640,
    0x833f15b01063ccb4,
    0xd105725ac19f8b11,
    0xe4d5c778a5724c2e,
    0x522d7560a3f5797e,
    0x86e7a5945e0c19f7,
    0x153c7e86547399f3,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0x1146fc2e2d3e9554,
    0x25ea335e3fa83b42,
    0x86e7a5945e0c19f7,
    0xcb5406851140e672,
];
#[cfg(feature = "racah-generated")]
const SU3_C64_STRUCTURE: [u64; 20] = [
    0x12cb72c30df11d0f,
    0x8cc387613de843c4,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0xf065848bb8da7bd1,
    0x66fd3d88cf5e7459,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
];
#[cfg(feature = "racah-generated")]
const SU3_C64: [u64; 20] = [
    0x2a24b0233aa370aa,
    0x6ce7c0fa70bc3f2a,
    0xc28cb0b2fc8ad53f,
    0x6f7dc8d068450aef,
    0x0b1fd04660a12253,
    0xa0dfb70e20f6d612,
    0x5e089b61c49000ca,
    0xae97c3bd61d46d67,
    0x2ceb842c8bd44d19,
    0xef1e1ecc13e24cee,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd90625198e16ea2c,
    0xd116c41479a9f973,
    0x4af583bb9acb5178,
    0x2ceb842c8bd44d19,
    0xffe5d99128cbf21d,
];
