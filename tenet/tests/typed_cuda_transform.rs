//! Real-device gates for typed structural transforms on CUDA tensors
//! (issue #1322, G2b-2): `permute`, `braid`, `transpose`, `transpose_axes`
//! and `repartition` on `TensorMap<R, D, CudaStorage<D>>`.
//!
//! Evidence chain, as the design review fixed it:
//!
//! 1. the device executor below the completed `TreeTransformStructure` is
//!    gated independently by the structure walker in
//!    `tenet-operations/tests/cuda_tree_transform.rs`;
//! 2. the typed *lowering* is gated here, by running the same public call on
//!    the Host and on the device — the Host side carries its own TensorKit and
//!    hand-computed oracles (`typed_facade.rs`, `semantic_suite.rs`,
//!    `tk_complex_su2.rs`, `lazy_adjoint_shared_owner.rs`), which these are
//!    device twins of;
//! 3. a dense oracle independent of both runs where a transform really is a
//!    plain axis permutation of the physical array: a bosonic permute within
//!    one side, for the two providers that have a physical basis (U(1), SU(2)).
//!    Its own agreement with the Host is checked on CPU, in CI.
//!
//! Every fixture additionally asserts non-vacuity — either that the payload
//! actually moved, or that the transform is not a bare reordering, which is
//! what proves a coefficient other than `1` or a recoupling block took part.
//!
//! Run with `cargo test -p tenet-rs --features cuda,cpu-faer --test \
//! typed_cuda_transform -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use num_complex::Complex64;
use tenet::core::CheckedFusionAlgebra;
use tenet::core::{
    product_sector, FermionParityFusionRule, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, SectorCodec, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{Runtime, TensorScalar};
use tenet::typed::{GradedSpace, TensorMap};

// ---------------------------------------------------------------------------
// Payload comparison
// ---------------------------------------------------------------------------

/// The two device payload dtypes, with the comparisons the gates need.
trait Payload: TensorScalar + Copy + PartialEq + std::fmt::Debug {
    fn parts(self) -> (f64, f64);
    fn distance(self, other: Self) -> f64 {
        let (ar, ai) = self.parts();
        let (br, bi) = other.parts();
        ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt()
    }
    fn magnitude(self) -> f64 {
        let (re, im) = self.parts();
        (re * re + im * im).sqrt()
    }
}

impl Payload for f64 {
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}

impl Payload for Complex64 {
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

fn assert_payload_close<D: Payload>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: payload length");
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(right) <= 1e-12 * (1.0 + right.magnitude()),
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

/// Sorted payload, used only to compare two payloads as multisets.
fn sorted_values<D: Payload>(data: &[D]) -> Vec<(f64, f64)> {
    let mut values: Vec<(f64, f64)> = data.iter().map(|value| value.parts()).collect();
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite fixture payload"));
    values
}

/// Non-vacuity, weak form: the transform actually moved the payload, so a
/// device replay that wrote nothing at all could not pass.
fn assert_moved<D: Payload>(source: &[D], transformed: &[D], what: &str) {
    assert_eq!(source.len(), transformed.len(), "{what}: length");
    assert!(
        source.iter().zip(transformed).any(|(a, b)| a != b),
        "{what}: the transform left the payload identical, so it proves nothing"
    );
}

/// Non-vacuity, strong form: the transform is not a bare reordering of the
/// source payload, so at least one structural coefficient differs from `1` or
/// a recoupling block mixed several source entries.
fn assert_not_a_reordering<D: Payload>(source: &[D], transformed: &[D], what: &str) {
    assert_ne!(
        sorted_values(source),
        sorted_values(transformed),
        "{what}: the payload is a permutation of the source, so no coefficient \
         other than 1 and no recoupling block took part"
    );
}

// ---------------------------------------------------------------------------
// Layout comparison
// ---------------------------------------------------------------------------

type Layout<S> = (
    Vec<(bool, Vec<usize>)>,
    Vec<tenet::typed::BlockFusionTrees<S>>,
);

fn layout<R, D>(tensor: &TensorMap<R, D>) -> Layout<R::Sector>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    (
        tensor
            .codomain()
            .iter()
            .chain(tensor.domain().iter())
            .map(|leg| (leg.is_dual(), leg.degeneracies().to_vec()))
            .collect(),
        (0..tensor.block_count())
            .map(|index| tensor.block_fusion_trees(index).unwrap())
            .collect(),
    )
}

/// Runs one transform on the Host and on the device and asserts the device
/// answer is the Host answer — payload to dtype tolerance, spaces and block
/// identities exactly. `$body` is written once and expanded against both
/// receiver types, so the two calls cannot drift.
macro_rules! device_matches_host {
    ($what:expr, $host:expr, |$t:ident| $body:expr) => {{
        let host_source = &$host;
        let device_source = host_source.to_cuda().unwrap();
        let expected = {
            let $t = host_source;
            $body
        }
        .unwrap();
        let actual = {
            let $t = &device_source;
            $body
        }
        .unwrap()
        .to_host()
        .unwrap();
        let what: &str = $what;
        assert_payload_close(actual.data(), expected.data(), what);
        assert_eq!(layout(&actual), layout(&expected), "{what}: layout");
        actual
    }};
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn runtime() -> Runtime {
    Runtime::builder().cuda(0).build().unwrap()
}

/// Distinct, finite, irrational-ish entries: a permutation of the payload is
/// then visible as a multiset, and no two entries cancel by accident.
fn real_fill(_trees: &tenet::typed::BlockFusionTrees<impl std::fmt::Debug>, idx: &[usize]) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

fn complex_fill(
    trees: &tenet::typed::BlockFusionTrees<impl std::fmt::Debug>,
    idx: &[usize],
) -> Complex64 {
    let re = real_fill(trees, idx);
    Complex64::new(re, -0.5 * re + 0.25)
}

fn u1_leg(charges: &[(i32, usize)], dual: bool) -> GradedSpace<U1FusionRule> {
    let space = GradedSpace::try_new_with_arc(
        Arc::new(U1FusionRule),
        charges
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    if dual {
        space.try_dual().unwrap()
    } else {
        space
    }
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

fn fz2_leg() -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new_with_arc(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    )
    .unwrap()
}

/// `[leg, leg] <- [leg]` over fZ2 with the counting fill of
/// `typed_facade.rs::fermionic_rank_three`, whose transposes and permutes are
/// hand-computed there.
fn fz2_rank_three(runtime: &Runtime) -> TensorMap<FermionParityFusionRule, f64> {
    let leg = fz2_leg();
    let mut next = 0.0;
    TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap()
}

// ---------------------------------------------------------------------------
// Dense oracle (bosonic permute within one side)
// ---------------------------------------------------------------------------

/// Permutes the axes of a row-major dense array: `out[i] = data[perm(i)]`,
/// where axis `k` of the output is axis `perm[k]` of the input.
fn permute_dense<D: Payload>(shape: &[usize], data: &[D], perm: &[usize]) -> (Vec<usize>, Vec<D>) {
    let out_shape: Vec<usize> = perm.iter().map(|&axis| shape[axis]).collect();
    let stride = |shape: &[usize]| {
        let mut strides = vec![1usize; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        strides
    };
    let in_strides = stride(shape);
    let out_strides = stride(&out_shape);
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        let mut source = 0usize;
        for (axis, &source_axis) in perm.iter().enumerate() {
            let index = (flat / out_strides[axis]) % out_shape[axis].max(1);
            source += index * in_strides[source_axis];
        }
        out.push(data[source]);
    }
    (out_shape, out)
}

/// The physical-basis oracle: a permute that stays within the codomain and
/// within the domain is, in the physical carrier basis, exactly that axis
/// permutation of the dense array — however many F moves the reduced-block
/// replay had to make to produce it.
macro_rules! assert_dense_oracle {
    ($what:expr, $tensor:expr, $codomain_axes:expr, $domain_axes:expr) => {{
        let tensor = &$tensor;
        let codomain_axes: &[usize] = $codomain_axes;
        let domain_axes: &[usize] = $domain_axes;
        let permuted = tensor.permute(codomain_axes, domain_axes).unwrap();
        let source = tensor.to_physical_dense().unwrap();
        let moved = permuted.to_physical_dense().unwrap();
        let perm: Vec<usize> = codomain_axes
            .iter()
            .chain(domain_axes.iter())
            .copied()
            .collect();
        let (shape, data) = permute_dense(&source.shape, &source.data, &perm);
        assert_eq!(moved.shape, shape, "{}: dense oracle shape", $what);
        assert_payload_close(&moved.data, &data, concat!($what, ": dense oracle"));
    }};
}

/// CPU gate for the oracle itself (#1322 acceptance: the oracle must be
/// independent evidence, so it is pinned against the Host before any device
/// result is compared to it). Runs in CI: no device is touched.
#[test]
fn the_dense_permute_oracle_agrees_with_the_host_for_u1_and_su2() {
    let runtime = Runtime::builder().build().unwrap();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let u1_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    assert_dense_oracle!("U(1) within-side permute", u1_tensor, &[1, 0], &[3, 2]);

    let su2 = su2_leg();
    let su2_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2], real_fill).unwrap();
    assert_dense_oracle!("SU(2) within-side permute", su2_tensor, &[1, 0], &[3, 2]);
    // The SU(2) case is the interesting one: the reduced-block replay must
    // recouple, so the payload is not a reordering of the source.
    let permuted = su2_tensor.permute(&[1, 0], &[3, 2]).unwrap();
    assert_not_a_reordering(su2_tensor.data(), permuted.data(), "SU(2) permute");
}

// ---------------------------------------------------------------------------
// Device gates
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a real CUDA device"]
fn device_permute_braid_and_planar_match_the_host_for_u1_and_su2() {
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let u1_dual = u1_leg(&[(-1, 1), (0, 2), (1, 1)], true);
    let su2 = su2_leg();

    // U(1) with a dual leg and several coupled sectors, both dtypes.
    let real: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1_dual], [&u1, &u1], real_fill).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1_dual], [&u1, &u1], complex_fill).unwrap();
    assert!(real.block_count() >= 2, "multi-block fixture");

    let permuted = device_matches_host!("U1/f64 permute", real, |t| t.permute(&[2, 0], &[1, 3]));
    assert_moved(real.data(), permuted.data(), "U1/f64 permute");
    // Coefficient-1 f64 block moves are bitwise on device.
    assert_eq!(
        permuted.data(),
        real.permute(&[2, 0], &[1, 3]).unwrap().data(),
        "coefficient-1 f64 permute must move bitwise"
    );
    device_matches_host!("U1/c64 permute", complex, |t| t.permute(&[2, 0], &[1, 3]));
    device_matches_host!("U1/f64 braid", real, |t| t.braid(
        &[2, 0],
        &[1, 3],
        &[3, 1, 4, 2]
    ));
    device_matches_host!("U1/f64 transpose", real, |t| t.transpose());
    device_matches_host!("U1/c64 transpose", complex, |t| t.transpose());
    device_matches_host!("U1/f64 transpose_axes", real, |t| t
        .transpose_axes(&[1, 3], &[0, 2]));

    // SU(2): recoupling, so Multi blocks and coefficients other than 1.
    let su2_real: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2], real_fill).unwrap();
    let su2_complex: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2], complex_fill).unwrap();
    let permuted =
        device_matches_host!("SU2/f64 permute", su2_real, |t| t.permute(&[1, 2], &[3, 0]));
    assert_not_a_reordering(su2_real.data(), permuted.data(), "SU2/f64 permute");
    let permuted = device_matches_host!("SU2/c64 permute", su2_complex, |t| t
        .permute(&[1, 2], &[3, 0]));
    assert_not_a_reordering(su2_complex.data(), permuted.data(), "SU2/c64 permute");
    let transposed = device_matches_host!("SU2/f64 transpose", su2_real, |t| t.transpose());
    assert_not_a_reordering(su2_real.data(), transposed.data(), "SU2/f64 transpose");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_permutes_match_the_dense_physical_oracle_within_one_side() {
    // The independent oracle of the acceptance gate: for a bosonic provider, a
    // permute inside the codomain and inside the domain is an axis permutation
    // of the physical dense array. SU(2) makes it non-trivial — the reduced
    // replay recouples to produce it.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let u1_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let device = u1_tensor.to_cuda().unwrap();
    let moved = device.permute(&[1, 0], &[3, 2]).unwrap().to_host().unwrap();
    let source = u1_tensor.to_physical_dense().unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = moved.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_payload_close(&actual.data, &data, "U(1) device dense oracle");

    let su2 = su2_leg();
    let su2_tensor: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2], complex_fill).unwrap();
    let device = su2_tensor.to_cuda().unwrap();
    let moved = device.permute(&[1, 0], &[3, 2]).unwrap().to_host().unwrap();
    assert_not_a_reordering(su2_tensor.data(), moved.data(), "SU(2) device permute");
    let source = su2_tensor.to_physical_dense().unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = moved.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_payload_close(&actual.data, &data, "SU(2) device dense oracle");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transform_laws_hold_on_device_tensors() {
    // Device twins of `semantic_suite.rs`: permute composition, braid inverse,
    // bosonic braid == permute, transpose involution — all evaluated entirely
    // on device, then compared to the Host's own answer for the same law.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let device = host.to_cuda().unwrap();

    // permute ∘ permute == permute(composition)
    let first = [2usize, 0, 3, 1];
    let second = [1usize, 0, 3, 2];
    let stepwise = device
        .permute(&first[..2], &first[2..])
        .unwrap()
        .permute(&second[..2], &second[2..])
        .unwrap()
        .to_host()
        .unwrap();
    let composed: Vec<usize> = second.iter().map(|&axis| first[axis]).collect();
    let direct = device
        .permute(&composed[..2], &composed[2..])
        .unwrap()
        .to_host()
        .unwrap();
    assert_payload_close(stepwise.data(), direct.data(), "permute composition");
    assert_moved(host.data(), direct.data(), "permute composition");

    // braid ∘ braid⁻¹ == id, with the levels carried along the strands.
    let axes = [2usize, 0, 3, 1];
    let levels = [4usize, 1, 3, 2];
    let braided = device.braid(&axes[..2], &axes[2..], &levels).unwrap();
    let mut inverse = [0usize; 4];
    for (index, &axis) in axes.iter().enumerate() {
        inverse[axis] = index;
    }
    let carried: Vec<usize> = axes.iter().map(|&axis| levels[axis]).collect();
    let back = braided
        .braid(&inverse[..2], &inverse[2..], &carried)
        .unwrap()
        .to_host()
        .unwrap();
    assert_payload_close(back.data(), host.data(), "braid inverse round trip");

    // Bosonic provider: braid == permute for any levels.
    let permuted = device.permute(&axes[..2], &axes[2..]).unwrap();
    assert_payload_close(
        braided.to_host().unwrap().data(),
        permuted.to_host().unwrap().data(),
        "bosonic braid == permute",
    );

    // Transpose is an involution.
    let twice = device
        .transpose()
        .unwrap()
        .transpose()
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(twice.data(), host.data(), "transpose involution is bitwise");
    assert_eq!(layout(&twice), layout(&host), "transpose involution layout");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_repartition_reaches_every_split_and_round_trips() {
    // The #1310 residual: codomain/domain repartition, i.e. bends, on device.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let device = host.to_cuda().unwrap();
    for num_codomain in 0..=4 {
        let what = format!("U(1) repartition to {num_codomain}");
        let moved = device.repartition(num_codomain).unwrap();
        let expected = host.repartition(num_codomain).unwrap();
        let actual = moved.to_host().unwrap();
        assert_payload_close(actual.data(), expected.data(), &what);
        assert_eq!(layout(&actual), layout(&expected), "{what}: layout");
        let back = moved.repartition(2).unwrap().to_host().unwrap();
        assert_eq!(back.data(), host.data(), "{what}: round trip is bitwise");
        assert_eq!(layout(&back), layout(&host), "{what}: round trip layout");
    }

    // SU(2): a bend carries a scalar that is neither 1 nor -1, so the payload
    // is not a reordering of the source.
    let su2 = su2_leg();
    let su2_host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2], real_fill).unwrap();
    for num_codomain in [0usize, 1, 3, 4] {
        let what = format!("SU(2) repartition to {num_codomain}");
        let moved = device_matches_host!(&what, su2_host, |t| t.repartition(num_codomain));
        assert_not_a_reordering(su2_host.data(), moved.data(), &what);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_fermionic_signs_match_the_hand_computed_fixture() {
    // Device twin of `typed_facade.rs::planar_transposes_bend_where_permute_braids`
    // and `repartition_is_sign_free_even_for_a_fermionic_provider`: the values
    // are hand-computed there, not read off any TeNeT descriptor.
    let runtime = runtime();
    let host = fz2_rank_three(&runtime);
    assert_eq!(host.data(), [1.0, 2.0, 3.0, 4.0]);
    let device = host.to_cuda().unwrap();

    assert_eq!(
        device.transpose().unwrap().to_host().unwrap().data(),
        [1.0, 2.0, 4.0, 3.0]
    );
    assert_eq!(
        device
            .permute(&[2], &[1, 0])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        [1.0, 2.0, -4.0, -3.0]
    );
    assert_eq!(
        device
            .transpose_axes(&[1, 2], &[0])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        [1.0, 4.0, 2.0, 3.0]
    );
    assert_eq!(
        device
            .permute(&[1, 2], &[0])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        [1.0, 4.0, -2.0, -3.0]
    );
    for num_codomain in [0usize, 1, 3] {
        assert_eq!(
            device
                .repartition(num_codomain)
                .unwrap()
                .to_host()
                .unwrap()
                .data(),
            host.repartition(num_codomain).unwrap().data(),
            "repartition to {num_codomain} is sign free",
        );
    }
    // Non-vacuity: the permutes above changed signs, so they are not
    // reorderings of the source payload.
    assert_not_a_reordering(
        host.data(),
        device
            .permute(&[2], &[1, 0])
            .unwrap()
            .to_host()
            .unwrap()
            .data(),
        "fZ2 permute",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_product_providers_match_the_host_for_signs_and_recoupling() {
    let runtime = runtime();

    // fZ2 x U(1): fermionic signs on top of a charge grading.
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
    )
    .unwrap();
    let host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], real_fill).unwrap();
    let permuted = device_matches_host!("fZ2xU1 permute", host, |t| t.permute(&[2, 1], &[0, 3]));
    assert_not_a_reordering(host.data(), permuted.data(), "fZ2xU1 permute");
    device_matches_host!("fZ2xU1 transpose", host, |t| t.transpose());
    device_matches_host!("fZ2xU1 braid", host, |t| t.braid(
        &[2, 1],
        &[0, 3],
        &[1, 4, 2, 3]
    ));
    device_matches_host!("fZ2xU1 repartition", host, |t| t.repartition(1));

    // fZ2 (x) SU(2): fermionic signs *and* recoupling in one provider.
    let rule = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&rule),
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let host: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], complex_fill).unwrap();
    let permuted = device_matches_host!("fZ2xSU2 permute", host, |t| t.permute(&[1, 2], &[3, 0]));
    assert_not_a_reordering(host.data(), permuted.data(), "fZ2xSU2 permute");
    let transposed = device_matches_host!("fZ2xSU2 transpose", host, |t| t.transpose());
    assert_not_a_reordering(host.data(), transposed.data(), "fZ2xSU2 transpose");
    device_matches_host!("fZ2xSU2 repartition", host, |t| t.repartition(3));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_rank_five_transforms_match_the_host() {
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 1)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&u1, &u1, &u1], [&u1, &u1], real_fill).unwrap();
    assert!(host.block_count() >= 3, "multi-block rank-5 fixture");
    let permuted = device_matches_host!("rank-5 permute", host, |t| t.permute(&[4, 1, 0], &[3, 2]));
    assert_moved(host.data(), permuted.data(), "rank-5 permute");
    device_matches_host!("rank-5 transpose", host, |t| t.transpose());
    device_matches_host!("rank-5 repartition", host, |t| t.repartition(1));

    let su2 = su2_leg();
    let su2_host: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2, &su2, &su2], [&su2, &su2], real_fill).unwrap();
    let permuted = device_matches_host!("rank-5 SU(2) permute", su2_host, |t| t
        .permute(&[4, 1, 0], &[3, 2]));
    assert_not_a_reordering(su2_host.data(), permuted.data(), "rank-5 SU(2) permute");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_complex_su2_permute_matches_the_tensorkit_fixture() {
    // Device twin of `tk_complex_su2.rs`: the norm is TensorKit's own value,
    // so this is a value oracle outside TeNeT entirely.
    let runtime = runtime();
    let space = GradedSpace::try_new_with_arc(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let fill = |c0: i64, labels: [i64; 5], idx: &[usize]| {
        let [l1, l2, m1, m2, lc] = labels;
        let value = c0
            + 7 * l1
            + 11 * l2
            + 13 * m1
            + 17 * m2
            + 19 * lc
            + 23 * (idx[0] as i64 + 1)
            + 29 * (idx[1] as i64 + 1)
            + 31 * (idx[2] as i64 + 1)
            + 37 * (idx[3] as i64 + 1);
        (value.rem_euclid(41) - 20) as f64
    };
    let host: TensorMap<_, Complex64> = TensorMap::from_block_fn(
        &runtime,
        [&space, &space],
        [&space, &space],
        |trees, idx| {
            let label = |sector: &SU2Irrep| sector.twice_spin() as i64;
            let codomain = trees.codomain_uncoupled();
            let domain = trees.domain_uncoupled();
            let labels = [
                label(&codomain[0]),
                label(&codomain[1]),
                label(&domain[0]),
                label(&domain[1]),
                label(trees.coupled()),
            ];
            Complex64::new(fill(11, labels, idx), fill(17, labels, idx) / 3.0)
        },
    )
    .unwrap();

    let permuted = device_matches_host!("tk SU(2) c64 permute", host, |t| t
        .permute(&[1, 0], &[3, 2]));
    assert_not_a_reordering(host.data(), permuted.data(), "tk SU(2) c64 permute");
    let norm = permuted.norm().unwrap();
    assert!(
        (norm - 40.741_733_994_626_32).abs() <= 1e-10 * 41.0,
        "TensorKit norm of the permuted tensor: got {norm}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_lazy_adjoint_transforms_lower_onto_the_parent_like_the_host() {
    // Device twin of `lazy_adjoint_shared_owner.rs`: every transform of a lazy
    // adjoint lowers onto the parent and re-wraps, so it must equal the Host's
    // transform of the same lazy adjoint.
    let runtime = runtime();
    let a = u1_leg(&[(-1, 2), (0, 1), (1, 3)], false);
    let b = u1_leg(&[(0, 2), (1, 1)], true);
    let c = u1_leg(&[(-1, 1), (1, 2)], false);
    let d = u1_leg(&[(-2, 2), (-1, 3), (0, 1), (1, 2), (2, 1)], false);
    let parent: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&a, &b, &c], [&d], complex_fill).unwrap();
    assert!(parent.block_count() >= 4, "multi-block lazy fixture");
    let host_lazy = parent.adjoint().unwrap();
    let device_lazy = parent.to_cuda().unwrap().adjoint().unwrap();

    let rank = host_lazy.rank();
    let split = host_lazy.codomain_rank();
    let mut axes: Vec<usize> = (0..rank).collect();
    axes.rotate_left(1);
    axes.swap(0, 1);
    let (codomain_axes, domain_axes) = axes.split_at(split);
    let levels: Vec<usize> = (0..rank).map(|axis| (axis * 7) % rank + 1).collect();
    let mut cycle: Vec<usize> = (0..split).chain((split..rank).rev()).collect();
    cycle.rotate_left(1);
    let (cyclic_codomain, reversed_domain) = cycle.split_at(split);
    let cyclic_domain: Vec<usize> = reversed_domain.iter().rev().copied().collect();
    let new_split = if split > 1 { split - 1 } else { split + 1 };

    let pairs = [
        (
            "lazy permute",
            host_lazy.permute(codomain_axes, domain_axes).unwrap(),
            device_lazy.permute(codomain_axes, domain_axes).unwrap(),
        ),
        (
            "lazy braid",
            host_lazy
                .braid(codomain_axes, domain_axes, &levels)
                .unwrap(),
            device_lazy
                .braid(codomain_axes, domain_axes, &levels)
                .unwrap(),
        ),
        (
            "lazy repartition",
            host_lazy.repartition(new_split).unwrap(),
            device_lazy.repartition(new_split).unwrap(),
        ),
        (
            "lazy transpose",
            host_lazy.transpose().unwrap(),
            device_lazy.transpose().unwrap(),
        ),
        (
            "lazy transpose_axes",
            host_lazy
                .transpose_axes(cyclic_codomain, &cyclic_domain)
                .unwrap(),
            device_lazy
                .transpose_axes(cyclic_codomain, &cyclic_domain)
                .unwrap(),
        ),
    ];
    for (what, expected, actual) in pairs {
        let actual = actual.to_host().unwrap();
        assert_payload_close(actual.data(), expected.data(), what);
        assert_eq!(layout(&actual), layout(&expected), "{what}: layout");
        assert_moved(parent.data(), actual.data(), what);
    }
}
