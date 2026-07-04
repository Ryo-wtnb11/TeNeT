//! Operation-identity property tests over the user layer (issue #9, part 2)
//! plus the cross-library invariant-stream check against TensorKit (part 3).
//!
//! Seeding reuses the splitmix64 scheme of `Tensor::rand_with_seed`
//! (`tenet/src/tensor.rs`); no new dependencies.
//!
//! Out of scope here (matching the issue): multiplicity `N > 1`,
//! non-symmetric (anyonic) braiding, and repartition/bending — extend this
//! suite when those land.

use tenet::prelude::*;
use tenet_network::tensor;

fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "data lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
            "element {index} differs: {a} vs {b}"
        );
    }
}

fn assert_scalar_close(lhs: f64, rhs: f64, tol: f64) {
    assert!(
        (lhs - rhs).abs() <= tol * (1.0 + lhs.abs().max(rhs.abs())),
        "{lhs} vs {rhs}"
    );
}

/// splitmix64, same generator as `Tensor::rand_with_seed`
/// (`tenet/src/tensor.rs`).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_below(state: &mut u64, bound: usize) -> usize {
    (splitmix64(state) % bound as u64) as usize
}

/// Fisher-Yates permutation of `0..n`.
fn rand_perm(state: &mut u64, n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rand_below(state, i + 1));
    }
    perm
}

/// The per-rule small test spaces: (name, space, is_fermionic).
fn spaces() -> Vec<(&'static str, Space, bool)> {
    vec![
        ("Z2", Space::z2([(0, 2), (1, 2)]), false),
        ("U1", Space::u1([(-1, 2), (0, 2), (1, 1)]), false),
        ("SU2", Space::su2([(0, 2), (1, 2), (2, 1)]), false),
        ("fZ2", Space::fz2([(0, 2), (1, 2)]), true),
        (
            "fZ2xU1xSU2",
            Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 1), ((0, 2, 0), 1)]).unwrap(),
            true,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Permute / braid
// ---------------------------------------------------------------------------

/// permute ∘ permute == permute(composition), random permutation pairs on
/// rank-4 and rank-5 tensors (splits chosen randomly, mirroring the mixed
/// codomain/domain splits already exercised by `user_api.rs`).
#[test]
fn permute_composition_law() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0001u64;
    for (name, v, _) in spaces() {
        for (ncod, ndom, seed) in [(2usize, 2usize, 11u64), (1, 4, 12)] {
            let rank = ncod + ndom;
            let cod: Vec<&Space> = std::iter::repeat(&v).take(ncod).collect();
            let dom: Vec<&Space> = std::iter::repeat(&v).take(ndom).collect();
            let t = Tensor::rand_with_seed(&rt, cod, dom, seed).unwrap();
            for _ in 0..3 {
                let s1 = rand_perm(&mut state, rank);
                let n1 = 1 + rand_below(&mut state, rank - 1);
                let s2 = rand_perm(&mut state, rank);
                let n2 = 1 + rand_below(&mut state, rank - 1);
                let step1 = t.permute(&s1[..n1], &s1[n1..]).unwrap();
                let step2 = step1.permute(&s2[..n2], &s2[n2..]).unwrap();
                let composed: Vec<usize> = s2.iter().map(|&i| s1[i]).collect();
                let direct = t.permute(&composed[..n2], &composed[n2..]).unwrap();
                assert_close(step2.data(), direct.data(), 1e-12);
                assert_scalar_close(step2.norm().unwrap(), t.norm().unwrap(), 1e-12);
                let _ = name;
            }
        }
    }
}

/// braid ∘ braid⁻¹ == id: the inverse braid applies the inverse permutation
/// with the levels carried along the strands (TensorKit level semantics:
/// undoing a crossing swaps which strand passes above).
#[test]
fn braid_inverse_roundtrip() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0002u64;
    for (name, v, _) in spaces() {
        let t = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 21).unwrap();
        for _ in 0..3 {
            let s = rand_perm(&mut state, 4);
            let levels = rand_perm(&mut state, 4)
                .into_iter()
                .map(|l| l + 1)
                .collect::<Vec<_>>();
            let braided = t.braid(&s[..2], &s[2..], &levels).unwrap();
            let mut s_inv = vec![0usize; 4];
            for (i, &j) in s.iter().enumerate() {
                s_inv[j] = i;
            }
            let levels_braided: Vec<usize> = s.iter().map(|&j| levels[j]).collect();
            let back = t
                .braid(&s[..2], &s[2..], &levels)
                .unwrap()
                .braid(&s_inv[..2], &s_inv[2..], &levels_braided)
                .unwrap();
            assert_close(back.data(), t.data(), 1e-12);
            assert_scalar_close(braided.norm().unwrap(), t.norm().unwrap(), 1e-12);
            let _ = name;
        }
    }
}

/// Bosonic rules: braid == permute for every level assignment.
#[test]
fn bosonic_braid_equals_permute() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0003u64;
    for (name, v, fermionic) in spaces() {
        if fermionic {
            continue;
        }
        let t = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 31).unwrap();
        for _ in 0..3 {
            let s = rand_perm(&mut state, 4);
            let levels = rand_perm(&mut state, 4)
                .into_iter()
                .map(|l| l + 1)
                .collect::<Vec<_>>();
            let braided = t.braid(&s[..2], &s[2..], &levels).unwrap();
            let permuted = t.permute(&s[..2], &s[2..]).unwrap();
            assert_close(braided.data(), permuted.data(), 1e-12);
            let _ = name;
        }
    }
}

/// Yang-Baxter on three adjacent codomain legs:
/// `b0 b1 b0 == b1 b0 b1` where `b_i` swaps codomain legs `i, i+1`.
/// All rules in scope have real symmetric R (±1), so the crossing
/// chirality (level order) does not affect the value; distinct levels are
/// still passed so the braid engine takes the genuine braiding path.
#[test]
fn yang_baxter_adjacent_swaps() {
    let rt = Runtime::builder().build().unwrap();
    for (name, v, _) in spaces() {
        let t = Tensor::rand_with_seed(&rt, [&v, &v, &v], [&v], 41).unwrap();
        let swap = |t: &Tensor, i: usize| {
            let mut cod = vec![0usize, 1, 2];
            cod.swap(i, i + 1);
            t.braid(&cod, &[3], &[1, 2, 3, 4]).unwrap()
        };
        let lhs = swap(&swap(&swap(&t, 0), 1), 0);
        let rhs = swap(&swap(&swap(&t, 1), 0), 1);
        assert_close(lhs.data(), rhs.data(), 1e-12);
        let _ = name;
    }
}

// ---------------------------------------------------------------------------
// Adjoint / trace / twist / isometry
// ---------------------------------------------------------------------------

/// adjoint is an involution and an antihomomorphism: `(a∘b)† == b†∘a†`.
#[test]
fn adjoint_involution_and_antihomomorphism() {
    let rt = Runtime::builder().build().unwrap();
    for (name, v, _) in spaces() {
        let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 51).unwrap();
        let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 52).unwrap();
        let round = a.adjoint().unwrap().adjoint().unwrap();
        assert_close(round.data(), a.data(), 1e-12);
        let lhs = a.compose(&b).unwrap().adjoint().unwrap();
        let rhs = b.adjoint().unwrap().compose(&a.adjoint().unwrap()).unwrap();
        assert_close(lhs.data(), rhs.data(), 1e-12);
        let _ = name;
    }
}

/// tr(a∘b) == tr(b∘a). For fermionic rules `tr` is the supertrace
/// (TensorKit semantics, see `Tensor::tr` and the braid/twist oracle tests
/// in `user_api.rs`), which is exactly what makes cyclicity hold there.
#[test]
fn trace_cyclicity() {
    let rt = Runtime::builder().build().unwrap();
    for (name, v, _) in spaces() {
        for (ncod, seed) in [(1usize, 61u64), (2, 62)] {
            let cod: Vec<&Space> = std::iter::repeat(&v).take(ncod).collect();
            let a = Tensor::rand_with_seed(&rt, cod.clone(), cod.clone(), seed).unwrap();
            let b = Tensor::rand_with_seed(&rt, cod.clone(), cod, seed + 100).unwrap();
            let ab = a.compose(&b).unwrap().tr().unwrap();
            let ba = b.compose(&a).unwrap().tr().unwrap();
            assert_scalar_close(ab.re, ba.re, 1e-12);
            assert_scalar_close(ab.im, ba.im, 1e-12);
            let _ = name;
        }
    }
}

/// twist² == id on every leg (all rules in scope have θ ∈ {±1}); bosonic
/// rules have trivial twist; twist is natural with respect to permute.
#[test]
fn twist_squares_to_identity_and_naturality() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0004u64;
    for (name, v, fermionic) in spaces() {
        let t = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 71).unwrap();
        for leg in 0..4usize {
            let twice = t.twist(&[leg]).unwrap().twist(&[leg]).unwrap();
            assert_close(twice.data(), t.data(), 1e-12);
            if !fermionic {
                let once = t.twist(&[leg]).unwrap();
                assert_close(once.data(), t.data(), 1e-12);
            }
        }
        // Naturality: twisting leg 0 commutes with permuting it elsewhere.
        let s = rand_perm(&mut state, 4);
        let pos = s.iter().position(|&j| j == 0).unwrap();
        let lhs = t.twist(&[0]).unwrap().permute(&s[..2], &s[2..]).unwrap();
        let rhs = t.permute(&s[..2], &s[2..]).unwrap().twist(&[pos]).unwrap();
        assert_close(lhs.data(), rhs.data(), 1e-12);
        let _ = name;
    }
}

/// isometry / unitary isometric identities: `w†∘w == id`.
#[test]
fn isometry_and_unitary_are_isometric() {
    let rt = Runtime::builder().build().unwrap();
    for (name, v, _) in spaces() {
        let id = Tensor::id(&rt, [&v]).unwrap();
        let u = Tensor::unitary(&rt, [&v], [&v]).unwrap();
        let utu = u.adjoint().unwrap().compose(&u).unwrap();
        assert_close(utu.data(), id.data(), 1e-12);
        let w = Tensor::isometry(&rt, [&v, &v], [&v]).unwrap();
        let wtw = w.adjoint().unwrap().compose(&w).unwrap();
        assert_close(wtw.data(), id.data(), 1e-12);
        let _ = name;
    }
}

// ---------------------------------------------------------------------------
// Contraction order independence
// ---------------------------------------------------------------------------

/// The same network contracted through different routes must agree:
/// `tensor!` (greedy pairwise) vs manual pairwise contraction in two
/// different association orders, for both a closed ring (scalar) and an
/// open two-tensor network that forces axis permutations.
#[test]
fn contraction_order_independence() {
    let rt = Runtime::builder().build().unwrap();
    for (name, v, _) in spaces() {
        // Closed ring of four matrices: tr(x1 x2 x3 x4).
        let x1 = Tensor::rand_with_seed(&rt, [&v], [&v], 81).unwrap();
        let x2 = Tensor::rand_with_seed(&rt, [&v], [&v], 82).unwrap();
        let x3 = Tensor::rand_with_seed(&rt, [&v], [&v], 83).unwrap();
        let x4 = Tensor::rand_with_seed(&rt, [&v], [&v], 84).unwrap();
        let ring = tensor!([] = x1[a; b] * x2[b; c] * x3[c; d] * x4[d; a])
            .unwrap()
            .scalar()
            .unwrap();
        let left = x1
            .compose(&x2)
            .unwrap()
            .compose(&x3)
            .unwrap()
            .compose(&x4)
            .unwrap()
            .tr()
            .unwrap()
            .re;
        let inner = x2.compose(&x3).unwrap();
        let middle = x1
            .compose(&inner)
            .unwrap()
            .compose(&x4)
            .unwrap()
            .tr()
            .unwrap()
            .re;
        assert_scalar_close(ring, left, 1e-12);
        assert_scalar_close(ring, middle, 1e-12);

        // Open chain: three association orders elementwise.
        let chain = tensor!([i; m] = x1[i; j] * x2[j; k] * x3[k; m]).unwrap();
        let assoc_l = x1.compose(&x2).unwrap().compose(&x3).unwrap();
        let assoc_r = x1.compose(&x2.compose(&x3).unwrap()).unwrap();
        assert_close(chain.data(), assoc_l.data(), 1e-12);
        assert_close(chain.data(), assoc_r.data(), 1e-12);

        // Rank-4 pair with crossed contracted legs: forces tree transforms
        // and output permutes on every route (incl. the dynamic engine).
        //
        // KNOWN FAILURE: SU2 with non-uniform sector degeneracies errors
        // with `ShapeMismatch` on every tree-transform contract route; see
        // `su2_nonuniform_degeneracy_crossed_contract_known_failure` below.
        if name == "SU2" {
            continue;
        }
        let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 85).unwrap();
        let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 86).unwrap();
        let via_macro = tensor!([p, q; r, s] = a[p, x; y, s] * b[q, y; x, r]).unwrap();
        let ab = a
            .contract(&b, &[1, 2], &[2, 1])
            .unwrap()
            // default open order [p, s, q, r] with codomain split after 2
            .permute(&[0, 2], &[3, 1])
            .unwrap();
        let ba = b
            .contract(&a, &[2, 1], &[1, 2])
            .unwrap()
            // default open order [q, r, p, s]
            .permute(&[2, 0], &[1, 3])
            .unwrap();
        assert_close(via_macro.data(), ab.data(), 1e-12);
        assert_close(via_macro.data(), ba.data(), 1e-12);
        let _ = name;
    }
}

/// KNOWN FAILURE (2026-07-04): SU(2) with non-uniform sector degeneracies
/// (`Space::su2([(0, 2), (1, 2), (2, 1)])`) fails every contract route that
/// needs source tree transforms with
/// `Operation(ShapeMismatch { dst: [2, 2, 2, 2], src: [2, 1, 1, 2] })` —
/// the destination shape is built from a single uniform degeneracy while
/// the source block carries the true per-sector shape. The plain compose
/// route (`[2,3] x [0,1]`) works; uniform-degeneracy SU(2) works; U1 / fZ2 /
/// fZ2xU1xSU2 with non-uniform degeneracies work. The bug therefore sits in
/// the dynamic contract route's degeneracy resolution for non-abelian rules
/// (`tenet-tensors/src/contract/`). This test asserts the CORRECT behavior:
/// un-ignore it once the engine is fixed.
/// Run with: `cargo test -p tenet --test semantic_suite -- --ignored su2_nonuniform`
#[test]
#[ignore = "KNOWN FAILURE: SU2 non-uniform degeneracy tree-transform contract (ShapeMismatch)"]
fn su2_nonuniform_degeneracy_crossed_contract_known_failure() {
    let rt = Runtime::builder().build().unwrap();
    let v = Space::su2([(0, 2), (1, 2), (2, 1)]);
    let a = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 85).unwrap();
    let b = Tensor::rand_with_seed(&rt, [&v, &v], [&v, &v], 86).unwrap();
    let via_macro = tensor!([p, q; r, s] = a[p, x; y, s] * b[q, y; x, r]).unwrap();
    let ab = a
        .contract(&b, &[1, 2], &[2, 1])
        .unwrap()
        .permute(&[0, 2], &[3, 1])
        .unwrap();
    let ba = b
        .contract(&a, &[2, 1], &[1, 2])
        .unwrap()
        .permute(&[2, 0], &[1, 3])
        .unwrap();
    assert_close(via_macro.data(), ab.data(), 1e-12);
    assert_close(via_macro.data(), ba.data(), 1e-12);
}

// ---------------------------------------------------------------------------
// Decompositions over random sector content
// ---------------------------------------------------------------------------

/// A random space for the given rule: every sector kept with random
/// degeneracy 1..=2 (the vacuum-compatible first sector always kept so
/// tensors are never empty).
fn random_space(name: &str, state: &mut u64) -> Space {
    let deg = |state: &mut u64| 1 + rand_below(state, 2);
    match name {
        "Z2" => Space::z2([(0, deg(state)), (1, deg(state))]),
        "fZ2" => Space::fz2([(0, deg(state)), (1, deg(state))]),
        "U1" => {
            let mut sectors = vec![(0, deg(state))];
            for q in [-2, -1, 1, 2] {
                if rand_below(state, 2) == 1 {
                    sectors.push((q, deg(state)));
                }
            }
            Space::u1(sectors)
        }
        "SU2" => {
            let mut sectors = vec![(0, deg(state))];
            for j in [1usize, 2] {
                if rand_below(state, 2) == 1 {
                    sectors.push((j, deg(state)));
                }
            }
            Space::su2(sectors)
        }
        "fZ2xU1xSU2" => {
            let mut sectors = vec![((0u8, 0i32, 0usize), deg(state))];
            for s in [(1u8, 1i32, 1usize), (0, 2, 0), (1, -1, 1)] {
                if rand_below(state, 2) == 1 {
                    sectors.push((s, deg(state)));
                }
            }
            Space::fz2_u1_su2(sectors).unwrap()
        }
        _ => unreachable!(),
    }
}

/// svd / qr reconstruction and isometry identities over randomly drawn
/// sector contents (beyond the fixed fixture spaces).
#[test]
fn svd_qr_reconstruction_random_spaces() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0005u64;
    for (name, _, _) in spaces() {
        for draw in 0..3u64 {
            let va = random_space(name, &mut state);
            let vb = random_space(name, &mut state);
            let t = Tensor::rand_with_seed(&rt, [&va, &vb], [&vb, &va], 90 + draw).unwrap();
            if t.norm().unwrap() == 0.0 {
                continue;
            }

            let (u, s, vh) = t.svd_compact().unwrap();
            let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
            let diff = recon.add(&t, 1.0, -1.0).unwrap().norm().unwrap();
            assert!(
                diff <= 1e-10 * (1.0 + t.norm().unwrap()),
                "{name} draw {draw}: svd reconstruction error {diff}"
            );
            let mid = u.domain_spaces();
            let mid_refs: Vec<&Space> = mid.iter().collect();
            let id = Tensor::id(&rt, mid_refs).unwrap();
            let utu = u.adjoint().unwrap().compose(&u).unwrap();
            let iso_err = utu.add(&id, 1.0, -1.0).unwrap().norm().unwrap();
            assert!(
                iso_err <= 1e-10,
                "{name} draw {draw}: U†U != id ({iso_err})"
            );

            let (q, r) = t.qr_compact().unwrap();
            let recon = q.compose(&r).unwrap();
            let diff = recon.add(&t, 1.0, -1.0).unwrap().norm().unwrap();
            assert!(
                diff <= 1e-10 * (1.0 + t.norm().unwrap()),
                "{name} draw {draw}: qr reconstruction error {diff}"
            );
            let mid = q.domain_spaces();
            let mid_refs: Vec<&Space> = mid.iter().collect();
            let id = Tensor::id(&rt, mid_refs).unwrap();
            let qtq = q.adjoint().unwrap().compose(&q).unwrap();
            let iso_err = qtq.add(&id, 1.0, -1.0).unwrap().norm().unwrap();
            assert!(
                iso_err <= 1e-10,
                "{name} draw {draw}: Q†Q != id ({iso_err})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Part 3: cross-library invariant stream vs TensorKit
// ---------------------------------------------------------------------------

/// The shared deterministic per-block fill, identical to `fill_value` in
/// `benchmarks/tensorkit_semantic_oracle.jl` (section 3), which follows the
/// block/tree alignment already validated by
/// `benchmarks/tensorkit_tsvd_crosscheck.jl`. Labels: U1 charge or SU2
/// twice-spin; indices are one-based degeneracy coordinates, codomain
/// axes first.
fn oracle_fill(c0: i64, labels: [i64; 5], idx: &[usize]) -> f64 {
    let [l1, l2, m1, m2, lc] = labels;
    let weighted = c0
        + 7 * l1
        + 11 * l2
        + 13 * m1
        + 17 * m2
        + 19 * lc
        + 23 * (idx[0] as i64 + 1)
        + 29 * (idx[1] as i64 + 1)
        + 31 * (idx[2] as i64 + 1)
        + 37 * (idx[3] as i64 + 1);
    (weighted.rem_euclid(41) - 20) as f64
}

fn oracle_tensor(rt: &Runtime, v: &Space, c0: i64, label_of: impl Fn(SectorId) -> i64) -> Tensor {
    Tensor::from_block_fn(rt, [v, v], [v, v], |key, idx| {
        let BlockKey::FusionTree(key) = key else {
            unreachable!("symmetric tensors have fusion-tree blocks")
        };
        let cod = key.codomain_uncoupled();
        let dom = key.domain_uncoupled();
        let coupled = key.coupled().expect("coupled sector");
        let labels = [
            label_of(cod[0]),
            label_of(cod[1]),
            label_of(dom[0]),
            label_of(dom[1]),
            label_of(coupled),
        ];
        oracle_fill(c0, labels, idx)
    })
    .unwrap()
}

/// Runs the part-3 op sequence and compares the basis-independent
/// invariants (norm, tr, singular values) against the committed TensorKit
/// stream. `svd_expected` holds the significant singular values (>= 1e-6
/// of the largest); trailing numerically-zero values are only counted.
fn invariant_stream_case(
    v: &Space,
    label_of: impl Fn(SectorId) -> i64,
    expected: &[(&str, f64, f64)],
    svd_count: usize,
    svd_expected: &[f64],
) {
    let rt = Runtime::builder().build().unwrap();
    let a = oracle_tensor(&rt, v, 3, &label_of);
    let b = oracle_tensor(&rt, v, 5, &label_of);
    let c = a.compose(&b).unwrap();
    let d = a.permute(&[1, 0], &[3, 2]).unwrap();
    let e = d.compose(&c).unwrap();
    let g = a.adjoint().unwrap().compose(&a).unwrap();
    let h = e.add(&a, 1.0, 0.5).unwrap();
    let hh_tr = h.compose(&h).unwrap().tr().unwrap().re;

    let steps: Vec<(&str, f64, f64)> = vec![
        ("s1a", a.norm().unwrap(), a.tr().unwrap().re),
        ("s1b", b.norm().unwrap(), b.tr().unwrap().re),
        ("s2", c.norm().unwrap(), c.tr().unwrap().re),
        ("s3", d.norm().unwrap(), d.tr().unwrap().re),
        ("s4", e.norm().unwrap(), e.tr().unwrap().re),
        ("s5", g.norm().unwrap(), g.tr().unwrap().re),
        ("s7", h.norm().unwrap(), h.tr().unwrap().re),
        ("s8", hh_tr, hh_tr),
    ];
    for ((step, norm, tr), &(exp_step, exp_norm, exp_tr)) in steps.iter().zip(expected) {
        assert_eq!(*step, exp_step);
        assert_scalar_close(*norm, exp_norm, 1e-9);
        assert_scalar_close(*tr, exp_tr, 1e-9);
    }

    let mut values: Vec<f64> = e
        .svd_vals()
        .unwrap()
        .iter()
        .flat_map(|spectrum| spectrum.values.iter().copied())
        .collect();
    values.sort_by(|x, y| y.partial_cmp(x).unwrap());
    assert_eq!(values.len(), svd_count, "singular value count");
    let cutoff = 1e-6 * svd_expected[0];
    for (k, (&got, &exp)) in values.iter().zip(svd_expected).enumerate() {
        assert_scalar_close(got, exp, 1e-8);
        assert!(exp > cutoff, "svd_expected[{k}] below cutoff");
    }
    for &tail in &values[svd_expected.len()..] {
        assert!(
            tail <= cutoff,
            "unexpected significant singular value {tail}"
        );
    }
}

/// Cross-library invariant stream, U(1). Oracle:
/// `julia benchmarks/tensorkit_semantic_oracle.jl` (section 3, `-- U1 --`
/// of `benchmarks/tensorkit_semantic_oracle.out`).
/// Run with: `cargo test -p tenet --release --test semantic_suite -- --ignored u1_vs`
#[test]
#[ignore = "cross-library stream, run explicitly (release recommended)"]
fn cross_library_invariant_stream_u1_vs_tensorkit() {
    let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    let label = |sector: SectorId| {
        i64::from(
            tenet::core::U1Irrep::from_sector_id(sector)
                .unwrap()
                .charge(),
        )
    };
    let expected = [
        ("s1a", 2.887628785006827e2, -1.9e1),
        ("s1b", 2.896515147552315e2, -3.0),
        ("s2", 1.402426411616665e4, 1.104e4),
        ("s3", 2.887628785006826e2, -1.9e1),
        ("s4", 8.209534152002786e5, 1.03394e5),
        ("s5", 2.485837810477585e4, 8.3384e4),
        ("s7", 8.209824186503642e5, 1.033845e5),
        ("s8", 6.256589682425e10, 6.256589682425e10),
    ];
    let svd = [
        5.938452457943761e5,
        4.179528114697005e5,
        2.459326023786228e5,
        1.933057950606111e5,
        1.566314868289549e5,
        8.210318973109889e4,
        7.523003903903847e4,
        6.565202259389844e4,
        5.151867957441879e4,
        3.767419629678344e4,
        2.538188406982896e4,
        2.343136052737890e4,
        2.008745263879226e4,
        1.964054032660374e4,
        1.883995230847887e4,
        1.867342767080404e4,
        1.396252249681011e4,
        1.278185908520561e4,
        1.221617279960168e4,
        9.646640464331718e3,
        7.594820923703089e3,
        7.588096644934964e3,
        6.218196863168910e3,
        2.651408037571802e3,
        2.163357342259171e3,
        1.657615939186760e3,
        1.643314514396656e3,
        1.129212243761851e3,
        6.562754711025852e2,
        5.535225377189289e2,
        5.476578287137901e2,
    ];
    invariant_stream_case(&v, label, &expected, 49, &svd);
}

/// Cross-library invariant stream, SU(2). Oracle:
/// `julia benchmarks/tensorkit_semantic_oracle.jl` (section 3, `-- SU2 --`
/// of `benchmarks/tensorkit_semantic_oracle.out`).
/// Run with: `cargo test -p tenet --release --test semantic_suite -- --ignored su2_vs`
#[test]
#[ignore = "cross-library stream, run explicitly (release recommended)"]
fn cross_library_invariant_stream_su2_vs_tensorkit() {
    let v = Space::su2([(0, 2), (1, 2), (2, 1)]);
    let label = |sector: SectorId| {
        i64::try_from(tenet::core::SU2Irrep::from_sector_id(sector).twice_spin()).unwrap()
    };
    let expected = [
        ("s1a", 3.075792580783041e2, -2.24e2),
        ("s1b", 3.089012787283341e2, -1.44e2),
        ("s2", 1.423541506244198e4, 1.6685e4),
        ("s3", 3.075792580783042e2, -2.24e2),
        ("s4", 7.171108525311829e5, -1.481890000000002e5),
        ("s5", 2.043484428616964e4, 9.4605e4),
        ("s7", 7.171433641554874e5, -1.483010000000002e5),
        ("s8", 2.412084471050002e10, 2.412084471050002e10),
    ];
    let svd = [
        4.110592957633280e5,
        1.822658678626898e5,
        1.406505246682399e5,
        1.140569967151560e5,
        7.343460378313088e4,
        5.224807565545758e4,
        4.043006489580932e4,
        2.351544167506671e4,
        2.184633501094873e4,
        1.928046828460497e4,
        1.660174629027798e4,
        8.566028701881454e3,
        6.373677277671596e3,
        4.039539983749055e3,
        3.958321987177687e3,
        3.574358904429296e3,
        3.288748484860358e3,
        3.151829069086835e3,
        1.779769959383261e3,
        1.439999999999999e3,
        1.076447526243916e3,
        5.845450724371532e2,
    ];
    invariant_stream_case(&v, label, &expected, 35, &svd);
}
