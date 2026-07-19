//! REFUTE harness for Stage B3c-1 (feat/su3-contract-b3c1).
//!
//! The shipped anchor (`tk_su3_correspondence.rs`) uses degeneracy-1 legs and a
//! SYMMETRIC fill `[3,5]` for both operands, so `M = 3·3 + 5·5 = 34`. That rules
//! out a cross-vertex swap (it would give `3·5 + 5·3 = 30`) but NOT:
//!   * degeneracy-index pairing (the anchor has no dense degeneracy index), or
//!   * a transpose/offset error in the coupled matrix that survives symmetry.
//!
//! This test does the independent evaluation the task asks for: a genuine
//! rank-3 ∘ rank-3 SU(3) compose over the `N(8,8,8)=2` OM vertex WITH degeneracy
//! 2 on every leg and DISTINCT values per (vertex, degeneracy) — then reproduces
//! the coupled-matrix sum by hand from the same fill definitions and checks it
//! against `contract`, including layout-independent invariants (element sum,
//! Frobenius norm).

use tenet::prelude::*;

/// The adjoint `8 = (1,1)` with degeneracy 2 (a real dense degeneracy index).
fn eight2() -> Space {
    Space::su3([((1, 1), 2)]).unwrap()
}

/// Distinct value for A's block element (vertex μ, codomain-deg i, domain-degs
/// k1,k2). Injective across all (μ,i,k1,k2).
fn a_fill(mu: usize, i: usize, k1: usize, k2: usize) -> f64 {
    1.0 + 100.0 * mu as f64 + 10.0 * i as f64 + 3.0 * k1 as f64 + 7.0 * k2 as f64
}

/// Distinct value for B's block element (vertex μ, codomain-degs k1,k2,
/// domain-deg j).
fn b_fill(mu: usize, k1: usize, k2: usize, j: usize) -> f64 {
    2.0 + 50.0 * mu as f64 + 4.0 * k1 as f64 + 9.0 * k2 as f64 + 11.0 * j as f64
}

fn vertex(tree: &tenet_core::FusionTreeKey) -> usize {
    tree.vertices().first().map(|s| s.get()).unwrap_or(0)
}

#[test]
fn refute_su3_om_layout_independent_eval() {
    let rt = Runtime::builder().build().unwrap();
    let v = eight2();

    // A : [8] <- [8,8]. Block index = [i (codomain 8 deg), k1, k2 (domain 8,8 deg)].
    let a = Tensor::from_block_fn(&rt, [&v], [&v, &v], |key, idx| match key {
        BlockKey::FusionTree(k) => {
            let mu = vertex(k.domain_tree());
            a_fill(mu, idx[0], idx[1], idx[2])
        }
        _ => 0.0,
    })
    .unwrap();

    // B : [8,8] <- [8]. Block index = [k1, k2 (codomain 8,8 deg), j (domain 8 deg)].
    let b = Tensor::from_block_fn(&rt, [&v, &v], [&v], |key, idx| match key {
        BlockKey::FusionTree(k) => {
            let mu = vertex(k.codomain_tree());
            b_fill(mu, idx[0], idx[1], idx[2])
        }
        _ => 0.0,
    })
    .unwrap();

    // A must have exactly two OM-vertex blocks, each 2x4 (deg 2 codomain, 2x2 domain).
    assert_eq!(a.data().len(), 2 * (2 * 4), "A = 2 vertex blocks x 2x4");
    assert_eq!(b.data().len(), 2 * (4 * 2), "B = 2 vertex blocks x 4x2");

    // Independent hand computation of the compose over coupled 8:
    //   M[i,j] = Σ_μ Σ_{k1,k2} A(μ,i,k1,k2) · B(μ,k1,k2,j)
    // NB: the SU(3) `N(8,8,8)=2` OM vertex labels are ids {1,2} (NOT {0,1});
    // that is exactly what the shipped anchor's `[3,5] = 1+2·{1,2}` encodes.
    let mut expected = [[0.0f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut s = 0.0;
            for mu in [1usize, 2] {
                for k1 in 0..2 {
                    for k2 in 0..2 {
                        s += a_fill(mu, i, k1, k2) * b_fill(mu, k1, k2, j);
                    }
                }
            }
            expected[i][j] = s;
        }
    }

    let m = a.contract(&b, &[1, 2], &[0, 1]).unwrap();
    assert_eq!(m.rank(), 2);
    assert_eq!(m.codomain_rank(), 1);
    assert_eq!(m.data().len(), 4, "single coupled-8 block, 2x2");

    // --- Layout-INDEPENDENT invariants (no assumption on result strides) ---
    let sum_all: f64 = m.data().iter().sum();
    let exp_sum: f64 = expected.iter().flatten().sum();
    assert!(
        (sum_all - exp_sum).abs() < 1e-6,
        "element sum: contract {sum_all} vs independent {exp_sum}"
    );

    // Frobenius: norm² = dim(8) · Σ_ij M[i,j]²  (dim 8 = 8).
    let exp_fro_sq: f64 = expected.iter().flatten().map(|x| x * x).sum();
    let n = m.norm().unwrap();
    assert!(
        (n * n - 8.0 * exp_fro_sq).abs() < 1e-4,
        "norm²: contract {} vs 8·ΣM² {}",
        n * n,
        8.0 * exp_fro_sq
    );

    // --- Element-wise (column-major result layout, the codebase-wide convention) ---
    // M[i,j] at data[i + 2*j]. Cross-checked by the two invariants above; a wrong
    // stride assumption here would clash with a matching sum/norm, flagging it.
    for i in 0..2 {
        for j in 0..2 {
            let got = m.data()[i + 2 * j];
            assert!(
                (got - expected[i][j]).abs() < 1e-6,
                "M[{i},{j}] = {got} vs independent {}",
                expected[i][j]
            );
        }
    }
}

/// Stage B3c-2 adjoint axiom `(A∘B)† == B†∘A†` on OM-vertex SU(3) tensors.
/// A lazy-adjoint operand is materialized (not folded) inside the SU(N)
/// contract path, so both sides ride the direct core/compose GEMM.
#[test]
fn su3_om_adjoint_reverses_composition() {
    let rt = Runtime::builder().build().unwrap();
    let v = eight2();
    let a = Tensor::from_block_fn(&rt, [&v], [&v, &v], |key, idx| match key {
        BlockKey::FusionTree(k) => a_fill(vertex(k.domain_tree()), idx[0], idx[1], idx[2]),
        _ => 0.0,
    })
    .unwrap();
    let b = Tensor::from_block_fn(&rt, [&v, &v], [&v], |key, idx| match key {
        BlockKey::FusionTree(k) => b_fill(vertex(k.codomain_tree()), idx[0], idx[1], idx[2]),
        _ => 0.0,
    })
    .unwrap();

    // LHS: (A∘B)† — compose over the two 8-legs, then dagger. [8] <- [8].
    let lhs = a
        .contract(&b, &[1, 2], &[0, 1])
        .unwrap()
        .adjoint()
        .unwrap()
        .scale(1.0)
        .unwrap();
    // RHS: B†∘A† — dagger each, then compose. B†:[8]<-[8,8], A†:[8,8]<-[8].
    let rhs = b
        .adjoint()
        .unwrap()
        .contract(&a.adjoint().unwrap(), &[1, 2], &[0, 1])
        .unwrap()
        .scale(1.0)
        .unwrap();

    assert_eq!(lhs.codomain_rank(), 1);
    assert_eq!(lhs.data().len(), rhs.data().len());
    assert!(lhs.data().iter().any(|&x| x.abs() > 1e-9), "non-trivial");
    for (x, y) in lhs.data().iter().zip(rhs.data().iter()) {
        assert!((x - y).abs() < 1e-10, "(A∘B)† vs B†∘A†: {x} vs {y}");
    }
}

/// FLIPPED (Stage B3c-2): the B3c-1 boundary pinned `.adjoint()` on an SU(3)
/// tensor as a loud panic. B3c-2 implements it via the generic block-relabel
/// materialization, so the adjoint now SUCCEEDS on an OM-vertex tensor — it
/// swaps codomain/domain, preserves the quantum-dim-weighted norm, and is an
/// involution (A†† == A). The `8` leg carries the `N(8,8,8)=2` outer
/// multiplicity, so this exercises the OM-sector adjoint, not just an abelian
/// relabel.
#[test]
fn su3_om_adjoint_swaps_and_is_involution() {
    let rt = Runtime::builder().build().unwrap();
    let v = eight2();
    // A : [8] <- [8,8], genuinely non-zero on both OM vertices.
    let a = Tensor::from_block_fn(&rt, [&v], [&v, &v], |key, idx| match key {
        BlockKey::FusionTree(k) => {
            let mu = vertex(k.domain_tree());
            a_fill(mu, idx[0], idx[1], idx[2])
        }
        _ => 0.0,
    })
    .unwrap();

    // Adjoint swaps the sides: [8] <- [8,8]  ==>  [8,8] <- [8].
    let ah = a.adjoint().unwrap();
    assert_eq!(ah.rank(), 3);
    assert_eq!(ah.codomain_rank(), 2);
    assert!((ah.norm().unwrap() - a.norm().unwrap()).abs() < 1e-12);

    // A†† == A at the data level (force materialization with scale(1.0) so the
    // block-relabel path runs, not the lazy involution short-circuit).
    let ahh = a.adjoint().unwrap().scale(1.0).unwrap().adjoint().unwrap();
    let ahh = ahh.scale(1.0).unwrap();
    assert_eq!(ahh.data().len(), a.data().len());
    for (x, y) in ahh.data().iter().zip(a.data().iter()) {
        assert!((x - y).abs() < 1e-12, "OM A†† round-trip: {x} vs {y}");
    }
}
