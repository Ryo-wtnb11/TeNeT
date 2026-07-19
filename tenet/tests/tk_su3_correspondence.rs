//! Numeric correspondence against TensorKit + SUNRepresentations for SU(3)
//! (`FusionStyleKind::Generic`, outer-multiplicity) closed-loop scalars — the
//! Stage B3c-1 permanent parity gate for the Generic contract + norm wiring
//! (the SU(3) analogue of the SU(2) χ32 anchor).
//!
//! # What is pinned, and why it is basis-independent
//!
//! Both scalars are built from a genuine SU(3) tensor whose adjoint coupled
//! sector `8 = (1,1)` is reached through the **outer-multiplicity vertex**
//! `N(8,8,8) = 2`: the `(8,8) → 8` fusion has TWO independent trees. The
//! contraction below genuinely SUMS over that vertex (verified structurally in
//! `su3_om_vertex_actually_participates`, not assumed).
//!
//! The pinned quantities are `norm(A)²` and `norm(A ∘ B)²`. Both are
//! `Σ` over the OM vertex of squares (`Σ aμ²`), which is **invariant under the
//! vertex-basis orthogonal rotation** (the residual gauge freedom of a real OM
//! fusion space) — so a mismatch is a genuine convention/structure divergence,
//! not a layout/gauge artifact.
//!
//! # TensorKit reference (provenance)
//!
//! Full `TensorMap{SUNIrrep{3}}` construction is broken in the pinned toolchain
//! (TensorKit 0.17 `jCjQQ` + SUNRepresentations 0.4.0 + TensorKitSectors 0.3.4):
//! `id(W)` / `zeros(Float64, V, V)` throw `SectorMismatch` from TensorKit's
//! `fusiontrees(::ProductSpace, blocksector)` block iterator (exactly why
//! Stage B3b validated against TK's `artin_braid`/`bendright` at the
//! FusionTreeBlock level, not full tensors). The B3c-1 contract **Core**
//! (compose) route and `norm` use NO F/R recoupling — a compose over matching
//! domain/codomain trees is a coefficient-free block matrix product (TensorKit
//! `mul!` parity), and `norm(t)² = Σ_c dim(c)·‖block_c‖²` (TensorKit
//! `vectorinterface.jl`). So the reference reduces to TWO SUNRepresentations
//! quantities plus those definitions, computed independently in
//! `scratchpad/tk_su3_anchor_reference.jl` (offline `/tmp/combenv`):
//!
//! ```text
//! dim(8)   = 8         # SUNRepresentations dim(SUNIrrep{3}(1,1))
//! N(8,8,8) = 2         # SUNRepresentations Nsymbol — the OM vertex
//! fill aμ  = [3, 5]    # deterministic 1 + 2·(vertex label), both A and B
//! norm(A)² = dim(8)·Σaμ²   = 8·(9+25)  = 272
//! M(A∘B)   = Σ aμ·bμ (bμ=aμ) = 9+25    = 34    (the OM sum)
//! norm(M)² = dim(8)·M²       = 8·34²    = 9248
//! ```
//!
//! The F/R recoupling (where a Generic bug could hide) was already TK-oracled in
//! Stage B3b's permute/braid OM anchors; the Core contract route deliberately
//! uses none, so these scalars close the contract + norm wiring.

use std::cell::RefCell;

use tenet::prelude::*;

/// The adjoint `8 = (1,1)`, degeneracy 1 (so every coupled block is a bare
/// scalar per fusion tree — the whole tensor is its OM structure, nothing else).
fn eight() -> Space {
    Space::su3([((1, 1), 1)]).unwrap()
}

/// Deterministic vertex fill: `1 + 2·μ` over the `(8,8) → 8` vertex label `μ`
/// (the SU(3) table labels the two vertices `1` and `2`, giving values `3, 5`).
/// `dom` selects the domain tree (for `A`, codomain `[8]`) or the codomain tree
/// (for `B`, codomain `[8,8]`) — whichever carries the two-leg `(8,8)` fusion.
fn vertex_fill(key: &FusionTreePairKey, dom: bool) -> f64 {
    let tree = if dom {
        key.domain_tree()
    } else {
        key.codomain_tree()
    };
    let mu = tree.vertices().first().map(|s| s.id()).unwrap_or(0);
    1.0 + 2.0 * mu as f64
}

fn norm_sq(t: &Tensor) -> f64 {
    let n = t.norm().unwrap();
    n * n
}

/// `A : [8] <- [8,8]` — its coupled-8 block is `1 × 2` (two `(8,8)→8` vertices).
fn build_a(rt: &Runtime) -> Tensor {
    let v = eight();
    Tensor::from_block_fn(rt, [&v], [&v, &v], |key, _| match key {
        BlockKey::FusionTree(k) => vertex_fill(k, true),
        _ => 0.0,
    })
    .unwrap()
}

/// `B : [8,8] <- [8]` — its coupled-8 block is `2 × 1`, filled identically.
fn build_b(rt: &Runtime) -> Tensor {
    let v = eight();
    Tensor::from_block_fn(rt, [&v, &v], [&v], |key, _| match key {
        BlockKey::FusionTree(k) => vertex_fill(k, false),
        _ => 0.0,
    })
    .unwrap()
}

/// Structural proof (NOT assumed) that the anchor's coupled sector is genuinely
/// reached through the `N(8,8,8) = 2` outer-multiplicity vertex: `A` must have
/// two blocks with identical external sectors `(8; 8,8)` differing ONLY in the
/// domain-tree vertex label.
#[test]
fn su3_om_vertex_actually_participates() {
    let rt = Runtime::builder().build().unwrap();
    let v = eight();
    let seen: RefCell<Vec<(Vec<usize>, Vec<usize>, usize)>> = RefCell::new(Vec::new());
    let _a = Tensor::from_block_fn(&rt, [&v], [&v, &v], |key, _| {
        if let BlockKey::FusionTree(k) = key {
            let cod: Vec<usize> = k
                .codomain_tree()
                .uncoupled()
                .iter()
                .map(|s| s.id())
                .collect();
            let dom: Vec<usize> = k.domain_tree().uncoupled().iter().map(|s| s.id()).collect();
            let mu = k
                .domain_tree()
                .vertices()
                .first()
                .map(|s| s.id())
                .unwrap_or(0);
            seen.borrow_mut().push((cod, dom, mu));
        }
        0.0
    })
    .unwrap();
    let blocks = seen.into_inner();
    // Exactly two blocks, same external (8; 8,8), distinct vertex labels ⇒ the
    // coupled sector is reached ONLY through the 2-fold OM vertex.
    assert_eq!(
        blocks.len(),
        2,
        "expected the two OM vertex blocks, got {blocks:?}"
    );
    assert_eq!(blocks[0].0, blocks[1].0, "same codomain externals");
    assert_eq!(blocks[0].1, blocks[1].1, "same domain externals (8,8)");
    assert_ne!(
        blocks[0].2, blocks[1].2,
        "the two blocks MUST differ only in the OM vertex label"
    );
}

/// Anchor (b): the norm of an OM-carrying tensor — `norm(A)² = dim(8)·Σaμ²`
/// sums BOTH vertex blocks weighted by the quantum dimension.
///
/// TensorKit + SUNRepresentations: `8·(3² + 5²) = 272` (see module docs).
#[test]
fn su3_norm_anchor_matches_tensorkit() {
    let rt = Runtime::builder().build().unwrap();
    let a = build_a(&rt);
    assert_eq!(a.data().len(), 2, "A is exactly its two OM vertex blocks");
    assert!(
        (norm_sq(&a) - 272.0).abs() < 1e-9,
        "norm(A)^2 = {} vs TK 272",
        norm_sq(&a)
    );
}

/// Anchor (a): a genuine contraction that SUMS over the OM vertex, closed to a
/// scalar. `M = A ∘ B` (Core/compose route) contracts the `(8,8)` legs at
/// coupled `8`, whose contracted basis is the TWO `(8,8)→8` vertex trees — so
/// the single `[8]<-[8]` block value is `Σ aμ·bμ = 34`, and `norm(M)² =
/// dim(8)·34² = 9248`. Reproducing `34` requires correctly summing the N=2
/// vertex; a bug that dropped or mis-paired a vertex would miss it.
///
/// TensorKit + SUNRepresentations: `M-block = 34`, `norm(M)² = 9248`.
#[test]
fn su3_om_contraction_scalar_matches_tensorkit() {
    let rt = Runtime::builder().build().unwrap();
    let a = build_a(&rt);
    let b = build_b(&rt);
    // Core/compose: contract ALL of A's domain (8,8) with ALL of B's codomain
    // (8,8) ⇒ M : [8] <- [8]. The contracted tree basis IS the OM vertex.
    let m = a.contract(&b, &[1, 2], &[0, 1]).unwrap();
    assert_eq!(m.rank(), 2, "M is a [8]<-[8] endomorphism");
    assert_eq!(m.codomain_rank(), 1);
    assert_eq!(m.data().len(), 1, "single coupled-8 block");
    assert!(
        (m.data()[0] - 34.0).abs() < 1e-9,
        "M block = {} vs TK 34 (the OM vertex sum 3·3 + 5·5)",
        m.data()[0]
    );
    assert!(
        (norm_sq(&m) - 9248.0).abs() < 1e-9,
        "norm(M)^2 = {} vs TK 9248",
        norm_sq(&m)
    );
}

/// FLIPPED (Stage B3c-2): the B3c-1 boundary pinned a non-core (open
/// contracted legs) SU(N) contraction as a clear error. B3c-2 wires the
/// source-transform route — a generic permute to core form followed by the
/// core GEMM — so contracting only ONE of A's two domain legs now succeeds
/// and must equal the explicit permute-then-core reference exactly (the
/// wiring adds no mathematics; see also `su3_b3c2_gates.rs`).
#[test]
fn su3_noncore_contract_matches_permute_then_core() {
    let rt = Runtime::builder().build().unwrap();
    let a = build_a(&rt);
    let b = build_b(&rt);
    let got = a.contract(&b, &[1], &[0]).unwrap();
    let want = a
        .permute(&[0, 2], &[1])
        .unwrap()
        .contract(&b.permute(&[0], &[1, 2]).unwrap(), &[2], &[0])
        .unwrap();
    assert_eq!(got.data().len(), want.data().len());
    assert!(!got.data().is_empty());
    for (x, y) in got.data().iter().zip(want.data().iter()) {
        assert!((x - y).abs() < 1e-12, "non-core route: {x} vs {y}");
    }
}
