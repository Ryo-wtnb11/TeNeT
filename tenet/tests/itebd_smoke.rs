//! Physics smoke test: iTEBD ground-state search for the spin-1/2 AFM
//! Heisenberg chain with U(1) (Sz) symmetry, user-layer API only. A compact
//! version of `examples/itebd_heisenberg.rs`; see that file for the physics
//! and charge-convention notes. Exact energy per bond: `1/4 - ln 2`.

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::Truncation;
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::tensor;

const E_EXACT: f64 = 0.25 - std::f64::consts::LN_2;
type Map = TensorMap<U1FusionRule, f64>;

fn space<const N: usize>(
    rule: &Arc<U1FusionRule>,
    sectors: [(i32, usize); N],
) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new_with_shared_provider(
        Arc::clone(rule),
        sectors.map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn heisenberg_two_site(rt: &Runtime, p: &GradedSpace<U1FusionRule>) -> Map {
    TensorMap::from_block_fn(rt, [p, p], [p, p], |trees, _| {
        let cod = trees.codomain_uncoupled();
        let dom = trees.domain_uncoupled();
        if cod == dom {
            if cod[0] == cod[1] {
                0.25
            } else {
                -0.25
            }
        } else {
            0.5
        }
    })
    .unwrap()
}

/// One Vidal bond update; returns `(g1', l_mid', g2')`.
fn bond_update(
    gate: &Map,
    l_out: &Map,
    g1: &Map,
    l_mid: &Map,
    g2: &Map,
    trunc: &Truncation,
) -> (Map, Map, Map) {
    let theta = tensor!([l, pa; pb, r] = l_out[l; x] * g1[x, qa; y] * l_mid[y; z]
        * g2[z, qb; w] * l_out[w; r] * gate[pa, pb; qa, qb])
    .unwrap();
    let svd = theta.svd_trunc(trunc).unwrap();
    let l_new = svd.s.scale(1.0 / svd.s.norm().unwrap());
    let l_out_inv = l_out.pinv(1e-12).unwrap();
    let (u, vh) = (svd.u, svd.vh);
    let g1_new = tensor!([l, pa; m] = l_out_inv[l; x] * u[x, pa; m]).unwrap();
    let g2_new = tensor!([m, pb; r] = vh[m; pb, x] * l_out_inv[x; r]).unwrap();
    (g1_new, l_new, g2_new)
}

fn bond_energy(h: &Map, l_out: &Map, g1: &Map, l_mid: &Map, g2: &Map) -> f64 {
    let theta = tensor!([l, pa, pb; r] = l_out[l; x] * g1[x, pa; y] * l_mid[y; z]
        * g2[z, pb; w] * l_out[w; r])
    .unwrap();
    let num = tensor!([] = conj(theta)[l, pa, pb; r] * h[pa, pb; qa, qb] * theta[l, qa, qb; r])
        .unwrap()
        .scalar()
        .unwrap();
    num / theta.inner(&theta).unwrap()
}

/// Runs the schedule from a Neel state; returns the energy per bond.
fn run_itebd(chi: usize, schedule: &[(f64, usize)]) -> f64 {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let p = space(&rule, [(1, 1), (-1, 1)]);
    let h = heisenberg_two_site(&rt, &p);
    let trunc = Truncation::rank(chi).and(Truncation::relative_cutoff(1e-8).unwrap());

    // Charge-balanced entangled start (converges faster than a product
    // state; a strict Neel start also works, see the regression test below).
    let vb = space(&rule, [(0, 1)]);
    let va = space(&rule, [(1, 1), (-1, 1)]);
    let mut ga = TensorMap::from_block_fn(&rt, [&vb, &p], [&va], |_, _| 1.0).unwrap();
    let mut la = TensorMap::from_block_fn(&rt, [&va], [&va], |_, _| 1.0).unwrap();
    let mut gb = TensorMap::from_block_fn(&rt, [&va, &p], [&vb], |_, _| 1.0).unwrap();
    let mut lb = TensorMap::from_block_fn(&rt, [&vb], [&vb], |_, _| 1.0).unwrap();

    for &(dt, steps) in schedule {
        let gate = h.scale(-dt).exp().unwrap();
        for _ in 0..steps {
            (ga, la, gb) = bond_update(&gate, &lb, &ga, &la, &gb, &trunc);
            (gb, lb, ga) = bond_update(&gate, &la, &gb, &lb, &ga, &trunc);
        }
    }
    0.5 * (bond_energy(&h, &lb, &ga, &la, &gb) + bond_energy(&h, &la, &gb, &lb, &ga))
}

/// Regression: a strict Neel product state built on the FULL physical space.
///
/// Each site tensor has a physical leg whose spaces contain both charges but
/// where one charge participates in NO fusion tree on that tensor (the
/// singleton bond legs fix the total charge): site A populates only `up`,
/// site B only `dn`. Legs used to carry sector sets without degeneracies, so
/// leg dimensions and result-block shapes were derived from populated blocks
/// only, and the contraction with the gate's full physical leg was rejected.
/// With graded legs (sector -> degeneracy on the leg itself) it must work.
#[test]
fn neel_product_state_contracts_with_the_full_gate() {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let p = space(&rule, [(1, 1), (-1, 1)]);
    let h = heisenberg_two_site(&rt, &p);

    // |up dn>: bonds {0} -> {+1} -> {0}; a has no tree with phys charge -1,
    // b none with +1.
    let vl = space(&rule, [(0, 1)]);
    let vm = space(&rule, [(1, 1)]);
    let vr = space(&rule, [(0, 1)]);
    let a = TensorMap::from_block_fn(&rt, [&vl, &p], [&vm], |_, _| 1.0).unwrap();
    let b = TensorMap::from_block_fn(&rt, [&vm, &p], [&vr], |_, _| 1.0).unwrap();

    // The legs report the full graded space, not just populated sectors.
    assert_eq!(a.leg_dims().unwrap(), vec![1, 2, 1]);
    assert_eq!(a.codomain()[1], p);

    let psi = tensor!([l, pa, pb; r] = a[l, pa; m] * b[m, pb; r]).unwrap();
    assert!((psi.norm().unwrap() - 1.0).abs() < 1e-12);

    // theta = h |psi>: this contraction used to be rejected with a leg
    // dimension mismatch against the gate's full physical leg.
    let theta = tensor!([l, pa, pb; r] = a[l, qa; m] * b[m, qb; r] * h[pa, pb; qa, qb]).unwrap();

    // h |up dn> = -1/4 |up dn> + 1/2 |dn up>, so <psi|h|psi> = -1/4 and
    // |h psi|^2 = 1/16 + 1/4 = 5/16.
    let energy = tensor!([] = conj(psi)[l, pa, pb; r] * h[pa, pb; qa, qb] * psi[l, qa, qb; r])
        .unwrap()
        .scalar()
        .unwrap();
    assert!((energy - (-0.25)).abs() < 1e-12, "energy = {energy}");
    let theta_norm = theta.norm().unwrap();
    assert!(
        (theta_norm - (5.0f64 / 16.0).sqrt()).abs() < 1e-12,
        "|h psi| = {theta_norm}"
    );
}

/// Fast variant: tiny chi and few steps, loose tolerance.
#[test]
fn itebd_heisenberg_reaches_the_ground_state_energy_coarsely() {
    let energy = run_itebd(8, &[(0.1, 100), (0.05, 100)]);
    assert!(
        (energy - E_EXACT).abs() < 5e-2,
        "E/bond = {energy} vs exact {E_EXACT}"
    );
}

/// Short but tighter run for release-mode machines:
/// `cargo test -p tenet --release -- --ignored itebd`.
#[test]
#[ignore = "several hundred iTEBD steps; run in release mode"]
fn itebd_heisenberg_short_run_release() {
    let energy = run_itebd(16, &[(0.1, 300), (0.05, 300), (0.01, 300)]);
    assert!(
        (energy - E_EXACT).abs() < 5e-3,
        "E/bond = {energy} vs exact {E_EXACT}"
    );
}
