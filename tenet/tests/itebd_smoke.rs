//! Physics smoke test: iTEBD ground-state search for the spin-1/2 AFM
//! Heisenberg chain with U(1) (Sz) symmetry, user-layer API only. A compact
//! version of `examples/itebd_heisenberg.rs`; see that file for the physics
//! and charge-convention notes. Exact energy per bond: `1/4 - ln 2`.

use tenet::prelude::*;
use tenet_network::tensor;

const E_EXACT: f64 = 0.25 - std::f64::consts::LN_2;

fn heisenberg_two_site(rt: &Runtime, p: &Space) -> Tensor {
    Tensor::from_block_fn(rt, [p, p], [p, p], |key, _| {
        let BlockKey::FusionTree(key) = key else {
            return 0.0;
        };
        let cod = key.codomain_uncoupled();
        let dom = key.domain_uncoupled();
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
    gate: &Tensor,
    l_out: &Tensor,
    g1: &Tensor,
    l_mid: &Tensor,
    g2: &Tensor,
    trunc: &Truncation,
) -> (Tensor, Tensor, Tensor) {
    let theta = tensor!([l, pa; pb, r] = l_out[l; x] * g1[x, qa; y] * l_mid[y; z]
        * g2[z, qb; w] * l_out[w; r] * gate[pa, pb; qa, qb])
    .unwrap();
    let svd = theta.svd_trunc(trunc).unwrap();
    let l_new = svd.s.scale(1.0 / svd.s.norm().unwrap()).unwrap();
    let l_out_inv = l_out.pinv(1e-12).unwrap();
    let (u, vh) = (svd.u, svd.vh);
    let g1_new = tensor!([l, pa; m] = l_out_inv[l; x] * u[x, pa; m]).unwrap();
    let g2_new = tensor!([m, pb; r] = vh[m; pb, x] * l_out_inv[x; r]).unwrap();
    (g1_new, l_new, g2_new)
}

fn bond_energy(h: &Tensor, l_out: &Tensor, g1: &Tensor, l_mid: &Tensor, g2: &Tensor) -> f64 {
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
    let p = Space::u1([(1, 1), (-1, 1)]);
    let h = heisenberg_two_site(&rt, &p);
    let trunc = Truncation::rank(chi).and(Truncation::relative_cutoff(1e-8));

    // Charge-balanced start so every leg's sectors are populated (the
    // network validator sizes legs from populated blocks; see the example).
    let vb = Space::u1([(0, 1)]);
    let va = Space::u1([(1, 1), (-1, 1)]);
    let mut ga = Tensor::from_block_fn(&rt, [&vb, &p], [&va], |_, _| 1.0).unwrap();
    let mut la = Tensor::from_block_fn(&rt, [&va], [&va], |_, _| 1.0).unwrap();
    let mut gb = Tensor::from_block_fn(&rt, [&va, &p], [&vb], |_, _| 1.0).unwrap();
    let mut lb = Tensor::from_block_fn(&rt, [&vb], [&vb], |_, _| 1.0).unwrap();

    for &(dt, steps) in schedule {
        let gate = h.scale(-dt).unwrap().exp().unwrap();
        for _ in 0..steps {
            (ga, la, gb) = bond_update(&gate, &lb, &ga, &la, &gb, &trunc);
            (gb, lb, ga) = bond_update(&gate, &la, &gb, &lb, &ga, &trunc);
        }
    }
    0.5 * (bond_energy(&h, &lb, &ga, &la, &gb) + bond_energy(&h, &la, &gb, &lb, &ga))
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
