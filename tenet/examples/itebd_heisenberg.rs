//! iTEBD ground-state search for the spin-1/2 antiferromagnetic Heisenberg
//! chain, U(1)-symmetric (Sz conservation), through the typed `TensorMap` API
//! and the `tensor!` macro.
//!
//! Charge convention: the physical leg has U(1) sectors `[(+1, 1), (-1, 1)]`
//! with charge `q = 2 Sz` (so `+1` = up, `-1` = down). Total charge is
//! conserved, which makes the two-site Hamiltonian and gate block-diagonal
//! in `q1 + q2 ∈ {-2, 0, +2}`: the `±2` blocks are the 1x1 `Sz Sz = +1/4`
//! diagonals and the `0` block is the 2x2 `[[-1/4, 1/2], [1/2, -1/4]]`
//! flip-flop matrix.
//!
//! Vidal form with a two-site unit cell A-B: state = `... λb Γa λa Γb λb ...`,
//! Γ tensors shaped `[left_bond, phys] <- [right_bond]`, λ diagonal bond
//! endomorphisms. One bond update contracts
//! `θ = λ_out Γ1 λ_mid Γ2 λ_out · gate`, truncates with `svd_trunc`, and
//! restores Vidal form by multiplying the outer `λ_out^{-1}` back in
//! (diagonal inverse via `TensorMap::pinv`).
//!
//! Reference: ground-state energy per bond `e = 1/4 - ln 2 ≈ -0.4431471`.
//!
//! Run with:
//! `cargo run --release -p tenet --example itebd_heisenberg --no-default-features --features cpu-faer`

use std::sync::Arc;
use std::time::Instant;

use tenet::prelude::{Error, GradedSpace, Runtime, TensorMap, Truncation, U1FusionRule, U1Irrep};
use tenet_network::tensor;

const E_EXACT: f64 = 0.25 - std::f64::consts::LN_2;
/// Relative cutoff on kept singular values: values this small get inverted
/// by the Vidal `λ^{-1}` step, so don't keep anything numerically dangerous.
const BOND_RTOL: f64 = 1e-8;
/// `pinv` cutoff (relative to the largest singular value of λ).
const PINV_RCOND: f64 = 1e-12;
type Map = TensorMap<U1FusionRule, f64>;

fn space<const N: usize>(
    rule: &Arc<U1FusionRule>,
    sectors: [(i32, usize); N],
) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(rule),
        sectors.map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
        false,
    )
    .expect("valid U(1) fixture")
}

/// Two-site Heisenberg Hamiltonian `h = S·S` on `[p, p] <- [p, p]`, built
/// block-by-block in the U(1) fusion-tree basis. Blocks are keyed by
/// (codomain uncoupled charges, domain uncoupled charges) with total charge
/// conserved, so only same-charge tree pairs appear:
/// `<s1 s2| h |s1 s2> = ±1/4` (Sz Sz), `<up dn| h |dn up> = 1/2` (flip-flop).
fn heisenberg_two_site(rt: &Runtime, p: &GradedSpace<U1FusionRule>) -> Result<Map, Error> {
    TensorMap::from_block_fn(rt, [p, p], [p, p], |trees, _| {
        let cod = trees.codomain_uncoupled();
        let dom = trees.domain_uncoupled();
        if cod == dom {
            if cod[0] == cod[1] {
                0.25 // up-up / dn-dn
            } else {
                -0.25 // up-dn / dn-up diagonal
            }
        } else {
            0.5 // flip-flop, only present in the charge-0 block
        }
    })
}

/// One iTEBD bond update in Vidal form: applies `gate` to the bond between
/// `g1` and `g2` (inner weight `l_mid`, outer weights `l_out` on both open
/// bonds), truncates, and returns `(g1', l_mid', g2', truncation_error)`.
fn bond_update(
    gate: &Map,
    l_out: &Map,
    g1: &Map,
    l_mid: &Map,
    g2: &Map,
    trunc: &Truncation,
) -> Result<(Map, Map, Map, f64), Error> {
    let theta = tensor!([l, pa; pb, r] = l_out[l; x] * g1[x, qa; y] * l_mid[y; z]
        * g2[z, qb; w] * l_out[w; r] * gate[pa, pb; qa, qb])?;
    let svd = theta.svd_trunc(trunc)?;
    let l_new = svd.s.scale(1.0 / svd.s.norm()?);
    // Divide the outer λ back out: diagonal inverse via pinv.
    let l_out_inv = l_out.pinv(PINV_RCOND)?;
    let u = svd.u;
    let vh = svd.vh;
    let g1_new = tensor!([l, pa; m] = l_out_inv[l; x] * u[x, pa; m])?;
    let g2_new = tensor!([m, pb; r] = vh[m; pb, x] * l_out_inv[x; r])?;
    Ok((g1_new, l_new, g2_new, svd.error))
}

/// Energy of one bond, `<θ|h|θ> / <θ|θ>`, on the two-site wavefunction
/// `θ = λ_out Γ1 λ_mid Γ2 λ_out` (no gate).
fn bond_energy(h: &Map, l_out: &Map, g1: &Map, l_mid: &Map, g2: &Map) -> Result<f64, Error> {
    let theta = tensor!([l, pa, pb; r] = l_out[l; x] * g1[x, pa; y] * l_mid[y; z]
        * g2[z, pb; w] * l_out[w; r])?;
    let num = tensor!([] = conj(theta)[l, pa, pb; r] * h[pa, pb; qa, qb] * theta[l, qa, qb; r])?
        .scalar()?;
    Ok(num / theta.inner(&theta)?)
}

/// Weighted bond dimension of a diagonal λ (its single codomain leg).
fn bond_dim(lambda: &Map) -> Result<usize, Error> {
    Ok(lambda.leg_dims()?[0])
}

struct State {
    ga: Map,
    la: Map, // bond A-B
    gb: Map,
    lb: Map, // bond B-A
}

impl State {
    /// Charge-balanced entangled start, `∝ ... (|up dn> + |dn up>) ...`
    /// per unit cell: bond B-A carries charge 0, bond A-B charges ±1.
    ///
    /// (A strict Neel product state — phys leg populated only in `+1` on A /
    /// `-1` on B — also works now that legs carry per-sector degeneracies;
    /// see the `neel_product_state_contracts_with_the_full_gate` test. The
    /// entangled start is kept because it converges faster.)
    fn init(
        rt: &Runtime,
        rule: &Arc<U1FusionRule>,
        p: &GradedSpace<U1FusionRule>,
    ) -> Result<Self, Error> {
        let vb = space(rule, [(0, 1)]);
        let va = space(rule, [(1, 1), (-1, 1)]);
        Ok(Self {
            ga: TensorMap::from_block_fn(rt, [&vb, p], [&va], |_, _| 1.0)?,
            la: TensorMap::from_block_fn(rt, [&va], [&va], |_, _| 1.0)?,
            gb: TensorMap::from_block_fn(rt, [&va, p], [&vb], |_, _| 1.0)?,
            lb: TensorMap::from_block_fn(rt, [&vb], [&vb], |_, _| 1.0)?,
        })
    }

    /// One full iTEBD step: update bond A-B, then bond B-A.
    fn step(&mut self, gate: &Map, trunc: &Truncation) -> Result<f64, Error> {
        let (ga, la, gb, err_a) = bond_update(gate, &self.lb, &self.ga, &self.la, &self.gb, trunc)?;
        self.ga = ga;
        self.la = la;
        self.gb = gb;
        let (gb, lb, ga, err_b) = bond_update(gate, &self.la, &self.gb, &self.lb, &self.ga, trunc)?;
        self.gb = gb;
        self.lb = lb;
        self.ga = ga;
        Ok(err_a.max(err_b))
    }

    /// Mean energy per bond, averaged over the A-B and B-A bonds.
    fn energy(&self, h: &Map) -> Result<f64, Error> {
        let e_ab = bond_energy(h, &self.lb, &self.ga, &self.la, &self.gb)?;
        let e_ba = bond_energy(h, &self.la, &self.gb, &self.lb, &self.ga)?;
        Ok(0.5 * (e_ab + e_ba))
    }
}

/// Runs the full schedule; returns the final energy per bond.
pub fn run(chi: usize, schedule: &[(f64, usize)], verbose: bool) -> Result<f64, Error> {
    let rt = Runtime::builder().build()?;
    let rule = Arc::new(U1FusionRule);
    let p = space(&rule, [(1, 1), (-1, 1)]);
    let h = heisenberg_two_site(&rt, &p)?;
    let trunc = Truncation::rank(chi).and(Truncation::relative_cutoff(BOND_RTOL)?);

    let mut state = State::init(&rt, &rule, &p)?;
    let mut energy = f64::NAN;
    for &(dt, steps) in schedule {
        let gate = h.scale(-dt).exp()?;
        let start = Instant::now();
        let mut max_err = 0.0f64;
        for _ in 0..steps {
            max_err = max_err.max(state.step(&gate, &trunc)?);
        }
        let elapsed = start.elapsed().as_secs_f64();
        energy = state.energy(&h)?;
        if verbose {
            println!(
                "dt = {dt:<7} steps = {steps:<5} chi = {:<3} E/bond = {energy:.8} \
                 (err vs exact {:+.2e}, max trunc err {max_err:.1e}, \
                 {:.1} steps/s, {elapsed:.2} s)",
                bond_dim(&state.la)?.max(bond_dim(&state.lb)?),
                energy - E_EXACT,
                steps as f64 / elapsed,
            );
        }
    }
    Ok(energy)
}

fn main() -> Result<(), Error> {
    let chi = 32;
    let schedule = [
        (0.1, 500),
        (0.05, 500),
        (0.01, 500),
        (0.005, 500),
        (0.001, 500),
    ];
    let start = Instant::now();
    let energy = run(chi, &schedule, true)?;
    let total = start.elapsed().as_secs_f64();
    println!();
    println!("final E/bond   = {energy:.8}");
    println!("exact  1/4-ln2 = {E_EXACT:.8}");
    println!("error          = {:+.3e}", energy - E_EXACT);
    println!("total wall time: {total:.1} s");
    assert!(
        (energy - E_EXACT).abs() < 5e-3,
        "iTEBD did not reach the exact ground-state energy: {energy} vs {E_EXACT}"
    );
    Ok(())
}
