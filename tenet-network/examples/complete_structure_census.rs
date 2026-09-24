//! Working-set census of the complete hom-space structure cache (#1365).
//!
//! Measurement only: prints the public `complete_hom_space_structure_cache_info`
//! counter deltas around every public call, one `STEP` row per call:
//! `STEP,workload,iter,step,hits,misses,admissions,evictions,bypasses,entries,charged`
//! (`iter` 0 is setup; every workload starts from a reset cache). The public
//! counters carry no key identity, so `benchmarks/complete_structure_census_lldb.py`
//! reads each lookup key from outside the process, and
//! `benchmarks/complete_structure_census.py` checks that trace against these
//! rows and replays it offline against FIFO and LRU at larger caps.
//!
//! ```text
//! cargo build -p tenet-network --example complete_structure_census
//! ```

use std::sync::Arc;

use tenet::core::{complete_hom_space_structure_cache_info, reset_core_intern_tables};
use tenet::prelude::*;
use tenet_network::tensor;

type Info = tenet::core::CompleteHomSpaceStructureCacheInfo;

struct Census {
    workload: String,
    iter: usize,
    steps: usize,
}

/// Step boundary for an external key tracer: `benchmarks/complete_structure_census_lldb.py`
/// breaks here and on the core lookup, so the lookups between two marks
/// belong to the step emitted at the second.
#[inline(never)]
fn census_mark(step: usize) {
    std::hint::black_box(step);
}

impl Census {
    fn start(&mut self, workload: String) {
        reset_core_intern_tables();
        self.workload = workload;
        self.iter = 0;
    }

    fn emit(&mut self, step: &str, before: Info, after: Info) {
        census_mark(self.steps);
        self.steps += 1;
        println!(
            "STEP,{},{},{step},{},{},{},{},{},{},{}",
            self.workload,
            self.iter,
            after.hits() - before.hits(),
            after.misses() - before.misses(),
            after.admissions() - before.admissions(),
            after.evictions() - before.evictions(),
            after.bypasses() - before.bypasses(),
            after.entries(),
            after.charged_bytes()
        );
    }
}

/// Runs one public call between two counter snapshots.
macro_rules! step {
    ($c:expr, $name:expr, $op:expr) => {{
        let before = complete_hom_space_structure_cache_info();
        let out = $op;
        let after = complete_hom_space_structure_cache_info();
        $c.emit($name, before, after);
        out
    }};
}

struct Case {
    name: &'static str,
    nc: usize,
    nd: usize,
    sectors: usize,
    deg: usize,
}

/// The E1 ledger shapes (`tenet/examples/eager_overhead_ledger.rs`).
const CASES: &[Case] = &[
    Case {
        name: "r2_s8_d2",
        nc: 1,
        nd: 1,
        sectors: 8,
        deg: 2,
    },
    Case {
        name: "r2_s8_d16",
        nc: 1,
        nd: 1,
        sectors: 8,
        deg: 16,
    },
    Case {
        name: "r3_s4_d4",
        nc: 2,
        nd: 1,
        sectors: 4,
        deg: 4,
    },
    Case {
        name: "r4_s3_d4",
        nc: 2,
        nd: 2,
        sectors: 3,
        deg: 4,
    },
    Case {
        name: "r5_s2_d2",
        nc: 3,
        nd: 2,
        sectors: 2,
        deg: 2,
    },
];

const E1_OPS: &[&str] = &[
    "compose",
    "contract",
    "permute",
    "repartition",
    "qr_compact",
    "svd_compact",
    "restrict_leg",
    "scale",
    "add",
    "norm",
    "add_adjoint",
    "adjoint_data",
    // #1368: conjugated (lazy adjoint) sources against owned sources of the
    // same space.
    "contract_lazy",
    "contract_owned",
    "compose_lazy",
    "compose_owned",
];
const E1_ITERS: usize = 4;

/// One row per `(case, op)`: the ledger's inputs, then `E1_ITERS` calls.
macro_rules! e1 {
    ($c:expr, $rt:expr, $symmetry:literal, $provider:expr, $dtype:ty, $dname:literal,
     $labels:expr, $one:expr) => {{
        for case in CASES {
            for &op in E1_OPS {
                let conj = op.ends_with("_lazy") || op.ends_with("_owned");
                if conj && $dname != "f64" {
                    continue;
                }
                let group = if conj { "conj" } else { "e1" };
                $c.start(format!(
                    "{group}/{}/{}/{}/{op}",
                    $symmetry, $dname, case.name
                ));
                let sectors: Vec<_> = ($labels)(case.sectors);
                let leg =
                    GradedSpace::try_new($provider, sectors.iter().map(|s| (s.clone(), case.deg)))
                        .unwrap();
                let cod = vec![&leg; case.nc];
                let dom = vec![&leg; case.nd];
                let rand = |c: &[&GradedSpace<_>], d: &[&GradedSpace<_>], seed| {
                    TensorMap::<_, $dtype>::rand_with_seed(
                        $rt,
                        c.iter().copied(),
                        d.iter().copied(),
                        seed,
                    )
                    .unwrap()
                };
                let a = step!($c, "setup_a", rand(&cod, &dom, 1));
                let a2 = step!($c, "setup_a2", rand(&cod, &dom, 2));
                let square = step!($c, "setup_square", rand(&dom, &dom, 3));
                let matrix = step!($c, "setup_matrix", rand(&[&leg], &[&leg], 4));
                let on_adjoint = step!($c, "setup_on_adjoint", rand(&dom, &cod, 5));
                let lazy = step!($c, "setup_lazy", a.adjoint().unwrap());
                let selection = LegSelection::try_new(
                    &leg,
                    sectors.iter().map(|s| (s.clone(), 0..case.deg.div_ceil(2))),
                )
                .unwrap();
                let rank = case.nc + case.nd;
                let rotated: Vec<usize> = (1..rank).chain([0]).collect();
                let (pc, pd) = rotated.split_at(case.nc);
                let to = if case.nd >= 2 {
                    case.nc + 1
                } else {
                    case.nc - 1
                };
                let open: Vec<usize> = (0..rank).collect();
                let one: $dtype = $one;
                for iter in 1..=E1_ITERS {
                    $c.iter = iter;
                    match op {
                        "compose" => {
                            step!($c, op, a.compose(&square).unwrap());
                        }
                        "contract" => {
                            step!($c, op, a.contract(&matrix, &[0], &[1], &open).unwrap());
                        }
                        "permute" => {
                            step!($c, op, a.permute(pc, pd).unwrap());
                        }
                        "repartition" => {
                            step!($c, op, a.repartition(to).unwrap());
                        }
                        "qr_compact" => {
                            step!($c, op, a.qr_compact().unwrap());
                        }
                        "svd_compact" => {
                            step!($c, op, a.svd_compact().unwrap());
                        }
                        "restrict_leg" => {
                            step!($c, op, a.restrict_leg(0, &selection).unwrap());
                        }
                        "scale" => {
                            step!($c, op, a.scale(one + one));
                        }
                        "add" => {
                            step!($c, op, a.add(&a2, one, one).unwrap());
                        }
                        "norm" => {
                            step!($c, op, a.norm().unwrap());
                        }
                        "add_adjoint" => {
                            step!($c, op, lazy.add(&on_adjoint, one, one).unwrap());
                        }
                        "adjoint_data" => {
                            step!($c, op, {
                                let t = a.adjoint().unwrap();
                                let _ = t.data().len();
                                t
                            });
                        }
                        "contract_lazy" => {
                            step!($c, op, lazy.contract(&matrix, &[0], &[1], &open).unwrap());
                        }
                        "contract_owned" => {
                            step!(
                                $c,
                                op,
                                on_adjoint.contract(&matrix, &[0], &[1], &open).unwrap()
                            );
                        }
                        "compose_lazy" => {
                            step!($c, op, lazy.compose(&a).unwrap());
                        }
                        "compose_owned" => {
                            step!($c, op, on_adjoint.compose(&a).unwrap());
                        }
                        _ => unreachable!("fixed op table"),
                    }
                }
            }
        }
    }};
}

/// Finite-chain sweeps: a two-site `contract` + `svd_trunc` + `compose` +
/// `permute` pass, then a `qr_compact` + `contract` canonicalization pass,
/// both left to right. `contract` puts the right site's physical leg in the
/// domain as its dual and `permute` returns it, so site spaces stay fixed.
macro_rules! mps {
    ($c:expr, $rt:expr, $name:literal, $sites:expr, $provider:expr, $phys:expr, $bond:expr, $edge:expr) => {{
        const SWEEPS: usize = 6;
        const SITES: usize = $sites;
        $c.start(format!("mps/{}/L{SITES}", $name));
        let p = GradedSpace::try_new($provider, $phys).unwrap();
        let v = GradedSpace::try_new($provider, $bond).unwrap();
        let e = GradedSpace::try_new($provider, $edge).unwrap();
        let trunc = Truncation::rank(16).and(Truncation::relative_cutoff(1e-10).unwrap());
        let mut sites: Vec<TensorMap<_, f64>> = (0..SITES)
            .map(|i| {
                let left = if i == 0 { &e } else { &v };
                let right = if i + 1 == SITES { &e } else { &v };
                step!(
                    $c,
                    "setup_site",
                    TensorMap::rand_with_seed($rt, [left, &p], [right], i as u64 + 1).unwrap()
                )
            })
            .collect();
        for sweep in 1..=SWEEPS {
            $c.iter = sweep;
            for i in 0..SITES - 1 {
                let theta = step!(
                    $c,
                    "svd_pass_contract",
                    sites[i]
                        .contract(&sites[i + 1], &[2], &[0], &[0, 1, 2, 3])
                        .unwrap()
                );
                let svd = step!($c, "svd_trunc", theta.svd_trunc(&trunc).unwrap());
                let sv = step!($c, "compose_s_vh", svd.s.compose(&svd.vh).unwrap());
                sites[i + 1] = step!($c, "svd_pass_permute", sv.permute(&[0, 1], &[2]).unwrap());
                sites[i] = svd.u;
            }
            for i in 0..SITES - 1 {
                let (q, r) = step!($c, "qr_compact", sites[i].qr_compact().unwrap());
                sites[i + 1] = step!(
                    $c,
                    "qr_pass_contract",
                    sites[i + 1].contract(&r, &[0], &[1], &[2, 0, 1]).unwrap()
                );
                sites[i] = q;
            }
        }
    }};
}

type Map = TensorMap<U1FusionRule, f64>;

/// The `tests/itebd_smoke.rs` iTEBD loop: `tensor!` networks of up to six
/// operands, `svd_trunc`, `pinv`, and a `conj` energy network.
fn itebd(c: &mut Census, rt: &Runtime) {
    const ITERS: usize = 30;
    c.start("itebd/U1".to_string());
    let rule = Arc::new(U1FusionRule);
    let space = |sectors: &[(i32, usize)]| {
        GradedSpace::try_new_with_arc(
            Arc::clone(&rule),
            sectors.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
        )
        .unwrap()
    };
    let p = space(&[(1, 1), (-1, 1)]);
    let h = step!(
        c,
        "setup_h",
        Map::from_block_fn(rt, [&p, &p], [&p, &p], |trees, _| {
            let (cod, dom) = (trees.codomain_uncoupled(), trees.domain_uncoupled());
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
    );
    let trunc = Truncation::rank(8).and(Truncation::relative_cutoff(1e-8).unwrap());
    let vb = space(&[(0, 1)]);
    let va = space(&[(1, 1), (-1, 1)]);
    let ones = |c: &[&GradedSpace<U1FusionRule>], d: &[&GradedSpace<U1FusionRule>]| {
        Map::from_block_fn(rt, c.iter().copied(), d.iter().copied(), |_, _| 1.0).unwrap()
    };
    let mut ga = step!(c, "setup", ones(&[&vb, &p], &[&va]));
    let mut la = step!(c, "setup", ones(&[&va], &[&va]));
    let mut gb = step!(c, "setup", ones(&[&va, &p], &[&vb]));
    let mut lb = step!(c, "setup", ones(&[&vb], &[&vb]));
    let gate = step!(c, "setup_gate", h.scale(-0.1).exp().unwrap());

    let update =
        |c: &mut Census, l_out: &Map, g1: &Map, l_mid: &Map, g2: &Map| -> (Map, Map, Map) {
            let theta = step!(
                c,
                "theta_network",
                tensor!([l, pa; pb, r] = l_out[l; x] * g1[x, qa; y]
            * l_mid[y; z] * g2[z, qb; w] * l_out[w; r] * gate[pa, pb; qa, qb])
                .unwrap()
            );
            let svd = step!(c, "svd_trunc", theta.svd_trunc(&trunc).unwrap());
            let norm = step!(c, "norm", svd.s.norm().unwrap());
            let l_new = step!(c, "scale", svd.s.scale(1.0 / norm));
            let inv = step!(c, "pinv", l_out.pinv(1e-12).unwrap());
            let (u, vh) = (svd.u, svd.vh);
            let g1n = step!(
                c,
                "absorb_left",
                tensor!([l, pa; m] = inv[l; x] * u[x, pa; m]).unwrap()
            );
            let g2n = step!(
                c,
                "absorb_right",
                tensor!([m, pb; r] = vh[m; pb, x] * inv[x; r]).unwrap()
            );
            (g1n, l_new, g2n)
        };
    for iter in 1..=ITERS {
        c.iter = iter;
        (ga, la, gb) = update(c, &lb, &ga, &la, &gb);
        (gb, lb, ga) = update(c, &la, &gb, &lb, &ga);
        let theta = step!(
            c,
            "energy_theta",
            tensor!([l, pa, pb; r] = lb[l; x] * ga[x, pa; y]
            * la[y; z] * gb[z, pb; w] * lb[w; r])
            .unwrap()
        );
        step!(
            c,
            "energy_network",
            tensor!([] = conj(theta)[l, pa, pb; r] * h[pa, pb; qa, qb]
            * theta[l, qa, qb; r])
            .unwrap()
        );
    }
}

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn centered(count: usize) -> impl Iterator<Item = i32> {
    let low = -((count as i32 - 1) / 2);
    (0..count as i32).map(move |i| low + i)
}

fn main() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let rt = &rt;
    let mut c = Census {
        workload: String::new(),
        iter: 0,
        steps: 0,
    };
    let info = complete_hom_space_structure_cache_info();
    eprintln!(
        "# cap={} byte_budget={} max_entry_bytes={}",
        info.entry_capacity(),
        info.byte_budget(),
        info.max_entry_bytes()
    );
    println!("STEP,workload,iter,step,hits,misses,admissions,evictions,bypasses,entries,charged");

    let u1 = |n: usize| centered(n).map(U1Irrep::new).collect::<Vec<_>>();
    let fz2u1 = |n: usize| {
        centered(n)
            .map(|q| {
                let parity = if q.rem_euclid(2) == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                };
                ProductSector::new(parity, U1Irrep::new(q))
            })
            .collect::<Vec<_>>()
    };
    let su2 = |n: usize| (0..n).map(SU2Irrep::from_twice_spin).collect::<Vec<_>>();
    let fz2u1_rule = || Fz2U1::new(FermionParityFusionRule, U1FusionRule);

    e1!(&mut c, rt, "U1", U1FusionRule, f64, "f64", u1, 1.0);
    e1!(
        &mut c,
        rt,
        "U1",
        U1FusionRule,
        Complex64,
        "c64",
        u1,
        Complex64::new(1.0, 0.0)
    );
    e1!(&mut c, rt, "fZ2xU1", fz2u1_rule(), f64, "f64", fz2u1, 1.0);
    e1!(
        &mut c,
        rt,
        "fZ2xU1",
        fz2u1_rule(),
        Complex64,
        "c64",
        fz2u1,
        Complex64::new(1.0, 0.0)
    );
    e1!(&mut c, rt, "SU2", SU2FusionRule, f64, "f64", su2, 1.0);
    e1!(
        &mut c,
        rt,
        "SU2",
        SU2FusionRule,
        Complex64,
        "c64",
        su2,
        Complex64::new(1.0, 0.0)
    );

    let u = |q: i32, d: usize| (U1Irrep::new(q), d);
    let s = |twice: usize, d: usize| (SU2Irrep::from_twice_spin(twice), d);
    macro_rules! chains {
        ($($sites:literal),*) => {$(
            mps!(&mut c, rt, "U1", $sites, U1FusionRule, [u(1, 1), u(-1, 1)],
                [u(-2, 2), u(-1, 2), u(0, 3), u(1, 2), u(2, 2)], [u(0, 1)]);
            mps!(&mut c, rt, "SU2", $sites, SU2FusionRule, [s(1, 1)],
                [s(0, 3), s(1, 3), s(2, 2), s(3, 1)], [s(0, 1)]);
        )*};
    }
    chains!(6, 24);

    itebd(&mut c, rt);
}
