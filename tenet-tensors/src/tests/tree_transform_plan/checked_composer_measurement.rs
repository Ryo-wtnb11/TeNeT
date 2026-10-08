//! Measurement only (#1962): provider calls by kind and coefficient
//! multiplies/adds of the checked Generic plan build across sharing regimes
//! (high sharing on SU(3) adjoint legs, no sharing on multiplicity-free
//! symmetric legs, identity). Allocation counts of the complete public path
//! are in the `checked_generic_composer_measurement` integration test; this
//! crate forbids the unsafe code a counting allocator needs.

use super::*;
use std::cell::Cell;
use std::ops::{Add, Mul};
use tenet_core::{
    BraidingStyleKind, CategoricalScalar, FusionStyleKind, SUNFusionRule, SUNFusionRuleError,
    SectorVec,
};

thread_local! {
    static MULS: Cell<u64> = const { Cell::new(0) };
    static ADDS: Cell<u64> = const { Cell::new(0) };
}

/// An `f64` that counts its coefficient multiplies and adds.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Counted(f64);

impl Add for Counted {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        ADDS.set(ADDS.get() + 1);
        Self(self.0 + other.0)
    }
}

impl Mul for Counted {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        MULS.set(MULS.get() + 1);
        Self(self.0 * other.0)
    }
}

impl num_traits::Zero for Counted {
    fn zero() -> Self {
        Self(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl CategoricalScalar for Counted {
    fn zero() -> Self {
        Self(0.0)
    }
    fn one() -> Self {
        Self(1.0)
    }
    fn conj(&self) -> Self {
        *self
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

const CALLS: [&str; 8] = ["channels", "dual", "n", "sqrt", "inv_sqrt", "fs", "f", "r"];

/// SU(3) with counted coefficients and per-kind provider call counts.
struct Measured {
    inner: SUNFusionRule,
    calls: Cell<[u64; 8]>,
}

impl Measured {
    fn hit(&self, kind: usize) {
        let mut calls = self.calls.get();
        calls[kind] += 1;
        self.calls.set(calls);
    }
}

impl CheckedGenericFusion for Measured {
    type Error = SUNFusionRuleError;
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.inner.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.inner.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.inner)
    }
    fn try_dual(&self, a: SectorId) -> Result<SectorId, Self::Error> {
        self.hit(1);
        self.inner.try_dual(a)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        self.hit(0);
        self.inner.try_fusion_channels(a, b)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.hit(0);
        self.inner.try_fusion_channels_in_table(a, b)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        self.hit(2);
        self.inner.try_nsymbol(a, b, c)
    }
}

impl CheckedGenericRigidSymbols for Measured {
    type Scalar = Counted;
    fn try_sqrt_dim_scalar(&self, a: SectorId) -> Result<Counted, Self::Error> {
        self.hit(3);
        self.inner.try_sqrt_dim_scalar(a).map(Counted)
    }
    fn try_inv_sqrt_dim_scalar(&self, a: SectorId) -> Result<Counted, Self::Error> {
        self.hit(4);
        self.inner.try_inv_sqrt_dim_scalar(a).map(Counted)
    }
    fn try_frobenius_schur_phase_scalar(&self, a: SectorId) -> Result<Counted, Self::Error> {
        self.hit(5);
        self.inner.try_frobenius_schur_phase_scalar(a).map(Counted)
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Counted>, Self::Error> {
        self.hit(6);
        let block = self.inner.try_f_symbol_generic(a, b, c, d, e, f)?;
        Ok(GenericFArray::new(
            block.data().iter().copied().map(Counted).collect(),
            block.shape(),
        ))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Counted>, Self::Error> {
        self.hit(7);
        let block = self.inner.try_r_symbol_generic(a, b, c)?;
        let (rows, cols) = block.shape();
        Ok(GenericRMatrix::new(
            block.data().iter().copied().map(Counted).collect(),
            rows,
            cols,
        ))
    }
}

fn homspace(legs: &[Vec<SectorId>], codomain_rank: usize) -> FusionTreeHomSpace {
    let leg =
        |sectors: &Vec<SectorId>| SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new(legs[..codomain_rank].iter().map(leg)),
        FusionProductSpace::new(legs[codomain_rank..].iter().map(leg)),
    )
}

/// The planar transpose rotating the linearized legs (codomain, then
/// reversed domain) by `shift` into a `split`-leg codomain.
fn rotation(
    codomain_rank: usize,
    domain_rank: usize,
    shift: usize,
    split: usize,
) -> TreeTransformOperation {
    let total = codomain_rank + domain_rank;
    let linear = (0..codomain_rank)
        .chain((codomain_rank..total).rev())
        .collect::<Vec<_>>();
    let rotated = (0..total)
        .map(|index| linear[(index + shift) % total])
        .collect::<Vec<_>>();
    TreeTransformOperation::transpose(
        rotated[..split].to_vec(),
        rotated[split..].iter().rev().copied().collect::<Vec<_>>(),
    )
}

fn median(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn measure_case(
    regime: &str,
    case: &str,
    legs: &[Vec<SectorId>],
    codomain_rank: usize,
    operation: TreeTransformOperation,
) {
    let plain = SUNFusionRule::new(3).unwrap();
    let structure = homspace(legs, codomain_rank)
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&plain)
        .unwrap();
    let groups = structure.fusion_tree_group_slice();
    let max_group = groups
        .iter()
        .map(|group| group.block_indices().len())
        .max()
        .unwrap_or(0);
    // Warm racah's symbol caches; TeNeT keeps no cache on this path.
    build_checked_generic_tree_pair_transform_group_plan(&plain, operation.clone(), &structure)
        .unwrap();
    let build_ns = median(
        (0..5)
            .map(|_| {
                let started = std::time::Instant::now();
                std::hint::black_box(
                    build_checked_generic_tree_pair_transform_group_plan(
                        &plain,
                        operation.clone(),
                        &structure,
                    )
                    .unwrap(),
                );
                started.elapsed().as_nanos()
            })
            .collect(),
    );

    let measured = Measured {
        inner: SUNFusionRule::new(3).unwrap(),
        calls: Cell::new([0; 8]),
    };
    MULS.set(0);
    ADDS.set(0);
    tenet_core::take_generic_block_step_counts();
    let plan = build_checked_generic_tree_pair_transform_group_plan(
        &measured,
        operation.clone(),
        &structure,
    )
    .unwrap();
    let (muls, adds) = (MULS.get(), ADDS.get());
    let step = tenet_core::take_generic_block_step_counts();
    let multis = plan
        .specs()
        .iter()
        .filter(|spec| spec.src_keys().len() > 1 || spec.dst_keys().len() > 1)
        .count();
    let stored: usize = plan
        .specs()
        .iter()
        .map(|spec| spec.recoupling_coefficients_dst_src().len())
        .sum();
    let calls = measured
        .calls
        .get()
        .iter()
        .zip(CALLS)
        .map(|(count, name)| format!("{name}={count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "regime={regime} case={case} blocks={} groups={} max_group={max_group} specs={} multi_specs={multis} stored_coeffs={stored} calls[{calls}] muls={muls} adds={adds} steps={} moves={} per_source_moves={} spread_muls={} build_median_ns={build_ns}",
        structure.block_count(),
        groups.len(),
        plan.specs().len(),
        step.steps,
        step.moves,
        step.per_source_moves,
        step.spread_multiplies,
    );
}

#[test]
#[ignore = "measurement: run explicitly with --ignored --nocapture"]
fn measure_checked_generic_composer_calls_and_arithmetic() {
    let rule = SUNFusionRule::new(3).unwrap();
    let adjoint = vec![rule.encode_dynkin(&[1, 1]).unwrap()];
    // Pieri: tensoring with a symmetric power (p, 0) or its dual is
    // multiplicity-free, so every vertex over these legs is multiplicity-free
    // and every bend is one-to-one.
    let symmetric = vec![
        rule.encode_dynkin(&[1, 0]).unwrap(),
        rule.encode_dynkin(&[2, 0]).unwrap(),
        rule.encode_dynkin(&[0, 2]).unwrap(),
    ];
    let adj = |rank: usize| vec![adjoint.clone(); rank];
    let sym = |rank: usize| vec![symmetric.clone(); rank];

    for (rank, codomain_rank) in [(2, 1), (4, 2), (6, 3)] {
        let domain_rank = rank - codomain_rank;
        measure_case(
            "high",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_transpose_anticlockwise"),
            &adj(rank),
            codomain_rank,
            rotation(codomain_rank, domain_rank, 1, codomain_rank),
        );
        measure_case(
            "high",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_transpose_clockwise"),
            &adj(rank),
            codomain_rank,
            rotation(codomain_rank, domain_rank, rank - 1, codomain_rank),
        );
        measure_case(
            "high",
            &format!("r{rank}_{rank}|0_permute_reverse"),
            &adj(rank),
            rank,
            TreeTransformOperation::permute((0..rank).rev().collect::<Vec<_>>(), []),
        );
        measure_case(
            "high",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_braid_levels"),
            &adj(rank),
            codomain_rank,
            TreeTransformOperation::braid(
                (0..codomain_rank).rev().collect::<Vec<_>>(),
                (codomain_rank..rank).rev().collect::<Vec<_>>(),
                (0..codomain_rank).collect::<Vec<_>>(),
                (codomain_rank..rank).rev().collect::<Vec<_>>(),
            ),
        );
        // A pure repartition is a chain of bends; over multiplicity-free
        // vertices each bend maps one tree to one tree, so no intermediate
        // tree is shared (c ≡ 1). Cycles are not used here: a fold
        // recouples the coupled line and fans out.
        for split in [codomain_rank + 1, codomain_rank - 1] {
            measure_case(
                "low",
                &format!("r{rank}_{codomain_rank}|{domain_rank}_repartition_to_{split}"),
                &sym(rank),
                codomain_rank,
                rotation(codomain_rank, domain_rank, 0, split),
            );
        }
        // Cycles over the same multiplicity-free legs: a fold recouples the
        // coupled line, so moves emit several terms and some sharing appears.
        measure_case(
            "fanout",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_transpose_anticlockwise"),
            &sym(rank),
            codomain_rank,
            rotation(codomain_rank, domain_rank, 1, codomain_rank),
        );
        measure_case(
            "identity",
            &format!("r{rank}_{codomain_rank}|{domain_rank}_permute_identity"),
            &adj(rank),
            codomain_rank,
            TreeTransformOperation::permute(
                (0..codomain_rank).collect::<Vec<_>>(),
                (codomain_rank..rank).collect::<Vec<_>>(),
            ),
        );
    }
}
