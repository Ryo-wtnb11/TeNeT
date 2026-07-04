//! User-layer symmetric tensor: dynamic rank, rule-erased, runtime-carrying.
//!
//! A [`Tensor`] wraps the expert-layer [`tenet_core::TensorMap`] machinery:
//! the concrete fusion rule is erased behind [`crate::space::RuleKind`] and
//! the const-generic codomain/domain ranks are erased behind an internal
//! enum. Operations lock the shared [`Runtime`] state once and dispatch to
//! the expert layer.

use std::collections::BTreeMap;
use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, FusionProductSpace, FusionRule, FusionTensorMapSpace,
    FusionTreeHomSpace, MultiplicityFreeRigidSymbols, SectorId, TensorMap, TensorMapSpace,
};
use tenet_tensors::{
    tree_transform_into_with_context, TensorContractSpec, TreeTransformOperation,
    TreeTransformRuleCacheKey,
};

use crate::error::Error;
use crate::runtime::{with_rule_ctx, Ctx, Runtime};
use crate::space::{with_rule, RuleKind, Space};

/// Current user-layer rank ceiling: legs per side (codomain or domain).
///
// ponytail: ceiling 2 legs per side (total rank <= 4). The expert layer is
// const-generic in (NOUT, NIN), so every extra split multiplies the
// monomorphized contract/transform stacks by ~5 rules; the dispatch tables
// below stay hand-written and small at 2. Upgrade path: extend `ErasedMap`
// and the three dispatch matches, or expose tenet-tensors' dynamic-space
// entry points publicly and delete the tables.
pub(crate) const MAX_LEGS_PER_SIDE: usize = 2;

/// Common trait bound bundle for the erased rules.
trait UserRule: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Sized {}
impl<R> UserRule for R where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Sized
{
}

// ---------------------------------------------------------------------------
// Leg bookkeeping: per-axis degeneracies keyed by internal leg sectors.
// ---------------------------------------------------------------------------

/// Degeneracy table of one tensor leg, keyed by the *internal* sectors of
/// the corresponding hom-space [`tenet_core::SectorLeg`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LegInfo {
    degs: BTreeMap<SectorId, usize>,
}

impl LegInfo {
    fn from_space(space: &Space) -> Self {
        Self {
            degs: space.sectors.iter().copied().collect(),
        }
    }

    /// Leg after crossing between codomain and domain: sector keys are
    /// dualized (degeneracies follow their sector).
    fn dualized<R: FusionRule>(&self, rule: &R) -> Self {
        Self {
            degs: self
                .degs
                .iter()
                .map(|(&sector, &deg)| (rule.dual(sector), deg))
                .collect(),
        }
    }

    fn deg(&self, sector: SectorId) -> Result<usize, Error> {
        self.degs.get(&sector).copied().ok_or_else(|| {
            Error::InvalidArgument(format!("sector {sector:?} not present on this leg"))
        })
    }

    fn dim<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(&self, rule: &R) -> usize {
        self.degs
            .iter()
            .map(|(&sector, &deg)| deg * (rule.dim_scalar(sector).round() as usize))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Rank erasure: enum over the supported (NOUT, NIN) splits.
// ---------------------------------------------------------------------------

/// Rank-erased tensor storage: one variant per supported codomain/domain
/// split, each holding the concretely typed expert-layer [`TensorMap`].
#[derive(Clone, Debug)]
enum ErasedMap {
    R0x0(TensorMap<f64, 0, 0>),
    R1x0(TensorMap<f64, 1, 0>),
    R0x1(TensorMap<f64, 0, 1>),
    R1x1(TensorMap<f64, 1, 1>),
    R2x0(TensorMap<f64, 2, 0>),
    R0x2(TensorMap<f64, 0, 2>),
    R2x1(TensorMap<f64, 2, 1>),
    R1x2(TensorMap<f64, 1, 2>),
    R2x2(TensorMap<f64, 2, 2>),
}

/// Runs `$body` with `$m` bound to the typed [`TensorMap`] of any variant.
macro_rules! on_map {
    ($map:expr, $m:ident => $body:expr) => {
        match $map {
            ErasedMap::R0x0($m) => $body,
            ErasedMap::R1x0($m) => $body,
            ErasedMap::R0x1($m) => $body,
            ErasedMap::R1x1($m) => $body,
            ErasedMap::R2x0($m) => $body,
            ErasedMap::R0x2($m) => $body,
            ErasedMap::R2x1($m) => $body,
            ErasedMap::R1x2($m) => $body,
            ErasedMap::R2x2($m) => $body,
        }
    };
}

impl ErasedMap {
    fn nout(&self) -> usize {
        match self {
            Self::R0x0(_) | Self::R0x1(_) | Self::R0x2(_) => 0,
            Self::R1x0(_) | Self::R1x1(_) | Self::R1x2(_) => 1,
            Self::R2x0(_) | Self::R2x1(_) | Self::R2x2(_) => 2,
        }
    }

    fn nin(&self) -> usize {
        match self {
            Self::R0x0(_) | Self::R1x0(_) | Self::R2x0(_) => 0,
            Self::R0x1(_) | Self::R1x1(_) | Self::R2x1(_) => 1,
            Self::R0x2(_) | Self::R1x2(_) | Self::R2x2(_) => 2,
        }
    }

    fn data(&self) -> &[f64] {
        on_map!(self, m => m.data())
    }

    fn data_vec(&self) -> Vec<f64> {
        self.data().to_vec()
    }

    fn structure(&self) -> &Arc<BlockStructure> {
        on_map!(self, m => m.structure())
    }

    fn hom(&self) -> &FusionTreeHomSpace {
        on_map!(self, m => m
            .fusion_space()
            .expect("user-layer tensors always carry a fusion space")
            .homspace())
    }

    /// Same tensor with replaced flat data (same fusion space and layout).
    fn with_data(&self, data: Vec<f64>) -> Result<Self, Error> {
        fn rebuilt<const N: usize, const I: usize>(
            map: &TensorMap<f64, N, I>,
            data: Vec<f64>,
        ) -> Result<TensorMap<f64, N, I>, Error> {
            let fusion_space = Arc::clone(
                map.fusion_space()
                    .expect("user-layer tensors always carry a fusion space"),
            );
            TensorMap::from_vec_with_shared_fusion_space(data, fusion_space).map_err(Into::into)
        }
        Ok(match self {
            Self::R0x0(m) => Self::R0x0(rebuilt(m, data)?),
            Self::R1x0(m) => Self::R1x0(rebuilt(m, data)?),
            Self::R0x1(m) => Self::R0x1(rebuilt(m, data)?),
            Self::R1x1(m) => Self::R1x1(rebuilt(m, data)?),
            Self::R2x0(m) => Self::R2x0(rebuilt(m, data)?),
            Self::R0x2(m) => Self::R0x2(rebuilt(m, data)?),
            Self::R2x1(m) => Self::R2x1(rebuilt(m, data)?),
            Self::R1x2(m) => Self::R1x2(rebuilt(m, data)?),
            Self::R2x2(m) => Self::R2x2(rebuilt(m, data)?),
        })
    }
}

// ---------------------------------------------------------------------------
// Typed kernels: the only places that touch const-generic expert entry points.
// ---------------------------------------------------------------------------

/// Builds the coupled-layout fusion space for `NOUT + NIN` legs from a hom
/// space and per-leg degeneracy tables.
fn build_fusion_space<R: UserRule, const NOUT: usize, const NIN: usize>(
    rule: &R,
    hom: FusionTreeHomSpace,
    legs: &[LegInfo],
) -> Result<FusionTensorMapSpace<NOUT, NIN>, Error> {
    debug_assert_eq!(legs.len(), NOUT + NIN);
    let keys = hom.fusion_tree_keys(rule);
    let mut shapes = Vec::with_capacity(keys.len());
    for key in &keys {
        let mut shape = Vec::with_capacity(NOUT + NIN);
        for (leg, &sector) in legs[..NOUT].iter().zip(key.codomain_uncoupled()) {
            shape.push(leg.deg(sector)?);
        }
        for (leg, &sector) in legs[NOUT..].iter().zip(key.domain_uncoupled()) {
            shape.push(leg.deg(sector)?);
        }
        shapes.push(shape);
    }
    let mut codomain_dims = [0usize; NOUT];
    for (dim, leg) in codomain_dims.iter_mut().zip(&legs[..NOUT]) {
        *dim = leg.dim(rule);
    }
    let mut domain_dims = [0usize; NIN];
    for (dim, leg) in domain_dims.iter_mut().zip(&legs[NOUT..]) {
        *dim = leg.dim(rule);
    }
    let dense = TensorMapSpace::<NOUT, NIN>::from_dims(codomain_dims, domain_dims)?;
    FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, rule, shapes).map_err(Into::into)
}

/// How a freshly built tensor is filled.
enum Fill<'f> {
    Zeros,
    Rand(u64),
    BlockFn(&'f mut dyn FnMut(&BlockKey, &[usize]) -> f64),
}

/// splitmix64: small deterministic RNG for [`Tensor::rand`]; no external
/// dependency, values uniform in `[-1, 1)`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64) / ((1u64 << 52) as f64) - 1.0
}

fn construct_typed<R: UserRule, const NOUT: usize, const NIN: usize>(
    rule: &R,
    hom: FusionTreeHomSpace,
    legs: &[LegInfo],
    fill: Fill<'_>,
) -> Result<TensorMap<f64, NOUT, NIN>, Error> {
    let space = build_fusion_space::<R, NOUT, NIN>(rule, hom, legs)?;
    match fill {
        Fill::Zeros => {
            let len = space.required_len()?;
            TensorMap::from_vec_with_fusion_space(vec![0.0; len], space).map_err(Into::into)
        }
        Fill::Rand(seed) => {
            let len = space.required_len()?;
            let mut state = seed;
            let data = (0..len).map(|_| rand_unit(&mut state)).collect();
            TensorMap::from_vec_with_fusion_space(data, space).map_err(Into::into)
        }
        Fill::BlockFn(fill) => {
            TensorMap::from_block_fn_with_fusion_space(space, 0.0, fill).map_err(Into::into)
        }
    }
}

/// Runtime `(nout, nin)` to const-generic construction dispatch.
fn construct_erased<R: UserRule>(
    rule: &R,
    hom: FusionTreeHomSpace,
    legs: &[LegInfo],
    nout: usize,
    nin: usize,
    fill: Fill<'_>,
) -> Result<ErasedMap, Error> {
    Ok(match (nout, nin) {
        (0, 0) => ErasedMap::R0x0(construct_typed::<R, 0, 0>(rule, hom, legs, fill)?),
        (1, 0) => ErasedMap::R1x0(construct_typed::<R, 1, 0>(rule, hom, legs, fill)?),
        (0, 1) => ErasedMap::R0x1(construct_typed::<R, 0, 1>(rule, hom, legs, fill)?),
        (1, 1) => ErasedMap::R1x1(construct_typed::<R, 1, 1>(rule, hom, legs, fill)?),
        (2, 0) => ErasedMap::R2x0(construct_typed::<R, 2, 0>(rule, hom, legs, fill)?),
        (0, 2) => ErasedMap::R0x2(construct_typed::<R, 0, 2>(rule, hom, legs, fill)?),
        (2, 1) => ErasedMap::R2x1(construct_typed::<R, 2, 1>(rule, hom, legs, fill)?),
        (1, 2) => ErasedMap::R1x2(construct_typed::<R, 1, 2>(rule, hom, legs, fill)?),
        (2, 2) => ErasedMap::R2x2(construct_typed::<R, 2, 2>(rule, hom, legs, fill)?),
        (nout, nin) => return Err(Error::UnsupportedRank { nout, nin }),
    })
}

fn transform_typed<
    R: UserRule,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
>(
    ctx: &mut Ctx<R::Key>,
    rule: &R,
    src: &TensorMap<f64, SRC_NOUT, SRC_NIN>,
    operation: TreeTransformOperation,
    dst_hom: FusionTreeHomSpace,
    dst_legs: &[LegInfo],
) -> Result<TensorMap<f64, DST_NOUT, DST_NIN>, Error> {
    let space = build_fusion_space::<R, DST_NOUT, DST_NIN>(rule, dst_hom, dst_legs)?;
    let len = space.required_len()?;
    let mut dst = TensorMap::from_vec_with_fusion_space(vec![0.0; len], space)?;
    tree_transform_into_with_context(
        ctx.tree_context_mut(),
        rule,
        operation,
        &mut dst,
        src,
        1.0,
        0.0,
    )?;
    Ok(dst)
}

/// Which tree transform a leg re-arrangement uses.
enum TransformKind<'a> {
    Permute,
    Braid { levels: &'a [usize] },
    Transpose,
}

/// Shared implementation of permute / braid / transpose: computes the
/// destination hom space and leg tables, then dispatches to the typed
/// tree-transform kernel.
fn transform_erased<R: UserRule>(
    ctx: &mut Ctx<R::Key>,
    rule: &R,
    map: &ErasedMap,
    legs: &[LegInfo],
    codomain_axes: &[usize],
    domain_axes: &[usize],
    kind: TransformKind<'_>,
) -> Result<(ErasedMap, Vec<LegInfo>), Error> {
    let nout = map.nout();
    let rank = legs.len();
    let dst_hom = map.hom().permute(rule, codomain_axes, domain_axes)?;

    let mut dst_legs = Vec::with_capacity(rank);
    for &axis in codomain_axes {
        dst_legs.push(if axis < nout {
            legs[axis].clone()
        } else {
            legs[axis].dualized(rule)
        });
    }
    for &axis in domain_axes {
        dst_legs.push(if axis >= nout {
            legs[axis].clone()
        } else {
            legs[axis].dualized(rule)
        });
    }

    let operation = match kind {
        TransformKind::Permute => TreeTransformOperation::permute(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
        ),
        TransformKind::Braid { levels } => {
            if levels.len() != rank {
                return Err(Error::InvalidArgument(format!(
                    "braid levels must list one level per source axis \
                     (expected {rank}, got {})",
                    levels.len()
                )));
            }
            TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            )
        }
        TransformKind::Transpose => TreeTransformOperation::transpose(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
        ),
    };

    let dst_nout = codomain_axes.len();
    let dst = match (map, dst_nout) {
        (ErasedMap::R0x0(m), 0) => ErasedMap::R0x0(transform_typed::<R, 0, 0, 0, 0>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R1x0(m), 1) => ErasedMap::R1x0(transform_typed::<R, 1, 0, 1, 0>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R1x0(m), 0) => ErasedMap::R0x1(transform_typed::<R, 0, 1, 1, 0>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R0x1(m), 1) => ErasedMap::R1x0(transform_typed::<R, 1, 0, 0, 1>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R0x1(m), 0) => ErasedMap::R0x1(transform_typed::<R, 0, 1, 0, 1>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R2x0(m), 2) => ErasedMap::R2x0(transform_typed::<R, 2, 0, 2, 0>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R2x0(m), 1) => ErasedMap::R1x1(transform_typed::<R, 1, 1, 2, 0>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R2x0(m), 0) => ErasedMap::R0x2(transform_typed::<R, 0, 2, 2, 0>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R1x1(m), 2) => ErasedMap::R2x0(transform_typed::<R, 2, 0, 1, 1>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R1x1(m), 1) => ErasedMap::R1x1(transform_typed::<R, 1, 1, 1, 1>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R1x1(m), 0) => ErasedMap::R0x2(transform_typed::<R, 0, 2, 1, 1>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R0x2(m), 2) => ErasedMap::R2x0(transform_typed::<R, 2, 0, 0, 2>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R0x2(m), 1) => ErasedMap::R1x1(transform_typed::<R, 1, 1, 0, 2>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R0x2(m), 0) => ErasedMap::R0x2(transform_typed::<R, 0, 2, 0, 2>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R2x1(m), 2) => ErasedMap::R2x1(transform_typed::<R, 2, 1, 2, 1>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R2x1(m), 1) => ErasedMap::R1x2(transform_typed::<R, 1, 2, 2, 1>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R1x2(m), 2) => ErasedMap::R2x1(transform_typed::<R, 2, 1, 1, 2>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R1x2(m), 1) => ErasedMap::R1x2(transform_typed::<R, 1, 2, 1, 2>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (ErasedMap::R2x2(m), 2) => ErasedMap::R2x2(transform_typed::<R, 2, 2, 2, 2>(
            ctx, rule, m, operation, dst_hom, &dst_legs,
        )?),
        (map, dst_nout) => {
            return Err(Error::UnsupportedRank {
                nout: dst_nout,
                nin: map.nout() + map.nin() - dst_nout,
            })
        }
    };
    Ok((dst, dst_legs))
}

fn contract_typed<R: UserRule, const P: usize, const K: usize, const Q: usize>(
    ctx: &mut Ctx<R::Key>,
    rule: &R,
    lhs: &TensorMap<f64, P, K>,
    rhs: &TensorMap<f64, K, Q>,
    dst_hom: FusionTreeHomSpace,
    dst_legs: &[LegInfo],
) -> Result<TensorMap<f64, P, Q>, Error> {
    let space = build_fusion_space::<R, P, Q>(rule, dst_hom, dst_legs)?;
    let len = space.required_len()?;
    let mut dst = TensorMap::from_vec_with_fusion_space(vec![0.0; len], space)?;
    let lhs_axes: Vec<usize> = (P..P + K).collect();
    let rhs_axes: Vec<usize> = (0..K).collect();
    ctx.tensorcontract_fusion_into(
        rule,
        &mut dst,
        lhs,
        rhs,
        TensorContractSpec::with_default_output_order(&lhs_axes, &rhs_axes),
        1.0,
        0.0,
    )?;
    Ok(dst)
}

/// Composition-shaped contraction dispatch: `lhs` must already be split as
/// `(open | contracted)` and `rhs` as `(contracted | open)`.
fn compose_erased<R: UserRule>(
    ctx: &mut Ctx<R::Key>,
    rule: &R,
    lhs: &ErasedMap,
    rhs: &ErasedMap,
    dst_hom: FusionTreeHomSpace,
    dst_legs: &[LegInfo],
) -> Result<ErasedMap, Error> {
    use ErasedMap as M;
    macro_rules! arm {
        ($l:expr, $r:expr, $DV:ident, $P:literal, $K:literal, $Q:literal) => {
            M::$DV(contract_typed::<R, $P, $K, $Q>(
                ctx, rule, $l, $r, dst_hom, dst_legs,
            )?)
        };
    }
    Ok(match (lhs, rhs) {
        // K = 0 (outer products)
        (M::R0x0(l), M::R0x0(r)) => arm!(l, r, R0x0, 0, 0, 0),
        (M::R0x0(l), M::R0x1(r)) => arm!(l, r, R0x1, 0, 0, 1),
        (M::R0x0(l), M::R0x2(r)) => arm!(l, r, R0x2, 0, 0, 2),
        (M::R1x0(l), M::R0x0(r)) => arm!(l, r, R1x0, 1, 0, 0),
        (M::R1x0(l), M::R0x1(r)) => arm!(l, r, R1x1, 1, 0, 1),
        (M::R1x0(l), M::R0x2(r)) => arm!(l, r, R1x2, 1, 0, 2),
        (M::R2x0(l), M::R0x0(r)) => arm!(l, r, R2x0, 2, 0, 0),
        (M::R2x0(l), M::R0x1(r)) => arm!(l, r, R2x1, 2, 0, 1),
        (M::R2x0(l), M::R0x2(r)) => arm!(l, r, R2x2, 2, 0, 2),
        // K = 1
        (M::R0x1(l), M::R1x0(r)) => arm!(l, r, R0x0, 0, 1, 0),
        (M::R0x1(l), M::R1x1(r)) => arm!(l, r, R0x1, 0, 1, 1),
        (M::R0x1(l), M::R1x2(r)) => arm!(l, r, R0x2, 0, 1, 2),
        (M::R1x1(l), M::R1x0(r)) => arm!(l, r, R1x0, 1, 1, 0),
        (M::R1x1(l), M::R1x1(r)) => arm!(l, r, R1x1, 1, 1, 1),
        (M::R1x1(l), M::R1x2(r)) => arm!(l, r, R1x2, 1, 1, 2),
        (M::R2x1(l), M::R1x0(r)) => arm!(l, r, R2x0, 2, 1, 0),
        (M::R2x1(l), M::R1x1(r)) => arm!(l, r, R2x1, 2, 1, 1),
        (M::R2x1(l), M::R1x2(r)) => arm!(l, r, R2x2, 2, 1, 2),
        // K = 2
        (M::R0x2(l), M::R2x0(r)) => arm!(l, r, R0x0, 0, 2, 0),
        (M::R0x2(l), M::R2x1(r)) => arm!(l, r, R0x1, 0, 2, 1),
        (M::R0x2(l), M::R2x2(r)) => arm!(l, r, R0x2, 0, 2, 2),
        (M::R1x2(l), M::R2x0(r)) => arm!(l, r, R1x0, 1, 2, 0),
        (M::R1x2(l), M::R2x1(r)) => arm!(l, r, R1x1, 1, 2, 1),
        (M::R1x2(l), M::R2x2(r)) => arm!(l, r, R1x2, 1, 2, 2),
        (M::R2x2(l), M::R2x0(r)) => arm!(l, r, R2x0, 2, 2, 0),
        (M::R2x2(l), M::R2x1(r)) => arm!(l, r, R2x1, 2, 2, 1),
        (M::R2x2(l), M::R2x2(r)) => arm!(l, r, R2x2, 2, 2, 2),
        (lhs, rhs) => {
            return Err(Error::InvalidArgument(format!(
                "composition shape mismatch: lhs domain rank {} vs rhs codomain rank {}",
                lhs.nin(),
                rhs.nout()
            )))
        }
    })
}

/// General axis contraction, lowered to permutes plus one composition-shaped
/// contraction (the same lowering the expert layer documents for
/// `tensorcontract!`). Returns the result in the default output order:
/// `lhs` open axes ascending (codomain side), then `rhs` open axes ascending
/// (domain side).
fn contract_erased<R: UserRule>(
    ctx: &mut Ctx<R::Key>,
    rule: &R,
    lhs: (&ErasedMap, &[LegInfo]),
    rhs: (&ErasedMap, &[LegInfo]),
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> Result<(ErasedMap, Vec<LegInfo>), Error> {
    let (lhs_map, lhs_legs) = lhs;
    let (rhs_map, rhs_legs) = rhs;
    if lhs_axes.len() != rhs_axes.len() {
        return Err(Error::InvalidArgument(format!(
            "contracted axis lists differ in length: {} vs {}",
            lhs_axes.len(),
            rhs_axes.len()
        )));
    }
    let lhs_open = open_axes(lhs_axes, lhs_legs.len())?;
    let rhs_open = open_axes(rhs_axes, rhs_legs.len())?;
    let contracted = lhs_axes.len();

    // Bring lhs to (open | contracted) and rhs to (contracted | open),
    // skipping the transform when the tensor is already in that shape.
    let lhs_ready = lhs_map.nout() == lhs_open.len()
        && lhs_open.iter().copied().eq(0..lhs_open.len())
        && lhs_axes.iter().copied().eq(lhs_open.len()..lhs_legs.len());
    let rhs_ready = rhs_map.nout() == contracted && rhs_axes.iter().copied().eq(0..contracted);

    let lhs_permuted;
    let (lhs_map, lhs_legs) = if lhs_ready {
        (lhs_map, lhs_legs)
    } else {
        lhs_permuted = transform_erased(
            ctx,
            rule,
            lhs_map,
            lhs_legs,
            &lhs_open,
            lhs_axes,
            TransformKind::Permute,
        )?;
        (&lhs_permuted.0, lhs_permuted.1.as_slice())
    };
    let rhs_permuted;
    let (rhs_map, rhs_legs) = if rhs_ready {
        (rhs_map, rhs_legs)
    } else {
        rhs_permuted = transform_erased(
            ctx,
            rule,
            rhs_map,
            rhs_legs,
            rhs_axes,
            &rhs_open,
            TransformKind::Permute,
        )?;
        (&rhs_permuted.0, rhs_permuted.1.as_slice())
    };

    let dst_hom = FusionTreeHomSpace::compose(rule, lhs_map.hom(), rhs_map.hom())?;
    let mut dst_legs = lhs_legs[..lhs_map.nout()].to_vec();
    dst_legs.extend_from_slice(&rhs_legs[rhs_map.nout()..]);
    let dst = compose_erased(ctx, rule, lhs_map, rhs_map, dst_hom, &dst_legs)?;
    Ok((dst, dst_legs))
}

fn open_axes(contracted: &[usize], rank: usize) -> Result<Vec<usize>, Error> {
    let mut seen = vec![false; rank];
    for &axis in contracted {
        if axis >= rank || seen[axis] {
            return Err(Error::InvalidArgument(format!(
                "invalid contracted axis list {contracted:?} for rank {rank}"
            )));
        }
        seen[axis] = true;
    }
    Ok((0..rank).filter(|&axis| !seen[axis]).collect())
}

/// Quantum-dimension-weighted Frobenius inner product over the stored
/// blocks: `sum_c dim(c) * <a_c, b_c>`, matching TensorKit's `dot`.
fn weighted_inner<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    rule: &R,
    structure: &BlockStructure,
    a: &[f64],
    b: &[f64],
) -> Result<f64, Error> {
    let mut total = 0.0;
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let weight = match block.key() {
            BlockKey::FusionTree(key) => {
                let coupled = key
                    .codomain_tree()
                    .coupled()
                    .unwrap_or_else(|| rule.vacuum());
                rule.dim_scalar(coupled)
            }
            _ => 1.0,
        };
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        // ponytail: odometer walk per element; blocks are small strided
        // views into coupled matrices. Vectorize per contiguous run if this
        // ever shows up in a profile.
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        let mut partial = 0.0;
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            partial += a[position] * b[position];
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
        total += weight * partial;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Public tensor type.
// ---------------------------------------------------------------------------

/// A block-sparse symmetric tensor with dynamic rank, tied to a [`Runtime`].
///
/// `Tensor` is the user-layer face of [`tenet_core::TensorMap`]: the fusion
/// rule (U1 / Z2 / fZ2 / SU2 / U1 x fZ2) is fixed per tensor by the
/// [`Space`]s it was built from, and the codomain/domain split is a runtime
/// property. Mixing tensors of different rules or different runtimes in one
/// operation is an error.
///
/// Scalar type is currently `f64`.
///
/// # Examples
///
/// ```
/// use tenet::prelude::*;
///
/// let rt = Runtime::builder().build()?;
/// let v = Space::z2([(0, 1), (1, 1)]);
///
/// // Same numbers as the tutorial's expert-layer Z2 example.
/// let a = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
///     BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
///     _ => 3.0,
/// })?;
/// let b = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
///     BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 5.0,
///     _ => 7.0,
/// })?;
/// let c = a.compose(&b)?;
/// assert_eq!(c.data(), &[10.0, 21.0]);
/// # Ok::<(), tenet::prelude::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Tensor {
    rt: Runtime,
    rule: RuleKind,
    /// Per-axis degeneracy tables (codomain axes first), keyed by the
    /// internal sectors of the corresponding hom-space legs.
    legs: Vec<LegInfo>,
    map: ErasedMap,
}

impl Tensor {
    fn build<'a, C, D>(rt: &Runtime, codomain: C, domain: D, fill: Fill<'_>) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        let codomain: Vec<&Space> = codomain.into_iter().collect();
        let domain: Vec<&Space> = domain.into_iter().collect();
        let mut spaces = codomain.iter().chain(domain.iter());
        let rule_kind = spaces
            .next()
            .ok_or_else(|| {
                Error::InvalidArgument(
                    "at least one leg is required to infer the fusion rule".to_string(),
                )
            })?
            .rule;
        if spaces.any(|space| space.rule != rule_kind) {
            return Err(Error::RuleMismatch);
        }

        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|space| space.sector_leg())),
            FusionProductSpace::new(domain.iter().map(|space| space.sector_leg())),
        );
        let legs: Vec<LegInfo> = codomain
            .iter()
            .chain(domain.iter())
            .map(|space| LegInfo::from_space(space))
            .collect();
        let map = with_rule!(rule_kind, rule, {
            construct_erased(rule, hom, &legs, codomain.len(), domain.len(), fill)
        })?;
        Ok(Self {
            rt: rt.clone(),
            rule: rule_kind,
            legs,
            map,
        })
    }

    /// Zero tensor on `codomain <- domain`. All spaces must share one
    /// fusion rule.
    pub fn zeros<'a, C, D>(rt: &Runtime, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::build(rt, codomain, domain, Fill::Zeros)
    }

    /// Random tensor on `codomain <- domain`, entries uniform in `[-1, 1)`.
    ///
    /// Deterministic per runtime: the n-th `rand` call on a given runtime
    /// always produces the same tensor. Use [`Self::rand_with_seed`] for an
    /// explicit stream.
    pub fn rand<'a, C, D>(rt: &Runtime, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::build(rt, codomain, domain, Fill::Rand(rt.next_rand_seed()))
    }

    /// Random tensor with an explicit seed (splitmix64 stream, entries
    /// uniform in `[-1, 1)`).
    pub fn rand_with_seed<'a, C, D>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        seed: u64,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::build(rt, codomain, domain, Fill::Rand(seed))
    }

    /// Tensor filled block-by-block: `fill(key, indices)` is called for
    /// every element of every symmetry-allowed block, with `indices` local
    /// to the block (degeneracy coordinates, codomain axes first). Mirrors
    /// [`tenet_core::TensorMap::from_block_fn_with_fusion_space`].
    pub fn from_block_fn<'a, C, D, F>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        mut fill: F,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
        F: FnMut(&BlockKey, &[usize]) -> f64,
    {
        Self::build(rt, codomain, domain, Fill::BlockFn(&mut fill))
    }

    /// Number of codomain legs.
    pub fn codomain_rank(&self) -> usize {
        self.map.nout()
    }

    /// Number of domain legs.
    pub fn domain_rank(&self) -> usize {
        self.map.nin()
    }

    /// Total number of legs.
    pub fn rank(&self) -> usize {
        self.legs.len()
    }

    /// Flat storage in the TensorKit-equivalent coupled-sector matrix
    /// layout (column-major inside each coupled block).
    pub fn data(&self) -> &[f64] {
        self.map.data()
    }

    fn check_same_world(&self, other: &Self) -> Result<(), Error> {
        if self.rule != other.rule {
            return Err(Error::RuleMismatch);
        }
        if !self.rt.same_runtime(&other.rt) {
            return Err(Error::RuntimeMismatch);
        }
        Ok(())
    }

    /// Categorical composition `self * rhs`: contracts `self`'s domain with
    /// `rhs`'s codomain, leg by leg. TensorKit `A * B`.
    pub fn compose(&self, rhs: &Self) -> Result<Self, Error> {
        if self.domain_rank() != rhs.codomain_rank() {
            return Err(Error::InvalidArgument(format!(
                "compose shape mismatch: lhs domain rank {} vs rhs codomain rank {}",
                self.domain_rank(),
                rhs.codomain_rank()
            )));
        }
        let lhs_axes: Vec<usize> = (self.codomain_rank()..self.rank()).collect();
        let rhs_axes: Vec<usize> = (0..rhs.codomain_rank()).collect();
        self.contract(rhs, &lhs_axes, &rhs_axes)
    }

    /// Contracts `lhs_axes` of `self` with `rhs_axes` of `rhs` (pairwise, in
    /// list order), with the default output order: `self`'s open axes
    /// ascending become the codomain, `rhs`'s open axes ascending become the
    /// domain. TensorKit `tensorcontract!` with default `pAB`.
    pub fn contract(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        self.check_same_world(rhs)?;
        let mut state = self.rt.lock();
        let (map, legs) = with_rule_ctx!(self.rule, state, rule, ctx, {
            contract_erased(
                ctx,
                rule,
                (&self.map, &self.legs),
                (&rhs.map, &rhs.legs),
                lhs_axes,
                rhs_axes,
            )
        })?;
        drop(state);
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            legs,
            map,
        })
    }

    /// Like [`Self::contract`], but with an explicit output axis order
    /// (`pAB`): `output_axes[i]` picks, for output position `i`, an index
    /// into the default output order (`self` open axes ascending, then
    /// `rhs` open axes ascending). The codomain/domain split of the result
    /// keeps `self`'s open-leg count on the codomain side.
    pub fn contract_ordered(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<Self, Error> {
        let contracted = self.contract(rhs, lhs_axes, rhs_axes)?;
        if output_axes.len() != contracted.rank() {
            return Err(Error::InvalidArgument(format!(
                "output axis list length {} does not match open rank {}",
                output_axes.len(),
                contracted.rank()
            )));
        }
        let split = contracted.codomain_rank();
        if output_axes.iter().copied().eq(0..contracted.rank()) {
            return Ok(contracted);
        }
        contracted.permute(&output_axes[..split], &output_axes[split..])
    }

    /// TensorKit `permute`: re-arranges legs with symmetric braiding.
    /// `codomain_axes` and `domain_axes` list source axis numbers
    /// (`0..rank`, codomain axes first) for the new codomain and domain.
    pub fn permute(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> Result<Self, Error> {
        self.transformed(codomain_axes, domain_axes, TransformKind::Permute)
    }

    /// TensorKit `braid`: explicit braid with one level per source axis
    /// (levels decide which strand crosses above at each transposition).
    pub fn braid(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, Error> {
        self.transformed(codomain_axes, domain_axes, TransformKind::Braid { levels })
    }

    /// TensorKit `transpose`: the planar transpose `codomain <- domain`
    /// to `domain' <- codomain'`, i.e. cyclic leg rotation without
    /// braiding. Equivalent to
    /// `transpose_into` with reversed domain axes as the new codomain and
    /// reversed codomain axes as the new domain.
    pub fn transpose(&self) -> Result<Self, Error> {
        let codomain_axes: Vec<usize> = (self.codomain_rank()..self.rank()).rev().collect();
        let domain_axes: Vec<usize> = (0..self.codomain_rank()).rev().collect();
        self.transformed(&codomain_axes, &domain_axes, TransformKind::Transpose)
    }

    fn transformed(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        kind: TransformKind<'_>,
    ) -> Result<Self, Error> {
        let mut state = self.rt.lock();
        let (map, legs) = with_rule_ctx!(self.rule, state, rule, ctx, {
            transform_erased(
                ctx,
                rule,
                &self.map,
                &self.legs,
                codomain_axes,
                domain_axes,
                kind,
            )
        })?;
        drop(state);
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            legs,
            map,
        })
    }

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block (real scalars: transpose only).
    pub fn adjoint(&self) -> Result<Self, Error> {
        use ErasedMap as M;
        let map = with_rule!(self.rule, rule, {
            Ok::<_, Error>(match &self.map {
                M::R0x0(m) => M::R0x0(tenet_tensors::adjoint(rule, m)?),
                M::R1x0(m) => M::R0x1(tenet_tensors::adjoint(rule, m)?),
                M::R0x1(m) => M::R1x0(tenet_tensors::adjoint(rule, m)?),
                M::R1x1(m) => M::R1x1(tenet_tensors::adjoint(rule, m)?),
                M::R2x0(m) => M::R0x2(tenet_tensors::adjoint(rule, m)?),
                M::R0x2(m) => M::R2x0(tenet_tensors::adjoint(rule, m)?),
                M::R2x1(m) => M::R1x2(tenet_tensors::adjoint(rule, m)?),
                M::R1x2(m) => M::R2x1(tenet_tensors::adjoint(rule, m)?),
                M::R2x2(m) => M::R2x2(tenet_tensors::adjoint(rule, m)?),
            })
        })?;
        let mut legs = self.legs[self.codomain_rank()..].to_vec();
        legs.extend_from_slice(&self.legs[..self.codomain_rank()]);
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            legs,
            map,
        })
    }

    /// Frobenius norm, weighted by coupled-sector quantum dimensions
    /// (`norm(t)^2 = sum_c dim(c) * |block_c|^2`), matching TensorKit's
    /// `norm`.
    pub fn norm(&self) -> Result<f64, Error> {
        let data = self.map.data();
        with_rule!(self.rule, rule, {
            weighted_inner(rule, self.map.structure(), data, data)
        })
        .map(f64::sqrt)
    }

    /// Returns `factor * self`.
    pub fn scale(&self, factor: f64) -> Result<Self, Error> {
        let mut data = self.map.data_vec();
        for value in &mut data {
            *value *= factor;
        }
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            legs: self.legs.clone(),
            map: self.map.with_data(data)?,
        })
    }

    /// Returns `alpha * self + beta * other`. Both tensors must live on the
    /// same spaces (identical hom space and block layout).
    pub fn add(&self, other: &Self, alpha: f64, beta: f64) -> Result<Self, Error> {
        self.check_same_space(other)?;
        let mut data = self.map.data_vec();
        for (value, &rhs) in data.iter_mut().zip(other.map.data()) {
            *value = alpha * *value + beta * rhs;
        }
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            legs: self.legs.clone(),
            map: self.map.with_data(data)?,
        })
    }

    /// Frobenius inner product `<self, other>`, weighted by coupled-sector
    /// quantum dimensions; `t.inner(&t)? == t.norm()?.powi(2)` up to
    /// floating-point error. Both tensors must live on the same spaces.
    pub fn inner(&self, other: &Self) -> Result<f64, Error> {
        self.check_same_space(other)?;
        with_rule!(self.rule, rule, {
            weighted_inner(
                rule,
                self.map.structure(),
                self.map.data(),
                other.map.data(),
            )
        })
    }

    fn check_same_space(&self, other: &Self) -> Result<(), Error> {
        self.check_same_world(other)?;
        if self.map.hom() != other.map.hom() || self.map.structure() != other.map.structure() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        Ok(())
    }
}
