//! User-layer symmetric tensor: dynamic rank, rule-erased, runtime-carrying.
//!
//! A [`Tensor`] stores a [`tenet_tensors::DynamicFusionMapSpace`] handle plus
//! flat `f64` storage in the TensorKit-equivalent coupled-sector matrix
//! layout. The concrete fusion rule is erased behind
//! [`crate::space::RuleKind`]; rank is fully dynamic (no ceiling), matching
//! TensorKit's `tensorcontract!`. Operations lock the shared [`Runtime`]
//! state once and dispatch to the dynamic expert entry points
//! (`tensorcontract_fusion_dyn_into`, `tree_transform_dyn_into`,
//! `adjoint_dyn`).

use std::collections::BTreeMap;
use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    SectorId,
};
use tenet_tensors::{DynamicFusionMapSpace, TensorContractSpec, TreeTransformOperation};

use crate::error::Error;
use crate::runtime::{with_rule_ctx, Runtime};
use crate::space::{with_rule, RuleKind, Space};

/// Degeneracy table of one tensor leg, keyed by the *internal* sectors of
/// the corresponding hom-space [`tenet_core::SectorLeg`]. Used only while
/// constructing a fresh tensor; afterwards the space handle carries all
/// structure.
#[derive(Clone, Debug)]
struct LegInfo {
    degs: BTreeMap<SectorId, usize>,
}

impl LegInfo {
    fn from_space(space: &Space) -> Self {
        Self {
            degs: space.sectors.iter().copied().collect(),
        }
    }

    fn deg(&self, sector: SectorId) -> Result<usize, Error> {
        self.degs.get(&sector).copied().ok_or_else(|| {
            Error::InvalidArgument(format!("sector {sector:?} not present on this leg"))
        })
    }
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

/// Builds the coupled-layout dynamic fusion space for the given hom space
/// and per-leg degeneracy tables.
fn build_space<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    rule: &R,
    hom: FusionTreeHomSpace,
    legs: &[LegInfo],
    nout: usize,
) -> Result<DynamicFusionMapSpace, Error> {
    debug_assert_eq!(legs.len(), nout + hom.domain().len());
    let keys = hom.fusion_tree_keys(rule);
    let mut shapes = Vec::with_capacity(keys.len());
    for key in &keys {
        let mut shape = Vec::with_capacity(legs.len());
        for (leg, &sector) in legs[..nout].iter().zip(key.codomain_uncoupled()) {
            shape.push(leg.deg(sector)?);
        }
        for (leg, &sector) in legs[nout..].iter().zip(key.domain_uncoupled()) {
            shape.push(leg.deg(sector)?);
        }
        shapes.push(shape);
    }
    DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, shapes).map_err(Into::into)
}

/// Fills every symmetry-allowed block element via `fill(key, indices)`,
/// mirroring [`tenet_core::TensorMap::from_block_fn_with_fusion_space`]
/// (degeneracy coordinates local to the block, codomain axes first, first
/// axis fastest).
fn fill_block_elements(
    structure: &BlockStructure,
    data: &mut [f64],
    fill: &mut dyn FnMut(&BlockKey, &[usize]) -> f64,
) -> Result<(), Error> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            data[position] = fill(block.key(), &indices);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok(())
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

/// Which tree transform a leg re-arrangement uses.
enum TransformKind<'a> {
    Permute,
    Braid { levels: &'a [usize] },
    Transpose,
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

// ---------------------------------------------------------------------------
// Public tensor type.
// ---------------------------------------------------------------------------

/// A block-sparse symmetric tensor with dynamic rank, tied to a [`Runtime`].
///
/// `Tensor` is the user-layer face of the expert layer's dynamic-rank
/// machinery: the fusion rule (U1 / Z2 / fZ2 / SU2 / U1 x fZ2) is fixed per
/// tensor by the [`Space`]s it was built from, and the codomain/domain split
/// is a runtime property with no rank ceiling. Mixing tensors of different
/// rules or different runtimes in one operation is an error.
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
    space: Arc<DynamicFusionMapSpace>,
    data: Vec<f64>,
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
        let (space, data) = with_rule!(rule_kind, rule, {
            let space = build_space(rule, hom, &legs, codomain.len())?;
            let len = space.required_len()?;
            let mut data = vec![0.0; len];
            match fill {
                Fill::Zeros => {}
                Fill::Rand(seed) => {
                    let mut state = seed;
                    for value in &mut data {
                        *value = rand_unit(&mut state);
                    }
                }
                Fill::BlockFn(fill) => {
                    fill_block_elements(space.structure(), &mut data, fill)?;
                }
            }
            Ok::<_, Error>((space, data))
        })?;
        Ok(Self {
            rt: rt.clone(),
            rule: rule_kind,
            space: Arc::new(space),
            data,
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
        self.space.nout()
    }

    /// Number of domain legs.
    pub fn domain_rank(&self) -> usize {
        self.space.nin()
    }

    /// Total number of legs.
    pub fn rank(&self) -> usize {
        self.space.rank()
    }

    /// Flat storage in the TensorKit-equivalent coupled-sector matrix
    /// layout (column-major inside each coupled block).
    pub fn data(&self) -> &[f64] {
        &self.data
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
        if lhs_axes.len() != rhs_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "contracted axis lists differ in length: {} vs {}",
                lhs_axes.len(),
                rhs_axes.len()
            )));
        }
        open_axes(lhs_axes, self.rank())?;
        open_axes(rhs_axes, rhs.rank())?;
        let mut state = self.rt.lock();
        let (space, data) = with_rule_ctx!(self.rule, state, rule, ctx, {
            let dst_space = DynamicFusionMapSpace::contracted(
                rule,
                &self.space,
                &rhs.space,
                lhs_axes,
                rhs_axes,
            )?;
            let mut data = vec![0.0; dst_space.required_len()?];
            ctx.tensorcontract_fusion_dyn_into(
                rule,
                &dst_space,
                &mut data,
                &self.space,
                &self.data,
                &rhs.space,
                &rhs.data,
                TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes),
                1.0,
                0.0,
            )?;
            Ok::<_, Error>((dst_space, data))
        })?;
        drop(state);
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data,
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
        let rank = self.rank();
        let nout = self.codomain_rank();
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

        let mut state = self.rt.lock();
        let (space, data) = with_rule_ctx!(self.rule, state, rule, ctx, {
            let dst_space = self.space.transformed(rule, &operation)?;
            let mut data = vec![0.0; dst_space.required_len()?];
            ctx.tree_context_mut().tree_transform_dyn_into(
                rule,
                operation,
                &Arc::clone(dst_space.structure()),
                self.space.structure(),
                &mut data,
                &self.data,
                1.0,
                0.0,
            )?;
            Ok::<_, Error>((dst_space, data))
        })?;
        drop(state);
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data,
        })
    }

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block (real scalars: transpose only).
    pub fn adjoint(&self) -> Result<Self, Error> {
        let (space, data) = with_rule!(self.rule, rule, {
            tenet_tensors::adjoint_dyn(rule, &self.space, &self.data)
        })?;
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data,
        })
    }

    /// Frobenius norm, weighted by coupled-sector quantum dimensions
    /// (`norm(t)^2 = sum_c dim(c) * |block_c|^2`), matching TensorKit's
    /// `norm`.
    pub fn norm(&self) -> Result<f64, Error> {
        with_rule!(self.rule, rule, {
            weighted_inner(rule, self.space.structure(), &self.data, &self.data)
        })
        .map(f64::sqrt)
    }

    /// Returns `factor * self`.
    pub fn scale(&self, factor: f64) -> Result<Self, Error> {
        let mut data = self.data.clone();
        for value in &mut data {
            *value *= factor;
        }
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data,
        })
    }

    /// Returns `alpha * self + beta * other`. Both tensors must live on the
    /// same spaces (identical hom space and block layout).
    pub fn add(&self, other: &Self, alpha: f64, beta: f64) -> Result<Self, Error> {
        self.check_same_space(other)?;
        let mut data = self.data.clone();
        for (value, &rhs) in data.iter_mut().zip(&other.data) {
            *value = alpha * *value + beta * rhs;
        }
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data,
        })
    }

    /// Frobenius inner product `<self, other>`, weighted by coupled-sector
    /// quantum dimensions; `t.inner(&t)? == t.norm()?.powi(2)` up to
    /// floating-point error. Both tensors must live on the same spaces.
    pub fn inner(&self, other: &Self) -> Result<f64, Error> {
        self.check_same_space(other)?;
        with_rule!(self.rule, rule, {
            weighted_inner(rule, self.space.structure(), &self.data, &other.data)
        })
    }

    fn check_same_space(&self, other: &Self) -> Result<(), Error> {
        self.check_same_world(other)?;
        if *self.space != *other.space {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        Ok(())
    }
}
