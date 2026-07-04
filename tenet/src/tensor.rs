//! User-layer symmetric tensor: dynamic rank, rule-erased, runtime-carrying.
//!
//! A [`Tensor`] stores a [`tenet_tensors::DynamicFusionMapSpace`] handle plus
//! flat scalar storage (`f64` or `Complex64`, chosen at construction) in the
//! TensorKit-equivalent coupled-sector matrix layout. The concrete fusion
//! rule is erased behind [`crate::space::RuleKind`] and the scalar type is
//! erased behind an internal storage enum, mirroring the dynamic-rank
//! decision; rank is fully dynamic (no ceiling), matching TensorKit's
//! `tensorcontract!`. Operations lock the shared [`Runtime`] state once,
//! dispatch on the stored dtype once per call (never per block), and forward
//! to the dynamic expert entry points (`tensorcontract_fusion_dyn_into`,
//! `tree_transform_dyn_into`, `adjoint_dyn`).

use std::hash::Hash;
use std::sync::Arc;

use num_complex::Complex64;
#[cfg(feature = "cuda")]
use tenet_core::TensorStorage;
use tenet_core::{
    BlockKey, BlockStructure, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    Placement, SectorId,
};
use tenet_matrixalgebra::{DynFactor, FactorScalar, SectorSpectrum, Truncation};
#[cfg(feature = "cuda")]
use tenet_tensors::cuda::{CudaStorage, CudaStorageGemm};
use tenet_tensors::{
    DynamicFusionMapSpace, RecouplingCoefficientAction, TensorContractSpec, TreeTransformOperation,
};

use crate::error::Error;
use crate::runtime::{with_rule_ctx, Ctx, Ctxs, Runtime};
use crate::space::{with_rule, RuleKind, Space};

/// The scalar type a [`Tensor`] stores, fixed at construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Dtype {
    /// Real double precision (`f64`).
    F64,
    /// Complex double precision ([`Complex64`]).
    C64,
}

/// Dtype-erased flat storage in the coupled-sector matrix layout. The
/// device variant shares the immutable buffer behind an `Arc` (operations
/// always write fresh destinations), keeping `Tensor: Clone` cheap and the
/// host paths untouched.
#[derive(Clone, Debug)]
enum Data {
    F64(Vec<f64>),
    C64(Vec<Complex64>),
    #[cfg(feature = "cuda")]
    CudaF64(Arc<CudaStorage>),
}

/// Explicit "no device kernel yet" error; device tensors never fall back
/// to host execution silently.
#[cfg(feature = "cuda")]
fn device_unsupported(what: &str) -> Error {
    Error::UnsupportedOnDevice(format!(
        "{what} has no device implementation yet; move the tensor to the \
         host explicitly with to_host()"
    ))
}

/// The scalar types the user layer stores: the expert-layer scalar machinery
/// plus the glue to lift typed data into the erased [`Data`] storage and to
/// pick the matching per-scalar execution context.
trait UserScalar: FactorScalar + RecouplingCoefficientAction<f64> {
    fn lift(data: Vec<Self>) -> Data;
    fn ctx_of<Key: Clone + Eq + Hash>(ctxs: &mut Ctxs<Key>) -> &mut Ctx<Self, Key>;
    fn rand_unit(state: &mut u64) -> Self;
}

impl UserScalar for f64 {
    fn lift(data: Vec<Self>) -> Data {
        Data::F64(data)
    }

    fn ctx_of<Key: Clone + Eq + Hash>(ctxs: &mut Ctxs<Key>) -> &mut Ctx<Self, Key> {
        &mut ctxs.f64
    }

    fn rand_unit(state: &mut u64) -> Self {
        rand_unit(state)
    }
}

impl UserScalar for Complex64 {
    fn lift(data: Vec<Self>) -> Data {
        Data::C64(data)
    }

    fn ctx_of<Key: Clone + Eq + Hash>(ctxs: &mut Ctxs<Key>) -> &mut Ctx<Self, Key> {
        &mut ctxs.c64
    }

    fn rand_unit(state: &mut u64) -> Self {
        Complex64::new(rand_unit(state), rand_unit(state))
    }
}

/// Dispatches once on the stored dtype of `$tensor`, binding `$data` to the
/// typed data vector in both arms; `$body` must be dtype-generic (the expert
/// entry points are generic over the scalar).
macro_rules! with_data {
    ($tensor:expr, $data:ident, $body:expr) => {
        match &$tensor.data {
            Data::F64($data) => $body,
            Data::C64($data) => $body,
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("this operation")),
        }
    };
}

/// Result of [`Tensor::svd_trunc`]: `t ~ u * s * vh` with the truncated bond
/// (TensorKit 0.17 / MatrixAlgebraKit `svd_trunc`). `singular_values` holds
/// the kept per-sector spectra and `error` the quantum-dimension-weighted
/// 2-norm of everything discarded.
#[derive(Clone, Debug)]
pub struct SvdTrunc {
    pub u: Tensor,
    pub s: Tensor,
    pub vh: Tensor,
    pub singular_values: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Result of [`Tensor::eigh_trunc`]: `t ~ v * d * v^H` with the truncated
/// bond; `error` is the quantum-dimension-weighted 2-norm of the discarded
/// eigenvalues.
#[derive(Clone, Debug)]
pub struct EighTrunc {
    pub d: Tensor,
    pub v: Tensor,
    pub eigenvalues: Vec<SectorSpectrum>,
    pub error: f64,
}

/// Result of [`Tensor::eig_trunc`]: `t ~ v * d * v^-1` with the truncated
/// bond. `d` and `v` are always c64 (the general eigendecomposition is
/// complex-valued even for real input); `error` is the
/// quantum-dimension-weighted 2-norm of the discarded `|eigenvalues|`.
#[derive(Clone, Debug)]
pub struct EigTrunc {
    pub d: Tensor,
    pub v: Tensor,
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
    pub error: f64,
}

/// How a freshly built tensor is filled.
enum Fill<'f, D> {
    Zeros,
    Rand(u64),
    BlockFn(&'f mut dyn FnMut(&BlockKey, &[usize]) -> D),
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

/// Builds the coupled-layout dynamic fusion space for the given hom space.
/// The hom-space legs carry the per-sector degeneracies, so the per-tree
/// degeneracy shapes are derived directly from them.
fn build_space<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    rule: &R,
    hom: FusionTreeHomSpace,
) -> Result<DynamicFusionMapSpace, Error> {
    let leg_deg = |leg: &tenet_core::SectorLeg, sector: SectorId| -> Result<usize, Error> {
        leg.degeneracy(sector).ok_or_else(|| {
            Error::InvalidArgument(format!("sector {sector:?} not present on this leg"))
        })
    };
    let keys = hom.fusion_tree_keys(rule);
    let mut shapes = Vec::with_capacity(keys.len());
    for key in &keys {
        let mut shape = Vec::with_capacity(hom.rank());
        for (leg, &sector) in hom.codomain().legs().iter().zip(key.codomain_uncoupled()) {
            shape.push(leg_deg(leg, sector)?);
        }
        for (leg, &sector) in hom.domain().legs().iter().zip(key.domain_uncoupled()) {
            shape.push(leg_deg(leg, sector)?);
        }
        shapes.push(shape);
    }
    DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, shapes).map_err(Into::into)
}

/// Fills every symmetry-allowed block element via `fill(key, indices)`,
/// mirroring [`tenet_core::TensorMap::from_block_fn_with_fusion_space`]
/// (degeneracy coordinates local to the block, codomain axes first, first
/// axis fastest).
fn fill_block_elements<D: UserScalar>(
    structure: &BlockStructure,
    data: &mut [D],
    fill: &mut dyn FnMut(&BlockKey, &[usize]) -> D,
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
/// blocks: `sum_c dim(c) * <a_c, b_c>` with the first argument conjugated,
/// matching TensorKit's `dot` (which conjugates its first argument). Real
/// tensors produce an exactly-real result.
fn weighted_inner<R, D>(
    rule: &R,
    structure: &BlockStructure,
    a: &[D],
    b: &[D],
) -> Result<Complex64, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: UserScalar,
{
    let mut total = Complex64::new(0.0, 0.0);
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
        let mut partial = D::from_real(0.0);
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            partial = partial + FactorScalar::adjoint(a[position]) * b[position];
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
        total += partial.widen_complex() * weight;
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
/// Scalar type: each tensor stores either real `f64` or complex
/// [`Complex64`] data, fixed at construction ([`Self::rand`] vs
/// [`Self::rand_c64`] and so on) and reported by [`Self::dtype`]. Operations
/// dispatch on the stored dtype internally; mixing dtypes in one operation
/// is [`Error::DtypeMismatch`] (widen explicitly with [`Self::to_c64`]).
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
    data: Data,
}

impl Tensor {
    fn build<'a, C, D, S>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        fill: Fill<'_, S>,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
        S: UserScalar,
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
        let (space, data) = with_rule!(rule_kind, rule, {
            let space = build_space(rule, hom)?;
            let len = space.required_len()?;
            let mut data = vec![S::from_real(0.0); len];
            match fill {
                Fill::Zeros => {}
                Fill::Rand(seed) => {
                    let mut state = seed;
                    for value in &mut data {
                        *value = S::rand_unit(&mut state);
                    }
                }
                Fill::BlockFn(fill) => {
                    fill_block_elements(space.structure(), &mut data, fill)?;
                }
            }
            Ok::<_, Error>((space, S::lift(data)))
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
        Self::build::<_, _, f64>(rt, codomain, domain, Fill::Zeros)
    }

    /// Complex (c64) zero tensor on `codomain <- domain`.
    pub fn zeros_c64<'a, C, D>(rt: &Runtime, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::build::<_, _, Complex64>(rt, codomain, domain, Fill::Zeros)
    }

    /// Random tensor on `codomain <- domain`, entries uniform in `[-1, 1)`.
    ///
    /// Deterministic per runtime: the n-th `rand`-family call on a given
    /// runtime always produces the same tensor. Use [`Self::rand_with_seed`]
    /// for an explicit stream.
    pub fn rand<'a, C, D>(rt: &Runtime, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::build::<_, _, f64>(rt, codomain, domain, Fill::Rand(rt.next_rand_seed()))
    }

    /// Complex (c64) random tensor: real and imaginary parts each uniform in
    /// `[-1, 1)`; same determinism as [`Self::rand`].
    pub fn rand_c64<'a, C, D>(rt: &Runtime, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::build::<_, _, Complex64>(rt, codomain, domain, Fill::Rand(rt.next_rand_seed()))
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
        Self::build::<_, _, f64>(rt, codomain, domain, Fill::Rand(seed))
    }

    /// Complex (c64) random tensor with an explicit seed.
    pub fn rand_with_seed_c64<'a, C, D>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        seed: u64,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::build::<_, _, Complex64>(rt, codomain, domain, Fill::Rand(seed))
    }

    /// Tensor filled block-by-block: `fill(key, indices)` is called for
    /// every element of every symmetry-allowed block, with `indices` local
    /// to the block (degeneracy coordinates, codomain axes first). Mirrors
    /// [`tenet_core::TensorMap::from_block_fn_with_fusion_space`].
    ///
    /// The fusion-tree `key` labels domain legs with the domain `Space`'s
    /// own sectors (TensorKit's `f2.uncoupled`), not their duals; on both
    /// sides the uncoupled sectors fuse to the tree's coupled sector.
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

    /// Complex (c64) [`Self::from_block_fn`]: `fill` returns [`Complex64`].
    pub fn from_block_fn_c64<'a, C, D, F>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        mut fill: F,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
        F: FnMut(&BlockKey, &[usize]) -> Complex64,
    {
        Self::build(rt, codomain, domain, Fill::BlockFn(&mut fill))
    }

    /// Wraps a same-runtime, same-rule result of an expert-layer call.
    fn with(&self, space: DynamicFusionMapSpace, data: Data) -> Self {
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data,
        }
    }

    /// The scalar type this tensor stores.
    pub fn dtype(&self) -> Dtype {
        match self.data {
            Data::F64(_) => Dtype::F64,
            Data::C64(_) => Dtype::C64,
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Dtype::F64,
        }
    }

    /// Where this tensor's data lives: [`Placement::Host`] or
    /// [`Placement::Cuda`] with the device ordinal. Transfers are always
    /// explicit (`to_cuda()` / `to_host()`).
    pub fn placement(&self) -> Placement {
        match &self.data {
            Data::F64(_) | Data::C64(_) => Placement::Host,
            #[cfg(feature = "cuda")]
            Data::CudaF64(storage) => storage.placement(),
        }
    }

    /// Uploads an f64 host tensor to the runtime's CUDA device (built with
    /// `Runtime::builder().cuda(device)`); a cheap clone when already
    /// device-resident. Explicit errors: c64 tensors (no device c64 storage
    /// yet) and runtimes built without a CUDA device.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Result<Self, Error> {
        let data = match &self.data {
            Data::CudaF64(storage) => Data::CudaF64(Arc::clone(storage)),
            Data::C64(_) => {
                return Err(device_unsupported("uploading a c64 tensor"));
            }
            Data::F64(host) => {
                let mut state = self.rt.lock();
                let cuda = state.cuda.as_mut().ok_or_else(|| {
                    Error::InvalidArgument(
                        "this runtime was built without a CUDA device; use \
                         Runtime::builder().cuda(device)"
                            .to_string(),
                    )
                })?;
                Data::CudaF64(Arc::new(CudaStorage::upload(cuda, host)?))
            }
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data,
        })
    }

    /// Downloads a device tensor back to host storage; a plain copy when
    /// already host-resident.
    #[cfg(feature = "cuda")]
    pub fn to_host(&self) -> Result<Self, Error> {
        let data = match &self.data {
            Data::F64(_) | Data::C64(_) => self.data.clone(),
            Data::CudaF64(storage) => {
                let mut state = self.rt.lock();
                let cuda = state.cuda.as_mut().ok_or_else(|| {
                    Error::InvalidArgument(
                        "this runtime was built without a CUDA device".to_string(),
                    )
                })?;
                Data::F64(storage.download(cuda)?)
            }
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data,
        })
    }

    /// The [`Runtime`] this tensor was created from (a shared handle).
    pub fn runtime(&self) -> &Runtime {
        &self.rt
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

    /// Flat `f64` storage in the TensorKit-equivalent coupled-sector matrix
    /// layout (column-major inside each coupled block).
    ///
    /// # Panics
    ///
    /// Panics if the tensor stores c64 data; use [`Self::data_c64`] then.
    pub fn data(&self) -> &[f64] {
        match &self.data {
            Data::F64(data) => data,
            Data::C64(_) => panic!("data(): tensor stores c64 data; use data_c64()"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => panic!("data(): tensor is device-resident; use to_host()"),
        }
    }

    /// Flat [`Complex64`] storage in the coupled-sector matrix layout.
    ///
    /// # Panics
    ///
    /// Panics if the tensor stores f64 data; use [`Self::data`] then.
    pub fn data_c64(&self) -> &[Complex64] {
        match &self.data {
            Data::C64(data) => data,
            Data::F64(_) => panic!("data_c64(): tensor stores f64 data; use data()"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => panic!("data_c64(): tensor is device-resident; use to_host()"),
        }
    }

    /// Widens to a c64 tensor (imaginary parts zero); a cheap clone when the
    /// tensor already stores c64 data.
    pub fn to_c64(&self) -> Self {
        let data = match &self.data {
            Data::F64(data) => Data::C64(
                data.iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            Data::C64(data) => Data::C64(data.clone()),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => panic!("to_c64(): tensor is device-resident; use to_host()"),
        };
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data,
        }
    }

    /// Quantum-dimension-weighted total dimension of every leg, in flat
    /// order (codomain legs first, then domain legs). This is the same
    /// notion as [`crate::prelude::Space::dim`] per leg; contraction
    /// planners use it as a size/FLOP proxy.
    pub fn leg_dims(&self) -> Result<Vec<usize>, Error> {
        let hom = self.space.homspace();
        with_rule!(self.rule, rule, {
            Ok(hom
                .codomain()
                .legs()
                .iter()
                .chain(hom.domain().legs())
                .map(|leg| {
                    leg.iter()
                        .map(|(sector, deg)| {
                            (deg as f64 * rule.dim_scalar(sector)).round() as usize
                        })
                        .sum()
                })
                .collect())
        })
    }

    /// The user-facing [`Space`] of flat leg `axis`, following TensorKit's
    /// `space(t, i)` convention: `codomain[i]` for `i < codomain_rank()`,
    /// `dual(domain[i - codomain_rank()])` otherwise.
    pub fn space(&self, axis: usize) -> Result<Space, Error> {
        let hom = self.space.homspace();
        let nout = hom.codomain().len();
        if axis < nout {
            Ok(Space::from_leg(self.rule, &hom.codomain().legs()[axis]))
        } else if axis < hom.rank() {
            Ok(Space::from_leg(self.rule, &hom.domain().legs()[axis - nout]).dual())
        } else {
            Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                hom.rank()
            )))
        }
    }

    /// The codomain spaces, in leg order.
    pub fn codomain_spaces(&self) -> Vec<Space> {
        let hom = self.space.homspace();
        hom.codomain()
            .legs()
            .iter()
            .map(|leg| Space::from_leg(self.rule, leg))
            .collect()
    }

    /// The domain spaces, in leg order (the spaces as written, i.e. *not*
    /// dualized; `t.space(codomain_rank() + i)` is their dual).
    pub fn domain_spaces(&self) -> Vec<Space> {
        let hom = self.space.homspace();
        hom.domain()
            .legs()
            .iter()
            .map(|leg| Space::from_leg(self.rule, leg))
            .collect()
    }

    fn check_rank0(&self) -> Result<(), Error> {
        if self.rank() != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar() requires a rank-0 tensor, got rank {}",
                self.rank()
            )));
        }
        Ok(())
    }

    /// The single element of a rank-0 (scalar) f64 tensor, e.g. the result
    /// of contracting every leg. Errors on tensors with legs and on c64
    /// tensors (use [`Self::scalar_c64`]).
    pub fn scalar(&self) -> Result<f64, Error> {
        self.check_rank0()?;
        match &self.data {
            Data::F64(data) => Ok(data.iter().sum()),
            Data::C64(_) => Err(Error::DtypeMismatch),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scalar()")),
        }
    }

    /// The single element of a rank-0 (scalar) tensor as [`Complex64`];
    /// works for both dtypes (f64 widens with zero imaginary part).
    pub fn scalar_c64(&self) -> Result<Complex64, Error> {
        self.check_rank0()?;
        match &self.data {
            Data::F64(data) => Ok(Complex64::new(data.iter().sum(), 0.0)),
            Data::C64(data) => Ok(data.iter().sum()),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scalar_c64()")),
        }
    }

    fn check_same_world(&self, other: &Self) -> Result<(), Error> {
        if self.rule != other.rule {
            return Err(Error::RuleMismatch);
        }
        if !self.rt.same_runtime(&other.rt) {
            return Err(Error::RuntimeMismatch);
        }
        if self.placement() != other.placement() {
            return Err(Error::PlacementMismatch);
        }
        if self.dtype() != other.dtype() {
            return Err(Error::DtypeMismatch);
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
        match (&self.data, &rhs.data) {
            (Data::F64(a), Data::F64(b)) => self.contract_impl(rhs, a, b, lhs_axes, rhs_axes),
            (Data::C64(a), Data::C64(b)) => self.contract_impl(rhs, a, b, lhs_axes, rhs_axes),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                self.contract_cuda_impl(rhs, a, b, lhs_axes, rhs_axes)
            }
            _ => Err(Error::DtypeMismatch),
        }
    }

    fn contract_impl<D: UserScalar>(
        &self,
        rhs: &Self,
        lhs_data: &[D],
        rhs_data: &[D],
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        let mut state = self.rt.lock();
        let (space, data) = with_rule_ctx!(self.rule, state, rule, ctxs, {
            let dst_space = DynamicFusionMapSpace::contracted(
                rule,
                &self.space,
                &rhs.space,
                lhs_axes,
                rhs_axes,
            )?;
            let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
            D::ctx_of(ctxs).tensorcontract_fusion_dyn_into(
                rule,
                &dst_space,
                &mut data,
                &self.space,
                lhs_data,
                &rhs.space,
                rhs_data,
                TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes),
                D::from_real(1.0),
                D::from_real(0.0),
            )?;
            Ok::<_, Error>((dst_space, D::lift(data)))
        })?;
        drop(state);
        Ok(self.with(space, data))
    }

    /// Device contraction: same plan compilation and resolution cache as the
    /// host path (spaces are host-side metadata), replayed directly on the
    /// device buffers via one offset GEMM per coupled-sector matrix.
    /// Phase-1 scope: only the canonical fully-direct route (exactly
    /// `contract`'s `alpha = 1`, `beta = 0` semantics); contractions that
    /// resolve to dynamic tree transforms return an explicit error.
    #[cfg(feature = "cuda")]
    fn contract_cuda_impl(
        &self,
        rhs: &Self,
        lhs_data: &CudaStorage,
        rhs_data: &CudaStorage,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = state.cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        let (space, data) = with_rule_ctx!(self.rule, state, rule, ctxs, {
            let dst_space = DynamicFusionMapSpace::contracted(
                rule,
                &self.space,
                &rhs.space,
                lhs_axes,
                rhs_axes,
            )?;
            // ponytail: destination allocated by uploading host zeros; a
            // device-side alloc/memset seam replaces this if upload cost
            // ever matters (the direct route overwrites every element).
            let mut dst = CudaStorage::upload(cuda, &vec![0.0; dst_space.required_len()?])?;
            ctxs.f64.tensorcontract_fusion_dyn_direct_on_storage(
                rule,
                &mut CudaStorageGemm::new(cuda),
                &dst_space,
                &mut dst,
                &self.space,
                lhs_data,
                &rhs.space,
                rhs_data,
                TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes),
            )?;
            Ok::<_, Error>((dst_space, Data::CudaF64(Arc::new(dst))))
        })?;
        drop(guard);
        Ok(self.with(space, data))
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

        with_data!(self, data, self.transformed_impl(data, operation))
    }

    fn transformed_impl<D: UserScalar>(
        &self,
        src_data: &[D],
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        let mut state = self.rt.lock();
        let (space, data) = with_rule_ctx!(self.rule, state, rule, ctxs, {
            let dst_space = self.space.transformed(rule, &operation)?;
            let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
            D::ctx_of(ctxs).tree_context_mut().tree_transform_dyn_into(
                rule,
                operation,
                &Arc::clone(dst_space.structure()),
                self.space.structure(),
                &mut data,
                src_data,
                D::from_real(1.0),
                D::from_real(0.0),
            )?;
            Ok::<_, Error>((dst_space, D::lift(data)))
        })?;
        drop(state);
        Ok(self.with(space, data))
    }

    /// Partial trace over pairs of mutually dual legs (TensorKit
    /// `tensortrace!` / TensorOperations `@tensor a[i, i; j]` semantics):
    /// each `(lhs, rhs)` pair of flat leg indices is traced, the remaining
    /// legs keep their order and codomain/domain sides. Symmetric fusion
    /// rules apply the categorical trace coefficients (quantum-dimension
    /// factors, and twists for fermionic rules: the supertrace).
    pub fn trace_pairs(&self, pairs: &[(usize, usize)]) -> Result<Self, Error> {
        let rank = self.rank();
        let mut seen = vec![false; rank];
        for &(lhs, rhs) in pairs {
            for axis in [lhs, rhs] {
                if axis >= rank || seen[axis] {
                    return Err(Error::InvalidArgument(format!(
                        "invalid trace pair list {pairs:?} for rank {rank} \
                         (axes must be in range and distinct)"
                    )));
                }
                seen[axis] = true;
            }
        }
        let output_axes: Vec<usize> = (0..rank).filter(|&axis| !seen[axis]).collect();
        let dst_codomain_rank = output_axes
            .iter()
            .filter(|&&axis| axis < self.codomain_rank())
            .count();
        let trace_lhs: Vec<usize> = pairs.iter().map(|&(lhs, _)| lhs).collect();
        let trace_rhs: Vec<usize> = pairs.iter().map(|&(_, rhs)| rhs).collect();
        with_data!(self, data, {
            self.trace_pairs_impl(
                data,
                &output_axes,
                dst_codomain_rank,
                &trace_lhs,
                &trace_rhs,
            )
        })
    }

    fn trace_pairs_impl<D: UserScalar>(
        &self,
        src_data: &[D],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        trace_lhs: &[usize],
        trace_rhs: &[usize],
    ) -> Result<Self, Error> {
        let (space, data) = with_rule!(self.rule, rule, {
            let hom = self.space.homspace().select(
                rule,
                &output_axes[..dst_codomain_rank],
                &output_axes[dst_codomain_rank..],
            )?;
            let dst_space = build_space(rule, hom)?;
            let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
            tenet_tensors::tensortrace_fusion_dyn_into(
                rule,
                &dst_space,
                &mut data,
                &self.space,
                src_data,
                tenet_tensors::TensorTraceAxisSpec::new(output_axes, trace_lhs, trace_rhs),
                D::from_real(1.0),
                D::from_real(0.0),
            )?;
            Ok::<_, Error>((dst_space, D::lift(data)))
        })?;
        Ok(self.with(space, data))
    }

    /// TensorKit `tr`: full trace of an endomorphism (`domain == codomain`)
    /// to a scalar, pairing codomain leg `i` with domain leg `i`. Returned
    /// as [`Complex64`] for both dtypes (f64 tensors give an exactly-real
    /// result). Fermionic rules give the supertrace, matching TensorKit.
    pub fn tr(&self) -> Result<Complex64, Error> {
        let hom = self.space.homspace();
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::InvalidArgument(
                "tr() requires an endomorphism (domain == codomain)".to_string(),
            ));
        }
        let nout = self.codomain_rank();
        let pairs: Vec<(usize, usize)> = (0..nout).map(|i| (i, nout + i)).collect();
        self.trace_pairs(&pairs)?.scalar_c64()
    }

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block (real scalars: transpose only, c64:
    /// entries conjugated).
    pub fn adjoint(&self) -> Result<Self, Error> {
        with_data!(self, data, {
            let (space, out) = with_rule!(self.rule, rule, {
                tenet_tensors::adjoint_dyn(rule, &self.space, data)
            })?;
            Ok(self.with(space, UserScalar::lift(out)))
        })
    }

    /// Frobenius norm, weighted by coupled-sector quantum dimensions
    /// (`norm(t)^2 = sum_c dim(c) * |block_c|^2`), matching TensorKit's
    /// `norm`. Always real, for both dtypes.
    pub fn norm(&self) -> Result<f64, Error> {
        let value = with_data!(self, data, {
            with_rule!(self.rule, rule, {
                weighted_inner(rule, self.space.structure(), data, data)
            })
        })?;
        Ok(value.re.sqrt())
    }

    /// Returns `factor * self` (real factor, both dtypes). Use
    /// [`Self::scale_c64`] for a complex factor.
    pub fn scale(&self, factor: f64) -> Result<Self, Error> {
        let data = match &self.data {
            Data::F64(data) => Data::F64(data.iter().map(|&value| value * factor).collect()),
            Data::C64(data) => Data::C64(data.iter().map(|&value| value * factor).collect()),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("scale()")),
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data,
        })
    }

    /// Returns `factor * self` for a c64 tensor. Errors with
    /// [`Error::DtypeMismatch`] on f64 tensors (widen with
    /// [`Self::to_c64`] first).
    pub fn scale_c64(&self, factor: Complex64) -> Result<Self, Error> {
        match &self.data {
            Data::C64(data) => Ok(Self {
                rt: self.rt.clone(),
                rule: self.rule,
                space: Arc::clone(&self.space),
                data: Data::C64(data.iter().map(|&value| value * factor).collect()),
            }),
            Data::F64(_) => Err(Error::DtypeMismatch),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scale_c64()")),
        }
    }

    /// Returns `alpha * self + beta * other` (real coefficients, both
    /// dtypes). Both tensors must live on the same spaces (identical hom
    /// space and block layout) and store the same dtype.
    pub fn add(&self, other: &Self, alpha: f64, beta: f64) -> Result<Self, Error> {
        self.check_same_space(other)?;
        let data = match (&self.data, &other.data) {
            (Data::F64(a), Data::F64(b)) => Data::F64(
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| alpha * x + beta * y)
                    .collect(),
            ),
            (Data::C64(a), Data::C64(b)) => Data::C64(
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| x * alpha + y * beta)
                    .collect(),
            ),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(_), _) | (_, Data::CudaF64(_)) => {
                return Err(device_unsupported("add()"))
            }
            _ => return Err(Error::DtypeMismatch),
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data,
        })
    }

    /// Returns `alpha * self + beta * other` with complex coefficients; both
    /// tensors must be c64 (widen with [`Self::to_c64`] first).
    pub fn add_c64(&self, other: &Self, alpha: Complex64, beta: Complex64) -> Result<Self, Error> {
        self.check_same_space(other)?;
        match (&self.data, &other.data) {
            (Data::C64(a), Data::C64(b)) => Ok(Self {
                rt: self.rt.clone(),
                rule: self.rule,
                space: Arc::clone(&self.space),
                data: Data::C64(
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| alpha * x + beta * y)
                        .collect(),
                ),
            }),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(_), _) | (_, Data::CudaF64(_)) => Err(device_unsupported("add_c64()")),
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Frobenius inner product `<self, other>` with `self` conjugated,
    /// weighted by coupled-sector quantum dimensions, matching TensorKit's
    /// `dot(x, y)`. Always returned as [`Complex64`]: real tensors give an
    /// exactly-zero imaginary part, so `t.inner(&t)?.re == t.norm()?.powi(2)`
    /// up to floating-point error. Both tensors must live on the same spaces
    /// and store the same dtype.
    pub fn inner(&self, other: &Self) -> Result<Complex64, Error> {
        self.check_same_space(other)?;
        match (&self.data, &other.data) {
            (Data::F64(a), Data::F64(b)) => with_rule!(self.rule, rule, {
                weighted_inner(rule, self.space.structure(), a, b)
            }),
            (Data::C64(a), Data::C64(b)) => with_rule!(self.rule, rule, {
                weighted_inner(rule, self.space.structure(), a, b)
            }),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(_), _) | (_, Data::CudaF64(_)) => Err(device_unsupported("inner()")),
            _ => Err(Error::DtypeMismatch),
        }
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

    // -----------------------------------------------------------------------
    // Decompositions and matrix functions (TensorKit 0.17 / MatrixAlgebraKit
    // names, transparently over the tenet-matrixalgebra dynamic cores).
    // -----------------------------------------------------------------------

    /// Wraps a dynamic factor produced by the matrixalgebra layer.
    fn from_dyn<D: UserScalar>(&self, (space, data): DynFactor<D>) -> Self {
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data: D::lift(data),
        }
    }

    /// Compact SVD `t = u * s * vh` (MatrixAlgebraKit `svd_compact`):
    /// per coupled sector the bond is `min(rows, cols)`.
    pub fn svd_compact(&self) -> Result<(Self, Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::svd_compact_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((
                self.from_dyn(out.u),
                self.from_dyn(out.s),
                self.from_dyn(out.vh),
            ))
        })
    }

    /// Full SVD `t = u * s * vh` (MatrixAlgebraKit `svd_full`): square
    /// unitaries per sector, rectangular diagonal `s`.
    pub fn svd_full(&self) -> Result<(Self, Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::svd_full_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((
                self.from_dyn(out.u),
                self.from_dyn(out.s),
                self.from_dyn(out.vh),
            ))
        })
    }

    /// Truncated SVD (MatrixAlgebraKit `svd_trunc`); see [`SvdTrunc`].
    pub fn svd_trunc(&self, truncation: &Truncation) -> Result<SvdTrunc, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::svd_trunc_dyn(
                    &mut state.dense,
                    rule,
                    &self.space,
                    data,
                    truncation,
                )
            })?;
            Ok(SvdTrunc {
                u: self.from_dyn(out.u),
                s: self.from_dyn(out.s),
                vh: self.from_dyn(out.vh),
                singular_values: out.singular_values,
                error: out.error,
            })
        })
    }

    /// All singular values per coupled sector, descending (MatrixAlgebraKit
    /// `svd_vals`). Real for both dtypes.
    pub fn svd_vals(&self) -> Result<Vec<SectorSpectrum>, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            with_rule!(self.rule, rule, {
                tenet_matrixalgebra::svd_vals_dyn(&mut state.dense, rule, &self.space, data)
            })
            .map_err(Into::into)
        })
    }

    /// Compact QR `t = q * r` (MatrixAlgebraKit `qr_compact`): `q` has
    /// orthonormal columns per coupled sector.
    pub fn qr_compact(&self) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let (q, r) = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::qr_compact_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((self.from_dyn(q), self.from_dyn(r)))
        })
    }

    /// Full QR `t = q * r` (MatrixAlgebraKit `qr_full`): square `q` per
    /// sector.
    pub fn qr_full(&self) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let (q, r) = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::qr_full_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((self.from_dyn(q), self.from_dyn(r)))
        })
    }

    /// Compact LQ `t = l * q` (MatrixAlgebraKit `lq_compact`): `q` has
    /// orthonormal rows per coupled sector.
    pub fn lq_compact(&self) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let (l, q) = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::lq_compact_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((self.from_dyn(l), self.from_dyn(q)))
        })
    }

    /// Full LQ `t = l * q` (MatrixAlgebraKit `lq_full`): square `q` per
    /// sector.
    pub fn lq_full(&self) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let (l, q) = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::lq_full_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((self.from_dyn(l), self.from_dyn(q)))
        })
    }

    /// Left isometry factorization `t = v * c` (TensorKit 0.17 `left_orth`,
    /// default QR kind): `v` isometric, `c` the corestriction.
    pub fn left_orth(&self) -> Result<(Self, Self), Error> {
        self.qr_compact()
    }

    /// Right isometry factorization `t = c * vh` (TensorKit 0.17
    /// `right_orth`, default LQ kind): `vh` has orthonormal rows.
    pub fn right_orth(&self) -> Result<(Self, Self), Error> {
        self.lq_compact()
    }

    /// Left null space `n : codomain <- W` with `n^H * t = 0` (MatrixAlgebraKit
    /// `left_null`).
    pub fn left_null(&self) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::left_null_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok(self.from_dyn(out))
        })
    }

    /// Right null space `n : W <- domain` with `t * n^H = 0` (MatrixAlgebraKit
    /// `right_null`).
    pub fn right_null(&self) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::right_null_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok(self.from_dyn(out))
        })
    }

    /// Left polar decomposition `t = w * p` (MatrixAlgebraKit `left_polar`):
    /// `w` isometric, `p` positive on the domain.
    pub fn left_polar(&self) -> Result<(Self, Self), Error> {
        with_data!(self, data, self.left_polar_impl(data))
    }

    fn left_polar_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let (w, p) = with_rule_ctx!(self.rule, state, rule, ctxs, {
            tenet_matrixalgebra::left_polar_dyn(
                &mut state.dense,
                D::ctx_of(ctxs),
                rule,
                &self.space,
                data,
            )
        })?;
        Ok((self.from_dyn(w), self.from_dyn(p)))
    }

    /// Right polar decomposition `t = p * w` (MatrixAlgebraKit
    /// `right_polar`): `p` positive on the codomain, `w` isometric.
    pub fn right_polar(&self) -> Result<(Self, Self), Error> {
        with_data!(self, data, self.right_polar_impl(data))
    }

    fn right_polar_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let (p, w) = with_rule_ctx!(self.rule, state, rule, ctxs, {
            tenet_matrixalgebra::right_polar_dyn(
                &mut state.dense,
                D::ctx_of(ctxs),
                rule,
                &self.space,
                data,
            )
        })?;
        Ok((self.from_dyn(p), self.from_dyn(w)))
    }

    /// Full Hermitian eigendecomposition `t = v * d * v^H` (MatrixAlgebraKit
    /// `eigh_full`), returned as `(d, v)`. Requires an endomorphism with
    /// Hermitian coupled blocks. The eigenvalues are real for both dtypes
    /// (TensorKit: real `D`); `d` keeps the input dtype so it composes with
    /// `v` directly.
    pub fn eigh_full(&self) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eigh_full_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((self.from_dyn(out.d), self.from_dyn(out.v)))
        })
    }

    /// Truncated Hermitian eigendecomposition (MatrixAlgebraKit
    /// `eigh_trunc`); see [`EighTrunc`].
    pub fn eigh_trunc(&self, truncation: &Truncation) -> Result<EighTrunc, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eigh_trunc_dyn(
                    &mut state.dense,
                    rule,
                    &self.space,
                    data,
                    truncation,
                )
            })?;
            Ok(EighTrunc {
                d: self.from_dyn(out.d),
                v: self.from_dyn(out.v),
                eigenvalues: out.eigenvalues,
                error: out.error,
            })
        })
    }

    /// All Hermitian eigenvalues per coupled sector, descending by magnitude
    /// (MatrixAlgebraKit `eigh_vals`). Real for both dtypes.
    pub fn eigh_vals(&self) -> Result<Vec<SectorSpectrum>, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eigh_vals_dyn(&mut state.dense, rule, &self.space, data)
            })
            .map_err(Into::into)
        })
    }

    /// Full general (non-Hermitian) eigendecomposition `t = v * d * v^-1`
    /// (MatrixAlgebraKit `eig_full`), returned as `(d, v)`. Requires an
    /// endomorphism. The output tensors are always c64, even for f64 input
    /// (real matrices have complex eigenpairs), matching TensorKit's
    /// `eigen`, whose `D` and `V` are `ComplexF64` for real input.
    pub fn eig_full(&self) -> Result<(Self, Self), Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eig_full_dyn(&mut state.dense, rule, &self.space, data)
            })?;
            Ok((self.from_dyn(out.d), self.from_dyn(out.v)))
        })
    }

    /// Truncated general eigendecomposition (MatrixAlgebraKit `eig_trunc`,
    /// kept by descending `|eigenvalue|`); see [`EigTrunc`]. Output tensors
    /// are always c64.
    pub fn eig_trunc(&self, truncation: &Truncation) -> Result<EigTrunc, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eig_trunc_dyn(
                    &mut state.dense,
                    rule,
                    &self.space,
                    data,
                    truncation,
                )
            })?;
            Ok(EigTrunc {
                d: self.from_dyn(out.d),
                v: self.from_dyn(out.v),
                eigenvalues: out.eigenvalues,
                error: out.error,
            })
        })
    }

    /// All general eigenvalues per coupled sector, descending by magnitude
    /// (MatrixAlgebraKit `eig_vals`). Complex for both dtypes.
    pub fn eig_vals(&self) -> Result<Vec<SectorSpectrum<Complex64>>, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eig_vals_dyn(&mut state.dense, rule, &self.space, data)
            })
            .map_err(Into::into)
        })
    }

    /// Matrix exponential of a Hermitian endomorphism, `exp(t) = v exp(d)
    /// v^H` (TensorKit `exp`, via the eigendecomposition).
    pub fn exp(&self) -> Result<Self, Error> {
        with_data!(self, data, self.exp_impl(data))
    }

    fn exp_impl<D: UserScalar>(&self, data: &[D]) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let out = with_rule_ctx!(self.rule, state, rule, ctxs, {
            tenet_matrixalgebra::exp_dyn(&mut state.dense, D::ctx_of(ctxs), rule, &self.space, data)
        })?;
        Ok(self.from_dyn(out))
    }

    /// True inverse of a full-rank endomorphism (MatrixAlgebraKit-style
    /// `inv`); fails when any coupled block is rank-deficient at working
    /// precision.
    pub fn inv(&self) -> Result<Self, Error> {
        with_data!(self, data, self.inv_impl(data))
    }

    fn inv_impl<D: UserScalar>(&self, data: &[D]) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let out = with_rule_ctx!(self.rule, state, rule, ctxs, {
            tenet_matrixalgebra::inv_dyn(&mut state.dense, D::ctx_of(ctxs), rule, &self.space, data)
        })?;
        Ok(self.from_dyn(out))
    }

    /// Moore-Penrose pseudo-inverse `t^+ = v s^+ u^H` (MatrixAlgebraKit
    /// `pinv`) with an `rcond * sigma_max` cutoff on the singular values.
    pub fn pinv(&self, rcond: f64) -> Result<Self, Error> {
        with_data!(self, data, self.pinv_impl(data, rcond))
    }

    fn pinv_impl<D: UserScalar>(&self, data: &[D], rcond: f64) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let out = with_rule_ctx!(self.rule, state, rule, ctxs, {
            tenet_matrixalgebra::pinv_dyn(
                &mut state.dense,
                D::ctx_of(ctxs),
                rule,
                &self.space,
                data,
                rcond,
            )
        })?;
        Ok(self.from_dyn(out))
    }
}
