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
use std::sync::{Arc, OnceLock};

use num_complex::Complex64;
use tenet_core::{
    BlockKey, BlockStructure, FusionProductSpace, FusionRule, FusionTreeHomSpace, FusionTreeKey,
    MultiplicityFreeRigidSymbols, Placement, SectorId, Su3FusionRule,
};
#[cfg(feature = "cuda")]
use tenet_core::{SectorLeg, TensorStorage};
#[cfg(feature = "cuda")]
use tenet_dense::{
    cuda_eigh_region, cuda_gemm_region_into, cuda_qr_region, cuda_svd_region, CudaDenseContext,
    CudaDenseStorage,
};
#[cfg(feature = "cuda")]
use tenet_matrixalgebra::{select_truncation, WeightedSpectrum};
use tenet_matrixalgebra::{DynFactor, FactorScalar, SectorSpectrum, Truncation};
#[cfg(feature = "cuda")]
use tenet_tensors::cuda::{CudaStorage, CudaStorageGemm};
#[cfg(feature = "cuda")]
use tenet_tensors::OperationError;
use tenet_tensors::{
    DynamicFusionMapSpace, OutputAxisOrder, RecouplingCoefficientAction, TensorContractSpec,
    TreeTransformOperation, TreeTransformRuleCacheKey,
};

use crate::error::Error;
use crate::runtime::{
    with_rule_ctx, BuiltinKey, Ctx, Ctxs, ProductKey, Runtime, Su3Key, TripleKey,
};
use crate::space::{with_rule, RuleKind, Space};

/// The scalar type a [`Tensor`] stores, fixed at construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Dtype {
    /// Real double precision (`f64`).
    F64,
    /// Complex double precision ([`Complex64`]).
    C64,
}

/// A scalar produced by a [`Tensor`] reduction ([`Tensor::scalar`],
/// [`Tensor::inner`], [`Tensor::tr`]): the variant matches the producing
/// tensor's [`Dtype`], mirroring TensorKit, where `dot`/`tr` on a real
/// tensor return a real scalar. Non-exhaustive so future precisions
/// (f32/c32) can add variants.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// Real double precision.
    F64(f64),
    /// Complex double precision.
    C64(Complex64),
}

impl Scalar {
    /// The real part (the value itself for real variants).
    pub fn re(self) -> f64 {
        match self {
            Self::F64(value) => value,
            Self::C64(value) => value.re,
        }
    }

    /// The imaginary part (exactly `0.0` for real variants).
    pub fn im(self) -> f64 {
        match self {
            Self::F64(_) => 0.0,
            Self::C64(value) => value.im,
        }
    }

    /// The value as `f64`; [`Error::DtypeMismatch`] on complex variants.
    /// Use [`Self::re`] when you deliberately want the real part of a
    /// complex scalar.
    pub fn try_f64(self) -> Result<f64, Error> {
        match self {
            Self::F64(value) => Ok(value),
            Self::C64(_) => Err(Error::DtypeMismatch),
        }
    }

    /// Widens to [`Complex64`] (exact for every variant).
    pub fn to_c64(self) -> Complex64 {
        match self {
            Self::F64(value) => Complex64::new(value, 0.0),
            Self::C64(value) => value,
        }
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64(value) => write!(f, "{value}"),
            Self::C64(value) => write!(f, "{value}"),
        }
    }
}

/// Dtype-erased flat storage in the coupled-sector matrix layout. The
/// device variant shares the immutable buffer behind an `Arc` (operations
/// always write fresh destinations), keeping `Tensor: Clone` cheap and the
/// host paths untouched.
#[derive(Clone, Debug)]
pub enum Data {
    F64(Vec<f64>),
    C64(Vec<Complex64>),
    /// Compact O(rank) diagonal storage for spectrum tensors (SVD `s`, eigh/eig
    /// `d`): only the per-sector diagonal values are held, not the dense
    /// block-diagonal matrix (issue #55). Materialized into the equivalent dense
    /// `F64`/`C64` on demand by [`Tensor::coupled_data`], so every existing value
    /// consumer is unaffected. Never surfaces *from* `coupled_data`.
    Diagonal(DiagonalData),
    #[cfg(feature = "cuda")]
    CudaF64(Arc<CudaStorage>),
}

/// The values of a [`Data::Diagonal`] tensor plus how they materialize into a
/// dense buffer, chosen to reproduce the former dense diagonal bit-for-bit:
/// SVD singular values / Hermitian eigenvalues are real (`RealF64`, or `RealC64`
/// when the source tensor is complex), general eigenvalues are complex (`C64`).
#[derive(Clone, Debug)]
pub enum DiagonalData {
    RealF64(Vec<SectorSpectrum<f64>>),
    RealC64(Vec<SectorSpectrum<f64>>),
    C64(Vec<SectorSpectrum<Complex64>>),
}

impl DiagonalData {
    fn dtype(&self) -> Dtype {
        match self {
            DiagonalData::RealF64(_) => Dtype::F64,
            DiagonalData::RealC64(_) | DiagonalData::C64(_) => Dtype::C64,
        }
    }

    /// Multiplies every stored value by a real factor, preserving the variant —
    /// so scaling a diagonal factor (e.g. itebd's `λ / |λ|`) keeps O(rank)
    /// storage instead of densifying.
    fn scaled(&self, factor: f64) -> DiagonalData {
        fn map_real(spectra: &[SectorSpectrum<f64>], factor: f64) -> Vec<SectorSpectrum<f64>> {
            spectra
                .iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: entry.values.iter().map(|&value| value * factor).collect(),
                })
                .collect()
        }
        match self {
            DiagonalData::RealF64(spectra) => DiagonalData::RealF64(map_real(spectra, factor)),
            DiagonalData::RealC64(spectra) => DiagonalData::RealC64(map_real(spectra, factor)),
            DiagonalData::C64(spectra) => DiagonalData::C64(
                spectra
                    .iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: entry.values.iter().map(|&value| value * factor).collect(),
                    })
                    .collect(),
            ),
        }
    }

    /// The largest `|entry|` over all sectors (for `pinv`'s relative cutoff).
    fn max_abs(&self) -> f64 {
        match self {
            DiagonalData::RealF64(s) | DiagonalData::RealC64(s) => s
                .iter()
                .flat_map(|e| e.values.iter())
                .fold(0.0f64, |m, &v| m.max(v.abs())),
            DiagonalData::C64(s) => s
                .iter()
                .flat_map(|e| e.values.iter())
                .fold(0.0f64, |m, &v| m.max(v.norm())),
        }
    }

    /// Elementwise reciprocal — the diagonal `inv` (TensorKit `inv.(d.data)`).
    /// Errors on a zero entry, like the dense `inv` on a rank-deficient block.
    fn try_recip(&self) -> Result<DiagonalData, Error> {
        fn recip_real(s: &[SectorSpectrum<f64>]) -> Result<Vec<SectorSpectrum<f64>>, Error> {
            s.iter()
                .map(|e| {
                    Ok(SectorSpectrum {
                        sector: e.sector,
                        values: e
                            .values
                            .iter()
                            .map(|&v| {
                                if v == 0.0 {
                                    Err(Error::InvalidArgument(
                                        "inv of a singular diagonal (zero entry)".to_string(),
                                    ))
                                } else {
                                    Ok(1.0 / v)
                                }
                            })
                            .collect::<Result<_, Error>>()?,
                    })
                })
                .collect()
        }
        Ok(match self {
            DiagonalData::RealF64(s) => DiagonalData::RealF64(recip_real(s)?),
            DiagonalData::RealC64(s) => DiagonalData::RealC64(recip_real(s)?),
            DiagonalData::C64(s) => DiagonalData::C64(
                s.iter()
                    .map(|e| {
                        Ok(SectorSpectrum {
                            sector: e.sector,
                            values: e
                                .values
                                .iter()
                                .map(|&v| {
                                    if v == Complex64::new(0.0, 0.0) {
                                        Err(Error::InvalidArgument(
                                            "inv of a singular diagonal (zero entry)".to_string(),
                                        ))
                                    } else {
                                        Ok(Complex64::new(1.0, 0.0) / v)
                                    }
                                })
                                .collect::<Result<_, Error>>()?,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
            ),
        })
    }

    /// Elementwise pseudo-inverse with an `rcond * max|entry|` cutoff (TensorKit
    /// `pinv` on a diagonal): entries at or below the cutoff map to 0, the rest
    /// to `1/entry`. Same variant (`1/entry` of a real entry stays real).
    fn pinv(&self, rcond: f64) -> DiagonalData {
        let cutoff = rcond * self.max_abs();
        fn map_real(s: &[SectorSpectrum<f64>], cutoff: f64) -> Vec<SectorSpectrum<f64>> {
            s.iter()
                .map(|e| SectorSpectrum {
                    sector: e.sector,
                    values: e
                        .values
                        .iter()
                        .map(|&v| if v.abs() > cutoff { 1.0 / v } else { 0.0 })
                        .collect(),
                })
                .collect()
        }
        match self {
            DiagonalData::RealF64(s) => DiagonalData::RealF64(map_real(s, cutoff)),
            DiagonalData::RealC64(s) => DiagonalData::RealC64(map_real(s, cutoff)),
            DiagonalData::C64(s) => DiagonalData::C64(
                s.iter()
                    .map(|e| SectorSpectrum {
                        sector: e.sector,
                        values: e
                            .values
                            .iter()
                            .map(|&v| {
                                if v.norm() > cutoff {
                                    Complex64::new(1.0, 0.0) / v
                                } else {
                                    Complex64::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    })
                    .collect(),
            ),
        }
    }

    /// Elementwise principal square root (TensorKit `sqrt.(d.data)`). A real
    /// (`RealF64`) diagonal errors on a negative entry (like the dense f64
    /// `sqrt`); a complex-typed real spectrum (`RealC64`) takes the complex root
    /// and promotes to `C64`, matching the dense c64 `sqrt`.
    fn try_sqrt(&self) -> Result<DiagonalData, Error> {
        let map_c64 = |s: &[SectorSpectrum<Complex64>]| -> Vec<SectorSpectrum<Complex64>> {
            s.iter()
                .map(|e| SectorSpectrum {
                    sector: e.sector,
                    values: e.values.iter().map(|&v| v.sqrt()).collect(),
                })
                .collect()
        };
        Ok(match self {
            DiagonalData::RealF64(s) => DiagonalData::RealF64(
                s.iter()
                    .map(|e| {
                        Ok(SectorSpectrum {
                            sector: e.sector,
                            values: e
                                .values
                                .iter()
                                .map(|&v| {
                                    if v < 0.0 {
                                        Err(Error::InvalidArgument(format!(
                                            "sqrt of a negative diagonal entry {v}; convert to \
                                             c64 with to_c64() for the complex square root"
                                        )))
                                    } else {
                                        Ok(v.sqrt())
                                    }
                                })
                                .collect::<Result<_, Error>>()?,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
            ),
            DiagonalData::RealC64(s) => DiagonalData::C64(map_c64(
                &s.iter()
                    .map(|e| SectorSpectrum {
                        sector: e.sector,
                        values: e.values.iter().map(|&v| Complex64::new(v, 0.0)).collect(),
                    })
                    .collect::<Vec<_>>(),
            )),
            DiagonalData::C64(s) => DiagonalData::C64(map_c64(s)),
        })
    }
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

/// The scalar types a [`Tensor`] can store: `f64` and [`Complex64`]. This
/// trait is **sealed** (its supertrait is crate-private); it exists so
/// [`Tensor::from_block_fn`] can infer the constructed dtype from the fill
/// closure's return type.
pub trait TensorScalar: UserScalar {}

impl TensorScalar for f64 {}
impl TensorScalar for Complex64 {}

/// The scalar types the user layer stores: the expert-layer scalar machinery
/// plus the glue to lift typed data into the erased [`Data`] storage and to
/// pick the matching per-scalar execution context. Crate-private supertrait
/// sealing [`TensorScalar`].
pub trait UserScalar: FactorScalar + RecouplingCoefficientAction<f64> {
    fn lift(data: Vec<Self>) -> Data;
    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key>;
    fn rand_unit(state: &mut u64) -> Self;
}

impl UserScalar for f64 {
    fn lift(data: Vec<Self>) -> Data {
        Data::F64(data)
    }

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key> {
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

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key> {
        &mut ctxs.c64
    }

    fn rand_unit(state: &mut u64) -> Self {
        Complex64::new(rand_unit(state), rand_unit(state))
    }
}

/// Caller-owned host execution state for dynamic destination operations.
#[derive(Default)]
pub struct TensorExecutionContext {
    u1: Ctxs<BuiltinKey>,
    z2: Ctxs<BuiltinKey>,
    fz2: Ctxs<BuiltinKey>,
    su2: Ctxs<BuiltinKey>,
    u1_fz2: Ctxs<ProductKey>,
    fz2_u1_su2: Ctxs<TripleKey>,
    su3: Ctxs<Su3Key>,
}

impl TensorExecutionContext {
    /// Builds caller-owned execution state with the same CPU execution
    /// configuration as `runtime`.
    pub fn for_runtime(runtime: &Runtime) -> Result<Self, Error> {
        let config = runtime.execution_config();
        let mut context = Self {
            u1: Ctxs::with_config(config.dense_threads, config.gemm_kind)?,
            z2: Ctxs::with_config(config.dense_threads, config.gemm_kind)?,
            fz2: Ctxs::with_config(config.dense_threads, config.gemm_kind)?,
            su2: Ctxs::with_config(config.dense_threads, config.gemm_kind)?,
            u1_fz2: Ctxs::with_config(config.dense_threads, config.gemm_kind)?,
            fz2_u1_su2: Ctxs::with_config(config.dense_threads, config.gemm_kind)?,
            su3: Ctxs::with_config(config.dense_threads, config.gemm_kind)?,
        };
        if let Some(threads) = config.recoupling_threads {
            context.set_recoupling_threads(threads);
        }
        if let Some(backend) = config.transpose_backend {
            context.set_transpose_backend(backend);
        }
        Ok(context)
    }

    fn set_recoupling_threads(&mut self, threads: usize) {
        macro_rules! apply {
            ($contexts:expr) => {
                $contexts
                    .f64
                    .tree_context_mut()
                    .backend_mut()
                    .set_recoupling_threads(threads);
                $contexts
                    .c64
                    .tree_context_mut()
                    .backend_mut()
                    .set_recoupling_threads(threads);
            };
        }
        apply!(self.u1);
        apply!(self.z2);
        apply!(self.fz2);
        apply!(self.su2);
        apply!(self.u1_fz2);
        apply!(self.fz2_u1_su2);
        apply!(self.su3);
    }

    fn set_transpose_backend(&mut self, backend: tenet_tensors::TransposeBackend) {
        macro_rules! apply {
            ($contexts:expr) => {
                $contexts
                    .f64
                    .tree_context_mut()
                    .backend_mut()
                    .set_transpose_backend(backend);
                $contexts
                    .f64
                    .contract_backend_mut()
                    .set_transpose_backend(backend);
                $contexts
                    .c64
                    .tree_context_mut()
                    .backend_mut()
                    .set_transpose_backend(backend);
                $contexts
                    .c64
                    .contract_backend_mut()
                    .set_transpose_backend(backend);
            };
        }
        apply!(self.u1);
        apply!(self.z2);
        apply!(self.fz2);
        apply!(self.su2);
        apply!(self.u1_fz2);
        apply!(self.fz2_u1_su2);
        apply!(self.su3);
    }
}

/// Dispatches once on the stored dtype of `$tensor`, binding `$data` to the
/// typed data vector in both arms; `$body` must be dtype-generic (the expert
/// entry points are generic over the scalar).
macro_rules! with_data {
    ($tensor:expr, $data:ident, $body:expr) => {
        match $tensor.coupled_data() {
            Data::F64($data) => $body,
            Data::C64($data) => $body,
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
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
    for key in keys.iter() {
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

/// Generic-fusion (SU(3)) sibling of [`build_space`]: builds the
/// multiplicity-aware coupled space. Same body, but enumerates via
/// `fusion_tree_keys_generic` and the Generic `from_degeneracy_shapes`.
fn build_space_generic<R: FusionRule>(
    rule: &R,
    hom: FusionTreeHomSpace,
) -> Result<DynamicFusionMapSpace, Error> {
    let leg_deg = |leg: &tenet_core::SectorLeg, sector: SectorId| -> Result<usize, Error> {
        leg.degeneracy(sector).ok_or_else(|| {
            Error::InvalidArgument(format!("sector {sector:?} not present on this leg"))
        })
    };
    let keys = hom
        .fusion_tree_keys_generic(rule)
        .map_err(tenet_tensors::OperationError::from_core_preserving_context)?;
    let mut shapes = Vec::with_capacity(keys.len());
    for key in keys.iter() {
        let mut shape = Vec::with_capacity(hom.rank());
        for (leg, &sector) in hom.codomain().legs().iter().zip(key.codomain_uncoupled()) {
            shape.push(leg_deg(leg, sector)?);
        }
        for (leg, &sector) in hom.domain().legs().iter().zip(key.domain_uncoupled()) {
            shape.push(leg_deg(leg, sector)?);
        }
        shapes.push(shape);
    }
    DynamicFusionMapSpace::from_degeneracy_shapes_generic(rule, hom, shapes).map_err(Into::into)
}

/// Fills a freshly-built coupled space (rule-agnostic: only touches the block
/// structure). Shared by the mult-free and SU(3) construction paths.
fn apply_fill<S: UserScalar>(
    space: &DynamicFusionMapSpace,
    fill: Fill<'_, S>,
) -> Result<Vec<S>, Error> {
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
    Ok(data)
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

/// Scales every fusion-tree block of `data` in place by the real factor
/// `factor_of(key)` (skipping factor-1 blocks). Backs [`Tensor::twist`] and
/// [`Tensor::flip`], whose effect on the storage is exactly a per-block
/// phase.
fn scale_blocks_impl<D: UserScalar>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    factor_of: &dyn Fn(&BlockKey) -> f64,
) -> Result<(), Error> {
    let structure = space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let factor = factor_of(block.key());
        if factor == 1.0 {
            continue;
        }
        let factor = D::from_real(factor);
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
            data[position] = data[position] * factor;
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
    // Abelian (UniqueFusion) fast path: every `dim(c) == 1`, and the coupled
    // buffer is a padding-free concatenation of the per-sector blocks, so the
    // weighted per-block sum collapses to one whole-buffer conjugated dot —
    // no `dim(c)` weights and no per-element odometer. Mirrors TensorKit's
    // `inner(tx.data, ty.data)` / `norm(t.data)` UniqueFusion specialization
    // (vectorinterface.jl:124, linalg.jl:277). Non-abelian keeps the weighted
    // block loop below, where `dim(c) != 1`.
    if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
        let mut total = D::from_real(0.0);
        for (&ai, &bi) in a.iter().zip(b) {
            total = total + FactorScalar::adjoint(ai) * bi;
        }
        return Ok(total.widen_complex());
    }
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

/// Generic-fusion (Stage B3c-1) sibling of [`weighted_inner`] for an
/// outer-multiplicity rule (SU(N)): `sum_c dim(c) * <a_c, b_c>`. Identical block
/// loop; the only difference is the coupled-sector weight. [`GenericRigidSymbols`]
/// exposes `sqrt_dim` rather than `dim`, and `dim(c) = sqrt_dim(c)^2` (an
/// integer for SU(N)) — so a Frobenius norm sums every multiplicity block
/// (e.g. both SU(3) `N(8,8,8)=2` vertices) weighted by the quantum dimension,
/// exactly matching TensorKit's `norm`. No Unique fast path: Generic is never
/// abelian.
fn weighted_inner_generic<R, D>(
    rule: &R,
    structure: &BlockStructure,
    a: &[D],
    b: &[D],
) -> Result<Complex64, Error>
where
    R: tenet_core::GenericRigidSymbols<Scalar = f64>,
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
                let sqrt = rule.sqrt_dim_scalar(coupled);
                sqrt * sqrt
            }
            _ => 1.0,
        };
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
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

/// Generic-fusion (Stage B3c-2) sibling of [`weighted_trace`] for an
/// outer-multiplicity rule (SU(N)): identical diagonal-block walk; the weight
/// is `dim(c) = sqrt_dim(c)²` and the twist is 1 (SU(N) is bosonic —
/// `GenericRigidSymbols` deliberately carries no twist because no non-bosonic
/// Generic rule ships; the contraction engine guards the same assumption).
/// A vertex-labelled (OM) block contributes only when its codomain and domain
/// trees coincide INCLUDING the vertex labels — off-diagonal vertex pairs are
/// off the coupled-block diagonal exactly like any other tree mismatch.
fn weighted_trace_generic<R, D>(
    rule: &R,
    structure: &BlockStructure,
    nout: usize,
    data: &[D],
) -> Result<Complex64, Error>
where
    R: tenet_core::GenericRigidSymbols<Scalar = f64>,
    D: UserScalar,
{
    let mut total = Complex64::new(0.0, 0.0);
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let key = match block.key() {
            BlockKey::FusionTree(key) => key,
            _ => {
                return Err(Error::InvalidArgument(
                    "tr() requires fusion-tree blocks".to_string(),
                ))
            }
        };
        if key.codomain_tree() != key.domain_tree() {
            continue;
        }
        let coupled = key
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| rule.vacuum());
        let sqrt = rule.sqrt_dim_scalar(coupled);
        let weight = sqrt * sqrt;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let count: usize = shape[..nout].iter().product();
        let mut indices = vec![0usize; nout];
        let mut partial = D::from_real(0.0);
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .enumerate()
                    .map(|(axis, &i)| i * strides[axis] + i * strides[nout + axis])
                    .sum::<usize>();
            partial = partial + data[position];
            for axis in 0..nout {
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

/// Quantum-dimension-weighted block trace of an endomorphism:
/// `sum_c dim(c) * tr(b_c)`, matching TensorKit's `tr` (`linalg.jl`, the
/// native `sum_c dim(c) * tr(block)`). Only fusion-tree blocks whose codomain
/// and domain trees coincide sit on the coupled-block diagonal; every other
/// block is off-diagonal in `b_c` and contributes nothing. Within a diagonal
/// block the trace pairs codomain degeneracy axis `i` with domain axis
/// `nout + i` (equal degeneracies, since the spaces match). Real tensors give
/// an exactly-real result. Fermionic rules fold their supertrace sign into the
/// coupled blocks, so the same sum yields the supertrace.
fn weighted_trace<R, D>(
    rule: &R,
    structure: &BlockStructure,
    nout: usize,
    data: &[D],
) -> Result<Complex64, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: UserScalar,
{
    let mut total = Complex64::new(0.0, 0.0);
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let key = match block.key() {
            BlockKey::FusionTree(key) => key,
            _ => {
                return Err(Error::InvalidArgument(
                    "tr() requires fusion-tree blocks".to_string(),
                ))
            }
        };
        // Off the coupled-block diagonal (codomain tree != domain tree): not on
        // the matrix diagonal of b_c, so it drops out of tr(b_c).
        if key.codomain_tree() != key.domain_tree() {
            continue;
        }
        let coupled = key
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| rule.vacuum());
        // Closing codomain leg i onto domain leg i bends the coupled loop, so
        // each block picks up its coupled charge's twist theta_c. For symmetric
        // (bosonic) categories theta == 1; for fermionic ones it is -1 on odd
        // sectors, which turns this sum into the supertrace — exactly matching
        // TensorKit's `tr` (and the partial-trace engine this replaces).
        let weight = rule.twist_scalar(coupled) * rule.dim_scalar(coupled);
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        // Walk only the codomain degeneracy multi-index and index both halves
        // diagonally (axis i and axis nout+i share the index) — the degeneracy
        // trace of this coupled sub-block.
        let count: usize = shape[..nout].iter().product();
        let mut indices = vec![0usize; nout];
        let mut partial = D::from_real(0.0);
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .enumerate()
                    .map(|(axis, &i)| i * strides[axis] + i * strides[nout + axis])
                    .sum::<usize>();
            partial = partial + data[position];
            for axis in 0..nout {
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
/// [`Complex64`] data, fixed at construction (the [`Dtype`] token of
/// [`Self::rand`], [`Self::zeros`] and so on; [`Self::from_block_fn`]
/// infers it from the fill closure) and reported by [`Self::dtype`].
/// Operations dispatch on the stored dtype internally; mixing dtypes in one
/// operation is [`Error::DtypeMismatch`] (widen explicitly with
/// [`Self::to_c64`]).
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
#[derive(Debug)]
pub struct Tensor {
    rt: Runtime,
    rule: RuleKind,
    // The tensor's own coupled space. For a lazy adjoint this is already the
    // *adjoint* coupled space, so all metadata is correct with no data touched.
    space: Arc<DynamicFusionMapSpace>,
    // Shared behind `Arc` so a lazy adjoint (see `adjoint`) can point at the
    // parent buffer with no copy; every value-read funnels through
    // `coupled_data`, so nothing else observes the sharing. For a lazy adjoint
    // this holds the *parent's* buffer in the parent's coupled layout.
    data: Arc<Data>,
    // `Some(parent_space)` marks a lazy adjoint (TensorKit's `AdjointTensorMap`):
    // `data` is the parent's buffer and `parent_space` its coupled space, so the
    // conjugate-transpose can be materialized on demand. `None` for an ordinary
    // tensor whose `data` already matches `space`.
    adjoint_source: Option<Arc<DynamicFusionMapSpace>>,
    // Memoized conjugate-transpose of a lazy adjoint's data, in `space`'s
    // layout; filled at most once by `coupled_data`. Empty for ordinary tensors.
    materialized: OnceLock<Arc<Data>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        // `OnceLock` isn't `Clone`; carry over an already-materialized buffer
        // (a cheap `Arc` bump) so a clone doesn't recompute the adjoint.
        let materialized = OnceLock::new();
        if let Some(buffer) = self.materialized.get() {
            let _ = materialized.set(Arc::clone(buffer));
        }
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::clone(&self.data),
            adjoint_source: self.adjoint_source.clone(),
            materialized,
        }
    }
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
        let (space, data) = if rule_kind == RuleKind::Su3 {
            // SU(3) is Generic: build the multiplicity-aware space directly (the
            // mult-free `with_rule!` binding cannot host a Generic rule).
            let space = build_space_generic(&Su3FusionRule::new(), hom)?;
            let data = apply_fill(&space, fill)?;
            (space, S::lift(data))
        } else {
            with_rule!(rule_kind, rule, {
                let space = build_space(rule, hom)?;
                let data = apply_fill(&space, fill)?;
                Ok::<_, Error>((space, S::lift(data)))
            })?
        };
        Ok(Self {
            rt: rt.clone(),
            rule: rule_kind,
            space: Arc::new(space),
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }

    /// Zero tensor of the given [`Dtype`] on `codomain <- domain`
    /// (TensorKit `zeros(Float64, W ← V)` / `zeros(ComplexF64, W ← V)`).
    /// All spaces must share one fusion rule.
    pub fn zeros<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        match dtype {
            Dtype::F64 => Self::build::<_, _, f64>(rt, codomain, domain, Fill::Zeros),
            Dtype::C64 => Self::build::<_, _, Complex64>(rt, codomain, domain, Fill::Zeros),
        }
    }

    /// Random tensor of the given [`Dtype`] on `codomain <- domain`
    /// (TensorKit `rand(Float64, W ← V)` / `rand(ComplexF64, W ← V)`):
    /// entries (real and imaginary parts for [`Dtype::C64`]) uniform in
    /// `[-1, 1)`.
    ///
    /// Deterministic per runtime: the n-th `rand`-family call on a given
    /// runtime always produces the same tensor. Use [`Self::rand_with_seed`]
    /// for an explicit stream.
    pub fn rand<'a, C, D>(rt: &Runtime, dtype: Dtype, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::rand_with_seed(rt, dtype, codomain, domain, rt.next_rand_seed())
    }

    /// Random tensor with an explicit seed (splitmix64 stream, entries
    /// uniform in `[-1, 1)`).
    pub fn rand_with_seed<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
        seed: u64,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        match dtype {
            Dtype::F64 => Self::build::<_, _, f64>(rt, codomain, domain, Fill::Rand(seed)),
            Dtype::C64 => Self::build::<_, _, Complex64>(rt, codomain, domain, Fill::Rand(seed)),
        }
    }

    /// Tensor filled block-by-block: `fill(key, indices)` is called for
    /// every element of every symmetry-allowed block, with `indices` local
    /// to the block (degeneracy coordinates, codomain axes first). Mirrors
    /// [`tenet_core::TensorMap::from_block_fn_with_fusion_space`].
    ///
    /// The constructed dtype follows the closure's return type (`f64` or
    /// [`Complex64`], the two [`TensorScalar`] impls) — no dtype token
    /// needed.
    ///
    /// The fusion-tree `key` labels domain legs with the domain `Space`'s
    /// own sectors (TensorKit's `f2.uncoupled`), not their duals; on both
    /// sides the uncoupled sectors fuse to the tree's coupled sector.
    pub fn from_block_fn<'a, C, D, S, F>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        mut fill: F,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
        S: TensorScalar,
        F: FnMut(&BlockKey, &[usize]) -> S,
    {
        Self::build(rt, codomain, domain, Fill::BlockFn(&mut fill))
    }

    /// Shared core of [`Self::id`] / [`Self::isomorphism`] /
    /// [`Self::isometry`]: checks that the domain fits in the codomain
    /// (exactly for `embed == false`, isometric embedding for
    /// `embed == true`) and fills every coupled-sector matrix with the
    /// (partial) identity, exactly TensorKit's `one!` per coupled block
    /// (`tensors/linalg.jl:102-158`).
    fn structural(
        rt: &Runtime,
        dtype: Dtype,
        codomain: Vec<&Space>,
        domain: Vec<&Space>,
        embed: bool,
        what: &str,
    ) -> Result<Self, Error> {
        let fused_codomain = Space::fuse_all(&codomain)?;
        let fused_domain = Space::fuse_all(&domain)?;
        let fits = if embed {
            // TensorKit `domain ≾ codomain`: sectorwise embeddable.
            fused_domain.sectors.iter().all(|&(sector, deg)| {
                fused_codomain
                    .sectors
                    .iter()
                    .any(|&(s, d)| s == sector && d >= deg)
            })
        } else {
            // TensorKit `domain ≅ codomain`: identical fused sector content.
            fused_codomain.sectors == fused_domain.sectors
        };
        if !fits {
            return Err(Error::InvalidArgument(format!(
                "{what}: codomain and domain are not {} (fused sector content differs)",
                if embed {
                    "isometrically embeddable"
                } else {
                    "isomorphic"
                }
            )));
        }
        let mut t = Self::build::<_, _, f64>(rt, codomain, domain, Fill::Zeros)?;
        let regions = sector_regions(t.space.structure(), t.space.nout())?;
        let Data::F64(data) = Arc::make_mut(&mut t.data) else {
            unreachable!("structural constructors build f64 host tensors");
        };
        for region in &regions {
            for i in 0..region.rows.min(region.cols) {
                data[region.offset + i * (region.rows + 1)] = 1.0;
            }
        }
        Ok(match dtype {
            Dtype::F64 => t,
            Dtype::C64 => t.to_c64(),
        })
    }

    /// The identity endomorphism on `spaces <- spaces` (TensorKit `id(V)`,
    /// `tensors/linalg.jl:75-82`): every coupled-sector block is the
    /// identity matrix.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 2), (1, 1)]);
    /// let t = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
    /// let id = Tensor::id(&rt, Dtype::F64, [&v])?;
    /// assert_eq!(id.compose(&t)?.data(), t.data());
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn id<'a, S>(rt: &Runtime, dtype: Dtype, spaces: S) -> Result<Self, Error>
    where
        S: IntoIterator<Item = &'a Space>,
    {
        let spaces: Vec<&Space> = spaces.into_iter().collect();
        Self::structural(rt, dtype, spaces.clone(), spaces, false, "id")
    }

    /// The canonical structural isomorphism `codomain <- domain` (TensorKit
    /// `isomorphism(W ← V)`, `tensors/linalg.jl:102-109`): every
    /// coupled-sector block is the identity matrix, which requires the fused
    /// codomain and domain to carry identical sector content. The
    /// finite-torus norm fuser is `isomorphism(fuse(dual(l) ⊗ l) ←
    /// dual(l) ⊗ l)`.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 1), (1, 1)]);
    /// let fused = v.dual().fuse(&v)?;
    /// let f = Tensor::isomorphism(&rt, Dtype::F64, [&fused], [&v.dual(), &v])?;
    /// // Unitary: f† ∘ f is the identity on the product space.
    /// let roundtrip = f.adjoint()?.compose(&f)?;
    /// assert_eq!(roundtrip.data(), Tensor::id(&rt, Dtype::F64, [&v.dual(), &v])?.data());
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn isomorphism<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::structural(
            rt,
            dtype,
            codomain.into_iter().collect(),
            domain.into_iter().collect(),
            false,
            "isomorphism",
        )
    }

    /// TensorKit `unitary(W ← V)` (`tensors/linalg.jl:129-132`): identical
    /// to [`Self::isomorphism`] — TensorKit only adds a Euclidean
    /// inner-product check, which every tenet fusion rule satisfies.
    pub fn unitary<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::structural(
            rt,
            dtype,
            codomain.into_iter().collect(),
            domain.into_iter().collect(),
            false,
            "unitary",
        )
    }

    /// The canonical isometry `codomain <- domain` (TensorKit
    /// `isometry(W ← V)`, `tensors/linalg.jl:149-158`): each coupled-sector
    /// block is the partial identity (the first `cols` columns of the
    /// identity), so `t† ∘ t = id(domain)`. Requires the domain to embed
    /// isometrically in the codomain (sectorwise `deg_domain <=
    /// deg_codomain`).
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let small = Space::su2([(0, 1), (1, 1)]);
    /// let big = Space::su2([(0, 2), (1, 3), (2, 1)]);
    /// let w = Tensor::isometry(&rt, Dtype::F64, [&big], [&small])?;
    /// let id = Tensor::id(&rt, Dtype::F64, [&small])?;
    /// assert_eq!(w.adjoint()?.compose(&w)?.data(), id.data());
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn isometry<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::structural(
            rt,
            dtype,
            codomain.into_iter().collect(),
            domain.into_iter().collect(),
            true,
            "isometry",
        )
    }

    /// TensorKit `twist(t, inds)` (`tensors/indexmanipulations.jl:62-97`):
    /// multiplies each fusion-tree block by the product over `legs` (flat
    /// leg indices, codomain first) of the ribbon-twist eigenvalue θ of that
    /// leg's uncoupled sector on the block's fusion tree. θ = −1 for odd
    /// fermionic sectors and +1 for every bosonic sector, so this is a no-op
    /// on purely bosonic legs and an involution on fermionic ones.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::fz2([(0, 1), (1, 1)]);
    /// let t = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
    /// let twisted = t.twist(&[0])?;
    /// assert_eq!(twisted.twist(&[0])?.data(), t.data()); // θ² = 1
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn twist(&self, legs: &[usize]) -> Result<Self, Error> {
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "twist leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let nout = self.codomain_rank();
        let rule = self.rule;
        // TensorKit `has_shared_twist` (`indexmanipulations.jl`): the twist is
        // the identity when every requested leg carries theta = 1 on every
        // block. Bosonic rules are all-theta=1 by construction (O(1)
        // short-circuit — the Z2/U1/SU2 finite-torus paths); a
        // fermionic/anyonic tensor still shares its buffer when no requested
        // leg touches a twisted sector. Either way, skip the whole-buffer
        // clone-and-scale-by-1 and return the shared data.
        let twist_is_identity = with_rule!(rule, rule, {
            rule.braiding_style().is_bosonic() || {
                let structure = self.space.structure();
                (0..structure.block_count()).try_fold(true, |noop, index| {
                    let block = structure.block(index)?;
                    Ok::<_, Error>(
                        noop && match block.key() {
                            BlockKey::FusionTree(key) => legs.iter().all(|&leg| {
                                let sector = if leg < nout {
                                    key.codomain_uncoupled()[leg]
                                } else {
                                    key.domain_uncoupled()[leg - nout]
                                };
                                rule.twist_scalar(sector) == 1.0
                            }),
                            _ => true,
                        },
                    )
                })?
            }
        });
        if twist_is_identity {
            return Ok(self.clone());
        }
        self.scaled_blocks(&self.space, &|key| match key {
            BlockKey::FusionTree(key) => with_rule!(rule, rule, {
                legs.iter()
                    .map(|&leg| {
                        rule.twist_scalar(if leg < nout {
                            key.codomain_uncoupled()[leg]
                        } else {
                            key.domain_uncoupled()[leg - nout]
                        })
                    })
                    .product()
            }),
            _ => 1.0,
        })
    }

    /// TensorKit `flip(t, I)` (`tensors/indexmanipulations.jl:8-29`): return
    /// a tensor isomorphic to `self` where the duality flag of each leg in
    /// `legs` (flat indices, codomain first; a leg listed twice is flipped
    /// twice, sequentially) is toggled, `space(t', i) = flip(space(t, i))`.
    /// The stored sectors and the block layout are unchanged; each
    /// fusion-tree block picks up the Z-isomorphism phase of
    /// `fusiontrees/braiding_manipulations.jl:384-414` per flipped leg with
    /// uncoupled sector `a` and pre-flip duality `d` (χ = Frobenius-Schur
    /// phase, θ = ribbon twist; both real for every rule in scope):
    /// codomain leg → `d ? χ·θ : 1`; domain leg → `d ? χ : θ`.
    ///
    /// Like TensorKit's, this `flip` is *not* an involution: flipping the
    /// same leg twice returns to the original spaces but can scale odd
    /// blocks (e.g. by θ = −1 on fermionic legs); only `flip⁴ = id` in
    /// general.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::fz2([(0, 1), (1, 1)]);
    /// let t = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
    ///     BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
    ///     _ => 3.0,
    /// })?;
    /// // TensorKit 0.17: flip(t, 2) on V ← V negates the odd block (θ = −1)
    /// // and re-orients the domain leg (see the flip oracle test).
    /// let flipped = t.flip(&[1])?;
    /// assert_eq!(flipped.data(), &[2.0, -3.0]);
    /// assert!(!flipped.space(1)?.is_dual()); // was dual: space(t, 1) = dual(v)
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn flip(&self, legs: &[usize]) -> Result<Self, Error> {
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "flip leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let hom = self.space.homspace();
        let nout = hom.codomain().len();
        let leg_of = |leg: usize| {
            if leg < nout {
                &hom.codomain().legs()[leg]
            } else {
                &hom.domain().legs()[leg - nout]
            }
        };
        // Sequential semantics for repeated legs (TensorKit flips one index
        // at a time): record the duality each occurrence sees.
        let mut flip_count = vec![0usize; rank];
        let occurrences: Vec<(usize, bool)> = legs
            .iter()
            .map(|&leg| {
                let dual = leg_of(leg).is_dual() ^ (flip_count[leg] % 2 == 1);
                flip_count[leg] += 1;
                (leg, dual)
            })
            .collect();

        let toggled_leg = |index: usize, leg: &tenet_core::SectorLeg| {
            if flip_count[index] % 2 == 1 {
                tenet_core::SectorLeg::new(leg.iter(), !leg.is_dual())
            } else {
                leg.clone()
            }
        };
        let new_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(
                hom.codomain()
                    .legs()
                    .iter()
                    .enumerate()
                    .map(|(index, leg)| toggled_leg(index, leg)),
            ),
            FusionProductSpace::new(
                hom.domain()
                    .legs()
                    .iter()
                    .enumerate()
                    .map(|(index, leg)| toggled_leg(nout + index, leg)),
            ),
        );
        let new_space = with_rule!(self.rule, rule, build_space(rule, new_hom))?;
        // Flipping preserves the stored sectors, so the flipped space must
        // reproduce the block layout exactly; anything else is a bug.
        let old_structure = self.space.structure();
        let new_structure = new_space.structure();
        if new_structure.block_count() != old_structure.block_count() {
            return Err(internal_layout_error("flip changed the block count"));
        }
        for index in 0..old_structure.block_count() {
            let old_block = old_structure.block(index)?;
            let new_block = new_structure.block(index)?;
            if old_block.offset() != new_block.offset() || old_block.shape() != new_block.shape() {
                return Err(internal_layout_error("flip changed the block layout"));
            }
        }

        let rule = self.rule;
        let flipped = self.scaled_blocks(&new_space, &|key| match key {
            BlockKey::FusionTree(key) => with_rule!(rule, rule, {
                occurrences
                    .iter()
                    .map(|&(leg, dual)| {
                        let sector = if leg < nout {
                            key.codomain_uncoupled()[leg]
                        } else {
                            key.domain_uncoupled()[leg - nout]
                        };
                        let chi = rule.frobenius_schur_phase_scalar(sector);
                        let theta = rule.twist_scalar(sector);
                        // TensorKit 0.17 flip coefficients (forward, real χ/θ).
                        if leg < nout {
                            if dual {
                                chi * theta
                            } else {
                                1.0
                            }
                        } else if dual {
                            chi
                        } else {
                            theta
                        }
                    })
                    .product()
            }),
            _ => 1.0,
        })?;
        Ok(Self {
            space: Arc::new(new_space),
            ..flipped
        })
    }

    /// Clones the storage scaled block-wise by `factor_of(key)` (evaluated
    /// on the blocks of `structure_space`, whose layout must match the
    /// stored one), shared by [`Self::twist`] and [`Self::flip`].
    fn scaled_blocks(
        &self,
        structure_space: &DynamicFusionMapSpace,
        factor_of: &dyn Fn(&BlockKey) -> f64,
    ) -> Result<Self, Error> {
        let data = match self.coupled_data() {
            Data::F64(data) => {
                let mut out = data.clone();
                scale_blocks_impl(structure_space, &mut out, factor_of)?;
                Data::F64(out)
            }
            Data::C64(data) => {
                let mut out = data.clone();
                scale_blocks_impl(structure_space, &mut out, factor_of)?;
                Data::C64(out)
            }
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("twist/flip")),
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }

    /// Wraps a same-runtime, same-rule result of an expert-layer call.
    fn with(&self, space: DynamicFusionMapSpace, data: Data) -> Self {
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
        }
    }

    /// The stored buffer resolved into this tensor's own coupled layout. The
    /// single funnel through which every value-read observes the data, so a
    /// lazy adjoint materializes its conjugate-transpose here exactly once
    /// (memoized) without any other method being aware of it. Ordinary tensors
    /// return the stored buffer directly.
    fn coupled_data(&self) -> &Data {
        if let Some(parent_space) = &self.adjoint_source {
            return self
                .materialized
                .get_or_init(|| Arc::new(self.materialize_adjoint(parent_space)))
                .as_ref();
        }
        // Compact diagonal storage materializes into its dense equivalent on
        // first value-read (issue #55), memoized like the lazy-adjoint case, so
        // no other method observes `Data::Diagonal`.
        if let Data::Diagonal(diagonal) = self.data.as_ref() {
            return self
                .materialized
                .get_or_init(|| Arc::new(self.materialize_diagonal(diagonal)))
                .as_ref();
        }
        self.data.as_ref()
    }

    /// A non-diagonal clone: `Data::Diagonal` materialized into its dense
    /// equivalent, everything else shared by `Arc` (cheap). Used at the entry of
    /// the binary ops that read the raw buffer (`contract`/`add`/`inner`), which
    /// do not yet have a diagonal-aware scaling path (that is issue #55 PR2).
    fn densified_if_diagonal(&self) -> Self {
        if !matches!(self.data.as_ref(), Data::Diagonal(_)) {
            return self.clone();
        }
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(self.coupled_data().clone()),
            adjoint_source: None,
            materialized: OnceLock::new(),
        }
    }

    /// Rebuilds the dense block-diagonal buffer of a [`Data::Diagonal`] tensor in
    /// its own (`space`) layout — the eager fallback for any consumer other than
    /// a diagonal-aware scaling (issue #55). Reproduces the former dense diagonal
    /// tensor bit-for-bit via [`tenet_matrixalgebra::diagonal_bond_data`].
    fn materialize_diagonal(&self, diagonal: &DiagonalData) -> Data {
        match diagonal {
            DiagonalData::RealF64(spectrum) => Data::F64(
                tenet_matrixalgebra::diagonal_bond_data(&self.space, spectrum, &|value| value)
                    .expect("diagonal fill is total on the stored bond space"),
            ),
            DiagonalData::RealC64(spectrum) => Data::C64(
                tenet_matrixalgebra::diagonal_bond_data(&self.space, spectrum, &|value| {
                    Complex64::new(value, 0.0)
                })
                .expect("diagonal fill is total on the stored bond space"),
            ),
            DiagonalData::C64(spectrum) => Data::C64(
                tenet_matrixalgebra::diagonal_bond_data(&self.space, spectrum, &|value| value)
                    .expect("diagonal fill is total on the stored bond space"),
            ),
        }
    }

    /// Conjugate-transpose of a lazy adjoint's shared parent buffer into this
    /// tensor's own coupled (`space`) layout — the eager fallback TensorKit
    /// takes (`convert(TensorMap, ::AdjointTensorMap)`) when an adjoint is
    /// consumed by something other than a contraction.
    fn materialize_adjoint(&self, parent_space: &DynamicFusionMapSpace) -> Data {
        // SU(N) (Generic): materialize through the generic block-relabel sibling.
        // The result is a genuine (non-lazy) SU(N) tensor's coupled data, so a
        // downstream consumer (norm/svd/contract) never has to fold a conjugate
        // through the mult-free-only Structure route — the route whose
        // non-self-dual coupled-sector mislabel bug the mult-free lazy-adjoint
        // fold was fixed for. SU(3) is non-self-dual (3 <-> 3̄), so materializing
        // here (rather than folding) is the deliberate, mislabel-proof choice.
        macro_rules! adjoint_dyn_dispatch {
            ($rule:ident, $parent:expr) => {
                if self.rule == RuleKind::Su3 {
                    tenet_tensors::adjoint_dyn_generic(&Su3FusionRule::new(), parent_space, $parent)
                } else {
                    with_rule!(self.rule, $rule, {
                        tenet_tensors::adjoint_dyn($rule, parent_space, $parent)
                    })
                }
            };
        }
        match self.data.as_ref() {
            Data::F64(parent) => {
                let (_space, out) = adjoint_dyn_dispatch!(rule, parent)
                    .expect("adjoint_dyn is total on a tensor's own space");
                Data::F64(out)
            }
            Data::C64(parent) => {
                let (_space, out) = adjoint_dyn_dispatch!(rule, parent)
                    .expect("adjoint_dyn is total on a tensor's own space");
                Data::C64(out)
            }
            Data::Diagonal(_) => unreachable!("adjoint() densifies diagonal storage first"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => unreachable!("adjoint() rejects device tensors"),
        }
    }

    /// The scalar type this tensor stores.
    pub fn dtype(&self) -> Dtype {
        // Discriminant only; dtype is adjoint-invariant, so read the stored
        // buffer directly (no need to materialize a lazy adjoint).
        match self.data.as_ref() {
            Data::F64(_) => Dtype::F64,
            Data::C64(_) => Dtype::C64,
            // Diagonal storage carries its own dtype tag (no materialization).
            Data::Diagonal(diagonal) => diagonal.dtype(),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Dtype::F64,
        }
    }

    /// Where this tensor's data lives: [`Placement::Host`] or
    /// [`Placement::Cuda`] with the device ordinal. Transfers are always
    /// explicit (`to_cuda()` / `to_host()`).
    pub fn placement(&self) -> Placement {
        match self.data.as_ref() {
            Data::F64(_) | Data::C64(_) | Data::Diagonal(_) => Placement::Host,
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
        let data = match self.coupled_data() {
            Data::CudaF64(storage) => Data::CudaF64(Arc::clone(storage)),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
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
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }

    /// Downloads a device tensor back to host storage; a plain copy when
    /// already host-resident.
    #[cfg(feature = "cuda")]
    pub fn to_host(&self) -> Result<Self, Error> {
        let data = match self.coupled_data() {
            Data::F64(_) | Data::C64(_) => self.coupled_data().clone(),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
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
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
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

    /// Number of tensors currently sharing this tensor's storage allocation.
    #[doc(hidden)]
    pub fn storage_strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Flat `f64` storage in the TensorKit-equivalent coupled-sector matrix
    /// layout (column-major inside each coupled block).
    ///
    /// This is an **internal-packing inspection API** (tests, debugging,
    /// oracle comparisons), not a general element-access API:
    ///
    /// - The slice is the internal buffer in the coupled-sector matrix
    ///   layout; element positions depend on block order, the fusion-tree
    ///   basis, and column-major packing.
    /// - That layout is **not a stable ABI**: it may change between
    ///   versions without notice.
    /// - There are no implicit device copies: on a device tensor this
    ///   panics — download explicitly with `to_host()` first.
    /// - For semantic access, prefer the operation APIs (contractions,
    ///   [`Self::scalar`], norms); a stable block iterator / dense export
    ///   would be a separate future API.
    ///
    /// # Panics
    ///
    /// Panics if the tensor stores c64 data (use [`Self::data_c64`]) or is
    /// device-resident (use `to_host()`).
    pub fn data(&self) -> &[f64] {
        match self.coupled_data() {
            Data::F64(data) => data,
            Data::C64(_) => panic!("data(): tensor stores c64 data; use data_c64()"),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => panic!("data(): tensor is device-resident; use to_host()"),
        }
    }

    /// Flat [`Complex64`] storage in the coupled-sector matrix layout.
    ///
    /// The same caveats as [`Self::data`] apply: this inspects the internal
    /// coupled-sector packing (layout-dependent, not a stable ABI, no
    /// implicit device copies; intended for tests and debugging).
    ///
    /// # Panics
    ///
    /// Panics if the tensor stores f64 data (use [`Self::data`]) or is
    /// device-resident (use `to_host()`).
    pub fn data_c64(&self) -> &[Complex64] {
        match self.coupled_data() {
            Data::C64(data) => data,
            Data::F64(_) => panic!("data_c64(): tensor stores f64 data; use data()"),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => panic!("data_c64(): tensor is device-resident; use to_host()"),
        }
    }

    /// Widens to a c64 tensor (imaginary parts zero); a cheap clone when the
    /// tensor already stores c64 data.
    pub fn to_c64(&self) -> Self {
        let data = match self.coupled_data() {
            Data::F64(data) => Data::C64(
                data.iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            Data::C64(data) => Data::C64(data.clone()),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => panic!("to_c64(): tensor is device-resident; use to_host()"),
        };
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
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

    /// Quantum-dimension-weighted size of one flat leg.
    pub fn leg_dim(&self, axis: usize) -> Result<usize, Error> {
        let hom = self.space.homspace();
        let leg = if axis < hom.codomain().len() {
            &hom.codomain().legs()[axis]
        } else if axis < hom.rank() {
            &hom.domain().legs()[axis - hom.codomain().len()]
        } else {
            return Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                hom.rank()
            )));
        };
        with_rule!(self.rule, rule, {
            Ok(leg
                .iter()
                .map(|(sector, deg)| (deg as f64 * rule.dim_scalar(sector)).round() as usize)
                .sum())
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

    /// The single element of a rank-0 (scalar) tensor, e.g. the result of
    /// contracting every leg. The returned [`Scalar`] variant matches
    /// [`Self::dtype`] (`F64` for f64 tensors, `C64` for c64 tensors);
    /// errors on tensors with legs.
    pub fn scalar(&self) -> Result<Scalar, Error> {
        self.check_rank0()?;
        match self.coupled_data() {
            Data::F64(data) => Ok(Scalar::F64(data.iter().sum())),
            Data::C64(data) => Ok(Scalar::C64(data.iter().sum())),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scalar()")),
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

    fn validate_host_destination(&self, input: &Self) -> Result<(), Error> {
        self.check_same_world(input)?;
        if self.placement() != Placement::Host {
            return Err(Error::PlacementMismatch);
        }
        if self.adjoint_source.is_some() || matches!(self.data.as_ref(), Data::Diagonal(_)) {
            return Err(Error::InvalidArgument(
                "destination must use ordinary dense host storage".to_string(),
            ));
        }
        if Arc::ptr_eq(&self.data, &input.data) {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_exact_destination_space(
        &self,
        expected: &DynamicFusionMapSpace,
    ) -> Result<(), Error> {
        if self.space.as_ref() != expected {
            return Err(Error::InvalidArgument(
                "destination fusion space or block layout does not match the operation result"
                    .to_string(),
            ));
        }
        let required = expected.required_len()?;
        let actual = match self.data.as_ref() {
            Data::F64(data) => data.len(),
            Data::C64(data) => data.len(),
            Data::Diagonal(_) => 0,
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => 0,
        };
        if actual != required {
            return Err(Error::InvalidArgument(format!(
                "destination storage length {actual} does not match required length {required}"
            )));
        }
        Ok(())
    }

    /// Categorical composition `self * rhs`: contracts `self`'s domain with
    /// `rhs`'s codomain, leg by leg. TensorKit `A * B` (`mul!` on coupled
    /// blocks); also available as the `&a * &b` operator (see the
    /// [`std::ops::Mul`] impl, which panics instead of returning `Result`).
    ///
    /// # Fermionic semantics: `compose` vs `contract`
    ///
    /// `compose` / `&a * &b` is TensorKit's `A * B` / `mul!`: **no**
    /// fermionic supertrace twist is inserted on dual composed legs.
    /// [`Self::contract`] and the `tensor!` macro are TensorKit's
    /// `tensorcontract!` / `@tensor`: dual contracted legs **are** twisted
    /// (TensorKit `tensoroperations.jl:388-420` twists only in
    /// `blas_contract!`, never in `mul!`). For bosonic rules the two agree
    /// exactly; for fermionic rules (fZ2 and products containing it) they
    /// can differ by signs. Worked example — the odd sector flips sign:
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::fz2([(0, 1), (1, 1)]);
    /// // a : v <- v*, b : v* <- v; the composed legs are dual (v*).
    /// let odd = |key: &BlockKey| matches!(key, BlockKey::FusionTree(k)
    ///     if k.codomain_uncoupled()[0].id() == 1);
    /// let a = Tensor::from_block_fn(&rt, [&v], [&v.dual()], |k, _| if odd(k) { 2.0 } else { 5.0 })?;
    /// let b = Tensor::from_block_fn(&rt, [&v.dual()], [&v], |k, _| if odd(k) { 3.0 } else { 7.0 })?;
    /// let composed = a.compose(&b)?;                    // mul! semantics: no twist
    /// let contracted = a.contract(&b, &[1], &[0])?;     // tensorcontract!: twist on v*
    /// assert_eq!(composed.data()[0], contracted.data()[0]);  // even sector agrees
    /// assert_eq!(composed.data()[1], -contracted.data()[1]); // odd sector: sign flip
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    ///
    /// Rule of thumb: use `compose` when you mean operator/matrix
    /// multiplication of tensor maps (TensorKit `A * B`); use
    /// [`Self::contract`] / `tensor!` when you mean index-notation
    /// contraction (TensorKit `@tensor`). Bosonic results are identical.
    #[doc(alias = "mul")]
    #[doc(alias = "matmul")]
    pub fn compose(&self, rhs: &Self) -> Result<Self, Error> {
        if self.domain_rank() != rhs.codomain_rank() {
            return Err(Error::InvalidArgument(format!(
                "compose shape mismatch: lhs domain rank {} vs rhs codomain rank {}",
                self.domain_rank(),
                rhs.codomain_rank()
            )));
        }
        // Diagonal fast-path (TensorKit `DiagonalTensorMap` `rmul!`/`lmul!`):
        // composing a dense operand with a real diagonal is a pure per-block bond
        // scaling — no GEMM, no braiding/twist (verified against TK
        // `diagonal.jl`; diagonal mul! never recouples). Only one operand
        // diagonal with a real spectrum; complex-spectrum (eig `d`) and
        // diagonal∘diagonal fall through to the densifying contract path below.
        match (self.real_diagonal_spectrum(), rhs.real_diagonal_spectrum()) {
            // `t * D`: scale `self`'s trailing bond axis (columns). `self.domain`
            // is the single bond leg == `D.codomain`, so the space is `self`'s.
            (None, Some(spectrum)) => return self.scaled_axis_copy(None, spectrum),
            // `D * t`: scale `rhs`'s leading bond axis (rows). `rhs.codomain` is
            // the single bond leg == `D.domain`, so the space is `rhs`'s.
            (Some(spectrum), None) => return rhs.scaled_axis_copy(Some(0), spectrum),
            _ => {}
        }
        let lhs_axes: Vec<usize> = (self.codomain_rank()..self.rank()).collect();
        let rhs_axes: Vec<usize> = (0..rhs.codomain_rank()).collect();
        // `contract` twists the dual contracted rhs legs (tensorcontract!
        // parity); the twist is involutive (θ = ±1), so pre-twisting those
        // legs cancels it exactly and yields mul! semantics. SU(N) (Generic)
        // is bosonic and cannot ride the mult-free `with_rule!` binding.
        let fermionic = self.rule != RuleKind::Su3
            && with_rule!(self.rule, rule, {
                rule.braiding_style() == tenet_core::BraidingStyleKind::Fermionic
            });
        if fermionic {
            let hom = rhs.space.homspace();
            let dual_legs: Vec<usize> = rhs_axes
                .iter()
                .copied()
                .filter(|&axis| hom.codomain().legs()[axis].is_dual())
                .collect();
            if !dual_legs.is_empty() {
                return self.contract(&rhs.twist(&dual_legs)?, &lhs_axes, &rhs_axes);
            }
        }
        self.contract(rhs, &lhs_axes, &rhs_axes)
    }

    /// Contracts `lhs_axes` of `self` with `rhs_axes` of `rhs` (pairwise, in
    /// list order), with the default output order: `self`'s open axes
    /// ascending become the codomain, `rhs`'s open axes ascending become the
    /// domain. TensorKit `tensorcontract!` with default `pAB`.
    ///
    /// **Fermionic semantics**: like TensorKit `tensorcontract!` / `@tensor`
    /// (and the `tensor!` macro), this **twists** dual contracted legs with
    /// the fermionic supertrace twist — unlike [`Self::compose`] / `&a * &b`
    /// (TensorKit `A * B` / `mul!`), which never does. Bosonic rules are
    /// unaffected; fermionic rules can differ by signs. See the worked
    /// example on [`Self::compose`].
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
        // SU(N) (Generic): a lazy-adjoint operand is materialized to plain
        // coupled data BEFORE contracting, rather than folded into the GEMM seam.
        // The mult-free seam folds a conjugate via its Structure route, whose
        // non-self-dual coupled-sector mislabel was the historical bug; SU(3) is
        // non-self-dual (3 <-> 3̄), so we route it through the mislabel-proof
        // eager materialization (`materialize_adjoint`, generic block-relabel)
        // instead. `scale(1.0)` reads `coupled_data` — the materialization point —
        // and returns an ordinary (non-lazy) tensor, so both operands then take
        // the direct core/compose GEMM with no conjugate flag. Gated on `Su3` to
        // keep the mult-free seam byte-for-byte unchanged (the χ32 guarantee).
        if self.rule == RuleKind::Su3
            && (self.adjoint_source.is_some() || rhs.adjoint_source.is_some())
        {
            let lhs = if self.adjoint_source.is_some() {
                self.scale(1.0)?
            } else {
                self.clone()
            };
            let rhs = if rhs.adjoint_source.is_some() {
                rhs.scale(1.0)?
            } else {
                rhs.clone()
            };
            return lhs.contract(&rhs, lhs_axes, rhs_axes);
        }
        // Order-parity fast path for a real diagonal operand (#75): instead of
        // densifying it to an O(d²) block-diagonal and running an O(d²·n) GEMM,
        // scale the OTHER operand's contracted leg by the spectrum (O(d·n)) and
        // `permute` the result into the contract output arrangement (O(n)). The
        // `permute` reuses the tested recoupling/repartition machinery, so the
        // result space — including leg duality and the codomain/domain split — is
        // correct for ANY geometry, not just edge legs. This is the same
        // scale + one-permute structure TensorKit runs (a `Diagonal` block scales
        // the recoupled operand); see docs/complexity_parity_policy.md.
        //
        // `contract` (tensorcontract!) applies a supertrace twist to `rhs`'s DUAL
        // contracted legs; `mul!` (the scaling below) does not. The exact
        // relation, from the cancellation `compose` performs, is
        // `contract(a, b) = mul!(a, b.twist(dual contracted rhs legs))`. So
        // pre-twist `rhs`'s dual contracted leg: when `rhs` is the diagonal, fold
        // θ into the spectrum (no densify); when `rhs` is the dense operand,
        // `twist` it. θ = ±1 by charge parity, identity for bosonic rules.
        // SU(N) (Generic) is bosonic and cannot ride the mult-free `with_rule!`
        // binding; short-circuit the twist probe (the diagonal fast path below
        // never fires for it — SU(N) has no `Data::Diagonal` factors yet).
        let fermionic = self.rule != RuleKind::Su3
            && with_rule!(self.rule, rule, {
                rule.braiding_style() == tenet_core::BraidingStyleKind::Fermionic
            });
        if lhs_axes.len() == 1 && rhs_axes.len() == 1 {
            let twist_rhs_leg = fermionic && rhs.leg_is_dual(rhs_axes[0]);
            match (self.real_diagonal_spectrum(), rhs.real_diagonal_spectrum()) {
                // A * D: scale A's contracted leg by the (twist-folded) spectrum,
                // then repartition to the output arrangement (A's open axes ->
                // codomain, the scaled leg -> domain).
                (None, Some(spectrum)) => {
                    let leg = lhs_axes[0];
                    let folded = self.twist_folded_spectrum(spectrum, twist_rhs_leg);
                    let scaled = self.scaled_axis_copy(Some(leg), &folded)?;
                    let codomain: Vec<usize> = (0..self.rank()).filter(|&a| a != leg).collect();
                    return scaled.permute(&codomain, &[leg]);
                }
                // D * A: pre-twist A's dual contracted leg, scale it, then
                // repartition (the scaled leg -> codomain 0, A's open -> domain).
                (Some(spectrum), None) => {
                    let leg = rhs_axes[0];
                    let pretwisted = if twist_rhs_leg {
                        rhs.twist(&[leg])?
                    } else {
                        rhs.clone()
                    };
                    let scaled = pretwisted.scaled_axis_copy(Some(leg), spectrum)?;
                    let domain: Vec<usize> = (0..rhs.rank()).filter(|&a| a != leg).collect();
                    return scaled.permute(&[leg], &domain);
                }
                _ => {}
            }
        }
        // Fallback (complex-spectrum diagonal, diagonal∘diagonal, or a multi-axis
        // contraction): materialize the diagonal to dense and run the ordinary
        // contraction. Densify is a no-op clone for non-diagonal operands.
        if matches!(self.data.as_ref(), Data::Diagonal(_))
            || matches!(rhs.data.as_ref(), Data::Diagonal(_))
        {
            return self.densified_if_diagonal().contract(
                &rhs.densified_if_diagonal(),
                lhs_axes,
                rhs_axes,
            );
        }
        // SU(N) (Generic), Stage B3c-2 source-transform route: the direct GEMM
        // engine only accepts core/compose form (lhs contracted axes == its whole
        // domain in order, rhs contracted axes == its whole codomain in order).
        // Any other arrangement is canonicalized here by composing ALREADY
        // TK-pinned primitives — one generic permute per operand (the B3a tree
        // transform, which carries all the recoupling) followed by the core
        // contract — so this wiring adds no mathematics of its own; the route-
        // equivalence gate pins `contract == explicit permute + core contract`.
        // Recursion is bounded: the permuted arrangement is canonical by
        // construction, so the recursive call falls through to the seam below.
        // Placed after the diagonal fast paths so a diagonal bond operand keeps
        // its O(d·n) scaling route, and after the adjoint materialization so
        // operands here are plain dense tensors.
        if self.rule == RuleKind::Su3 {
            let canonical_lhs = (self.codomain_rank()..self.rank()).collect::<Vec<_>>();
            let canonical_rhs = (0..rhs.codomain_rank()).collect::<Vec<_>>();
            if lhs_axes != canonical_lhs.as_slice() || rhs_axes != canonical_rhs.as_slice() {
                let lhs_open: Vec<usize> =
                    (0..self.rank()).filter(|a| !lhs_axes.contains(a)).collect();
                let rhs_open: Vec<usize> =
                    (0..rhs.rank()).filter(|a| !rhs_axes.contains(a)).collect();
                let lhs = self.permute(&lhs_open, lhs_axes)?;
                let rhs = rhs.permute(rhs_axes, &rhs_open)?;
                let contracted = (lhs_open.len()..lhs.rank()).collect::<Vec<_>>();
                let rhs_contracted = (0..rhs_axes.len()).collect::<Vec<_>>();
                return lhs.contract(&rhs, &contracted, &rhs_contracted);
            }
        }
        // Fold a lazy-adjoint operand into the GEMM with no copy. The adjoint is
        // transpose (codomain<->domain) + elementwise conjugate; both fold into
        // the contraction seam: feed it the shared PARENT buffer + parent space,
        // remap the contracted axes adjoint->parent, and raise the conjugate
        // flag. The seam applies the adjoint inside the GEMM — a transpose plus a
        // data-only conjugation (BLAS `op='T'` for real, `op='C'` for complex) —
        // so f64 and c64 take the same route with no materialized conjugate-
        // transpose. The result (`dst`) is still built from the adjoint space, so
        // callers see exactly the materialized-adjoint result. This is
        // TensorKit's `AdjointTensorMap` contraction; verified against TensorKit
        // (`A'*B == @tensor conj(A[v;w])*B[v;x]`) and, for non-self-dual (U(1))
        // symmetries, against the eager-adjoint oracle in tenet-tensors.
        let (lhs_space, lhs_axes_seam, lhs_conj) = self.seam_operand(lhs_axes);
        let (rhs_space, rhs_axes_seam, rhs_conj) = rhs.seam_operand(rhs_axes);
        // The seam always consumes the raw stored buffer (it never materializes):
        // for a lazy adjoint that buffer is the shared parent, conjugated by the
        // flag; for an ordinary tensor it is just the stored data.
        match (self.data.as_ref(), rhs.data.as_ref()) {
            (Data::F64(a), Data::F64(b)) => self.contract_impl(
                &lhs_space,
                a,
                &lhs_axes_seam,
                lhs_conj,
                &rhs_space,
                b,
                &rhs_axes_seam,
                rhs_conj,
                rhs,
                lhs_axes,
                rhs_axes,
            ),
            (Data::C64(a), Data::C64(b)) => self.contract_impl(
                &lhs_space,
                a,
                &lhs_axes_seam,
                lhs_conj,
                &rhs_space,
                b,
                &rhs_axes_seam,
                rhs_conj,
                rhs,
                lhs_axes,
                rhs_axes,
            ),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                // Device tensors are never lazy adjoints (`adjoint` rejects them).
                self.contract_cuda_impl(rhs, a, b, lhs_axes, rhs_axes)
            }
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// The `(seam_space, seam_contracted_axes, conjugate)` an operand feeds to the
    /// contraction seam:
    /// - ordinary tensor: its own space, the axes unchanged, no conjugation;
    /// - real lazy adjoint: the `adjoint_view` (a strided view over the shared
    ///   parent buffer presenting the adjoint space in adjoint axis order) with no
    ///   conjugation — a real adjoint is a pure transpose, so this feeds the fast
    ///   direct-GEMM route with no copy and no remap;
    /// - complex lazy adjoint: the PARENT space (its stored buffer is the shared
    ///   parent), the contracted axes remapped adjoint->parent, and
    ///   `conjugate = true`, so the seam folds the conjugate-transpose into the
    ///   GEMM (BLAS `op='C'`) with no materialized copy.
    ///
    /// The adjoint->parent axis map: the adjoint space's codomain (axis `< nin_p`)
    /// is the parent's domain (`nout_p + a`), its domain is the parent's codomain
    /// (`a - nin_p`) — the inverse of `adjointtensorindex`. The seam's lowering
    /// re-applies `adjointtensorindex` to these, recovering the adjoint contraction
    /// against the parent buffer.
    fn seam_operand(&self, user_axes: &[usize]) -> (DynamicFusionMapSpace, Vec<usize>, bool) {
        match &self.adjoint_source {
            None => ((*self.space).clone(), user_axes.to_vec(), false),
            Some(parent) if self.dtype() == Dtype::F64 => (
                parent
                    .adjoint_view()
                    .expect("adjoint_view is total on a tensor's own space"),
                user_axes.to_vec(),
                false,
            ),
            Some(parent) => {
                let (nout_p, nin_p) = (parent.nout(), parent.nin());
                let axes = user_axes
                    .iter()
                    .map(|&a| if a < nin_p { nout_p + a } else { a - nin_p })
                    .collect();
                ((**parent).clone(), axes, true)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn contract_impl<D: UserScalar>(
        &self,
        lhs_space: &DynamicFusionMapSpace,
        lhs_data: &[D],
        lhs_axes_seam: &[usize],
        lhs_conj: bool,
        rhs_space: &DynamicFusionMapSpace,
        rhs_data: &[D],
        rhs_axes_seam: &[usize],
        rhs_conj: bool,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        let mut state = self.rt.lock();
        // SU(N) (Generic): dedicated non-macro core/compose route — the mult-free
        // `with_rule_ctx!` binding cannot host a Generic rule. Only the
        // fully-direct GEMM (compose) route runs here: the block GEMM is
        // symmetry-agnostic and the outer-multiplicity vertices ride in the
        // fusion-tree keys, so an OM vertex is summed by the contracted-tree
        // GEMM. `contract` guarantees the operands arrive in core/compose form
        // (non-core arrangements are canonicalized by generic permutes upstream,
        // Stage B3c-2) and materializes lazy adjoints, so a conjugate flag here
        // is a routing bug, not user input.
        if self.rule == RuleKind::Su3 {
            if lhs_conj || rhs_conj {
                return Err(Error::InvalidArgument(
                    "internal: SU(N) contraction reached the seam with a conjugate flag; \
                     lazy adjoints must be materialized upstream (contract() does this)"
                        .to_string(),
                ));
            }
            let rule = Su3FusionRule::new();
            let dst_space = DynamicFusionMapSpace::contracted_generic(
                &rule,
                lhs_space,
                rhs_space,
                lhs_axes_seam,
                rhs_axes_seam,
            )?;
            let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
            D::ctx_of(&mut state.su3).tensorcontract_fusion_dyn_into_generic(
                &rule,
                &dst_space,
                &mut data,
                lhs_space,
                lhs_data,
                rhs_space,
                rhs_data,
                TensorContractSpec::new_with_conjugation(
                    lhs_axes_seam,
                    rhs_axes_seam,
                    OutputAxisOrder::identity(),
                    false,
                    false,
                ),
                D::from_real(1.0),
                D::from_real(0.0),
            )?;
            drop(state);
            return Ok(self.with(dst_space, D::lift(data)));
        }
        let (space, data) = with_rule_ctx!(self.rule, state, rule, ctxs, {
            // `dst` is the user-facing result: a lazy-adjoint operand contributes
            // its adjoint space (`self.space`/`rhs.space` already are adjoint), so
            // this matches the materialized-adjoint result exactly.
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
                lhs_space,
                lhs_data,
                rhs_space,
                rhs_data,
                TensorContractSpec::new_with_conjugation(
                    lhs_axes_seam,
                    rhs_axes_seam,
                    OutputAxisOrder::identity(),
                    lhs_conj,
                    rhs_conj,
                ),
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
        // Identity permute (new arrangement == current codomain/domain, natural
        // order) is a no-op: return the tensor unchanged, sharing its buffer,
        // instead of allocating and running a copy. Matches TensorKit's
        // `has_shared_permute(t, ...) && return t` (indexmanipulations.jl:91).
        // Only `Permute` (not `Braid`/`Transpose`) — a braid may carry a
        // nontrivial crossing even with identity axes, and transpose swaps
        // sides. Measured ~27% of itebd's permutes.
        if matches!(kind, TransformKind::Permute)
            && codomain_axes.iter().copied().eq(0..nout)
            && domain_axes.iter().copied().eq(nout..rank)
        {
            return Ok(self.clone());
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

        with_data!(self, data, self.transformed_impl(data, operation))
    }

    fn transformed_impl<D: UserScalar>(
        &self,
        src_data: &[D],
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        let mut state = self.rt.lock();
        // SU(3) (Generic): dedicated non-macro path — build the generic result
        // space and drive the non-memoized generic tree-transform. The recoupling
        // coefficient scalar is f64 for either data dtype, so the generic braid
        // math is identical to the tree-level layer this stage proved against TK.
        if self.rule == RuleKind::Su3 {
            let rule = Su3FusionRule::new();
            let dst_space = self.space.transformed_generic(&rule, &operation)?;
            let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
            D::ctx_of(&mut state.su3)
                .tree_context_mut()
                .tree_transform_dyn_into_generic(
                    &rule,
                    operation,
                    &Arc::clone(dst_space.structure()),
                    self.space.structure(),
                    &mut data,
                    src_data,
                    D::from_real(1.0),
                    D::from_real(0.0),
                )?;
            let out = (dst_space, D::lift(data));
            drop(state);
            return Ok(self.with(out.0, out.1));
        }
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
        // SU(N) (Generic): the partial-trace engine rides the mult-free
        // recoupling (`multiplicity_free_permute_tree_pair`); its generic
        // sibling is Stage B3c-3. Full trace (`tr`) IS wired generically.
        if self.rule == RuleKind::Su3 {
            return Err(Error::InvalidArgument(
                "SU(3) partial trace (trace_pairs) is not yet wired (Stage B3c-3); \
                 tr() supports SU(3)"
                    .to_string(),
            ));
        }
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
    /// to a scalar, pairing codomain leg `i` with domain leg `i`. The
    /// returned [`Scalar`] variant matches [`Self::dtype`]. Fermionic rules
    /// give the supertrace, matching TensorKit.
    pub fn tr(&self) -> Result<Scalar, Error> {
        let hom = self.space.homspace();
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::InvalidArgument(
                "tr() requires an endomorphism (domain == codomain)".to_string(),
            ));
        }
        // Block-local weighted trace (TensorKit `tr`): sum the coupled-block
        // diagonals weighted by quantum dimension, directly on the stored
        // blocks. Avoids the generic partial-trace engine's per-call recoupling
        // compile, rank-0 destination allocation, and kernel dispatch that the
        // former `trace_pairs` route paid to produce a single scalar.
        let nout = self.codomain_rank();
        // SU(N) (Generic): same block-local weighted trace through the
        // generic-dim sibling (mult-free `with_rule!` cannot host it).
        if self.rule == RuleKind::Su3 {
            let rule = Su3FusionRule::new();
            return match self.coupled_data() {
                Data::F64(data) => {
                    weighted_trace_generic(&rule, self.space.structure(), nout, data)
                        .map(|v| Scalar::F64(v.re))
                }
                Data::C64(data) => {
                    weighted_trace_generic(&rule, self.space.structure(), nout, data)
                        .map(Scalar::C64)
                }
                Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
                #[cfg(feature = "cuda")]
                Data::CudaF64(_) => Err(device_unsupported("tr()")),
            };
        }
        match self.coupled_data() {
            Data::F64(data) => with_rule!(self.rule, rule, {
                weighted_trace(rule, self.space.structure(), nout, data).map(|v| Scalar::F64(v.re))
            }),
            Data::C64(data) => with_rule!(self.rule, rule, {
                weighted_trace(rule, self.space.structure(), nout, data).map(Scalar::C64)
            }),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("tr()")),
        }
    }

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block (real scalars: transpose only, c64:
    /// entries conjugated).
    ///
    /// Lazy, exactly like TensorKit's `AdjointTensorMap`: no data is copied or
    /// conjugated here — the result shares the parent buffer and presents the
    /// adjoint coupled space (O(blocks) metadata). A contraction folds the
    /// conjugate-transpose into its GEMM; any other consumer (`data`, `svd`, …)
    /// materializes it once, on demand, via [`Self::coupled_data`].
    pub fn adjoint(&self) -> Result<Self, Error> {
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(_) = self.data.as_ref() {
            return Err(device_unsupported("adjoint()"));
        }
        // Materialize compact diagonal storage before taking a lazy adjoint, so
        // the adjoint materialization path never has to handle it (issue #55).
        if matches!(self.data.as_ref(), Data::Diagonal(_)) {
            return self.densified_if_diagonal().adjoint();
        }
        if let Some(parent_space) = &self.adjoint_source {
            // Involution: the adjoint of a lazy adjoint is the original parent,
            // rebuilt with no copy and no pending materialization.
            return Ok(Self {
                rt: self.rt.clone(),
                rule: self.rule,
                space: Arc::clone(parent_space),
                data: Arc::clone(&self.data),
                adjoint_source: None,
                materialized: OnceLock::new(),
            });
        }
        // SU(N) (Generic): the adjoint space is a pure codomain/domain swap +
        // per-block transpose (no leg bending, no recoupling), so it takes the
        // generic key-enumeration sibling. The lazy `adjoint_source` machinery
        // is symmetry-agnostic (metadata only) and shared with the mult-free
        // path below.
        let adjoint_space = if self.rule == RuleKind::Su3 {
            tenet_tensors::adjoint_space_dyn_generic(&Su3FusionRule::new(), &self.space)?
        } else {
            with_rule!(self.rule, rule, {
                tenet_tensors::adjoint_space_dyn(rule, &self.space)
            })?
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(adjoint_space),
            data: Arc::clone(&self.data),
            adjoint_source: Some(Arc::clone(&self.space)),
            materialized: OnceLock::new(),
        })
    }

    /// Frobenius norm, weighted by coupled-sector quantum dimensions
    /// (`norm(t)^2 = sum_c dim(c) * |block_c|^2`), matching TensorKit's
    /// `norm`. Always real, for both dtypes.
    pub fn norm(&self) -> Result<f64, Error> {
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.data.as_ref() {
            return Ok(self.weighted_inner_cuda(storage, storage)?.re.sqrt());
        }
        // SU(N) (Generic): dedicated non-macro path — the Frobenius norm is a
        // storage-level block sum weighted by dim(c) = sqrt_dim(c)², so it needs
        // only `GenericRigidSymbols`, no contract. Sums over OM vertices.
        if self.rule == RuleKind::Su3 {
            let value = with_data!(self, data, {
                weighted_inner_generic(&Su3FusionRule::new(), self.space.structure(), data, data)
            })?;
            return Ok(value.re.sqrt());
        }
        let value = with_data!(self, data, {
            with_rule!(self.rule, rule, {
                weighted_inner(rule, self.space.structure(), data, data)
            })
        })?;
        Ok(value.re.sqrt())
    }

    /// Entrywise infinity norm over TensorKit tensor blocks:
    /// `maximum(norm(block, Inf) for block in blocks(t))`.
    ///
    /// Julia's `norm(array, Inf)` is the maximum absolute element, including
    /// for matrices. TensorKit applies that to each block, so the coupled
    /// storage equivalent is the maximum absolute stored entry. Unlike
    /// [`Self::norm`], this is not quantum-dimension weighted.
    pub fn norm_inf(&self) -> Result<f64, Error> {
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(_) = self.data.as_ref() {
            return Err(device_unsupported("norm_inf()"));
        }
        match self.coupled_data() {
            Data::F64(data) => Ok(data.iter().map(|value| value.abs()).fold(0.0, f64::max)),
            Data::C64(data) => Ok(data.iter().map(|value| value.norm()).fold(0.0, f64::max)),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => unreachable!("returned above"),
        }
    }

    /// Returns `factor * self` (real factor, both dtypes). Use
    /// [`Self::scale_c64`] for a complex factor.
    pub fn scale(&self, factor: f64) -> Result<Self, Error> {
        // Scaling a diagonal stays diagonal (O(rank)); itebd normalizes λ this
        // way, and keeping it diagonal lets the next contract scale the bond.
        if let Data::Diagonal(diagonal) = self.data.as_ref() {
            return Ok(self.with_diagonal(diagonal.scaled(factor)));
        }
        let data = match self.coupled_data() {
            Data::F64(data) => Data::F64(data.iter().map(|&value| value * factor).collect()),
            Data::C64(data) => Data::C64(data.iter().map(|&value| value * factor).collect()),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(storage) => {
                Data::CudaF64(Arc::new(self.axpby_cuda(factor, storage, None)?))
            }
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }

    /// Returns `factor * self` for a c64 tensor. Errors with
    /// [`Error::DtypeMismatch`] on f64 tensors (widen with
    /// [`Self::to_c64`] first).
    pub fn scale_c64(&self, factor: Complex64) -> Result<Self, Error> {
        match self.coupled_data() {
            Data::C64(data) => Ok(Self {
                rt: self.rt.clone(),
                rule: self.rule,
                space: Arc::clone(&self.space),
                data: Arc::new(Data::C64(
                    data.iter().map(|&value| value * factor).collect(),
                )),
                adjoint_source: None,
                materialized: OnceLock::new(),
            }),
            Data::F64(_) => Err(Error::DtypeMismatch),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scale_c64()")),
        }
    }

    /// Returns `alpha * self + beta * other` (real coefficients, both
    /// dtypes). Both tensors must live on the same spaces (identical hom
    /// space and block layout) and store the same dtype.
    pub fn add(&self, other: &Self, alpha: f64, beta: f64) -> Result<Self, Error> {
        self.check_same_space(other)?;
        let data = match (self.coupled_data(), other.coupled_data()) {
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
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                Data::CudaF64(Arc::new(self.axpby_cuda(alpha, a, Some((beta, b)))?))
            }
            _ => return Err(Error::DtypeMismatch),
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }

    /// Returns `alpha * self + beta * other` with complex coefficients; both
    /// tensors must be c64 (widen with [`Self::to_c64`] first).
    pub fn add_c64(&self, other: &Self, alpha: Complex64, beta: Complex64) -> Result<Self, Error> {
        self.check_same_space(other)?;
        match (self.coupled_data(), other.coupled_data()) {
            (Data::C64(a), Data::C64(b)) => Ok(Self {
                rt: self.rt.clone(),
                rule: self.rule,
                space: Arc::clone(&self.space),
                data: Arc::new(Data::C64(
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| alpha * x + beta * y)
                        .collect(),
                )),
                adjoint_source: None,
                materialized: OnceLock::new(),
            }),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(_), _) | (_, Data::CudaF64(_)) => Err(device_unsupported("add_c64()")),
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Frobenius inner product `<self, other>` with `self` conjugated,
    /// weighted by coupled-sector quantum dimensions, matching TensorKit's
    /// `dot(x, y)`. The returned [`Scalar`] variant matches the operands'
    /// dtype: f64 tensors give `Scalar::F64` (the result is exactly real),
    /// so `t.inner(&t)?.re() == t.norm()?.powi(2)` up to floating-point
    /// error. Both tensors must live on the same spaces and store the same
    /// dtype.
    pub fn inner(&self, other: &Self) -> Result<Scalar, Error> {
        self.check_same_space(other)?;
        match (self.coupled_data(), other.coupled_data()) {
            (Data::F64(a), Data::F64(b)) => with_rule!(self.rule, rule, {
                weighted_inner(rule, self.space.structure(), a, b).map(|v| Scalar::F64(v.re))
            }),
            (Data::C64(a), Data::C64(b)) => with_rule!(self.rule, rule, {
                weighted_inner(rule, self.space.structure(), a, b).map(Scalar::C64)
            }),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                self.weighted_inner_cuda(a, b).map(|v| Scalar::F64(v.re))
            }
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
            data: Arc::new(D::lift(data)),
            adjoint_source: None,
            materialized: OnceLock::new(),
        }
    }

    /// Wraps a real per-sector spectrum (svd `S`, eigh `D`) as a diagonal-storage
    /// tensor: the bond space is built eagerly, but the values stay O(rank) in
    /// `Data::Diagonal` instead of a dense O(rank²) block-diagonal buffer (issue
    /// #55). `complex` picks the materialized dtype so the on-demand dense buffer
    /// matches the former dense `S` — a complex input yields a complex-valued but
    /// real-magnitude `S` (`RealC64`), a real input a real `S` (`RealF64`).
    fn from_diagonal_real_spectrum(
        &self,
        spectrum: Vec<SectorSpectrum<f64>>,
        complex: bool,
    ) -> Result<Self, Error> {
        // SU(N) (Generic): the bond space is a rank-1/rank-1 hom whose trees
        // are trivial, but the key enumeration must still be the generic one.
        let space = if self.rule == RuleKind::Su3 {
            tenet_matrixalgebra::diagonal_bond_space_generic(&Su3FusionRule::new(), &spectrum)
        } else {
            with_rule!(self.rule, rule, {
                tenet_matrixalgebra::diagonal_bond_space(rule, &spectrum)
            })
        }?;
        let data = if complex {
            DiagonalData::RealC64(spectrum)
        } else {
            DiagonalData::RealF64(spectrum)
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data: Arc::new(Data::Diagonal(data)),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }

    /// Wraps a complex per-sector spectrum (eig `D`) as diagonal storage. The
    /// general eigendecomposition is complex-valued even for real input, so `d`
    /// is always c64; the spectrum stays O(rank) in `DiagonalData::C64`. Compose
    /// densifies it (no real-spectrum scaling path), but storage is O(rank).
    fn from_diagonal_complex_spectrum(
        &self,
        spectrum: Vec<SectorSpectrum<Complex64>>,
    ) -> Result<Self, Error> {
        let space = with_rule!(self.rule, rule, {
            tenet_matrixalgebra::diagonal_bond_space(rule, &spectrum)
        })?;
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::new(space),
            data: Arc::new(Data::Diagonal(DiagonalData::C64(spectrum))),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }

    /// Rewraps `data` in this tensor's own (shared) space — for ops that leave
    /// the space unchanged (bond scaling), so the space `Arc` is reused instead
    /// of deep-cloned.
    fn with_same_space<D: UserScalar>(&self, data: Vec<D>) -> Self {
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(D::lift(data)),
            adjoint_source: None,
            materialized: OnceLock::new(),
        }
    }

    /// Reuse this tensor's space with a new diagonal payload (elementwise
    /// scale/inv/pinv/sqrt keep the same bond space).
    fn with_diagonal(&self, data: DiagonalData) -> Self {
        Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(Data::Diagonal(data)),
            adjoint_source: None,
            materialized: OnceLock::new(),
        }
    }

    /// The per-sector real spectrum of a diagonal-storage operand, if it stores
    /// one (svd `S`, eigh `D`). `None` for dense tensors and for complex-spectrum
    /// diagonals (eig `D`), which take the densifying compose/contract path.
    fn real_diagonal_spectrum(&self) -> Option<&[SectorSpectrum]> {
        match self.data.as_ref() {
            Data::Diagonal(DiagonalData::RealF64(s) | DiagonalData::RealC64(s)) => Some(s),
            _ => None,
        }
    }

    /// Whether leg `axis` (flat, codomain-first) carries a dual space, read from
    /// the hom-space legs (the same duality `compose` checks to decide which
    /// contracted legs `contract` twists).
    fn leg_is_dual(&self, axis: usize) -> bool {
        let hom = self.space.homspace();
        let nout = self.codomain_rank();
        if axis < nout {
            hom.codomain().legs()[axis].is_dual()
        } else {
            hom.domain().legs()[axis - nout].is_dual()
        }
    }

    /// The spectrum with each value multiplied by its sector's supertrace twist
    /// `θ` (±1) when `apply` — folds `contract`'s fermionic twist of a diagonal
    /// operand's dual contracted leg into the scaling instead of densifying it.
    /// Identity (a plain copy) when `!apply` or for bosonic rules (`θ = 1`).
    fn twist_folded_spectrum(
        &self,
        spectrum: &[SectorSpectrum],
        apply: bool,
    ) -> Vec<SectorSpectrum> {
        if !apply {
            return spectrum.to_vec();
        }
        with_rule!(self.rule, rule, {
            spectrum
                .iter()
                .map(|entry| {
                    let theta = rule.twist_scalar(entry.sector);
                    SectorSpectrum {
                        sector: entry.sector,
                        values: entry.values.iter().map(|&value| value * theta).collect(),
                    }
                })
                .collect()
        })
    }

    /// Scales this (dense) operand along one bond axis by `spectrum`, keeping the
    /// same space — TensorKit's `DiagonalTensorMap` `rmul!` (`axis = None`,
    /// trailing/columns) or `lmul!` (`axis = Some(0)`, leading/rows) as a
    /// block-local scaling instead of a GEMM against a materialized diagonal.
    fn scaled_axis_copy(
        &self,
        axis: Option<usize>,
        spectrum: &[SectorSpectrum],
    ) -> Result<Self, Error> {
        with_data!(self, data, {
            let mut buf = data.to_vec();
            tenet_matrixalgebra::scale_axis_by_spectrum(&self.space, &mut buf, axis, spectrum)?;
            Ok(self.with_same_space(buf))
        })
    }

    /// Compact SVD `t = u * s * vh` (MatrixAlgebraKit `svd_compact`):
    /// per coupled sector the bond is `min(rows, cols)`.
    pub fn svd_compact(&self) -> Result<(Self, Self, Self), Error> {
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.data.as_ref() {
            let out = self.svd_cuda(storage, None)?;
            return Ok((out.u, out.s, out.vh));
        }
        let complex = self.dtype() == Dtype::C64;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            // SU(N) (Generic): the block-level SVD engine is symmetry-agnostic;
            // only the factor-space builders differ (multiplicity-aware keys).
            let (u, vh, spectrum) = if self.rule == RuleKind::Su3 {
                tenet_matrixalgebra::svd_compact_factors_dyn_generic(
                    &mut *state.dense,
                    &Su3FusionRule::new(),
                    &self.space,
                    data,
                )
            } else {
                with_rule!(self.rule, rule, {
                    tenet_matrixalgebra::svd_compact_factors_dyn(
                        &mut *state.dense,
                        rule,
                        &self.space,
                        data,
                    )
                })
            }?;
            Ok((
                self.from_dyn(u),
                self.from_diagonal_real_spectrum(spectrum, complex)?,
                self.from_dyn(vh),
            ))
        })
    }

    /// Full SVD `t = u * s * vh` (MatrixAlgebraKit `svd_full`): square
    /// unitaries per sector, rectangular diagonal `s`.
    pub fn svd_full(&self) -> Result<(Self, Self, Self), Error> {
        // ponytail: the square-unitary completion path has no generic sibling
        // yet — svd_compact/svd_trunc cover the physics workflows; add the
        // `build_left_right_pair_generic` chain in B3c-3 if a caller needs it.
        if self.rule == RuleKind::Su3 {
            return Err(Error::InvalidArgument(
                "SU(3) svd_full is not yet wired (Stage B3c-3); use svd_compact or svd_trunc"
                    .to_string(),
            ));
        }
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::svd_full_dyn(&mut *state.dense, rule, &self.space, data)
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
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.data.as_ref() {
            return self.svd_cuda(storage, Some(truncation));
        }
        // Singular values are real => `s` is a real diagonal in O(rank) storage
        // (see `svd_compact`). `out.singular_values` is also returned, so it is
        // cloned into the diagonal factor.
        let complex = self.dtype() == Dtype::C64;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            // SU(N) (Generic): same engine, generic factor spaces, and the
            // integer-rounded sqrt_dim² truncation weight.
            let out = if self.rule == RuleKind::Su3 {
                tenet_matrixalgebra::svd_trunc_dyn_generic(
                    &mut *state.dense,
                    &Su3FusionRule::new(),
                    &self.space,
                    data,
                    truncation,
                )
            } else {
                with_rule!(self.rule, rule, {
                    tenet_matrixalgebra::svd_trunc_dyn(
                        &mut *state.dense,
                        rule,
                        &self.space,
                        data,
                        truncation,
                    )
                })
            }?;
            Ok(SvdTrunc {
                u: self.from_dyn(out.u),
                s: self.from_diagonal_real_spectrum(out.singular_values.clone(), complex)?,
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
            if self.rule == RuleKind::Su3 {
                tenet_matrixalgebra::svd_vals_dyn_generic(
                    &mut *state.dense,
                    &Su3FusionRule::new(),
                    &self.space,
                    data,
                )
            } else {
                with_rule!(self.rule, rule, {
                    tenet_matrixalgebra::svd_vals_dyn(&mut *state.dense, rule, &self.space, data)
                })
            }
            .map_err(Into::into)
        })
    }

    /// Compact QR `t = q * r` (MatrixAlgebraKit `qr_compact`): `q` has
    /// orthonormal columns per coupled sector.
    pub fn qr_compact(&self) -> Result<(Self, Self), Error> {
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.data.as_ref() {
            return self.qr_cuda(storage);
        }
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let (q, r) = if self.rule == RuleKind::Su3 {
                tenet_matrixalgebra::qr_compact_dyn_generic(
                    &mut *state.dense,
                    &Su3FusionRule::new(),
                    &self.space,
                    data,
                )
            } else {
                with_rule!(self.rule, rule, {
                    tenet_matrixalgebra::qr_compact_dyn(&mut *state.dense, rule, &self.space, data)
                })
            }?;
            Ok((self.from_dyn(q), self.from_dyn(r)))
        })
    }

    /// Full QR `t = q * r` (MatrixAlgebraKit `qr_full`): square `q` per
    /// sector.
    pub fn qr_full(&self) -> Result<(Self, Self), Error> {
        // ponytail: see svd_full — the square-Q completion has no generic
        // sibling yet (B3c-3); qr_compact covers left_orth and the workflows.
        if self.rule == RuleKind::Su3 {
            return Err(Error::InvalidArgument(
                "SU(3) qr_full is not yet wired (Stage B3c-3); use qr_compact".to_string(),
            ));
        }
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let (q, r) = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::qr_full_dyn(&mut *state.dense, rule, &self.space, data)
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
            let (l, q) = if self.rule == RuleKind::Su3 {
                tenet_matrixalgebra::lq_compact_dyn_generic(
                    &mut *state.dense,
                    &Su3FusionRule::new(),
                    &self.space,
                    data,
                )
            } else {
                with_rule!(self.rule, rule, {
                    tenet_matrixalgebra::lq_compact_dyn(&mut *state.dense, rule, &self.space, data)
                })
            }?;
            Ok((self.from_dyn(l), self.from_dyn(q)))
        })
    }

    /// Full LQ `t = l * q` (MatrixAlgebraKit `lq_full`): square `q` per
    /// sector.
    pub fn lq_full(&self) -> Result<(Self, Self), Error> {
        // ponytail: see svd_full/qr_full (B3c-3); lq_compact covers right_orth.
        if self.rule == RuleKind::Su3 {
            return Err(Error::InvalidArgument(
                "SU(3) lq_full is not yet wired (Stage B3c-3); use lq_compact".to_string(),
            ));
        }
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let (l, q) = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::lq_full_dyn(&mut *state.dense, rule, &self.space, data)
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
                tenet_matrixalgebra::left_null_dyn(&mut *state.dense, rule, &self.space, data)
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
                tenet_matrixalgebra::right_null_dyn(&mut *state.dense, rule, &self.space, data)
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
                &mut *state.dense,
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
                &mut *state.dense,
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
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.data.as_ref() {
            let out = self.eigh_cuda(storage, None)?;
            return Ok((out.d, out.v));
        }
        // eigh eigenvalues are real, so `d` is a real diagonal (`RealC64` for
        // c64 input). Build it as O(rank) diagonal storage from the spectrum;
        // `eigh_full_dyn` returns only the spectrum + eigenvectors (no dense d),
        // so nothing O(rank²) is materialized and discarded here (#56 item N).
        let complex = self.dtype() == Dtype::C64;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eigh_full_dyn(&mut *state.dense, rule, &self.space, data)
            })?;
            Ok((
                self.from_diagonal_real_spectrum(out.eigenvalues, complex)?,
                self.from_dyn(out.v),
            ))
        })
    }

    /// Truncated Hermitian eigendecomposition (MatrixAlgebraKit
    /// `eigh_trunc`); see [`EighTrunc`].
    pub fn eigh_trunc(&self, truncation: &Truncation) -> Result<EighTrunc, Error> {
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.data.as_ref() {
            return self.eigh_cuda(storage, Some(truncation));
        }
        // Real eigenvalues => real diagonal `d` in O(rank) storage (see
        // `eigh_full`). `out.eigenvalues` is also returned to the caller, so it
        // is cloned into the diagonal factor.
        let complex = self.dtype() == Dtype::C64;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        with_data!(self, data, {
            let out = with_rule!(self.rule, rule, {
                tenet_matrixalgebra::eigh_trunc_dyn(
                    &mut *state.dense,
                    rule,
                    &self.space,
                    data,
                    truncation,
                )
            })?;
            Ok(EighTrunc {
                d: self.from_diagonal_real_spectrum(out.eigenvalues.clone(), complex)?,
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
                tenet_matrixalgebra::eigh_vals_dyn(&mut *state.dense, rule, &self.space, data)
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
                tenet_matrixalgebra::eig_full_dyn(&mut *state.dense, rule, &self.space, data)
            })?;
            Ok((
                self.from_diagonal_complex_spectrum(out.eigenvalues)?,
                self.from_dyn(out.v),
            ))
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
                    &mut *state.dense,
                    rule,
                    &self.space,
                    data,
                    truncation,
                )
            })?;
            Ok(EigTrunc {
                d: self.from_diagonal_complex_spectrum(out.eigenvalues.clone())?,
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
                tenet_matrixalgebra::eig_vals_dyn(&mut *state.dense, rule, &self.space, data)
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
            tenet_matrixalgebra::exp_dyn(
                &mut *state.dense,
                D::ctx_of(ctxs),
                rule,
                &self.space,
                data,
            )
        })?;
        Ok(self.from_dyn(out))
    }

    /// True inverse of a full-rank endomorphism (MatrixAlgebraKit-style
    /// `inv`); fails when any coupled block is rank-deficient at working
    /// precision.
    pub fn inv(&self) -> Result<Self, Error> {
        // A diagonal inverse is elementwise (O(rank)), not a block inversion;
        // keep it diagonal so the next contract still scales the bond.
        if let Data::Diagonal(diagonal) = self.data.as_ref() {
            return Ok(self.with_diagonal(diagonal.try_recip()?));
        }
        with_data!(self, data, self.inv_impl(data))
    }

    fn inv_impl<D: UserScalar>(&self, data: &[D]) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let out = with_rule_ctx!(self.rule, state, rule, ctxs, {
            tenet_matrixalgebra::inv_dyn(
                &mut *state.dense,
                D::ctx_of(ctxs),
                rule,
                &self.space,
                data,
            )
        })?;
        Ok(self.from_dyn(out))
    }

    /// Moore-Penrose pseudo-inverse `t^+ = v s^+ u^H` (MatrixAlgebraKit
    /// `pinv`) with an `rcond * sigma_max` cutoff on the singular values.
    pub fn pinv(&self, rcond: f64) -> Result<Self, Error> {
        // A diagonal pseudo-inverse is an elementwise cutoff+reciprocal on the
        // spectrum (O(rank)) — its own singular values are |entry| — so skip the
        // SVD and keep it diagonal (itebd's `l_out.pinv` fires this).
        if let Data::Diagonal(diagonal) = self.data.as_ref() {
            return Ok(self.with_diagonal(diagonal.pinv(rcond)));
        }
        with_data!(self, data, self.pinv_impl(data, rcond))
    }

    fn pinv_impl<D: UserScalar>(&self, data: &[D], rcond: f64) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let out = with_rule_ctx!(self.rule, state, rule, ctxs, {
            tenet_matrixalgebra::pinv_dyn(
                &mut *state.dense,
                D::ctx_of(ctxs),
                rule,
                &self.space,
                data,
                rcond,
            )
        })?;
        Ok(self.from_dyn(out))
    }

    /// Elementwise square root of a diagonal bond tensor, i.e. the
    /// TensorKit 0.17 `sqrt(::DiagonalTensorMap)` idiom
    /// (`tensors/diagonal.jl:384-390`: `sqrt.(d.data)` on the diagonal)
    /// used to split singular values as `√S · √S = S` in Vidal-gauge /
    /// gate-application updates.
    ///
    /// The receiver must be a diagonal bond tensor as produced by the
    /// factorization paths ([`Self::svd_trunc`]'s `s`, eigenvalue factors):
    /// one codomain leg equal to the one domain leg and every stored block
    /// diagonal (off-diagonal entries exactly zero). Anything else — the
    /// analog of calling this on a non-`DiagonalTensorMap` — is an
    /// [`Error::InvalidArgument`]. For f64 tensors every diagonal entry must
    /// be `>= 0` (Julia's real `sqrt` throws a `DomainError` there too;
    /// convert with [`Self::to_c64`] first for the complex branch); c64
    /// tensors take the principal complex square root.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 2), (1, 2)]);
    /// let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 7)?;
    /// let s = t.svd_trunc(&Truncation::Full)?.s;
    /// let sqrt_s = s.sqrt()?;
    /// let composed = sqrt_s.compose(&sqrt_s)?;
    /// let max_err = composed
    ///     .data()
    ///     .iter()
    ///     .zip(s.data())
    ///     .map(|(a, b)| (a - b).abs())
    ///     .fold(0.0f64, f64::max);
    /// assert!(max_err < 1e-12);
    /// assert!(t.sqrt().is_err()); // not a diagonal bond tensor
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn sqrt(&self) -> Result<Self, Error> {
        let hom = self.space.homspace();
        if hom.codomain().len() != 1
            || hom.domain().len() != 1
            || hom.codomain().legs() != hom.domain().legs()
        {
            return Err(Error::InvalidArgument(
                "sqrt requires a diagonal bond tensor `[v] <- [v]` (equal single \
                 codomain and domain legs), like the `s` factor of svd_trunc"
                    .to_string(),
            ));
        }
        // Diagonal storage: sqrt is elementwise on the spectrum (O(rank)) and
        // stays diagonal, so √S · √S = S keeps scaling the bond.
        if let Data::Diagonal(diagonal) = self.data.as_ref() {
            return Ok(self.with_diagonal(diagonal.try_sqrt()?));
        }
        let data = match self.coupled_data() {
            Data::F64(data) => Data::F64(sqrt_diagonal_impl(&self.space, data, &|value| {
                if value < 0.0 {
                    Err(Error::InvalidArgument(format!(
                        "sqrt of a negative diagonal entry {value}; convert to c64 \
                         with to_c64() for the complex square root"
                    )))
                } else {
                    Ok(value.sqrt())
                }
            })?),
            Data::C64(data) => Data::C64(sqrt_diagonal_impl(&self.space, data, &|value| {
                Ok(value.sqrt())
            })?),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("sqrt")),
        };
        Ok(Self {
            rt: self.rt.clone(),
            rule: self.rule,
            space: Arc::clone(&self.space),
            data: Arc::new(data),
            adjoint_source: None,
            materialized: OnceLock::new(),
        })
    }
}

impl TensorExecutionContext {
    /// Whether `dst` has the exact host layout required by a contraction.
    pub fn can_contract_overwrite_into(
        &self,
        dst: &Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<bool, Error> {
        if lhs.check_same_world(rhs).is_err()
            || dst.check_same_world(lhs).is_err()
            || dst.check_same_world(rhs).is_err()
            || dst.placement() != Placement::Host
            || lhs.adjoint_source.is_some()
            || rhs.adjoint_source.is_some()
            || dst.adjoint_source.is_some()
            || matches!(lhs.data.as_ref(), Data::Diagonal(_))
            || matches!(rhs.data.as_ref(), Data::Diagonal(_))
            || matches!(dst.data.as_ref(), Data::Diagonal(_))
            || Arc::ptr_eq(&dst.data, &lhs.data)
            || Arc::ptr_eq(&dst.data, &rhs.data)
        {
            return Ok(false);
        }
        open_axes(lhs_axes, lhs.rank())?;
        open_axes(rhs_axes, rhs.rank())?;
        let expected = if lhs.rule == RuleKind::Su3 {
            DynamicFusionMapSpace::contracted_generic(
                &Su3FusionRule::new(),
                &lhs.space,
                &rhs.space,
                lhs_axes,
                rhs_axes,
            )?
        } else {
            with_rule!(lhs.rule, rule, {
                DynamicFusionMapSpace::contracted(rule, &lhs.space, &rhs.space, lhs_axes, rhs_axes)
            })?
        };
        Ok(dst.validate_exact_destination_space(&expected).is_ok()
            && Arc::strong_count(&dst.data) == 1)
    }

    /// Whether `dst` has the exact host layout required by a permutation.
    pub fn can_permute_overwrite_into(
        &self,
        dst: &Tensor,
        src: &Tensor,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<bool, Error> {
        if dst.check_same_world(src).is_err()
            || dst.placement() != Placement::Host
            || src.adjoint_source.is_some()
            || dst.adjoint_source.is_some()
            || matches!(src.data.as_ref(), Data::Diagonal(_))
            || matches!(dst.data.as_ref(), Data::Diagonal(_))
            || Arc::ptr_eq(&dst.data, &src.data)
        {
            return Ok(false);
        }
        let operation = TreeTransformOperation::permute(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
        );
        let expected = if src.rule == RuleKind::Su3 {
            src.space
                .transformed_generic(&Su3FusionRule::new(), &operation)?
        } else {
            with_rule!(src.rule, rule, src.space.transformed(rule, &operation))?
        };
        Ok(dst.validate_exact_destination_space(&expected).is_ok()
            && Arc::strong_count(&dst.data) == 1)
    }

    /// Overwrites an exact-layout dense host destination with
    /// `alpha * contract(lhs, rhs)`. Validation errors leave `dst` unchanged;
    /// an error returned after backend execution begins may leave it partially
    /// overwritten.
    pub fn contract_overwrite_into(
        &mut self,
        dst: &mut Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        alpha: Scalar,
    ) -> Result<(), Error> {
        lhs.check_same_world(rhs)?;
        dst.validate_host_destination(lhs)?;
        dst.validate_host_destination(rhs)?;
        if lhs.adjoint_source.is_some()
            || rhs.adjoint_source.is_some()
            || matches!(lhs.data.as_ref(), Data::Diagonal(_))
            || matches!(rhs.data.as_ref(), Data::Diagonal(_))
        {
            return Err(Error::InvalidArgument(
                "dynamic destination contraction requires ordinary dense inputs".to_string(),
            ));
        }
        if lhs_axes.len() != rhs_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "contracted axis lists differ in length: {} vs {}",
                lhs_axes.len(),
                rhs_axes.len()
            )));
        }
        open_axes(lhs_axes, lhs.rank())?;
        open_axes(rhs_axes, rhs.rank())?;

        let expected = if lhs.rule == RuleKind::Su3 {
            DynamicFusionMapSpace::contracted_generic(
                &Su3FusionRule::new(),
                &lhs.space,
                &rhs.space,
                lhs_axes,
                rhs_axes,
            )?
        } else {
            with_rule!(lhs.rule, rule, {
                DynamicFusionMapSpace::contracted(rule, &lhs.space, &rhs.space, lhs_axes, rhs_axes)
            })?
        };
        dst.validate_exact_destination_space(&expected)?;

        match (
            Arc::get_mut(&mut dst.data),
            lhs.data.as_ref(),
            rhs.data.as_ref(),
            alpha,
        ) {
            (
                Some(Data::F64(dst_data)),
                Data::F64(lhs_data),
                Data::F64(rhs_data),
                Scalar::F64(alpha),
            ) => dispatch_contract_into(
                self, dst.rule, &expected, dst_data, lhs, lhs_data, rhs, rhs_data, lhs_axes,
                rhs_axes, alpha, 0.0,
            ),
            (
                Some(Data::C64(dst_data)),
                Data::C64(lhs_data),
                Data::C64(rhs_data),
                Scalar::C64(alpha),
            ) => dispatch_contract_into(
                self,
                dst.rule,
                &expected,
                dst_data,
                lhs,
                lhs_data,
                rhs,
                rhs_data,
                lhs_axes,
                rhs_axes,
                alpha,
                Complex64::new(0.0, 0.0),
            ),
            (None, _, _, _) => Err(Error::InvalidArgument(
                "destination storage must be uniquely owned".to_string(),
            )),
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Overwrites an exact-layout dense host destination with
    /// `alpha * permute(src)`. Validation errors leave `dst` unchanged; an
    /// error returned after backend execution begins may leave it partially
    /// overwritten.
    pub fn permute_overwrite_into(
        &mut self,
        dst: &mut Tensor,
        src: &Tensor,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        alpha: Scalar,
    ) -> Result<(), Error> {
        dst.validate_host_destination(src)?;
        if src.adjoint_source.is_some() || matches!(src.data.as_ref(), Data::Diagonal(_)) {
            return Err(Error::InvalidArgument(
                "dynamic destination permutation requires an ordinary dense input".to_string(),
            ));
        }
        let operation = TreeTransformOperation::permute(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
        );
        let expected = if src.rule == RuleKind::Su3 {
            src.space
                .transformed_generic(&Su3FusionRule::new(), &operation)?
        } else {
            with_rule!(src.rule, rule, src.space.transformed(rule, &operation))?
        };
        dst.validate_exact_destination_space(&expected)?;

        match (Arc::get_mut(&mut dst.data), src.data.as_ref(), alpha) {
            (Some(Data::F64(dst_data)), Data::F64(src_data), Scalar::F64(alpha)) => {
                dispatch_permute_into(
                    self, dst.rule, operation, &expected, dst_data, src, src_data, alpha, 0.0,
                )
            }
            (Some(Data::C64(dst_data)), Data::C64(src_data), Scalar::C64(alpha)) => {
                dispatch_permute_into(
                    self,
                    dst.rule,
                    operation,
                    &expected,
                    dst_data,
                    src,
                    src_data,
                    alpha,
                    Complex64::new(0.0, 0.0),
                )
            }
            (None, _, _) => Err(Error::InvalidArgument(
                "destination storage must be uniquely owned".to_string(),
            )),
            _ => Err(Error::DtypeMismatch),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn contract_into_with_rule<R, D, Key>(
    contexts: &mut Ctxs<Key>,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs: &Tensor,
    lhs_data: &[D],
    rhs: &Tensor,
    rhs_data: &[D],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    alpha: D,
    beta: D,
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = Key>,
    D: UserScalar,
    Key: Clone + Eq + Hash + Send + Sync + 'static,
{
    D::ctx_of(contexts)
        .tensorcontract_fusion_dyn_into(
            rule,
            dst_space,
            dst_data,
            &lhs.space,
            lhs_data,
            &rhs.space,
            rhs_data,
            TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes),
            alpha,
            beta,
        )
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_contract_into<D: UserScalar>(
    context: &mut TensorExecutionContext,
    rule_kind: RuleKind,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs: &Tensor,
    lhs_data: &[D],
    rhs: &Tensor,
    rhs_data: &[D],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    alpha: D,
    beta: D,
) -> Result<(), Error> {
    match rule_kind {
        RuleKind::U1 => contract_into_with_rule(
            &mut context.u1,
            &tenet_core::U1FusionRule,
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            lhs_axes,
            rhs_axes,
            alpha,
            beta,
        ),
        RuleKind::Z2 => contract_into_with_rule(
            &mut context.z2,
            &tenet_core::Z2FusionRule,
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            lhs_axes,
            rhs_axes,
            alpha,
            beta,
        ),
        RuleKind::FZ2 => contract_into_with_rule(
            &mut context.fz2,
            &tenet_core::FermionParityFusionRule,
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            lhs_axes,
            rhs_axes,
            alpha,
            beta,
        ),
        RuleKind::SU2 => contract_into_with_rule(
            &mut context.su2,
            &tenet_core::SU2FusionRule,
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            lhs_axes,
            rhs_axes,
            alpha,
            beta,
        ),
        RuleKind::U1FZ2 => contract_into_with_rule(
            &mut context.u1_fz2,
            &tenet_core::ProductFusionRule::<
                tenet_core::U1FusionRule,
                tenet_core::FermionParityFusionRule,
            >::new(
                tenet_core::U1FusionRule,
                tenet_core::FermionParityFusionRule,
            ),
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            lhs_axes,
            rhs_axes,
            alpha,
            beta,
        ),
        RuleKind::FZ2U1SU2 => contract_into_with_rule(
            &mut context.fz2_u1_su2,
            &tenet_core::ProductFusionRule::<
                tenet_core::ProductFusionRule<
                    tenet_core::FermionParityFusionRule,
                    tenet_core::U1FusionRule,
                >,
                tenet_core::SU2FusionRule,
            >::new(
                tenet_core::ProductFusionRule::<
                    tenet_core::FermionParityFusionRule,
                    tenet_core::U1FusionRule,
                >::new(
                    tenet_core::FermionParityFusionRule,
                    tenet_core::U1FusionRule,
                ),
                tenet_core::SU2FusionRule,
            ),
            dst_space,
            dst_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            lhs_axes,
            rhs_axes,
            alpha,
            beta,
        ),
        RuleKind::Su3 => D::ctx_of(&mut context.su3)
            .tensorcontract_fusion_dyn_into_generic(
                &Su3FusionRule::new(),
                dst_space,
                dst_data,
                &lhs.space,
                lhs_data,
                &rhs.space,
                rhs_data,
                TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes),
                alpha,
                beta,
            )
            .map_err(Into::into),
    }
}

#[allow(clippy::too_many_arguments)]
fn permute_into_with_rule<R, D, Key>(
    contexts: &mut Ctxs<Key>,
    rule: &R,
    operation: TreeTransformOperation,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src: &Tensor,
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = Key>,
    D: UserScalar,
    Key: Clone + Eq + Hash + Send + Sync + 'static,
{
    D::ctx_of(contexts)
        .tree_context_mut()
        .tree_transform_dyn_into(
            rule,
            operation,
            dst_space.structure(),
            src.space.structure(),
            dst_data,
            src_data,
            alpha,
            beta,
        )
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_permute_into<D: UserScalar>(
    context: &mut TensorExecutionContext,
    rule_kind: RuleKind,
    operation: TreeTransformOperation,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src: &Tensor,
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), Error> {
    macro_rules! apply {
        ($contexts:expr, $rule:expr) => {
            permute_into_with_rule(
                $contexts, $rule, operation, dst_space, dst_data, src, src_data, alpha, beta,
            )
        };
    }
    match rule_kind {
        RuleKind::U1 => apply!(&mut context.u1, &tenet_core::U1FusionRule),
        RuleKind::Z2 => apply!(&mut context.z2, &tenet_core::Z2FusionRule),
        RuleKind::FZ2 => apply!(&mut context.fz2, &tenet_core::FermionParityFusionRule),
        RuleKind::SU2 => apply!(&mut context.su2, &tenet_core::SU2FusionRule),
        RuleKind::U1FZ2 => apply!(
            &mut context.u1_fz2,
            &tenet_core::ProductFusionRule::<
                tenet_core::U1FusionRule,
                tenet_core::FermionParityFusionRule,
            >::new(
                tenet_core::U1FusionRule,
                tenet_core::FermionParityFusionRule,
            )
        ),
        RuleKind::FZ2U1SU2 => apply!(
            &mut context.fz2_u1_su2,
            &tenet_core::ProductFusionRule::<
                tenet_core::ProductFusionRule<
                    tenet_core::FermionParityFusionRule,
                    tenet_core::U1FusionRule,
                >,
                tenet_core::SU2FusionRule,
            >::new(
                tenet_core::ProductFusionRule::<
                    tenet_core::FermionParityFusionRule,
                    tenet_core::U1FusionRule,
                >::new(
                    tenet_core::FermionParityFusionRule,
                    tenet_core::U1FusionRule,
                ),
                tenet_core::SU2FusionRule,
            )
        ),
        RuleKind::Su3 => D::ctx_of(&mut context.su3)
            .tree_context_mut()
            .tree_transform_dyn_into_generic(
                &Su3FusionRule::new(),
                operation,
                dst_space.structure(),
                src.space.structure(),
                dst_data,
                src_data,
                alpha,
                beta,
            )
            .map_err(Into::into),
    }
}

/// TensorKit `A * B` as an operator: `&a * &b` is exactly
/// [`Tensor::compose`] (categorical composition / `mul!` on coupled blocks,
/// **no** fermionic supertrace twist — see the fermionic-semantics note on
/// [`Tensor::compose`]).
///
/// # Panics
///
/// Panics on any composition error (space/rule/runtime/dtype mismatch),
/// printing both hom spaces — mirroring TensorKit, where `A * B` throws
/// `SpaceMismatch` (nalgebra and ndarray panic on shape mismatch the same
/// way). Use [`Tensor::compose`] directly when you want the `Result`.
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        match self.compose(rhs) {
            Ok(out) => out,
            Err(err) => panic!(
                "Tensor * Tensor (compose) failed: {err}\n  lhs: {:?} <- {:?}\n  rhs: {:?} <- {:?}",
                self.codomain_spaces(),
                self.domain_spaces(),
                rhs.codomain_spaces(),
                rhs.domain_spaces(),
            ),
        }
    }
}

/// Takes the elementwise square root of the diagonal of every `[n, n]`
/// block, verifying that all off-diagonal entries are exactly zero (the
/// storage invariant of the diagonal bond tensors built by the
/// factorization paths).
fn sqrt_diagonal_impl<D: UserScalar + PartialEq>(
    space: &DynamicFusionMapSpace,
    data: &[D],
    sqrt_of: &dyn Fn(D) -> Result<D, Error>,
) -> Result<Vec<D>, Error> {
    let zero = D::from_real(0.0);
    let mut out = vec![zero; data.len()];
    let structure = space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        debug_assert_eq!(shape.len(), 2);
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                let position = offset + row * strides[0] + col * strides[1];
                if row == col {
                    out[position] = sqrt_of(data[position])?;
                } else if data[position] != zero {
                    return Err(Error::InvalidArgument(format!(
                        "sqrt requires a diagonal bond tensor, but block {:?} has a \
                         nonzero off-diagonal entry at ({row}, {col})",
                        block.key()
                    )));
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CUDA device paths (f64 only).
//
// The user-layer storage is always the TensorKit-equivalent coupled-sector
// matrix layout, so every coupled sector is one contiguous column-major
// matrix region of the flat device buffer. All device work is expressed as
// (a) per-sector cuSOLVER decompositions on those regions and (b) region
// GEMMs against small host-built selector matrices (identity / prefix /
// sign / permutation) that also perform factor assembly into freshly
// allocated coupled-layout buffers. Only scalars ever cross PCIe implicitly:
// per-sector reduction partials, singular/eigen values and R diagonals
// (truncation and gauge decisions are host scalar logic).
// ---------------------------------------------------------------------------

/// One coupled sector of the packed coupled-sector matrix layout: a
/// contiguous column-major `rows x cols` region at `offset`, with the
/// per-fusion-tree row/column extents needed for factor assembly.
struct SectorRegion {
    /// Coupled sector (`None` only for degenerate vacuum-coupled trees).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    coupled: Option<SectorId>,
    rows: usize,
    cols: usize,
    offset: usize,
    /// `(codomain tree, row offset, row count)` in row order.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    row_trees: Vec<(FusionTreeKey, usize, usize)>,
    /// `(domain tree, column offset, column count)` in column order.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    col_trees: Vec<(FusionTreeKey, usize, usize)>,
}

#[cfg(feature = "cuda")]
fn dense_err(err: tenet_dense::DenseError) -> Error {
    Error::from(OperationError::Dense(err))
}

#[cfg(feature = "cuda")]
fn require_cuda(cuda: Option<&mut CudaDenseContext>) -> Result<&mut CudaDenseContext, Error> {
    cuda.ok_or_else(|| {
        Error::InvalidArgument(
            "this runtime was built without a CUDA device; use \
             Runtime::builder().cuda(device)"
                .to_string(),
        )
    })
}

fn internal_layout_error(what: &str) -> Error {
    Error::InvalidArgument(format!(
        "internal coupled-layout invariant violated ({what}); please report this"
    ))
}

/// Enumerates the coupled-sector matrix regions of a coupled-layout block
/// structure and verifies that every fusion-tree block addresses exactly the
/// packed column-major sector matrix. The structural constructors and the
/// device paths rely on these offsets, so any other layout is an explicit
/// error, never silent misaddressing.
fn sector_regions(structure: &BlockStructure, nout: usize) -> Result<Vec<SectorRegion>, Error> {
    let mut regions: Vec<SectorRegion> = Vec::new();
    let mut block_index = 0usize;
    let mut next_offset = 0usize;
    while block_index < structure.block_count() {
        let first = structure.block(block_index)?;
        let BlockKey::FusionTree(first_key) = first.key() else {
            return Err(internal_layout_error("non-fusion-tree block layout"));
        };
        let coupled = first_key.codomain_tree().coupled();

        // Pass 1: extents of this sector's matrix and per-tree offsets.
        let mut row_trees: Vec<(FusionTreeKey, usize, usize)> = Vec::new();
        let mut col_trees: Vec<(FusionTreeKey, usize, usize)> = Vec::new();
        let mut rows = 0usize;
        let mut cols = 0usize;
        let mut end = block_index;
        while end < structure.block_count() {
            let block = structure.block(end)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(internal_layout_error("non-fusion-tree block layout"));
            };
            if key.codomain_tree().coupled() != coupled {
                break;
            }
            let shape = block.shape();
            let sub_rows: usize = shape[..nout].iter().product();
            let sub_cols: usize = shape[nout..].iter().product();
            if !row_trees
                .iter()
                .any(|(tree, _, _)| tree == key.codomain_tree())
            {
                row_trees.push((key.codomain_tree().clone(), rows, sub_rows));
                rows += sub_rows;
            }
            if !col_trees
                .iter()
                .any(|(tree, _, _)| tree == key.domain_tree())
            {
                col_trees.push((key.domain_tree().clone(), cols, sub_cols));
                cols += sub_cols;
            }
            end += 1;
        }

        // Pass 2: verify packed addressing for every block of the sector.
        let offset = next_offset;
        let mut covered = 0usize;
        for index in block_index..end {
            let block = structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!("checked in pass 1");
            };
            let row_offset = row_trees
                .iter()
                .find(|(tree, _, _)| tree == key.codomain_tree())
                .map(|(_, offset, _)| *offset)
                .expect("recorded in pass 1");
            let col_offset = col_trees
                .iter()
                .find(|(tree, _, _)| tree == key.domain_tree())
                .map(|(_, offset, _)| *offset)
                .expect("recorded in pass 1");
            let shape = block.shape();
            let mut expected_strides = Vec::with_capacity(shape.len());
            let mut stride = 1usize;
            for axis in 0..nout {
                expected_strides.push(stride);
                stride *= shape[axis];
            }
            let mut stride = rows;
            for axis in nout..shape.len() {
                expected_strides.push(stride);
                stride *= shape[axis];
            }
            if block.strides() != expected_strides.as_slice()
                || block.offset() != offset + row_offset + rows * col_offset
            {
                return Err(internal_layout_error("non-packed coupled-sector layout"));
            }
            covered += shape.iter().product::<usize>();
        }
        if covered != rows * cols {
            return Err(internal_layout_error("coupled sector with layout holes"));
        }

        regions.push(SectorRegion {
            coupled,
            rows,
            cols,
            offset,
            row_trees,
            col_trees,
        });
        next_offset = offset + rows * cols;
        block_index = end;
    }
    Ok(regions)
}

#[cfg(feature = "cuda")]
fn coupled_sector_of<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    region: &SectorRegion,
    rule: &R,
) -> SectorId {
    region.coupled.unwrap_or_else(|| rule.vacuum())
}

#[cfg(feature = "cuda")]
fn find_source<'a>(
    regions: &'a [SectorRegion],
    target: &SectorRegion,
) -> Result<(usize, &'a SectorRegion), Error> {
    regions
        .iter()
        .enumerate()
        .find(|(_, region)| region.coupled == target.coupled)
        .ok_or_else(|| internal_layout_error("factor bond sector missing in the source tensor"))
}

/// Shared host-side truncation decision (exactly the host cores' flow:
/// `select_truncation` over quantum-dimension-weighted magnitudes, kept
/// prefixes, empty sectors dropped, discarded weighted 2-norm as `error`;
/// a no-op decision keeps the full factorization with `error == 0`).
#[cfg(feature = "cuda")]
fn decide_kept<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    rule: &R,
    spectra: &[SectorSpectrum],
    truncation: Option<&Truncation>,
) -> (Vec<SectorSpectrum>, f64) {
    let Some(truncation) = truncation else {
        return (spectra.to_vec(), 0.0);
    };
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.abs()).collect())
        .collect();
    let weighted: Vec<WeightedSpectrum<'_>> = spectra
        .iter()
        .zip(&magnitudes)
        .map(|(entry, values)| WeightedSpectrum {
            weight: rule.dim_scalar(entry.sector),
            values,
        })
        .collect();
    let decision = select_truncation(&weighted, truncation);
    if spectra
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return (spectra.to_vec(), 0.0);
    }
    let mut kept = spectra.to_vec();
    for (entry, &count) in kept.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    kept.retain(|entry| !entry.values.is_empty());
    (kept, decision.error)
}

/// Uploads a small host-built selector matrix (`rows x cols`, column-major,
/// zero except `entries`) used by the assembly GEMMs.
#[cfg(feature = "cuda")]
fn upload_selector(
    cuda: &mut CudaDenseContext,
    rows: usize,
    cols: usize,
    entries: impl Iterator<Item = (usize, usize, f64)>,
) -> Result<CudaStorage, Error> {
    let mut data = vec![0.0; rows * cols];
    for (row, col, value) in entries {
        data[row + rows * col] = value;
    }
    CudaStorage::upload(cuda, &data).map_err(Error::from)
}

/// Writes `factor_rows x kept` slices of `factor * selector` into the target
/// sector region of a left factor (`codomain <- bond`), one GEMM per
/// codomain tree so correctness never relies on tree enumeration order
/// matching between the source and factor spaces.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn assemble_left_factor(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage,
    target: &SectorRegion,
    source: &SectorRegion,
    factor: &CudaDenseStorage,
    k_full: usize,
    selector: &CudaStorage,
    kept: usize,
) -> Result<(), Error> {
    for (tree, dst_row, sub_rows) in &target.row_trees {
        if *sub_rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees
            .iter()
            .find(|(source_tree, _, _)| source_tree == tree)
            .map(|(_, offset, _)| *offset)
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        cuda_gemm_region_into(
            cuda,
            &mut dst.0,
            target.offset + dst_row,
            target.rows,
            factor,
            src_row,
            source.rows,
            &selector.0,
            0,
            k_full,
            *sub_rows,
            k_full,
            kept,
            1.0,
            0.0,
        )
        .map_err(dense_err)?;
    }
    Ok(())
}

/// Writes `kept x factor_cols` slices of `selector * factor` into the target
/// sector region of a right factor (`bond <- domain`), one GEMM per domain
/// tree.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn assemble_right_factor(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage,
    target: &SectorRegion,
    source: &SectorRegion,
    selector: &CudaStorage,
    kept: usize,
    k_full: usize,
    factor: &CudaDenseStorage,
) -> Result<(), Error> {
    for (tree, dst_col, sub_cols) in &target.col_trees {
        if *sub_cols == 0 {
            continue;
        }
        let src_col = source
            .col_trees
            .iter()
            .find(|(source_tree, _, _)| source_tree == tree)
            .map(|(_, offset, _)| *offset)
            .ok_or_else(|| internal_layout_error("domain tree missing in the source sector"))?;
        cuda_gemm_region_into(
            cuda,
            &mut dst.0,
            target.offset + target.rows * dst_col,
            target.rows,
            &selector.0,
            0,
            kept,
            factor,
            k_full * src_col,
            k_full,
            kept,
            k_full,
            *sub_cols,
            1.0,
            0.0,
        )
        .map_err(dense_err)?;
    }
    Ok(())
}

/// Fills the diagonal of a coupled-layout `W <- W` buffer from per-sector
/// spectra, mirroring the host `diagonal_bond_tensor_dyn`.
#[cfg(feature = "cuda")]
fn fill_diagonal_values(
    structure: &BlockStructure,
    data: &mut [f64],
    spectra: &[SectorSpectrum],
) -> Result<(), Error> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree
            .codomain_tree()
            .coupled()
            .unwrap_or_else(|| tree.codomain_tree().uncoupled()[0]);
        let Some(entry) = spectra.iter().find(|entry| entry.sector == sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            data[offset + position * (strides[0] + strides[1])] = entry.values[position];
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
impl Tensor {
    /// Device weighted Frobenius inner product: one `[1, len] x [len, 1]`
    /// region GEMM per coupled sector into a device partials buffer, then a
    /// single download of the per-sector scalars, weighted by quantum
    /// dimensions on the host.
    fn weighted_inner_cuda(&self, a: &CudaStorage, b: &CudaStorage) -> Result<Complex64, Error> {
        let regions = sector_regions(self.space.structure(), self.space.nout())?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let mut partials = CudaStorage::upload(cuda, &vec![0.0; regions.len().max(1)])?;
        for (index, region) in regions.iter().enumerate() {
            let len = region.rows * region.cols;
            if len == 0 {
                continue;
            }
            cuda_gemm_region_into(
                cuda,
                &mut partials.0,
                index,
                1,
                &a.0,
                region.offset,
                1,
                &b.0,
                region.offset,
                len,
                1,
                len,
                1,
                1.0,
                0.0,
            )
            .map_err(dense_err)?;
        }
        let values = partials.download(cuda)?;
        drop(guard);
        let total = with_rule!(self.rule, rule, {
            regions
                .iter()
                .zip(&values)
                .map(|(region, &value)| value * rule.dim_scalar(coupled_sector_of(region, rule)))
                .sum::<f64>()
        });
        Ok(Complex64::new(total, 0.0))
    }

    /// Device `alpha * x (+ beta * y)`: whole-buffer region GEMVs against a
    /// `[1, 1]` ones operand (tenferro has no axpby/scale primitive; this
    /// stays on the one proven dot-general seam).
    fn axpby_cuda(
        &self,
        alpha: f64,
        x: &CudaStorage,
        other: Option<(f64, &CudaStorage)>,
    ) -> Result<CudaStorage, Error> {
        let len = TensorStorage::<f64>::len(x);
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let ones = CudaStorage::upload(cuda, &[1.0])?;
        // ponytail: destination allocated by uploading host zeros, same seam
        // as the device contraction path; replace with a device alloc if the
        // upload ever shows up in profiles.
        let mut dst = CudaStorage::upload(cuda, &vec![0.0; len])?;
        if len > 0 {
            cuda_gemm_region_into(
                cuda, &mut dst.0, 0, len, &x.0, 0, len, &ones.0, 0, 1, len, 1, 1, alpha, 0.0,
            )
            .map_err(dense_err)?;
            if let Some((beta, y)) = other {
                cuda_gemm_region_into(
                    cuda, &mut dst.0, 0, len, &y.0, 0, len, &ones.0, 0, 1, len, 1, 1, beta, 1.0,
                )
                .map_err(dense_err)?;
            }
        }
        drop(guard);
        Ok(dst)
    }

    /// Device SVD: per-sector cuSOLVER SVD on the packed regions, values
    /// downloaded for the (shared, host-side) truncation decision, factors
    /// assembled on device through prefix selectors. `truncation: None` is
    /// `svd_compact`.
    fn svd_cuda(
        &self,
        storage: &CudaStorage,
        truncation: Option<&Truncation>,
    ) -> Result<SvdTrunc, Error> {
        let regions = sector_regions(self.space.structure(), self.space.nout())?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let out = with_rule!(self.rule, rule, {
            let mut spectra: Vec<SectorSpectrum> = Vec::with_capacity(regions.len());
            let mut factors: Vec<Option<(CudaDenseStorage, CudaDenseStorage)>> =
                Vec::with_capacity(regions.len());
            for region in &regions {
                let sector = coupled_sector_of(region, rule);
                if region.rows == 0 || region.cols == 0 {
                    spectra.push(SectorSpectrum {
                        sector,
                        values: Vec::new(),
                    });
                    factors.push(None);
                    continue;
                }
                let (u, s, vt) =
                    cuda_svd_region(cuda, &storage.0, region.offset, region.rows, region.cols)
                        .map_err(dense_err)?;
                spectra.push(SectorSpectrum { sector, values: s });
                factors.push(Some((u, vt)));
            }
            let (kept_spectra, error) = decide_kept(rule, &spectra, truncation);

            let hom = self.space.homspace();
            let bond_leg = SectorLeg::new(
                kept_spectra
                    .iter()
                    .map(|entry| (entry.sector, entry.values.len())),
                false,
            );
            let u_space = build_space(
                rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                    FusionProductSpace::new([bond_leg.clone()]),
                ),
            )?;
            let s_space = build_space(
                rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond_leg.clone()]),
                    FusionProductSpace::new([bond_leg.clone()]),
                ),
            )?;
            let vh_space = build_space(
                rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond_leg]),
                    FusionProductSpace::new(hom.domain().legs().iter().cloned()),
                ),
            )?;

            let mut u_data = CudaStorage::upload(cuda, &vec![0.0; u_space.required_len()?])?;
            for target in &sector_regions(u_space.structure(), u_space.nout())? {
                let kept = target.cols;
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((u_dev, _)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let k_full = source.rows.min(source.cols);
                let selector = upload_selector(cuda, k_full, kept, (0..kept).map(|j| (j, j, 1.0)))?;
                assemble_left_factor(
                    cuda,
                    &mut u_data,
                    target,
                    source,
                    u_dev,
                    k_full,
                    &selector,
                    kept,
                )?;
            }

            let mut vh_data = CudaStorage::upload(cuda, &vec![0.0; vh_space.required_len()?])?;
            for target in &sector_regions(vh_space.structure(), vh_space.nout())? {
                let kept = target.rows;
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((_, vt_dev)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let k_full = source.rows.min(source.cols);
                let selector = upload_selector(cuda, kept, k_full, (0..kept).map(|j| (j, j, 1.0)))?;
                assemble_right_factor(
                    cuda,
                    &mut vh_data,
                    target,
                    source,
                    &selector,
                    kept,
                    k_full,
                    vt_dev,
                )?;
            }

            let mut s_host = vec![0.0; s_space.required_len()?];
            fill_diagonal_values(s_space.structure(), &mut s_host, &kept_spectra)?;
            let s_data = CudaStorage::upload(cuda, &s_host)?;

            Ok::<_, Error>(SvdTrunc {
                u: self.with(u_space, Data::CudaF64(Arc::new(u_data))),
                s: self.with(s_space, Data::CudaF64(Arc::new(s_data))),
                vh: self.with(vh_space, Data::CudaF64(Arc::new(vh_data))),
                singular_values: kept_spectra,
                error,
            })
        })?;
        drop(guard);
        Ok(out)
    }

    /// Device compact QR with the host's positive-diagonal gauge: only `R`'s
    /// diagonal crosses to the host (sign decisions), the gauge is applied by
    /// the sign-selector assembly GEMMs.
    fn qr_cuda(&self, storage: &CudaStorage) -> Result<(Self, Self), Error> {
        let regions = sector_regions(self.space.structure(), self.space.nout())?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let out = with_rule!(self.rule, rule, {
            let mut factors: Vec<Option<(CudaDenseStorage, CudaDenseStorage, Vec<f64>)>> =
                Vec::with_capacity(regions.len());
            let mut bond_pairs: Vec<(SectorId, usize)> = Vec::with_capacity(regions.len());
            for region in &regions {
                let sector = coupled_sector_of(region, rule);
                if region.rows == 0 || region.cols == 0 {
                    bond_pairs.push((sector, 0));
                    factors.push(None);
                    continue;
                }
                let (q, r, diag) =
                    cuda_qr_region(cuda, &storage.0, region.offset, region.rows, region.cols)
                        .map_err(dense_err)?;
                // Positive-diagonal gauge (host `positive_diagonal_gauge`,
                // real scalars): flip where R's diagonal is negative, leave
                // exact zeros untouched.
                let signs: Vec<f64> = diag
                    .iter()
                    .map(|&value| if value < 0.0 { -1.0 } else { 1.0 })
                    .collect();
                bond_pairs.push((sector, region.rows.min(region.cols)));
                factors.push(Some((q, r, signs)));
            }

            let hom = self.space.homspace();
            let bond_leg = SectorLeg::new(bond_pairs.iter().copied(), false);
            let q_space = build_space(
                rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                    FusionProductSpace::new([bond_leg.clone()]),
                ),
            )?;
            let r_space = build_space(
                rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond_leg]),
                    FusionProductSpace::new(hom.domain().legs().iter().cloned()),
                ),
            )?;

            let mut q_data = CudaStorage::upload(cuda, &vec![0.0; q_space.required_len()?])?;
            for target in &sector_regions(q_space.structure(), q_space.nout())? {
                let kept = target.cols;
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((q_dev, _, signs)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let selector = upload_selector(
                    cuda,
                    kept,
                    kept,
                    signs.iter().enumerate().map(|(j, &sign)| (j, j, sign)),
                )?;
                assemble_left_factor(
                    cuda,
                    &mut q_data,
                    target,
                    source,
                    q_dev,
                    kept,
                    &selector,
                    kept,
                )?;
            }

            let mut r_data = CudaStorage::upload(cuda, &vec![0.0; r_space.required_len()?])?;
            for target in &sector_regions(r_space.structure(), r_space.nout())? {
                let kept = target.rows;
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((_, r_dev, signs)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let selector = upload_selector(
                    cuda,
                    kept,
                    kept,
                    signs.iter().enumerate().map(|(j, &sign)| (j, j, sign)),
                )?;
                assemble_right_factor(
                    cuda,
                    &mut r_data,
                    target,
                    source,
                    &selector,
                    kept,
                    kept,
                    r_dev,
                )?;
            }

            Ok::<_, Error>((
                self.with(q_space, Data::CudaF64(Arc::new(q_data))),
                self.with(r_space, Data::CudaF64(Arc::new(r_data))),
            ))
        })?;
        drop(guard);
        Ok(out)
    }

    /// Device Hermitian eigendecomposition: eigenvalues cross to the host
    /// (descending-by-magnitude ordering and truncation are host decisions),
    /// eigenvectors are reordered / truncated on device via a permutation
    /// selector. `truncation: None` is `eigh_full`.
    fn eigh_cuda(
        &self,
        storage: &CudaStorage,
        truncation: Option<&Truncation>,
    ) -> Result<EighTrunc, Error> {
        {
            let hom = self.space.homspace();
            if hom.codomain() != hom.domain() {
                return Err(Error::InvalidArgument(
                    "eigh requires an endomorphism (codomain == domain)".to_string(),
                ));
            }
        }
        let regions = sector_regions(self.space.structure(), self.space.nout())?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let out = with_rule!(self.rule, rule, {
            let mut spectra: Vec<SectorSpectrum> = Vec::with_capacity(regions.len());
            let mut factors: Vec<Option<(CudaDenseStorage, Vec<usize>)>> =
                Vec::with_capacity(regions.len());
            for region in &regions {
                let sector = coupled_sector_of(region, rule);
                let n = region.rows;
                if n == 0 {
                    spectra.push(SectorSpectrum {
                        sector,
                        values: Vec::new(),
                    });
                    factors.push(None);
                    continue;
                }
                let (values, vectors) =
                    cuda_eigh_region(cuda, &storage.0, region.offset, n).map_err(dense_err)?;
                // Host ordering contract: descending by |eigenvalue|,
                // stable on ties (mirrors `eigh_full_dyn`).
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| {
                    values[b]
                        .abs()
                        .partial_cmp(&values[a].abs())
                        .expect("finite eigenvalues")
                        .then(a.cmp(&b))
                });
                let sorted: Vec<f64> = order.iter().map(|&index| values[index]).collect();
                spectra.push(SectorSpectrum {
                    sector,
                    values: sorted,
                });
                factors.push(Some((vectors, order)));
            }
            let (kept_spectra, error) = decide_kept(rule, &spectra, truncation);

            let hom = self.space.homspace();
            let bond_leg = SectorLeg::new(
                kept_spectra
                    .iter()
                    .map(|entry| (entry.sector, entry.values.len())),
                false,
            );
            let v_space = build_space(
                rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                    FusionProductSpace::new([bond_leg.clone()]),
                ),
            )?;
            let d_space = build_space(
                rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond_leg.clone()]),
                    FusionProductSpace::new([bond_leg]),
                ),
            )?;

            let mut v_data = CudaStorage::upload(cuda, &vec![0.0; v_space.required_len()?])?;
            for target in &sector_regions(v_space.structure(), v_space.nout())? {
                let kept = target.cols;
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((v_dev, order)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let n = source.rows;
                let selector = upload_selector(
                    cuda,
                    n,
                    kept,
                    order
                        .iter()
                        .take(kept)
                        .enumerate()
                        .map(|(j, &original)| (original, j, 1.0)),
                )?;
                assemble_left_factor(cuda, &mut v_data, target, source, v_dev, n, &selector, kept)?;
            }

            let mut d_host = vec![0.0; d_space.required_len()?];
            fill_diagonal_values(d_space.structure(), &mut d_host, &kept_spectra)?;
            let d_data = CudaStorage::upload(cuda, &d_host)?;

            Ok::<_, Error>(EighTrunc {
                d: self.with(d_space, Data::CudaF64(Arc::new(d_data))),
                v: self.with(v_space, Data::CudaF64(Arc::new(v_data))),
                eigenvalues: kept_spectra,
                error,
            })
        })?;
        drop(guard);
        Ok(out)
    }
}

/// Concise, explicit tensor constructors on the runtime itself: `rt.zeros(…)`
/// is exactly `Tensor::zeros(&rt, …)`, one per common builder. Use these when
/// juggling several runtimes (each call names its own); use the argument-free
/// free functions ([`zeros`], [`rand`], …) when one default runtime suffices.
impl Runtime {
    /// [`Tensor::zeros`] on this runtime.
    pub fn zeros<'a, C, D>(&self, dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::zeros(self, dtype, codomain, domain)
    }

    /// [`Tensor::rand`] on this runtime.
    pub fn rand<'a, C, D>(&self, dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::rand(self, dtype, codomain, domain)
    }

    /// [`Tensor::rand_with_seed`] on this runtime.
    pub fn rand_with_seed<'a, C, D>(
        &self,
        dtype: Dtype,
        codomain: C,
        domain: D,
        seed: u64,
    ) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::rand_with_seed(self, dtype, codomain, domain, seed)
    }

    /// [`Tensor::id`] on this runtime.
    pub fn id<'a, S>(&self, dtype: Dtype, spaces: S) -> Result<Tensor, Error>
    where
        S: IntoIterator<Item = &'a Space>,
    {
        Tensor::id(self, dtype, spaces)
    }
}

/// Zero tensor on the calling thread's default runtime — [`Tensor::zeros`]
/// without the runtime argument. Set the default once with
/// [`crate::set_default_runtime`] / [`crate::default!`]; errors if none is set.
pub fn zeros<'a, C, D>(dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
where
    C: IntoIterator<Item = &'a Space>,
    D: IntoIterator<Item = &'a Space>,
{
    Tensor::zeros(&crate::runtime::default_runtime()?, dtype, codomain, domain)
}

/// Random tensor on the calling thread's default runtime; see [`zeros`] and
/// [`Tensor::rand`].
pub fn rand<'a, C, D>(dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
where
    C: IntoIterator<Item = &'a Space>,
    D: IntoIterator<Item = &'a Space>,
{
    Tensor::rand(&crate::runtime::default_runtime()?, dtype, codomain, domain)
}

/// Seeded random tensor on the calling thread's default runtime; see [`zeros`]
/// and [`Tensor::rand_with_seed`].
pub fn rand_with_seed<'a, C, D>(
    dtype: Dtype,
    codomain: C,
    domain: D,
    seed: u64,
) -> Result<Tensor, Error>
where
    C: IntoIterator<Item = &'a Space>,
    D: IntoIterator<Item = &'a Space>,
{
    Tensor::rand_with_seed(
        &crate::runtime::default_runtime()?,
        dtype,
        codomain,
        domain,
        seed,
    )
}

/// Identity tensor on the calling thread's default runtime; see [`zeros`] and
/// [`Tensor::id`].
pub fn id<'a, S>(dtype: Dtype, spaces: S) -> Result<Tensor, Error>
where
    S: IntoIterator<Item = &'a Space>,
{
    Tensor::id(&crate::runtime::default_runtime()?, dtype, spaces)
}
