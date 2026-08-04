//! User-layer symmetric tensor: dynamic rank, rule-erased, runtime-carrying.
//!
//! A [`Tensor`] stores one enum-erased provider-bound dynamic fusion space plus
//! flat scalar storage (`f64` or `Complex64`, chosen at construction) in the
//! TensorKit-equivalent coupled-sector matrix layout. Not every payload is
//! that dense buffer: a spectrum factor (SVD `s`, eigh/eig `d`) holds compact
//! [`Data::Diagonal`] per-sector values — `Σ_c k_c` instead of `Σ_c k_c²`,
//! TensorKit's `DiagonalTensorMap` — materialized into the dense layout only
//! at the `coupled_data` boundary, and with the `cuda` feature a payload may
//! live in device storage instead of a host `Vec`. On host-dense payloads
//! [`Tensor::adjoint`] is lazy: it returns a view sharing the parent buffer
//! in O(1), and consumers not lowered for the view materialize one shared
//! owned adjoint on demand (compact spectra are adjointed directly, without
//! the view; a device payload's view stays metadata-lazy, and a consumer not
//! lowered for it errors rather than materializing on the host).
//! The scalar type is
//! erased behind an internal storage enum; rank is fully dynamic (no ceiling),
//! matching TensorKit's `tensorcontract!`. CPU operations briefly acquire a
//! per-operation context and/or executor lease, then run with that resource
//! exclusively without holding the [`Runtime`]'s coarse shared-state lock.
//! They dispatch on the stored rule and dtype once per call (never per block)
//! and forward the bound authority to the expert layer.

use std::borrow::Cow;
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use num_complex::Complex64;
use smallvec::SmallVec;
#[cfg(test)]
use tenet_core::FusionTreePairKey;
use tenet_core::{
    validate_unit_layout_correspondence_checked, BlockKey, BlockStructure, CheckedFusionSpaceError,
    CoupledSectorRegion, FusionProductSpace, FusionRule, FusionTreeHomSpace,
    FusionTreePairOrientation, LoweredMultiplicityFreeAlgebra, MultiplicityFreeRigidSymbols,
    OrientedFusionTreeHomSpace, Placement, SectorId, UnitLegInsertion,
};
#[cfg(feature = "cuda")]
use tenet_core::{SectorLeg, TensorStorage};
#[cfg(feature = "cuda")]
use tenet_dense::{cuda_eigh_region, cuda_gemm_region_into, CudaDenseContext, CudaDenseStorage};
#[cfg(feature = "cuda")]
use tenet_matrixalgebra::validate_hermitian_regions;
use tenet_matrixalgebra::{
    BoundDynFactor, BoundDynamicTensorRef, FactorScalar, SectorSpectrum, SvdTruncFactorsDyn,
    Truncation,
};
#[cfg(feature = "cuda")]
use tenet_tensors::cuda::{CudaStorage, CudaStorageGemm};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, OperationError, OutputAxisOrder,
    OwnedCatC64Source as CatC64Source, TensorContractSpec, TreeTransformOperation,
};

use crate::error::Error;
use crate::runtime::Runtime;
#[cfg(test)]
use crate::runtime::TensorExecutionContext;
use crate::space::{Fz2U1Su2Rule, Space, U1Fz2Rule, UserRuleContext};
pub(crate) use crate::tensor_core::internal_layout_error;
use crate::tensor_core::{
    pow_by_squaring, tensorcontract_owned_multiplicity_free, tensorproduct_owned_multiplicity_free,
    tree_transform_owned_multiplicity_free,
};
use crate::typed::{
    absorb_mapped, apply_fill, cat_homspace, check_flip_layout_identity, compile_cat_plan,
    coupled_region_pow_sum, flip_block_factor, flip_toggled_homspace,
    logical_adjoint_axes_to_parent, lower_adjoint_tree_transform_operation,
    map_checked_unit_layout_error, reject_unbraided_nonunit_legs, scale_blocks_impl,
    sector_regions, twist_block_factor, twist_is_identity_over_blocks, validate_axis_permutation,
    validate_contracted_axes, validate_norm_p, weighted_inner, weighted_trace, with_planar_axes,
    CatCopyPlan, CatOperandLayout, CatSide, Fill, PlanarRequestKind, ScalarOps, TensorOrientation,
};
#[cfg(feature = "cuda")]
use crate::typed::{
    assemble_left_factor, assemble_right_factor, cuda_qr_region, cuda_svd_region, decide_kept,
    dense_err, fill_diagonal_values, upload_selector,
};
#[cfg(test)]
use crate::typed::{cat_logical_block_key, coupled_region_inner, CAT_RESULT_LAYOUT_BUILDS};

mod diagonal;
use diagonal::{
    axpby_dense_c64, axpby_dense_real, axpy_diagonal_into, compact_inner_with_weight,
    dense_inner_with_weight, oriented_dense_inner_with_weight,
};

#[cfg(test)]
thread_local! {
    static ORDERED_CONTRACT_FUSED_ROUTE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static SELECTED_RESULT_LAYOUT_BUILDS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static DIAGONAL_RESULT_LAYOUT_BUILDS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static TWIST_CALLS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn observe_ordered_contract_fused_route() {
    ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| {
        if observation.get().is_some() {
            observation.set(Some(true));
        }
    });
}

#[cfg(test)]
fn observe_selected_result_layout_build() {
    SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| {
        if let Some(builds) = observation.get() {
            observation.set(Some(builds + 1));
        }
    });
}

#[cfg(test)]
fn observe_diagonal_result_layout_build() {
    DIAGONAL_RESULT_LAYOUT_BUILDS.with(|observation| {
        if let Some(builds) = observation.get() {
            observation.set(Some(builds + 1));
        }
    });
}

#[cfg(test)]
fn observe_twist_call() {
    TWIST_CALLS.with(|observation| {
        if let Some(calls) = observation.get() {
            observation.set(Some(calls + 1));
        }
    });
}

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
    /// block-diagonal matrix (issue #55). Storage-local operations consume this
    /// representation directly; dense-only operations use
    /// [`Tensor::coupled_data`] as the explicit materialization boundary.
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

    /// TensorKit `tr` on compact diagonal storage: sum the reduced diagonal
    /// values with their quantum dimensions. Why not reuse `trace_pairs`: that
    /// contraction API intentionally inserts orientation-dependent fermionic
    /// twists, while matrix trace uses TensorKit's positive trace formalism.
    fn ordinary_trace_with(&self, dim: impl Fn(SectorId) -> f64) -> Complex64 {
        let trace_real = |spectra: &[SectorSpectrum<f64>]| {
            spectra
                .iter()
                .map(|entry| entry.values.iter().sum::<f64>() * dim(entry.sector))
                .sum::<f64>()
        };
        match self {
            Self::RealF64(spectra) | Self::RealC64(spectra) => {
                Complex64::new(trace_real(spectra), 0.0)
            }
            Self::C64(spectra) => spectra
                .iter()
                .map(|entry| entry.values.iter().sum::<Complex64>() * dim(entry.sector))
                .sum(),
        }
    }

    fn ordinary_trace<R>(&self, rule: &R) -> Complex64
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        self.ordinary_trace_with(|sector| rule.dim_scalar(sector))
    }

    fn elementwise_product(&self, rhs: &Self) -> Option<Self> {
        fn multiply<V: Copy>(
            lhs: &[SectorSpectrum<V>],
            rhs: &[SectorSpectrum<V>],
            mul: impl Fn(V, V) -> V,
        ) -> Option<Vec<SectorSpectrum<V>>> {
            if lhs.len() != rhs.len() {
                return None;
            }
            lhs.iter()
                .zip(rhs)
                .map(|(lhs, rhs)| {
                    if lhs.sector != rhs.sector || lhs.values.len() != rhs.values.len() {
                        return None;
                    }
                    Some(SectorSpectrum {
                        sector: lhs.sector,
                        values: lhs
                            .values
                            .iter()
                            .copied()
                            .zip(rhs.values.iter().copied())
                            .map(|(lhs, rhs)| mul(lhs, rhs))
                            .collect(),
                    })
                })
                .collect()
        }

        fn real_complex_product(
            real: &[SectorSpectrum<f64>],
            complex: &[SectorSpectrum<Complex64>],
        ) -> Option<Vec<SectorSpectrum<Complex64>>> {
            if real.len() != complex.len() {
                return None;
            }
            real.iter()
                .zip(complex)
                .map(|(real, complex)| {
                    if real.sector != complex.sector || real.values.len() != complex.values.len() {
                        return None;
                    }
                    Some(SectorSpectrum {
                        sector: real.sector,
                        values: real
                            .values
                            .iter()
                            .copied()
                            .zip(complex.values.iter().copied())
                            .map(|(real, complex)| real * complex)
                            .collect(),
                    })
                })
                .collect()
        }

        match (self, rhs) {
            (Self::RealF64(lhs), Self::RealF64(rhs)) => {
                multiply(lhs, rhs, |lhs, rhs| lhs * rhs).map(Self::RealF64)
            }
            (Self::RealC64(lhs), Self::RealC64(rhs)) => {
                multiply(lhs, rhs, |lhs, rhs| lhs * rhs).map(Self::RealC64)
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                multiply(lhs, rhs, |lhs, rhs| lhs * rhs).map(Self::C64)
            }
            (Self::RealC64(real), Self::C64(complex)) => {
                real_complex_product(real, complex).map(Self::C64)
            }
            (Self::C64(complex), Self::RealC64(real)) => {
                real_complex_product(real, complex).map(Self::C64)
            }
            _ => None,
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

    /// TensorKit `_norm(blocks(t), p, 0)`'s finite-`p` accumulator on compact
    /// storage (`linalg.jl:262-270`): `Σ_c dim(c) · Σ_i |λ_i|^p`.
    ///
    /// Reads the `Σ_c k_c` stored values only. Materializing first would visit
    /// `Σ_c k_c²` scalars and add `k_c² − k_c` exact zeros per sector, which
    /// contribute nothing for any `p > 0` — the whole point of the compact arm.
    fn abs_pow_sum_with(&self, p: f64, dim: impl Fn(SectorId) -> f64) -> f64 {
        fn accumulate<V: Copy>(
            spectra: &[SectorSpectrum<V>],
            p: f64,
            dim: impl Fn(SectorId) -> f64,
            magnitude: impl Fn(V) -> f64,
        ) -> f64 {
            spectra
                .iter()
                .map(|entry| {
                    dim(entry.sector)
                        * entry
                            .values
                            .iter()
                            .map(|&value| magnitude(value).powf(p))
                            .sum::<f64>()
                })
                .sum()
        }
        match self {
            Self::RealF64(spectra) | Self::RealC64(spectra) => {
                accumulate(spectra, p, dim, f64::abs)
            }
            Self::C64(spectra) => accumulate(spectra, p, dim, |value: Complex64| value.norm()),
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

    /// Compact `one(d)`: preserve each sector and degeneracy, replacing values
    /// by the multiplicative identity.
    fn ones_like(&self) -> DiagonalData {
        fn real(spectra: &[SectorSpectrum<f64>]) -> Vec<SectorSpectrum<f64>> {
            spectra
                .iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: vec![1.0; entry.values.len()],
                })
                .collect()
        }
        match self {
            Self::RealF64(spectra) => Self::RealF64(real(spectra)),
            Self::RealC64(spectra) => Self::RealC64(real(spectra)),
            Self::C64(spectra) => Self::C64(
                spectra
                    .iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: vec![Complex64::new(1.0, 0.0); entry.values.len()],
                    })
                    .collect(),
            ),
        }
    }

    /// Compact exact zero: keep sector keys, dtype, and degeneracies without
    /// reading the stored values (in particular NaN and infinities).
    fn zeros_like(&self) -> DiagonalData {
        fn real(spectra: &[SectorSpectrum<f64>]) -> Vec<SectorSpectrum<f64>> {
            spectra
                .iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: vec![0.0; entry.values.len()],
                })
                .collect()
        }
        match self {
            Self::RealF64(spectra) => Self::RealF64(real(spectra)),
            Self::RealC64(spectra) => Self::RealC64(real(spectra)),
            Self::C64(spectra) => Self::C64(
                spectra
                    .iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: vec![Complex64::new(0.0, 0.0); entry.values.len()],
                    })
                    .collect(),
            ),
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

    /// Elementwise exponential (TensorKit `exp(::DiagonalTensorMap)`,
    /// `tensors/diagonal.jl:383-390`). Variant-preserving, `RealC64` included:
    /// `exp` of a real value is real, so a real spectrum inside a c64 tensor
    /// stays real instead of promoting to `C64` and doubling the stored payload
    /// for no information.
    fn exp(&self) -> DiagonalData {
        fn map<V: Copy>(
            spectra: &[SectorSpectrum<V>],
            exp: impl Fn(V) -> V,
        ) -> Vec<SectorSpectrum<V>> {
            spectra
                .iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: entry.values.iter().map(|&value| exp(value)).collect(),
                })
                .collect()
        }
        match self {
            DiagonalData::RealF64(spectra) => DiagonalData::RealF64(map(spectra, f64::exp)),
            DiagonalData::RealC64(spectra) => DiagonalData::RealC64(map(spectra, f64::exp)),
            DiagonalData::C64(spectra) => {
                DiagonalData::C64(map(spectra, |value: Complex64| value.exp()))
            }
        }
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

fn map_trace_preflight_error(error: OperationError) -> Error {
    match error {
        OperationError::Core(error) => Error::Core(Box::new(error)),
        OperationError::FusionAlgebra(error) => Error::FusionAlgebra(error),
        other => Error::from(other),
    }
}

/// Legacy-only glue between typed scalar operations and erased [`Data`].
#[allow(private_bounds)]
pub trait UserScalar: crate::typed::ScalarOps {
    fn lift(data: Vec<Self>) -> Data;
    fn data_slice(data: &Data) -> Option<&[Self]>;
}

impl UserScalar for f64 {
    fn lift(data: Vec<Self>) -> Data {
        Data::F64(data)
    }

    fn data_slice(data: &Data) -> Option<&[Self]> {
        match data {
            Data::F64(data) => Some(data),
            _ => None,
        }
    }
}

impl UserScalar for Complex64 {
    fn lift(data: Vec<Self>) -> Data {
        Data::C64(data)
    }

    fn data_slice(data: &Data) -> Option<&[Self]> {
        match data {
            Data::C64(data) => Some(data),
            _ => None,
        }
    }
}

/// Dispatches once on the stored dtype of `$tensor`, binding `$data` to the
/// typed data vector in both arms; `$body` must be dtype-generic (the expert
/// entry points are generic over the scalar).
macro_rules! with_data {
    ($tensor:expr, $data:ident, $body:expr) => {
        match $tensor.coupled_data()? {
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
    /// Left isometry `U` (codomain legs `<- bond`).
    pub u: Tensor,
    /// Diagonal singular-value tensor `S` (`bond <- bond`).
    pub s: Tensor,
    /// Right isometry `V†` (`bond <- domain legs`).
    pub vh: Tensor,
    /// Kept singular values per coupled sector.
    pub singular_values: Vec<SectorSpectrum>,
    /// Quantum-dimension-weighted 2-norm of the discarded singular values.
    pub error: f64,
}

/// Result of [`Tensor::eigh_trunc`]: `t ~ v * d * v^H` with the truncated
/// bond; `error` is the quantum-dimension-weighted 2-norm of the discarded
/// eigenvalues.
#[derive(Clone, Debug)]
pub struct EighTrunc {
    /// Diagonal eigenvalue tensor `D` (`bond <- bond`), real for Hermitian input.
    pub d: Tensor,
    /// Eigenvector isometry `V` (codomain legs `<- bond`).
    pub v: Tensor,
    /// Kept eigenvalues per coupled sector.
    pub eigenvalues: Vec<SectorSpectrum>,
    /// Quantum-dimension-weighted 2-norm of the discarded eigenvalues.
    pub error: f64,
}

/// Result of [`Tensor::eig_trunc`]: `t ~ v * d * v^-1` with the truncated
/// bond. `d` and `v` are always c64 (the general eigendecomposition is
/// complex-valued even for real input); `error` is the
/// quantum-dimension-weighted 2-norm of the discarded `|eigenvalues|`.
#[derive(Clone, Debug)]
pub struct EigTrunc {
    /// Diagonal eigenvalue tensor `D` (`bond <- bond`), always c64.
    pub d: Tensor,
    /// Eigenvector tensor `V` (codomain legs `<- bond`), always c64.
    pub v: Tensor,
    /// Kept (complex) eigenvalues per coupled sector.
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
    /// Quantum-dimension-weighted 2-norm of the discarded `|eigenvalues|`.
    pub error: f64,
}

struct CatDescriptor {
    space: UserBoundSpace,
    plan: CatCopyPlan,
}

impl CatDescriptor {
    fn new(
        lhs: &TensorBody,
        rhs: &TensorBody,
        axis: usize,
        side: CatSide,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, Error> {
        let lhs = TensorMetadataView {
            body: lhs,
            orientation: TensorOrientation::Owned,
        };
        let rhs = TensorMetadataView {
            body: rhs,
            orientation: TensorOrientation::Owned,
        };
        let source_regions = [
            sector_regions(lhs.body.space.structure(), lhs.body.space.nout())?,
            sector_regions(rhs.body.space.structure(), rhs.body.space.nout())?,
        ];
        Self::compile(&lhs, &rhs, source_regions, axis, side, homspace)?.ok_or_else(|| {
            internal_layout_error("owned concatenation unexpectedly declined its layout")
        })
    }

    fn try_new_oriented(
        lhs: &TensorMetadataView<'_>,
        rhs: &TensorMetadataView<'_>,
        axis: usize,
        side: CatSide,
        homspace: FusionTreeHomSpace,
    ) -> Result<Option<Self>, Error> {
        // Why not use materialized_body: it constructs the complete adjoint
        // block grid that this borrowed cat route avoids.
        let source_regions = [
            lhs.body
                .space
                .structure()
                .coupled_sector_regions(lhs.body.space.nout())?,
            rhs.body
                .space
                .structure()
                .coupled_sector_regions(rhs.body.space.nout())?,
        ];
        let [Some(lhs_regions), Some(rhs_regions)] = source_regions else {
            return Ok(None);
        };
        let Some(descriptor) =
            Self::compile(lhs, rhs, [lhs_regions, rhs_regions], axis, side, homspace)?
        else {
            return Ok(None);
        };
        descriptor.plan.preflight([
            lhs.body.space.structure().required_len()?,
            rhs.body.space.structure().required_len()?,
        ])?;
        Ok(Some(descriptor))
    }

    /// Erased-facade wrapper of [`compile_cat_plan`]: derives the erased
    /// destination space from the homspace and lowers the two operand views to
    /// structure-level layouts. Orientation (adjoint-view) handling lives in
    /// the layouts; the plan itself is facade-agnostic.
    fn compile(
        lhs: &TensorMetadataView<'_>,
        rhs: &TensorMetadataView<'_>,
        source_regions: [Arc<[CoupledSectorRegion]>; 2],
        axis: usize,
        side: CatSide,
        homspace: FusionTreeHomSpace,
    ) -> Result<Option<Self>, Error> {
        let space = lhs.body.space.from_homspace(homspace)?;
        let [lhs_regions, rhs_regions] = source_regions;
        let Some(plan) = compile_cat_plan(
            space.structure(),
            space.nout(),
            [
                CatOperandLayout::from_regions(
                    lhs.body.space.structure(),
                    lhs_regions,
                    lhs.orientation,
                    lhs.nout(),
                    lhs.body.space.nout(),
                    lhs.rank(),
                ),
                CatOperandLayout::from_regions(
                    rhs.body.space.structure(),
                    rhs_regions,
                    rhs.orientation,
                    rhs.nout(),
                    rhs.body.space.nout(),
                    rhs.rank(),
                ),
            ],
            axis,
            side,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(Self { space, plan }))
    }
}

fn execute_cat_data(descriptor: &CatDescriptor, lhs: &Data, rhs: &Data) -> Result<Data, Error> {
    match (lhs, rhs) {
        (Data::F64(lhs), Data::F64(rhs)) => Ok(Data::F64(descriptor.plan.execute(lhs, rhs)?)),
        (Data::C64(lhs), Data::C64(rhs)) => Ok(Data::C64(descriptor.plan.execute(lhs, rhs)?)),
        (Data::F64(lhs), Data::C64(rhs)) => Ok(Data::C64(
            descriptor
                .plan
                .execute_c64(CatC64Source::F64(lhs), CatC64Source::C64(rhs))?,
        )),
        (Data::C64(lhs), Data::F64(rhs)) => Ok(Data::C64(
            descriptor
                .plan
                .execute_c64(CatC64Source::C64(lhs), CatC64Source::F64(rhs))?,
        )),
        (Data::Diagonal(_), _) | (_, Data::Diagonal(_)) => Err(internal_layout_error(
            "compact diagonal reached cat execution",
        )),
        #[cfg(feature = "cuda")]
        (Data::CudaF64(_), _) | (_, Data::CudaF64(_)) => Err(Error::UnsupportedOnDevice(
            "catdomain/catcodomain has no device implementation yet".to_string(),
        )),
    }
}

fn validate_absorb_layout(structure: &BlockStructure, data: &Data) -> Result<(), Error> {
    let actual = match data {
        Data::F64(data) => data.len(),
        Data::C64(data) => data.len(),
        Data::Diagonal(_) => {
            return Err(internal_layout_error(
                "compact diagonal reached absorb execution",
            ));
        }
        #[cfg(feature = "cuda")]
        Data::CudaF64(_) => {
            return Err(Error::UnsupportedOnDevice(
                "Tensor::absorb has no device implementation yet".to_string(),
            ));
        }
    };
    if structure.required_len()? != actual {
        return Err(internal_layout_error(
            "absorb block layout does not cover scalar storage",
        ));
    }
    Ok(())
}

fn execute_absorb_data(
    destination_structure: &BlockStructure,
    destination: &Data,
    source_structure: &BlockStructure,
    source: &Data,
) -> Result<Data, Error> {
    match (destination, source) {
        (Data::F64(destination), Data::F64(source)) => {
            let mut output = destination.clone();
            absorb_mapped(
                destination_structure,
                &mut output,
                source_structure,
                source,
                Ok,
            )?;
            Ok(Data::F64(output))
        }
        (Data::C64(destination), Data::C64(source)) => {
            let mut output = destination.clone();
            absorb_mapped(
                destination_structure,
                &mut output,
                source_structure,
                source,
                Ok,
            )?;
            Ok(Data::C64(output))
        }
        (Data::C64(destination), Data::F64(source)) => {
            let mut output = destination.clone();
            absorb_mapped(
                destination_structure,
                &mut output,
                source_structure,
                source,
                |value| Ok(Complex64::new(value, 0.0)),
            )?;
            Ok(Data::C64(output))
        }
        (Data::F64(destination), Data::C64(source)) => {
            let mut output = destination.clone();
            absorb_mapped(
                destination_structure,
                &mut output,
                source_structure,
                source,
                |value| {
                    if value.im == 0.0 {
                        Ok(value.re)
                    } else {
                        Err(Error::InexactScalarConversion {
                            operation: "Tensor::absorb",
                            from: Dtype::C64,
                            to: Dtype::F64,
                        })
                    }
                },
            )?;
            Ok(Data::F64(output))
        }
        (Data::Diagonal(_), _) | (_, Data::Diagonal(_)) => Err(internal_layout_error(
            "compact diagonal reached absorb execution",
        )),
        #[cfg(feature = "cuda")]
        (Data::CudaF64(_), _) | (_, Data::CudaF64(_)) => Err(Error::UnsupportedOnDevice(
            "Tensor::absorb has no device implementation yet".to_string(),
        )),
    }
}

#[cfg(test)]
mod absorb_tests {
    use super::*;
    use std::collections::HashMap;
    use tenet_core::{BlockSpec, FusionTreeKey, SU2FusionRule};

    fn scalar_structure(keys: &[FusionTreePairKey]) -> BlockStructure {
        BlockStructure::from_blocks(
            keys.iter()
                .cloned()
                .enumerate()
                .map(|(offset, key)| {
                    let rank = key.codomain_uncoupled().len() + key.domain_uncoupled().len();
                    BlockSpec::column_major_with_key(key.into(), vec![1; rank], offset).unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn assert_exact_key_oracle(
        mut destination_keys: Vec<FusionTreePairKey>,
        mut source_keys: Vec<FusionTreePairKey>,
    ) {
        destination_keys.sort();
        source_keys.sort();
        let destination_structure = scalar_structure(&destination_keys);
        let source_structure = scalar_structure(&source_keys);
        let mut destination = (0..destination_keys.len())
            .map(|index| 100.0 + index as f64)
            .collect::<Vec<_>>();
        let before = destination.clone();
        let source = (0..source_keys.len())
            .map(|index| 10.0 + index as f64)
            .collect::<Vec<_>>();
        let source_values = source_keys
            .iter()
            .cloned()
            .zip(source.iter().copied())
            .collect::<HashMap<_, _>>();

        absorb_mapped(
            &destination_structure,
            &mut destination,
            &source_structure,
            &source,
            Ok,
        )
        .unwrap();

        for index in 0..destination_structure.block_count() {
            let block = destination_structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!()
            };
            assert_eq!(
                destination[block.offset()],
                source_values
                    .get(key)
                    .copied()
                    .unwrap_or(before[block.offset()])
            );
        }
    }

    #[test]
    fn absorb_merge_matches_only_complete_asymmetric_interleaved_keys() {
        // What: equal external sectors do not alias distinct coupled sectors
        // or inner lines when destination/source key sets interleave.
        let rule = SU2FusionRule;
        let pair = |coupled, innerline| {
            let tree = FusionTreeKey::try_from_sector_ids_for_rule(
                &rule,
                [1, 1, 1],
                coupled,
                [false; 3],
                [innerline],
                [1, 1],
            )
            .unwrap();
            FusionTreePairKey::pair(tree.clone(), tree)
        };
        let inner_zero = pair(1, 0);
        let inner_two = pair(1, 2);
        let coupled_three = pair(3, 2);
        let destination_keys = vec![inner_zero.clone(), inner_two.clone(), coupled_three.clone()];
        assert_exact_key_oracle(destination_keys.clone(), vec![inner_two]);
        assert_exact_key_oracle(destination_keys, vec![inner_zero, coupled_three]);
    }

    #[test]
    fn absorb_duality_precedes_layout_and_layout_precedes_conversion() {
        // What: malformed materialized storage is reported before a nonreal
        // overlapping C64 value can reach scalar conversion.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 1)]);
        let mut destination = Tensor::zeros(&runtime, Dtype::F64, [&space], [&space]).unwrap();
        let dual = space.dual();
        let dual_source = Tensor::zeros(&runtime, Dtype::F64, [&dual], [&dual]).unwrap();
        let source = Tensor::from_block_fn(&runtime, [&space], [&space], |_, _| {
            Complex64::new(1.0, 2.0)
        })
        .unwrap();
        let body = destination.owned_body_mut().unwrap();
        let Data::F64(data) = Arc::make_mut(&mut body.data) else {
            unreachable!()
        };
        data.clear();

        // What: duality validation wins when the materialized block layout is
        // also malformed.
        assert!(matches!(
            destination.absorb(&dual_source),
            Err(Error::InvalidArgument(message)) if message.contains("duality")
        ));
        assert!(matches!(
            destination.absorb(&source),
            Err(Error::InvalidArgument(message)) if message.contains("block layout")
        ));
    }
}

fn build_bound_space<
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + tenet_core::CheckedFusionAlgebra,
>(
    provider: Arc<R>,
    hom: FusionTreeHomSpace,
) -> Result<BoundDynamicFusionMapSpace<R>, Error> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(provider, hom)
        .map_err(Into::into)
}

fn build_bound_space_like<
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + tenet_core::CheckedFusionAlgebra,
>(
    authority: &BoundDynamicFusionMapSpace<R>,
    hom: FusionTreeHomSpace,
) -> Result<BoundDynamicFusionMapSpace<R>, Error> {
    authority
        .derive_from_final_homspace(hom)
        .map_err(Into::into)
}

/// Which tree transform a leg re-arrangement uses.
enum TransformKind<'a> {
    Permute,
    Braid { levels: &'a [usize] },
    Transpose,
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
#[derive(Debug)]
enum UserBoundSpace {
    U1(BoundDynamicFusionMapSpace<tenet_core::U1FusionRule>),
    CU1(BoundDynamicFusionMapSpace<tenet_core::CU1FusionRule>),
    Z2(BoundDynamicFusionMapSpace<tenet_core::Z2FusionRule>),
    ZN(BoundDynamicFusionMapSpace<tenet_core::ZNFusionRule>),
    FZ2(BoundDynamicFusionMapSpace<tenet_core::FermionParityFusionRule>),
    SU2(BoundDynamicFusionMapSpace<tenet_core::SU2FusionRule>),
    U1FZ2(BoundDynamicFusionMapSpace<U1Fz2Rule>),
    FZ2U1SU2(BoundDynamicFusionMapSpace<Fz2U1Su2Rule>),
}

trait IntoUserBoundDynamicSpace: FusionRule + Sized {
    fn into_user_bound(
        expected: &UserBoundSpace,
        bound: BoundDynamicFusionMapSpace<Self>,
    ) -> Result<UserBoundSpace, Error>;
}

macro_rules! impl_into_user_bound {
    ($rule:ty, $context:ident, $inner:ident) => {
        impl IntoUserBoundDynamicSpace for $rule {
            fn into_user_bound(
                expected: &UserBoundSpace,
                bound: BoundDynamicFusionMapSpace<Self>,
            ) -> Result<UserBoundSpace, Error> {
                let UserBoundSpace::$inner(existing) = expected else {
                    return Err(Error::InvalidArgument(
                        "SVD factor provider type does not match tensor context".to_string(),
                    ));
                };
                if !Arc::ptr_eq(existing.provider_arc(), bound.provider_arc())
                    || existing.provider().rule_identity() != bound.provider().rule_identity()
                {
                    return Err(Error::InvalidArgument(
                        "SVD factor provider identity does not match tensor context".to_string(),
                    ));
                }
                Ok(UserBoundSpace::$inner(bound))
            }
        }
    };
}

impl_into_user_bound!(tenet_core::U1FusionRule, U1, U1);
impl_into_user_bound!(tenet_core::CU1FusionRule, CU1, CU1);
impl_into_user_bound!(tenet_core::Z2FusionRule, Z2, Z2);
impl_into_user_bound!(tenet_core::ZNFusionRule, ZN, ZN);
impl_into_user_bound!(tenet_core::FermionParityFusionRule, FZ2, FZ2);
impl_into_user_bound!(tenet_core::SU2FusionRule, SU2, SU2);
impl_into_user_bound!(U1Fz2Rule, U1FZ2, U1FZ2);
impl_into_user_bound!(Fz2U1Su2Rule, FZ2U1SU2, FZ2U1SU2);

impl PartialEq for UserBoundSpace {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity() && self.raw() == other.raw()
    }
}

impl Eq for UserBoundSpace {}

impl UserBoundSpace {
    fn from_bound<R>(expected: &Self, bound: BoundDynamicFusionMapSpace<R>) -> Result<Self, Error>
    where
        R: IntoUserBoundDynamicSpace,
    {
        R::into_user_bound(expected, bound)
    }

    fn transformed(&self, operation: &TreeTransformOperation) -> Result<Self, Error> {
        macro_rules! transform {
            ($space:expr, $variant:ident, $method:ident) => {
                Ok(UserBoundSpace::$variant($space.$method(operation)?))
            };
        }
        match self {
            Self::U1(space) => transform!(space, U1, transformed_multiplicity_free),
            Self::CU1(space) => transform!(space, CU1, transformed_multiplicity_free),
            Self::Z2(space) => transform!(space, Z2, transformed_multiplicity_free),
            Self::ZN(space) => transform!(space, ZN, transformed_multiplicity_free),
            Self::FZ2(space) => transform!(space, FZ2, transformed_multiplicity_free),
            Self::SU2(space) => transform!(space, SU2, transformed_multiplicity_free),
            Self::U1FZ2(space) => transform!(space, U1FZ2, transformed_multiplicity_free),
            Self::FZ2U1SU2(space) => {
                transform!(space, FZ2U1SU2, transformed_multiplicity_free)
            }
        }
    }

    fn from_homspace(&self, homspace: FusionTreeHomSpace) -> Result<Self, Error> {
        macro_rules! build {
            ($space:expr, $variant:ident) => {
                Ok(UserBoundSpace::$variant(build_bound_space_like(
                    $space, homspace,
                )?))
            };
        }
        match self {
            Self::U1(space) => build!(space, U1),
            Self::CU1(space) => build!(space, CU1),
            Self::Z2(space) => build!(space, Z2),
            Self::ZN(space) => build!(space, ZN),
            Self::FZ2(space) => build!(space, FZ2),
            Self::SU2(space) => build!(space, SU2),
            Self::U1FZ2(space) => build!(space, U1FZ2),
            Self::FZ2U1SU2(space) => build!(space, FZ2U1SU2),
        }
    }

    fn from_selected_homspace(&self, homspace: FusionTreeHomSpace) -> Result<Self, Error> {
        #[cfg(test)]
        observe_selected_result_layout_build();
        macro_rules! build {
            ($space:expr, $variant:ident) => {
                Ok(UserBoundSpace::$variant(
                    $space.derive_from_final_homspace(homspace)?,
                ))
            };
        }
        match self {
            Self::U1(space) => build!(space, U1),
            Self::CU1(space) => build!(space, CU1),
            Self::Z2(space) => build!(space, Z2),
            Self::ZN(space) => build!(space, ZN),
            Self::FZ2(space) => build!(space, FZ2),
            Self::SU2(space) => build!(space, SU2),
            Self::U1FZ2(space) => build!(space, U1FZ2),
            Self::FZ2U1SU2(space) => build!(space, FZ2U1SU2),
        }
    }

    fn raw(&self) -> &DynamicFusionMapSpace {
        match self {
            UserBoundSpace::U1(space) => space.space(),
            UserBoundSpace::CU1(space) => space.space(),
            UserBoundSpace::Z2(space) => space.space(),
            UserBoundSpace::ZN(space) => space.space(),
            UserBoundSpace::FZ2(space) => space.space(),
            UserBoundSpace::SU2(space) => space.space(),
            UserBoundSpace::U1FZ2(space) => space.space(),
            UserBoundSpace::FZ2U1SU2(space) => space.space(),
        }
    }

    fn context(&self) -> UserRuleContext {
        match self {
            UserBoundSpace::U1(space) => UserRuleContext::U1(Arc::clone(space.provider_arc())),
            UserBoundSpace::CU1(space) => UserRuleContext::CU1(Arc::clone(space.provider_arc())),
            UserBoundSpace::Z2(space) => UserRuleContext::Z2(Arc::clone(space.provider_arc())),
            UserBoundSpace::ZN(space) => UserRuleContext::ZN(Arc::clone(space.provider_arc())),
            UserBoundSpace::FZ2(space) => UserRuleContext::FZ2(Arc::clone(space.provider_arc())),
            UserBoundSpace::SU2(space) => UserRuleContext::SU2(Arc::clone(space.provider_arc())),
            UserBoundSpace::U1FZ2(space) => {
                UserRuleContext::U1FZ2(Arc::clone(space.provider_arc()))
            }
            UserBoundSpace::FZ2U1SU2(space) => {
                UserRuleContext::FZ2U1SU2(Arc::clone(space.provider_arc()))
            }
        }
    }

    fn identity(&self) -> tenet_core::RuleIdentity {
        match self {
            UserBoundSpace::U1(space) => space.provider().rule_identity(),
            UserBoundSpace::CU1(space) => space.provider().rule_identity(),
            UserBoundSpace::Z2(space) => space.provider().rule_identity(),
            UserBoundSpace::ZN(space) => space.provider().rule_identity(),
            UserBoundSpace::FZ2(space) => space.provider().rule_identity(),
            UserBoundSpace::SU2(space) => space.provider().rule_identity(),
            UserBoundSpace::U1FZ2(space) => space.provider().rule_identity(),
            UserBoundSpace::FZ2U1SU2(space) => space.provider().rule_identity(),
        }
    }

    #[cfg(test)]
    fn provider_matches_context_allocation(&self, context: &UserRuleContext) -> bool {
        match (self, context) {
            (Self::U1(space), UserRuleContext::U1(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::CU1(space), UserRuleContext::CU1(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::Z2(space), UserRuleContext::Z2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::ZN(space), UserRuleContext::ZN(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::FZ2(space), UserRuleContext::FZ2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::SU2(space), UserRuleContext::SU2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::U1FZ2(space), UserRuleContext::U1FZ2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::FZ2U1SU2(space), UserRuleContext::FZ2U1SU2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            _ => false,
        }
    }
}

impl std::ops::Deref for UserBoundSpace {
    type Target = DynamicFusionMapSpace;

    fn deref(&self) -> &Self::Target {
        self.raw()
    }
}

macro_rules! with_bound_multiplicity_free {
    ($space:expr, $bound:ident, $body:expr) => {
        match $space.as_ref() {
            UserBoundSpace::U1($bound) => $body,
            UserBoundSpace::CU1($bound) => $body,
            UserBoundSpace::Z2($bound) => $body,
            UserBoundSpace::ZN($bound) => $body,
            UserBoundSpace::FZ2($bound) => $body,
            UserBoundSpace::SU2($bound) => $body,
            UserBoundSpace::U1FZ2($bound) => $body,
            UserBoundSpace::FZ2U1SU2($bound) => $body,
        }
    };
}

/// Static dispatch from the tensor's sole bound authority. Why not rebuild a
/// `UserRuleContext`: ordinary operations only need a provider borrow, and an
/// enum reconstruction plus Arc refcount traffic would make the hot path pay
/// for a user-facing `Space` view it never creates.
macro_rules! with_user_rule {
    ($space:expr, $rule:ident, $body:expr) => {
        match $space.as_ref() {
            UserBoundSpace::U1(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::CU1(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::Z2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::ZN(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::FZ2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::SU2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::U1FZ2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::FZ2U1SU2(bound) => {
                let $rule = bound.provider();
                $body
            }
        }
    };
}

macro_rules! with_bound_ctx {
    ($space:expr, $state:expr, $bound:ident, $ctxs:ident, $body:expr) => {
        match $space.as_ref() {
            UserBoundSpace::U1($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
            UserBoundSpace::CU1($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
            UserBoundSpace::Z2($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
            UserBoundSpace::ZN($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
            UserBoundSpace::FZ2($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
            UserBoundSpace::SU2($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
            UserBoundSpace::U1FZ2($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
            UserBoundSpace::FZ2U1SU2($bound) => {
                let $ctxs = &mut $state.mf;
                $body
            }
        }
    };
}

#[derive(Clone, Debug)]
struct TensorBody {
    space: Arc<UserBoundSpace>,
    data: Arc<Data>,
}

#[derive(Debug)]
struct AdjointView {
    parent: TensorBody,
    materialized: OnceLock<Arc<TensorBody>>,
    // Why not rely on OnceLock::set races: losing builders would still repeat
    // the expensive block-grid/data work before publication.
    init: Mutex<()>,
    #[cfg(test)]
    materialized_body_builds: std::sync::atomic::AtomicUsize,
}

#[derive(Debug)]
enum TensorRepr {
    Owned(TensorBody),
    Adjoint(Arc<AdjointView>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContractionSemantics {
    TensorContract,
    Composition,
}

struct TensorMetadataView<'a> {
    body: &'a TensorBody,
    orientation: TensorOrientation,
}

impl TensorMetadataView<'_> {
    fn codomain(&self) -> &FusionProductSpace {
        match self.orientation {
            TensorOrientation::Owned => self.body.space.homspace().codomain(),
            TensorOrientation::Adjoint => self.body.space.homspace().domain(),
        }
    }

    fn domain(&self) -> &FusionProductSpace {
        match self.orientation {
            TensorOrientation::Owned => self.body.space.homspace().domain(),
            TensorOrientation::Adjoint => self.body.space.homspace().codomain(),
        }
    }

    fn nout(&self) -> usize {
        self.codomain().len()
    }

    fn nin(&self) -> usize {
        self.domain().len()
    }

    fn rank(&self) -> usize {
        self.nout() + self.nin()
    }

    fn oriented_homspace(&self) -> OrientedFusionTreeHomSpace<'_> {
        let orientation = match self.orientation {
            TensorOrientation::Owned => FusionTreePairOrientation::Direct,
            TensorOrientation::Adjoint => FusionTreePairOrientation::Adjoint,
        };
        OrientedFusionTreeHomSpace::new(self.body.space.homspace(), orientation)
    }
}

/// A block-sparse symmetric tensor map `codomain <- domain` with dynamic rank,
/// carrying its [`Runtime`] and a rule-erased fusion space. This is the
/// everyday user-layer type; see the crate-level docs for the execution model.
#[derive(Debug)]
pub struct Tensor {
    rt: Runtime,
    repr: TensorRepr,
    compact_dense: OnceLock<Arc<Data>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let compact_dense = OnceLock::new();
        if let Some(data) = self.compact_dense.get() {
            let _ = compact_dense.set(Arc::clone(data));
        }
        let repr = match &self.repr {
            TensorRepr::Owned(body) => TensorRepr::Owned(body.clone()),
            TensorRepr::Adjoint(view) => TensorRepr::Adjoint(Arc::clone(view)),
        };
        Self {
            rt: self.rt.clone(),
            repr,
            compact_dense,
        }
    }
}

impl Tensor {
    fn owned(rt: Runtime, space: Arc<UserBoundSpace>, data: Arc<Data>) -> Self {
        Self {
            rt,
            repr: TensorRepr::Owned(TensorBody { space, data }),
            compact_dense: OnceLock::new(),
        }
    }

    fn metadata(&self) -> TensorMetadataView<'_> {
        match &self.repr {
            TensorRepr::Owned(body) => TensorMetadataView {
                body,
                orientation: TensorOrientation::Owned,
            },
            TensorRepr::Adjoint(view) => TensorMetadataView {
                body: &view.parent,
                orientation: TensorOrientation::Adjoint,
            },
        }
    }

    fn parent_body_for_lowering(&self) -> &TensorBody {
        match &self.repr {
            TensorRepr::Owned(body) => body,
            TensorRepr::Adjoint(view) => &view.parent,
        }
    }

    fn parent_tensor_for_lowering(&self) -> Self {
        let body = self.parent_body_for_lowering();
        Self::owned(
            self.rt.clone(),
            Arc::clone(&body.space),
            Arc::clone(&body.data),
        )
    }

    fn parent_layout_supports_inner_identity(&self) -> Result<bool, Error> {
        let parent = self.parent_body_for_lowering();
        Ok(parent
            .space
            .structure()
            .coupled_sector_regions(parent.space.nout())?
            .is_some())
    }

    fn oriented_add_adjoint<D: UserScalar>(
        &self,
        other: &Self,
        alpha: D,
        beta: D,
    ) -> Result<Self, Error> {
        debug_assert!(self.is_adjoint_view() ^ other.is_adjoint_view());
        let (logical, parent, logical_factor, adjoint_factor) = if self.is_adjoint_view() {
            (
                other.parent_body_for_lowering(),
                self.parent_body_for_lowering(),
                beta,
                alpha,
            )
        } else {
            (
                self.parent_body_for_lowering(),
                other.parent_body_for_lowering(),
                alpha,
                beta,
            )
        };
        let parent_operand = tenet_tensors::FusionOperand::adjoint(parent.space.raw());
        let data = if let Data::Diagonal(diagonal) = logical.data.as_ref() {
            let mut data = oriented_add_data(
                logical.space.raw(),
                parent_operand,
                parent.data.as_ref(),
                parent_operand,
                parent.data.as_ref(),
                adjoint_factor,
                D::from_real(0.0),
            )?;
            axpy_diagonal_into(
                logical.space.raw(),
                &mut data,
                diagonal,
                logical_factor.widen_complex(),
            )?;
            data
        } else {
            oriented_add_data(
                logical.space.raw(),
                tenet_tensors::FusionOperand::direct(logical.space.raw()),
                logical.data.as_ref(),
                parent_operand,
                parent.data.as_ref(),
                logical_factor,
                adjoint_factor,
            )?
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&logical.space),
            Arc::new(data),
        ))
    }

    fn ordinary_body(&self) -> &TensorBody {
        // Why not return the adjoint parent: its space and bytes describe a
        // different tensor, which is exactly the incoherent pair this split
        // prevents from reaching general consumers.
        match &self.repr {
            TensorRepr::Owned(body) => body,
            TensorRepr::Adjoint(_) => {
                panic!("internal: an adjoint view reached an owned-only consumer")
            }
        }
    }

    fn stored_data(&self) -> &Data {
        // Dtype, placement, and view-native lowering inspect physical storage;
        // they do not interpret it in the logical adjoint block layout.
        self.parent_body_for_lowering().data.as_ref()
    }

    fn stored_data_arc(&self) -> &Arc<Data> {
        &self.parent_body_for_lowering().data
    }

    fn rule_authority_space(&self) -> &Arc<UserBoundSpace> {
        // Fusion-rule/provider identity is adjoint-invariant. Why not use this
        // for an owned layout: only materialized_body may supply that.
        &self.parent_body_for_lowering().space
    }

    fn owned_body_mut(&mut self) -> Option<&mut TensorBody> {
        let TensorRepr::Owned(body) = &mut self.repr else {
            return None;
        };
        Some(body)
    }

    fn is_adjoint_view(&self) -> bool {
        matches!(self.repr, TensorRepr::Adjoint(_))
    }

    #[cfg(test)]
    fn has_cached_materialization(&self) -> bool {
        match &self.repr {
            TensorRepr::Owned(_) => self.compact_dense.get().is_some(),
            TensorRepr::Adjoint(view) => view.materialized.get().is_some(),
        }
    }

    #[cfg(test)]
    fn adjoint_body_builds(&self) -> usize {
        let TensorRepr::Adjoint(view) = &self.repr else {
            return 0;
        };
        view.materialized_body_builds
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn rule_context(&self) -> UserRuleContext {
        self.rule_authority_space().context()
    }

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
        let context = Arc::clone(
            spaces
                .next()
                .ok_or_else(|| {
                    Error::InvalidArgument(
                        "at least one leg is required to infer the fusion rule".to_string(),
                    )
                })?
                .rule_context(),
        );
        if spaces.any(|space| {
            !Arc::ptr_eq(space.rule_context(), &context)
                && space.rule_context().identity() != context.identity()
        }) {
            return Err(Error::RuleMismatch);
        }

        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|space| space.sector_leg())),
            FusionProductSpace::new(domain.iter().map(|space| space.sector_leg())),
        );
        macro_rules! build {
            ($provider:expr, $variant:ident) => {{
                let bound = build_bound_space(Arc::clone($provider), hom)?;
                let data = S::lift(apply_fill(bound.space(), fill)?);
                Ok::<_, Error>((UserBoundSpace::$variant(bound), data))
            }};
        }
        let (space, data) = match context.as_ref() {
            UserRuleContext::U1(provider) => build!(provider, U1),
            UserRuleContext::CU1(provider) => build!(provider, CU1),
            UserRuleContext::Z2(provider) => build!(provider, Z2),
            UserRuleContext::ZN(provider) => build!(provider, ZN),
            UserRuleContext::FZ2(provider) => build!(provider, FZ2),
            UserRuleContext::SU2(provider) => build!(provider, SU2),
            UserRuleContext::U1FZ2(provider) => build!(provider, U1FZ2),
            UserRuleContext::FZ2U1SU2(provider) => build!(provider, FZ2U1SU2),
        }?;
        Ok(Self::owned(rt.clone(), Arc::new(space), Arc::new(data)))
    }

    /// Zero tensor of the given [`Dtype`] on `codomain <- domain`
    /// (TensorKit `zeros(Float64, W ← V)` / `zeros(ComplexF64, W ← V)`).
    /// All spaces must share one fusion rule.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when no leg is given (the fusion rule is
    ///   inferred from the legs).
    /// - [`Error::RuleMismatch`] when the legs disagree on the rule identity,
    ///   reported before any provider algebra runs.
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    ///   when the provider cannot certify the requested layout. Nothing is
    ///   published in that case.
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
    /// for an explicit stream. Note the stream position is drawn *before*
    /// validation, so a failing call still advances the runtime's seedless
    /// stream — unlike the typed sibling, which draws after admission.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports.
    pub fn rand<'a, C, D>(rt: &Runtime, dtype: Dtype, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::rand_with_seed(rt, dtype, codomain, domain, rt.next_rand_seed())
    }

    /// Random tensor with an explicit seed (splitmix64 stream, entries
    /// uniform in `[-1, 1)`).
    ///
    /// Reproducibility is defined for the same TeNeT version and tensor
    /// layout. The stream fills internal storage order, so a sector codec or
    /// block-layout migration can produce a different semantic tensor from
    /// the same seed. Cross-version fixtures should use
    /// [`Self::from_block_fn`] with semantic [`BlockKey`] labels.
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
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports; the fill itself is infallible.
    ///
    /// # Complexity
    ///
    /// One layout admission plus one `O(stored_len)` allocation with one
    /// `fill` call per symmetry-allowed element.
    pub fn from_block_fn<'a, C, D, S, F>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        mut fill: F,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
        S: UserScalar,
        F: FnMut(&BlockKey, &[usize]) -> S,
    {
        Self::build(rt, codomain, domain, Fill::BlockFn(&mut fill))
    }

    /// TensorKit-compatible immutable `absorb`: copies the common per-axis
    /// prefix of every exact shared fusion-tree block from `source` into a
    /// deep copy of `self`.
    ///
    /// The result keeps `self`'s HomSpace and dtype. Coordinates outside the
    /// common prefix are unchanged. Real source values widen exactly into a
    /// complex destination; complex values require a zero imaginary part when
    /// copied into a real destination.
    pub fn absorb(&self, source: &Self) -> Result<Self, Error> {
        let destination_metadata = self.metadata();
        let source_metadata = source.metadata();
        if destination_metadata.nout() != source_metadata.nout()
            || destination_metadata.nin() != source_metadata.nin()
        {
            return Err(Error::InvalidArgument(format!(
                "Tensor::absorb requires equal codomain/domain ranks, got {}|{} and {}|{}",
                destination_metadata.nout(),
                destination_metadata.nin(),
                source_metadata.nout(),
                source_metadata.nin()
            )));
        }
        if self.rule_authority_space().identity() != source.rule_authority_space().identity() {
            return Err(Error::RuleMismatch);
        }
        if !self.rt.same_runtime(&source.rt) {
            return Err(Error::RuntimeMismatch);
        }
        if self.placement() != source.placement() {
            return Err(Error::PlacementMismatch);
        }
        if self.placement() != Placement::Host {
            return Err(Error::UnsupportedOnDevice(
                "Tensor::absorb has no device implementation yet; move both tensors to the host \
                 explicitly with to_host()"
                    .to_string(),
            ));
        }
        for (destination, source) in destination_metadata
            .codomain()
            .legs()
            .iter()
            .chain(destination_metadata.domain().legs())
            .zip(
                source_metadata
                    .codomain()
                    .legs()
                    .iter()
                    .chain(source_metadata.domain().legs()),
            )
        {
            if destination.is_dual() != source.is_dual() {
                return Err(Error::InvalidArgument(
                    "Tensor::absorb requires corresponding legs to have equal duality".to_string(),
                ));
            }
        }

        let destination_body = self.materialized_body()?;
        let source_body = source.materialized_body()?;
        let destination_data = self.coupled_data()?;
        let source_data = source.coupled_data()?;
        validate_absorb_layout(destination_body.space.structure(), destination_data)?;
        validate_absorb_layout(source_body.space.structure(), source_data)?;
        let data = execute_absorb_data(
            destination_body.space.structure(),
            destination_data,
            source_body.space.structure(),
            source_data,
        )?;
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&destination_body.space),
            Arc::new(data),
        ))
    }

    /// TensorKit `catdomain(t1, t2)`: concatenate two `Nout | 1` tensor maps
    /// along their sole domain leg. The codomain product spaces must match
    /// exactly; the two domain spaces are combined by direct sum, and reduced
    /// data is copied into adjacent column slabs for every complete fusion-tree
    /// pair key. Mixed f64/c64 operands produce a c64 tensor.
    ///
    /// Rust uses a method (`t1.catdomain(&t2)`) because binary tensor
    /// operations in this API are methods; the name and operand order match
    /// TensorKit's free function.
    ///
    pub fn catdomain(&self, other: &Self) -> Result<Self, Error> {
        self.cat(other, CatSide::Domain)
    }

    /// TensorKit `catcodomain(t1, t2)`: concatenate two `1 | Nin` tensor maps
    /// along their sole codomain leg. The domain product spaces must match
    /// exactly; the two codomain spaces are combined by direct sum, and
    /// reduced data is copied into adjacent row slabs for every complete
    /// fusion-tree pair key. Mixed f64/c64 operands produce a c64 tensor.
    ///
    /// Rust uses a method (`t1.catcodomain(&t2)`) because binary tensor
    /// operations in this API are methods; the name and operand order match
    /// TensorKit's free function.
    ///
    pub fn catcodomain(&self, other: &Self) -> Result<Self, Error> {
        self.cat(other, CatSide::Codomain)
    }

    fn cat(&self, other: &Self, side: CatSide) -> Result<Self, Error> {
        self.check_same_execution_world(other)?;
        if self.placement() != Placement::Host {
            let operation = match side {
                CatSide::Domain => "Tensor::catdomain",
                CatSide::Codomain => "Tensor::catcodomain",
            };
            return Err(Error::UnsupportedOnDevice(format!(
                "{operation} has no device implementation yet; move both tensors to the host \
                 explicitly with to_host()"
            )));
        }
        let lhs = self.metadata();
        let rhs = other.metadata();
        // `domain_spaces()`/`codomain_spaces()` used to be round-tripped here
        // through user `Space`s solely to call `Space::oplus`; the oriented
        // homspace legs are the same `SectorLeg`s (`Space::from_leg` /
        // `sector_leg` are verbatim inverses), so the core reads them directly.
        let (axis, homspace) = cat_homspace(
            lhs.codomain(),
            lhs.domain(),
            rhs.codomain(),
            rhs.domain(),
            side,
        )?;

        if (self.is_adjoint_view() || other.is_adjoint_view())
            && !matches!(self.stored_data(), Data::Diagonal(_))
            && !matches!(other.stored_data(), Data::Diagonal(_))
        {
            if let Some(descriptor) =
                CatDescriptor::try_new_oriented(&lhs, &rhs, axis, side, homspace.clone())?
            {
                let data = execute_cat_data(&descriptor, self.stored_data(), other.stored_data())?;
                return Ok(Self::owned(
                    self.rt.clone(),
                    Arc::new(descriptor.space),
                    Arc::new(data),
                ));
            }
        }

        let lhs_body = self.materialized_body()?;
        let rhs_body = other.materialized_body()?;
        let descriptor = CatDescriptor::new(lhs_body, rhs_body, axis, side, homspace)?;
        let data = execute_cat_data(&descriptor, self.coupled_data()?, other.coupled_data()?)?;
        Ok(Self::owned(
            self.rt.clone(),
            Arc::new(descriptor.space),
            Arc::new(data),
        ))
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
        let regions = sector_regions(
            t.ordinary_body().space.structure(),
            t.ordinary_body().space.nout(),
        )?;
        let body = t.owned_body_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "fresh structural tensor unexpectedly shares its owned authority".to_string(),
            )
        })?;
        let Data::F64(data) = Arc::make_mut(&mut body.data) else {
            unreachable!("structural constructors build f64 host tensors");
        };
        for region in regions.iter() {
            for i in 0..region.rows().min(region.cols()) {
                data[region.range().start + i * (region.rows() + 1)] = 1.0;
            }
        }
        Ok(match dtype {
            Dtype::F64 => t,
            Dtype::C64 => t.to_c64(),
        })
    }

    /// The identity endomorphism on `spaces <- spaces` (TensorKit `id(V)`):
    /// every coupled-sector block is the identity matrix.
    ///
    pub fn id<'a, S>(rt: &Runtime, dtype: Dtype, spaces: S) -> Result<Self, Error>
    where
        S: IntoIterator<Item = &'a Space>,
    {
        let spaces: Vec<&Space> = spaces.into_iter().collect();
        Self::structural(rt, dtype, spaces.clone(), spaces, false, "id")
    }

    /// Builds a compact diagonal tensor `bond <- bond` from canonical sector-order values.
    ///
    /// This is TeNeT's immutable, type-erased counterpart of TensorKit's
    /// `DiagonalTensorMap` / `diagm` (`tensors/diagonal.jl`): it stores only
    /// `Σ_c k_c` values, not the dense `Σ_c k_c²` diagonal blocks. The vectors
    /// are positional in [`Space::sectors`] order; their order is therefore
    /// not inferred from their values.
    ///
    /// Every vector must have its sector's exact degeneracy and every scalar
    /// must match `dtype` exactly. Validation completes before layout admission.
    /// The checked admission root retains the supplied leg, including its dual
    /// flag.
    ///
    /// # Complexity
    ///
    /// Construction stores `O(Σ_c k_c)` values. Dense storage is first allocated
    /// only by [`Self::data`] or [`Self::data_c64`].
    pub fn diagonal<I>(
        rt: &Runtime,
        dtype: Dtype,
        bond: &Space,
        sector_values: I,
    ) -> Result<Self, Error>
    where
        I: IntoIterator<Item = Vec<Scalar>>,
    {
        let values: Vec<Vec<Scalar>> = sector_values.into_iter().collect();
        if values.len() != bond.sectors.len() {
            return Err(Error::InvalidArgument(
                "diagonal spectrum sector count does not match bond".into(),
            ));
        }
        for ((_, degeneracy), values) in bond.sectors.iter().zip(&values) {
            if values.len() != *degeneracy {
                return Err(Error::InvalidArgument(
                    "diagonal spectrum length does not match bond degeneracy".into(),
                ));
            }
            if values.iter().any(|value| {
                !matches!(
                    (dtype, value),
                    (Dtype::F64, Scalar::F64(_)) | (Dtype::C64, Scalar::C64(_))
                )
            }) {
                return Err(Error::DtypeMismatch);
            }
        }
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([bond.sector_leg()]),
            FusionProductSpace::new([bond.sector_leg()]),
        );
        let space = match bond.rule_context().as_ref() {
            UserRuleContext::U1(provider) => {
                UserBoundSpace::U1(build_bound_space(Arc::clone(provider), hom)?)
            }
            UserRuleContext::CU1(provider) => {
                UserBoundSpace::CU1(build_bound_space(Arc::clone(provider), hom)?)
            }
            UserRuleContext::Z2(provider) => {
                UserBoundSpace::Z2(build_bound_space(Arc::clone(provider), hom)?)
            }
            UserRuleContext::ZN(provider) => {
                UserBoundSpace::ZN(build_bound_space(Arc::clone(provider), hom)?)
            }
            UserRuleContext::FZ2(provider) => {
                UserBoundSpace::FZ2(build_bound_space(Arc::clone(provider), hom)?)
            }
            UserRuleContext::SU2(provider) => {
                UserBoundSpace::SU2(build_bound_space(Arc::clone(provider), hom)?)
            }
            UserRuleContext::U1FZ2(provider) => {
                UserBoundSpace::U1FZ2(build_bound_space(Arc::clone(provider), hom)?)
            }
            UserRuleContext::FZ2U1SU2(provider) => {
                UserBoundSpace::FZ2U1SU2(build_bound_space(Arc::clone(provider), hom)?)
            }
        };
        let data = match dtype {
            Dtype::F64 => DiagonalData::RealF64(
                bond.sectors
                    .iter()
                    .zip(values)
                    .map(|(&(sector, _), values)| {
                        Ok(SectorSpectrum {
                            sector,
                            values: values
                                .into_iter()
                                .map(|value| value.try_f64())
                                .collect::<Result<_, _>>()?,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
            ),
            Dtype::C64 => DiagonalData::C64(
                bond.sectors
                    .iter()
                    .zip(values)
                    .map(|(&(sector, _), values)| SectorSpectrum {
                        sector,
                        values: values.into_iter().map(Scalar::to_c64).collect(),
                    })
                    .collect(),
            ),
        };
        Ok(Self::owned(
            rt.clone(),
            Arc::new(space),
            Arc::new(Data::Diagonal(data)),
        ))
    }

    /// Returns compact diagonal values in the tensor's public dtype without materializing.
    ///
    /// This is TeNeT's `diag` readback. It returns [`None`] for dense storage;
    /// otherwise vectors are returned in the canonical bond-sector order and
    /// cost `O(Σ_c k_c)`. A complex tensor with internally real compact values
    /// returns [`Scalar::C64`] values with zero imaginary part.
    pub fn diagonal_spectrum(&self) -> Result<Option<Vec<Vec<Scalar>>>, Error> {
        Ok(match self.stored_data() {
            Data::Diagonal(DiagonalData::RealF64(spectrum)) => Some(
                spectrum
                    .iter()
                    .map(|entry| entry.values.iter().copied().map(Scalar::F64).collect())
                    .collect(),
            ),
            Data::Diagonal(DiagonalData::RealC64(spectrum)) => Some(
                spectrum
                    .iter()
                    .map(|entry| {
                        entry
                            .values
                            .iter()
                            .copied()
                            .map(|value| Scalar::C64(Complex64::new(value, 0.0)))
                            .collect()
                    })
                    .collect(),
            ),
            Data::Diagonal(DiagonalData::C64(spectrum)) => Some(
                spectrum
                    .iter()
                    .map(|entry| entry.values.iter().copied().map(Scalar::C64).collect())
                    .collect(),
            ),
            _ => None,
        })
    }

    /// Tests rank-one host tensors for blockwise diagonality.
    ///
    /// This is TensorKit's `isdiag` at `tol = 0` for finite data. Positive
    /// tolerance accepts `max_offdiag <= tol * max(norm_inf, 1)`. Negative or
    /// non-finite tolerances are errors before any rank or storage shortcut;
    /// device tensors return the existing unsupported-on-device error.
    pub fn is_diagonal(&self, tol: f64) -> Result<bool, Error> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidArgument(
                "diagonal tolerance must be finite and nonnegative".into(),
            ));
        }
        if matches!(self.stored_data(), Data::Diagonal(_)) {
            return Ok(true);
        }
        if self.rank() != 2 || self.numout() != 1 || self.numin() != 1 {
            return Ok(false);
        }
        if self.placement() != Placement::Host {
            #[cfg(feature = "cuda")]
            return Err(device_unsupported("is_diagonal()"));
            #[cfg(not(feature = "cuda"))]
            return Err(Error::PlacementMismatch);
        }
        let body = self.materialized_body()?;
        let mut norm = 0.0_f64;
        let mut offdiag = 0.0_f64;
        macro_rules! scan {
            ($data:expr) => {{
                for index in 0..body.space.structure().block_count() {
                    let block = body.space.structure().block(index)?;
                    for row in 0..block.shape()[0] {
                        for col in 0..block.shape()[1] {
                            let value = $data[block.offset()
                                + row * block.strides()[0]
                                + col * block.strides()[1]]
                                .abs_value();
                            norm = norm.max(value);
                            if row != col {
                                if !value.is_finite() {
                                    return Ok(false);
                                }
                                offdiag = offdiag.max(value);
                            }
                        }
                    }
                }
            }};
        }
        match body.data.as_ref() {
            Data::F64(data) => scan!(data),
            Data::C64(data) => scan!(data),
            Data::Diagonal(_) => unreachable!("compact storage returned above"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("is_diagonal()")),
        }
        Ok(offdiag <= tol * norm.max(1.0))
    }

    /// The canonical structural isomorphism `codomain <- domain` (TensorKit
    /// `isomorphism(W ← V)`): every
    /// coupled-sector block is the identity matrix, which requires the fused
    /// codomain and domain to carry identical sector content. The
    /// finite-torus norm fuser is `isomorphism(fuse(dual(l) ⊗ l) ←
    /// dual(l) ⊗ l)`.
    ///
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

    /// TensorKit `unitary(W ← V)`: identical
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
    /// `isometry(W ← V)`): each coupled-sector
    /// block is the partial identity (the first `cols` columns of the
    /// identity), so `t† ∘ t = id(domain)`. Requires the domain to embed
    /// isometrically in the codomain (sectorwise `deg_domain <=
    /// deg_codomain`).
    ///
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

    /// TensorKit `twist(t, inds)`:
    /// multiplies each fusion-tree block by the product over `legs` (flat
    /// leg indices, codomain first) of the ribbon-twist eigenvalue θ of that
    /// leg's uncoupled sector on the block's fusion tree. θ = −1 for odd
    /// fermionic sectors and +1 for every bosonic sector, so this is a no-op
    /// on purely bosonic legs and an involution on fermionic ones.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when a leg index is out of range, or when
    ///   the fusion rule has no braiding and a requested leg carries non-unit
    ///   sectors.
    /// - [`Error::UnsupportedOnDevice`] for a device payload.
    /// - [`Error::Core`] when the stored block layout cannot be walked — an
    ///   engine invariant, not a caller mistake.
    ///
    pub fn twist(&self, legs: &[usize]) -> Result<Self, Error> {
        #[cfg(test)]
        observe_twist_call();
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.twist(legs);
        }
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "twist leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        with_user_rule!(self.ordinary_body().space, rule, {
            reject_unbraided_nonunit_legs(
                rule,
                self.ordinary_body().space.homspace(),
                legs,
                "twist",
                true,
            )
        })?;
        let nout = self.codomain_rank();
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return with_user_rule!(self.ordinary_body().space, rule, {
                let sector_factor = |sector| {
                    legs.iter()
                        .map(|_| rule.twist_scalar(sector))
                        .product::<f64>()
                };
                if diagonal.sectors_all(|sector| sector_factor(sector) == 1.0) {
                    Ok(self.clone())
                } else {
                    Ok(self.with_diagonal(diagonal.scaled_by_sector(sector_factor)))
                }
            });
        }
        // TensorKit `has_shared_twist` (`indexmanipulations.jl:34-51`): the
        // twist is the identity when every requested leg carries theta = 1 on
        // every block (bosonic rules O(1); a fermionic/anyonic tensor still
        // shares its buffer when no requested leg touches a twisted sector).
        // Skip the whole-buffer clone-and-scale-by-1 and return shared data.
        let twist_is_identity = with_user_rule!(self.ordinary_body().space, rule, {
            twist_is_identity_over_blocks(rule, self.ordinary_body().space.structure(), nout, legs)?
        });
        if twist_is_identity {
            return Ok(self.clone());
        }
        self.scaled_blocks(&self.ordinary_body().space, &|key| match key {
            BlockKey::FusionTree(key) => with_user_rule!(self.ordinary_body().space, rule, {
                twist_block_factor(rule, key, nout, legs, false)
            }),
            _ => 1.0,
        })
    }

    /// TensorKit `flip(t, I)`: return
    /// a tensor isomorphic to `self` where the duality flag of each leg in
    /// `legs` (flat indices, codomain first; a leg listed twice is flipped
    /// twice, sequentially) is toggled, `space(t', i) = flip(space(t, i))`.
    /// The stored sectors and the block layout are unchanged; each
    /// fusion-tree block picks up the Z-isomorphism phase of
    /// TensorKit's fusion-tree `flip` per flipped leg with
    /// uncoupled sector `a` and pre-flip duality `d` (χ = Frobenius-Schur
    /// phase, θ = ribbon twist; both real for every rule in scope):
    /// codomain leg → `d ? χ·θ : 1`; domain leg → `d ? χ : θ`.
    ///
    /// Like TensorKit's, this `flip` is *not* an involution: flipping the
    /// same leg twice returns to the original spaces but can scale odd
    /// blocks (e.g. by θ = −1 on fermionic legs); only `flip⁴ = id` in
    /// general.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when a leg index is out of range, or —
    ///   stricter than [`Self::twist`] — when the fusion rule has no braiding
    ///   and *any* leg is requested: a flip always needs the twist and
    ///   Frobenius-Schur coefficients.
    /// - [`Error::UnsupportedOnDevice`] for a device payload.
    /// - [`Error::Core`] / [`Error::InvalidArgument`] (with a "please report
    ///   this" message) when the toggled layout does not match the stored one
    ///   — engine invariants, not caller mistakes.
    ///
    pub fn flip(&self, legs: &[usize]) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.flip(legs);
        }
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "flip leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let hom = self.ordinary_body().space.homspace();
        with_user_rule!(self.ordinary_body().space, rule, {
            reject_unbraided_nonunit_legs(rule, hom, legs, "flip", false)
        })?;
        let nout = hom.codomain().len();
        // Sequential semantics for repeated legs (TensorKit flips one index
        // at a time): the shared helper records the duality each occurrence
        // sees alongside the toggled hom space.
        let (new_hom, occurrences) = flip_toggled_homspace(hom, legs);
        let new_space = self.ordinary_body().space.from_homspace(new_hom)?;
        check_flip_layout_identity(
            self.ordinary_body().space.structure(),
            new_space.structure(),
        )?;

        let flipped = self.scaled_blocks(new_space.raw(), &|key| match key {
            BlockKey::FusionTree(key) => with_user_rule!(self.ordinary_body().space, rule, {
                flip_block_factor(rule, key, nout, &occurrences, false)
            }),
            _ => 1.0,
        })?;
        Ok(Self::owned(
            flipped.rt.clone(),
            Arc::new(new_space),
            Arc::clone(&flipped.ordinary_body().data),
        ))
    }

    /// Clones the storage scaled block-wise by `factor_of(key)` (evaluated
    /// on the blocks of `structure_space`, whose layout must match the
    /// stored one), shared by [`Self::twist`] and [`Self::flip`].
    fn scaled_blocks(
        &self,
        structure_space: &DynamicFusionMapSpace,
        factor_of: &dyn Fn(&BlockKey) -> f64,
    ) -> Result<Self, Error> {
        let data = match self.coupled_data()? {
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
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        ))
    }

    fn with_bound(&self, space: UserBoundSpace, data: Data) -> Result<Self, Error> {
        Ok(Self::owned(
            self.rt.clone(),
            Arc::new(space),
            Arc::new(data),
        ))
    }

    /// Resolves the stored representation into this tensor's dense coupled
    /// layout. Why not require every operation to call this: compact-aware
    /// operations must preserve O(r) storage and therefore inspect
    /// [`Data::Diagonal`] before reaching this dense fallback.
    fn coupled_data(&self) -> Result<&Data, Error> {
        let body = self.materialized_body()?;
        if let Data::Diagonal(diagonal) = body.data.as_ref() {
            return Ok(self
                .compact_dense
                .get_or_init(|| Arc::new(Self::materialize_diagonal(body, diagonal)))
                .as_ref());
        }
        Ok(body.data.as_ref())
    }

    fn materialized_body(&self) -> Result<&TensorBody, Error> {
        let TensorRepr::Adjoint(view) = &self.repr else {
            return Ok(self.ordinary_body());
        };
        if let Some(body) = view.materialized.get() {
            return Ok(body);
        }
        let _guard = view
            .init
            .lock()
            .map_err(|_| Error::InvalidArgument("adjoint initializer was poisoned".to_string()))?;
        if let Some(body) = view.materialized.get() {
            return Ok(body);
        }
        let built = Self::build_adjoint_body(&view.parent)?;
        #[cfg(test)]
        view.materialized_body_builds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = view.materialized.set(built);
        view.materialized.get().map(Arc::as_ref).ok_or_else(|| {
            Error::InvalidArgument(
                "adjoint materialization completed without publishing its owned body".to_string(),
            )
        })
    }

    fn materialized_tensor(&self) -> Result<Self, Error> {
        let body = self.materialized_body()?;
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&body.space),
            Arc::clone(&body.data),
        ))
    }

    /// Builds an operation-local logical tensor without publishing the
    /// receiver's reusable materialization cache, but still constructs a full
    /// receiver-sized logical payload. Prefer an oriented kernel or algebraic
    /// redirect when one implements the same semantics.
    fn materialized_tensor_uncached(&self) -> Result<Self, Error> {
        let TensorRepr::Adjoint(view) = &self.repr else {
            return Ok(self.clone());
        };
        let body = Self::build_adjoint_body(&view.parent)?;
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&body.space),
            Arc::clone(&body.data),
        ))
    }

    fn materialized_dense_data_arc(&self, body: &TensorBody) -> Result<Arc<Data>, Error> {
        match body.data.as_ref() {
            Data::F64(_) | Data::C64(_) => Ok(Arc::clone(&body.data)),
            Data::Diagonal(_) => {
                let _ = self.coupled_data()?;
                self.compact_dense.get().cloned().ok_or_else(|| {
                    Error::InvalidArgument(
                        "compact diagonal materialization completed without publishing dense storage"
                            .to_string(),
                    )
                })
            }
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("unit-leg layout operation")),
        }
    }

    fn insert_unit(&self, insertion: UnitLegInsertion) -> Result<Self, Error> {
        let (position, operation) = match insertion {
            UnitLegInsertion::Left { position, .. } => (position, "Tensor::insert_left_unit"),
            UnitLegInsertion::Right { position, .. } => (position, "Tensor::insert_right_unit"),
        };
        if position > self.rank() {
            return Err(Error::InvalidArgument(format!(
                "{operation}: position {position} exceeds rank {}",
                self.rank()
            )));
        }
        #[cfg(feature = "cuda")]
        if matches!(self.stored_data(), Data::CudaF64(_)) {
            return Err(device_unsupported(operation));
        }

        let source = self.materialized_body()?;
        with_user_rule!(source.space, rule, {
            let homspace = match insertion {
                UnitLegInsertion::Left { position, dual } => source
                    .space
                    .homspace()
                    .insert_left_unit(rule, position, dual)?,
                UnitLegInsertion::Right { position, dual } => source
                    .space
                    .homspace()
                    .insert_right_unit(rule, position, dual)?,
            };
            let destination = source.space.from_homspace(homspace)?;
            validate_unit_layout_correspondence_checked(
                rule,
                (source.space.homspace(), source.space.structure()),
                (destination.homspace(), destination.structure()),
                insertion,
            )
            .map_err(map_checked_unit_layout_error)?;
            let data = self.materialized_dense_data_arc(source)?;
            Ok(Self::owned(self.rt.clone(), Arc::new(destination), data))
        })
    }

    fn remove_unit_layout(&self, axis: usize) -> Result<Self, Error> {
        if axis >= self.rank() {
            return Err(Error::InvalidArgument(format!(
                "Tensor::remove_unit: axis {axis} is out of range for rank {}",
                self.rank()
            )));
        }
        with_user_rule!(self.rule_authority_space(), rule, {
            let metadata = self.metadata();
            let leg = if axis < metadata.nout() {
                &metadata.codomain().legs()[axis]
            } else {
                &metadata.domain().legs()[axis - metadata.nout()]
            };
            if leg.sectors() != [rule.vacuum()] || leg.degeneracy(rule.vacuum()) != Some(1) {
                return Err(Error::InvalidArgument(format!(
                    "Tensor::remove_unit: axis {axis} is not a canonical unit leg"
                )));
            }
            Ok::<_, Error>(())
        })?;
        #[cfg(feature = "cuda")]
        if matches!(self.stored_data(), Data::CudaF64(_)) {
            return Err(device_unsupported("Tensor::remove_unit"));
        }

        let source = self.materialized_body()?;
        with_user_rule!(source.space, rule, {
            let stored_leg = if axis < source.space.nout() {
                &source.space.homspace().codomain().legs()[axis]
            } else {
                &source.space.homspace().domain().legs()[axis - source.space.nout()]
            };
            let insertion = if axis < source.space.nout() {
                UnitLegInsertion::Right {
                    position: axis,
                    dual: stored_leg.is_dual(),
                }
            } else {
                UnitLegInsertion::Left {
                    position: axis,
                    dual: stored_leg.is_dual(),
                }
            };
            let homspace = source.space.homspace().remove_unit(rule, axis)?;
            let destination = source.space.from_homspace(homspace)?;
            validate_unit_layout_correspondence_checked(
                rule,
                (destination.homspace(), destination.structure()),
                (source.space.homspace(), source.space.structure()),
                insertion,
            )
            .map_err(map_checked_unit_layout_error)?;
            let data = self.materialized_dense_data_arc(source)?;
            Ok(Self::owned(self.rt.clone(), Arc::new(destination), data))
        })
    }

    fn build_adjoint_body(parent: &TensorBody) -> Result<Arc<TensorBody>, Error> {
        macro_rules! materialize {
            ($space:expr, $variant:ident, $function:ident, $data:expr, $lift:ident) => {{
                let (space, data) = tenet_tensors::$function($space, $data)?;
                Ok::<_, Error>(Arc::new(TensorBody {
                    space: Arc::new(UserBoundSpace::$variant(space)),
                    data: Arc::new(Data::$lift(data)),
                }))
            }};
        }
        match (parent.space.as_ref(), parent.data.as_ref()) {
            (UserBoundSpace::U1(space), Data::F64(data)) => {
                materialize!(space, U1, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::U1(space), Data::C64(data)) => {
                materialize!(space, U1, adjoint_bound_dyn, data, C64)
            }
            (UserBoundSpace::CU1(space), Data::F64(data)) => {
                materialize!(space, CU1, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::CU1(space), Data::C64(data)) => {
                materialize!(space, CU1, adjoint_bound_dyn, data, C64)
            }
            (UserBoundSpace::Z2(space), Data::F64(data)) => {
                materialize!(space, Z2, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::Z2(space), Data::C64(data)) => {
                materialize!(space, Z2, adjoint_bound_dyn, data, C64)
            }
            (UserBoundSpace::ZN(space), Data::F64(data)) => {
                materialize!(space, ZN, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::ZN(space), Data::C64(data)) => {
                materialize!(space, ZN, adjoint_bound_dyn, data, C64)
            }
            (UserBoundSpace::FZ2(space), Data::F64(data)) => {
                materialize!(space, FZ2, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::FZ2(space), Data::C64(data)) => {
                materialize!(space, FZ2, adjoint_bound_dyn, data, C64)
            }
            (UserBoundSpace::SU2(space), Data::F64(data)) => {
                materialize!(space, SU2, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::SU2(space), Data::C64(data)) => {
                materialize!(space, SU2, adjoint_bound_dyn, data, C64)
            }
            (UserBoundSpace::U1FZ2(space), Data::F64(data)) => {
                materialize!(space, U1FZ2, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::U1FZ2(space), Data::C64(data)) => {
                materialize!(space, U1FZ2, adjoint_bound_dyn, data, C64)
            }
            (UserBoundSpace::FZ2U1SU2(space), Data::F64(data)) => {
                materialize!(space, FZ2U1SU2, adjoint_bound_dyn, data, F64)
            }
            (UserBoundSpace::FZ2U1SU2(space), Data::C64(data)) => {
                materialize!(space, FZ2U1SU2, adjoint_bound_dyn, data, C64)
            }
            (_, Data::Diagonal(_)) => Err(Error::InvalidArgument(
                "compact diagonal tensors do not use the lazy adjoint representation".to_string(),
            )),
            #[cfg(feature = "cuda")]
            (_, Data::CudaF64(_)) => {
                Err(device_unsupported("materializing an adjoint device tensor"))
            }
        }
    }

    /// A non-diagonal clone: `Data::Diagonal` materialized into its dense
    /// equivalent, everything else shared by `Arc` (cheap). Why not use this in
    /// compact-aware arithmetic: it would recreate the O(r²) payload those
    /// paths are designed to avoid.
    fn densified_if_diagonal(&self) -> Self {
        if !matches!(self.stored_data(), Data::Diagonal(_)) {
            return self.clone();
        }
        let data = self
            .coupled_data()
            .expect("a valid compact diagonal tensor has a total dense representation")
            .clone();
        Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        )
    }

    /// Rebuilds the dense block-diagonal buffer of a [`Data::Diagonal`] tensor in
    /// its own (`space`) layout. This is the eager fallback for dense-only
    /// consumers and reproduces the former dense diagonal tensor bit-for-bit via
    /// [`tenet_matrixalgebra::diagonal_bond_data`].
    fn materialize_diagonal(body: &TensorBody, diagonal: &DiagonalData) -> Data {
        match diagonal {
            DiagonalData::RealF64(spectrum) => Data::F64(
                tenet_matrixalgebra::diagonal_bond_data(&body.space, spectrum, &|value| value)
                    .expect("diagonal fill is total on the stored bond space"),
            ),
            DiagonalData::RealC64(spectrum) => Data::C64(
                tenet_matrixalgebra::diagonal_bond_data(&body.space, spectrum, &|value| {
                    Complex64::new(value, 0.0)
                })
                .expect("diagonal fill is total on the stored bond space"),
            ),
            DiagonalData::C64(spectrum) => Data::C64(
                tenet_matrixalgebra::diagonal_bond_data(&body.space, spectrum, &|value| value)
                    .expect("diagonal fill is total on the stored bond space"),
            ),
        }
    }

    /// The scalar type this tensor stores.
    pub fn dtype(&self) -> Dtype {
        // Discriminant only; dtype is adjoint-invariant, so read the stored
        // buffer directly (no need to materialize a lazy adjoint).
        match self.stored_data() {
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
        match self.stored_data() {
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
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .to_cuda()?;
            return parent.adjoint();
        }
        let data = match self.coupled_data()? {
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
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        ))
    }

    /// Downloads a device tensor back to host storage; a plain copy when
    /// already host-resident.
    #[cfg(feature = "cuda")]
    pub fn to_host(&self) -> Result<Self, Error> {
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .to_host()?;
            return parent.adjoint();
        }
        let data = match self.coupled_data()? {
            Data::F64(_) | Data::C64(_) => self.coupled_data()?.clone(),
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
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        ))
    }

    /// The [`Runtime`] this tensor was created from (a shared handle).
    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    /// Number of codomain legs.
    pub fn codomain_rank(&self) -> usize {
        self.metadata().nout()
    }

    /// Number of domain legs.
    pub fn domain_rank(&self) -> usize {
        self.metadata().nin()
    }

    /// Total number of legs.
    pub fn rank(&self) -> usize {
        self.metadata().rank()
    }

    /// Number of codomain (output) legs. TensorKit `numout`; alias of
    /// [`Self::codomain_rank`].
    pub fn numout(&self) -> usize {
        self.codomain_rank()
    }

    /// Number of domain (input) legs. TensorKit `numin`; alias of
    /// [`Self::domain_rank`].
    pub fn numin(&self) -> usize {
        self.domain_rank()
    }

    /// Total number of legs. TensorKit `numind`; alias of [`Self::rank`].
    pub fn numind(&self) -> usize {
        self.rank()
    }

    /// Inserts the canonical unit leg at zero-based external slot `position`.
    ///
    /// This follows TensorKit's left seam convention: the codomain/domain seam
    /// belongs to the domain side. The returned tensor has the corresponding
    /// HomSpace and block layout. Owned dense input shares its `Arc<Data>`;
    /// compact and lazy input materialize once through their existing route,
    /// then the output shares that resulting dense `Arc<Data>`.
    pub fn insert_left_unit(&self, position: usize, dual: bool) -> Result<Self, Error> {
        self.insert_unit(UnitLegInsertion::Left { position, dual })
    }

    /// Inserts the canonical unit leg at zero-based external slot `position`.
    ///
    /// This follows TensorKit's right seam convention: the codomain/domain
    /// seam belongs to the codomain side. Owned dense input shares its
    /// `Arc<Data>`; compact and lazy input materialize once through their
    /// existing route, then the output shares that resulting dense `Arc<Data>`.
    pub fn insert_right_unit(&self, position: usize, dual: bool) -> Result<Self, Error> {
        self.insert_unit(UnitLegInsertion::Right { position, dual })
    }

    /// Removes the canonical unit leg at flat external axis `axis`.
    ///
    /// The selected leg must contain exactly the vacuum sector with degeneracy
    /// one. Owned dense input shares its `Arc<Data>`; compact and lazy input
    /// materialize once through their existing route, then the output shares
    /// that resulting dense `Arc<Data>`.
    pub fn remove_unit(&self, axis: usize) -> Result<Self, Error> {
        self.remove_unit_layout(axis)
    }

    /// Number of tensors currently sharing this tensor's storage allocation.
    #[doc(hidden)]
    pub fn storage_strong_count(&self) -> usize {
        Arc::strong_count(self.stored_data_arc())
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
    /// device-resident (use `to_host()`). Both are legal tensor states, so
    /// prefer [`Self::try_data`] when the dtype/placement is not statically
    /// known — this method is the panicking half of that pair (#128).
    pub fn data(&self) -> &[f64] {
        self.try_data()
            .expect("data(): tensor is not host f64; use try_data()/data_c64()/to_host()")
    }

    /// Flat host `f64` storage, or a typed error when the tensor is not in that
    /// state. The recoverable counterpart of [`Self::data`]: a c64 tensor
    /// yields [`Error::DtypeMismatch`] and a device tensor
    /// [`Error::PlacementMismatch`] instead of panicking (#128). Same
    /// internal-packing caveats as [`Self::data`].
    pub fn try_data(&self) -> Result<&[f64], Error> {
        if self.placement() != Placement::Host {
            return Err(Error::PlacementMismatch);
        }
        match self.coupled_data()? {
            Data::F64(data) => Ok(data),
            Data::C64(_) => Err(Error::DtypeMismatch),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(Error::PlacementMismatch),
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
    /// device-resident (use `to_host()`). Both are legal tensor states, so
    /// prefer [`Self::try_data_c64`] when the dtype/placement is not
    /// statically known — this method is the panicking half of that pair
    /// (#128).
    pub fn data_c64(&self) -> &[Complex64] {
        self.try_data_c64()
            .expect("data_c64(): tensor is not host c64; use try_data_c64()/data()/to_host()")
    }

    /// Flat host [`Complex64`] storage, or a typed error when the tensor is not
    /// in that state. The recoverable counterpart of [`Self::data_c64`]: an
    /// f64 tensor yields [`Error::DtypeMismatch`] and a device tensor
    /// [`Error::PlacementMismatch`] instead of panicking (#128). Same
    /// internal-packing caveats as [`Self::data`].
    pub fn try_data_c64(&self) -> Result<&[Complex64], Error> {
        if self.placement() != Placement::Host {
            return Err(Error::PlacementMismatch);
        }
        match self.coupled_data()? {
            Data::C64(data) => Ok(data),
            Data::F64(_) => Err(Error::DtypeMismatch),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(Error::PlacementMismatch),
        }
    }

    /// Widens to a c64 tensor (imaginary parts zero); a cheap clone when the
    /// tensor already stores c64 data.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is device-resident (a legal state); prefer
    /// [`Self::try_to_c64`], the recoverable half of this pair (#128).
    pub fn to_c64(&self) -> Self {
        self.try_to_c64()
            .expect("to_c64(): tensor is device-resident; use try_to_c64()/to_host()")
    }

    /// Widens to a c64 tensor, or a typed error when widening is not possible
    /// in place: a device-resident tensor yields [`Error::PlacementMismatch`]
    /// instead of panicking (#128). The recoverable counterpart of
    /// [`Self::to_c64`].
    pub fn try_to_c64(&self) -> Result<Self, Error> {
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.to_c64_storage()));
        }
        let data = match self.coupled_data()? {
            Data::F64(data) => Data::C64(
                data.iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            Data::C64(data) => Data::C64(data.clone()),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(Error::PlacementMismatch),
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
    }

    /// A zero tensor on the same spaces and dtype as `self` (TensorKit
    /// `zerovector` / `zero`). Every stored scalar is freshly initialized to
    /// exact positive zero, independently of non-finite source values.
    pub fn zeros_like(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.parent_tensor_for_lowering().zeros_like()?.adjoint();
        }
        let data = match self.stored_data() {
            Data::F64(data) => Data::F64(vec![0.0; data.len()]),
            Data::C64(data) => Data::C64(vec![Complex64::new(0.0, 0.0); data.len()]),
            Data::Diagonal(diagonal) => Data::Diagonal(diagonal.zeros_like()),
            #[cfg(feature = "cuda")]
            Data::CudaF64(storage) => {
                let len = self.ordinary_body().space.raw().required_len()?;
                if TensorStorage::<f64>::len(storage.as_ref()) != len {
                    return Err(internal_layout_error(
                        "CUDA payload length does not match its admitted tensor space",
                    ));
                }
                let mut state = self.rt.lock();
                let cuda = require_cuda(state.cuda.as_mut())?;
                validate_cuda_zero_placement(cuda.device(), storage.placement())?;
                Data::CudaF64(Arc::new(CudaStorage::upload(cuda, &vec![0.0; len])?))
            }
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        ))
    }

    /// Quantum-dimension-weighted total dimension of every leg, in flat
    /// order (codomain legs first, then domain legs). This is the same
    /// notion as [`crate::space::Space::dim`] per leg; contraction
    /// planners use it as a size/FLOP proxy.
    pub fn leg_dims(&self) -> Result<Vec<usize>, Error> {
        let metadata = self.metadata();
        with_user_rule!(self.rule_authority_space(), rule, {
            Ok(metadata
                .codomain()
                .legs()
                .iter()
                .chain(metadata.domain().legs())
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
        let metadata = self.metadata();
        let leg = if axis < metadata.nout() {
            &metadata.codomain().legs()[axis]
        } else if axis < metadata.rank() {
            &metadata.domain().legs()[axis - metadata.nout()]
        } else {
            return Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                metadata.rank()
            )));
        };
        with_user_rule!(self.rule_authority_space(), rule, {
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
        let metadata = self.metadata();
        let nout = metadata.nout();
        if axis < nout {
            Ok(Space::from_leg(
                Arc::new(self.rule_context()),
                &metadata.codomain().legs()[axis],
            ))
        } else if axis < metadata.rank() {
            Ok(Space::from_leg(
                Arc::new(self.rule_context()),
                &metadata.domain().legs()[axis - nout],
            )
            .dual())
        } else {
            Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                metadata.rank()
            )))
        }
    }

    /// The codomain spaces, in leg order.
    pub fn codomain_spaces(&self) -> Vec<Space> {
        let metadata = self.metadata();
        let context = Arc::new(self.rule_context());
        metadata
            .codomain()
            .legs()
            .iter()
            .map(|leg| Space::from_leg(Arc::clone(&context), leg))
            .collect()
    }

    /// The domain spaces, in leg order (the spaces as written, i.e. *not*
    /// dualized; `t.space(codomain_rank() + i)` is their dual).
    pub fn domain_spaces(&self) -> Vec<Space> {
        let metadata = self.metadata();
        let context = Arc::new(self.rule_context());
        metadata
            .domain()
            .legs()
            .iter()
            .map(|leg| Space::from_leg(Arc::clone(&context), leg))
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
        match self.coupled_data()? {
            Data::F64(data) => Ok(Scalar::F64(data.iter().sum())),
            Data::C64(data) => Ok(Scalar::C64(data.iter().sum())),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scalar()")),
        }
    }

    fn check_same_world(&self, other: &Self) -> Result<(), Error> {
        self.check_same_execution_world(other)?;
        if self.dtype() != other.dtype() {
            return Err(Error::DtypeMismatch);
        }
        Ok(())
    }

    fn check_same_execution_world(&self, other: &Self) -> Result<(), Error> {
        if self.rule_authority_space().identity() != other.rule_authority_space().identity() {
            return Err(Error::RuleMismatch);
        }
        if !self.rt.same_runtime(&other.rt) {
            return Err(Error::RuntimeMismatch);
        }
        if self.placement() != other.placement() {
            return Err(Error::PlacementMismatch);
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
    /// (TensorKit twists only in
    /// `blas_contract!`, never in `mul!`). For bosonic rules the two agree
    /// exactly; for fermionic rules (fZ2 and products containing it) they
    /// can differ by signs. Worked example — the odd sector flips sign:
    ///
    ///
    /// Rule of thumb: use `compose` when you mean operator/matrix
    /// multiplication of tensor maps (TensorKit `A * B`); use
    /// [`Self::contract`] / `tensor!` when you mean index-notation
    /// contraction (TensorKit `@tensor`). Bosonic results are identical.
    ///
    /// # Complexity
    ///
    /// Dense operands: one GEMM per coupled sector. A compact-diagonal
    /// operand (an `s` from [`Self::svd_trunc`], a `d` from
    /// [`Self::eigh_full`]) takes TensorKit's `DiagonalTensorMap` route
    /// instead: `t * D` and `D * t` scale one bond axis of `t` in a single
    /// pass over `t`'s payload with the diagonal never densified, and
    /// `D * D` multiplies the two spectra elementwise in `O(Σ_c k_c)`,
    /// staying compact. Operands or destinations the compact arms cannot
    /// prove fall through to the dense route — same result, different cost.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when the ranks do not compose (lhs domain
    ///   rank ≠ rhs codomain rank).
    /// - [`Error::RuleMismatch`] / [`Error::RuntimeMismatch`] /
    ///   [`Error::PlacementMismatch`] / [`Error::DtypeMismatch`] when the
    ///   operands come from different worlds (rule, runtime, placement,
    ///   scalar type).
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    ///   from the contraction seam when the composed legs are not mutually
    ///   dual — the expert layer owns those rules.
    /// - [`Error::UnsupportedOnDevice`] where [`Self::contract`] reports it.
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
        self.check_same_world(rhs)?;
        let lhs_axes: Vec<usize> = (self.codomain_rank()..self.rank()).collect();
        let rhs_axes: Vec<usize> = (0..rhs.codomain_rank()).collect();
        let mut diagonal_dst = if self.diagonal_data().is_some() || rhs.diagonal_data().is_some() {
            Some(self.contraction_output_space(rhs, &lhs_axes, &rhs_axes)?)
        } else {
            None
        };
        // Why not send a proven diagonal composition through GEMM: TensorKit's
        // `DiagonalTensorMap` `rmul!`/`lmul!` shows it is only per-block bond
        // scaling. The compact product is valid only after the derived output
        // proves the same rank-2 diagonal invariant.
        match (self.diagonal_data(), rhs.diagonal_data()) {
            (Some(lhs), Some(rhs_diagonal)) => {
                let dst_space = diagonal_dst
                    .take()
                    .expect("diagonal destination prepared when both operands are diagonal");
                if Self::is_diagonal_bond_space(dst_space.raw()) {
                    if let Some(product) = lhs.elementwise_product(rhs_diagonal) {
                        return self.with_bound(dst_space, Data::Diagonal(product));
                    }
                }
            }
            // `t * D`: scale `self`'s trailing bond axis (columns). `self.domain`
            // is the single bond leg == `D.codomain`, so the space is `self`'s.
            (None, Some(diagonal))
                if diagonal_dst
                    .as_ref()
                    .is_some_and(|candidate| Self::space_matches_metadata(candidate, self)) =>
            {
                return self.scaled_axis_copy_diagonal(None, diagonal);
            }
            // `D * t`: scale `rhs`'s leading bond axis (rows). `rhs.codomain` is
            // the single bond leg == `D.domain`, so the space is `rhs`'s.
            (Some(diagonal), None)
                if diagonal_dst
                    .as_ref()
                    .is_some_and(|candidate| Self::space_matches_metadata(candidate, rhs)) =>
            {
                return rhs.scaled_axis_copy_diagonal(Some(0), diagonal);
            }
            _ => {}
        }
        let fermionic = with_user_rule!(self.rule_authority_space(), rule, {
            rule.braiding_style() == tenet_core::BraidingStyleKind::Fermionic
        });
        if fermionic {
            match (self.stored_data(), rhs.stored_data()) {
                (Data::F64(_), Data::F64(_)) | (Data::C64(_), Data::C64(_)) => {
                    return self.compose_host_fusion_impl(rhs, &lhs_axes, &rhs_axes);
                }
                _ => {}
            }
        }
        self.contract(rhs, &lhs_axes, &rhs_axes)
    }

    /// Integer tensor-map power (TensorKit `t ^ p`), using `O(log |p|)`
    /// compositions. Zero returns the multiplicative identity (staying compact
    /// for compact input); negative powers invert once.
    ///
    /// Returns [`Error::InvalidArgument`] unless this is an endomorphism.
    pub fn powi(&self, exponent: i32) -> Result<Self, Error> {
        let metadata = self.metadata();
        if metadata.codomain().legs() != metadata.domain().legs() {
            return Err(Error::InvalidArgument(
                "powi() requires an endomorphism (domain == codomain)".to_string(),
            ));
        }
        if exponent == 0 {
            if let Some(diagonal) = self.diagonal_data() {
                return Ok(self.with_diagonal(diagonal.ones_like()));
            }
            return Self::id(&self.rt, self.dtype(), &self.domain_spaces());
        }

        let power = if exponent < 0 {
            self.inv()?
        } else {
            self.clone()
        };
        pow_by_squaring(power, exponent.unsigned_abs(), Self::compose)
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
    ///
    /// # Complexity
    ///
    /// Dense operands: one GEMM per coupled sector. A compact-diagonal
    /// operand in one of the proven canonical rank-2 bond geometries never
    /// becomes an `O(d²)` block-diagonal fed to an `O(d²·n)` GEMM: the other
    /// operand's contracted leg is scaled by the stored spectrum (`O(d·n)`)
    /// and the result is permuted into the contract output arrangement — the
    /// same scale-plus-permute structure TensorKit runs. Geometries the
    /// compact arm has not proven (outer products, scalar-output
    /// contractions, unproved layouts) densify the diagonal and take the
    /// ordinary dense route.
    ///
    /// # Errors
    ///
    /// - [`Error::RuleMismatch`] / [`Error::RuntimeMismatch`] /
    ///   [`Error::PlacementMismatch`] / [`Error::DtypeMismatch`] when the
    ///   operands come from different worlds.
    /// - [`Error::InvalidArgument`] when the axis lists differ in length or
    ///   an axis list is malformed (out of range, repeated).
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    ///   from the contraction seam when the paired legs are not mutually
    ///   dual or the provider cannot carry the required recoupling.
    /// - [`Error::UnsupportedOnDevice`] when a device operand is a lazy
    ///   adjoint view (materialize with `to_host()` first).
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
        validate_contracted_axes(lhs_axes, self.rank())?;
        validate_contracted_axes(rhs_axes, rhs.rank())?;
        // Order-parity fast path for a real or complex diagonal operand (#75): instead of
        // densifying it to an O(d²) block-diagonal and running an O(d²·n) GEMM,
        // scale the OTHER operand's contracted leg by the spectrum (O(d·n)) and
        // `permute` the result into the contract output arrangement (O(n)). The
        // `permute` reuses the tested recoupling/repartition machinery, so the
        // result space — including leg duality and the codomain/domain split — is
        // correct for the proven canonical geometries at any leg position within
        // the preserved partition side. This is the same
        // scale + one-permute structure TensorKit runs (a `Diagonal` block scales
        // the recoupled operand); see docs/complexity_parity_policy.md.
        //
        // `contract` (tensorcontract!) applies a supertrace twist to `rhs`'s
        // externally dual contracted legs; `mul!` does not. The canonical
        // diagonal routes below therefore fold that RHS twist into the scaled
        // operand. θ = ±1 by charge parity, identity for bosonic rules.
        if let Some(output) = self.try_contract_diagonal_fast_path(
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::identity(),
        )? {
            return Ok(output);
        }
        // Why not generalize compact storage to every diagonal contraction: a
        // zero-axis outer product is rank 4 and a two-axis contraction is scalar,
        // neither fits `DiagonalData`'s rank-2 bond invariant. Those shapes and
        // any unproved rank-2 layout retain the ordinary dense fallback.
        if matches!(self.stored_data(), Data::Diagonal(_))
            || matches!(rhs.stored_data(), Data::Diagonal(_))
        {
            return self.densified_if_diagonal().contract(
                &rhs.densified_if_diagonal(),
                lhs_axes,
                rhs_axes,
            );
        }
        // Fold a lazy adjoint into contraction without copying its blocks.
        // Planning derives logical HomSpace geometry from parent storage plus
        // orientation, while execution maps only referenced blocks and strides
        // onto that storage and conjugates their numerical values.
        //
        // Why not retain a second owned adjoint layout: oriented metadata carries
        // the non-self-dual relabeling needed before the fusion-tree plan is
        // built, while the completed plan can still reference parent storage.
        match (self.stored_data(), rhs.stored_data()) {
            (Data::F64(_), Data::F64(_)) | (Data::C64(_), Data::C64(_)) => {
                self.contract_host_fusion_impl(rhs, lhs_axes, rhs_axes, OutputAxisOrder::identity())
            }
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                if self.is_adjoint_view() || rhs.is_adjoint_view() {
                    return Err(device_unsupported(
                        "contracting a lazy adjoint device tensor",
                    ));
                }
                self.contract_cuda_impl(rhs, a, b, lhs_axes, rhs_axes)
            }
            _ => Err(Error::DtypeMismatch),
        }
    }

    fn contract_host_fusion_impl(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Self, Error> {
        self.contract_host_fusion_impl_with_semantics(
            rhs,
            lhs_axes,
            rhs_axes,
            output_order,
            ContractionSemantics::TensorContract,
        )
    }

    fn compose_host_fusion_impl(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        self.contract_host_fusion_impl_with_semantics(
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::identity(),
            ContractionSemantics::Composition,
        )
    }

    fn contract_host_fusion_impl_with_semantics(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        semantics: ContractionSemantics,
    ) -> Result<Self, Error> {
        let (lhs_storage, lhs_orientation) = self.seam_operand();
        let (rhs_storage, rhs_orientation) = rhs.seam_operand();
        // The seam always consumes the raw stored buffer (it never materializes):
        // for a lazy adjoint that buffer is the shared parent, conjugated by the
        // flag; for an ordinary tensor it is just the stored data.
        match (self.stored_data(), rhs.stored_data()) {
            (Data::F64(a), Data::F64(b)) => self.contract_impl(
                lhs_storage,
                a,
                lhs_orientation,
                rhs_storage,
                b,
                rhs_orientation,
                rhs,
                lhs_axes,
                rhs_axes,
                output_order,
                semantics,
            ),
            (Data::C64(a), Data::C64(b)) => self.contract_impl(
                lhs_storage,
                a,
                lhs_orientation,
                rhs_storage,
                b,
                rhs_orientation,
                rhs,
                lhs_axes,
                rhs_axes,
                output_order,
                semantics,
            ),
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Returns parent storage layout and its logical orientation.
    fn seam_operand(&self) -> (&UserBoundSpace, FusionTreePairOrientation) {
        match &self.repr {
            TensorRepr::Owned(body) => (body.space.as_ref(), FusionTreePairOrientation::Direct),
            TensorRepr::Adjoint(view) => (
                view.parent.space.as_ref(),
                FusionTreePairOrientation::Adjoint,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn contract_impl<D: UserScalar>(
        &self,
        lhs_storage: &UserBoundSpace,
        lhs_data: &[D],
        lhs_orientation: FusionTreePairOrientation,
        rhs_storage: &UserBoundSpace,
        rhs_data: &[D],
        rhs_orientation: FusionTreePairOrientation,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        semantics: ContractionSemantics,
    ) -> Result<Self, Error> {
        // Lease an execution context so independent operations on one runtime
        // do not serialize while bound spaces remain the fusion authority.
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        if semantics == ContractionSemantics::TensorContract
            && lhs_orientation == FusionTreePairOrientation::Direct
            && rhs_orientation == FusionTreePairOrientation::Direct
        {
            macro_rules! contract_owned {
                ($variant:ident, $lhs:expr, $rhs:expr) => {{
                    let (space, data) = tensorcontract_owned_multiplicity_free(
                        context.multiplicity_free_lane::<D>(),
                        BoundDynamicTensorRef::try_new($lhs, lhs_data)?,
                        BoundDynamicTensorRef::try_new($rhs, rhs_data)?,
                        lhs_axes,
                        rhs_axes,
                        output_order,
                    )?;
                    return self.with_bound(UserBoundSpace::$variant(space), D::lift(data));
                }};
            }
            match (lhs_storage, rhs_storage) {
                (UserBoundSpace::U1(lhs), UserBoundSpace::U1(rhs)) => {
                    contract_owned!(U1, lhs, rhs)
                }
                (UserBoundSpace::CU1(lhs), UserBoundSpace::CU1(rhs)) => {
                    contract_owned!(CU1, lhs, rhs)
                }
                (UserBoundSpace::Z2(lhs), UserBoundSpace::Z2(rhs)) => {
                    contract_owned!(Z2, lhs, rhs)
                }
                (UserBoundSpace::ZN(lhs), UserBoundSpace::ZN(rhs)) => {
                    contract_owned!(ZN, lhs, rhs)
                }
                (UserBoundSpace::FZ2(lhs), UserBoundSpace::FZ2(rhs)) => {
                    contract_owned!(FZ2, lhs, rhs)
                }
                (UserBoundSpace::SU2(lhs), UserBoundSpace::SU2(rhs)) => {
                    contract_owned!(SU2, lhs, rhs)
                }
                (UserBoundSpace::U1FZ2(lhs), UserBoundSpace::U1FZ2(rhs)) => {
                    contract_owned!(U1FZ2, lhs, rhs)
                }
                (UserBoundSpace::FZ2U1SU2(lhs), UserBoundSpace::FZ2U1SU2(rhs)) => {
                    contract_owned!(FZ2U1SU2, lhs, rhs)
                }
                _ => {}
            }
        }
        let dst_bound =
            self.contraction_output_space_oriented(rhs, lhs_axes, rhs_axes, output_order)?;
        let mut data = vec![D::from_real(0.0); dst_bound.raw().required_len()?];
        let lhs_is_adjoint = lhs_orientation == FusionTreePairOrientation::Adjoint;
        let rhs_is_adjoint = rhs_orientation == FusionTreePairOrientation::Adjoint;
        let spec = TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            output_order,
            lhs_is_adjoint,
            rhs_is_adjoint,
        );
        macro_rules! contract_bound {
            ($contexts:expr, $dst:expr, $lhs_storage:expr, $rhs_storage:expr) => {{
                // Why not use the generalized prelowered route unconditionally:
                // ordinary operands must retain the established accumulation
                // order and bitwise output; only lazy operands need categorical
                // geometry separated from their parent storage.
                if semantics == ContractionSemantics::Composition {
                    let lhs = if lhs_is_adjoint {
                        tenet_tensors::FusionOperand::adjoint($lhs_storage.space())
                    } else {
                        tenet_tensors::FusionOperand::direct($lhs_storage.space())
                    };
                    let rhs = if rhs_is_adjoint {
                        tenet_tensors::FusionOperand::adjoint($rhs_storage.space())
                    } else {
                        tenet_tensors::FusionOperand::direct($rhs_storage.space())
                    };
                    D::ctx_of($contexts).tensorcompose_fusion_dyn_into(
                        $dst,
                        &mut data,
                        lhs,
                        lhs_data,
                        rhs,
                        rhs_data,
                        lhs_axes,
                        rhs_axes,
                        D::from_real(1.0),
                        D::from_real(0.0),
                    )
                } else if !lhs_is_adjoint && !rhs_is_adjoint {
                    // E1 (#586): the plain entry point. The `_lowered` context
                    // twin was a verbatim delegate to this call; the lowered
                    // cold staging travels with the bound spaces' layout
                    // capability, not with the entry-point name.
                    D::ctx_of($contexts).tensorcontract_fusion_dyn_into(
                        $dst,
                        &mut data,
                        $lhs_storage,
                        lhs_data,
                        $rhs_storage,
                        rhs_data,
                        TensorContractSpec::new(lhs_axes, rhs_axes, output_order),
                        D::from_real(1.0),
                        D::from_real(0.0),
                    )
                } else {
                    let lhs = if lhs_is_adjoint {
                        tenet_tensors::FusionOperand::adjoint($lhs_storage.space())
                    } else {
                        tenet_tensors::FusionOperand::direct($lhs_storage.space())
                    };
                    let rhs = if rhs_is_adjoint {
                        tenet_tensors::FusionOperand::adjoint($rhs_storage.space())
                    } else {
                        tenet_tensors::FusionOperand::direct($rhs_storage.space())
                    };
                    D::ctx_of($contexts).tensorcontract_fusion_dyn_prelowered_into(
                        $dst,
                        &mut data,
                        lhs,
                        lhs_data,
                        rhs,
                        rhs_data,
                        spec,
                        D::from_real(1.0),
                        D::from_real(0.0),
                    )
                }
            }};
        }
        match (&dst_bound, lhs_storage, rhs_storage) {
            (
                UserBoundSpace::U1(dst),
                UserBoundSpace::U1(lhs_storage),
                UserBoundSpace::U1(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            (
                UserBoundSpace::CU1(dst),
                UserBoundSpace::CU1(lhs_storage),
                UserBoundSpace::CU1(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            (
                UserBoundSpace::Z2(dst),
                UserBoundSpace::Z2(lhs_storage),
                UserBoundSpace::Z2(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            (
                UserBoundSpace::ZN(dst),
                UserBoundSpace::ZN(lhs_storage),
                UserBoundSpace::ZN(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            (
                UserBoundSpace::FZ2(dst),
                UserBoundSpace::FZ2(lhs_storage),
                UserBoundSpace::FZ2(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            (
                UserBoundSpace::SU2(dst),
                UserBoundSpace::SU2(lhs_storage),
                UserBoundSpace::SU2(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            (
                UserBoundSpace::U1FZ2(dst),
                UserBoundSpace::U1FZ2(lhs_storage),
                UserBoundSpace::U1FZ2(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            (
                UserBoundSpace::FZ2U1SU2(dst),
                UserBoundSpace::FZ2U1SU2(lhs_storage),
                UserBoundSpace::FZ2U1SU2(rhs_storage),
            ) => contract_bound!(&mut context.mf, dst, lhs_storage, rhs_storage),
            _ => return Err(Error::RuleMismatch),
        }?;
        let data = D::lift(data);
        self.with_bound(dst_bound, data)
    }

    /// Device contraction: same eager route compilation as the host path
    /// (spaces are host-side metadata), replayed directly on the
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
        if self.is_adjoint_view() || rhs.is_adjoint_view() {
            return Err(device_unsupported(
                "contracting a lazy adjoint device tensor",
            ));
        }
        let dst_bound = self.ordinary_body().space.contracted(
            &rhs.ordinary_body().space,
            lhs_axes,
            rhs_axes,
        )?;
        // ponytail: destination allocated by uploading host zeros; a
        // device-side alloc/memset seam replaces this if upload cost
        // ever matters (the direct route overwrites every element).
        let mut dst = CudaStorage::upload(cuda, &vec![0.0; dst_bound.raw().required_len()?])?;
        let spec = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        macro_rules! contract_cuda_bound {
            ($contexts:expr, $dst:expr, $lhs:expr, $rhs:expr) => {
                $contexts.f64.tensorcontract_fusion_dyn_direct_on_storage(
                    &mut CudaStorageGemm::new(cuda),
                    $dst,
                    &mut dst,
                    $lhs,
                    lhs_data,
                    $rhs,
                    rhs_data,
                    spec,
                )
            };
        }
        match (
            &dst_bound,
            self.ordinary_body().space.as_ref(),
            rhs.ordinary_body().space.as_ref(),
        ) {
            (UserBoundSpace::U1(dst), UserBoundSpace::U1(lhs), UserBoundSpace::U1(rhs)) => {
                contract_cuda_bound!(&mut state.mf, dst, lhs, rhs)
            }
            (UserBoundSpace::CU1(dst), UserBoundSpace::CU1(lhs), UserBoundSpace::CU1(rhs)) => {
                contract_cuda_bound!(&mut state.mf, dst, lhs, rhs)
            }
            (UserBoundSpace::Z2(dst), UserBoundSpace::Z2(lhs), UserBoundSpace::Z2(rhs)) => {
                contract_cuda_bound!(&mut state.mf, dst, lhs, rhs)
            }
            (UserBoundSpace::ZN(dst), UserBoundSpace::ZN(lhs), UserBoundSpace::ZN(rhs)) => {
                contract_cuda_bound!(&mut state.mf, dst, lhs, rhs)
            }
            (UserBoundSpace::FZ2(dst), UserBoundSpace::FZ2(lhs), UserBoundSpace::FZ2(rhs)) => {
                contract_cuda_bound!(&mut state.mf, dst, lhs, rhs)
            }
            (UserBoundSpace::SU2(dst), UserBoundSpace::SU2(lhs), UserBoundSpace::SU2(rhs)) => {
                contract_cuda_bound!(&mut state.mf, dst, lhs, rhs)
            }
            (
                UserBoundSpace::U1FZ2(dst),
                UserBoundSpace::U1FZ2(lhs),
                UserBoundSpace::U1FZ2(rhs),
            ) => contract_cuda_bound!(&mut state.mf, dst, lhs, rhs),
            (
                UserBoundSpace::FZ2U1SU2(dst),
                UserBoundSpace::FZ2U1SU2(lhs),
                UserBoundSpace::FZ2U1SU2(rhs),
            ) => contract_cuda_bound!(&mut state.mf, dst, lhs, rhs),
            _ => return Err(Error::RuleMismatch),
        }?;
        let data = Data::CudaF64(Arc::new(dst));
        drop(guard);
        self.with_bound(dst_bound, data)
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
        self.check_same_world(rhs)?;
        if lhs_axes.len() != rhs_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "contracted axis lists differ in length: {} vs {}",
                lhs_axes.len(),
                rhs_axes.len()
            )));
        }
        validate_contracted_axes(lhs_axes, self.rank())?;
        validate_contracted_axes(rhs_axes, rhs.rank())?;
        let open_rank = self.rank() - lhs_axes.len() + rhs.rank() - rhs_axes.len();

        let host_mult_free_dense = self.placement() == Placement::Host
            && !matches!(self.stored_data(), Data::Diagonal(_))
            && !matches!(rhs.stored_data(), Data::Diagonal(_));
        if host_mult_free_dense {
            let output_error = if output_axes.len() != open_rank {
                Some(Error::InvalidArgument(format!(
                    "output axis list length {} does not match open rank {}",
                    output_axes.len(),
                    open_rank
                )))
            } else {
                validate_axis_permutation(output_axes, open_rank).err()
            };
            if let Some(output_error) = output_error {
                // Why not report pAB first: the public contract historically
                // validates contracted spaces before inspecting output order.
                self.validate_oriented_contracted_homspace(rhs, lhs_axes, rhs_axes)?;
                return Err(output_error);
            }
        }
        if !host_mult_free_dense
            && (matches!(self.stored_data(), Data::Diagonal(_))
                || matches!(rhs.stored_data(), Data::Diagonal(_)))
            && output_axes.len() == open_rank
            && validate_axis_permutation(output_axes, open_rank).is_ok()
        {
            if let Some(output) = self.try_contract_diagonal_fast_path(
                rhs,
                lhs_axes,
                rhs_axes,
                OutputAxisOrder::from_axes(output_axes),
            )? {
                return Ok(output);
            }
        }
        if !host_mult_free_dense {
            // Why not force generic fusion, compact diagonal, or device storage
            // through the multiplicity-free host plan: those routes have distinct
            // complexity or placement contracts. Preserve their proven sequential
            // operation, including validation order, until each backend can consume
            // pAB directly.
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
            return contracted.permute(&output_axes[..split], &output_axes[split..]);
        }

        if output_axes.iter().copied().eq(0..open_rank) {
            return self.contract(rhs, lhs_axes, rhs_axes);
        }

        #[cfg(test)]
        observe_ordered_contract_fused_route();
        self.contract_host_fusion_impl(
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::from_axes(output_axes),
        )
    }

    /// Tensor product in one category, ordered as
    /// `codomain(self), codomain(rhs); domain(self), domain(rhs)`.
    ///
    /// The two codomain trees and the two domain trees are merged independently
    /// with F moves, without an R symbol or a dense Kronecker temporary.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOnDevice`] for device storage.
    pub fn otimes(&self, rhs: &Self) -> Result<Self, Error> {
        self.check_same_world(rhs)?;
        if self.placement() != Placement::Host {
            return Err(Error::UnsupportedOnDevice(
                "Tensor::otimes requires host storage".to_string(),
            ));
        }
        if self.is_adjoint_view() || rhs.is_adjoint_view() {
            let lhs = if self.is_adjoint_view() {
                self.materialized_tensor()?
            } else {
                self.clone()
            };
            let rhs = if rhs.is_adjoint_view() {
                rhs.materialized_tensor()?
            } else {
                rhs.clone()
            };
            return lhs.otimes(&rhs);
        }
        if matches!(self.stored_data(), Data::Diagonal(_))
            || matches!(rhs.stored_data(), Data::Diagonal(_))
        {
            return self
                .densified_if_diagonal()
                .otimes(&rhs.densified_if_diagonal());
        }

        {
            macro_rules! product {
                ($variant:ident, $data:ident, $lhs:expr, $lhs_data:expr, $rhs:expr, $rhs_data:expr) => {{
                    let (space, data) = tensorproduct_owned_multiplicity_free(
                        BoundDynamicTensorRef::try_new($lhs, $lhs_data)?,
                        BoundDynamicTensorRef::try_new($rhs, $rhs_data)?,
                    )?;
                    return self.with_bound(UserBoundSpace::$variant(space), Data::$data(data));
                }};
            }
            match (
                self.ordinary_body().space.as_ref(),
                self.stored_data(),
                rhs.ordinary_body().space.as_ref(),
                rhs.stored_data(),
            ) {
                (UserBoundSpace::U1(a), Data::F64(ad), UserBoundSpace::U1(b), Data::F64(bd)) => {
                    product!(U1, F64, a, ad, b, bd)
                }
                (UserBoundSpace::CU1(a), Data::F64(ad), UserBoundSpace::CU1(b), Data::F64(bd)) => {
                    product!(CU1, F64, a, ad, b, bd)
                }
                (UserBoundSpace::Z2(a), Data::F64(ad), UserBoundSpace::Z2(b), Data::F64(bd)) => {
                    product!(Z2, F64, a, ad, b, bd)
                }
                (UserBoundSpace::ZN(a), Data::F64(ad), UserBoundSpace::ZN(b), Data::F64(bd)) => {
                    product!(ZN, F64, a, ad, b, bd)
                }
                (UserBoundSpace::FZ2(a), Data::F64(ad), UserBoundSpace::FZ2(b), Data::F64(bd)) => {
                    product!(FZ2, F64, a, ad, b, bd)
                }
                (UserBoundSpace::SU2(a), Data::F64(ad), UserBoundSpace::SU2(b), Data::F64(bd)) => {
                    product!(SU2, F64, a, ad, b, bd)
                }
                (
                    UserBoundSpace::U1FZ2(a),
                    Data::F64(ad),
                    UserBoundSpace::U1FZ2(b),
                    Data::F64(bd),
                ) => product!(U1FZ2, F64, a, ad, b, bd),
                (
                    UserBoundSpace::FZ2U1SU2(a),
                    Data::F64(ad),
                    UserBoundSpace::FZ2U1SU2(b),
                    Data::F64(bd),
                ) => product!(FZ2U1SU2, F64, a, ad, b, bd),
                (UserBoundSpace::U1(a), Data::C64(ad), UserBoundSpace::U1(b), Data::C64(bd)) => {
                    product!(U1, C64, a, ad, b, bd)
                }
                (UserBoundSpace::CU1(a), Data::C64(ad), UserBoundSpace::CU1(b), Data::C64(bd)) => {
                    product!(CU1, C64, a, ad, b, bd)
                }
                (UserBoundSpace::Z2(a), Data::C64(ad), UserBoundSpace::Z2(b), Data::C64(bd)) => {
                    product!(Z2, C64, a, ad, b, bd)
                }
                (UserBoundSpace::ZN(a), Data::C64(ad), UserBoundSpace::ZN(b), Data::C64(bd)) => {
                    product!(ZN, C64, a, ad, b, bd)
                }
                (UserBoundSpace::FZ2(a), Data::C64(ad), UserBoundSpace::FZ2(b), Data::C64(bd)) => {
                    product!(FZ2, C64, a, ad, b, bd)
                }
                (UserBoundSpace::SU2(a), Data::C64(ad), UserBoundSpace::SU2(b), Data::C64(bd)) => {
                    product!(SU2, C64, a, ad, b, bd)
                }
                (
                    UserBoundSpace::U1FZ2(a),
                    Data::C64(ad),
                    UserBoundSpace::U1FZ2(b),
                    Data::C64(bd),
                ) => product!(U1FZ2, C64, a, ad, b, bd),
                (
                    UserBoundSpace::FZ2U1SU2(a),
                    Data::C64(ad),
                    UserBoundSpace::FZ2U1SU2(b),
                    Data::C64(bd),
                ) => product!(FZ2U1SU2, C64, a, ad, b, bd),
                _ => {}
            }
        }

        Err(Error::InvalidArgument(
            "tensor-product host dispatch does not match tensor storage".to_string(),
        ))
    }

    /// TensorKit `permute`: re-arranges legs with symmetric braiding.
    /// `codomain_axes` and `domain_axes` list source axis numbers
    /// (`0..rank`, codomain axes first) for the new codomain and domain.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] when
    /// the axis lists are malformed (out of range, repeated, or not a
    /// partition of `0..rank`) or the provider cannot support the braiding
    /// the requested motion needs — the expert layer owns that validation,
    /// exactly as on the typed sibling. Plus [`Error::UnsupportedOnDevice`]
    /// for a device payload.
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
    /// [`Self::transpose_axes`] with reversed domain axes as the new codomain
    /// and reversed codomain axes as the new domain.
    pub fn transpose(&self) -> Result<Self, Error> {
        with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            PlanarRequestKind::FullTranspose,
            |codomain_axes, domain_axes| {
                self.transformed(codomain_axes, domain_axes, TransformKind::Transpose)
            },
        )
    }

    /// TensorKit `transpose` with an explicit cyclic axis map.
    ///
    /// The Rust name distinguishes this from [`Self::transpose`], which uses
    /// TensorKit's conventional full planar transpose. `codomain_axes` and
    /// `domain_axes` must together describe one cyclic rotation of the planar
    /// source axes. They are zero-based flat source axis numbers, with
    /// codomain axes first and domain axes second; unlike [`Self::permute`],
    /// this operation never braids legs.
    pub fn transpose_axes(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, Error> {
        with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            PlanarRequestKind::Explicit {
                codomain_axes,
                domain_axes,
            },
            |codomain_axes, domain_axes| {
                self.transformed(codomain_axes, domain_axes, TransformKind::Transpose)
            },
        )
    }

    /// TensorKit `repartition(t, N₁, N₂)`: move the planar boundary so the
    /// codomain holds `num_codomain` legs and the domain holds the rest. The
    /// boundary order is codomain followed by reversed domain; legs which cross
    /// the boundary are bent without introducing a symmetric braid.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `num_codomain` exceeds the rank, and
    /// otherwise as [`Self::permute`]: [`Error::Operation`] /
    /// [`Error::Core`] / [`Error::FusionAlgebra`] from the expert layer,
    /// plus [`Error::UnsupportedOnDevice`] for a device payload.
    pub fn repartition(&self, num_codomain: usize) -> Result<Self, Error> {
        if num_codomain == self.codomain_rank() {
            return Ok(self.clone());
        }

        with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            PlanarRequestKind::Repartition { num_codomain },
            |codomain_axes, domain_axes| {
                // Why not identity `permute`: domain trees run opposite to the planar
                // boundary, and flattening them would braid a different leg across it.
                self.transformed(codomain_axes, domain_axes, TransformKind::Transpose)
            },
        )
    }

    fn transformed(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        kind: TransformKind<'_>,
    ) -> Result<Self, Error> {
        let rank = self.rank();
        let nout = self.codomain_rank();
        if let TransformKind::Braid { levels } = &kind {
            if levels.len() != rank {
                return Err(Error::InvalidArgument(format!(
                    "braid levels must list one level per source axis \
                     (expected {rank}, got {})",
                    levels.len()
                )));
            }
        }
        // Identity tree transforms have no axis motion or adjacent braid swaps,
        // so return the tensor unchanged and share its owned storage. Levels
        // cannot contribute a phase when there is no crossing.
        let shares_identity_storage = matches!(
            &kind,
            TransformKind::Permute | TransformKind::Braid { .. } | TransformKind::Transpose
        ) && codomain_axes.iter().copied().eq(0..nout)
            && domain_axes.iter().copied().eq(nout..rank);
        if shares_identity_storage {
            return Ok(self.clone());
        }
        let operation = match &kind {
            TransformKind::Permute => TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
            TransformKind::Braid { levels } => TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            ),
            TransformKind::Transpose => TreeTransformOperation::transpose(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
        };
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent_nout = view.parent.space.homspace().codomain().len();
            let parent_nin = view.parent.space.homspace().domain().len();
            let lowered =
                lower_adjoint_tree_transform_operation(parent_nout, parent_nin, &operation)?;
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            );
            let transformed = with_data!(parent, data, parent.transformed_impl(data, lowered))?;
            return transformed.adjoint();
        }

        if let Data::Diagonal(diagonal) = self.stored_data() {
            // Why not include explicit braid or Generic fusion: their compact
            // single-term scalar predicates are not proved, so they retain the
            // existing dense fallback. The geometry test is shared with the
            // typed facade so the two cannot drift apart on which swaps are
            // compact.
            let is_rank_one_swap = crate::tensor_core::is_rank_one_diagonal_swap(
                self.codomain_rank(),
                self.domain_rank(),
                &operation,
            );
            if is_rank_one_swap {
                let destination = self.ordinary_body().space.transformed(&operation)?;
                let data = with_user_rule!(self.ordinary_body().space, rule, {
                    diagonal.transformed_rank_one_swap(
                        rule,
                        self.ordinary_body().space.raw(),
                        destination.raw(),
                        &operation,
                    )
                })?;
                return self.with_bound(destination, Data::Diagonal(data));
            }
        }

        with_data!(self, data, self.transformed_impl(data, operation))
    }

    fn transformed_impl<D: UserScalar>(
        &self,
        src_data: &[D],
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        // Tree transforms use a leased context and retain the source provider
        // proof in the derived destination.
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let (dst_bound, data) = tree_transform_owned_multiplicity_free(
                context.multiplicity_free_lane::<D>(),
                BoundDynamicTensorRef::try_new(bound, src_data)?,
                operation,
            )?;
            self.with_bound(
                UserBoundSpace::from_bound(&self.ordinary_body().space, dst_bound)?,
                D::lift(data),
            )
        })
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
        #[cfg(feature = "cuda")]
        if matches!(self.stored_data(), Data::CudaF64(_)) {
            return Err(device_unsupported("Tensor::trace_pairs"));
        }
        if pairs.is_empty() {
            return Ok(self.clone());
        }
        let output_axes: Vec<usize> = (0..rank).filter(|&axis| !seen[axis]).collect();
        let dst_codomain_rank = output_axes
            .iter()
            .filter(|&&axis| axis < self.codomain_rank())
            .count();
        let trace_lhs: Vec<usize> = pairs.iter().map(|&(lhs, _)| lhs).collect();
        let trace_rhs: Vec<usize> = pairs.iter().map(|&(_, rhs)| rhs).collect();
        let source_conjugate = self.is_adjoint_view();
        let source = self.parent_body_for_lowering();
        let parent_nout = source.space.homspace().codomain().len();
        let parent_nin = source.space.homspace().domain().len();
        let operation_output_axes = if source_conjugate {
            Cow::Owned(logical_adjoint_axes_to_parent(
                parent_nout,
                parent_nin,
                &output_axes,
            ))
        } else {
            Cow::Borrowed(output_axes.as_slice())
        };
        let operation_trace_lhs = if source_conjugate {
            Cow::Owned(logical_adjoint_axes_to_parent(
                parent_nout,
                parent_nin,
                &trace_lhs,
            ))
        } else {
            Cow::Borrowed(trace_lhs.as_slice())
        };
        let operation_trace_rhs = if source_conjugate {
            Cow::Owned(logical_adjoint_axes_to_parent(
                parent_nout,
                parent_nin,
                &trace_rhs,
            ))
        } else {
            Cow::Borrowed(trace_rhs.as_slice())
        };
        let operation = tenet_tensors::TensorTraceAxisSpec::new_with_conjugation(
            &operation_output_axes,
            &operation_trace_lhs,
            &operation_trace_rhs,
            source_conjugate,
        );
        let hom = with_bound_multiplicity_free!(source.space, bound, {
            tenet_tensors::tensortrace_fusion_dyn_selected_homspace_checked(
                bound,
                operation,
                dst_codomain_rank,
            )
            .map_err(map_trace_preflight_error)
        })?;
        // Compact arm: the full trace of a rank-(1,1) tensor over its only pair
        // is a reduction of the stored diagonal, so there is nothing to
        // materialize (#585). This is the *categorical* trace, not `tr()`:
        // TensorOperations' `tensortrace!` carries the quantum dimension of the
        // traced channel and the fermionic twist of its orientation, which is
        // what makes it the supertrace for a fermionic rule and what makes the
        // coefficient here `dim(c) * θ(c)` rather than `tr()`'s `dim(c)`.
        //
        // Why the guard is this narrow: with one pair and rank two the
        // destination is the empty tree, so the traced channel is a single
        // uncoupled sector and the coefficient collapses to a per-sector
        // scalar. Any wider geometry leaves an open destination tree whose
        // recoupling is not a per-sector scaling of a diagonal. The
        // adjoint-view exclusion is defensive and unreachable today, for the
        // reason spelled out at the compact `is_posdef` arm: `adjoint`
        // short-circuits `Data::Diagonal` and never builds a view over it.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            if !source_conjugate && rank == 2 && self.codomain_rank() == 1 && pairs.len() == 1 {
                // The traced channel's twist is applied exactly when the traced
                // leg is *not* dual — `tensortrace`'s own rule for the
                // uncoupled legs of a trace channel (see
                // `tenet_tensors::tensortrace`'s `trace_channel_factor`), and
                // the reason a compact `transpose`, which flips both bond legs,
                // trades the supertrace for the ordinary one. The whole
                // coefficient is pinned against the engine route by the value
                // oracles in `compact_diagonal_tests.rs`, which sweep both
                // orientations on a fermionic rule; it is not derivable from
                // `tr()`, whose weight is `dim(c)` unconditionally.
                let traced_leg_is_dual =
                    self.ordinary_body().space.homspace().codomain().legs()[0].is_dual();
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    diagonal.ordinary_trace_with(|sector| {
                        if traced_leg_is_dual {
                            rule.dim_scalar(sector)
                        } else {
                            rule.dim_scalar(sector) * rule.twist_scalar(sector)
                        }
                    })
                });
                let dst_bound = source.space.from_selected_homspace(hom)?;
                if dst_bound.raw().required_len()? != 1 {
                    return Err(internal_layout_error(
                        "a fully traced rank-one destination is not a single scalar",
                    ));
                }
                let data = match diagonal {
                    DiagonalData::RealF64(_) => Data::F64(vec![value.re]),
                    DiagonalData::RealC64(_) | DiagonalData::C64(_) => Data::C64(vec![value]),
                };
                return self.with_bound(dst_bound, data);
            }
        }
        let data = if source_conjugate {
            self.stored_data()
        } else {
            self.coupled_data()?
        };
        match data {
            Data::F64(data) => self.trace_pairs_impl(data, hom, operation),
            Data::C64(data) => self.trace_pairs_impl(data, hom, operation),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("Tensor::trace_pairs")),
        }
    }

    fn trace_pairs_impl<D: UserScalar>(
        &self,
        src_data: &[D],
        hom: FusionTreeHomSpace,
        operation: tenet_tensors::TensorTraceAxisSpec<'_>,
    ) -> Result<Self, Error> {
        let source = self.parent_body_for_lowering();
        let dst_bound = source.space.from_selected_homspace(hom)?;
        macro_rules! trace_bound {
            ($dst:expr, $src:expr) => {
                tenet_tensors::tensortrace_fusion_dyn_owned_checked(
                    $dst,
                    $src,
                    src_data,
                    operation,
                    D::from_real(1.0),
                )
            };
        }
        let data = match (&dst_bound, source.space.as_ref()) {
            (UserBoundSpace::U1(dst), UserBoundSpace::U1(src)) => trace_bound!(dst, src),
            (UserBoundSpace::CU1(dst), UserBoundSpace::CU1(src)) => trace_bound!(dst, src),
            (UserBoundSpace::Z2(dst), UserBoundSpace::Z2(src)) => trace_bound!(dst, src),
            (UserBoundSpace::ZN(dst), UserBoundSpace::ZN(src)) => trace_bound!(dst, src),
            (UserBoundSpace::FZ2(dst), UserBoundSpace::FZ2(src)) => trace_bound!(dst, src),
            (UserBoundSpace::SU2(dst), UserBoundSpace::SU2(src)) => trace_bound!(dst, src),
            (UserBoundSpace::U1FZ2(dst), UserBoundSpace::U1FZ2(src)) => {
                trace_bound!(dst, src)
            }
            (UserBoundSpace::FZ2U1SU2(dst), UserBoundSpace::FZ2U1SU2(src)) => {
                trace_bound!(dst, src)
            }
            _ => return Err(Error::RuleMismatch),
        }?;
        let data = D::lift(data);
        self.with_bound(dst_bound, data)
    }

    /// TensorKit `tr`: full trace of an endomorphism (`domain == codomain`)
    /// to a scalar, pairing codomain leg `i` with domain leg `i`. The
    /// returned [`Scalar`] variant matches [`Self::dtype`]. This is TensorKit's
    /// positive/ordinary trace; [`Self::trace_pairs`] retains the fermionic
    /// supertrace semantics used by tensor contractions.
    ///
    /// # Complexity
    ///
    /// One pass over the coupled-block diagonals of a dense payload — no
    /// recoupling, no destination tensor. Compact diagonal storage is traced
    /// directly on its `O(Σ_c k_c)` stored spectrum, never materialized.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when the tensor is not an endomorphism.
    /// - [`Error::UnsupportedOnDevice`] for a device payload.
    /// - [`Error::Core`] when the stored block layout cannot be walked — an
    ///   engine invariant, not a caller mistake.
    pub fn tr(&self) -> Result<Scalar, Error> {
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            );
            return Ok(match parent.tr()? {
                Scalar::F64(value) => Scalar::F64(value),
                Scalar::C64(value) => Scalar::C64(value.conj()),
            });
        }
        let hom = self.ordinary_body().space.homspace();
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
        if let Data::Diagonal(diagonal) = self.stored_data() {
            let value = with_user_rule!(
                self.ordinary_body().space,
                rule,
                diagonal.ordinary_trace(rule)
            );
            return Ok(match diagonal {
                DiagonalData::RealF64(_) => Scalar::F64(value.re),
                DiagonalData::RealC64(_) | DiagonalData::C64(_) => Scalar::C64(value),
            });
        }
        match self.coupled_data()? {
            Data::F64(data) => with_user_rule!(self.ordinary_body().space, rule, {
                weighted_trace(rule, self.ordinary_body().space.structure(), nout, data)
                    .map(|v| Scalar::F64(v.re))
            }),
            Data::C64(data) => with_user_rule!(self.ordinary_body().space, rule, {
                weighted_trace(rule, self.ordinary_body().space.structure(), nout, data)
                    .map(Scalar::C64)
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
    /// conjugated here. The result shares the parent buffer and reverses the
    /// borrowed space orientation in O(1). An explicit payload read such as
    /// `data` materializes one shared owned adjoint on demand; decomposition
    /// fallbacks instead use operation-local logical copies and leave that
    /// reusable receiver cache cold. Compact diagonal storage is handled
    /// directly and does not use this lazy dense-adjoint route.
    pub fn adjoint(&self) -> Result<Self, Error> {
        if let Data::Diagonal(diagonal) = self.stored_data() {
            // Why not use the lazy dense-adjoint wrapper: real compact spectra
            // are self-adjoint and can share their Data Arc; only genuinely
            // complex entries require an owned O(r) conjugated result.
            return Ok(match diagonal {
                DiagonalData::RealF64(_) | DiagonalData::RealC64(_) => self.clone(),
                DiagonalData::C64(_) => self.with_diagonal(diagonal.conjugated_complex()?),
            });
        }
        Ok(match &self.repr {
            TensorRepr::Owned(parent) => Self {
                rt: self.rt.clone(),
                repr: TensorRepr::Adjoint(Arc::new(AdjointView {
                    parent: parent.clone(),
                    materialized: OnceLock::new(),
                    init: Mutex::new(()),
                    #[cfg(test)]
                    materialized_body_builds: std::sync::atomic::AtomicUsize::new(0),
                })),
                compact_dense: OnceLock::new(),
            },
            TensorRepr::Adjoint(view) => Self {
                rt: self.rt.clone(),
                repr: TensorRepr::Owned(view.parent.clone()),
                compact_dense: OnceLock::new(),
            },
        })
    }

    /// Frobenius norm, weighted by coupled-sector quantum dimensions
    /// (`norm(t)^2 = sum_c dim(c) * |block_c|^2`), matching TensorKit's
    /// `norm`. Always real, for both dtypes.
    ///
    /// # Complexity
    ///
    /// One pass over the payload: `O(N)` for a dense payload of `N` scalars,
    /// `O(Σ_c k_c)` on compact diagonal storage — the stored spectrum is
    /// reduced directly, never materialized. A lazy adjoint delegates to its
    /// parent (conjugate-transposing blocks changes no magnitude), so it
    /// pays no materialization either.
    ///
    /// # Errors
    ///
    /// No caller-input failure mode: every well-formed tensor — dense,
    /// compact diagonal, device, lazy adjoint — has a norm. What remains is
    /// [`Error::Core`] / [`Error::Operation`] when the stored block layout
    /// cannot be walked or the device reduction fails: engine invariants and
    /// execution failures, not argument errors.
    pub fn norm(&self) -> Result<f64, Error> {
        if let TensorRepr::Adjoint(view) = &self.repr {
            return Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .norm();
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return Ok(self.weighted_inner_cuda(storage, storage)?.re.sqrt());
        }
        if let Data::Diagonal(diagonal) = self.stored_data() {
            macro_rules! reduce {
                ($weight:expr) => {
                    match diagonal {
                        DiagonalData::RealF64(spectrum) => compact_inner_with_weight(
                            spectrum,
                            spectrum,
                            $weight,
                            |value| value,
                            |value| value,
                        ),
                        DiagonalData::RealC64(spectrum) => compact_inner_with_weight(
                            spectrum,
                            spectrum,
                            $weight,
                            |value| Complex64::new(value, 0.0),
                            |value| Complex64::new(value, 0.0),
                        ),
                        DiagonalData::C64(spectrum) => compact_inner_with_weight(
                            spectrum,
                            spectrum,
                            $weight,
                            |value| value,
                            |value| value,
                        ),
                    }
                };
            }
            let value = with_user_rule!(self.ordinary_body().space, rule, {
                reduce!(|sector| rule.dim_scalar(sector))
            })
            .ok_or_else(|| {
                internal_layout_error("a diagonal spectrum is incompatible with itself")
            })?;
            return Ok(value.re.sqrt());
        }
        let value = with_data!(self, data, {
            with_user_rule!(self.ordinary_body().space, rule, {
                weighted_inner(
                    rule,
                    self.ordinary_body().space.structure(),
                    self.ordinary_body().space.nout(),
                    data,
                    data,
                )
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
        if let TensorRepr::Adjoint(view) = &self.repr {
            return Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .norm_inf();
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(_) = self.stored_data() {
            return Err(device_unsupported("norm_inf()"));
        }
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(diagonal.max_abs());
        }
        match self.coupled_data()? {
            Data::F64(data) => Ok(data.iter().map(|value| value.abs()).fold(0.0, f64::max)),
            Data::C64(data) => Ok(data.iter().map(|value| value.norm()).fold(0.0, f64::max)),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => unreachable!("returned above"),
        }
    }

    /// TensorKit `norm(t, p)` for a general exponent:
    ///
    /// ```text
    /// p == Inf     -> maximum entry magnitude over blocks(t)
    /// finite p > 0 -> (Σ_c dim(c) * norm(block_c, p)^p)^(1/p)
    /// ```
    ///
    /// `norm(block, p)` is Julia's *entrywise* p-norm — including for matrices,
    /// so this is never an operator norm. Only `p == 2` is the quantum-dimension
    /// weighted Frobenius norm; every other exponent weights the same `dim(c)`
    /// against a different power sum.
    ///
    /// A separate method rather than an optional argument because Rust has no
    /// overloading: [`Self::norm`] stays the zero-argument two-norm, and
    /// `p == 2.0` / `p == f64::INFINITY` delegate to [`Self::norm`] and
    /// [`Self::norm_inf`] so the three never drift apart.
    ///
    /// # Complexity
    ///
    /// One pass over the payload: `O(N)` for a dense payload of `N` scalars,
    /// and `O(Σ_c k_c)` on compact diagonal storage — the stored spectra are
    /// read directly, never materialized, since the `k_c² − k_c` off-diagonal
    /// zeros contribute nothing to any `p > 0`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] for `p` that is NaN, zero, negative, or
    ///   `-inf`. TensorKit throws `ArgumentError` on the same domain.
    /// - Whatever [`Self::norm`] reports for a payload whose block structure
    ///   cannot be walked, plus device-unsupported on CUDA storage.
    pub fn norm_p(&self, p: f64) -> Result<f64, Error> {
        // Domain check first, so an invalid `p` is rejected identically on
        // every storage and rule path instead of only where it happens to be
        // reached.
        validate_norm_p(p)?;
        if p == 2.0 {
            return self.norm();
        }
        if p.is_infinite() {
            return self.norm_inf();
        }
        if let TensorRepr::Adjoint(view) = &self.repr {
            // `t†` conjugate-transposes each block, which permutes a coupled
            // sector's entries and conjugates them — neither changes `|x|`, so
            // the parent's answer is this one. Same delegation `norm` makes.
            return Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .norm_p(p);
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(_) = self.stored_data() {
            return Err(device_unsupported("norm_p()"));
        }
        if let Data::Diagonal(diagonal) = self.stored_data() {
            let total = with_user_rule!(self.ordinary_body().space, rule, {
                diagonal.abs_pow_sum_with(p, |sector| rule.dim_scalar(sector))
            });
            return Ok(total.powf(p.recip()));
        }
        with_data!(self, data, {
            with_user_rule!(self.ordinary_body().space, rule, {
                coupled_region_pow_sum(
                    self.ordinary_body().space.structure(),
                    self.ordinary_body().space.nout(),
                    data,
                    p,
                    |coupled| rule.dim_scalar(coupled),
                )
            })
        })
    }

    /// Returns `factor * self` (real factor, both dtypes). Use
    /// [`Self::scale_c64`] for a complex factor.
    pub fn scale(&self, factor: f64) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            return self.parent_tensor_for_lowering().scale(factor)?.adjoint();
        }
        // Scaling a diagonal stays diagonal (O(rank)); itebd normalizes λ this
        // way, and keeping it diagonal lets the next contract scale the bond.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.scaled(factor)));
        }
        let data = match self.coupled_data()? {
            Data::F64(data) => Data::F64(data.iter().map(|&value| value * factor).collect()),
            Data::C64(data) => Data::C64(data.iter().map(|&value| value * factor).collect()),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(storage) => {
                Data::CudaF64(Arc::new(self.axpby_cuda(factor, storage, None)?))
            }
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
    }

    /// Returns `factor * self` for a c64 tensor. Errors with
    /// [`Error::DtypeMismatch`] on f64 tensors (widen with
    /// [`Self::to_c64`] first).
    pub fn scale_c64(&self, factor: Complex64) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            if self.dtype() != Dtype::C64 {
                return Err(Error::DtypeMismatch);
            }
            return self
                .parent_tensor_for_lowering()
                .scale_c64(factor.conj())?
                .adjoint();
        }
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.scaled_c64(factor)?));
        }
        match self.coupled_data()? {
            Data::C64(data) => Ok(Self::owned(
                self.rt.clone(),
                Arc::clone(&self.materialized_body()?.space),
                Arc::new(Data::C64(
                    data.iter().map(|&value| value * factor).collect(),
                )),
            )),
            Data::F64(_) => Err(Error::DtypeMismatch),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scale_c64()")),
        }
    }

    /// Returns `alpha * self + beta * other` (real coefficients, both
    /// dtypes). Both tensors must live on the same spaces (identical hom
    /// space and block layout) and store the same dtype.
    ///
    /// TensorKit's counterpart is `VectorInterface.add(ty, tx, α, β)`
    /// (also behind `t1 + t2`), which
    /// computes `β*ty + α*tx` — a **false friend**: there the first
    /// coefficient `α` belongs to the *second* argument. Here `alpha`
    /// belongs to `self` and `beta` to `other`; go by argument order, not
    /// coefficient names.
    ///
    /// # Complexity
    ///
    /// One elementwise pass, `O(N)` on dense payloads. Two compact-diagonal
    /// operands combine spectrum-to-spectrum in `O(Σ_c k_c)` and **stay
    /// compact**; diagonal + dense allocates only the one owned `O(N)` dense
    /// result, scattering the spectrum onto its diagonal without densifying
    /// the diagonal operand separately.
    ///
    /// # Errors
    ///
    /// - [`Error::RuleMismatch`] / [`Error::RuntimeMismatch`] /
    ///   [`Error::PlacementMismatch`] / [`Error::DtypeMismatch`] when the
    ///   operands come from different worlds.
    /// - [`Error::InvalidArgument`] when they do not live on the same space
    ///   or block layout.
    pub fn add(&self, other: &Self, alpha: f64, beta: f64) -> Result<Self, Error> {
        if self.is_adjoint_view() || other.is_adjoint_view() {
            self.check_same_logical_space(other)?;
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_))
                || matches!(other.stored_data(), Data::CudaF64(_))
            {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            if self.is_adjoint_view() && other.is_adjoint_view() {
                return self
                    .parent_tensor_for_lowering()
                    .add(&other.parent_tensor_for_lowering(), alpha, beta)?
                    .adjoint();
            }
            return match self.dtype() {
                Dtype::F64 => self.oriented_add_adjoint(other, alpha, beta),
                Dtype::C64 => self.oriented_add_adjoint(
                    other,
                    Complex64::new(alpha, 0.0),
                    Complex64::new(beta, 0.0),
                ),
            };
        }
        self.check_same_space(other)?;
        match (self.diagonal_data(), other.diagonal_data()) {
            (Some(lhs), Some(rhs)) => {
                let data = lhs.axpby_real(rhs, alpha, beta).ok_or_else(|| {
                    internal_layout_error("equal diagonal spaces carry incompatible spectra")
                })?;
                return Ok(self.with_diagonal(data));
            }
            (Some(diagonal), None) => {
                // Why not materialize `diagonal`: the owned dense result is the
                // only O(n²) allocation required by diagonal+dense addition.
                let data = axpby_dense_real(
                    &self.ordinary_body().space,
                    other.coupled_data()?,
                    diagonal,
                    beta,
                    alpha,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, Some(diagonal)) => {
                let data = axpby_dense_real(
                    &self.ordinary_body().space,
                    self.coupled_data()?,
                    diagonal,
                    alpha,
                    beta,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, None) => {}
        }
        let data = match (self.coupled_data()?, other.coupled_data()?) {
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
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
    }

    /// Returns `alpha * self + beta * other` with complex coefficients; both
    /// tensors must be c64 (widen with [`Self::to_c64`] first).
    pub fn add_c64(&self, other: &Self, alpha: Complex64, beta: Complex64) -> Result<Self, Error> {
        if self.is_adjoint_view() || other.is_adjoint_view() {
            self.check_same_world(other)?;
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_))
                || matches!(other.stored_data(), Data::CudaF64(_))
            {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            if self.dtype() != Dtype::C64 {
                return Err(Error::DtypeMismatch);
            }
            self.check_same_logical_space(other)?;
            if self.is_adjoint_view() && other.is_adjoint_view() {
                return self
                    .parent_tensor_for_lowering()
                    .add_c64(
                        &other.parent_tensor_for_lowering(),
                        alpha.conj(),
                        beta.conj(),
                    )?
                    .adjoint();
            }
            return self.oriented_add_adjoint(other, alpha, beta);
        }
        self.check_same_space(other)?;
        match (self.diagonal_data(), other.diagonal_data()) {
            (Some(lhs), Some(rhs)) => {
                let data = lhs
                    .axpby_c64(rhs, alpha, beta)
                    .ok_or(Error::DtypeMismatch)?;
                return Ok(self.with_diagonal(data));
            }
            (Some(diagonal), None) => {
                let data = axpby_dense_c64(
                    &self.ordinary_body().space,
                    other.coupled_data()?,
                    diagonal,
                    beta,
                    alpha,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, Some(diagonal)) => {
                let data = axpby_dense_c64(
                    &self.ordinary_body().space,
                    self.coupled_data()?,
                    diagonal,
                    alpha,
                    beta,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, None) => {}
        }
        match (self.coupled_data()?, other.coupled_data()?) {
            (Data::C64(a), Data::C64(b)) => Ok(Self::owned(
                self.rt.clone(),
                Arc::clone(&self.materialized_body()?.space),
                Arc::new(Data::C64(
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| alpha * x + beta * y)
                        .collect(),
                )),
            )),
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
        if self.is_adjoint_view() || other.is_adjoint_view() {
            self.check_same_logical_space(other)?;
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_))
                || matches!(other.stored_data(), Data::CudaF64(_))
            {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            if self.is_adjoint_view()
                && other.is_adjoint_view()
                && self.parent_layout_supports_inner_identity()?
                && other.parent_layout_supports_inner_identity()?
            {
                return other
                    .parent_tensor_for_lowering()
                    .inner(&self.parent_tensor_for_lowering());
            }
            let lhs = self.metadata();
            let rhs = other.metadata();
            let (logical, lhs_operand, rhs_operand, swap_dense_data) =
                oriented_inner_layout(&lhs, &rhs);
            macro_rules! reduce {
                ($weight:expr) => {
                    match (lhs.body.data.as_ref(), rhs.body.data.as_ref()) {
                        (Data::Diagonal(diagonal), parent_data) => {
                            oriented_dense_inner_with_weight(
                                lhs.body.space.raw(),
                                diagonal,
                                rhs.body.space.raw(),
                                parent_data,
                                true,
                                $weight,
                            )
                            .map(|value| match self.dtype() {
                                Dtype::F64 => Scalar::F64(value.re),
                                Dtype::C64 => Scalar::C64(value),
                            })
                        }
                        (parent_data, Data::Diagonal(diagonal)) => {
                            oriented_dense_inner_with_weight(
                                rhs.body.space.raw(),
                                diagonal,
                                lhs.body.space.raw(),
                                parent_data,
                                false,
                                $weight,
                            )
                            .map(|value| match self.dtype() {
                                Dtype::F64 => Scalar::F64(value.re),
                                Dtype::C64 => Scalar::C64(value),
                            })
                        }
                        (Data::F64(lhs_data), Data::F64(rhs_data)) => oriented_dense_inner(
                            logical,
                            lhs_operand,
                            lhs_data,
                            rhs_operand,
                            rhs_data,
                            swap_dense_data,
                            $weight,
                        )
                        .map(|value| Scalar::F64(value.re)),
                        (Data::C64(lhs_data), Data::C64(rhs_data)) => oriented_dense_inner(
                            logical,
                            lhs_operand,
                            lhs_data,
                            rhs_operand,
                            rhs_data,
                            swap_dense_data,
                            $weight,
                        )
                        .map(Scalar::C64),
                        #[cfg(feature = "cuda")]
                        (Data::CudaF64(_), Data::CudaF64(_)) => {
                            Err(device_unsupported("inner() with one lazy adjoint"))
                        }
                        _ => Err(Error::DtypeMismatch),
                    }
                };
            }
            return with_user_rule!(self.rule_authority_space(), rule, {
                reduce!(|sector| rule.dim_scalar(sector))
            });
        }
        self.check_same_space(other)?;
        match (self.diagonal_data(), other.diagonal_data()) {
            (Some(lhs), Some(rhs)) => {
                macro_rules! reduce {
                    ($weight:expr) => {
                        match (lhs, rhs) {
                            (DiagonalData::RealF64(lhs), DiagonalData::RealF64(rhs)) => {
                                compact_inner_with_weight(
                                    lhs,
                                    rhs,
                                    $weight,
                                    |value| value,
                                    |value| value,
                                )
                                .map(|value| Scalar::F64(value.re))
                            }
                            (DiagonalData::RealC64(lhs), DiagonalData::RealC64(rhs)) => {
                                compact_inner_with_weight(
                                    lhs,
                                    rhs,
                                    $weight,
                                    |value| Complex64::new(value, 0.0),
                                    |value| Complex64::new(value, 0.0),
                                )
                                .map(Scalar::C64)
                            }
                            (DiagonalData::RealC64(lhs), DiagonalData::C64(rhs)) => {
                                compact_inner_with_weight(
                                    lhs,
                                    rhs,
                                    $weight,
                                    |value| Complex64::new(value, 0.0),
                                    |value| value,
                                )
                                .map(Scalar::C64)
                            }
                            (DiagonalData::C64(lhs), DiagonalData::RealC64(rhs)) => {
                                compact_inner_with_weight(
                                    lhs,
                                    rhs,
                                    $weight,
                                    |value| value,
                                    |value| Complex64::new(value, 0.0),
                                )
                                .map(Scalar::C64)
                            }
                            (DiagonalData::C64(lhs), DiagonalData::C64(rhs)) => {
                                compact_inner_with_weight(
                                    lhs,
                                    rhs,
                                    $weight,
                                    |value| value,
                                    |value| value,
                                )
                                .map(Scalar::C64)
                            }
                            _ => None,
                        }
                    };
                }
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    reduce!(|sector| rule.dim_scalar(sector))
                })
                .ok_or(Error::DtypeMismatch)?;
                return Ok(value);
            }
            (Some(diagonal), None) => {
                let dense_data = other.coupled_data()?;
                macro_rules! reduce {
                    ($weight:expr) => {
                        match (diagonal, dense_data) {
                            (DiagonalData::RealF64(spectrum), Data::F64(dense)) => {
                                dense_inner_with_weight(
                                    &self.ordinary_body().space,
                                    spectrum,
                                    dense,
                                    true,
                                    $weight,
                                    |value| value,
                                )
                                .map(|value| Scalar::F64(value.re))
                            }
                            (DiagonalData::RealC64(spectrum), Data::C64(dense)) => {
                                dense_inner_with_weight(
                                    &self.ordinary_body().space,
                                    spectrum,
                                    dense,
                                    true,
                                    $weight,
                                    |value| Complex64::new(value, 0.0),
                                )
                                .map(Scalar::C64)
                            }
                            (DiagonalData::C64(spectrum), Data::C64(dense)) => {
                                dense_inner_with_weight(
                                    &self.ordinary_body().space,
                                    spectrum,
                                    dense,
                                    true,
                                    $weight,
                                    |value| value,
                                )
                                .map(Scalar::C64)
                            }
                            _ => Err(Error::DtypeMismatch),
                        }
                    };
                }
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    reduce!(|sector| rule.dim_scalar(sector))
                })?;
                return Ok(value);
            }
            (None, Some(diagonal)) => {
                let dense_data = self.coupled_data()?;
                macro_rules! reduce {
                    ($weight:expr) => {
                        match (dense_data, diagonal) {
                            (Data::F64(dense), DiagonalData::RealF64(spectrum)) => {
                                dense_inner_with_weight(
                                    &self.ordinary_body().space,
                                    spectrum,
                                    dense,
                                    false,
                                    $weight,
                                    |value| value,
                                )
                                .map(|value| Scalar::F64(value.re))
                            }
                            (Data::C64(dense), DiagonalData::RealC64(spectrum)) => {
                                dense_inner_with_weight(
                                    &self.ordinary_body().space,
                                    spectrum,
                                    dense,
                                    false,
                                    $weight,
                                    |value| Complex64::new(value, 0.0),
                                )
                                .map(Scalar::C64)
                            }
                            (Data::C64(dense), DiagonalData::C64(spectrum)) => {
                                dense_inner_with_weight(
                                    &self.ordinary_body().space,
                                    spectrum,
                                    dense,
                                    false,
                                    $weight,
                                    |value| value,
                                )
                                .map(Scalar::C64)
                            }
                            _ => Err(Error::DtypeMismatch),
                        }
                    };
                }
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    reduce!(|sector| rule.dim_scalar(sector))
                })?;
                return Ok(value);
            }
            (None, None) => {}
        }
        match (self.coupled_data()?, other.coupled_data()?) {
            (Data::F64(a), Data::F64(b)) => {
                with_user_rule!(self.ordinary_body().space, rule, {
                    weighted_inner(
                        rule,
                        self.ordinary_body().space.structure(),
                        self.ordinary_body().space.nout(),
                        a,
                        b,
                    )
                    .map(|v| Scalar::F64(v.re))
                })
            }
            (Data::C64(a), Data::C64(b)) => {
                with_user_rule!(self.ordinary_body().space, rule, {
                    weighted_inner(
                        rule,
                        self.ordinary_body().space.structure(),
                        self.ordinary_body().space.nout(),
                        a,
                        b,
                    )
                    .map(Scalar::C64)
                })
            }
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                self.weighted_inner_cuda(a, b).map(|v| Scalar::F64(v.re))
            }
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Frobenius inner product `<self, other>` with `self` conjugated — an
    /// alias for [`Self::inner`], matching `LinearAlgebra.dot` / TensorKit's
    /// `dot(x, y)`. Provided for callers who reach for the `dot` name; the
    /// semantics (conjugate-linear in the first argument, quantum-dimension
    /// weighted) are identical.
    ///
    pub fn dot(&self, other: &Self) -> Result<Scalar, Error> {
        self.inner(other)
    }

    /// Returns `self / norm(self)`, the unit-norm tensor pointing the same way
    /// (TensorKit's `normalize`, LinearAlgebra's 2-norm normalization). The
    /// norm is the quantum-dimension-weighted Frobenius norm from
    /// [`Self::norm`]; the result satisfies `t.normalize()?.norm()? == 1`.
    /// Works for both dtypes (a c64 tensor is scaled by the real reciprocal
    /// norm).
    ///
    /// Like TensorKit, a zero-norm tensor is not special-cased: normalizing it
    /// divides by zero and yields non-finite entries. Guard the caller if that
    /// input is reachable.
    ///
    pub fn normalize(&self) -> Result<Self, Error> {
        self.scale(1.0 / self.norm()?)
    }

    /// Tests whether the tensor equals its own adjoint within `tol`, relative
    /// to its norm (TensorKit `ishermitian`). Non-endomorphisms (codomain and
    /// domain spaces differ) are never Hermitian and return `false` without
    /// error, unlike TensorKit which throws — the predicate form is friendlier.
    pub fn is_hermitian(&self, tol: f64) -> Result<bool, Error> {
        if self.codomain_spaces() != self.domain_spaces() {
            return Ok(false);
        }
        let diff = self.add(&self.adjoint()?, 1.0, -1.0)?.norm()?;
        Ok(diff <= tol * self.norm()?.max(1.0))
    }

    /// Tests whether `adjoint(t) ∘ t` is the identity on the domain within
    /// `tol` (TensorKit `isisometric`): the columns are orthonormal. Works for
    /// any rectangular shape with `codomain_dim >= domain_dim`.
    pub fn is_isometric(&self, tol: f64) -> Result<bool, Error> {
        let gram = self.adjoint()?.compose(self)?;
        let identity = Self::id(&self.rt, self.dtype(), &self.domain_spaces())?;
        Ok(gram.add(&identity, 1.0, -1.0)?.norm()? <= tol * gram.norm()?.max(1.0))
    }

    /// Tests whether the tensor is unitary within `tol` (TensorKit
    /// `isunitary`): isometric in both directions, i.e. `adjoint(t) ∘ t` and
    /// `t ∘ adjoint(t)` are both identities.
    pub fn is_unitary(&self, tol: f64) -> Result<bool, Error> {
        Ok(self.is_isometric(tol)? && self.adjoint()?.is_isometric(tol)?)
    }

    /// Tests whether the tensor is Hermitian and positive definite (TensorKit
    /// `isposdef`, which is Cholesky-based and strict): every Hermitian
    /// eigenvalue must exceed `tol * max(norm, 1)`. Positive *semi*definite
    /// spectra (an eigenvalue at zero) return `false`; with `tol = 0.0` the
    /// check is exact strict positivity up to floating point.
    pub fn is_posdef(&self, tol: f64) -> Result<bool, Error> {
        if !self.is_hermitian(tol)? {
            return Ok(false);
        }
        let threshold = tol * self.norm()?.max(1.0);
        // Compact arm: a spectrum factor's stored values *are* its Hermitian
        // eigenvalues, so there is nothing to factorize and nothing to
        // materialize (#585). The adjoint-view exclusion is defensive and is
        // unreachable today: `Tensor::adjoint` short-circuits `Data::Diagonal`
        // and never builds an `AdjointView` over it, so no compact tensor is
        // ever a view. Kept anyway — if compact adjoint views ever appear, the
        // stored spectrum would be the parent's rather than this tensor's
        // logical one, and this arm would silently read the wrong values.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            if !self.is_adjoint_view() {
                return Ok(diagonal.is_posdef(threshold));
            }
        }
        Ok(self
            .eigh_vals()?
            .iter()
            .flat_map(|spectrum| spectrum.values.iter())
            .all(|&lambda| lambda > threshold))
    }

    /// Tests whether the tensor equals minus its own adjoint within `tol`,
    /// relative to its norm (TensorKit `isantihermitian`). Non-endomorphisms
    /// return `false` without error (cf. [`Self::is_hermitian`]).
    pub fn is_antihermitian(&self, tol: f64) -> Result<bool, Error> {
        if self.codomain_spaces() != self.domain_spaces() {
            return Ok(false);
        }
        let sum = self.add(&self.adjoint()?, 1.0, 1.0)?.norm()?;
        Ok(sum <= tol * self.norm()?.max(1.0))
    }

    /// The Hermitian part `(t + t†)/2` (TensorKit `project_hermitian`), the
    /// nearest Hermitian tensor. Requires an endomorphism.
    pub fn project_hermitian(&self) -> Result<Self, Error> {
        self.add(&self.adjoint()?, 0.5, 0.5)
    }

    /// The anti-Hermitian part `(t - t†)/2` (TensorKit `project_antihermitian`).
    /// Requires an endomorphism.
    pub fn project_antihermitian(&self) -> Result<Self, Error> {
        self.add(&self.adjoint()?, 0.5, -0.5)
    }

    fn check_same_space(&self, other: &Self) -> Result<(), Error> {
        self.check_same_world(other)?;
        if self.ordinary_body().space != other.ordinary_body().space {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        Ok(())
    }

    fn check_same_logical_space(&self, other: &Self) -> Result<(), Error> {
        self.check_same_world(other)?;
        let lhs = self.metadata();
        let rhs = other.metadata();
        if lhs.codomain() != rhs.codomain() || lhs.domain() != rhs.domain() {
            return Err(logical_space_mismatch());
        }
        match (lhs.orientation, rhs.orientation) {
            (TensorOrientation::Owned, TensorOrientation::Owned)
            | (TensorOrientation::Adjoint, TensorOrientation::Adjoint) => {
                if lhs.body.space != rhs.body.space {
                    return Err(logical_space_mismatch());
                }
            }
            (TensorOrientation::Owned, TensorOrientation::Adjoint) => {
                tenet_tensors::validate_oriented_fusion_layout(
                    lhs.body.space.structure(),
                    tenet_tensors::FusionOperand::adjoint(rhs.body.space.raw()),
                )
                .map_err(|_| logical_space_mismatch())?;
            }
            (TensorOrientation::Adjoint, TensorOrientation::Owned) => {
                tenet_tensors::validate_oriented_fusion_layout(
                    rhs.body.space.structure(),
                    tenet_tensors::FusionOperand::adjoint(lhs.body.space.raw()),
                )
                .map_err(|_| logical_space_mismatch())?;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Decompositions and matrix functions (TensorKit 0.17 / MatrixAlgebraKit
    // names, transparently over the tenet-matrixalgebra dynamic cores).
    // -----------------------------------------------------------------------

    fn from_bound_factor<R, D>(&self, factor: BoundDynFactor<R, D>) -> Result<Self, Error>
    where
        R: IntoUserBoundDynamicSpace,
        D: UserScalar,
    {
        let (space, data) = factor.into_parts();
        Ok(Self::owned(
            self.rt.clone(),
            Arc::new(UserBoundSpace::from_bound(
                self.materialized_body()?.space.as_ref(),
                space,
            )?),
            Arc::new(D::lift(data)),
        ))
    }

    fn from_bound_factors<R, D>(
        &self,
        factors: (
            BoundDynFactor<R, D>,
            BoundDynFactor<R, D>,
            Vec<SectorSpectrum>,
        ),
        complex: bool,
    ) -> Result<(Self, Self, Self), Error>
    where
        R: IntoUserBoundDynamicSpace,
        D: UserScalar,
    {
        let (u, vh, spectrum) = factors;
        Ok((
            self.from_bound_factor(u)?,
            self.from_diagonal_real_spectrum(spectrum, complex)?,
            self.from_bound_factor(vh)?,
        ))
    }

    fn from_svd_trunc_factors<R, D>(
        &self,
        output: SvdTruncFactorsDyn<R, D>,
        complex: bool,
    ) -> Result<SvdTrunc, Error>
    where
        R: IntoUserBoundDynamicSpace,
        D: UserScalar,
    {
        let (u, vh, singular_values, error) = output;
        Ok(SvdTrunc {
            u: self.from_bound_factor(u)?,
            s: self.from_diagonal_real_spectrum(singular_values.clone(), complex)?,
            vh: self.from_bound_factor(vh)?,
            singular_values,
            error,
        })
    }

    /// Wraps a real per-sector spectrum (svd `S`, eigh `D`) as a diagonal-storage
    /// tensor: the bond space is built eagerly, but the values stay O(rank) in
    /// `Data::Diagonal` instead of a dense O(rank²) block-diagonal buffer (issue
    /// #55). `complex` preserves the public dtype: a complex input yields a
    /// complex-valued but real-magnitude `S` (`RealC64`), while a real input
    /// yields `RealF64`.
    fn from_diagonal_real_spectrum(
        &self,
        mut spectrum: Vec<SectorSpectrum<f64>>,
        complex: bool,
    ) -> Result<Self, Error> {
        spectrum.sort_unstable_by_key(|entry| entry.sector);
        #[cfg(test)]
        observe_diagonal_result_layout_build();
        let space = with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let space = tenet_matrixalgebra::diagonal_bond_bound_space_like(bound, &spectrum)?;
            UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
        })?;
        let data = if complex {
            DiagonalData::RealC64(spectrum)
        } else {
            DiagonalData::RealF64(spectrum)
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::new(space),
            Arc::new(Data::Diagonal(data)),
        ))
    }

    /// Wraps a complex per-sector spectrum (eig `D`) as diagonal storage. The
    /// general eigendecomposition is complex-valued even for real input, so `d`
    /// is always c64 and stays compact through block-local scaling/products.
    fn from_diagonal_complex_spectrum(
        &self,
        mut spectrum: Vec<SectorSpectrum<Complex64>>,
    ) -> Result<Self, Error> {
        spectrum.sort_unstable_by_key(|entry| entry.sector);
        with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let space = tenet_matrixalgebra::diagonal_bond_bound_space_like(bound, &spectrum)?;
            let space = UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)?;
            self.with_bound(space, Data::Diagonal(DiagonalData::C64(spectrum)))
        })
    }

    fn with_same_data(&self, data: Data) -> Self {
        Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        )
    }

    /// Reuse this tensor's space with a new diagonal payload (elementwise
    /// scale/inv/pinv/sqrt keep the same bond space).
    fn with_diagonal(&self, data: DiagonalData) -> Self {
        Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(Data::Diagonal(data)),
        )
    }

    fn diagonal_data(&self) -> Option<&DiagonalData> {
        match self.stored_data() {
            Data::Diagonal(diagonal) => Some(diagonal),
            _ => None,
        }
    }

    fn is_diagonal_bond_space(space: &DynamicFusionMapSpace) -> bool {
        let homspace = space.homspace();
        space.nout() == 1
            && space.nin() == 1
            && homspace.codomain().legs() == homspace.domain().legs()
    }

    fn contraction_output_space(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<UserBoundSpace, Error> {
        self.contraction_output_space_oriented(rhs, lhs_axes, rhs_axes, OutputAxisOrder::identity())
    }

    fn contraction_output_space_ordered(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<UserBoundSpace, Error> {
        self.contraction_output_space_oriented(rhs, lhs_axes, rhs_axes, output_order)
    }

    fn contraction_output_space_oriented(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<UserBoundSpace, Error> {
        let homspace = self.oriented_contraction_homspace(rhs, lhs_axes, rhs_axes, output_order)?;
        self.metadata().body.space.from_selected_homspace(homspace)
    }

    fn space_matches_metadata(candidate: &UserBoundSpace, operand: &Self) -> bool {
        let metadata = operand.metadata();
        // Why not build and compare an adjoint layout: canonical rule identity
        // plus the final HomSpace fully determine eligibility and leave a
        // rejected operand lazy.
        candidate.identity() == metadata.body.space.identity()
            && candidate.raw().homspace().codomain() == metadata.codomain()
            && candidate.raw().homspace().domain() == metadata.domain()
    }

    fn validate_oriented_contracted_homspace(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<(), Error> {
        self.oriented_contraction_homspace(rhs, lhs_axes, rhs_axes, OutputAxisOrder::identity())?;
        Ok(())
    }

    fn oriented_contraction_homspace(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<FusionTreeHomSpace, Error> {
        let lhs_metadata = self.metadata();
        let rhs_metadata = rhs.metadata();
        let lhs_open_rank = lhs_metadata
            .rank()
            .checked_sub(lhs_axes.len())
            .ok_or_else(|| {
                Error::InvalidArgument("contracted axis count exceeds tensor rank".to_string())
            })?;
        let rhs_open_rank = rhs_metadata
            .rank()
            .checked_sub(rhs_axes.len())
            .ok_or_else(|| {
                Error::InvalidArgument("contracted axis count exceeds tensor rank".to_string())
            })?;
        let output_rank = lhs_open_rank + rhs_open_rank;
        let identity_axes;
        let output_axes = match output_order {
            OutputAxisOrder::Identity => {
                identity_axes = (0..output_rank).collect::<SmallVec<[usize; 8]>>();
                identity_axes.as_slice()
            }
            OutputAxisOrder::Axes(output_axes) => {
                if let Err(output_error) = validate_axis_permutation(output_axes, output_rank) {
                    self.validate_oriented_contracted_homspace(rhs, lhs_axes, rhs_axes)?;
                    return Err(output_error);
                }
                output_axes
            }
        };
        with_user_rule!(self.rule_authority_space(), rule, {
            OrientedFusionTreeHomSpace::try_tensorcontract_homspace_checked(
                rule,
                lhs_metadata.oriented_homspace(),
                rhs_metadata.oriented_homspace(),
                lhs_axes,
                rhs_axes,
                output_axes,
                lhs_open_rank,
            )
            .map_err(|error| match error {
                CheckedFusionSpaceError::Core(error) => {
                    OperationError::from_core_preserving_context(*error).into()
                }
                CheckedFusionSpaceError::FusionAlgebra(error) => Error::FusionAlgebra(error),
                _ => Error::InvalidArgument("unknown checked fusion metadata error".to_string()),
            })
        })
    }

    fn output_source_axes_for_order(
        default_source_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Vec<usize>, Error> {
        match output_order {
            OutputAxisOrder::Identity => Ok(default_source_axes.to_vec()),
            OutputAxisOrder::Axes(output_axes) => {
                validate_axis_permutation(output_axes, default_source_axes.len())?;
                Ok(output_axes
                    .iter()
                    .map(|&axis| default_source_axes[axis])
                    .collect())
            }
        }
    }

    fn try_contract_diagonal_fast_path(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Option<Self>, Error> {
        if lhs_axes.len() != 1
            || rhs_axes.len() != 1
            || (self.diagonal_data().is_none() && rhs.diagonal_data().is_none())
        {
            return Ok(None);
        }

        let fermionic = with_user_rule!(self.rule_authority_space(), rule, {
            rule.braiding_style() == tenet_core::BraidingStyleKind::Fermionic
        });
        let twist_rhs_leg = fermionic && rhs.external_axis_is_dual(rhs_axes[0])?;
        let dst_space =
            self.contraction_output_space_ordered(rhs, lhs_axes, rhs_axes, output_order)?;

        match (self.diagonal_data(), rhs.diagonal_data()) {
            (Some(lhs), Some(rhs_diagonal)) if lhs_axes == [1] && rhs_axes == [0] => {
                let identity_order = match output_order {
                    OutputAxisOrder::Identity => true,
                    OutputAxisOrder::Axes(output_axes) => output_axes.iter().copied().eq(0..2),
                };
                if !identity_order {
                    // Why not bind the product spectrum to a reordered output:
                    // pAB can move the surviving diagonal leg across the
                    // codomain/domain split, and #453 oracle checks showed
                    // raw-label rebinding is not equivalent to permute.
                    return Ok(None);
                }
                let folded_rhs = self.twist_folded_diagonal(rhs_diagonal, twist_rhs_leg);
                if Self::is_diagonal_bond_space(dst_space.raw()) {
                    if let Some(product) = lhs.elementwise_product(&folded_rhs) {
                        return Ok(Some(self.with_bound(dst_space, Data::Diagonal(product))?));
                    }
                }
            }
            (None, Some(diagonal)) if lhs_axes[0] >= self.codomain_rank() && rhs_axes[0] == 0 => {
                let leg = lhs_axes[0];
                let folded = self.twist_folded_diagonal(diagonal, twist_rhs_leg);
                let scaled = self.scaled_axis_copy_diagonal(Some(leg), &folded)?;
                let mut default_source_axes: Vec<usize> =
                    (0..self.rank()).filter(|&axis| axis != leg).collect();
                default_source_axes.push(leg);
                let ordered_axes =
                    Self::output_source_axes_for_order(&default_source_axes, output_order)?;
                let split = dst_space.raw().nout();
                let output = scaled.permute(&ordered_axes[..split], &ordered_axes[split..])?;
                debug_assert_eq!(output.ordinary_body().space.raw(), dst_space.raw());
                return Ok(Some(output));
            }
            (Some(diagonal), None) if lhs_axes[0] == 1 && rhs_axes[0] < rhs.codomain_rank() => {
                let leg = rhs_axes[0];
                let pretwisted = if twist_rhs_leg {
                    rhs.twist(&[leg])?
                } else {
                    rhs.clone()
                };
                let scaled = pretwisted.scaled_axis_copy_diagonal(Some(leg), diagonal)?;
                let mut default_source_axes = Vec::with_capacity(rhs.rank());
                default_source_axes.push(leg);
                default_source_axes.extend((0..rhs.rank()).filter(|&axis| axis != leg));
                let ordered_axes =
                    Self::output_source_axes_for_order(&default_source_axes, output_order)?;
                let split = dst_space.raw().nout();
                let output = scaled.permute(&ordered_axes[..split], &ordered_axes[split..])?;
                debug_assert_eq!(output.ordinary_body().space.raw(), dst_space.raw());
                return Ok(Some(output));
            }
            _ => {}
        }
        Ok(None)
    }

    fn external_axis_is_dual(&self, axis: usize) -> Result<bool, Error> {
        let metadata = self.metadata();
        metadata
            .oriented_homspace()
            .external_axis_is_dual(axis)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "axis {axis} is out of range for rank {}",
                    self.rank()
                ))
            })
    }

    /// Folds the supertrace twist into compact values. Why not call `twist` on
    /// a temporary tensor: this helper already has the spectrum and avoids an
    /// extra compact result allocation before contraction.
    fn twist_folded_diagonal(&self, diagonal: &DiagonalData, apply: bool) -> DiagonalData {
        if !apply {
            return diagonal.clone();
        }
        fn fold<V: Copy>(
            spectrum: &[SectorSpectrum<V>],
            factor: impl Fn(SectorId, V) -> V,
        ) -> Vec<SectorSpectrum<V>> {
            spectrum
                .iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: entry
                        .values
                        .iter()
                        .copied()
                        .map(|value| factor(entry.sector, value))
                        .collect(),
                })
                .collect()
        }
        with_user_rule!(self.ordinary_body().space, rule, {
            match diagonal {
                DiagonalData::RealF64(spectrum) => {
                    DiagonalData::RealF64(fold(spectrum, |sector, value| {
                        value * rule.twist_scalar(sector)
                    }))
                }
                DiagonalData::RealC64(spectrum) => {
                    DiagonalData::RealC64(fold(spectrum, |sector, value| {
                        value * rule.twist_scalar(sector)
                    }))
                }
                DiagonalData::C64(spectrum) => {
                    DiagonalData::C64(fold(spectrum, |sector, value| {
                        value * rule.twist_scalar(sector)
                    }))
                }
            }
        })
    }

    /// Why not materialize a diagonal matrix: TensorKit `lmul!`/`rmul!` only
    /// scales the selected block-local axis for every compact scalar variant.
    fn scaled_axis_copy_diagonal(
        &self,
        axis: Option<usize>,
        diagonal: &DiagonalData,
    ) -> Result<Self, Error> {
        let space = Arc::clone(&self.materialized_body()?.space);
        match (self.coupled_data()?, diagonal) {
            (Data::F64(data), DiagonalData::RealF64(spectrum)) => {
                let mut buf = data.clone();
                tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
                    &space,
                    &mut buf,
                    axis,
                    spectrum,
                    |value| value,
                )?;
                Ok(Self::owned(
                    self.rt.clone(),
                    Arc::clone(&space),
                    Arc::new(Data::F64(buf)),
                ))
            }
            (Data::C64(data), DiagonalData::RealC64(spectrum)) => {
                let mut buf = data.clone();
                tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
                    &space,
                    &mut buf,
                    axis,
                    spectrum,
                    |value| Complex64::new(value, 0.0),
                )?;
                Ok(Self::owned(
                    self.rt.clone(),
                    Arc::clone(&space),
                    Arc::new(Data::C64(buf)),
                ))
            }
            (Data::C64(data), DiagonalData::C64(spectrum)) => {
                let mut buf = data.clone();
                tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
                    &space,
                    &mut buf,
                    axis,
                    spectrum,
                    |value| value,
                )?;
                Ok(Self::owned(
                    self.rt.clone(),
                    Arc::clone(&space),
                    Arc::new(Data::C64(buf)),
                ))
            }
            (Data::F64(_) | Data::C64(_), _) => Err(Error::DtypeMismatch),
            (Data::Diagonal(_), _) => Err(Error::InvalidArgument(
                "internal: diagonal scaling requires a non-diagonal operand".to_string(),
            )),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(_), _) => Err(device_unsupported("diagonal scaling")),
        }
    }

    /// Compact SVD `t = u * s * vh` (MatrixAlgebraKit `svd_compact`):
    /// per coupled sector the bond is `min(rows, cols)`.
    pub fn svd_compact(&self) -> Result<(Self, Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            let parent = self.parent_tensor_for_lowering();
            let complex = parent.dtype() == Dtype::C64;
            let mut dense = parent.rt.lease_dense();
            return with_data!(&parent, data, {
                with_bound_multiplicity_free!(parent.ordinary_body().space, bound, {
                    let factors = tenet_matrixalgebra::svd_compact_adjoint_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(bound, data)?,
                    )?;
                    parent.from_bound_factors(factors, complex)
                })
            });
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            let out = self.svd_cuda(storage, None)?;
            return Ok((out.u, out.s, out.vh));
        }
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let factors = tenet_matrixalgebra::svd_compact_factors_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&bound, data)?,
                )?;
                self.from_bound_factors(factors, complex)
            })
        })
    }

    /// Full SVD `t = u * s * vh` (MatrixAlgebraKit `svd_full`): square
    /// unitaries per sector, rectangular diagonal `s`.
    pub fn svd_full(&self) -> Result<(Self, Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            let parent = self.parent_tensor_for_lowering();
            let mut dense = parent.rt.lease_dense();
            return with_data!(&parent, data, {
                with_bound_multiplicity_free!(parent.ordinary_body().space, bound, {
                    let out = tenet_matrixalgebra::svd_full_adjoint_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(bound, data)?,
                    )?;
                    let (u, s, vh, _) = out.into_parts();
                    Ok::<_, Error>((
                        parent.from_bound_factor(u)?,
                        parent.from_bound_factor(s)?,
                        parent.from_bound_factor(vh)?,
                    ))
                })
            });
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::svd_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                let (u, s, vh, _) = out.into_parts();
                Ok::<_, Error>((
                    self.from_bound_factor(u)?,
                    self.from_bound_factor(s)?,
                    self.from_bound_factor(vh)?,
                ))
            })
        })
    }

    /// Truncated SVD (MatrixAlgebraKit `svd_trunc`); see [`SvdTrunc`].
    ///
    /// # Complexity
    ///
    /// Sectorwise cubic (one dense SVD per coupled sector); a
    /// compact-diagonal input is materialized dense first. On host storage
    /// the returned `s` is held in `O(Σ_c k_c)` compact diagonal storage —
    /// the `DiagonalTensorMap` TensorKit's own `svd_trunc` returns, not the
    /// `Σ_c k_c²` block-diagonal buffer — so a downstream `u.compose(&s)` or
    /// `s.compose(&vh)` takes the bond-scaling path rather than a GEMM. A
    /// device (`Data::CudaF64`) receiver instead returns `s` as a dense
    /// `Σ_c k_c²` device block-diagonal, so composing with it is a device
    /// GEMM, not a bond scaling.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    /// from the matrix-algebra seam, including a malformed `truncation` —
    /// the truncation policy is validated where it is applied, not here. On
    /// a device receiver a malformed `truncation` is instead reported as
    /// [`Error::InvalidArgument`]: the device route validates it on this
    /// side of the seam.
    pub fn svd_trunc(&self, truncation: &Truncation) -> Result<SvdTrunc, Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            let parent = self.parent_tensor_for_lowering();
            let complex = parent.dtype() == Dtype::C64;
            let mut dense = parent.rt.lease_dense();
            return with_data!(parent, data, {
                with_bound_multiplicity_free!(parent.ordinary_body().space, bound, {
                    let output = tenet_matrixalgebra::svd_trunc_adjoint_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(&bound, data)?,
                        truncation,
                    )?;
                    parent.from_svd_trunc_factors(output, complex)
                })
            });
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return self.svd_cuda(storage, Some(truncation));
        }
        // Singular values are real => `s` is a real diagonal in O(rank) storage
        // (see `svd_compact`). `out.singular_values` is also returned, so it is
        // cloned into the diagonal factor.
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let output = tenet_matrixalgebra::svd_trunc_factors_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&bound, data)?,
                    truncation,
                )?;
                self.from_svd_trunc_factors(output, complex)
            })
        })
    }

    /// All singular values per coupled sector, descending (MatrixAlgebraKit
    /// `svd_vals`). Real for both dtypes.
    pub fn svd_vals(&self) -> Result<Vec<SectorSpectrum>, Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return Err(device_unsupported("materializing an adjoint device tensor"));
            }
            return self.parent_tensor_for_lowering().svd_vals();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                tenet_matrixalgebra::svd_vals_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&bound, data)?,
                )
            })
            .map_err(Into::into)
        })
    }

    /// Compact QR `t = q * r` (MatrixAlgebraKit `qr_compact`): `q` has
    /// orthonormal columns per coupled sector.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    /// straight from the matrix-algebra seam, unfiltered — as everywhere in
    /// the factorization group there are no pre-checks here; the seam owns
    /// the rules.
    pub fn qr_compact(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.qr_compact();
            }
            return self.materialized_tensor_uncached()?.qr_compact();
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return self.qr_cuda(storage);
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let (q, r) = tenet_matrixalgebra::qr_compact_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok::<_, Error>((self.from_bound_factor(q)?, self.from_bound_factor(r)?))
            })
        })
    }

    /// Full QR `t = q * r` (MatrixAlgebraKit `qr_full`): square `q` per
    /// sector.
    pub fn qr_full(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.qr_full();
            }
            return self.materialized_tensor_uncached()?.qr_full();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let (q, r) = tenet_matrixalgebra::qr_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok::<_, Error>((self.from_bound_factor(q)?, self.from_bound_factor(r)?))
            })
        })
    }

    /// Compact LQ `t = l * q` (MatrixAlgebraKit `lq_compact`): `q` has
    /// orthonormal rows per coupled sector.
    pub fn lq_compact(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.lq_compact();
            }
            let (q, r) = self.adjoint()?.qr_compact()?;
            return Ok((
                r.adjoint()?.materialized_tensor_uncached()?,
                q.adjoint()?.materialized_tensor_uncached()?,
            ));
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let (l, q) = tenet_matrixalgebra::lq_compact_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok::<_, Error>((self.from_bound_factor(l)?, self.from_bound_factor(q)?))
            })
        })
    }

    /// Full LQ `t = l * q` (MatrixAlgebraKit `lq_full`): square `q` per
    /// sector.
    pub fn lq_full(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.lq_full();
            }
            let (q, r) = self.adjoint()?.qr_full()?;
            return Ok((
                r.adjoint()?.materialized_tensor_uncached()?,
                q.adjoint()?.materialized_tensor_uncached()?,
            ));
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let (l, q) = tenet_matrixalgebra::lq_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok::<_, Error>((self.from_bound_factor(l)?, self.from_bound_factor(q)?))
            })
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
    /// `left_null`). A host lazy adjoint redirects through the owned parent's
    /// right null space and returns a detached owned factor.
    pub fn left_null(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.left_null();
            }
            return self
                .adjoint()?
                .right_null()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::left_null_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                self.from_bound_factor(out)
            })
        })
    }

    /// Right null space `n : W <- domain` with `t * n^H = 0` (MatrixAlgebraKit
    /// `right_null`). A host lazy adjoint redirects through the owned parent's
    /// left null space and returns a detached owned factor.
    pub fn right_null(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.right_null();
            }
            return self
                .adjoint()?
                .left_null()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::right_null_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                self.from_bound_factor(out)
            })
        })
    }

    /// Left polar decomposition `t = w * p` (MatrixAlgebraKit `left_polar`):
    /// `w` isometric, `p` positive on the domain. Every coupled-sector matrix
    /// must have at least as many rows as columns.
    pub fn left_polar(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.left_polar();
            }
            let parent = self.adjoint()?;
            return with_data!(&parent, data, parent.left_polar_adjoint_impl(data));
        }
        with_data!(self, data, self.left_polar_impl(data))
    }

    fn left_polar_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let (w, p) = tenet_matrixalgebra::left_polar_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            Ok::<_, Error>((self.from_bound_factor(w)?, self.from_bound_factor(p)?))
        })
    }

    fn left_polar_adjoint_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let (w, p) = tenet_matrixalgebra::left_polar_adjoint_parent_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            Ok::<_, Error>((self.from_bound_factor(w)?, self.from_bound_factor(p)?))
        })
    }

    /// Right polar decomposition `t = p * w` (MatrixAlgebraKit
    /// `right_polar`): `p` positive on the codomain, `w` isometric. Every
    /// coupled-sector matrix must have at least as many columns as rows.
    pub fn right_polar(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.right_polar();
            }
            let parent = self.adjoint()?;
            return with_data!(&parent, data, parent.right_polar_adjoint_impl(data));
        }
        with_data!(self, data, self.right_polar_impl(data))
    }

    fn right_polar_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let (p, w) = tenet_matrixalgebra::right_polar_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            Ok::<_, Error>((self.from_bound_factor(p)?, self.from_bound_factor(w)?))
        })
    }

    fn right_polar_adjoint_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let (p, w) = tenet_matrixalgebra::right_polar_adjoint_parent_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            Ok::<_, Error>((self.from_bound_factor(p)?, self.from_bound_factor(w)?))
        })
    }

    /// Full Hermitian eigendecomposition `t = v * d * v^H` (MatrixAlgebraKit
    /// `eigh_full`), returned as `(d, v)`. Requires an endomorphism with
    /// Hermitian coupled blocks. The eigenvalues are real for both dtypes
    /// (TensorKit: real `D`); `d` keeps the input dtype so it composes with
    /// `v` directly.
    ///
    /// # Complexity
    ///
    /// Sectorwise cubic (one dense EIGH per coupled sector); a
    /// compact-diagonal input is materialized dense first. On host storage
    /// the returned `d` is built as `O(Σ_c k_c)` compact diagonal storage
    /// straight from the spectrum — nothing `O(Σ_c k_c²)` is materialized
    /// and discarded (#56 item N) — so `v.compose(&d)` takes the
    /// bond-scaling path. A device (`Data::CudaF64`) receiver instead
    /// returns `d` as a dense `Σ_c k_c²` device block-diagonal.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] when the tensor is not an endomorphism or its
    ///   coupled blocks are not Hermitian — the seam is where that surfaces.
    ///   On a device receiver the endomorphism check runs here instead, ahead
    ///   of the seam, and reports [`Error::InvalidArgument`]; non-Hermitian
    ///   blocks still surface as [`Error::Operation`] from the shared
    ///   validator.
    /// - [`Error::Core`] / [`Error::FusionAlgebra`] otherwise from the seam,
    ///   which owns those rules.
    pub fn eigh_full(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.eigh_full();
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            let out = self.eigh_cuda(storage, None)?;
            return Ok((out.d, out.v));
        }
        // eigh eigenvalues are real, so `d` is a real diagonal (`RealC64` for
        // c64 input). Build it as O(rank) diagonal storage from the spectrum;
        // `eigh_full_dyn` returns only the spectrum + eigenvectors (no dense d),
        // so nothing O(rank²) is materialized and discarded here (#56 item N).
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eigh_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                let (v, eigenvalues) = out.into_parts();
                Ok::<_, Error>((
                    self.from_diagonal_real_spectrum(eigenvalues, complex)?,
                    self.from_bound_factor(v)?,
                ))
            })
        })
    }

    /// Truncated Hermitian eigendecomposition (MatrixAlgebraKit
    /// `eigh_trunc`); see [`EighTrunc`].
    pub fn eigh_trunc(&self, truncation: &Truncation) -> Result<EighTrunc, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.eigh_trunc(truncation);
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return self.eigh_cuda(storage, Some(truncation));
        }
        // Real eigenvalues => real diagonal `d` in O(rank) storage (see
        // `eigh_full`). `out.eigenvalues` is also returned to the caller, so it
        // is cloned into the diagonal factor.
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eigh_trunc_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                    truncation,
                )?;
                let (v, eigenvalues, error) = out.into_parts();
                Ok::<_, Error>(EighTrunc {
                    d: self.from_diagonal_real_spectrum(eigenvalues.clone(), complex)?,
                    v: self.from_bound_factor(v)?,
                    eigenvalues,
                    error,
                })
            })
        })
    }

    /// All Hermitian eigenvalues per coupled sector, descending by magnitude
    /// (MatrixAlgebraKit `eigh_vals`). Real for both dtypes.
    pub fn eigh_vals(&self) -> Result<Vec<SectorSpectrum>, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.eigh_vals();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                tenet_matrixalgebra::eigh_vals_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )
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
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.eig_full();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eig_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                let (v, eigenvalues) = out.into_parts();
                Ok::<_, Error>((
                    self.from_diagonal_complex_spectrum(eigenvalues)?,
                    self.from_bound_factor(v)?,
                ))
            })
        })
    }

    /// Truncated general eigendecomposition (MatrixAlgebraKit `eig_trunc`,
    /// kept by descending `|eigenvalue|`); see [`EigTrunc`]. Output tensors
    /// are always c64.
    pub fn eig_trunc(&self, truncation: &Truncation) -> Result<EigTrunc, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.eig_trunc(truncation);
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eig_trunc_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                    truncation,
                )?;
                let (v, eigenvalues, error) = out.into_parts();
                Ok::<_, Error>(EigTrunc {
                    d: self.from_diagonal_complex_spectrum(eigenvalues.clone())?,
                    v: self.from_bound_factor(v)?,
                    eigenvalues,
                    error,
                })
            })
        })
    }

    /// All general eigenvalues per coupled sector, descending by magnitude
    /// (MatrixAlgebraKit `eig_vals`). Complex for both dtypes.
    pub fn eig_vals(&self) -> Result<Vec<SectorSpectrum<Complex64>>, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.eig_vals();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                tenet_matrixalgebra::eig_vals_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )
            })
            .map_err(Into::into)
        })
    }

    /// Matrix exponential per coupled sector (TensorKit `exp`, which copies
    /// and calls `exp!`) — or, on compact diagonal storage, `exp` of
    /// each stored value.
    ///
    /// # Domain, and what storage decides
    ///
    /// Like TensorKit's, the dense arm accepts **any endomorphism** and picks
    /// its algorithm from the blocks (issue #577):
    ///
    /// - Hermitian blocks take the spectral function `v exp(d) v^H` of the
    ///   Hermitian eigendecomposition — exact, and the cheaper route;
    /// - every other block takes blockwise scaling-and-squaring Padé [13/13]
    ///   (Higham 2005), which is what the `LinearAlgebra.exp!` behind
    ///   TensorKit's own `exp!` runs. Nothing is symmetrized on the way.
    ///
    /// The **compact** arm is TensorKit's `exp(::DiagonalTensorMap)`, which is unconditionally elementwise,
    /// and so is this one: a diagonal's exponential is elementwise whether or
    /// not its spectrum is real. Storage therefore decides *how* `exp` is
    /// computed, not whether it is defined — a genuinely complex spectrum from
    /// [`Self::eig_full`] comes back the same either way, and the compact route
    /// simply never builds the dense buffer (issue #578).
    ///
    /// # Complexity
    ///
    /// Dense input: `O(Σ_c n_c³)` on both routes — one Hermitian
    /// eigendecomposition per coupled sector plus the composition that
    /// reassembles `v exp(d) v^H`, or six GEMMs, one solve and
    /// `s = max(0, ceil(log2(||A_c||_1 / theta_13)))` squarings per sector —
    /// over the balanced block, so a badly scaled one pays for its true
    /// magnitude and not its scaling — with an `O(max_c n_c²)` Padé workspace
    /// that every sector reuses. That workspace is the whole of the scratch on
    /// the canonical layout; a payload whose coupled sectors are not laid out
    /// in contiguous regions takes a fallback that matricizes them all first,
    /// adding `O(Σ_c n_c²)`. Sectors are never coupled. Compact input: `O(Σ_c k_c)` time and storage over the stored
    /// spectra, with no dense buffer, no EIGH and no GEMM — the result stays
    /// compact, so a following `compose` is still a bond scaling (issue #578).
    /// A dense lazy adjoint builds one operation-local logical payload per call
    /// without publishing its reusable receiver cache.
    ///
    /// # Errors
    ///
    /// The compact arm is infallible: any tensor that holds compact diagonal
    /// storage is an admissible input, on every rule. The dense arm reports
    ///
    /// - [`Error::Operation`] wrapping an `InvalidArgument` for a payload that
    ///   is not an endomorphism (`codomain != domain`), for a nonfinite entry
    ///   in a block bound for the general route, or for such a block whose
    ///   column 1-norm overflows to infinity although every entry of it is
    ///   finite;
    /// - [`Error::Operation`] wrapping a `Dense` failure from the backend,
    ///   including `DenseError::Unsupported` when the selected executor has no
    ///   dense solve — the general route needs one, and there is no implicit
    ///   host copy. Nothing is published unless every coupled sector succeeded;
    pub fn exp(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.exp();
        }
        // Compact diagonal: `exp` is elementwise on the spectrum (O(Σ_c k_c))
        // and stays diagonal, so it must not reach the dense lease below —
        // materializing to eigendecompose a matrix already given in its
        // eigenbasis is the complexity gap this arm closes.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.exp()));
        }
        with_data!(self, data, self.exp_impl(data))
    }

    fn exp_impl<D: UserScalar>(&self, data: &[D]) -> Result<Self, Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let out = tenet_matrixalgebra::exp_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            self.from_bound_factor(out)
        })
    }

    /// True inverse of a nonsingular map between isomorphic spaces
    /// (MatrixAlgebraKit-style `inv`). A host lazy adjoint solves its owned
    /// parent and returns a detached owned adjoint of that inverse, without
    /// allocating or publishing a separate receiver-materialization payload.
    pub fn inv(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.inv();
            }
            // (A†)^-1 = (A^-1)†. Solve the owned parent without publishing
            // the receiver's lazy materialization, then detach the result.
            return self
                .adjoint()?
                .inv()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        // A diagonal inverse is elementwise (O(rank)), not a block inversion;
        // keep it diagonal so the next contract still scales the bond.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.try_recip()?));
        }
        with_data!(self, data, self.inv_impl(data))
    }

    fn inv_impl<D: UserScalar>(&self, data: &[D]) -> Result<Self, Error> {
        let mut dense = self.rt.lease_dense();
        with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let out = tenet_matrixalgebra::inv_direct_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            self.from_bound_factor(out)
        })
    }

    /// Thresholded pseudo-inverse `t^+ = v s^+ u^H` (MatrixAlgebraKit `pinv`)
    /// with an `rcond * sigma_max` cutoff on the singular values. It is the
    /// exact Moore-Penrose inverse of the hard-thresholded effective-rank
    /// tensor, and of the original tensor only when no nonzero mode is cut.
    ///
    /// # Complexity
    ///
    /// Dense input: one compact SVD, sectorwise cubic, plus a bond scaling
    /// and one composition. Compact diagonal input skips the SVD entirely —
    /// its singular values are its `|entry|`s — and applies the
    /// cutoff-and-reciprocal elementwise in `O(Σ_c k_c)`, staying compact
    /// (itebd's `l_out.pinv` rides this arm).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `rcond` is not finite or is
    ///   negative — checked before any work, on both storages.
    /// - [`Error::Operation`] / [`Error::Core`] from the SVD, on the dense
    ///   arm; [`Error::UnsupportedOnDevice`] for a device payload.
    ///
    /// There is no singular-input failure: sending the offending directions
    /// to zero is what a pseudo-inverse is for.
    pub fn pinv(&self, rcond: f64) -> Result<Self, Error> {
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            ));
        }
        // A diagonal pseudo-inverse is an elementwise cutoff+reciprocal on the
        // spectrum (O(rank)) — its own singular values are |entry| — so skip the
        // SVD and keep it diagonal (itebd's `l_out.pinv` fires this).
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.pinv(rcond)));
        }
        if self.is_adjoint_view() {
            #[cfg(feature = "cuda")]
            if matches!(self.stored_data(), Data::CudaF64(_)) {
                return self.materialized_tensor()?.pinv(rcond);
            }
            let parent = self.adjoint()?;
            return with_data!(&parent, data, parent.pinv_adjoint_parent_impl(data, rcond));
        }
        with_data!(self, data, self.pinv_impl(data, rcond))
    }

    fn pinv_adjoint_parent_impl<D: UserScalar>(
        &self,
        data: &[D],
        rcond: f64,
    ) -> Result<Self, Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let out = tenet_matrixalgebra::pinv_adjoint_parent_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
                rcond,
            )?;
            self.from_bound_factor(out)
        })
    }

    fn pinv_impl<D: UserScalar>(&self, data: &[D], rcond: f64) -> Result<Self, Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let out = tenet_matrixalgebra::pinv_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
                rcond,
            )?;
            self.from_bound_factor(out)
        })
    }

    /// Elementwise square root of a diagonal bond tensor, i.e. the
    /// TensorKit 0.17 `sqrt(::DiagonalTensorMap)` idiom
    /// (`sqrt.(d.data)` on the diagonal)
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
    /// A dense lazy adjoint builds one operation-local logical payload without
    /// publishing its reusable receiver cache.
    ///
    pub fn sqrt(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor_uncached()?.sqrt();
        }
        let hom = self.ordinary_body().space.homspace();
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
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.try_sqrt()?));
        }
        let data = match self.coupled_data()? {
            Data::F64(data) => Data::F64(sqrt_diagonal_impl(
                &self.ordinary_body().space,
                data,
                &|value| {
                    if value < 0.0 {
                        Err(Error::InvalidArgument(format!(
                            "sqrt of a negative diagonal entry {value}; convert to c64 \
                         with to_c64() for the complex square root"
                        )))
                    } else {
                        Ok(value.sqrt())
                    }
                },
            )?),
            Data::C64(data) => Data::C64(sqrt_diagonal_impl(
                &self.ordinary_body().space,
                data,
                &|value| Ok(value.sqrt()),
            )?),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("sqrt")),
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
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

fn logical_space_mismatch() -> Error {
    Error::InvalidArgument("tensors live on different spaces or block layouts".to_string())
}

#[allow(clippy::too_many_arguments)]
fn oriented_add_data<D: UserScalar>(
    destination: &DynamicFusionMapSpace,
    lhs: tenet_tensors::FusionOperand<'_>,
    lhs_data: &Data,
    rhs: tenet_tensors::FusionOperand<'_>,
    rhs_data: &Data,
    alpha: D,
    beta: D,
) -> Result<Data, Error> {
    let lhs_data = D::data_slice(lhs_data).ok_or(Error::DtypeMismatch)?;
    let rhs_data = D::data_slice(rhs_data).ok_or(Error::DtypeMismatch)?;
    let mut output = vec![D::from_real(0.0); destination.structure().required_len()?];
    tenet_tensors::oriented_fusion_add_into(
        destination.structure(),
        &mut output,
        lhs,
        lhs_data,
        rhs,
        rhs_data,
        alpha,
        beta,
    )?;
    Ok(D::lift(output))
}

fn oriented_inner_layout<'a>(
    lhs: &TensorMetadataView<'a>,
    rhs: &TensorMetadataView<'a>,
) -> (
    &'a BlockStructure,
    tenet_tensors::FusionOperand<'a>,
    tenet_tensors::FusionOperand<'a>,
    bool,
) {
    match (lhs.orientation, rhs.orientation) {
        (TensorOrientation::Adjoint, TensorOrientation::Adjoint) => (
            lhs.body.space.structure(),
            tenet_tensors::FusionOperand::direct(rhs.body.space.raw()),
            tenet_tensors::FusionOperand::direct(lhs.body.space.raw()),
            true,
        ),
        (lhs_orientation, rhs_orientation) => (
            if matches!(lhs_orientation, TensorOrientation::Owned) {
                lhs.body.space.structure()
            } else {
                rhs.body.space.structure()
            },
            if matches!(lhs_orientation, TensorOrientation::Adjoint) {
                tenet_tensors::FusionOperand::adjoint(lhs.body.space.raw())
            } else {
                tenet_tensors::FusionOperand::direct(lhs.body.space.raw())
            },
            if matches!(rhs_orientation, TensorOrientation::Adjoint) {
                tenet_tensors::FusionOperand::adjoint(rhs.body.space.raw())
            } else {
                tenet_tensors::FusionOperand::direct(rhs.body.space.raw())
            },
            false,
        ),
    }
}

fn oriented_dense_inner<D: UserScalar>(
    logical: &BlockStructure,
    lhs_operand: tenet_tensors::FusionOperand<'_>,
    lhs_data: &[D],
    rhs_operand: tenet_tensors::FusionOperand<'_>,
    rhs_data: &[D],
    swap_data: bool,
    weight: impl Fn(SectorId) -> f64,
) -> Result<Complex64, Error> {
    let (lhs_data, rhs_data) = if swap_data {
        (rhs_data, lhs_data)
    } else {
        (lhs_data, rhs_data)
    };
    tenet_tensors::oriented_fusion_inner(
        logical,
        lhs_operand,
        lhs_data,
        rhs_operand,
        rhs_data,
        |sector| D::from_real(weight(sector)),
    )
    .map(FactorScalar::widen_complex)
    .map_err(Error::from)
}

#[cfg(test)]
fn odometer_inner_oracle<D, W>(
    structure: &BlockStructure,
    a: &[D],
    b: &[D],
    mut weight_of: W,
) -> Result<Complex64, Error>
where
    D: UserScalar,
    W: FnMut(SectorId) -> f64,
{
    let mut total = Complex64::new(0.0, 0.0);
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(internal_layout_error(
                "inner-product oracle requires fusion-tree blocks",
            ));
        };
        let coupled = key.codomain_tree().coupled();
        let shape = block.shape();
        let strides = block.strides();
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        let mut partial = D::from_real(0.0);
        for _ in 0..count {
            let position = block.offset()
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &stride)| i * stride)
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
        total += partial.widen_complex() * weight_of(coupled);
    }
    Ok(total)
}

#[cfg(test)]
mod coupled_region_inner_tests {
    use super::*;

    fn assert_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual - expected).norm() <= 1.0e-11 * (1.0 + expected.norm()),
            "actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_multiplicity_free_oracle(space: Space, seed: u64) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        for dtype in [Dtype::F64, Dtype::C64] {
            let lhs =
                Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed).unwrap();
            let rhs = Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed + 1)
                .unwrap();
            let structure = lhs.ordinary_body().space.structure();
            let expected = match (lhs.coupled_data().unwrap(), rhs.coupled_data().unwrap()) {
                (Data::F64(a), Data::F64(b)) => {
                    with_user_rule!(lhs.ordinary_body().space, rule, {
                        odometer_inner_oracle(structure, a, b, |coupled| rule.dim_scalar(coupled))
                    })
                }
                (Data::C64(a), Data::C64(b)) => {
                    with_user_rule!(lhs.ordinary_body().space, rule, {
                        odometer_inner_oracle(structure, a, b, |coupled| rule.dim_scalar(coupled))
                    })
                }
                _ => unreachable!(),
            }
            .unwrap();
            let actual = lhs.inner(&rhs).unwrap().to_c64();
            assert_close(actual, expected);
            assert_close(rhs.inner(&lhs).unwrap().to_c64(), actual.conj());
            assert_close(
                lhs.inner(&lhs).unwrap().to_c64(),
                Complex64::new(lhs.norm().unwrap().powi(2), 0.0),
            );
        }
    }

    #[test]
    fn non_abelian_regions_match_the_block_odometer_oracle() {
        // What: contiguous coupled-sector reduction preserves every block of
        // multi-sector, multi-tree SU(2) and fermionic product tensors.
        assert_multiplicity_free_oracle(
            Space::su2([(0, 2), (1, 2), (2, 1), (3, 1)]).unwrap(),
            282_001,
        );
        assert_multiplicity_free_oracle(
            Space::fz2_u1_su2([
                ((0, -2, 0), 2),
                ((0, 1, 2), 1),
                ((1, -1, 1), 2),
                ((1, 2, 3), 1),
            ])
            .unwrap(),
            282_101,
        );
    }

    #[test]
    fn malformed_scalar_range_is_a_typed_internal_layout_error() {
        // What: the lowest contiguous-region boundary rejects a scalar buffer
        // that cannot be covered exactly instead of entering an odometer path.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap();
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 282_301).unwrap();
        let data = tensor.data();
        let error = coupled_region_inner(
            tensor.ordinary_body().space.structure(),
            tensor.ordinary_body().space.nout(),
            &data[..data.len() - 1],
            data,
            |_| 1.0,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(message) if
            message.contains("internal coupled-layout invariant violated")));
    }

    #[test]
    fn empty_and_non_fusion_structures_keep_explicit_boundary_semantics() {
        // What: a canonical empty structure reduces to zero, while an ordinal
        // dense structure is rejected as non-packed coupled-sector storage.
        let empty = BlockStructure::empty(3);
        assert_eq!(
            coupled_region_inner::<f64, _>(&empty, 1, &[], &[], |_| 7.0).unwrap(),
            Complex64::new(0.0, 0.0)
        );

        let trivial = BlockStructure::trivial(&[2, 2]).unwrap();
        let error = coupled_region_inner(&trivial, 1, &[1.0; 4], &[1.0; 4], |_| 1.0).unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(message) if
            message.contains("non-packed coupled-sector layout")));
    }
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

#[cfg(feature = "cuda")]
fn validate_cuda_zero_placement(device: usize, actual: Placement) -> Result<(), Error> {
    if actual != Placement::Cuda(device) {
        return Err(Error::PlacementMismatch);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn coupled_sector_of(region: &CoupledSectorRegion) -> SectorId {
    region.coupled()
}

#[cfg(feature = "cuda")]
fn find_source<'a>(
    regions: &'a [CoupledSectorRegion],
    target: &CoupledSectorRegion,
) -> Result<(usize, &'a CoupledSectorRegion), Error> {
    regions
        .iter()
        .enumerate()
        .find(|(_, region)| region.coupled() == target.coupled())
        .ok_or_else(|| internal_layout_error("factor bond sector missing in the source tensor"))
}

#[cfg(feature = "cuda")]
impl Tensor {
    /// Device weighted Frobenius inner product: one `[1, len] x [len, 1]`
    /// region GEMM per coupled sector into a device partials buffer, then a
    /// single download of the per-sector scalars, weighted by quantum
    /// dimensions on the host.
    fn weighted_inner_cuda(&self, a: &CudaStorage, b: &CudaStorage) -> Result<Complex64, Error> {
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let mut partials = CudaStorage::upload(cuda, &vec![0.0; regions.len().max(1)])?;
        for (index, region) in regions.iter().enumerate() {
            let len = region.rows() * region.cols();
            if len == 0 {
                continue;
            }
            cuda_gemm_region_into(
                cuda,
                &mut partials.0,
                index,
                1,
                &a.0,
                region.range().start,
                1,
                &b.0,
                region.range().start,
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
        let total = with_user_rule!(self.ordinary_body().space, rule, {
            regions
                .iter()
                .zip(&values)
                .map(|(region, &value)| value * rule.dim_scalar(coupled_sector_of(region)))
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
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let out = with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let rule = bound.provider();
            let mut spectra: Vec<SectorSpectrum> = Vec::with_capacity(regions.len());
            let mut factors: Vec<Option<(CudaDenseStorage, CudaDenseStorage)>> =
                Vec::with_capacity(regions.len());
            for region in regions.iter() {
                let sector = coupled_sector_of(region);
                if region.rows() == 0 || region.cols() == 0 {
                    spectra.push(SectorSpectrum {
                        sector,
                        values: Vec::new(),
                    });
                    factors.push(None);
                    continue;
                }
                let (u, s, vt) = cuda_svd_region(
                    cuda,
                    &storage.0,
                    region.range().start,
                    region.rows(),
                    region.cols(),
                )?;
                spectra.push(SectorSpectrum { sector, values: s });
                factors.push(Some((u, vt)));
            }
            let (kept_spectra, error) = decide_kept(rule, &spectra, truncation)?;

            let hom = self.ordinary_body().space.homspace();
            let bond_leg = SectorLeg::new(
                kept_spectra
                    .iter()
                    .map(|entry| (entry.sector, entry.values.len())),
                false,
            );
            let build_output_space = |hom| {
                let space = build_bound_space_like(bound, hom)?;
                UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
            };
            let u_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let s_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg.clone()]),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let vh_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg]),
                FusionProductSpace::new(hom.domain().legs().iter().cloned()),
            ))?;

            let mut u_data = CudaStorage::upload(cuda, &vec![0.0; u_space.required_len()?])?;
            for target in sector_regions(u_space.structure(), u_space.nout())?.iter() {
                let kept = target.cols();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((u_dev, _)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let k_full = source.rows().min(source.cols());
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
            for target in sector_regions(vh_space.structure(), vh_space.nout())?.iter() {
                let kept = target.rows();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((_, vt_dev)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let k_full = source.rows().min(source.cols());
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
                u: self.with_bound(u_space, Data::CudaF64(Arc::new(u_data)))?,
                s: self.with_bound(s_space, Data::CudaF64(Arc::new(s_data)))?,
                vh: self.with_bound(vh_space, Data::CudaF64(Arc::new(vh_data)))?,
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
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let out = with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let mut factors: Vec<Option<(CudaDenseStorage, CudaDenseStorage, Vec<f64>)>> =
                Vec::with_capacity(regions.len());
            let mut bond_pairs: Vec<(SectorId, usize)> = Vec::with_capacity(regions.len());
            for region in regions.iter() {
                let sector = coupled_sector_of(region);
                if region.rows() == 0 || region.cols() == 0 {
                    bond_pairs.push((sector, 0));
                    factors.push(None);
                    continue;
                }
                let (q, r, diag) = cuda_qr_region(
                    cuda,
                    &storage.0,
                    region.range().start,
                    region.rows(),
                    region.cols(),
                )?;
                // Positive-diagonal gauge (host `positive_diagonal_gauge`,
                // real scalars): flip where R's diagonal is negative, leave
                // exact zeros untouched.
                let signs: Vec<f64> = diag
                    .iter()
                    .map(|&value| if value < 0.0 { -1.0 } else { 1.0 })
                    .collect();
                bond_pairs.push((sector, region.rows().min(region.cols())));
                factors.push(Some((q, r, signs)));
            }

            let hom = self.ordinary_body().space.homspace();
            let bond_leg = SectorLeg::new(bond_pairs.iter().copied(), false);
            let build_output_space = |hom| {
                let space = build_bound_space_like(bound, hom)?;
                UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
            };
            let q_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let r_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg]),
                FusionProductSpace::new(hom.domain().legs().iter().cloned()),
            ))?;

            let mut q_data = CudaStorage::upload(cuda, &vec![0.0; q_space.required_len()?])?;
            for target in sector_regions(q_space.structure(), q_space.nout())?.iter() {
                let kept = target.cols();
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
            for target in sector_regions(r_space.structure(), r_space.nout())?.iter() {
                let kept = target.rows();
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
                self.with_bound(q_space, Data::CudaF64(Arc::new(q_data)))?,
                self.with_bound(r_space, Data::CudaF64(Arc::new(r_data)))?,
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
            let hom = self.ordinary_body().space.homspace();
            if hom.codomain() != hom.domain() {
                return Err(Error::InvalidArgument(
                    "eigh requires an endomorphism (codomain == domain)".to_string(),
                ));
            }
        }
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        // No device validator exists; skipping this copy lets cuSOLVER silently trust one triangle.
        let host_data = storage.0.download_f64(cuda).map_err(dense_err)?;
        validate_hermitian_regions(&host_data, &regions)?;
        let out = with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let rule = bound.provider();
            let mut spectra: Vec<SectorSpectrum> = Vec::with_capacity(regions.len());
            let mut factors: Vec<Option<(CudaDenseStorage, Vec<usize>)>> =
                Vec::with_capacity(regions.len());
            for region in regions.iter() {
                let sector = coupled_sector_of(region);
                let n = region.rows();
                if n == 0 {
                    spectra.push(SectorSpectrum {
                        sector,
                        values: Vec::new(),
                    });
                    factors.push(None);
                    continue;
                }
                let (values, vectors) = cuda_eigh_region(cuda, &storage.0, region.range().start, n)
                    .map_err(dense_err)?;
                if !values.iter().all(|value| value.is_finite()) {
                    return Err(Error::InvalidArgument(
                        "eigenvalues must be finite".to_string(),
                    ));
                }
                // Host ordering contract: descending by |eigenvalue|,
                // stable on ties (mirrors `eigh_full_dyn`).
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| values[b].abs().total_cmp(&values[a].abs()).then(a.cmp(&b)));
                let sorted: Vec<f64> = order.iter().map(|&index| values[index]).collect();
                spectra.push(SectorSpectrum {
                    sector,
                    values: sorted,
                });
                factors.push(Some((vectors, order)));
            }
            let (kept_spectra, error) = decide_kept(rule, &spectra, truncation)?;

            let hom = self.ordinary_body().space.homspace();
            let bond_leg = SectorLeg::new(
                kept_spectra
                    .iter()
                    .map(|entry| (entry.sector, entry.values.len())),
                false,
            );
            let build_output_space = |hom| {
                let space = build_bound_space_like(bound, hom)?;
                UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
            };
            let v_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let d_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg.clone()]),
                FusionProductSpace::new([bond_leg]),
            ))?;

            let mut v_data = CudaStorage::upload(cuda, &vec![0.0; v_space.required_len()?])?;
            for target in sector_regions(v_space.structure(), v_space.nout())?.iter() {
                let kept = target.cols();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((v_dev, order)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let n = source.rows();
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
                d: self.with_bound(d_space, Data::CudaF64(Arc::new(d_data)))?,
                v: self.with_bound(v_space, Data::CudaF64(Arc::new(v_data)))?,
                eigenvalues: kept_spectra,
                error,
            })
        })?;
        drop(guard);
        Ok(out)
    }
}

/// Legacy erased tensor constructors on the runtime itself.
impl Runtime {
    /// Constructs a legacy erased zero tensor on this runtime.
    pub fn zeros<'a, C, D>(&self, dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::zeros(self, dtype, codomain, domain)
    }

    /// Constructs a legacy erased random tensor on this runtime.
    pub fn rand<'a, C, D>(&self, dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::rand(self, dtype, codomain, domain)
    }

    /// Constructs a seeded legacy erased random tensor on this runtime.
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

    /// Constructs a legacy erased identity tensor on this runtime.
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

#[cfg(test)]
mod unit_layout_tensor_tests {
    use super::*;

    fn assert_unit_roundtrip(tensor: &Tensor, position: usize, dual: bool, left: bool) {
        let data = Arc::clone(&tensor.ordinary_body().data);
        let inserted = if left {
            tensor.insert_left_unit(position, dual).unwrap()
        } else {
            tensor.insert_right_unit(position, dual).unwrap()
        };
        assert!(Arc::ptr_eq(&inserted.ordinary_body().data, &data));
        let restored = inserted.remove_unit(position).unwrap();
        assert_eq!(restored.codomain_spaces(), tensor.codomain_spaces());
        assert_eq!(restored.domain_spaces(), tensor.domain_spaces());
        assert!(Arc::ptr_eq(&restored.ordinary_body().data, &data));
    }

    #[test]
    fn unit_layout_uses_tensorkit_slots_without_symmetry_branches() {
        // What: U1 covers rank-zero/start/seam/end slots and both dual flags;
        // SU2, odd fZ2, and product spaces take the same metadata-only path.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-1, 1), (0, 2), (1, 1)]);
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 549_001).unwrap();
        let scalar = tensor.trace_pairs(&[(0, 1)]).unwrap();
        for dual in [false, true] {
            assert_unit_roundtrip(&scalar, 0, dual, true);
            assert_unit_roundtrip(&scalar, 0, dual, false);
            for position in 0..=tensor.rank() {
                assert_unit_roundtrip(&tensor, position, dual, true);
                assert_unit_roundtrip(&tensor, position, dual, false);
            }
        }

        let left_seam = tensor.insert_left_unit(1, false).unwrap();
        assert_eq!((left_seam.codomain_rank(), left_seam.domain_rank()), (1, 2));
        assert!(left_seam.space(1).unwrap().is_dual());
        let right_seam = tensor.insert_right_unit(1, true).unwrap();
        assert_eq!(
            (right_seam.codomain_rank(), right_seam.domain_rank()),
            (2, 1)
        );

        let su2 = Space::su2([(0, 1), (1, 2), (2, 1)]).unwrap();
        let su2_tensor =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&su2, &su2], [&su2], 549_003).unwrap();
        assert!(su2_tensor.ordinary_body().space.structure().block_count() > 1);
        assert_unit_roundtrip(&su2_tensor, 1, true, false);

        let odd = Space::fz2([(1, 1)]).unwrap();
        let odd_tensor = Tensor::zeros(&runtime, Dtype::F64, [&odd], [&odd]).unwrap();
        assert_unit_roundtrip(&odd_tensor, 1, false, true);

        let product = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, -1, 1), 2)]).unwrap();
        let product_tensor = Tensor::zeros(&runtime, Dtype::F64, [&product], [&product]).unwrap();
        assert_unit_roundtrip(&product_tensor, 1, true, false);
    }

    #[test]
    fn unit_layout_materializes_compact_and_lazy_once_then_shares() {
        // What: compact diagonal and lazy-adjoint inputs only materialize at
        // the documented boundary, and the result reuses that Arc<Data>.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(0, 2), (1, 1)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 549_002).unwrap();

        let compact = source.svd_compact().unwrap().1;
        assert!(!compact.has_cached_materialization());
        let compact_out = compact.insert_right_unit(1, false).unwrap();
        assert!(compact.has_cached_materialization());
        assert!(Arc::ptr_eq(
            &compact_out.ordinary_body().data,
            compact.compact_dense.get().unwrap()
        ));

        let lazy = source.adjoint().unwrap();
        assert_eq!(lazy.adjoint_body_builds(), 0);
        let lazy_out = lazy.insert_left_unit(0, true).unwrap();
        assert_eq!(lazy.adjoint_body_builds(), 1);
        assert!(Arc::ptr_eq(
            &lazy_out.ordinary_body().data,
            &lazy.materialized_body().unwrap().data
        ));
    }

    #[test]
    fn unit_layout_preflight_leaves_lazy_adjoint_unmaterialized() {
        // What: invalid axis and non-unit requests fail before the lazy
        // adjoint's dense body is built.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(1, 1)]);
        let lazy = Tensor::zeros(&runtime, Dtype::F64, [&space], [&space])
            .unwrap()
            .adjoint()
            .unwrap();

        assert!(matches!(
            lazy.insert_left_unit(lazy.rank() + 1, false),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            lazy.remove_unit(0),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }
}

#[cfg(test)]
mod compact_diagonal_tests;

#[cfg(test)]
mod adjoint_parent_view_tests {
    use super::*;
    use crate::space::SectorLabel;
    use tenet_core::FusionAlgebraError;

    fn assert_scalar_close(actual: Scalar, expected: Scalar) {
        assert!((actual.to_c64() - expected.to_c64()).norm() < 1e-11);
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.codomain_spaces(), expected.codomain_spaces());
        assert_eq!(actual.domain_spaces(), expected.domain_spaces());
        assert_eq!(actual.dtype(), expected.dtype());
        match (
            actual.coupled_data().unwrap(),
            expected.coupled_data().unwrap(),
        ) {
            (Data::F64(actual), Data::F64(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (&actual, &expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-11);
                }
            }
            (Data::C64(actual), Data::C64(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (&actual, &expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).norm() < 1e-11);
                }
            }
            _ => panic!("dtype mismatch"),
        }
    }

    fn assert_spectra_close(actual: &[SectorSpectrum], expected: &[SectorSpectrum]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.sector, expected.sector);
            assert_eq!(actual.values.len(), expected.values.len());
            for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
                assert!((actual - expected).abs() < 1e-11);
            }
        }
    }

    fn assert_erased_eigh_uses_a_cold_logical_copy(parent: &Tensor) {
        let eager = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected_vals = eager.eigh_vals().unwrap();
        let expected_full = eager.eigh_full().unwrap();
        let expected_trunc = eager.eigh_trunc(&Truncation::rank(1)).unwrap();
        let parent_space = Arc::clone(&parent.ordinary_body().space);
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let lazy = parent.adjoint().unwrap();

        for _ in 0..2 {
            // This fixture's logical/parent split is only about 4e-15: keep
            // eigenvalues exact so a parent-result redirect cannot hide under
            // the general reconstruction tolerance used by the C64 tests.
            assert_eq!(lazy.clone().eigh_vals().unwrap(), expected_vals);
            let full = lazy.clone().eigh_full().unwrap();
            assert_eq!(full.0.data(), expected_full.0.data());
            assert_close(&full.1, &expected_full.1);
            let trunc = lazy.clone().eigh_trunc(&Truncation::rank(1)).unwrap();
            assert_eq!(trunc.eigenvalues, expected_trunc.eigenvalues);
            assert_eq!(trunc.error, expected_trunc.error);
            assert_eq!(trunc.d.data(), expected_trunc.d.data());
            assert_close(&trunc.v, &expected_trunc.v);
            for output in [&full.0, &full.1, &trunc.d, &trunc.v] {
                assert!(!output.is_adjoint_view());
                assert!(output
                    .rule_authority_space()
                    .provider_matches_context_allocation(&parent.rule_authority_space().context()));
            }
        }

        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || {
                    let vals = clone.eigh_vals().unwrap();
                    let full = clone.eigh_full().unwrap();
                    let trunc = clone.eigh_trunc(&Truncation::rank(1)).unwrap();
                    (vals, full.0.data().to_vec(), trunc.error)
                })
            })
            .collect::<Vec<_>>();
        for call in calls {
            let (vals, diagonal, error) = call.join().unwrap();
            assert_eq!(vals, expected_vals);
            assert_eq!(diagonal, expected_full.0.data());
            assert_eq!(error, expected_trunc.error);
        }
        assert!(Arc::ptr_eq(&parent.ordinary_body().space, &parent_space));
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn erased_eigh_dense_lazy_near_hermitian_uses_logical_triangle_and_stays_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(0, 2)]);
        let parent = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) | (1, 1) => 1.0,
                (0, 1) => 4.0e-15,
                _ => 0.0,
            }
        })
        .unwrap();
        let logical = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let logical_vals = logical.eigh_vals().unwrap();
        let parent_vals = parent.eigh_vals().unwrap();
        assert!(logical_vals[0]
            .values
            .iter()
            .zip(&parent_vals[0].values)
            .any(|(logical, parent)| (logical - parent).abs() > 1.0e-15));
        let logical_trunc = logical.eigh_trunc(&Truncation::rank(1)).unwrap();
        let parent_trunc = parent.eigh_trunc(&Truncation::rank(1)).unwrap();
        assert_ne!(logical_trunc.eigenvalues, parent_trunc.eigenvalues);
        assert_ne!(logical_trunc.error, parent_trunc.error);

        assert_erased_eigh_uses_a_cold_logical_copy(&parent);
    }

    #[test]
    fn erased_eigh_dense_lazy_complex_failure_matches_logical_oracles() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(0, 2)]);
        let hermitian = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => Complex64::new(2.0, 0.0),
                (1, 1) => Complex64::new(3.0, 0.0),
                (0, 1) => Complex64::new(0.0, 1.0),
                (1, 0) => Complex64::new(0.0, -1.0),
                _ => unreachable!(),
            }
        })
        .unwrap();
        let eager = hermitian
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected = eager.eigh_full().unwrap();
        let lazy = hermitian.adjoint().unwrap();
        let actual = lazy.eigh_full().unwrap();
        assert_close(&actual.0, &expected.0);
        assert_close(&actual.1, &expected.1);
        let reconstructed = actual
            .1
            .compose(&actual.0)
            .unwrap()
            .compose(&actual.1.adjoint().unwrap())
            .unwrap();
        assert_close(&reconstructed, &eager);
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());

        let nonhermitian = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => 1.0,
                (1, 1) => 2.0,
                (0, 1) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
        let eager = nonhermitian
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected = [
            eager.eigh_vals().unwrap_err().to_string(),
            eager.eigh_full().unwrap_err().to_string(),
            eager
                .eigh_trunc(&Truncation::rank(1))
                .unwrap_err()
                .to_string(),
        ];
        let lazy = nonhermitian.adjoint().unwrap();
        for _ in 0..2 {
            assert_eq!(lazy.eigh_vals().unwrap_err().to_string(), expected[0]);
            assert_eq!(lazy.eigh_full().unwrap_err().to_string(), expected[1]);
            assert_eq!(
                lazy.eigh_trunc(&Truncation::rank(1))
                    .unwrap_err()
                    .to_string(),
                expected[2]
            );
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_erased_eig_uses_a_cold_logical_copy(parent: &Tensor) {
        let eager = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected_vals = eager.eig_vals().unwrap();
        let expected_full = eager.eig_full().unwrap();
        let expected_trunc = eager.eig_trunc(&Truncation::rank(1)).unwrap();
        let parent_space = Arc::clone(&parent.ordinary_body().space);
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let lazy = parent.adjoint().unwrap();

        for _ in 0..2 {
            assert_eq!(lazy.clone().eig_vals().unwrap(), expected_vals);
            let full = lazy.clone().eig_full().unwrap();
            assert_eq!(full.0.try_data_c64().unwrap(), expected_full.0.data_c64());
            assert_eq!(full.1.try_data_c64().unwrap(), expected_full.1.data_c64());
            let trunc = lazy.clone().eig_trunc(&Truncation::rank(1)).unwrap();
            assert_eq!(trunc.eigenvalues, expected_trunc.eigenvalues);
            assert_eq!(trunc.error, expected_trunc.error);
            assert_eq!(trunc.d.try_data_c64().unwrap(), expected_trunc.d.data_c64());
            assert_eq!(trunc.v.try_data_c64().unwrap(), expected_trunc.v.data_c64());
            for output in [&full.0, &full.1, &trunc.d, &trunc.v] {
                assert!(!output.is_adjoint_view());
                assert!(output
                    .rule_authority_space()
                    .provider_matches_context_allocation(&parent.rule_authority_space().context()));
            }
        }

        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || {
                    let vals = clone.eig_vals().unwrap();
                    let full = clone.eig_full().unwrap();
                    let trunc = clone.eig_trunc(&Truncation::rank(1)).unwrap();
                    (vals, full.0.data_c64().to_vec(), trunc.error)
                })
            })
            .collect::<Vec<_>>();
        for call in calls {
            let (vals, diagonal, error) = call.join().unwrap();
            assert_eq!(vals, expected_vals);
            assert_eq!(diagonal, expected_full.0.data_c64());
            assert_eq!(error, expected_trunc.error);
        }
        assert!(Arc::ptr_eq(&parent.ordinary_body().space, &parent_space));
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());

        let _ = lazy.data();
        assert_eq!(lazy.adjoint_body_builds(), 1);
        assert!(lazy.has_cached_materialization());
    }

    #[test]
    fn erased_eig_dense_lazy_nonnormal_is_logical_owned_repeatable_and_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(0, 2)]);
        let parent = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => 1.0,
                (1, 1) => 2.0,
                (0, 1) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
        let logical = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let (d, v) = parent.adjoint().unwrap().eig_full().unwrap();
        let lhs = logical.to_c64().compose(&v).unwrap();
        let rhs = v.compose(&d).unwrap();
        let residual = lhs.add(&rhs, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(residual < 1.0e-12, "A^H V - V D residual={residual:e}");

        assert_erased_eig_uses_a_cold_logical_copy(&parent);
    }

    #[test]
    fn erased_eig_dense_lazy_signed_zero_order_complex_tie_and_jordan_match_oracles() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let scalar_leg = Space::u1([(0, 1)]);
        let negative =
            Tensor::from_block_fn(&runtime, [&scalar_leg], [&scalar_leg], |_, _| -2.0).unwrap();
        let lazy = negative.adjoint().unwrap();
        let value = lazy.eig_vals().unwrap()[0].values[0];
        assert_eq!(value, Complex64::new(-2.0, 0.0));
        assert_eq!(value.im.to_bits(), 0.0f64.to_bits());
        assert_eq!(lazy.adjoint_body_builds(), 0);

        let leg = Space::u1([(0, 2)]);
        let rotation = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 1) => -1.0,
                (1, 0) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
        let eager = rotation
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected = eager.eig_vals().unwrap();
        let lazy = rotation.adjoint().unwrap();
        assert_eq!(lazy.eig_vals().unwrap(), expected);
        assert_eq!(expected[0].values[0].im, 1.0);
        assert_eq!(expected[0].values[1].im, -1.0);
        let actual = lazy.eig_trunc(&Truncation::rank(1)).unwrap();
        let expected_trunc = eager.eig_trunc(&Truncation::rank(1)).unwrap();
        assert_eq!(actual.eigenvalues, expected_trunc.eigenvalues);
        assert_eq!(actual.d.data_c64(), expected_trunc.d.data_c64());
        assert_eq!(actual.v.data_c64(), expected_trunc.v.data_c64());
        assert_eq!(actual.error, expected_trunc.error);
        assert_eq!(lazy.adjoint_body_builds(), 0);

        let complex = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                if indices[0] == 0 {
                    Complex64::new(1.0, 1.0)
                } else {
                    Complex64::new(-1.0, 1.0)
                }
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();
        let eager = complex
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let lazy = complex.adjoint().unwrap();
        assert_eq!(lazy.eig_vals().unwrap(), eager.eig_vals().unwrap());
        let actual = lazy.eig_full().unwrap();
        let expected = eager.eig_full().unwrap();
        assert_eq!(actual.0.data_c64(), expected.0.data_c64());
        assert_eq!(actual.1.data_c64(), expected.1.data_c64());
        assert_eq!(lazy.adjoint_body_builds(), 0);

        for epsilon in [0.0, 1.0e-12] {
            let jordan = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                match (indices[0], indices[1]) {
                    (0, 0) | (1, 1) => 1.0,
                    (0, 1) => 1.0,
                    (1, 0) => epsilon,
                    _ => unreachable!(),
                }
            })
            .unwrap();
            let eager = jordan
                .adjoint()
                .unwrap()
                .materialized_tensor_uncached()
                .unwrap();
            let lazy = jordan.adjoint().unwrap();
            assert_eq!(lazy.eig_vals().unwrap(), eager.eig_vals().unwrap());
            let actual = lazy.eig_full().unwrap();
            let expected = eager.eig_full().unwrap();
            assert_eq!(actual.0.data_c64(), expected.0.data_c64());
            assert_eq!(actual.1.data_c64(), expected.1.data_c64());
            assert_eq!(lazy.adjoint_body_builds(), 0);
        }
    }

    #[test]
    fn erased_eig_dense_lazy_failures_match_exact_logical_errors() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let left = Space::u1([(0, 2)]);
        let right = Space::u1([(0, 3)]);
        let parent = Tensor::from_block_fn(&runtime, [&left], [&right], |_, indices| {
            (indices[0] + indices[1]) as f64
        })
        .unwrap();
        let eager = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected = [
            eager.eig_vals().unwrap_err().to_string(),
            eager.eig_full().unwrap_err().to_string(),
            eager
                .eig_trunc(&Truncation::rank(1))
                .unwrap_err()
                .to_string(),
        ];
        let lazy = parent.adjoint().unwrap();
        for _ in 0..2 {
            assert_eq!(lazy.eig_vals().unwrap_err().to_string(), expected[0]);
            assert_eq!(lazy.eig_full().unwrap_err().to_string(), expected[1]);
            assert_eq!(
                lazy.eig_trunc(&Truncation::rank(1))
                    .unwrap_err()
                    .to_string(),
                expected[2]
            );
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn erased_exp_of_a_near_hermitian_adjoint_uses_the_logical_orientation_and_stays_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(0, 2)]);
        let delta = 4.0e-15;
        let parent = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => Complex64::new(0.25, 0.0),
                (1, 1) => Complex64::new(-0.5, 0.0),
                (0, 1) => Complex64::new(delta, 0.0),
                _ => Complex64::new(0.0, 0.0),
            }
        })
        .unwrap();
        let expected = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap()
            .exp()
            .unwrap();
        let parent_redirect = parent
            .exp()
            .unwrap()
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let lazy = parent.adjoint().unwrap();
        let actual = lazy.exp().unwrap();

        assert_close(&actual, &expected);
        assert!(actual
            .data_c64()
            .iter()
            .zip(expected.data_c64())
            .all(|(&actual, &expected)| (actual - expected).norm() < 1.0e-20));
        assert!(actual
            .data_c64()
            .iter()
            .zip(parent_redirect.data_c64())
            .any(|(&actual, &redirect)| (actual - redirect).norm() > 1.0e-16));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_erased_exp_uses_a_cold_logical_copy(parent: &Tensor) {
        let expected = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap()
            .exp()
            .unwrap();
        let parent_space = Arc::clone(&parent.ordinary_body().space);
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let lazy = parent.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().exp().unwrap();
            assert_close(&actual, &expected);
            assert!(!actual.is_adjoint_view());
            assert!(actual
                .rule_authority_space()
                .provider_matches_context_allocation(&parent.rule_authority_space().context()));
            assert!(!Arc::ptr_eq(&actual.ordinary_body().data, &parent_data));
            let _ = actual.coupled_data().unwrap();
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.exp().unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_close(&call.join().unwrap(), &expected);
        }
        assert!(Arc::ptr_eq(&parent.ordinary_body().space, &parent_space));
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn erased_exp_returns_owned_provider_native_outputs_and_keeps_receivers_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1 = Space::u1([(-1, 2), (0, 3), (2, 1)]);
        let parent = Tensor::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            let re = if indices[0] == indices[1] {
                0.25 + indices[0] as f64 / 10.0
            } else {
                (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
            };
            Complex64::new(re, (indices[0] + indices[1] + 1) as f64 / 200.0)
        })
        .unwrap();
        assert_erased_exp_uses_a_cold_logical_copy(&parent);

        let half = Space::su2([(1, 1)]).unwrap();
        let parent = Tensor::from_block_fn(
            &runtime,
            [&half, &half, &half],
            [&half, &half, &half],
            |_, indices| (indices.iter().sum::<usize>() + 1) as f64 / 20.0,
        )
        .unwrap();
        assert!(parent.ordinary_body().space.structure().block_count() > 1);
        assert_erased_exp_uses_a_cold_logical_copy(&parent);
    }

    #[test]
    fn erased_exp_failures_keep_the_receiver_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(0, 2)]);
        let parent = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices == [0, 1] {
                Complex64::new(f64::NAN, 0.0)
            } else {
                Complex64::new((indices[0] + indices[1] + 1) as f64, 0.0)
            }
        })
        .unwrap();
        let before = parent.data_c64().to_vec();
        let data = Arc::clone(&parent.ordinary_body().data);
        let lazy = parent.adjoint().unwrap();
        assert!(matches!(lazy.exp(), Err(Error::Operation(_))));
        assert!(parent
            .data_c64()
            .iter()
            .zip(&before)
            .all(|(actual, expected)| {
                actual.re.to_bits() == expected.re.to_bits()
                    && actual.im.to_bits() == expected.im.to_bits()
            }));
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &data));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_erased_sqrt_uses_a_cold_logical_copy(parent: &Tensor) {
        let expected = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap()
            .sqrt()
            .unwrap();
        let parent_space = Arc::clone(&parent.ordinary_body().space);
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let lazy = parent.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().sqrt().unwrap();
            assert_close(&actual, &expected);
            assert!(!actual.is_adjoint_view());
            assert!(actual
                .rule_authority_space()
                .provider_matches_context_allocation(&parent.rule_authority_space().context()));
            assert!(!Arc::ptr_eq(&actual.ordinary_body().data, &parent_data));
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.sqrt().unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_close(&call.join().unwrap(), &expected);
        }
        assert!(Arc::ptr_eq(&parent.ordinary_body().space, &parent_space));
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());

        let _ = lazy.coupled_data().unwrap();
        assert_eq!(lazy.adjoint_body_builds(), 1);
        assert!(lazy.has_cached_materialization());
    }

    #[test]
    fn erased_sqrt_dense_lazy_success_is_owned_provider_native_repeatable_and_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1 = Space::u1([(-1, 2), (2, 4)]);
        let f64_parent = Tensor::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 1) as f64
            } else {
                0.0
            }
        })
        .unwrap();
        assert_erased_sqrt_uses_a_cold_logical_copy(&f64_parent);

        let c64_parent = Tensor::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            if indices[0] == indices[1] {
                match indices[0] % 4 {
                    0 => Complex64::new(-4.0, 0.0),
                    1 => Complex64::new(-4.0, -0.0),
                    2 => Complex64::new(-4.0, 1.0e-300),
                    _ => Complex64::new(-4.0, -1.0e-300),
                }
            } else {
                Complex64::new(0.0, -0.0)
            }
        })
        .unwrap();
        let expected = c64_parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap()
            .sqrt()
            .unwrap();
        let lazy = c64_parent.adjoint().unwrap();
        let actual = lazy.sqrt().unwrap();
        assert!(actual
            .data_c64()
            .iter()
            .zip(expected.data_c64())
            .all(|(actual, expected)| {
                actual.re.to_bits() == expected.re.to_bits()
                    && actual.im.to_bits() == expected.im.to_bits()
            }));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
        assert_erased_sqrt_uses_a_cold_logical_copy(&c64_parent);

        let su2 = Space::su2([(0, 2), (1, 4)]).unwrap();
        let su2_parent = Tensor::from_block_fn(&runtime, [&su2], [&su2], |_, indices| {
            if indices[0] == indices[1] {
                Complex64::new(-4.0, (indices[0] as f64 - 1.0) / 10.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();
        assert_erased_sqrt_uses_a_cold_logical_copy(&su2_parent);
    }

    #[test]
    fn erased_sqrt_dense_lazy_failures_preserve_logical_order_and_stay_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(0, 3)]);

        let offdiag = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices == [0, 1] {
                1.0
            } else {
                0.0
            }
        })
        .unwrap();
        let lazy = offdiag.adjoint().unwrap();
        let error = lazy.sqrt().unwrap_err().to_string();
        assert!(error.contains("(1, 0)"), "{error}");
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());

        let mixed = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (1, 1) => -1.0,
                (0, 2) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
        let eager_error = mixed
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap()
            .sqrt()
            .unwrap_err()
            .to_string();
        let lazy = mixed.adjoint().unwrap();
        assert_eq!(lazy.sqrt().unwrap_err().to_string(), eager_error);
        assert!(eager_error.contains("negative"), "{eager_error}");
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());

        let negative = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                -1.0
            } else {
                0.0
            }
        })
        .unwrap();
        let eager_error = negative
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap()
            .sqrt()
            .unwrap_err()
            .to_string();
        let lazy = negative.adjoint().unwrap();
        for _ in 0..2 {
            assert_eq!(lazy.clone().sqrt().unwrap_err().to_string(), eager_error);
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_lazy_compact_svd_matches_eager(
        runtime: &Runtime,
        left: &Space,
        right: &Space,
        dtype: Dtype,
        seed: u64,
    ) {
        let parent = Tensor::rand_with_seed(runtime, dtype, [left], [right], seed).unwrap();
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
        let (actual_u, actual_s, actual_vh) = lazy.svd_compact().unwrap();
        let (expected_u, expected_s, expected_vh) = eager.svd_compact().unwrap();

        assert_close(&actual_u, &expected_u);
        assert_close(&actual_s, &expected_s);
        assert_close(&actual_vh, &expected_vh);
        assert_close(
            &actual_u
                .compose(&actual_s)
                .unwrap()
                .compose(&actual_vh)
                .unwrap(),
            &eager,
        );
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_lazy_full_svd_matches_eager(
        runtime: &Runtime,
        left: &Space,
        right: &Space,
        dtype: Dtype,
        seed: u64,
    ) {
        let parent = Tensor::rand_with_seed(runtime, dtype, [left], [right], seed).unwrap();
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
        let (actual_u, actual_s, actual_vh) = lazy.svd_full().unwrap();
        let (expected_u, expected_s, expected_vh) = eager.svd_full().unwrap();

        assert_close(&actual_u, &expected_u);
        assert_close(&actual_s, &expected_s);
        assert_close(&actual_vh, &expected_vh);
        assert_close(
            &actual_u
                .compose(&actual_s)
                .unwrap()
                .compose(&actual_vh)
                .unwrap(),
            &eager,
        );
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_lazy_truncated_svd_matches_eager(
        runtime: &Runtime,
        left: &Space,
        right: &Space,
        dtype: Dtype,
        seed: u64,
    ) {
        let parent = Tensor::rand_with_seed(runtime, dtype, [left], [right], seed).unwrap();
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
        let actual = lazy.svd_trunc(&Truncation::rank(4)).unwrap();
        let expected = eager.svd_trunc(&Truncation::rank(4)).unwrap();

        assert_close(&actual.u, &expected.u);
        assert_close(&actual.s, &expected.s);
        assert_close(&actual.vh, &expected.vh);
        assert_spectra_close(&actual.singular_values, &expected.singular_values);
        assert!((actual.error - expected.error).abs() < 1e-12);
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_metadata_and_materialization(space: Space, seed: u64) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], seed).unwrap();
        let adjoint = source.adjoint().unwrap();
        assert!(adjoint.is_adjoint_view());
        assert_eq!(adjoint.codomain_rank(), 1);
        assert_eq!(adjoint.domain_rank(), 2);
        assert_eq!(adjoint.rank(), 3);
        assert_eq!(adjoint.dtype(), source.dtype());
        assert_eq!(adjoint.placement(), source.placement());
        assert!(adjoint.runtime().same_runtime(source.runtime()));
        assert_eq!(adjoint.codomain_spaces(), source.domain_spaces());
        assert_eq!(adjoint.domain_spaces(), source.codomain_spaces());
        assert_eq!(adjoint.adjoint_body_builds(), 0);

        tenet_tensors::reset_global_operation_caches();
        assert_eq!(adjoint.space(0).unwrap(), source.domain_spaces()[0]);
        assert_eq!(adjoint.leg_dims().unwrap().len(), 3);
        assert_eq!(adjoint.adjoint_body_builds(), 0);

        let clone = adjoint.clone();
        let expected = clone.try_data_c64().unwrap().to_vec();
        assert_eq!(adjoint.adjoint_body_builds(), 1);
        let published_space = Arc::clone(&adjoint.materialized_body().unwrap().space);
        tenet_tensors::reset_global_operation_caches();
        assert_eq!(adjoint.try_data_c64().unwrap(), expected);
        assert_eq!(adjoint.adjoint_body_builds(), 1);
        assert!(Arc::ptr_eq(
            &published_space,
            &adjoint.materialized_body().unwrap().space
        ));

        let round_trip = adjoint.adjoint().unwrap();
        assert!(!round_trip.is_adjoint_view());
        assert!(Arc::ptr_eq(
            &round_trip.ordinary_body().data,
            &source.ordinary_body().data
        ));
        assert!(Arc::ptr_eq(
            &round_trip.ordinary_body().space,
            &source.ordinary_body().space
        ));
    }

    fn assert_elementary_adjoint_operations(space: Space, dtype: Dtype, seed: u64) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let parent_a =
            Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed).unwrap();
        let parent_b =
            Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed + 1).unwrap();
        let owned =
            Tensor::rand_with_seed(&runtime, dtype, [&space], [&space, &space], seed + 2).unwrap();
        let eager_a = parent_a.adjoint().unwrap().materialized_tensor().unwrap();
        let eager_b = parent_b.adjoint().unwrap().materialized_tensor().unwrap();
        let lazy_a = parent_a.adjoint().unwrap();
        let lazy_b = parent_b.adjoint().unwrap();

        assert_close(
            &lazy_a.scale(-1.25).unwrap(),
            &eager_a.scale(-1.25).unwrap(),
        );
        assert_close(
            &lazy_a.add(&lazy_b, 0.75, -1.5).unwrap(),
            &eager_a.add(&eager_b, 0.75, -1.5).unwrap(),
        );
        for (lhs, rhs, eager_lhs, eager_rhs) in [
            (&lazy_a, &owned, &eager_a, &owned),
            (&owned, &lazy_a, &owned, &eager_a),
        ] {
            assert_close(
                &lhs.add(rhs, 0.75, -1.5).unwrap(),
                &eager_lhs.add(eager_rhs, 0.75, -1.5).unwrap(),
            );
            assert_scalar_close(lhs.inner(rhs).unwrap(), eager_lhs.inner(eager_rhs).unwrap());
        }
        assert_scalar_close(
            lazy_a.inner(&lazy_b).unwrap(),
            eager_a.inner(&eager_b).unwrap(),
        );

        if dtype == Dtype::C64 {
            let alpha = Complex64::new(0.75, -0.5);
            let beta = Complex64::new(-1.5, 0.25);
            assert_close(
                &lazy_a.scale_c64(alpha).unwrap(),
                &eager_a.scale_c64(alpha).unwrap(),
            );
            assert_close(
                &lazy_a.add_c64(&lazy_b, alpha, beta).unwrap(),
                &eager_a.add_c64(&eager_b, alpha, beta).unwrap(),
            );
            assert_close(
                &lazy_a.add_c64(&owned, alpha, beta).unwrap(),
                &eager_a.add_c64(&owned, alpha, beta).unwrap(),
            );
            assert_close(
                &owned.add_c64(&lazy_a, alpha, beta).unwrap(),
                &owned.add_c64(&eager_a, alpha, beta).unwrap(),
            );
        }
        assert!(!lazy_a.has_cached_materialization());
        assert!(!lazy_b.has_cached_materialization());
    }

    #[test]
    fn elementary_adjoint_operations_match_eager_rule_oracles_without_materializing_inputs() {
        let spaces = [
            Space::u1([(-1, 2), (0, 1), (1, 3)]),
            Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
            Space::fz2([(0, 2), (1, 3)]).unwrap(),
            Space::fz2_u1_su2([((0, 0, 0), 2), ((1, -1, 1), 2), ((1, 1, 1), 1)]).unwrap(),
        ];
        for (space_index, space) in spaces.into_iter().enumerate() {
            for (dtype_index, dtype) in [Dtype::F64, Dtype::C64].into_iter().enumerate() {
                assert_elementary_adjoint_operations(
                    space.clone(),
                    dtype,
                    666_000 + 10 * space_index as u64 + dtype_index as u64,
                );
            }
        }
    }

    #[test]
    fn compact_diagonal_and_lazy_dense_elementary_operations_stay_compact_aware() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-1, 2), (0, 3), (1, 1)]);
        for (dtype, seed) in [(Dtype::F64, 666_101), (Dtype::C64, 666_102)] {
            let parent = Tensor::rand_with_seed(&runtime, dtype, [&space], [&space], seed).unwrap();
            let diagonal = parent.svd_compact().unwrap().1;
            let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
            let lazy = parent.adjoint().unwrap();

            assert_close(
                &lazy.add(&diagonal, 0.75, -1.25).unwrap(),
                &eager.add(&diagonal, 0.75, -1.25).unwrap(),
            );
            assert_close(
                &diagonal.add(&lazy, 0.75, -1.25).unwrap(),
                &diagonal.add(&eager, 0.75, -1.25).unwrap(),
            );
            assert_scalar_close(
                lazy.inner(&diagonal).unwrap(),
                eager.inner(&diagonal).unwrap(),
            );
            assert_scalar_close(
                diagonal.inner(&lazy).unwrap(),
                diagonal.inner(&eager).unwrap(),
            );
            if dtype == Dtype::C64 {
                let alpha = Complex64::new(0.5, -0.25);
                let beta = Complex64::new(-1.0, 0.75);
                assert_close(
                    &lazy.add_c64(&diagonal, alpha, beta).unwrap(),
                    &eager.add_c64(&diagonal, alpha, beta).unwrap(),
                );
                assert_close(
                    &diagonal.add_c64(&lazy, alpha, beta).unwrap(),
                    &diagonal.add_c64(&eager, alpha, beta).unwrap(),
                );
            }
            assert!(!lazy.has_cached_materialization());
            assert!(!diagonal.has_cached_materialization());
        }
    }

    #[test]
    fn elementary_adjoint_errors_precede_materialization_and_result_publication() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1 = Space::u1([(0, 2), (1, 1)]);
        let other_u1 = Space::u1([(0, 1), (1, 2)]);
        let z2 = Space::z2([(0, 2), (1, 1)]);
        let lazy = Tensor::rand_with_seed(&runtime, Dtype::C64, [&u1], [&u1], 666_201)
            .unwrap()
            .adjoint()
            .unwrap();
        let cases = [
            (
                Tensor::rand_with_seed(&runtime, Dtype::C64, [&z2], [&z2], 666_202).unwrap(),
                Error::RuleMismatch,
            ),
            (
                Tensor::rand_with_seed(&other_runtime, Dtype::C64, [&u1], [&u1], 666_203).unwrap(),
                Error::RuntimeMismatch,
            ),
            (
                Tensor::rand_with_seed(&runtime, Dtype::F64, [&u1], [&u1], 666_204).unwrap(),
                Error::DtypeMismatch,
            ),
            (
                Tensor::rand_with_seed(&runtime, Dtype::C64, [&other_u1], [&other_u1], 666_205)
                    .unwrap(),
                Error::InvalidArgument(
                    "tensors live on different spaces or block layouts".to_string(),
                ),
            ),
        ];
        for (other, expected) in cases {
            assert_eq!(lazy.add(&other, 1.0, 1.0).unwrap_err(), expected);
            assert_eq!(
                lazy.add_c64(&other, Complex64::new(1.0, 0.5), Complex64::new(-0.5, 0.25),)
                    .unwrap_err(),
                expected
            );
            assert_eq!(lazy.inner(&other).unwrap_err(), expected);
            assert!(!lazy.has_cached_materialization());
        }

        let simultaneous =
            Tensor::rand_with_seed(&other_runtime, Dtype::F64, [&z2], [&z2], 666_206).unwrap();
        assert_eq!(
            lazy.add(&simultaneous, 1.0, 1.0).unwrap_err(),
            Error::RuleMismatch
        );

        let f64_lazy = Tensor::rand_with_seed(&runtime, Dtype::F64, [&u1], [&u1], 666_207)
            .unwrap()
            .adjoint()
            .unwrap();
        assert_eq!(
            f64_lazy.scale_c64(Complex64::new(1.0, 0.5)).unwrap_err(),
            Error::DtypeMismatch
        );
        let f64_wrong_space =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&other_u1], [&other_u1], 666_208)
                .unwrap();
        assert_eq!(
            f64_lazy
                .add_c64(
                    &f64_wrong_space,
                    Complex64::new(1.0, 0.5),
                    Complex64::new(-0.5, 0.25),
                )
                .unwrap_err(),
            Error::DtypeMismatch
        );
        assert!(!f64_lazy.has_cached_materialization());
    }

    #[test]
    fn metadata_and_shared_materialization_cover_supported_rules() {
        // What: an adjoint view swaps only logical orientation until one owned consumer reads it.
        assert_metadata_and_materialization(Space::u1([(-1, 1), (0, 2), (1, 1)]), 261_001);
        assert_metadata_and_materialization(Space::fz2([(0, 2), (1, 2)]).unwrap(), 261_002);
        assert_metadata_and_materialization(Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap(), 261_003);
        assert_metadata_and_materialization(
            Space::product([((-1, 0), 1), ((0, 1), 2), ((1, 0), 1)]).unwrap(),
            261_004,
        );
    }

    #[test]
    fn external_axis_duality_reads_the_parent_orientation() {
        // What: direct and lazy-adjoint leg duality matches the materialized
        // logical space, preserves the invalid-axis error, and builds no adjoint layout.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-1, 1), (0, 2), (2, 1)]);
        let dual = space.dual();
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &dual], [&space], 477_001)
                .unwrap();
        let lazy = source.adjoint().unwrap();
        let oracle = source.adjoint().unwrap().materialized_tensor().unwrap();

        for axis in 0..source.rank() {
            assert_eq!(
                source.external_axis_is_dual(axis).unwrap(),
                source
                    .ordinary_body()
                    .space
                    .homspace()
                    .external_axis_is_dual(axis)
                    .unwrap()
            );
            assert_eq!(
                lazy.external_axis_is_dual(axis).unwrap(),
                oracle
                    .ordinary_body()
                    .space
                    .homspace()
                    .external_axis_is_dual(axis)
                    .unwrap()
            );
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert_eq!(
            lazy.external_axis_is_dual(lazy.rank()).unwrap_err(),
            Error::InvalidArgument(format!(
                "axis {} is out of range for rank {}",
                lazy.rank(),
                lazy.rank()
            ))
        );
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }

    #[test]
    fn norm_inf_reads_the_adjoint_parent() {
        // What: an adjoint view preserves the entrywise maximum without building its layout.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let left = Space::u1([(-2, 1), (-1, 2), (0, 1), (1, 2)]);
        let right = Space::u1([(-1, 1), (0, 3), (2, 1)]);

        for (dtype, seed) in [(Dtype::F64, 482_001), (Dtype::C64, 482_002)] {
            let source = Tensor::rand_with_seed(&runtime, dtype, [&left], [&right], seed).unwrap();
            let lazy = source.adjoint().unwrap();

            assert_eq!(lazy.norm_inf().unwrap(), source.norm_inf().unwrap());
            assert_eq!(lazy.adjoint_body_builds(), 0);
        }
    }

    #[test]
    fn svd_vals_reads_parent_without_adjoint_materialization() {
        // What: values-only SVD preserves every host rule's sector spectrum
        // across cold, repeated, cloned, and concurrent lazy-adjoint reads.
        let runtime = Runtime::builder().dense_threads(4).build().unwrap();
        let fixtures = [
            (Space::u1([(-1, 1), (0, 3)]), Space::u1([(-1, 2), (0, 1)])),
            (
                Space::su2([(0, 2), (1, 1)]).unwrap(),
                Space::su2([(0, 1), (1, 3)]).unwrap(),
            ),
            (
                Space::product([((-1, 0), 1), ((0, 1), 3)]).unwrap(),
                Space::product([((-1, 0), 2), ((0, 1), 1)]).unwrap(),
            ),
        ];

        for (fixture, (left, right)) in fixtures.into_iter().enumerate() {
            for (dtype, lane) in [(Dtype::F64, 0), (Dtype::C64, 1)] {
                let parent = Tensor::rand_with_seed(
                    &runtime,
                    dtype,
                    [&left],
                    [&right],
                    603_100 + 2 * fixture as u64 + lane,
                )
                .unwrap();
                let expected = parent.svd_vals().unwrap();
                let lazy = parent.adjoint().unwrap();

                assert_eq!(lazy.svd_vals().unwrap(), expected);
                assert_eq!(lazy.svd_vals().unwrap(), expected);
                assert_eq!(lazy.clone().svd_vals().unwrap(), expected);
                assert_eq!(lazy.adjoint_body_builds(), 0);
                assert!(!lazy.has_cached_materialization());
            }
        }

        let space = Space::u1([(-1, 2), (0, 3), (1, 1)]);
        let parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 603_200).unwrap();
        let expected = parent.svd_vals().unwrap();
        let lazy = parent.adjoint().unwrap();
        std::thread::scope(|scope| {
            let calls: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| lazy.svd_vals().unwrap()))
                .collect();
            for call in calls {
                assert_eq!(call.join().unwrap(), expected);
            }
        });
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn asymmetric_u1_consumers_match_an_eager_adjoint_oracle() {
        // What: transform, trace, and rectangular SVD consume one coherent adjoint body.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let left = Space::u1([(-2, 1), (-1, 2), (0, 1), (1, 2)]);
        let right = Space::u1([(-1, 1), (0, 3), (2, 1)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&left], [&right], 261_101).unwrap();
        let lazy = source.adjoint().unwrap();
        let eager = source.adjoint().unwrap().materialized_tensor().unwrap();

        assert_close(
            &lazy.permute(&[0], &[1]).unwrap(),
            &eager.permute(&[0], &[1]).unwrap(),
        );
        let (lazy_u, lazy_s, lazy_vh) = lazy.svd_compact().unwrap();
        let (eager_u, eager_s, eager_vh) = eager.svd_compact().unwrap();
        assert_close(&lazy_u, &eager_u);
        assert_close(&lazy_s, &eager_s);
        assert_close(&lazy_vh, &eager_vh);

        let endomorphism =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&left], [&left], 261_102).unwrap();
        let trace = endomorphism.tr().unwrap().to_c64();
        let adjoint_trace = endomorphism.adjoint().unwrap().tr().unwrap().to_c64();
        assert!((adjoint_trace - trace.conj()).norm() < 1e-12);
    }

    #[test]
    fn compact_svd_reads_multiplicity_free_parent_without_materialization() {
        // What: asymmetric real/complex U(1), SU(2), and product factors match
        // the owned-adjoint gauge oracle while the lazy input cache stays empty.
        let runtime = Runtime::builder().dense_threads(4).build().unwrap();
        let fixtures = [
            (
                Space::u1([(-1, 1), (0, 3), (1, 2)]),
                Space::u1([(-1, 2), (0, 1), (1, 3)]),
            ),
            (
                Space::su2([(0, 1), (1, 3), (2, 2)]).unwrap(),
                Space::su2([(0, 2), (1, 1), (2, 3)]).unwrap(),
            ),
            (
                Space::product([((-1, 0), 1), ((0, 1), 3), ((1, 0), 2)]).unwrap(),
                Space::product([((-1, 0), 2), ((0, 1), 1), ((1, 0), 3)]).unwrap(),
            ),
        ];

        for (fixture, (left, right)) in fixtures.into_iter().enumerate() {
            for (dtype, lane) in [(Dtype::F64, 0), (Dtype::C64, 1)] {
                assert_lazy_compact_svd_matches_eager(
                    &runtime,
                    &left,
                    &right,
                    dtype,
                    603_300 + 2 * fixture as u64 + lane,
                );
            }
        }

        let space = Space::u1([(-1, 2), (0, 3), (1, 1)]);
        let parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 603_400).unwrap();
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
        std::thread::scope(|scope| {
            let calls: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| lazy.svd_compact().unwrap()))
                .collect();
            for call in calls {
                let (u, s, vh) = call.join().unwrap();
                assert_close(&u.compose(&s).unwrap().compose(&vh).unwrap(), &eager);
            }
        });
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn full_svd_reads_multiplicity_free_parent_without_materialization() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let fixtures = [
            (
                Space::u1([(-1, 1), (0, 3), (1, 2)]),
                Space::u1([(-1, 2), (0, 1), (1, 3)]),
            ),
            (
                Space::su2([(0, 1), (1, 3), (2, 2)]).unwrap(),
                Space::su2([(0, 2), (1, 1), (2, 3)]).unwrap(),
            ),
        ];

        for (fixture, (left, right)) in fixtures.into_iter().enumerate() {
            for (dtype, lane) in [(Dtype::F64, 0), (Dtype::C64, 1)] {
                assert_lazy_full_svd_matches_eager(
                    &runtime,
                    &left,
                    &right,
                    dtype,
                    603_800 + 2 * fixture as u64 + lane,
                );
            }
        }
    }

    #[test]
    fn compact_svd_whole_degenerate_cluster_uses_semantic_oracle() {
        // What: a fully retained repeated-singular-value cluster is checked by
        // reconstruction and isometry rather than an arbitrary raw basis.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-1, 2), (0, 3), (1, 2)]);
        for dtype in [Dtype::F64, Dtype::C64] {
            let parent = Tensor::id(&runtime, dtype, [&space]).unwrap();
            let lazy = parent.adjoint().unwrap();
            let (u, s, vh) = lazy.svd_compact().unwrap();

            assert_close(&u.compose(&s).unwrap().compose(&vh).unwrap(), &parent);
            assert!(u.is_isometric(1e-12).unwrap());
            assert!(vh.adjoint().unwrap().is_isometric(1e-12).unwrap());
            assert_eq!(lazy.adjoint_body_builds(), 0);
            assert!(!lazy.has_cached_materialization());
        }
    }

    #[test]
    fn truncated_svd_reads_multiplicity_free_parent_without_materialization() {
        // What: truncation keeps the compact-SVD gauge, policy, and error while
        // avoiding an owned adjoint input for every multiplicity-free rule.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let fixtures = [
            (
                Space::u1([(-1, 1), (0, 3), (1, 2)]),
                Space::u1([(-1, 2), (0, 1), (1, 3)]),
            ),
            (
                Space::su2([(0, 1), (1, 3), (2, 2)]).unwrap(),
                Space::su2([(0, 2), (1, 1), (2, 3)]).unwrap(),
            ),
            (
                Space::product([((-1, 0), 1), ((0, 1), 3), ((1, 0), 2)]).unwrap(),
                Space::product([((-1, 0), 2), ((0, 1), 1), ((1, 0), 3)]).unwrap(),
            ),
        ];

        for (fixture, (left, right)) in fixtures.into_iter().enumerate() {
            for (dtype, lane) in [(Dtype::F64, 0), (Dtype::C64, 1)] {
                assert_lazy_truncated_svd_matches_eager(
                    &runtime,
                    &left,
                    &right,
                    dtype,
                    603_600 + 2 * fixture as u64 + lane,
                );
            }
        }

        let space = Space::u1([(-1, 2), (0, 3), (1, 1)]);
        let parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 603_699).unwrap();
        let lazy = parent.adjoint().unwrap();
        let expected = parent
            .adjoint()
            .unwrap()
            .materialized_tensor()
            .unwrap()
            .svd_trunc(&Truncation::rank(4))
            .unwrap();
        std::thread::scope(|scope| {
            let calls: Vec<_> = (0..4)
                .map(|_| {
                    let lazy = lazy.clone();
                    scope.spawn(move || lazy.svd_trunc(&Truncation::rank(4)).unwrap())
                })
                .collect();
            for call in calls {
                let actual = call.join().unwrap();
                assert_close(&actual.u, &expected.u);
                assert_close(&actual.s, &expected.s);
                assert_close(&actual.vh, &expected.vh);
                assert_spectra_close(&actual.singular_values, &expected.singular_values);
                assert!((actual.error - expected.error).abs() < 1e-12);
            }
        });
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn truncated_svd_split_degenerate_cluster_uses_semantic_oracle() {
        // What: raw singular vectors are not compared when rank truncation
        // splits an exactly degenerate cluster; the retained subspace is
        // checked by isometry and its approximation error.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(0, 4)]);
        let parent = Tensor::from_block_fn(&runtime, [&space], [&space], |_, indices| {
            if indices[0] == indices[1] {
                Complex64::new([5.0, 2.0, 2.0, 0.5][indices[0]], 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();
        let lazy = parent.adjoint().unwrap();
        let output = lazy.svd_trunc(&Truncation::rank(2)).unwrap();
        let approximation = output
            .u
            .compose(&output.s)
            .unwrap()
            .compose(&output.vh)
            .unwrap();

        assert_eq!(output.singular_values.len(), 1);
        assert_eq!(output.singular_values[0].values, vec![5.0, 2.0]);
        assert!(output.u.is_isometric(1e-12).unwrap());
        assert!(output.vh.adjoint().unwrap().is_isometric(1e-12).unwrap());
        assert!(
            (parent
                .add(&approximation, 1.0, -1.0)
                .unwrap()
                .norm()
                .unwrap()
                - output.error)
                .abs()
                < 1e-12
        );
        assert!((output.error - (2.0_f64.powi(2) + 0.5_f64.powi(2)).sqrt()).abs() < 1e-12);

        // The retained sigma=2 directions may rotate within span{e1,e2}, but
        // they must not leak into the complete sigma=5 or sigma=0.5 clusters.
        let u = output.u.try_data_c64().unwrap();
        let vh = output.vh.try_data_c64().unwrap();
        for value in [u[4], u[7], vh[1], vh[7]] {
            assert!(value.norm() < 1e-12);
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn truncated_svd_rejects_foreign_truncspace_without_materialization() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-1, 2), (0, 3), (1, 1)]);
        let foreign = Space::su2([(0, 2), (1, 1)]).unwrap();
        let parent =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 603_701).unwrap();
        let before = parent.data().to_vec();
        let lazy = parent.adjoint().unwrap();

        assert!(matches!(
            lazy.svd_trunc(&Truncation::space(foreign.truncspace())),
            Err(Error::Operation(_))
        ));
        assert_eq!(parent.data(), before);
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_adjoint_qr_lq_matches_eager_oracle(
        left: Space,
        right: Space,
        dtype: Dtype,
        seed: u64,
    ) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let source = Tensor::rand_with_seed(&runtime, dtype, [&left], [&right], seed).unwrap();
        assert_adjoint_qr_lq_tensor(&source);
    }

    fn assert_adjoint_qr_lq_tensor(source: &Tensor) {
        let lazy = source.adjoint().unwrap();
        let eager = source.adjoint().unwrap().materialized_tensor().unwrap();

        let (lazy_q, lazy_r) = lazy.qr_compact().unwrap();
        assert!(!lazy_q.is_adjoint_view() && !lazy_r.is_adjoint_view());
        let (eager_q, eager_r) = eager.qr_compact().unwrap();
        assert_close(&lazy_q, &eager_q);
        assert_close(&lazy_r, &eager_r);
        assert_close(&lazy_q.compose(&lazy_r).unwrap(), &eager);
        assert!(lazy_q.is_isometric(1e-12).unwrap());

        let (lazy_l, lazy_q) = lazy.lq_compact().unwrap();
        assert!(!lazy_l.is_adjoint_view() && !lazy_q.is_adjoint_view());
        let (eager_l, eager_q) = eager.lq_compact().unwrap();
        assert_close(&lazy_l, &eager_l);
        assert_close(&lazy_q, &eager_q);
        assert_close(&lazy_l.compose(&lazy_q).unwrap(), &eager);
        assert!(lazy_q.adjoint().unwrap().is_isometric(1e-12).unwrap());

        let (lazy_q, lazy_r) = lazy.qr_full().unwrap();
        assert!(!lazy_q.is_adjoint_view() && !lazy_r.is_adjoint_view());
        assert_close(&lazy_q.compose(&lazy_r).unwrap(), &eager);
        assert!(lazy_q.is_isometric(1e-12).unwrap());

        let (lazy_l, lazy_q) = lazy.lq_full().unwrap();
        assert!(!lazy_l.is_adjoint_view() && !lazy_q.is_adjoint_view());
        assert_close(&lazy_l.compose(&lazy_q).unwrap(), &eager);
        assert!(lazy_q.adjoint().unwrap().is_isometric(1e-12).unwrap());

        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn qr_lq_adjoint_dispatch_preserves_semantics_without_publishing_input_materialization() {
        // What: rectangular adjoint QR uses an operation-local logical copy
        // while LQ uses the parent QR, preserving compact gauge, full semantics,
        // and cold lazy-input storage.
        assert_adjoint_qr_lq_matches_eager_oracle(
            Space::u1([(-2, 2), (0, 3), (1, 2)]),
            Space::u1([(-2, 1), (0, 2)]),
            Dtype::F64,
            261_201,
        );
        assert_adjoint_qr_lq_matches_eager_oracle(
            Space::u1([(-2, 1), (0, 2)]),
            Space::u1([(-2, 2), (0, 3), (1, 2)]),
            Dtype::C64,
            261_202,
        );
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let su2 = Space::su2([(0, 2), (1, 2)]).unwrap();
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&su2, &su2], [&su2], 261_203).unwrap();
        assert!(source.ordinary_body().space.structure().block_count() > 1);
        assert_adjoint_qr_lq_tensor(&source);
        assert_adjoint_qr_lq_matches_eager_oracle(
            Space::product([((-1, 0), 1), ((0, 1), 2), ((1, 0), 1)]).unwrap(),
            Space::product([((-1, 0), 2), ((0, 1), 3), ((1, 0), 2)]).unwrap(),
            Dtype::C64,
            261_204,
        );
    }

    fn assert_adjoint_null_spaces(source: &Tensor) {
        let lazy = source.adjoint().unwrap();
        let eager = source.adjoint().unwrap().materialized_tensor().unwrap();
        let source_context = source.rule_authority_space().context();
        for (actual, expected, left) in [
            (lazy.left_null().unwrap(), eager.left_null().unwrap(), true),
            (
                lazy.right_null().unwrap(),
                eager.right_null().unwrap(),
                false,
            ),
        ] {
            assert!(!actual.is_adjoint_view());
            assert!(actual
                .rule_authority_space()
                .provider_matches_context_allocation(&source_context));
            assert_eq!(actual.codomain_spaces(), expected.codomain_spaces());
            assert_eq!(actual.domain_spaces(), expected.domain_spaces());
            let actual_projector = if left {
                actual.compose(&actual.adjoint().unwrap()).unwrap()
            } else {
                actual.adjoint().unwrap().compose(&actual).unwrap()
            };
            let expected_projector = if left {
                expected.compose(&expected.adjoint().unwrap()).unwrap()
            } else {
                expected.adjoint().unwrap().compose(&expected).unwrap()
            };
            assert_close(&actual_projector, &expected_projector);
            let residual = if left {
                actual.adjoint().unwrap().compose(&eager).unwrap()
            } else {
                eager.compose(&actual.adjoint().unwrap()).unwrap()
            };
            assert!(residual.norm().unwrap() < 1e-10 * (1.0 + eager.norm().unwrap()));
            assert!(if left {
                actual.is_isometric(1e-11).unwrap()
            } else {
                actual.adjoint().unwrap().is_isometric(1e-11).unwrap()
            });
            let _ = actual.coupled_data().unwrap();
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    fn assert_erased_polar_factors(
        parent: &Tensor,
        target: &Tensor,
        factors: &(Tensor, Tensor),
        left: bool,
    ) {
        let reconstructed = factors.0.compose(&factors.1).unwrap();
        assert_close(&reconstructed, target);
        let (positive, isometry) = if left {
            (&factors.1, &factors.0)
        } else {
            (&factors.0, &factors.1)
        };
        assert!(if left {
            isometry.is_isometric(1e-11).unwrap()
        } else {
            isometry.adjoint().unwrap().is_isometric(1e-11).unwrap()
        });
        assert!(positive.is_hermitian(1e-11).unwrap());
        assert!(positive
            .eigh_vals()
            .unwrap()
            .iter()
            .all(|entry| entry.values.iter().all(|&value| value >= -1e-11)));
        for factor in [&factors.0, &factors.1] {
            assert!(!factor.is_adjoint_view());
            assert!(factor
                .rule_authority_space()
                .provider_matches_context_allocation(&parent.rule_authority_space().context()));
            let _ = factor.coupled_data().unwrap();
        }
    }

    fn assert_erased_inverse_redirect(parent: &Tensor) {
        let eager = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected = eager.inv().unwrap();
        let parent_space = Arc::clone(&parent.ordinary_body().space);
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let lazy = parent.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().inv().unwrap();
            assert_close(&actual, &expected);
            assert_close(
                &eager.compose(&actual).unwrap(),
                &Tensor::id(&parent.rt, parent.dtype(), eager.codomain_spaces().iter()).unwrap(),
            );
            assert_close(
                &actual.compose(&eager).unwrap(),
                &Tensor::id(&parent.rt, parent.dtype(), eager.domain_spaces().iter()).unwrap(),
            );
            assert!(!actual.is_adjoint_view());
            assert!(actual
                .rule_authority_space()
                .provider_matches_context_allocation(&parent.rule_authority_space().context()));
            assert!(!Arc::ptr_eq(&actual.ordinary_body().data, &parent_data));
            let _ = actual.coupled_data().unwrap();
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.inv().unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_close(&call.join().unwrap(), &expected);
        }
        assert!(Arc::ptr_eq(&parent.ordinary_body().space, &parent_space));
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn erased_null_spaces_redirect_through_the_parent_without_materializing_the_adjoint() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let matched_left = Space::u1([(0, 3)]);
        let matched_right = Space::u1([(0, 2)]);
        let unmatched_left = Space::u1([(0, 3), (1, 2)]);
        let unmatched_right = Space::u1([(0, 2), (1, 2)]);
        let disjoint_left = Space::u1([(1, 2)]);
        let disjoint_right = Space::u1([(0, 3)]);
        for (fixture, (left, right)) in [
            (&matched_left, &matched_right),
            (&unmatched_left, &matched_right),
            (&matched_left, &unmatched_right),
            (&disjoint_left, &disjoint_right),
        ]
        .into_iter()
        .enumerate()
        {
            for (dtype, lane) in [(Dtype::F64, 0), (Dtype::C64, 1)] {
                let source = Tensor::rand_with_seed(
                    &runtime,
                    dtype,
                    [left],
                    [right],
                    703_000 + 2 * fixture as u64 + lane,
                )
                .unwrap();
                assert_adjoint_null_spaces(&source);
            }
        }

        let rank_deficient =
            Tensor::from_block_fn(&runtime, [&matched_left], [&matched_left], |_, indices| {
                Complex64::new(
                    (indices[0] + indices[1] + 1) as f64,
                    (indices[0] + indices[1] + 1) as f64 / 7.0,
                )
            })
            .unwrap();
        assert_adjoint_null_spaces(&rank_deficient);

        let su2 = Space::su2([(0, 2), (1, 1)]).unwrap();
        for (dtype, seed) in [(Dtype::F64, 703_010), (Dtype::C64, 703_011)] {
            let source =
                Tensor::rand_with_seed(&runtime, dtype, [&su2, &su2], [&su2], seed).unwrap();
            assert!(source.ordinary_body().space.structure().block_count() > 1);
            assert_adjoint_null_spaces(&source);
        }
    }

    #[test]
    fn erased_inverse_redirect_is_owned_provider_native_repeatable_and_cold() {
        // What: U(1) complex blocks and a genuine SU(2) multitree redirect
        // through the parent across repeated, cloned, and concurrent calls,
        // preserving provider authority and cold lazy storage.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1 = Space::u1([(-1, 2), (0, 3), (2, 1)]);
        let parent = Tensor::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            let re = if indices[0] == indices[1] {
                20.0 + indices[0] as f64
            } else {
                (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
            };
            Complex64::new(re, (indices[0] + indices[1] + 1) as f64 / 200.0)
        })
        .unwrap();
        let identity = Tensor::id(&runtime, Dtype::C64, [&u1]).unwrap();
        let parent = parent.add(&identity, 1.0, 100.0).unwrap();
        assert_erased_inverse_redirect(&parent);

        let wide = Space::u1([(0, 4)]);
        let narrow = Space::u1([(0, 2)]);
        let parent = Tensor::from_block_fn(&runtime, [&wide], [&narrow, &narrow], |_, indices| {
            let column = 2 * indices[1] + indices[2];
            if indices[0] == column {
                10.0 + indices[0] as f64
            } else {
                (indices[0] + column + 1) as f64 / 100.0
            }
        })
        .unwrap();
        assert_ne!(parent.codomain_rank(), parent.domain_rank());
        assert_erased_inverse_redirect(&parent);

        let half = Space::su2([(1, 1)]).unwrap();
        let parent = Tensor::from_block_fn(
            &runtime,
            [&half, &half, &half],
            [&half, &half, &half],
            |_, indices| {
                if indices[..3] == indices[3..] {
                    20.0 + indices.iter().sum::<usize>() as f64
                } else {
                    (indices.iter().sum::<usize>() + 1) as f64 / 100.0
                }
            },
        )
        .unwrap();
        let identity = Tensor::id(&runtime, Dtype::F64, [&half, &half, &half]).unwrap();
        let parent = parent.add(&identity, 1.0, 100.0).unwrap();
        assert!(parent.ordinary_body().space.structure().block_count() > 1);
        assert_erased_inverse_redirect(&parent);
    }

    #[test]
    fn erased_inverse_redirect_failure_and_powi_stay_cold() {
        // What: negative powers inherit the redirect; singular and unsupported
        // inputs preserve parent Arc/bytes and reject with a cold receiver.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1 = Space::u1([(0, 3)]);
        let parent = Tensor::from_block_fn(&runtime, [&u1], [&u1], |_, indices| {
            if indices[0] == indices[1] {
                4.0 + indices[0] as f64
            } else {
                (indices[0] + indices[1] + 1) as f64 / 20.0
            }
        })
        .unwrap();
        let lazy = parent.adjoint().unwrap();
        let eager = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        assert_close(&lazy.powi(-3).unwrap(), &eager.powi(-3).unwrap());
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());

        let singular = parent.scale(0.0).unwrap();
        let space = Arc::clone(&singular.ordinary_body().space);
        let data = Arc::clone(&singular.ordinary_body().data);
        let before = singular.data().to_vec();
        let cold = singular.adjoint().unwrap();
        assert!(matches!(cold.inv(), Err(Error::Operation(_))));
        assert_eq!(singular.data(), before);
        assert!(Arc::ptr_eq(&singular.ordinary_body().space, &space));
        assert!(Arc::ptr_eq(&singular.ordinary_body().data, &data));
        assert_eq!(cold.adjoint_body_builds(), 0);
        assert!(!cold.has_cached_materialization());

        // The charge-zero block solves before the later singular block fails:
        // no partially produced inverse or receiver materialization may leak.
        let leg = Space::u1([(0, 1), (1, 2)]);
        let late = Tensor::from_block_fn(&runtime, [&leg], [&leg], |key, indices| match key {
            BlockKey::FusionTree(key)
                if key.codomain_uncoupled()[0].id() == 0 && indices[0] == indices[1] =>
            {
                2.0
            }
            _ => 0.0,
        })
        .unwrap();
        let before = late.data().to_vec();
        let data = Arc::clone(&late.ordinary_body().data);
        let cold = late.adjoint().unwrap();
        assert!(matches!(cold.inv(), Err(Error::Operation(_))));
        assert_eq!(late.data(), before);
        assert!(Arc::ptr_eq(&late.ordinary_body().data, &data));
        assert_eq!(cold.adjoint_body_builds(), 0);
        assert!(!cold.has_cached_materialization());
    }

    #[test]
    fn erased_pinv_rejects_invalid_rcond_before_materializing_a_lazy_receiver() {
        // What: argument validation precedes storage, rule, device, provider,
        // and lazy dispatch. None of the invalid values may publish the view.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(-1, 1), (0, 2), (2, 1)]);
        let parent = Tensor::rand_with_seed(&runtime, Dtype::C64, [&leg], [&leg], 711_001).unwrap();
        let lazy = parent.adjoint().unwrap();

        for rcond in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(lazy.pinv(rcond), Err(Error::InvalidArgument(_))));
            assert_eq!(lazy.adjoint_body_builds(), 0);
            assert!(!lazy.has_cached_materialization());
        }
    }

    fn assert_erased_pinv_redirect(parent: &Tensor, rcond: f64, exact_original: bool) {
        let eager = parent
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let expected = eager.pinv(rcond).unwrap();
        let parent_space = Arc::clone(&parent.ordinary_body().space);
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let lazy = parent.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().pinv(rcond).unwrap();
            assert_close(&actual, &expected);
            assert_close(
                &actual.compose(&eager).unwrap().compose(&actual).unwrap(),
                &actual,
            );
            assert!(eager.compose(&actual).unwrap().is_hermitian(1e-9).unwrap());
            assert!(actual.compose(&eager).unwrap().is_hermitian(1e-9).unwrap());
            if exact_original {
                assert_close(
                    &eager.compose(&actual).unwrap().compose(&eager).unwrap(),
                    &eager,
                );
            }
            assert!(!actual.is_adjoint_view());
            assert!(actual
                .rule_authority_space()
                .provider_matches_context_allocation(&parent.rule_authority_space().context()));
            assert!(!Arc::ptr_eq(&actual.ordinary_body().data, &parent_data));
            let _ = actual.coupled_data().unwrap();
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.pinv(rcond).unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_close(&call.join().unwrap(), &expected);
        }
        assert!(Arc::ptr_eq(&parent.ordinary_body().space, &parent_space));
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn erased_pinv_redirect_preserves_semantics_ownership_and_cold_concurrency() {
        // What: real/complex U(1), rectangular and empty support, plus a
        // genuine SU(2) multitree match a materialized oracle and projector
        // laws without publishing the lazy receiver.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(-1, 2), (0, 3)]);
        let full = Tensor::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            let re = if indices[0] == indices[1] {
                30.0 + indices[0] as f64
            } else {
                (indices[0] + indices[1] + 1) as f64 / 100.0
            };
            Complex64::new(re, (indices[0] + 2 * indices[1] + 1) as f64 / 200.0)
        })
        .unwrap();
        assert_erased_pinv_redirect(&full, 1e-12, true);

        let left = Space::u1([(0, 3), (1, 2)]);
        let right = Space::u1([(0, 2)]);
        let rectangular =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&left], [&right], 711_010).unwrap();
        assert_erased_pinv_redirect(&rectangular, 1e-10, false);
        let disjoint = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&Space::u1([(1, 2)])],
            [&right],
            711_011,
        )
        .unwrap();
        assert_erased_pinv_redirect(&disjoint, 1e-10, false);

        let half = Space::su2([(1, 1)]).unwrap();
        let su2 = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&half, &half, &half],
            [&half],
            711_012,
        )
        .unwrap();
        assert!(su2.ordinary_body().space.structure().block_count() > 1);
        assert_erased_pinv_redirect(&su2, 1e-10, false);
    }

    #[test]
    fn erased_c64_multitree_polar_redirect_returns_owned_psd_factors_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let half = Space::su2([(1, 1)]).unwrap();
        assert_eq!(
            Space::fuse_all(&[&half, &half, &half])
                .unwrap()
                .degeneracy(SectorLabel::SU2 { twice_spin: 1 }),
            Some(2)
        );
        let parent = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&half, &half, &half],
            [&half, &half, &half],
            706,
        )
        .unwrap();
        assert!(parent
            .data_c64()
            .iter()
            .any(|value| value.im.abs() > f64::EPSILON));
        assert!(parent.ordinary_body().space.structure().block_count() > 1);
        let lazy = parent.adjoint().unwrap();
        let eager = lazy.materialized_tensor_uncached().unwrap();

        for left in [true, false] {
            let factors = if left {
                lazy.left_polar().unwrap()
            } else {
                lazy.right_polar().unwrap()
            };
            assert_erased_polar_factors(&parent, &eager, &factors, left);
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn erased_polar_redirect_repeats_clones_and_runs_concurrently_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = Space::u1([(-1, 1), (0, 2), (2, 1)]);
        let parent = Tensor::rand_with_seed(&runtime, Dtype::C64, [&leg], [&leg], 706_010).unwrap();
        let lazy = parent.adjoint().unwrap();
        let target = lazy.materialized_tensor_uncached().unwrap();
        for left in [true, false] {
            for _ in 0..2 {
                let factors = if left {
                    lazy.clone().left_polar().unwrap()
                } else {
                    lazy.clone().right_polar().unwrap()
                };
                assert_erased_polar_factors(&parent, &target, &factors, left);
            }
            let calls = (0..4)
                .map(|_| {
                    let clone = lazy.clone();
                    std::thread::spawn(move || {
                        if left {
                            clone.left_polar().unwrap()
                        } else {
                            clone.right_polar().unwrap()
                        }
                    })
                })
                .collect::<Vec<_>>();
            for call in calls {
                let factors = call.join().unwrap();
                assert_erased_polar_factors(&parent, &target, &factors, left);
            }
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn erased_polar_redirect_errors_keep_requested_names() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        for (source, left, message) in [
            (
                Tensor::rand_with_seed(
                    &runtime,
                    Dtype::F64,
                    [&Space::u1([(0, 3)])],
                    [&Space::u1([(0, 2)])],
                    706_001,
                )
                .unwrap(),
                true,
                "left_polar requires rows >= columns in every coupled-sector matrix",
            ),
            (
                Tensor::rand_with_seed(
                    &runtime,
                    Dtype::F64,
                    [&Space::u1([(0, 2)])],
                    [&Space::u1([(0, 3)])],
                    706_002,
                )
                .unwrap(),
                false,
                "right_polar requires columns >= rows in every coupled-sector matrix",
            ),
        ] {
            let lazy = source.adjoint().unwrap();
            let error = if left {
                lazy.left_polar().unwrap_err()
            } else {
                lazy.right_polar().unwrap_err()
            };
            assert!(matches!(
                error,
                Error::Operation(error)
                    if matches!(
                        error.as_ref(),
                        tenet_tensors::OperationError::InvalidArgument { message: actual }
                            if *actual == message
                    )
            ));
            assert_eq!(lazy.adjoint_body_builds(), 0);
            assert!(!lazy.has_cached_materialization());
        }
    }

    fn assert_adjoint_trace_matches_eager_oracle(
        space: Space,
        dtype: Dtype,
        seed: u64,
        pairs: &[(usize, usize)],
    ) {
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let parent =
            Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space, &space], seed)
                .unwrap();
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let parent_f64 = (dtype == Dtype::F64).then(|| parent.data().to_vec());
        let parent_c64 = (dtype == Dtype::C64).then(|| parent.data_c64().to_vec());
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();

        if dtype == Dtype::C64 {
            assert!(parent
                .data_c64()
                .iter()
                .any(|value| value.im.abs() > f64::EPSILON));
        }
        let actual = lazy.trace_pairs(pairs).unwrap();
        let expected = eager.trace_pairs(pairs).unwrap();

        assert_close(&actual, &expected);
        assert_eq!(lazy.adjoint_body_builds(), 0);
        match parent_data.as_ref() {
            Data::F64(data) => assert_eq!(data, parent_f64.as_ref().unwrap()),
            Data::C64(data) => assert_eq!(data, parent_c64.as_ref().unwrap()),
            _ => panic!("unexpected parent storage"),
        }
    }

    #[test]
    fn adjoint_trace_matches_eager_oracles_without_materializing_parent_storage() {
        // What: logical adjoint traces preserve non-self-dual labels, SU2
        // recoupling, fermionic twists, products, pair order, and complex conjugation.
        let asymmetric_u1 = Space::u1([(-3, 1), (-1, 2), (0, 1), (2, 1)]);
        assert_adjoint_trace_matches_eager_oracle(
            asymmetric_u1.clone(),
            Dtype::F64,
            261_401,
            &[(0, 2)],
        );
        assert_adjoint_trace_matches_eager_oracle(
            asymmetric_u1,
            Dtype::C64,
            261_402,
            &[(1, 3), (0, 2)],
        );
        assert_adjoint_trace_matches_eager_oracle(
            Space::su2([(0, 1), (1, 2), (2, 1), (3, 1)]).unwrap(),
            Dtype::C64,
            261_403,
            &[(1, 3)],
        );
        assert_adjoint_trace_matches_eager_oracle(
            Space::fz2([(1, 2)]).unwrap(),
            Dtype::F64,
            261_404,
            &[(1, 3), (0, 2)],
        );
        assert_adjoint_trace_matches_eager_oracle(
            nested_product_space([1, 2, 1, 1]),
            Dtype::C64,
            261_405,
            &[(0, 2)],
        );
    }

    #[test]
    fn asymmetric_rank_adjoint_trace_maps_logical_axes_to_parent_once() {
        // What: a 3|1 parent is a 1|3 logical adjoint, and tracing logical
        // axes (0, 1) retains logical domain axes (2, 3) without a double rotation.
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let space = Space::u1([(-2, 1), (0, 2), (1, 1)]);
        let parent = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space, &space, &space],
            [&space],
            261_407,
        )
        .unwrap();
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();

        assert_eq!(
            logical_adjoint_axes_to_parent(3, 1, &[0, 1, 2, 3]),
            [3, 0, 1, 2]
        );
        let actual = lazy.trace_pairs(&[(0, 1)]).unwrap();
        let expected = eager.trace_pairs(&[(0, 1)]).unwrap();

        assert_close(&actual, &expected);
        assert_eq!(actual.codomain_rank(), 0);
        assert_eq!(actual.domain_rank(), 2);
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }

    #[test]
    fn padded_parent_adjoint_trace_reads_custom_strides_without_materialization() {
        // What: a lazy adjoint trace reads the parent's padded offsets and
        // strides directly, leaves every source cell unchanged, and matches
        // the eager owned-adjoint oracle.
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let p1 = Space::u1([(1, 2)]);
        let m1 = Space::u1([(-1, 2)]);
        let p2 = Space::u1([(2, 2)]);
        let m2 = Space::u1([(-2, 3)]);
        let canonical =
            Tensor::zeros(&runtime, Dtype::C64, [&p1, &m1, &p2, &m2], [&p1, &m1]).unwrap();
        let UserBoundSpace::U1(authority) = canonical.ordinary_body().space.as_ref() else {
            unreachable!()
        };
        let canonical_block = authority.space().structure().block(0).unwrap();
        assert_eq!(authority.space().structure().block_count(), 1);
        assert_eq!(canonical_block.shape(), [2, 2, 2, 3, 2, 2]);
        let structure = BlockStructure::from_blocks_with_rank(
            6,
            vec![tenet_core::BlockSpec::with_key(
                canonical_block.key().clone(),
                canonical_block.shape().to_vec(),
                vec![1, 3, 6, 12, 36, 72],
                1,
            )
            .unwrap()],
        )
        .unwrap();
        let typed = tenet_core::FusionTensorMapSpace::<4, 2>::new_unbound(
            tenet_core::TensorMapSpace::from_dims([2, 2, 2, 3], [2, 2]).unwrap(),
            authority.space().homspace().clone(),
            structure,
        )
        .unwrap()
        .try_bind_rule(authority.provider())
        .unwrap();
        let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            DynamicFusionMapSpace::from_typed(&typed),
            Arc::clone(authority.provider_arc()),
        )
        .unwrap();
        let source = (0..bound.space().required_len().unwrap())
            .map(|index| Complex64::new(index as f64, index as f64 + 0.5))
            .collect::<Vec<_>>();
        let mut canonical_data = vec![
            Complex64::new(0.0, 0.0);
            canonical
                .ordinary_body()
                .space
                .raw()
                .required_len()
                .unwrap()
        ];
        for linear in 0..canonical_block.shape().iter().product() {
            let mut remainder = linear;
            let mut padded_offset = 1;
            for (&dim, stride) in canonical_block.shape().iter().zip([1, 3, 6, 12, 36, 72]) {
                padded_offset += (remainder % dim) * stride;
                remainder /= dim;
            }
            canonical_data[canonical_block.offset() + linear] = source[padded_offset];
        }
        let oracle_parent = Tensor::owned(
            runtime.clone(),
            Arc::clone(&canonical.ordinary_body().space),
            Arc::new(Data::C64(canonical_data)),
        );
        let parent = Tensor::owned(
            runtime,
            Arc::new(UserBoundSpace::U1(bound)),
            Arc::new(Data::C64(source.clone())),
        );
        let lazy = parent.adjoint().unwrap();
        let eager = oracle_parent
            .adjoint()
            .unwrap()
            .materialized_tensor()
            .unwrap();

        let actual = lazy.trace_pairs(&[(0, 2), (1, 3)]).unwrap();
        let expected = eager.trace_pairs(&[(0, 2), (1, 3)]).unwrap();

        assert_close(&actual, &expected);
        assert_eq!(parent.data_c64(), source);
        assert_eq!(lazy.adjoint_body_builds(), 0);

        let owned = eager.scale_c64(Complex64::new(-0.25, 0.5)).unwrap();
        let scale_factor = Complex64::new(0.75, -0.5);
        let scaled = lazy.scale_c64(scale_factor).unwrap();
        assert!(scaled.is_adjoint_view());
        assert_close(
            &scaled.parent_tensor_for_lowering(),
            &parent.scale_c64(scale_factor.conj()).unwrap(),
        );
        assert_close(
            &lazy
                .add_c64(
                    &owned,
                    Complex64::new(0.5, -0.25),
                    Complex64::new(-1.0, 0.75),
                )
                .unwrap(),
            &eager
                .add_c64(
                    &owned,
                    Complex64::new(0.5, -0.25),
                    Complex64::new(-1.0, 0.75),
                )
                .unwrap(),
        );
        assert_scalar_close(lazy.inner(&owned).unwrap(), eager.inner(&owned).unwrap());
        assert_scalar_close(owned.inner(&lazy).unwrap(), owned.inner(&eager).unwrap());
        let alpha = Complex64::new(0.5, -0.25);
        let beta = Complex64::new(-1.0, 0.75);
        let added = lazy.add_c64(&lazy, alpha, beta).unwrap();
        assert!(added.is_adjoint_view());
        assert_close(
            &added.parent_tensor_for_lowering(),
            &parent.add_c64(&parent, alpha.conj(), beta.conj()).unwrap(),
        );
        assert_scalar_close(lazy.inner(&lazy).unwrap(), eager.inner(&eager).unwrap());
        assert!(!lazy.has_cached_materialization());
    }

    #[test]
    fn malformed_adjoint_trace_pairs_fail_before_view_or_destination_builds() {
        // What: the public logical-axis error contract is unchanged and no
        // adjoint grid or result layout is built for an invalid pair list.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-1, 1), (0, 2), (2, 1)]);
        let lazy = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space, &space],
            [&space, &space],
            261_406,
        )
        .unwrap()
        .adjoint()
        .unwrap();

        SELECTED_RESULT_LAYOUT_BUILDS.with(|builds| builds.set(Some(0)));
        for pairs in [&[(0, 4)][..], &[(0, 2), (0, 3)][..]] {
            assert!(matches!(
                lazy.trace_pairs(pairs),
                Err(Error::InvalidArgument(_))
            ));
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
        SELECTED_RESULT_LAYOUT_BUILDS.with(|builds| assert_eq!(builds.get(), Some(0)));
        SELECTED_RESULT_LAYOUT_BUILDS.with(|builds| builds.set(None));
    }

    #[test]
    fn finite_u1_trace_failure_is_atomic_for_owned_and_lazy_adjoint_sources() {
        // What: both public trace routes report the exact finite U1 dual
        // failure without mutating source bytes, materializing an adjoint, or
        // building a result layout.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(0, 1), (i32::MIN, 1)]);
        let parent = Tensor::from_block_fn(&runtime, [&space], [&space], |key, _| match key {
            BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
            _ => 3.0,
        })
        .unwrap();
        let parent_data = Arc::clone(&parent.ordinary_body().data);
        let parent_bytes = parent.data().to_vec();
        let lazy = parent.adjoint().unwrap();
        let expected = Error::FusionAlgebra(Box::new(FusionAlgebraError::U1DualOverflow {
            charge: i32::MIN,
        }));

        SELECTED_RESULT_LAYOUT_BUILDS.with(|builds| builds.set(Some(0)));
        let ordinary = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            parent.trace_pairs(&[(0, 1)])
        }));
        assert_eq!(
            ordinary.expect("owned trace must not unwind").unwrap_err(),
            expected
        );
        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(parent.data(), parent_bytes);
        SELECTED_RESULT_LAYOUT_BUILDS.with(|builds| assert_eq!(builds.get(), Some(0)));

        let adjoint =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lazy.trace_pairs(&[(0, 1)])));
        assert_eq!(
            adjoint.expect("adjoint trace must not unwind").unwrap_err(),
            expected
        );

        assert!(Arc::ptr_eq(&parent.ordinary_body().data, &parent_data));
        assert_eq!(parent.data(), parent_bytes);
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert!(!lazy.has_cached_materialization());
        SELECTED_RESULT_LAYOUT_BUILDS.with(|builds| assert_eq!(builds.get(), Some(0)));
        SELECTED_RESULT_LAYOUT_BUILDS.with(|builds| builds.set(None));
    }

    #[test]
    fn lazy_u1_contract_requiring_dynamic_tree_reports_dual_overflow_without_unwinding() {
        // What: a public lazy-adjoint contraction whose layout requires the
        // dynamic-tree fallback preserves the finite U(1) algebra error.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let minimum = Space::u1([(i32::MIN, 1)]);
        let zero = Space::u1([(0, 1)]);
        let parent = Tensor::zeros(&runtime, Dtype::F64, [&minimum, &zero], [&minimum]).unwrap();
        let rhs = Tensor::zeros(&runtime, Dtype::F64, [&minimum, &zero], [&minimum]).unwrap();
        let lazy = parent.adjoint().unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lazy.contract(&rhs, &[1], &[0])
        }));

        assert_eq!(
            result
                .expect("lazy contraction must not unwind")
                .unwrap_err(),
            Error::FusionAlgebra(Box::new(FusionAlgebraError::U1DualOverflow {
                charge: i32::MIN,
            }))
        );
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn product_trace_preserves_nested_u1_dual_failure() {
        // What: checked product trace exposes its U1 child closure failure,
        // rather than replacing it with a packed-sector codec error.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::product([((0, 0), 1), ((i32::MIN, 1), 1)]).unwrap();
        let tensor = Tensor::zeros(&runtime, Dtype::F64, [&space], [&space]).unwrap();

        assert_eq!(
            tensor.trace_pairs(&[(0, 1)]).unwrap_err(),
            Error::FusionAlgebra(Box::new(FusionAlgebraError::U1DualOverflow {
                charge: i32::MIN,
            }))
        );
    }

    #[test]
    fn public_trace_validation_precedes_finite_algebra_preflight() {
        // What: malformed public axes retain their error precedence over a
        // representable tensor whose traced labels have no finite U1 dual.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(0, 1), (i32::MIN, 1)]);
        let tensor = Tensor::zeros(&runtime, Dtype::F64, [&space], [&space]).unwrap();

        for pairs in [&[(0, 0)][..], &[(0, 1), (0, 1)][..], &[(0, 2)][..]] {
            assert!(matches!(
                tensor.trace_pairs(pairs),
                Err(Error::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn empty_trace_pairs_is_metadata_noop() {
        // What: an explicit empty trace pair list is a metadata-only no-op.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(0, 2), (1, 1)]);
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_508).unwrap();
        let traced = tensor.trace_pairs(&[]).unwrap();

        assert!(Arc::ptr_eq(
            &traced.ordinary_body().space,
            &tensor.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &traced.ordinary_body().data,
            &tensor.ordinary_body().data
        ));
    }

    #[test]
    fn public_trace_matches_fz2_and_su2_hand_oracles() {
        // What: public full partial-trace syntax applies the fermionic odd
        // sign and the spin-half quantum dimension.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let fz2 = Space::fz2([(0, 1), (1, 1)]).unwrap();
        let fermionic = Tensor::from_block_fn(&runtime, [&fz2], [&fz2], |key, _| match key {
            BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
            _ => 3.0,
        })
        .unwrap();
        assert_eq!(
            fermionic
                .trace_pairs(&[(0, 1)])
                .unwrap()
                .scalar()
                .unwrap()
                .try_f64()
                .unwrap(),
            -1.0
        );

        let spin_half = Space::su2([(1, 1)]).unwrap();
        let su2 = Tensor::from_block_fn(&runtime, [&spin_half], [&spin_half], |_, _| 7.0).unwrap();
        assert_eq!(
            su2.trace_pairs(&[(0, 1)])
                .unwrap()
                .scalar()
                .unwrap()
                .try_f64()
                .unwrap(),
            14.0
        );
    }

    fn assert_lowered_transform_matches_eager_oracle(
        lazy: &Tensor,
        actual: Tensor,
        expected: Tensor,
    ) {
        assert_eq!(actual.codomain_spaces(), expected.codomain_spaces());
        assert_eq!(actual.domain_spaces(), expected.domain_spaces());
        assert_eq!(actual.dtype(), expected.dtype());
        assert!(actual.is_adjoint_view());
        assert_eq!(actual.adjoint_body_builds(), 0);
        assert_eq!(lazy.adjoint_body_builds(), 0);

        let expected_parent = expected.adjoint().unwrap().materialized_tensor().unwrap();
        let actual_parent = actual.adjoint().unwrap();
        assert_close(&actual_parent, &expected_parent);

        assert_eq!(actual.adjoint_body_builds(), 0);
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }

    fn assert_adjoint_transforms_stay_parent_lowered(space: Space, dtype: Dtype, seed: u64) {
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let parent =
            Tensor::rand_with_seed(&runtime, dtype, [&space, &space, &space], [&space], seed)
                .unwrap();
        if dtype == Dtype::C64 {
            assert!(parent
                .data_c64()
                .iter()
                .any(|value| value.im.abs() > f64::EPSILON));
        }
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
        let codomain_axes = [3, 0, 2];
        let domain_axes = [1];
        let levels = [17, 3, 11, 5];

        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.permute(&codomain_axes, &domain_axes).unwrap(),
            eager.permute(&codomain_axes, &domain_axes).unwrap(),
        );
        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.braid(&codomain_axes, &domain_axes, &levels).unwrap(),
            eager.braid(&codomain_axes, &domain_axes, &levels).unwrap(),
        );
        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.repartition(3).unwrap(),
            eager.repartition(3).unwrap(),
        );
        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.transpose().unwrap(),
            eager.transpose().unwrap(),
        );

        let involution = lazy.adjoint().unwrap();
        assert!(!involution.is_adjoint_view());
        assert!(Arc::ptr_eq(
            &involution.ordinary_body().space,
            &parent.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &involution.ordinary_body().data,
            &parent.ordinary_body().data
        ));
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }

    fn nested_product_space(degeneracies: [usize; 4]) -> Space {
        Space::fz2_u1_su2([
            ((0, 0, 0), degeneracies[0]),
            ((0, 0, 2), degeneracies[1]),
            ((1, -1, 1), degeneracies[2]),
            ((1, 1, 1), degeneracies[3]),
        ])
        .unwrap()
    }

    #[test]
    fn adjoint_transforms_match_eager_oracles_without_building_adjoint_grids() {
        // What: non-self-dual, fermionic, SU2 inner-line, and product
        // transforms lower to the parent for real and genuinely complex data.
        let spaces = [
            Space::u1([(-2, 1), (-1, 2), (0, 1), (1, 1)]),
            Space::fz2([(1, 2)]).unwrap(),
            Space::su2([(0, 1), (1, 2), (2, 1)]).unwrap(),
            nested_product_space([1, 1, 1, 1]),
        ];
        for (case, space) in spaces.into_iter().enumerate() {
            assert_adjoint_transforms_stay_parent_lowered(
                space.clone(),
                Dtype::F64,
                261_300 + case as u64 * 10,
            );
            assert_adjoint_transforms_stay_parent_lowered(
                space,
                Dtype::C64,
                261_301 + case as u64 * 10,
            );
        }
    }

    #[test]
    fn asymmetric_product_lazy_transpose_matches_eager_materialization() {
        // What: transpose of a complex 3|1 product adjoint preserves the exact
        // reversed split without materializing either lazy transform result.
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let first = nested_product_space([1, 1, 1, 1]);
        let second = nested_product_space([2, 1, 1, 1]);
        let third = nested_product_space([1, 2, 1, 1]);
        let domain = nested_product_space([1, 1, 2, 1]);
        let parent = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&first, &second, &third],
            [&domain],
            261_380,
        )
        .unwrap();
        assert!(parent
            .data_c64()
            .iter()
            .any(|value| value.im.abs() > f64::EPSILON));
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();

        let actual = lazy.transpose().unwrap();
        let expected = eager.transpose().unwrap();

        assert_eq!(
            actual.codomain_spaces(),
            vec![third.dual(), second.dual(), first.dual()]
        );
        assert_eq!(actual.domain_spaces(), vec![domain.dual()]);
        assert_lowered_transform_matches_eager_oracle(&lazy, actual, expected);
    }

    #[test]
    fn adjoint_braid_levels_follow_tensorkit_parent_axis_order() {
        // What: a 3|1 parent's logical levels map to [3, 11, 5, 17], with
        // unchanged values, while the output tuples swap around the adjoint.
        let operation = TreeTransformOperation::braid([3, 0, 2], [1], [17], [3, 11, 5]);
        let lowered = lower_adjoint_tree_transform_operation(3, 1, &operation).unwrap();

        assert_eq!(lowered.codomain_permutation(), [0]);
        assert_eq!(lowered.domain_permutation(), [2, 3, 1]);
        assert_eq!(lowered.codomain_levels(), [3, 11, 5]);
        assert_eq!(lowered.domain_levels(), [17]);
    }

    #[test]
    fn lowered_adjoint_braid_preserves_the_exact_fermionic_swap_sign() {
        // What: swapping two odd fZ2 legs through a lazy adjoint keeps the
        // TensorKit fermionic minus sign and does not build an adjoint grid.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let odd = Space::fz2([(1, 1)]).unwrap();
        let parent = Tensor::from_block_fn(
            &runtime,
            [&odd, &odd],
            std::iter::empty::<&Space>(),
            |_, _| 1.0,
        )
        .unwrap();
        let lazy = parent.adjoint().unwrap();

        let transformed = lazy.braid(&[], &[1, 0], &[0, 1]).unwrap();
        let transformed_parent = transformed.adjoint().unwrap();

        assert!(transformed_parent.data().iter().all(|&value| value == -1.0));
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert_eq!(transformed.adjoint_body_builds(), 0);
    }

    #[test]
    fn malformed_adjoint_transform_errors_precede_any_view_build() {
        // What: level-count errors precede axis errors, and all invalid
        // transform requests leave the lazy adjoint representation untouched.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-2, 1), (0, 2), (1, 1)]);
        let lazy = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space, &space, &space],
            [&space],
            261_390,
        )
        .unwrap()
        .adjoint()
        .unwrap();

        let bad_levels_and_axes = lazy.braid(&[4, 0, 2], &[1], &[17, 3, 11]).unwrap_err();
        assert!(matches!(bad_levels_and_axes, Error::InvalidArgument(_)));
        assert_eq!(lazy.adjoint_body_builds(), 0);

        let bad_axes = lazy.permute(&[4, 0, 2], &[1]).unwrap_err();
        let Error::Operation(error) = bad_axes else {
            panic!("invalid axes returned the wrong error layer");
        };
        assert!(matches!(
            error.as_ref(),
            OperationError::Core(tenet_core::CoreError::InvalidPermutation { .. })
        ));
        assert_eq!(lazy.adjoint_body_builds(), 0);

        assert!(matches!(
            lazy.repartition(5),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }

    fn assert_concurrent_raw_reads_initialize_one_shared_body(dtype: Dtype, seed: u64) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1((-8..=8).map(|charge| (charge, 2)));
        let adjoint = Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed)
            .unwrap()
            .adjoint()
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let clone = adjoint.clone();
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    match dtype {
                        Dtype::F64 => assert!(!clone.try_data().unwrap().is_empty()),
                        Dtype::C64 => assert!(!clone.try_data_c64().unwrap().is_empty()),
                    }
                });
            }
        });
        assert_eq!(adjoint.adjoint_body_builds(), 1);
    }

    #[test]
    fn concurrent_raw_reads_initialize_one_shared_body() {
        // What: f64 and c64 clones racing on their first raw read each publish
        // one coherent materialized body exactly once.
        assert_concurrent_raw_reads_initialize_one_shared_body(Dtype::F64, 261_103);
        assert_concurrent_raw_reads_initialize_one_shared_body(Dtype::C64, 261_105);
    }

    fn assert_host_contract_stays_view_native(dtype: Dtype, seed: u64) {
        let runtime = Runtime::builder().dense_threads(2).build().unwrap();
        let space = Space::u1([(-2, 1), (-1, 2), (0, 3), (1, 2), (2, 1)]);
        let lhs = Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed)
            .unwrap()
            .adjoint()
            .unwrap();
        let rhs =
            Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed + 1).unwrap();

        let barrier = Arc::new(std::sync::Barrier::new(4));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let lhs = lhs.clone();
                let rhs = rhs.clone();
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    assert_eq!(lhs.compose(&rhs).unwrap().rank(), 2);
                });
            }
        });
        // What: contraction consumes the parent plus orientation directly;
        // neither a duplicate logical grid nor owned adjoint data is built.
        assert_eq!(lhs.adjoint_body_builds(), 0);

        for _ in 0..3 {
            let output = lhs.compose(&rhs).unwrap();
            assert_eq!(output.rank(), 2);
        }
        assert_eq!(lhs.adjoint_body_builds(), 0);
    }

    #[test]
    fn fermionic_compose_keeps_lazy_lhs_and_rhs_parent_native() {
        // What: A† * B and A† * B† over the non-Abelian product read both
        // parent buffers through the Core batch without building logical grids
        // or owned adjoints.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2), ((1, 1, 2), 1)]).unwrap();
        let lhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_801)
                .unwrap();
        let rhs = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_802)
            .unwrap();
        let rhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space.dual()], 353_803)
                .unwrap();
        let lhs = lhs_parent.adjoint().unwrap();
        let rhs_lazy = rhs_parent.adjoint().unwrap();

        let lhs_result = lhs.compose(&rhs).unwrap();
        assert_eq!(lhs.adjoint_body_builds(), 0);
        let both_result = lhs.compose(&rhs_lazy).unwrap();
        assert_eq!(lhs.adjoint_body_builds(), 0);
        assert_eq!(rhs_lazy.adjoint_body_builds(), 0);

        let eager_lhs = lhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        let eager_rhs = rhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        assert_close(&lhs_result, &eager_lhs.compose(&rhs).unwrap());
        assert_close(&both_result, &eager_lhs.compose(&eager_rhs).unwrap());
    }

    fn assert_lazy_contract_matches_eager_oracle(space: Space, seed: u64) {
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let lhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], seed).unwrap();
        let rhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], seed + 1)
                .unwrap();
        let plain =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], seed + 2)
                .unwrap();
        let lhs_eager = lhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        let rhs_eager = rhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        let output_axes = [2, 0, 3, 1];

        for (lhs, rhs, eager_lhs, eager_rhs) in [
            (
                lhs_parent.adjoint().unwrap(),
                plain.clone(),
                lhs_eager.clone(),
                plain.clone(),
            ),
            (
                plain.clone(),
                rhs_parent.adjoint().unwrap(),
                plain.clone(),
                rhs_eager.clone(),
            ),
            (
                lhs_parent.adjoint().unwrap(),
                rhs_parent.adjoint().unwrap(),
                lhs_eager.clone(),
                rhs_eager.clone(),
            ),
        ] {
            let expected = eager_lhs
                .contract_ordered(&eager_rhs, &[2], &[0], &output_axes)
                .unwrap();
            let actual = lhs
                .contract_ordered(&rhs, &[2], &[0], &output_axes)
                .unwrap();
            assert_close(&actual, &expected);
            assert_eq!(lhs.adjoint_body_builds(), 0);
            assert_eq!(rhs.adjoint_body_builds(), 0);
        }
    }

    #[test]
    fn oriented_u1_contraction_metadata_matches_materialized_adjoint() {
        // What: F64/C64 lazy adjoints derive ordered destinations and invalid
        // contraction errors from parent metadata before building either view.
        for dtype in [Dtype::F64, Dtype::C64] {
            let runtime = Runtime::builder().dense_threads(1).build().unwrap();
            let bond = Space::u1([(-2, 1), (0, 2), (1, 1)]);
            let bad_bond = Space::u1([(-2, 1), (0, 3), (1, 1)]);
            let lhs_a = Space::u1([(-1, 2), (0, 1), (2, 1)]);
            let lhs_b = Space::u1([(-3, 1), (0, 2), (1, 1)]);
            let rhs_a = Space::u1([(-2, 1), (0, 1), (3, 2)]);
            let rhs_b = Space::u1([(-1, 1), (0, 3), (2, 1)]);
            let parent = Tensor::rand_with_seed(
                &runtime,
                dtype,
                [&bond],
                [&lhs_a, &lhs_b],
                485_000 + dtype as u64,
            )
            .unwrap();
            let rhs = Tensor::rand_with_seed(
                &runtime,
                dtype,
                [&bond],
                [&rhs_a, &rhs_b],
                485_010 + dtype as u64,
            )
            .unwrap();
            let bad_rhs = Tensor::zeros(&runtime, dtype, [&bad_bond], [&rhs_a, &rhs_b]).unwrap();
            let lazy = parent.adjoint().unwrap();
            let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
            let output_axes = [2, 0, 3, 1];
            let expected = eager
                .contract_ordered(&rhs, &[2], &[0], &output_axes)
                .unwrap();

            SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.set(Some(0)));
            let actual = lazy
                .contraction_output_space_oriented(
                    &rhs,
                    &[2],
                    &[0],
                    OutputAxisOrder::from_axes(&output_axes),
                )
                .unwrap();
            assert_eq!(&actual, expected.ordinary_body().space.as_ref());
            assert_eq!(lazy.adjoint_body_builds(), 0);
            assert_eq!(
                SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.replace(None)),
                Some(1)
            );

            let invalid_lazy = parent.adjoint().unwrap();
            SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.set(Some(0)));
            let expected_error = eager
                .contract_ordered(&bad_rhs, &[2], &[0], &[0, 0, 2, 3])
                .unwrap_err();
            let actual_error = invalid_lazy
                .contract_ordered(&bad_rhs, &[2], &[0], &[0, 0, 2, 3])
                .unwrap_err();
            assert_eq!(actual_error, expected_error);
            assert_eq!(invalid_lazy.adjoint_body_builds(), 0);
            assert_eq!(
                SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.replace(None)),
                Some(0)
            );

            let output = lazy
                .contract_ordered(&rhs, &[2], &[0], &output_axes)
                .unwrap();
            assert_eq!(
                output.ordinary_body().space.as_ref(),
                expected.ordinary_body().space.as_ref()
            );
        }
    }

    fn assert_lazy_core_adjoint_matches_eager(
        rows: Space,
        contracted: Space,
        cols: Space,
        dtype: Dtype,
        seed: u64,
    ) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let lhs_parent =
            Tensor::rand_with_seed(&runtime, dtype, [&contracted], [&rows], seed).unwrap();
        let rhs_parent =
            Tensor::rand_with_seed(&runtime, dtype, [&cols], [&contracted], seed + 1).unwrap();
        let lhs_direct =
            Tensor::rand_with_seed(&runtime, dtype, [&rows], [&contracted], seed + 2).unwrap();
        let rhs_direct =
            Tensor::rand_with_seed(&runtime, dtype, [&contracted], [&cols], seed + 3).unwrap();

        let lhs_lazy = lhs_parent.adjoint().unwrap();
        let rhs_lazy = rhs_parent.adjoint().unwrap();
        let lhs_eager = lhs_lazy.materialized_tensor().unwrap();
        let rhs_eager = rhs_lazy.materialized_tensor().unwrap();
        for (lhs, rhs, eager_lhs, eager_rhs) in [
            (
                lhs_lazy.clone(),
                rhs_direct.clone(),
                lhs_eager.clone(),
                rhs_direct.clone(),
            ),
            (
                lhs_direct.clone(),
                rhs_lazy.clone(),
                lhs_direct.clone(),
                rhs_eager.clone(),
            ),
            (lhs_lazy, rhs_lazy, lhs_eager, rhs_eager),
        ] {
            let expected = eager_lhs.compose(&eager_rhs).unwrap();
            let actual = lhs.compose(&rhs).unwrap();
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn repeated_and_parallel_host_contractions_do_not_materialize_adjoint_data() {
        // What: f64 and c64 contraction derive logical geometry from parent
        // orientation and reuse parent storage.
        assert_host_contract_stays_view_native(Dtype::F64, 261_104);
        assert_host_contract_stays_view_native(Dtype::C64, 261_106);
    }

    #[test]
    fn lazy_contraction_matches_eager_oracles_for_supported_rule_families() {
        // What: lhs, rhs, and double adjoints preserve non-self-dual labels,
        // recoupling coefficients, fermionic signs, and crossed output order.
        assert_lazy_contract_matches_eager_oracle(Space::u1([(-2, 1), (-1, 2), (1, 3)]), 261_201);
        assert_lazy_contract_matches_eager_oracle(
            Space::zn(3, [(0, 2), (1, 3), (2, 1)]).unwrap(),
            261_206,
        );
        assert_lazy_contract_matches_eager_oracle(
            Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap(),
            261_211,
        );
        assert_lazy_contract_matches_eager_oracle(Space::fz2([(0, 2), (1, 3)]).unwrap(), 261_221);
        assert_lazy_contract_matches_eager_oracle(
            Space::fz2_u1_su2([
                ((0, 0, 0), 2),
                ((1, -1, 1), 2),
                ((1, 1, 1), 1),
                ((0, 2, 2), 1),
            ])
            .unwrap(),
            261_231,
        );
        assert_lazy_contract_matches_eager_oracle(
            Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2), ((0, 0, 2), 1)]).unwrap(),
            261_241,
        );
    }

    #[test]
    fn lazy_core_adjoint_handles_rectangular_real_and_complex_blocks() {
        // What: lhs, rhs, and both-adjoint Core replay transposes rectangular
        // parent matrices and conjugates complex values exactly once.
        for (dtype, seed) in [(Dtype::F64, 272_100), (Dtype::C64, 272_200)] {
            assert_lazy_core_adjoint_matches_eager(
                Space::su2([(0, 2), (1, 1)]).unwrap(),
                Space::su2([(0, 1), (1, 3)]).unwrap(),
                Space::su2([(0, 3), (1, 2)]).unwrap(),
                dtype,
                seed,
            );
            assert_lazy_core_adjoint_matches_eager(
                Space::u1([(-1, 2), (0, 1), (1, 1)]),
                Space::u1([(-1, 1), (0, 2), (1, 3)]),
                Space::u1([(-1, 3), (0, 1), (1, 2)]),
                dtype,
                seed + 20,
            );
            assert_lazy_core_adjoint_matches_eager(
                Space::fz2([(0, 2), (1, 1)]).unwrap(),
                Space::fz2([(0, 1), (1, 3)]).unwrap(),
                Space::fz2([(0, 3), (1, 2)]).unwrap(),
                dtype,
                seed + 40,
            );
            assert_lazy_core_adjoint_matches_eager(
                Space::fz2_u1_su2([((0, 0, 0), 2), ((1, -1, 1), 1), ((1, 1, 1), 1)]).unwrap(),
                Space::fz2_u1_su2([((0, 0, 0), 1), ((1, -1, 1), 3), ((1, 1, 1), 2)]).unwrap(),
                Space::fz2_u1_su2([((0, 0, 0), 3), ((1, -1, 1), 1), ((1, 1, 1), 2)]).unwrap(),
                dtype,
                seed + 60,
            );
        }
    }
}

#[cfg(test)]
mod shared_context_tests {
    use super::*;

    /// Every runtime-minted executor shares one CPU context, avoiding one
    /// eager rayon pool per rule, dtype, and concurrent lease.
    #[test]
    fn runtime_and_leased_contexts_share_one_cpu_context() {
        let rt = Runtime::builder().build().expect("runtime");
        let shared = rt.execution_config().shared_ctx.clone();
        {
            let mut state = rt.lock();
            assert!(state.shares_cpu_context(&shared));
        }

        let mut lease = rt.lease_context().expect("lease");
        assert!(lease.context().shares_cpu_context(&shared));
        let mut network_context =
            TensorExecutionContext::for_config(rt.execution_config()).expect("context");
        assert!(network_context.shares_cpu_context(&shared));
    }

    #[test]
    fn runtime_builder_recoupling_threads_reach_every_runtime_and_context_lane() {
        fn assert_runtime_and_context(runtime: &Runtime, expected: usize) {
            {
                let mut state = runtime.lock();
                assert!(state.recoupling_threads_are(expected));
            }

            let mut context =
                TensorExecutionContext::for_config(runtime.execution_config()).expect("context");
            assert!(context.recoupling_threads_are(expected));
        }

        let configured = Runtime::builder()
            .recoupling_threads(3)
            .build()
            .expect("configured runtime");
        assert_runtime_and_context(&configured, 3);

        let default = Runtime::builder().build().expect("default runtime");
        assert_runtime_and_context(&default, 1);
    }

    #[test]
    fn runtime_transform_stores_are_isolated_and_expired_weak_handles_run_eagerly() {
        // What: Runtime stores are isolated, clear is local, and a detached
        // context keeps executing after its Runtime store owner is gone.
        let runtime_a = Runtime::builder().build().unwrap();
        let runtime_b = Runtime::builder().build().unwrap();
        let space = Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap();
        let source_a =
            Tensor::rand_with_seed(&runtime_a, Dtype::F64, [&space, &space], [&space], 475_002)
                .unwrap();
        let source_b =
            Tensor::rand_with_seed(&runtime_b, Dtype::F64, [&space, &space], [&space], 475_002)
                .unwrap();
        let expected_a = source_a.permute(&[1], &[2, 0]).unwrap();
        let expected_b = source_b.permute(&[1], &[2, 0]).unwrap();
        assert_eq!(runtime_a.tree_transform_cache_info().entries(), 1);
        assert_eq!(runtime_b.tree_transform_cache_info().entries(), 1);

        runtime_a.clear_tree_transform_cache();
        assert_eq!(runtime_a.tree_transform_cache_info().entries(), 0);
        assert_eq!(runtime_a.tree_transform_cache_info().misses(), 0);
        assert_eq!(runtime_b.tree_transform_cache_info().entries(), 1);
        assert_eq!(runtime_b.tree_transform_cache_info().misses(), 1);

        let mut context = TensorExecutionContext::for_config(runtime_b.execution_config()).unwrap();
        let store = runtime_b.execution_config().tree_transform_store.clone();
        let UserBoundSpace::SU2(source_space) = source_b.ordinary_body().space.as_ref() else {
            unreachable!()
        };
        let rule = Arc::clone(source_space.provider_arc());
        let source_structure = Arc::clone(source_space.space().structure());
        let destination_structure = Arc::clone(expected_b.ordinary_body().space.raw().structure());
        let source_data = source_b.data().to_vec();
        let expected_data = expected_b.data().to_vec();
        let operation = TreeTransformOperation::permute([1], [2, 0]);
        drop(source_a);
        drop(source_b);
        drop(expected_a);
        drop(expected_b);
        drop(runtime_a);
        drop(runtime_b);
        assert!(store.upgrade().is_none());

        let mut actual = vec![f64::NAN; expected_data.len()];
        context
            .mf
            .f64
            .tree_context_mut()
            .tree_transform_dyn_overwrite_into_ref(
                rule.as_ref(),
                &operation,
                &destination_structure,
                &source_structure,
                &mut actual,
                &source_data,
                1.0,
            )
            .unwrap();
        assert_eq!(actual, expected_data);
    }
}

#[cfg(test)]
mod bound_provider_tests {
    use super::*;

    #[test]
    fn construction_and_svd_factors_share_one_provider_allocation() {
        // What: construction and owned factors retain the originating provider allocation.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::z2([(0, 2), (1, 1)]);
        let provider = space.rule_context().as_ref().clone();
        let tensor = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 7).unwrap();
        assert!(tensor
            .ordinary_body()
            .space
            .provider_matches_context_allocation(&provider));

        let (u, s, vh) = tensor.svd_compact().unwrap();
        for factor in [&u, &s, &vh] {
            assert!(factor
                .ordinary_body()
                .space
                .provider_matches_context_allocation(&provider));
        }

        let lazy = tensor.adjoint().unwrap();
        let (u, s, vh) = lazy.svd_compact().unwrap();
        for factor in [&u, &s, &vh] {
            assert!(factor
                .ordinary_body()
                .space
                .provider_matches_context_allocation(&provider));
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);

        for rank in [1, 0] {
            let output = lazy.svd_trunc(&Truncation::rank(rank)).unwrap();
            for factor in [&output.u, &output.s, &output.vh] {
                assert!(factor
                    .ordinary_body()
                    .space
                    .provider_matches_context_allocation(&provider));
            }
        }
        assert_eq!(lazy.adjoint_body_builds(), 0);
    }
}

#[cfg(test)]
mod tk_user_api_tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use tenet_core::{
        Fz2SectorLayout, PackedProductCodec, ProductSectorCodec, ProductSectorLayout,
        Su2SectorLayout, U1Irrep, U1SectorLayout,
    };

    type NestedLabel = (usize, i32, usize);
    type NestedInnerCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type NestedInnerLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type NestedOuterCodec = PackedProductCodec<NestedInnerLayout, Su2SectorLayout>;

    const E: NestedLabel = (0, 0, 0);
    const O: NestedLabel = (1, 0, 1);
    const T: NestedLabel = (0, 0, 2);

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct NestedSemanticElement {
        codomain: Vec<NestedLabel>,
        domain: Vec<NestedLabel>,
        codomain_inner: Vec<NestedLabel>,
        domain_inner: Vec<NestedLabel>,
        codomain_is_dual: Vec<bool>,
        domain_is_dual: Vec<bool>,
        coupled: NestedLabel,
        indices: Vec<usize>,
    }

    fn nested_element(
        codomain: &[NestedLabel],
        domain: &[NestedLabel],
        codomain_inner: &[NestedLabel],
        domain_inner: &[NestedLabel],
        coupled: NestedLabel,
        indices: &[usize],
    ) -> NestedSemanticElement {
        NestedSemanticElement {
            codomain: codomain.to_vec(),
            domain: domain.to_vec(),
            codomain_inner: codomain_inner.to_vec(),
            domain_inner: domain_inner.to_vec(),
            codomain_is_dual: vec![false; codomain.len()],
            domain_is_dual: vec![false; domain.len()],
            coupled,
            indices: indices.to_vec(),
        }
    }

    fn nested_label(sector: SectorId) -> NestedLabel {
        let (inner, spin) = NestedOuterCodec::decode(sector).unwrap();
        let (parity, charge) = NestedInnerCodec::decode(inner).unwrap();
        (
            parity.id(),
            U1Irrep::from_sector_id(charge).unwrap().charge(),
            spin.id(),
        )
    }

    fn normalized_nested_element(key: &BlockKey, indices: &[usize]) -> NestedSemanticElement {
        let BlockKey::FusionTree(key) = key else {
            panic!("nested product fixture must use fusion-tree blocks");
        };
        let labels = |sectors: &[SectorId]| {
            sectors
                .iter()
                .copied()
                .map(nested_label)
                .collect::<Vec<_>>()
        };
        NestedSemanticElement {
            codomain: labels(key.codomain_uncoupled()),
            domain: labels(key.domain_uncoupled()),
            codomain_inner: labels(key.codomain_innerlines()),
            domain_inner: labels(key.domain_innerlines()),
            codomain_is_dual: key.codomain_is_dual().to_vec(),
            domain_is_dual: key.domain_is_dual().to_vec(),
            coupled: nested_label(key.coupled()),
            indices: indices.to_vec(),
        }
    }

    // Why not derive these orders through the legacy Cantor codec: the
    // coefficient oracle must survive another internal SectorId encoding
    // change. These normalized keys are copied from the TensorKit oracle.
    fn nested_source_order() -> Vec<NestedSemanticElement> {
        vec![
            nested_element(&[E, E], &[E, E], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O], &[E, E], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E], &[O, O], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O], &[O, O], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E], &[O, O], &[], &[], E, &[0, 0, 0, 1]),
            nested_element(&[O, O], &[O, O], &[], &[], E, &[0, 0, 0, 1]),
            nested_element(&[O, O], &[O, O], &[], &[], T, &[0, 0, 0, 0]),
            nested_element(&[O, O], &[O, O], &[], &[], T, &[0, 0, 0, 1]),
            nested_element(&[O, E], &[O, E], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, O], &[O, E], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, E], &[E, O], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, O], &[E, O], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, E], &[E, O], &[], &[], O, &[0, 0, 0, 1]),
            nested_element(&[E, O], &[E, O], &[], &[], O, &[0, 0, 0, 1]),
        ]
    }

    fn nested_repartition_3_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[E, E, E], &[E], &[E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, E], &[E], &[E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O], &[E], &[O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O], &[E], &[O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[E, O, O], &[E], &[O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, O, O], &[E], &[O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, E, E], &[O], &[O], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, O, E], &[O], &[O], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, E, O], &[O], &[E], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, E, O], &[O], &[E], &[], O, &[0, 0, 1, 0]),
            nested_element(&[O, O, O], &[O], &[E], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, O, O], &[O], &[E], &[], O, &[0, 0, 1, 0]),
            nested_element(&[O, O, O], &[O], &[T], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, O, O], &[O], &[T], &[], O, &[0, 0, 1, 0]),
        ];
        for element in &mut order {
            element.codomain_is_dual = vec![false, false, true];
        }
        order
    }

    fn nested_repartition_1_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[E], &[E, E, E], &[], &[E], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[O, O, E], &[], &[E], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[O, O, E], &[], &[E], E, &[0, 0, 1, 0]),
            nested_element(&[E], &[O, E, O], &[], &[O], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[E, O, O], &[], &[O], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[E, O, O], &[], &[O], E, &[0, 0, 1, 0]),
            nested_element(&[O], &[O, E, E], &[], &[O], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[E, O, E], &[], &[O], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[E, O, E], &[], &[O], O, &[0, 0, 1, 0]),
            nested_element(&[O], &[E, E, O], &[], &[E], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[O, O, O], &[], &[E], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[O, O, O], &[], &[E], O, &[0, 0, 1, 0]),
            nested_element(&[O], &[O, O, O], &[], &[T], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[O, O, O], &[], &[T], O, &[0, 0, 1, 0]),
        ];
        for element in &mut order {
            element.domain_is_dual = vec![false, false, true];
        }
        order
    }

    fn nested_repartition_0_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[], &[E, E, E, E], &[], &[E, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, E, E], &[], &[E, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, E, E], &[], &[E, E], E, &[0, 1, 0, 0]),
            nested_element(&[], &[O, E, O, E], &[], &[O, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, O, E], &[], &[O, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, O, E], &[], &[O, E], E, &[0, 1, 0, 0]),
            nested_element(&[], &[O, E, E, O], &[], &[O, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, E, O], &[], &[O, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, E, O], &[], &[O, O], E, &[0, 1, 0, 0]),
            nested_element(&[], &[E, E, O, O], &[], &[E, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[E, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[E, O], E, &[0, 1, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[T, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[T, O], E, &[0, 1, 0, 0]),
        ];
        for element in &mut order {
            element.domain_is_dual = vec![false, false, true, true];
        }
        order
    }

    fn nested_repartition_4_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[E, E, E, E], &[], &[E, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, E, E], &[], &[E, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O, E], &[], &[O, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O, E], &[], &[O, E], &[], E, &[0, 0, 1, 0]),
            nested_element(&[E, O, O, E], &[], &[O, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, O, O, E], &[], &[O, E], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, E, E, O], &[], &[O, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, O, E, O], &[], &[O, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E, O, O], &[], &[E, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E, O, O], &[], &[E, O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, O, O, O], &[], &[E, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, O, O], &[], &[E, O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, O, O, O], &[], &[T, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, O, O], &[], &[T, O], &[], E, &[0, 0, 1, 0]),
        ];
        for element in &mut order {
            element.codomain_is_dual = vec![false, false, true, true];
        }
        order
    }

    fn nested_semantic_sequence_tensor(
        rt: &Runtime,
        codomain: &[&Space],
        domain: &[&Space],
    ) -> Tensor {
        // What: TensorKit's sequential fixture values are attached to
        // normalized fusion-tree elements rather than TeNeT storage offsets.
        let order = nested_source_order();
        let assignments = order
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, semantic)| (semantic, (index + 1) as f64))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            assignments.len(),
            order.len(),
            "TensorKit source oracle contains duplicate elements"
        );
        let mut visited = BTreeSet::new();
        let tensor = Tensor::from_block_fn(
            rt,
            codomain.iter().copied(),
            domain.iter().copied(),
            |key, indices| {
                let element = normalized_nested_element(key, indices);
                assert!(
                    visited.insert(element.clone()),
                    "TeNeT enumerated a duplicate source semantic element"
                );
                assignments
                    .get(&element)
                    .copied()
                    .expect("TensorKit source oracle covers every semantic element")
            },
        )
        .unwrap();
        assert_eq!(
            visited,
            assignments.keys().cloned().collect(),
            "TensorKit and TeNeT source semantic element sets differ"
        );
        tensor
    }

    fn assert_nested_semantic_fixture(
        actual: &Tensor,
        order: &[NestedSemanticElement],
        expected: &[f64],
    ) {
        // What: each expected coefficient/sign is selected by normalized
        // fusion-tree labels, dual flags, and local degeneracy indices, with
        // exact coverage and no duplicate or extra elements.
        assert_eq!(order.len(), expected.len());
        let expected = order
            .iter()
            .cloned()
            .zip(expected.iter().copied())
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            expected.len(),
            order.len(),
            "TensorKit semantic oracle contains duplicate elements"
        );

        let structure = actual.ordinary_body().space.structure();
        let mut observed = BTreeMap::new();
        for block_index in 0..structure.block_count() {
            let block = structure.block(block_index).unwrap();
            let mut indices = vec![0usize; block.shape().len()];
            let count: usize = block.shape().iter().product();
            for _ in 0..count {
                let position = block.offset()
                    + indices
                        .iter()
                        .zip(block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                let semantic = normalized_nested_element(block.key(), &indices);
                assert!(
                    observed.insert(semantic, actual.data()[position]).is_none(),
                    "TeNeT produced a duplicate normalized semantic element"
                );
                for (axis, index) in indices.iter_mut().enumerate() {
                    *index += 1;
                    if *index < block.shape()[axis] {
                        break;
                    }
                    *index = 0;
                }
            }
        }
        assert_eq!(observed.len(), actual.data().len());
        assert_eq!(
            observed.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "TensorKit and TeNeT semantic element sets differ"
        );
        for (semantic, expected) in expected {
            let actual = observed[&semantic];
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "TensorKit semantic fixture mismatch for {semantic:?}: \
                 actual={actual}, expected={expected}"
            );
        }
    }

    fn sequential_f64_tensor(rt: &Runtime, codomain: &[&Space], domain: &[&Space]) -> Tensor {
        let mut tensor = Tensor::zeros(
            rt,
            Dtype::F64,
            codomain.iter().copied(),
            domain.iter().copied(),
        )
        .unwrap();
        let body = tensor.owned_body_mut().unwrap();
        let Data::F64(data) = Arc::get_mut(&mut body.data).unwrap() else {
            unreachable!("requested f64 tensor")
        };
        for (index, value) in data.iter_mut().enumerate() {
            *value = (index + 1) as f64;
        }
        tensor
    }

    fn assert_external_axis_order(output: &Tensor, source: &Tensor, axes: &[usize]) {
        assert_eq!(output.rank(), axes.len());
        for (output_axis, &source_axis) in axes.iter().enumerate() {
            assert_eq!(
                output.space(output_axis).unwrap(),
                source.space(source_axis).unwrap()
            );
        }
    }

    fn assert_tensorkit_fixture(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "TensorKit fixture mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn index_count_aliases_match_rank_accessors() {
        // What: numout/numin/numind are exact TK-named aliases of the rank accessors.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let t = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v]).unwrap();
        assert_eq!(t.numout(), t.codomain_rank());
        assert_eq!(t.numin(), t.domain_rank());
        assert_eq!(t.numind(), t.rank());
        assert_eq!((t.numout(), t.numin(), t.numind()), (2, 1, 3));
    }

    #[test]
    fn repartition_moves_the_split_and_round_trips() {
        // What: repartition re-splits legs at the given codomain count, invertibly.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let t = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v]).unwrap();
        let r = t.repartition(1).unwrap();
        assert_eq!((r.codomain_rank(), r.domain_rank()), (1, 2));
        // Back to the original split recovers the original data (planar move).
        let back = r.repartition(2).unwrap();
        assert_eq!(back.data(), t.data());
        assert!(t.repartition(4).is_err());
    }

    #[test]
    fn repartition_uses_tensorkit_planar_axis_order_for_heterogeneous_u1_legs() {
        // What: a 2|2 -> 3|1 repartition moves the last domain leg across the
        // boundary and matches `tensorkit_semantic_oracle.out` section 4,
        // `U1 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::u1([(0, 1)]);
        let b = Space::u1([(0, 2)]);
        let c = Space::u1([(0, 3)]);
        let d = Space::u1([(0, 4)]);
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_eq!((output.codomain_rank(), output.domain_rank()), (3, 1));
        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0, 2.0, 7.0, 8.0, 13.0, 14.0, 19.0, 20.0, 3.0, 4.0, 9.0, 10.0, 15.0, 16.0, 21.0,
                22.0, 5.0, 6.0, 11.0, 12.0, 17.0, 18.0, 23.0, 24.0,
            ],
        );
    }

    #[test]
    fn repartition_same_split_shares_storage_without_transforming() {
        // What: repartitioning to the current split is a zero-copy no-op.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::su2([(0, 1), (1, 2)]).unwrap();
        let source = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 191).unwrap();

        let output = source.repartition(source.codomain_rank()).unwrap();

        assert!(Arc::ptr_eq(
            &output.ordinary_body().space,
            &source.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &output.ordinary_body().data,
            &source.ordinary_body().data
        ));
    }

    #[test]
    fn identity_braid_shares_storage_for_multiplicity_free_rules() {
        // What: exact-axis braids share owned storage for fermionic,
        // non-Abelian, and nested-product tensors even with nonmonotone levels.
        let rt = Runtime::builder().build().unwrap();
        let spaces = [
            Space::fz2([(0, 1), (1, 2)]).unwrap(),
            Space::su2([(0, 1), (1, 2), (2, 1)]).unwrap(),
            Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 2)]).unwrap(),
        ];

        for (case, space) in spaces.iter().enumerate() {
            let source =
                Tensor::rand_with_seed(&rt, Dtype::F64, [space, space], [space], 200 + case as u64)
                    .unwrap();
            let output = source.braid(&[0, 1], &[2], &[17, 3, 11]).unwrap();

            assert!(
                Arc::ptr_eq(&output.ordinary_body().space, &source.ordinary_body().space),
                "case {case}"
            );
            assert!(
                Arc::ptr_eq(&output.ordinary_body().data, &source.ordinary_body().data),
                "case {case}"
            );
        }
    }

    #[test]
    fn identity_braid_validates_levels_before_sharing() {
        // What: malformed braid levels remain an error even when the axis map
        // itself is the identity.
        let rt = Runtime::builder().build().unwrap();
        let space = Space::fz2([(0, 1), (1, 1)]).unwrap();
        let source =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&space, &space], [&space], 203).unwrap();

        assert!(source.braid(&[0, 1], &[2], &[7, 5]).is_err());
    }

    #[test]
    fn identity_braid_shares_rank_zero_storage() {
        // What: the empty axis map is a zero-copy identity braid for a scalar.
        let rt = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 1)]);
        let vector =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&space], std::iter::empty::<&Space>(), 204)
                .unwrap();
        let scalar = vector
            .contract(&vector.adjoint().unwrap(), &[0], &[0])
            .unwrap();

        let output = scalar.braid(&[], &[], &[]).unwrap();

        assert!(Arc::ptr_eq(
            &output.ordinary_body().space,
            &scalar.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &output.ordinary_body().data,
            &scalar.ordinary_body().data
        ));
    }

    #[test]
    fn identity_transpose_shares_rank_zero_storage() {
        // What: TensorKit-style scalar transpose is a zero-copy identity tree
        // transform, not a replay through the general transform path.
        let rt = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 1)]);
        let vector =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&space], std::iter::empty::<&Space>(), 205)
                .unwrap();
        let scalar = vector
            .contract(&vector.adjoint().unwrap(), &[0], &[0])
            .unwrap();

        let output = scalar.transpose().unwrap();

        assert_eq!(output.rank(), 0);
        assert!(Arc::ptr_eq(
            &output.ordinary_body().space,
            &scalar.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &output.ordinary_body().data,
            &scalar.ordinary_body().data
        ));
    }

    #[test]
    fn transpose_axes_accepts_only_planar_cyclic_axis_maps() {
        // What: the public TensorKit-style transpose API accepts a cyclic
        // rotation and rejects an ordinary noncyclic permutation as CoreError.
        let rt = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 1), (1, 1)]);
        let source =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&space, &space], [&space, &space], 206)
                .unwrap();

        assert!(source.transpose_axes(&[1, 3], &[0, 2]).is_ok());
        assert!(matches!(
            source.transpose_axes(&[0, 2], &[1, 3]),
            Err(Error::Core(error))
                if matches!(error.as_ref(), tenet_core::CoreError::InvalidPermutation { .. })
        ));
    }

    #[test]
    fn transpose_axes_keeps_planar_fermionic_signs_distinct_from_permute() {
        // What: a nontrivial odd fZ2 planar transpose is not a braided
        // permutation, so the public operations produce distinct raw data.
        let rt = Runtime::builder().build().unwrap();
        let odd = Space::fz2([(1, 1)]).unwrap();
        let source =
            Tensor::from_block_fn(&rt, [&odd, &odd], std::iter::empty::<&Space>(), |_, _| 1.0)
                .unwrap();

        let transpose = source.transpose_axes(&[1, 0], &[]).unwrap();
        let permute = source.permute(&[1, 0], &[]).unwrap();

        assert_ne!(transpose.data(), permute.data());
    }

    #[test]
    fn transpose_axes_boundary_move_matches_repartition_space_and_data() {
        // What: the explicit cyclic map used by repartition has the same
        // TensorKit planar output spaces and reduced data as repartition.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::fz2([(0, 1), (1, 1)]).unwrap();
        let b = Space::fz2([(0, 2), (1, 1)]).unwrap();
        let c = Space::fz2([(0, 1), (1, 2)]).unwrap();
        let d = Space::fz2([(0, 2), (1, 2)]).unwrap();
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let transpose = source.transpose_axes(&[0, 1, 3], &[2]).unwrap();
        let repartition = source.repartition(3).unwrap();

        assert_eq!(transpose.codomain_spaces(), repartition.codomain_spaces());
        assert_eq!(transpose.domain_spaces(), repartition.domain_spaces());
        assert_eq!(transpose.data(), repartition.data());
    }

    #[test]
    fn nonidentity_braid_keeps_fermionic_odd_swap_sign() {
        // What: the identity shortcut does not absorb a real crossing of two
        // odd fZ2 legs, whose reduced data acquires the fermionic minus sign.
        let rt = Runtime::builder().build().unwrap();
        let odd = Space::fz2([(1, 1)]).unwrap();
        let source =
            Tensor::from_block_fn(&rt, [&odd, &odd], std::iter::empty::<&Space>(), |_, _| 1.0)
                .unwrap();

        let output = source.braid(&[1, 0], &[], &[0, 1]).unwrap();

        assert!(output.data().iter().all(|&value| value == -1.0));
        assert!(!Arc::ptr_eq(
            &output.ordinary_body().data,
            &source.ordinary_body().data
        ));
    }

    #[test]
    fn repartition_matches_tensorkit_for_fermion_odd_sectors() {
        // What: a planar boundary move preserves TensorKit's fZ2 odd-sector
        // signs from semantic oracle section 4, `fZ2 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::fz2([(0, 1), (1, 1)]).unwrap();
        let b = Space::fz2([(0, 2), (1, 1)]).unwrap();
        let c = Space::fz2([(0, 1), (1, 2)]).unwrap();
        let d = Space::fz2([(0, 2), (1, 2)]).unwrap();
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0, 2.0, 4.0, 5.0, 3.0, 6.0, 31.0, 32.0, 34.0, 35.0, 33.0, 36.0, 19.0, 20.0, 25.0,
                26.0, 21.0, 27.0, 7.0, 8.0, 13.0, 14.0, 9.0, 15.0, 22.0, 23.0, 28.0, 29.0, 24.0,
                30.0, 10.0, 11.0, 16.0, 17.0, 12.0, 18.0,
            ],
        );
    }

    #[test]
    fn repartition_matches_tensorkit_for_su2_recoupling() {
        // What: SU2 repartition with nontrivial inner lines reproduces the
        // TensorKit F-move coefficients from semantic oracle section 4,
        // `SU2 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::su2([(0, 1), (1, 1)]).unwrap();
        let b = Space::su2([(0, 1), (1, 1)]).unwrap();
        let c = Space::su2([(0, 1), (1, 1)]).unwrap();
        let d = Space::su2([(0, 1), (1, 2)]).unwrap();
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0,
                2.0,
                12.727_922_061_357_859,
                15.556_349_186_104_049,
                14.142_135_623_730_955,
                16.970_562_748_477_143,
                7.000_000_000_000_002,
                8.000_000_000_000_002,
                -2.121_320_343_559_643,
                -3.535_533_905_932_737_8,
                -2.828_427_124_746_190_3,
                -4.242_640_687_119_286,
                15.921_683_328_090_658,
                17.146_428_199_482_248,
            ],
        );
        assert!(output
            .ordinary_body()
            .space
            .structure()
            .sector_structure()
            .blocks()
            .iter()
            .any(|block| matches!(block.key(), BlockKey::FusionTree(key) if !key.codomain_innerlines().is_empty())));
    }

    #[test]
    fn repartition_matches_tensorkit_for_nested_fz2_u1_su2() {
        // What: nested product coefficients retain both odd parity and SU2
        // recoupling semantics from semantic oracle section 4,
        // `fZ2xU1xSU2 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let base = [((0, 0, 0), 1), ((1, 0, 1), 1)];
        let a = Space::fz2_u1_su2(base).unwrap();
        let b = Space::fz2_u1_su2(base).unwrap();
        let c = Space::fz2_u1_su2(base).unwrap();
        let d = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let source = nested_semantic_sequence_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_nested_semantic_fixture(
            &output,
            &nested_repartition_3_order(),
            &[
                1.0,
                2.0,
                15.556_349_186_104_049,
                18.384_776_310_850_24,
                16.970_562_748_477_143,
                19.798_989_873_223_334,
                9.000_000_000_000_002,
                10.000_000_000_000_002,
                -2.121_320_343_559_643,
                -3.535_533_905_932_737_8,
                -2.828_427_124_746_190_3,
                -4.242_640_687_119_286,
                8.573_214_099_741_124,
                9.797_958_971_132_713,
            ],
        );
    }

    #[test]
    fn threaded_owned_transform_fallback_preserves_nonabelian_results() {
        // What: configuring threaded recoupling leaves the new serial-only
        // owned writer and reproduces the initialized SU2/product path for
        // both real and complex storage.
        let serial = Runtime::builder().build().unwrap();
        let threaded = Runtime::builder().recoupling_threads(2).build().unwrap();

        let su2 = Space::su2([(0, 1), (1, 2)]).unwrap();
        let serial_su2 =
            Tensor::rand_with_seed(&serial, Dtype::C64, [&su2, &su2], [&su2, &su2], 226)
                .unwrap()
                .repartition(3)
                .unwrap();
        let threaded_su2 =
            Tensor::rand_with_seed(&threaded, Dtype::C64, [&su2, &su2], [&su2, &su2], 226)
                .unwrap()
                .repartition(3)
                .unwrap();
        assert_eq!(threaded_su2.data_c64(), serial_su2.data_c64());

        let product = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let serial_product = Tensor::rand_with_seed(
            &serial,
            Dtype::F64,
            [&product, &product],
            [&product, &product],
            227,
        )
        .unwrap()
        .repartition(1)
        .unwrap();
        let threaded_product = Tensor::rand_with_seed(
            &threaded,
            Dtype::F64,
            [&product, &product],
            [&product, &product],
            227,
        )
        .unwrap()
        .repartition(1)
        .unwrap();
        assert_eq!(threaded_product.data(), serial_product.data());
    }

    #[test]
    fn repartition_decreasing_boundary_matches_tensorkit_for_fermion_odd_sectors() {
        // What: moving the boundary in the opposite direction preserves the
        // fZ2 oracle signs from section 4, `fZ2 2|2 -> 1|3`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::fz2([(0, 1), (1, 1)]).unwrap();
        let b = Space::fz2([(0, 2), (1, 1)]).unwrap();
        let c = Space::fz2([(0, 1), (1, 2)]).unwrap();
        let d = Space::fz2([(0, 2), (1, 2)]).unwrap();
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(1).unwrap();

        assert_eq!((output.codomain_rank(), output.domain_rank()), (1, 3));
        assert_external_axis_order(&output, &source, &[0, 2, 3, 1]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0, 4.0, 2.0, 5.0, 7.0, 10.0, 13.0, 16.0, 8.0, 11.0, 14.0, 17.0, 21.0, 24.0, 27.0,
                30.0, 33.0, 36.0, 19.0, 22.0, 25.0, 28.0, 20.0, 23.0, 26.0, 29.0, 31.0, 34.0, 32.0,
                35.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0,
            ],
        );
    }

    #[test]
    fn repartition_decreasing_boundary_matches_tensorkit_for_nested_product() {
        // What: decreasing the boundary retains the nested product's odd
        // parity and SU2 coefficients from semantic oracle section 4,
        // `fZ2xU1xSU2 2|2 -> 1|3`.
        let rt = Runtime::builder().build().unwrap();
        let base = [((0, 0, 0), 1), ((1, 0, 1), 1)];
        let a = Space::fz2_u1_su2(base).unwrap();
        let b = Space::fz2_u1_su2(base).unwrap();
        let c = Space::fz2_u1_su2(base).unwrap();
        let d = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let source = nested_semantic_sequence_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(1).unwrap();

        assert_eq!((output.codomain_rank(), output.domain_rank()), (1, 3));
        assert_external_axis_order(&output, &source, &[0, 2, 3, 1]);
        assert_nested_semantic_fixture(
            &output,
            &nested_repartition_1_order(),
            &[
                1.0,
                3.0,
                5.0,
                14.142_135_623_730_95,
                16.970_562_748_477_14,
                19.798_989_873_223_33,
                9.000_000_000_000_002,
                11.0,
                13.0,
                -std::f64::consts::SQRT_2,
                -2.0 * std::f64::consts::SQRT_2,
                -3.0 * std::f64::consts::SQRT_2,
                8.573_214_099_741_124,
                9.797_958_971_132_713,
            ],
        );
    }

    #[test]
    fn repartition_supports_empty_codomain_empty_domain_and_rank_zero() {
        // What: N=0 and N=rank match the nested-product endpoint fixtures in
        // semantic oracle section 4, while rank zero remains a shared no-op.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::u1([(0, 1)]);
        let b = Space::u1([(0, 2)]);
        let c = Space::u1([(0, 3)]);
        let d = Space::u1([(0, 4)]);
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let all_domain = source.repartition(0).unwrap();
        assert_eq!(
            (all_domain.codomain_rank(), all_domain.domain_rank()),
            (0, 4)
        );
        assert_external_axis_order(&all_domain, &source, &[2, 3, 1, 0]);

        let all_codomain = source.repartition(source.rank()).unwrap();
        assert_eq!(
            (all_codomain.codomain_rank(), all_codomain.domain_rank()),
            (4, 0)
        );
        assert_external_axis_order(&all_codomain, &source, &[0, 1, 3, 2]);

        let base = [((0, 0, 0), 1), ((1, 0, 1), 1)];
        let na = Space::fz2_u1_su2(base).unwrap();
        let nb = Space::fz2_u1_su2(base).unwrap();
        let nc = Space::fz2_u1_su2(base).unwrap();
        let nd = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let nested = nested_semantic_sequence_tensor(&rt, &[&na, &nb], &[&nc, &nd]);

        let nested_all_domain = nested.repartition(0).unwrap();
        assert_external_axis_order(&nested_all_domain, &nested, &[2, 3, 1, 0]);
        assert_nested_semantic_fixture(
            &nested_all_domain,
            &nested_repartition_0_order(),
            &[
                1.0,
                3.0,
                5.0,
                14.142_135_623_730_95,
                16.970_562_748_477_14,
                19.798_989_873_223_33,
                12.727_922_061_357_86,
                15.556_349_186_104_05,
                18.384_776_310_850_24,
                -2.0,
                -4.000_000_000_000_001,
                -6.000_000_000_000_002,
                12.124_355_652_982_14,
                13.856_406_460_551_02,
            ],
        );

        let nested_all_codomain = nested.repartition(nested.rank()).unwrap();
        assert_external_axis_order(&nested_all_codomain, &nested, &[0, 1, 3, 2]);
        assert_nested_semantic_fixture(
            &nested_all_codomain,
            &nested_repartition_4_order(),
            &[
                1.0,
                2.0,
                15.556_349_186_104_05,
                18.384_776_310_850_24,
                16.970_562_748_477_14,
                19.798_989_873_223_33,
                12.727_922_061_357_86,
                14.142_135_623_730_96,
                -3.000_000_000_000_001,
                -5.000_000_000_000_001,
                -4.000_000_000_000_001,
                -6.000_000_000_000_002,
                12.124_355_652_982_14,
                13.856_406_460_551_02,
            ],
        );

        let vector =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&a], std::iter::empty::<&Space>(), 192)
                .unwrap();
        let scalar = vector
            .contract(&vector.adjoint().unwrap(), &[0], &[0])
            .unwrap();
        let repartitioned_scalar = scalar.repartition(0).unwrap();
        assert_eq!(repartitioned_scalar.rank(), 0);
        assert!(Arc::ptr_eq(
            &repartitioned_scalar.ordinary_body().space,
            &scalar.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &repartitioned_scalar.ordinary_body().data,
            &scalar.ordinary_body().data
        ));
    }

    #[test]
    fn zeros_like_is_exact_for_dense_compact_lazy_and_empty_host_storage() {
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
        let index = std::cell::Cell::new(0usize);
        let dense = Tensor::from_block_fn(&rt, [&v], [&v], |_, _| {
            let i = index.get();
            index.set(i + 1);
            values[i % values.len()]
        })
        .unwrap();
        let source_bits: Vec<_> = dense.data().iter().map(|value| value.to_bits()).collect();
        let source_space = Arc::clone(dense.rule_authority_space());
        let zero = dense.zeros_like().unwrap();
        assert!(zero.data().iter().all(|value| value.to_bits() == 0));
        assert_eq!(
            dense
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            source_bits
        );
        assert!(Arc::ptr_eq(zero.rule_authority_space(), &source_space));
        assert!(zero.runtime().same_runtime(dense.runtime()));
        assert_eq!(zero.codomain_spaces(), dense.codomain_spaces());
        assert_eq!(zero.domain_spaces(), dense.domain_spaces());

        let complex = Tensor::from_block_fn(&rt, [&v], [&v], |_, indices| {
            Complex64::new(
                values[indices.iter().sum::<usize>() % values.len()],
                values[(indices.iter().sum::<usize>() + 1) % values.len()],
            )
        })
        .unwrap();
        let complex_zero = complex.zeros_like().unwrap();
        assert!(complex_zero
            .data_c64()
            .iter()
            .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));

        let diagonal = Tensor::diagonal(
            &rt,
            Dtype::F64,
            &v,
            [
                vec![Scalar::F64(f64::NAN), Scalar::F64(f64::INFINITY)],
                vec![Scalar::F64(f64::NEG_INFINITY)],
            ],
        )
        .unwrap();
        let diagonal_zero = diagonal.zeros_like().unwrap();
        let Data::Diagonal(DiagonalData::RealF64(spectrum)) = diagonal_zero.stored_data() else {
            panic!("f64 compact zero must remain compact")
        };
        assert!(spectrum
            .iter()
            .flat_map(|entry| &entry.values)
            .all(|value| value.to_bits() == 0));
        assert!(!diagonal.has_cached_materialization());
        assert!(!diagonal_zero.has_cached_materialization());

        let complex_diagonal = Tensor::diagonal(
            &rt,
            Dtype::C64,
            &v,
            [
                vec![
                    Scalar::C64(Complex64::new(f64::NAN, f64::INFINITY)),
                    Scalar::C64(Complex64::new(f64::NEG_INFINITY, -0.0)),
                ],
                vec![Scalar::C64(Complex64::new(-0.0, f64::NAN))],
            ],
        )
        .unwrap();
        let complex_diagonal_zero = complex_diagonal.zeros_like().unwrap();
        let Data::Diagonal(DiagonalData::C64(spectrum)) = complex_diagonal_zero.stored_data()
        else {
            panic!("c64 compact zero must remain compact")
        };
        assert!(spectrum
            .iter()
            .flat_map(|entry| &entry.values)
            .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));

        let lazy = complex.adjoint().unwrap();
        let lazy_zero = lazy.zeros_like().unwrap();
        assert!(lazy_zero.is_adjoint_view());
        assert!(!lazy.has_cached_materialization());
        assert!(!lazy_zero.has_cached_materialization());
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert_eq!(lazy_zero.adjoint_body_builds(), 0);
        let Data::C64(parent_zero) = lazy_zero.stored_data() else {
            panic!("lazy c64 zero must keep its dense canonical parent")
        };
        assert!(parent_zero
            .iter()
            .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));

        let empty = Space::u1([(0, 0)]);
        let empty = Tensor::from_block_fn(&rt, [&empty], [&empty], |_, _| f64::NAN).unwrap();
        assert!(empty.data().is_empty());
        assert!(empty.zeros_like().unwrap().data().is_empty());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_zero_placement_validation_is_exact() {
        assert!(validate_cuda_zero_placement(0, Placement::Cuda(0)).is_ok());
        assert_eq!(
            validate_cuda_zero_placement(1, Placement::Cuda(0)).unwrap_err(),
            Error::PlacementMismatch
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn cuda_zeros_like_uses_exact_host_zero_upload_and_keeps_lazy_cold() {
        let rt = Runtime::builder().cuda(0).build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
        let index = std::cell::Cell::new(0usize);
        let host = Tensor::from_block_fn(&rt, [&v], [&v], |_, _| {
            let i = index.get();
            index.set(i + 1);
            values[i % values.len()]
        })
        .unwrap();
        let source_bits: Vec<_> = host.data().iter().map(|value| value.to_bits()).collect();
        let device = host.to_cuda().unwrap();
        let authority = Arc::clone(device.rule_authority_space());
        for _ in 0..3 {
            let zero = device.zeros_like().unwrap();
            assert_eq!(zero.placement(), Placement::Cuda(0));
            assert!(Arc::ptr_eq(zero.rule_authority_space(), &authority));
            assert!(zero.runtime().same_runtime(device.runtime()));
            assert!(zero
                .to_host()
                .unwrap()
                .data()
                .iter()
                .all(|value| value.to_bits() == 0));
        }

        let empty_space = Space::u1([(0, 0)]);
        let empty_host =
            Tensor::from_block_fn(&rt, [&empty_space], [&empty_space], |_, _| f64::NAN).unwrap();
        let empty_device = empty_host.to_cuda().unwrap();
        let empty_authority = Arc::clone(empty_device.rule_authority_space());
        let empty_zero = empty_device.zeros_like().unwrap();
        assert_eq!(empty_zero.placement(), Placement::Cuda(0));
        assert!(Arc::ptr_eq(
            empty_zero.rule_authority_space(),
            &empty_authority
        ));
        assert!(empty_zero.runtime().same_runtime(empty_device.runtime()));
        assert!(empty_zero.to_host().unwrap().data().is_empty());

        assert_eq!(
            device
                .to_host()
                .unwrap()
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            source_bits
        );

        let lazy = device.adjoint().unwrap();
        let lazy_zero = lazy.zeros_like().unwrap();
        assert!(lazy_zero.is_adjoint_view());
        assert!(!lazy.has_cached_materialization());
        assert!(!lazy_zero.has_cached_materialization());
        assert_eq!(lazy.adjoint_body_builds(), 0);
        assert_eq!(lazy_zero.adjoint_body_builds(), 0);

        let mut missing_context = device.clone();
        missing_context.rt = Runtime::builder().build().unwrap();
        assert!(matches!(
            missing_context.zeros_like(),
            Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
        ));
        assert_eq!(device.placement(), Placement::Cuda(0));
    }

    #[test]
    fn identity_is_hermitian_isometric_unitary_posdef() {
        // What: the identity endomorphism satisfies every structural predicate.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let id = Tensor::id(&rt, Dtype::F64, [&v, &v]).unwrap();
        assert!(id.is_hermitian(1e-12).unwrap());
        assert!(id.is_isometric(1e-12).unwrap());
        assert!(id.is_unitary(1e-12).unwrap());
        assert!(id.is_posdef(1e-12).unwrap());
    }

    #[test]
    fn non_endomorphism_is_not_hermitian() {
        // What: a rectangular map returns false rather than erroring.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let w = Space::u1([(0, 3), (1, 2)]);
        let t = Tensor::rand(&rt, Dtype::F64, [&v], [&w]).unwrap();
        assert!(!t.is_hermitian(1e-12).unwrap());
    }

    #[test]
    fn negative_identity_is_hermitian_but_not_posdef() {
        // What: is_posdef rejects a Hermitian tensor with a negative eigenvalue.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let minus_id = Tensor::id(&rt, Dtype::F64, [&v])
            .unwrap()
            .scale(-1.0)
            .unwrap();
        assert!(minus_id.is_hermitian(1e-12).unwrap());
        assert!(!minus_id.is_posdef(1e-12).unwrap());
    }

    #[test]
    fn zero_tensor_is_not_posdef() {
        // What: a zero spectrum is positive SEMIdefinite, so strict posdef
        // (TK isposdef = Cholesky) must reject it.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let zero = Tensor::zeros(&rt, Dtype::F64, [&v], [&v]).unwrap();
        let zero2 = Tensor::zeros(&rt, Dtype::F64, [&v], [&v]).unwrap();
        assert_eq!(
            zero.rule_authority_space().identity(),
            zero2.rule_authority_space().identity()
        );
        assert!(zero.is_hermitian(1e-12).unwrap());
        assert!(!zero.is_posdef(1e-12).unwrap());
    }

    #[test]
    fn zn_erased_nonzero_tensor_transposes_with_sector_routing() {
        let rt = Runtime::builder().build().unwrap();
        let v = Space::zn(3, [(0, 1), (1, 2)]).unwrap();
        let tensor = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
            BlockKey::FusionTree(key) => key.codomain_uncoupled()[0].id() as f64 + 1.0,
            _ => 1.0,
        })
        .unwrap();
        let transposed = tensor.transpose().unwrap();
        assert_eq!(
            transposed.codomain_spaces()[0].zn_sectors().unwrap(),
            vec![(0, 1), (2, 2)]
        );
        assert_eq!(
            transposed.domain_spaces()[0].zn_sectors().unwrap(),
            vec![(0, 1), (2, 2)]
        );
        assert_eq!(transposed.data(), tensor.data());
    }

    #[test]
    fn zn_erased_compose_matches_sector_oracle() {
        let rt = Runtime::builder().build().unwrap();
        let v = Space::zn(2, [(0, 1), (1, 1)]).unwrap();
        let vd = v.dual();
        let lhs = Tensor::from_block_fn(&rt, [&v], [&vd], |key, _| {
            if matches!(key, BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0) {
                2.0
            } else {
                3.0
            }
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&rt, [&vd], [&v], |key, _| {
            if matches!(key, BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0) {
                5.0
            } else {
                7.0
            }
        })
        .unwrap();
        let composed = lhs.compose(&rhs).unwrap();
        assert_eq!(composed.data(), &[10.0, 21.0]);
    }

    #[test]
    fn zn3_nonselfdual_compose_routes_both_dual_channels() {
        let rt = Runtime::builder().build().unwrap();
        let v = Space::zn(3, [(1, 1), (2, 1)]).unwrap();
        let vd = v.dual();
        let lhs = Tensor::from_block_fn(&rt, [&v], [&vd], |key, _| match key {
            BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 1 => 2.0,
            _ => 3.0,
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&rt, [&vd], [&v], |key, _| match key {
            BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 2 => 5.0,
            _ => 7.0,
        })
        .unwrap();
        assert_eq!(lhs.compose(&rhs).unwrap().data(), &[14.0, 15.0]);
        assert_eq!(
            lhs.contract(&rhs, &[1], &[0]).unwrap().data(),
            &[14.0, 15.0]
        );
    }

    #[test]
    fn hermitian_projectors_split_a_general_endomorphism() {
        // What: t = project_hermitian(t) + project_antihermitian(t), and each
        // part satisfies its predicate.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let t = Tensor::rand(&rt, Dtype::C64, [&v], [&v]).unwrap();
        let herm = t.project_hermitian().unwrap();
        let anti = t.project_antihermitian().unwrap();
        assert!(herm.is_hermitian(1e-10).unwrap());
        assert!(anti.is_antihermitian(1e-10).unwrap());
        // Reassembled parts recover the original tensor.
        let recomposed = herm.add(&anti, 1.0, 1.0).unwrap();
        assert!(recomposed.add(&t, 1.0, -1.0).unwrap().norm().unwrap() < 1e-10);
    }
}

#[cfg(test)]
mod cat_tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    fn oracle_value(source: usize, key: &BlockKey, indices: &[usize]) -> f64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let key_part = (hasher.finish() & 0x000f_ffff) as f64;
        let index_part = indices
            .iter()
            .enumerate()
            .map(|(axis, &index)| (axis + 1) * (index + 1))
            .sum::<usize>() as f64;
        source as f64 * 2_000_000.0 + key_part + index_part / 1024.0
    }

    fn changed_sector(key: &BlockKey, side: CatSide) -> SectorId {
        let BlockKey::FusionTree(key) = key else {
            panic!("cat output must use fusion-tree keys");
        };
        match side {
            CatSide::Domain => key.domain_uncoupled()[0],
            CatSide::Codomain => key.codomain_uncoupled()[0],
        }
    }

    fn assert_f64_oracle(lhs: &Tensor, output: &Tensor, side: CatSide) {
        let axis = match side {
            CatSide::Domain => output.codomain_rank(),
            CatSide::Codomain => 0,
        };
        let lhs_metadata = lhs.metadata();
        let lhs_leg = match side {
            CatSide::Domain => &lhs_metadata.domain().legs()[0],
            CatSide::Codomain => &lhs_metadata.codomain().legs()[0],
        };
        let structure = output.ordinary_body().space.structure();
        for block_index in 0..structure.block_count() {
            let block = structure.block(block_index).unwrap();
            let lhs_deg = lhs_leg
                .degeneracy(changed_sector(block.key(), side))
                .unwrap_or(0);
            let mut indices = vec![0; block.shape().len()];
            for _ in 0..block.shape().iter().product() {
                let (source, shift) = if indices[axis] < lhs_deg {
                    (0, 0)
                } else {
                    (1, lhs_deg)
                };
                let mut source_indices = indices.clone();
                source_indices[axis] -= shift;
                let position = block.offset()
                    + indices
                        .iter()
                        .zip(block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                assert_eq!(
                    output.data()[position],
                    oracle_value(source, block.key(), &source_indices),
                    "key={:?}, indices={indices:?}",
                    block.key()
                );
                advance_indices(&mut indices, block.shape());
            }
        }
    }

    fn assert_c64_oracle(lhs: &Tensor, output: &Tensor, side: CatSide) {
        let axis = match side {
            CatSide::Domain => output.codomain_rank(),
            CatSide::Codomain => 0,
        };
        let lhs_metadata = lhs.metadata();
        let lhs_leg = match side {
            CatSide::Domain => &lhs_metadata.domain().legs()[0],
            CatSide::Codomain => &lhs_metadata.codomain().legs()[0],
        };
        let structure = output.ordinary_body().space.structure();
        for block_index in 0..structure.block_count() {
            let block = structure.block(block_index).unwrap();
            let lhs_deg = lhs_leg
                .degeneracy(changed_sector(block.key(), side))
                .unwrap_or(0);
            let mut indices = vec![0; block.shape().len()];
            for _ in 0..block.shape().iter().product() {
                let (source, shift) = if indices[axis] < lhs_deg {
                    (0, 0)
                } else {
                    (1, lhs_deg)
                };
                let mut source_indices = indices.clone();
                source_indices[axis] -= shift;
                let expected = oracle_value(source, block.key(), &source_indices);
                let position = block.offset()
                    + indices
                        .iter()
                        .zip(block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                assert_eq!(
                    output.data_c64()[position],
                    Complex64::new(expected, -expected)
                );
                advance_indices(&mut indices, block.shape());
            }
        }
    }

    fn advance_indices(indices: &mut [usize], shape: &[usize]) {
        for axis in 0..indices.len() {
            indices[axis] += 1;
            if indices[axis] < shape[axis] {
                return;
            }
            indices[axis] = 0;
        }
    }

    fn for_each_cat_element(
        lhs: &Tensor,
        rhs: &Tensor,
        output: &Tensor,
        side: CatSide,
        mut check: impl FnMut(usize, usize, usize),
    ) {
        let axis = match side {
            CatSide::Domain => output.codomain_rank(),
            CatSide::Codomain => 0,
        };
        let lhs_metadata = lhs.metadata();
        let lhs_leg = match side {
            CatSide::Domain => &lhs_metadata.domain().legs()[0],
            CatSide::Codomain => &lhs_metadata.codomain().legs()[0],
        };
        let source_bodies = [
            lhs.materialized_body().unwrap(),
            rhs.materialized_body().unwrap(),
        ];
        let output_structure = output.ordinary_body().space.structure();
        for block_index in 0..output_structure.block_count() {
            let output_block = output_structure.block(block_index).unwrap();
            let lhs_degeneracy = lhs_leg
                .degeneracy(changed_sector(output_block.key(), side))
                .unwrap_or(0);
            let mut indices = vec![0; output_block.shape().len()];
            for _ in 0..output_block.shape().iter().product() {
                let (source, shift) = if indices[axis] < lhs_degeneracy {
                    (0, 0)
                } else {
                    (1, lhs_degeneracy)
                };
                let source_structure = source_bodies[source].space.structure();
                let source_block = source_structure
                    .find_block_index_by_key(output_block.key())
                    .and_then(|index| source_structure.block(index).ok())
                    .expect("cat output key must belong to one source");
                let mut source_indices = indices.clone();
                source_indices[axis] -= shift;
                let source_position = source_block.offset()
                    + source_indices
                        .iter()
                        .zip(source_block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                let output_position = output_block.offset()
                    + indices
                        .iter()
                        .zip(output_block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                check(source, source_position, output_position);
                advance_indices(&mut indices, output_block.shape());
            }
        }
    }

    fn assert_f64_sources(lhs: &Tensor, rhs: &Tensor, output: &Tensor, side: CatSide) {
        let sources = [lhs.try_data().unwrap(), rhs.try_data().unwrap()];
        let destination = output.try_data().unwrap();
        for_each_cat_element(
            lhs,
            rhs,
            output,
            side,
            |source, source_position, output_position| {
                assert_eq!(
                    destination[output_position],
                    sources[source][source_position]
                );
            },
        );
    }

    fn independent_materialized(tensor: &Tensor) -> Tensor {
        let equivalent = match &tensor.repr {
            TensorRepr::Owned(_) => tensor.clone(),
            TensorRepr::Adjoint(view) => Tensor::owned(
                tensor.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .adjoint()
            .unwrap(),
        };
        equivalent.materialized_tensor().unwrap()
    }

    fn assert_dense_storage_eq(actual: &Data, expected: &Data) {
        match (actual, expected) {
            (Data::F64(actual), Data::F64(expected)) => assert_eq!(actual, expected),
            (Data::C64(actual), Data::C64(expected)) => assert_eq!(actual, expected),
            _ => panic!("cat oracle storage dtype changed"),
        }
    }

    fn assert_lazy_cat_matches_eager(lhs: &Tensor, rhs: &Tensor, side: CatSide) -> Tensor {
        let lhs_before = lhs.stored_data().clone();
        let rhs_before = rhs.stored_data().clone();
        let eager_lhs = independent_materialized(lhs);
        let eager_rhs = independent_materialized(rhs);
        let expected = match side {
            CatSide::Domain => eager_lhs.catdomain(&eager_rhs).unwrap(),
            CatSide::Codomain => eager_lhs.catcodomain(&eager_rhs).unwrap(),
        };

        let actual = match side {
            CatSide::Domain => lhs.catdomain(rhs).unwrap(),
            CatSide::Codomain => lhs.catcodomain(rhs).unwrap(),
        };

        assert_eq!(
            actual.ordinary_body().space.homspace(),
            expected.ordinary_body().space.homspace()
        );
        assert_eq!(
            actual.ordinary_body().space.structure(),
            expected.ordinary_body().space.structure()
        );
        assert_dense_storage_eq(actual.stored_data(), expected.stored_data());
        assert_dense_storage_eq(lhs.stored_data(), &lhs_before);
        assert_dense_storage_eq(rhs.stored_data(), &rhs_before);
        assert_eq!(lhs.adjoint_body_builds(), 0);
        assert_eq!(rhs.adjoint_body_builds(), 0);
        actual
    }

    fn u1_cat_operand(
        runtime: &Runtime,
        common: &Space,
        changed: &Space,
        side: CatSide,
        adjoint: bool,
        dtype: Dtype,
        source: usize,
    ) -> Tensor {
        let (codomain, domain): (Vec<&Space>, Vec<&Space>) = match (side, adjoint) {
            (CatSide::Domain, false) | (CatSide::Codomain, true) => (vec![common], vec![changed]),
            (CatSide::Domain, true) | (CatSide::Codomain, false) => (vec![changed], vec![common]),
        };
        let tensor = match dtype {
            Dtype::F64 => Tensor::from_block_fn(runtime, codomain, domain, |key, indices| {
                oracle_value(source, key, indices)
            })
            .unwrap(),
            Dtype::C64 => Tensor::from_block_fn(runtime, codomain, domain, |key, indices| {
                let value = oracle_value(source, key, indices);
                Complex64::new(value, value / (source + 3) as f64 + 0.25)
            })
            .unwrap(),
        };
        if adjoint {
            tensor.adjoint().unwrap()
        } else {
            tensor
        }
    }

    fn repack_c64_by_key(source: &Tensor, destination: &BlockStructure) -> Vec<Complex64> {
        let source_structure = source.ordinary_body().space.structure();
        let source_data = source.data_c64();
        let mut destination_data =
            vec![Complex64::new(0.0, 0.0); destination.required_len().unwrap()];
        for destination_index in 0..destination.block_count() {
            let destination_block = destination.block(destination_index).unwrap();
            let source_block = source_structure
                .find_block_index_by_key(destination_block.key())
                .and_then(|index| source_structure.block(index).ok())
                .unwrap();
            assert_eq!(source_block.shape(), destination_block.shape());
            let mut indices = vec![0; destination_block.shape().len()];
            for _ in 0..destination_block.shape().iter().product() {
                let source_position = source_block.offset()
                    + indices
                        .iter()
                        .zip(source_block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                let destination_position = destination_block.offset()
                    + indices
                        .iter()
                        .zip(destination_block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                destination_data[destination_position] = source_data[source_position];
                advance_indices(&mut indices, destination_block.shape());
            }
        }
        destination_data
    }

    fn reordered_complete_su2_parent(source: &Tensor) -> Tensor {
        let UserBoundSpace::SU2(authority) = source.ordinary_body().space.as_ref() else {
            unreachable!()
        };
        let canonical = authority.space().structure();
        let mut blocks = (0..canonical.block_count())
            .map(|index| {
                let block = canonical.block(index).unwrap();
                let BlockKey::FusionTree(key) = block.key() else {
                    unreachable!()
                };
                (key.clone(), block.shape().to_vec())
            })
            .collect::<Vec<_>>();
        let mut start = 0;
        while start < blocks.len() {
            let coupled = blocks[start].0.codomain_tree().coupled();
            let end = blocks[start..]
                .iter()
                .position(|(key, _)| key.codomain_tree().coupled() != coupled)
                .map_or(blocks.len(), |relative| start + relative);
            blocks[start..end].reverse();
            start = end;
        }
        let reordered = BlockStructure::coupled_sector_matrix_with_keys(
            authority.provider(),
            authority.space().nout(),
            authority.space().rank(),
            blocks,
        )
        .unwrap();
        let data = repack_c64_by_key(source, &reordered);
        let dense = tenet_core::TensorMapSpace::<1, 2>::from_dims(
            [source.codomain_spaces()[0].dim()],
            [
                source.domain_spaces()[0].dim(),
                source.domain_spaces()[1].dim(),
            ],
        )
        .unwrap();
        let typed = tenet_core::FusionTensorMapSpace::<1, 2>::new_unbound(
            dense,
            authority.space().homspace().clone(),
            reordered,
        )
        .unwrap()
        .try_bind_rule(authority.provider())
        .unwrap();
        let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            DynamicFusionMapSpace::from_typed(&typed),
            Arc::clone(authority.provider_arc()),
        )
        .unwrap();
        Tensor::owned(
            source.rt.clone(),
            Arc::new(UserBoundSpace::SU2(bound)),
            Arc::new(Data::C64(data)),
        )
    }

    #[test]
    fn catdomain_u1_overlapping_and_disjoint_sectors_matches_hand_oracle() {
        let runtime = Runtime::builder().build().unwrap();
        let c0 = Space::u1([(-1, 2), (0, 1), (1, 2)]);
        let c1 = Space::u1([(-1, 1), (0, 2), (1, 1)]);
        let left = Space::u1([(-1, 2), (0, 1)]);
        let right = Space::u1([(0, 3), (1, 2)]);
        let lhs = Tensor::from_block_fn(&runtime, [&c0, &c1], [&left], |key, indices| {
            oracle_value(0, key, indices)
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, [&c0, &c1], [&right], |key, indices| {
            oracle_value(1, key, indices)
        })
        .unwrap();
        let lhs_before = lhs.data().to_vec();
        let rhs_before = rhs.data().to_vec();

        let output = lhs.catdomain(&rhs).unwrap();

        assert_eq!(output.codomain_spaces(), vec![c0, c1]);
        assert_eq!(output.domain_spaces()[0], left.oplus(&right).unwrap());
        assert_f64_oracle(&lhs, &output, CatSide::Domain);
        assert_eq!(lhs.data(), lhs_before);
        assert_eq!(rhs.data(), rhs_before);
    }

    #[test]
    fn catdomain_with_empty_codomain_still_concatenates_columns() {
        let runtime = Runtime::builder().build().unwrap();
        let left = Space::u1([(0, 1)]);
        let right = Space::u1([(0, 2)]);
        let lhs = Tensor::from_block_fn(&runtime, std::iter::empty(), [&left], |_, indices| {
            (indices[0] + 1) as f64
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, std::iter::empty(), [&right], |_, indices| {
            (indices[0] + 2) as f64
        })
        .unwrap();

        let output = lhs.catdomain(&rhs).unwrap();

        assert_eq!(output.codomain_rank(), 0);
        assert_eq!(output.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn catcodomain_su2_complex_matches_independent_row_slab_oracle() {
        let runtime = Runtime::builder().build().unwrap();
        let left = Space::su2([(0, 2), (1, 1)]).unwrap();
        let right = Space::su2([(1, 3), (2, 2)]).unwrap();
        let d0 = Space::su2([(0, 1), (1, 2), (2, 1)]).unwrap();
        let d1 = Space::su2([(0, 2), (1, 1)]).unwrap();
        let lhs = Tensor::from_block_fn(&runtime, [&left], [&d0, &d1], |key, indices| {
            let value = oracle_value(0, key, indices);
            Complex64::new(value, -value)
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, [&right], [&d0, &d1], |key, indices| {
            let value = oracle_value(1, key, indices);
            Complex64::new(value, -value)
        })
        .unwrap();
        let lhs_before = lhs.data_c64().to_vec();
        let rhs_before = rhs.data_c64().to_vec();

        let output = lhs.catcodomain(&rhs).unwrap();

        assert_eq!(output.codomain_spaces()[0], left.oplus(&right).unwrap());
        assert_eq!(output.domain_spaces(), vec![d0, d1]);
        assert_c64_oracle(&lhs, &output, CatSide::Codomain);
        assert_eq!(lhs.data_c64(), lhs_before);
        assert_eq!(rhs.data_c64(), rhs_before);
    }

    #[test]
    fn cat_handles_fermionic_odd_and_product_fusion_keys() {
        let runtime = Runtime::builder().build().unwrap();
        let f0 = Space::fz2([(0, 1), (1, 2)]).unwrap();
        let f1 = Space::fz2([(1, 3)]).unwrap();
        let unchanged = Space::fz2([(0, 2), (1, 1)]).unwrap();
        let lhs = Tensor::from_block_fn(&runtime, [&unchanged], [&f0], |key, indices| {
            oracle_value(0, key, indices)
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, [&unchanged], [&f1], |key, indices| {
            oracle_value(1, key, indices)
        })
        .unwrap();
        let fermionic = lhs.catdomain(&rhs).unwrap();
        assert_f64_oracle(&lhs, &fermionic, CatSide::Domain);

        let p0 = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, -1, 1), 1)]).unwrap();
        let p1 = Space::fz2_u1_su2([((1, -1, 1), 2), ((1, 1, 1), 1)]).unwrap();
        let d0 = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, -1, 1), 1)]).unwrap();
        let d1 = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 1, 1), 1)]).unwrap();
        let lhs = Tensor::from_block_fn(&runtime, [&p0], [&d0, &d1], |key, indices| {
            oracle_value(0, key, indices)
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, [&p1], [&d0, &d1], |key, indices| {
            oracle_value(1, key, indices)
        })
        .unwrap();
        let product = lhs.catcodomain(&rhs).unwrap();
        assert_f64_oracle(&lhs, &product, CatSide::Codomain);
    }

    #[test]
    fn cat_promotes_mixed_f64_c64_in_both_operand_orders() {
        let runtime = Runtime::builder().build().unwrap();
        let codomain = Space::u1([(0, 2)]);
        let left = Space::u1([(0, 1)]);
        let right = Space::u1([(0, 2)]);
        let lhs = Tensor::from_block_fn(&runtime, [&codomain], [&left], |_, indices| {
            (indices[0] + 1) as f64
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, [&codomain], [&right], |_, indices| {
            let value = (indices[0] + 2 * indices[1] + 3) as f64;
            Complex64::new(value, -value)
        })
        .unwrap();

        let domain = lhs.catdomain(&rhs).unwrap();
        let reverse_domain = rhs.catdomain(&lhs).unwrap();
        assert_eq!(domain.dtype(), Dtype::C64);
        assert_eq!(
            domain.data_c64(),
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, -3.0),
                Complex64::new(4.0, -4.0),
                Complex64::new(5.0, -5.0),
                Complex64::new(6.0, -6.0),
            ]
        );
        assert_eq!(
            reverse_domain.data_c64(),
            &[
                Complex64::new(3.0, -3.0),
                Complex64::new(4.0, -4.0),
                Complex64::new(5.0, -5.0),
                Complex64::new(6.0, -6.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ]
        );

        let domain_space = Space::u1([(0, 2)]);
        let upper = Space::u1([(0, 1)]);
        let lower = Space::u1([(0, 2)]);
        let lhs = Tensor::from_block_fn(&runtime, [&upper], [&domain_space], |_, indices| {
            let value = (indices[1] + 1) as f64;
            Complex64::new(value, -value)
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, [&lower], [&domain_space], |_, indices| {
            (indices[0] + 2 * indices[1] + 3) as f64
        })
        .unwrap();

        let codomain = lhs.catcodomain(&rhs).unwrap();
        let reverse_codomain = rhs.catcodomain(&lhs).unwrap();
        assert_eq!(codomain.dtype(), Dtype::C64);
        assert_eq!(
            codomain.data_c64(),
            &[
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, -2.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]
        );
        assert_eq!(
            reverse_codomain.data_c64(),
            &[
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(2.0, -2.0),
            ]
        );
    }

    #[test]
    fn catcodomain_materializes_compact_diagonal_operands() {
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 2), (1, 1)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 11_055).unwrap();
        let diagonal = source.svd_compact().unwrap().1;
        assert!(matches!(diagonal.stored_data(), Data::Diagonal(_)));
        assert!(!diagonal.has_cached_materialization());

        let output = diagonal.catcodomain(&diagonal).unwrap();

        assert!(matches!(diagonal.stored_data(), Data::Diagonal(_)));
        assert!(diagonal.has_cached_materialization());
        assert!(matches!(output.stored_data(), Data::F64(_)));
        assert_f64_sources(&diagonal, &diagonal, &output, CatSide::Codomain);
    }

    #[test]
    fn lazy_adjoint_cat_u1_matches_eager_for_both_sides_and_operand_orientations() {
        // What: AD, DA, and AA concatenation read asymmetric dual U1 parent
        // storage directly for both cat sides, including both mixed-dtype orders.
        let runtime = Runtime::builder().build().unwrap();
        let common = Space::u1([(-2, 1), (0, 2), (1, 1)]);
        let left = Space::u1([(-1, 2), (2, 1)]).dual();
        let right = Space::u1([(-1, 1), (3, 2)]).dual();
        for side in [CatSide::Domain, CatSide::Codomain] {
            for (lhs_adjoint, rhs_adjoint, lhs_dtype, rhs_dtype) in [
                (true, false, Dtype::C64, Dtype::F64),
                (false, true, Dtype::F64, Dtype::C64),
                (true, true, Dtype::C64, Dtype::C64),
            ] {
                let lhs = u1_cat_operand(&runtime, &common, &left, side, lhs_adjoint, lhs_dtype, 0);
                let rhs =
                    u1_cat_operand(&runtime, &common, &right, side, rhs_adjoint, rhs_dtype, 1);
                assert_lazy_cat_matches_eager(&lhs, &rhs, side);
            }
        }
    }

    #[test]
    fn lazy_adjoint_catdomain_su2_routes_multi_tree_keys_without_sorted_projection() {
        // What: complete SU2 fusion-tree pairs remain exact when swapping the
        // parent key order does not preserve the destination's sorted order.
        let runtime = Runtime::builder().build().unwrap();
        let common0 = Space::su2([(0, 2), (1, 1), (2, 1)]).unwrap();
        let common1 = Space::su2([(0, 1), (1, 2), (2, 1)]).unwrap();
        let left = Space::su2([(0, 1), (1, 2)]).unwrap();
        let right = Space::su2([(1, 1), (2, 2)]).unwrap();
        let lhs_parent =
            Tensor::from_block_fn(&runtime, [&left], [&common0, &common1], |key, indices| {
                let value = oracle_value(0, key, indices);
                Complex64::new(value, value / 3.0 + 0.5)
            })
            .unwrap();
        let rhs_parent =
            Tensor::from_block_fn(&runtime, [&right], [&common0, &common1], |key, indices| {
                let value = oracle_value(1, key, indices);
                Complex64::new(value, -value / 3.0 - 0.25)
            })
            .unwrap();
        let lhs = lhs_parent.adjoint().unwrap();
        let rhs = rhs_parent.adjoint().unwrap();
        let source = lhs.parent_body_for_lowering().space.structure();
        let projected = source
            .sector_structure()
            .sorted_indices()
            .iter()
            .map(|&index| cat_logical_block_key(source.block(index).unwrap().key()).unwrap())
            .collect::<Vec<_>>();
        assert!(
            projected.windows(2).any(|pair| pair[0] > pair[1]),
            "fixture must exercise a non-sorted swapped-key projection"
        );

        assert_lazy_cat_matches_eager(&lhs, &rhs, CatSide::Domain);
    }

    #[test]
    fn lazy_adjoint_cat_declines_reordered_complete_coupled_regions() {
        // What: a valid complete region whose parent tree extents use a
        // different order takes the eager oracle path instead of misrouting
        // whole-matrix columns between fusion-tree subblocks.
        let runtime = Runtime::builder().build().unwrap();
        let common0 = Space::su2([(0, 2), (1, 1), (2, 1)]).unwrap();
        let common1 = Space::su2([(0, 1), (1, 2), (2, 1)]).unwrap();
        let left = Space::su2([(0, 1), (1, 2)]).unwrap();
        let right = Space::su2([(1, 1), (2, 2)]).unwrap();
        let canonical_parent =
            Tensor::from_block_fn(&runtime, [&left], [&common0, &common1], |key, indices| {
                let value = oracle_value(0, key, indices);
                Complex64::new(value, value / 5.0 + 0.75)
            })
            .unwrap();
        let reordered_parent = reordered_complete_su2_parent(&canonical_parent);
        let canonical_regions = canonical_parent
            .ordinary_body()
            .space
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap();
        let reordered_regions = reordered_parent
            .ordinary_body()
            .space
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap();
        assert_eq!(canonical_regions.len(), reordered_regions.len());
        assert_eq!(
            canonical_regions
                .iter()
                .map(CoupledSectorRegion::coupled)
                .collect::<Vec<_>>(),
            reordered_regions
                .iter()
                .map(CoupledSectorRegion::coupled)
                .collect::<Vec<_>>()
        );
        assert!(
            reordered_regions
                .windows(2)
                .all(|pair| pair[0].coupled() < pair[1].coupled()),
            "fixture must preserve monotone coupled-sector mapping"
        );
        assert!(
            canonical_regions
                .iter()
                .zip(reordered_regions.iter())
                .any(|(canonical, reordered)| {
                    canonical.coupled() == reordered.coupled()
                        && canonical.col_trees() != reordered.col_trees()
                }),
            "fixture must retain a complete region with reordered domain-tree extents"
        );

        let lhs = reordered_parent.adjoint().unwrap();
        let rhs =
            Tensor::from_block_fn(&runtime, [&common0, &common1], [&right], |key, indices| {
                let value = oracle_value(1, key, indices);
                Complex64::new(value, -value / 7.0 - 0.5)
            })
            .unwrap();
        let lhs_metadata = lhs.metadata();
        let rhs_metadata = rhs.metadata();
        let (axis, homspace) = cat_homspace(
            lhs_metadata.codomain(),
            lhs_metadata.domain(),
            rhs_metadata.codomain(),
            rhs_metadata.domain(),
            CatSide::Domain,
        )
        .unwrap();
        assert!(
            CatDescriptor::try_new_oriented(
                &lhs_metadata,
                &rhs_metadata,
                axis,
                CatSide::Domain,
                homspace
            )
            .unwrap()
            .is_none(),
            "reordered unchanged-side tree extents must decline the whole-region copy"
        );
        assert_eq!(lhs.adjoint_body_builds(), 0);

        let eager_lhs = Tensor::owned(
            lhs.rt.clone(),
            Arc::clone(&lhs.parent_body_for_lowering().space),
            Arc::clone(&lhs.parent_body_for_lowering().data),
        )
        .adjoint()
        .unwrap();
        let expected_error = eager_lhs.materialized_tensor().unwrap_err();
        let actual_error = lhs.catdomain(&rhs).unwrap_err();

        assert_eq!(actual_error, expected_error);
        assert_eq!(lhs.adjoint_body_builds(), 0);
    }

    #[test]
    fn lazy_adjoint_cat_preserves_fermionic_and_product_tree_pairs() {
        // What: odd fZ2 sectors and nested-product fusion-tree identities use
        // the same borrowed adjoint route as bosonic multiplicity-free rules.
        let runtime = Runtime::builder().build().unwrap();
        let f_common0 = Space::fz2([(0, 2), (1, 1)]).unwrap();
        let f_common1 = Space::fz2([(0, 1), (1, 2)]).unwrap();
        let f_left = Space::fz2([(0, 1), (1, 2)]).unwrap();
        let f_right = Space::fz2([(1, 3)]).unwrap();
        let f_lhs_parent = Tensor::from_block_fn(
            &runtime,
            [&f_left],
            [&f_common0, &f_common1],
            |key, indices| oracle_value(0, key, indices),
        )
        .unwrap();
        let f_rhs_parent = Tensor::from_block_fn(
            &runtime,
            [&f_right],
            [&f_common0, &f_common1],
            |key, indices| oracle_value(1, key, indices),
        )
        .unwrap();
        assert_lazy_cat_matches_eager(
            &f_lhs_parent.adjoint().unwrap(),
            &f_rhs_parent.adjoint().unwrap(),
            CatSide::Domain,
        );

        let p_common0 = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, -1, 1), 1)]).unwrap();
        let p_common1 = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 2)]).unwrap();
        let p_left = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, -1, 1), 2)]).unwrap();
        let p_right = Space::fz2_u1_su2([((1, -1, 1), 1), ((1, 1, 1), 2)]).unwrap();
        let p_lhs_parent = Tensor::from_block_fn(
            &runtime,
            [&p_common0, &p_common1],
            [&p_left],
            |key, indices| oracle_value(0, key, indices),
        )
        .unwrap();
        let p_rhs_parent = Tensor::from_block_fn(
            &runtime,
            [&p_common0, &p_common1],
            [&p_right],
            |key, indices| oracle_value(1, key, indices),
        )
        .unwrap();
        assert_lazy_cat_matches_eager(
            &p_lhs_parent.adjoint().unwrap(),
            &p_rhs_parent.adjoint().unwrap(),
            CatSide::Codomain,
        );
    }

    #[test]
    fn lazy_adjoint_cat_supports_empty_unchanged_products() {
        // What: rank-one adjoint vectors concatenate when the unchanged
        // codomain or domain product is empty.
        let runtime = Runtime::builder().build().unwrap();
        let left = Space::u1([(0, 1)]);
        let right = Space::u1([(0, 2)]);
        let domain_lhs_parent =
            Tensor::from_block_fn(&runtime, [&left], std::iter::empty(), |_, indices| {
                (indices[0] + 1) as f64
            })
            .unwrap();
        let domain_rhs_parent =
            Tensor::from_block_fn(&runtime, [&right], std::iter::empty(), |_, indices| {
                (indices[0] + 2) as f64
            })
            .unwrap();
        let domain = assert_lazy_cat_matches_eager(
            &domain_lhs_parent.adjoint().unwrap(),
            &domain_rhs_parent.adjoint().unwrap(),
            CatSide::Domain,
        );
        assert_eq!((domain.codomain_rank(), domain.domain_rank()), (0, 1));

        let codomain_lhs_parent =
            Tensor::from_block_fn(&runtime, std::iter::empty(), [&left], |_, indices| {
                (indices[0] + 1) as f64
            })
            .unwrap();
        let codomain_rhs_parent =
            Tensor::from_block_fn(&runtime, std::iter::empty(), [&right], |_, indices| {
                (indices[0] + 2) as f64
            })
            .unwrap();
        let codomain = assert_lazy_cat_matches_eager(
            &codomain_lhs_parent.adjoint().unwrap(),
            &codomain_rhs_parent.adjoint().unwrap(),
            CatSide::Codomain,
        );
        assert_eq!((codomain.codomain_rank(), codomain.domain_rank()), (1, 0));
    }

    #[test]
    fn lazy_adjoint_cat_rejects_invalid_contracts_before_building_layouts() {
        // What: lazy metadata preserves rule/runtime/rank/unchanged-space/dual
        // validation precedence without constructing either logical input.
        let runtime = Runtime::builder().build().unwrap();
        let other_runtime = Runtime::builder().build().unwrap();
        let codomain = Space::u1([(0, 2)]);
        let bad_codomain = Space::u1([(0, 3)]);
        let left = Space::u1([(0, 1)]);
        let right = Space::u1([(0, 2)]);
        let lhs = Tensor::zeros(&runtime, Dtype::F64, [&left], [&codomain])
            .unwrap()
            .adjoint()
            .unwrap();
        let bad_rank = Tensor::zeros(&runtime, Dtype::F64, [&right, &right], [&codomain])
            .unwrap()
            .adjoint()
            .unwrap();
        let bad_runtime = Tensor::zeros(&other_runtime, Dtype::F64, [&right], [&codomain])
            .unwrap()
            .adjoint()
            .unwrap();
        let other_rule = Space::z2([(0, 1)]);
        let bad_provider = Tensor::zeros(&runtime, Dtype::F64, [&other_rule], [&other_rule])
            .unwrap()
            .adjoint()
            .unwrap();
        let bad_unchanged = Tensor::zeros(&runtime, Dtype::F64, [&right], [&bad_codomain])
            .unwrap()
            .adjoint()
            .unwrap();
        let bad_dual = Tensor::zeros(&runtime, Dtype::F64, [&right.dual()], [&codomain])
            .unwrap()
            .adjoint()
            .unwrap();

        CAT_RESULT_LAYOUT_BUILDS.with(|observation| observation.set(Some(0)));
        assert!(lhs.catdomain(&bad_rank).is_err());
        assert!(matches!(
            lhs.catdomain(&bad_runtime),
            Err(Error::RuntimeMismatch)
        ));
        assert!(matches!(
            lhs.catdomain(&bad_provider),
            Err(Error::RuleMismatch)
        ));
        assert!(lhs.catdomain(&bad_unchanged).is_err());
        assert!(lhs.catdomain(&bad_dual).is_err());
        assert_eq!(
            CAT_RESULT_LAYOUT_BUILDS.with(|observation| observation.replace(None)),
            Some(0)
        );
        for tensor in [
            &lhs,
            &bad_rank,
            &bad_runtime,
            &bad_provider,
            &bad_unchanged,
            &bad_dual,
        ] {
            assert_eq!(tensor.adjoint_body_builds(), 0);
        }
    }

    #[test]
    fn cat_validates_every_contract_before_result_layout_build() {
        let runtime = Runtime::builder().build().unwrap();
        let other_runtime = Runtime::builder().build().unwrap();
        let codomain = Space::u1([(0, 2)]);
        let left = Space::u1([(0, 1)]);
        let right = Space::u1([(0, 2)]);
        let lhs = Tensor::zeros(&runtime, Dtype::F64, [&codomain], [&left]).unwrap();
        let rhs = Tensor::zeros(&runtime, Dtype::F64, [&codomain], [&right]).unwrap();
        let bad_rank = Tensor::zeros(&runtime, Dtype::F64, [&codomain], [&right, &right]).unwrap();
        let bad_runtime = Tensor::zeros(&other_runtime, Dtype::F64, [&codomain], [&right]).unwrap();
        let other_rule = Space::z2([(0, 1)]);
        let bad_provider =
            Tensor::zeros(&runtime, Dtype::F64, [&other_rule], [&other_rule]).unwrap();
        let bad_codomain = Space::u1([(0, 3)]);
        let bad_unchanged = Tensor::zeros(&runtime, Dtype::F64, [&bad_codomain], [&right]).unwrap();
        let dual = right.dual();
        let bad_dual = Tensor::zeros(&runtime, Dtype::F64, [&codomain], [&dual]).unwrap();

        CAT_RESULT_LAYOUT_BUILDS.with(|observation| observation.set(Some(0)));
        assert!(lhs.catdomain(&bad_rank).is_err());
        assert!(matches!(
            lhs.catdomain(&bad_runtime),
            Err(Error::RuntimeMismatch)
        ));
        assert!(matches!(
            lhs.catdomain(&bad_provider),
            Err(Error::RuleMismatch)
        ));
        assert!(lhs.catdomain(&bad_unchanged).is_err());
        assert!(lhs.catdomain(&bad_dual).is_err());
        assert_eq!(
            CAT_RESULT_LAYOUT_BUILDS.with(|observation| observation.get()),
            Some(0)
        );
        lhs.catdomain(&rhs).unwrap();
        assert_eq!(
            CAT_RESULT_LAYOUT_BUILDS.with(|observation| observation.replace(None)),
            Some(1)
        );
    }
}

#[cfg(test)]
mod compose_direct_tests {
    use super::*;

    #[test]
    fn direct_fermionic_odd_dual_contract_matches_hand_oracle() {
        // What: direct tensorcontract keeps the TensorKit odd dual-leg twist.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::fz2([(0, 1), (1, 1)]).unwrap();
        let lhs = Tensor::from_block_fn(&runtime, [&space], [&space.dual()], |key, _| {
            let BlockKey::FusionTree(key) = key else {
                return 0.0;
            };
            if key.codomain_uncoupled()[0].id() == 0 {
                5.0
            } else {
                2.0
            }
        })
        .unwrap();
        let rhs = Tensor::from_block_fn(&runtime, [&space.dual()], [&space], |key, _| {
            let BlockKey::FusionTree(key) = key else {
                return 0.0;
            };
            if key.codomain_uncoupled()[0].id() == 0 {
                7.0
            } else {
                3.0
            }
        })
        .unwrap();

        let actual = lhs.contract(&rhs, &[1], &[0]).unwrap();

        assert_eq!(actual.data(), [35.0, -6.0]);
    }

    #[test]
    fn direct_compose_does_not_invoke_tensor_twist() {
        // What: fermionic map composition takes the direct `mul!` route,
        // while the retained integration oracle exercises twist-then-contract.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::fz2([(0, 24), (1, 24)]).unwrap();
        let lhs = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space.dual()], 353_524)
            .unwrap();
        let rhs = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_624)
            .unwrap();

        TWIST_CALLS.with(|observation| observation.set(Some(0)));
        let composed = lhs.compose(&rhs).unwrap();
        let calls = TWIST_CALLS.with(|observation| observation.replace(None));

        assert_eq!(calls, Some(0));
        assert_eq!(composed.codomain_rank(), 1);
        assert_eq!(composed.domain_rank(), 1);
    }
}

#[cfg(test)]
mod ordered_contract_route_tests {
    use super::*;

    #[test]
    fn contracted_axis_validation_handles_high_rank_and_late_duplicates() {
        let valid = (0..64).rev().collect::<Vec<_>>();
        validate_contracted_axes(&valid, 64).unwrap();
        let mut duplicate = valid;
        duplicate[63] = duplicate[0];
        // What: validation remains correct after the inline common-rank mark
        // storage spills to its linear-time high-rank fallback.
        assert!(validate_contracted_axes(&duplicate, 64).is_err());
        assert!(validate_contracted_axes(&[64], 64).is_err());
    }

    #[test]
    fn ordinary_multiplicity_free_ordered_contract_uses_fused_plan_route() {
        // What: a crossed SU2 pAB is handed to the contraction plan instead of
        // returning a default-order owned tensor to a second public permute.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap();
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space, &space],
            [&space, &space],
            224_501,
        )
        .unwrap();
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space, &space],
            [&space, &space],
            224_502,
        )
        .unwrap();

        ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.set(Some(false)));
        let _ = lhs
            .contract_ordered(&rhs, &[3, 2], &[0, 1], &[2, 0, 3, 1])
            .unwrap();
        let observed = ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.replace(None));

        assert_eq!(observed, Some(true));
    }

    #[test]
    fn compact_diagonal_ordered_contract_keeps_sequential_fallback() {
        // What: compact diagonal complexity dispatch is not bypassed by the
        // new host fusion route.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 2), (1, 2)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_503).unwrap();
        let diagonal = source.svd_compact().unwrap().1;

        ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.set(Some(false)));
        let _ = diagonal
            .contract_ordered(&diagonal, &[1], &[0], &[1, 0])
            .unwrap();
        let observed = ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.replace(None));

        assert_eq!(observed, Some(false));
    }

    #[test]
    fn u1_dense_diagonal_ordered_contract_matches_sequential_oracle() {
        // What: pAB is folded into the single output transform after compact
        // diagonal scaling.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 2), (1, 2)]);
        let dense =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_506).unwrap();
        let diagonal = dense.svd_compact().unwrap().1;

        let actual = dense
            .contract_ordered(&diagonal, &[1], &[0], &[1, 0])
            .unwrap();
        let expected = dense
            .contract(&diagonal, &[1], &[0])
            .unwrap()
            .permute(&[1], &[0])
            .unwrap();

        assert_eq!(actual.data().len(), expected.data().len());
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1.0e-11);
        }

        let actual = diagonal
            .contract_ordered(&dense, &[1], &[0], &[1, 0])
            .unwrap();
        let expected = diagonal
            .contract(&dense, &[1], &[0])
            .unwrap()
            .permute(&[1], &[0])
            .unwrap();

        assert_eq!(actual.data().len(), expected.data().len());
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
    }

    #[test]
    fn fz2_dense_diagonal_ordered_contract_matches_sequential_oracle() {
        // What: folded pAB preserves the existing fermionic contract signs.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::fz2([(0, 2), (1, 2)]).unwrap();
        let dense =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_507).unwrap();
        let diagonal = dense.svd_compact().unwrap().1;

        let actual = dense
            .contract_ordered(&diagonal, &[1], &[0], &[1, 0])
            .unwrap();
        let expected = dense
            .contract(&diagonal, &[1], &[0])
            .unwrap()
            .permute(&[1], &[0])
            .unwrap();

        assert_eq!(actual.data().len(), expected.data().len());
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1.0e-11);
        }

        let actual = diagonal
            .contract_ordered(&dense, &[1], &[0], &[1, 0])
            .unwrap();
        let expected = diagonal
            .contract(&dense, &[1], &[0])
            .unwrap()
            .permute(&[1], &[0])
            .unwrap();

        assert_eq!(actual.data().len(), expected.data().len());
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
    }

    #[test]
    fn partial_trace_builds_selected_result_layout_once() {
        // What: nested-product partial trace enters the selected-result layout
        // builder once and returns the expected rank-zero tensor.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, -1, 1), 1), ((1, 1, 1), 1)]).unwrap();
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_506).unwrap();

        SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.set(Some(0)));
        let traced = tensor.trace_pairs(&[(0, 1)]).unwrap();
        let builds = SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.replace(None));

        assert_eq!(builds, Some(1));
        assert_eq!(traced.rank(), 0);
    }
}
