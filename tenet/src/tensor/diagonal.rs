use num_complex::Complex64;
use tenet_core::{BlockKey, SectorId};
use tenet_matrixalgebra::{FactorScalar, SectorSpectrum};
use tenet_tensors::DynamicFusionMapSpace;

use super::{Data, DiagonalData, Error, UserScalar};

fn map_spectra<I: Copy, O>(
    spectra: &[SectorSpectrum<I>],
    map: impl Fn(I) -> O,
) -> Vec<SectorSpectrum<O>> {
    spectra
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().copied().map(&map).collect(),
        })
        .collect()
}

fn zip_spectra<L: Copy, R: Copy, O>(
    lhs: &[SectorSpectrum<L>],
    rhs: &[SectorSpectrum<R>],
    map: impl Fn(L, R) -> O,
) -> Option<Vec<SectorSpectrum<O>>> {
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
                    .map(|(lhs, rhs)| map(lhs, rhs))
                    .collect(),
            })
        })
        .collect()
}

impl DiagonalData {
    pub(super) fn conjugated_complex(&self) -> Result<Self, Error> {
        match self {
            Self::C64(spectra) => Ok(Self::C64(map_spectra(spectra, |value| value.conj()))),
            Self::RealF64(_) | Self::RealC64(_) => Err(Error::DtypeMismatch),
        }
    }

    pub(super) fn scaled_c64(&self, factor: Complex64) -> Result<Self, Error> {
        match self {
            Self::RealF64(_) => Err(Error::DtypeMismatch),
            Self::RealC64(spectra) => Ok(Self::C64(map_spectra(spectra, |value| value * factor))),
            Self::C64(spectra) => Ok(Self::C64(map_spectra(spectra, |value| value * factor))),
        }
    }

    pub(super) fn to_c64_storage(&self) -> Self {
        match self {
            Self::RealF64(spectra) => Self::RealC64(spectra.clone()),
            Self::RealC64(_) | Self::C64(_) => self.clone(),
        }
    }

    pub(super) fn sectors_all(&self, predicate: impl Fn(SectorId) -> bool) -> bool {
        match self {
            Self::RealF64(spectra) | Self::RealC64(spectra) => {
                spectra.iter().all(|entry| predicate(entry.sector))
            }
            Self::C64(spectra) => spectra.iter().all(|entry| predicate(entry.sector)),
        }
    }

    pub(super) fn scaled_by_sector(&self, factor: impl Fn(SectorId) -> f64) -> Self {
        match self {
            Self::RealF64(spectra) => Self::RealF64(
                spectra
                    .iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| value * factor(entry.sector))
                            .collect(),
                    })
                    .collect(),
            ),
            Self::RealC64(spectra) => Self::RealC64(
                spectra
                    .iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| value * factor(entry.sector))
                            .collect(),
                    })
                    .collect(),
            ),
            Self::C64(spectra) => Self::C64(
                spectra
                    .iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| value * factor(entry.sector))
                            .collect(),
                    })
                    .collect(),
            ),
        }
    }

    pub(super) fn axpby_real(&self, rhs: &Self, alpha: f64, beta: f64) -> Option<Self> {
        let real = |lhs, rhs| lhs * alpha + rhs * beta;
        match (self, rhs) {
            (Self::RealF64(lhs), Self::RealF64(rhs)) => {
                zip_spectra(lhs, rhs, real).map(Self::RealF64)
            }
            (Self::RealC64(lhs), Self::RealC64(rhs)) => {
                zip_spectra(lhs, rhs, real).map(Self::RealC64)
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                zip_spectra(lhs, rhs, |lhs, rhs| lhs * alpha + rhs * beta).map(Self::C64)
            }
            (Self::RealC64(lhs), Self::C64(rhs)) => {
                zip_spectra(lhs, rhs, |lhs, rhs| lhs * alpha + rhs * beta).map(Self::C64)
            }
            (Self::C64(lhs), Self::RealC64(rhs)) => {
                zip_spectra(lhs, rhs, |lhs, rhs| lhs * alpha + rhs * beta).map(Self::C64)
            }
            _ => None,
        }
    }

    pub(super) fn axpby_c64(&self, rhs: &Self, alpha: Complex64, beta: Complex64) -> Option<Self> {
        let real = |value| Complex64::new(value, 0.0);
        match (self, rhs) {
            (Self::RealC64(lhs), Self::RealC64(rhs)) => {
                zip_spectra(lhs, rhs, |lhs, rhs| real(lhs) * alpha + real(rhs) * beta)
                    .map(Self::C64)
            }
            (Self::RealC64(lhs), Self::C64(rhs)) => {
                zip_spectra(lhs, rhs, |lhs, rhs| real(lhs) * alpha + rhs * beta).map(Self::C64)
            }
            (Self::C64(lhs), Self::RealC64(rhs)) => {
                zip_spectra(lhs, rhs, |lhs, rhs| lhs * alpha + real(rhs) * beta).map(Self::C64)
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                zip_spectra(lhs, rhs, |lhs, rhs| lhs * alpha + rhs * beta).map(Self::C64)
            }
            _ => None,
        }
    }
}

/// Walks the logical dense diagonal without building it. Why not route through
/// `coupled_data`: binary operations already own a dense result, so a second
/// dense allocation for the compact input can never contribute useful data.
fn visit_entries<V: Copy>(
    space: &DynamicFusionMapSpace,
    spectrum: &[SectorSpectrum<V>],
    mut visit: impl FnMut(usize, SectorId, V),
) -> Result<(), Error> {
    debug_assert!(spectrum
        .windows(2)
        .all(|entries| entries[0].sector < entries[1].sector));
    let structure = space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        // Why not build a per-call HashMap: compact spectra are canonicalized
        // once at construction, so binary search keeps replay allocation-free.
        let Ok(spectrum_index) = spectrum.binary_search_by_key(&sector, |entry| entry.sector)
        else {
            continue;
        };
        let entry = &spectrum[spectrum_index];
        let strides = block.strides();
        let count = block.shape()[0].min(block.shape()[1]);
        for (diagonal_index, &value) in entry.values[..count].iter().enumerate() {
            visit(
                block.offset() + diagonal_index * (strides[0] + strides[1]),
                sector,
                value,
            );
        }
    }
    Ok(())
}

fn axpy_into<D: UserScalar, V: Copy>(
    space: &DynamicFusionMapSpace,
    dst: &mut [D],
    spectrum: &[SectorSpectrum<V>],
    alpha: D,
    map: impl Fn(V) -> D,
) -> Result<(), Error> {
    visit_entries(space, spectrum, |position, _, value| {
        dst[position] = dst[position] + alpha * map(value);
    })
}

pub(super) fn axpby_dense_real(
    space: &DynamicFusionMapSpace,
    dense: &Data,
    diagonal: &DiagonalData,
    dense_factor: f64,
    diagonal_factor: f64,
) -> Result<Data, Error> {
    match (dense, diagonal) {
        (Data::F64(dense), DiagonalData::RealF64(spectrum)) => {
            let mut out = dense
                .iter()
                .map(|&value| value * dense_factor)
                .collect::<Vec<_>>();
            axpy_into(space, &mut out, spectrum, diagonal_factor, |value| value)?;
            Ok(Data::F64(out))
        }
        (Data::C64(dense), DiagonalData::RealC64(spectrum)) => {
            let mut out = dense
                .iter()
                .map(|&value| value * dense_factor)
                .collect::<Vec<_>>();
            axpy_into(
                space,
                &mut out,
                spectrum,
                Complex64::new(diagonal_factor, 0.0),
                |value| Complex64::new(value, 0.0),
            )?;
            Ok(Data::C64(out))
        }
        (Data::C64(dense), DiagonalData::C64(spectrum)) => {
            let mut out = dense
                .iter()
                .map(|&value| value * dense_factor)
                .collect::<Vec<_>>();
            axpy_into(
                space,
                &mut out,
                spectrum,
                Complex64::new(diagonal_factor, 0.0),
                |value| value,
            )?;
            Ok(Data::C64(out))
        }
        _ => Err(Error::DtypeMismatch),
    }
}

pub(super) fn axpby_dense_c64(
    space: &DynamicFusionMapSpace,
    dense: &Data,
    diagonal: &DiagonalData,
    dense_factor: Complex64,
    diagonal_factor: Complex64,
) -> Result<Data, Error> {
    let Data::C64(dense) = dense else {
        return Err(Error::DtypeMismatch);
    };
    let mut out = dense
        .iter()
        .map(|&value| value * dense_factor)
        .collect::<Vec<_>>();
    match diagonal {
        DiagonalData::RealC64(spectrum) => {
            axpy_into(space, &mut out, spectrum, diagonal_factor, |value| {
                Complex64::new(value, 0.0)
            })?
        }
        DiagonalData::C64(spectrum) => {
            axpy_into(space, &mut out, spectrum, diagonal_factor, |value| value)?
        }
        DiagonalData::RealF64(_) => return Err(Error::DtypeMismatch),
    }
    Ok(Data::C64(out))
}

pub(super) fn dense_inner_with_weight<D, V>(
    space: &DynamicFusionMapSpace,
    spectrum: &[SectorSpectrum<V>],
    dense: &[D],
    diagonal_first: bool,
    weight: impl Fn(SectorId) -> f64,
    map: impl Fn(V) -> D,
) -> Result<Complex64, Error>
where
    D: UserScalar,
    V: Copy,
{
    let mut total = Complex64::new(0.0, 0.0);
    visit_entries(space, spectrum, |position, sector, value| {
        let diagonal = map(value);
        let product = if diagonal_first {
            FactorScalar::adjoint(diagonal) * dense[position]
        } else {
            FactorScalar::adjoint(dense[position]) * diagonal
        };
        total += product.widen_complex() * weight(sector);
    })?;
    Ok(total)
}

pub(super) fn compact_inner_with_weight<D, L, Rhs>(
    lhs: &[SectorSpectrum<L>],
    rhs: &[SectorSpectrum<Rhs>],
    weight: impl Fn(SectorId) -> f64,
    map_lhs: impl Fn(L) -> D,
    map_rhs: impl Fn(Rhs) -> D,
) -> Option<Complex64>
where
    D: UserScalar,
    L: Copy,
    Rhs: Copy,
{
    if lhs.len() != rhs.len() {
        return None;
    }
    let mut total = Complex64::new(0.0, 0.0);
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        if lhs.sector != rhs.sector || lhs.values.len() != rhs.values.len() {
            return None;
        }
        let mut partial = D::from_real(0.0);
        for (&lhs, &rhs) in lhs.values.iter().zip(&rhs.values) {
            partial = partial + FactorScalar::adjoint(map_lhs(lhs)) * map_rhs(rhs);
        }
        total += partial.widen_complex() * weight(lhs.sector);
    }
    Some(total)
}
