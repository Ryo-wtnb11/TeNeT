//! Spectrum truncation policies for the fusion-tensor factorizations.
//!
//! Design (informed by MatrixAlgebraKit / the legacy `TruncationStrategy`, but
//! intentionally narrower): every policy here is a magnitude-monotone rule
//! over per-sector spectra that are non-negative and descending, so a
//! selection is always a per-sector *prefix count*. That keeps the host-side
//! decision a pure scalar computation and keeps the device-side application a
//! leading-columns/rows gather. Rules that can keep non-prefix index sets
//! (arbitrary filters, signed eigenvalue windows) get their own layer when a
//! decomposition needs them.
//!
//! All budgets are weighted by the coupled sector's quantum dimension: one
//! kept value of an SU(2) spin-j sector consumes `2j + 1` of a rank budget
//! and contributes `(2j + 1) * value^2` to the 2-norm.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

/// Truncation policy over per-sector descending spectra.
#[derive(Clone, Debug, PartialEq)]
pub enum Truncation {
    /// Keep everything.
    Full,
    /// Keep the largest values while the quantum-dimension-weighted total
    /// dimension stays at or below the bound.
    Rank(usize),
    /// Discard values below `max(atol, rtol * norm)`, where `norm` is the
    /// weighted 2-norm of the full spectrum.
    #[non_exhaustive]
    Tolerance { atol: f64, rtol: f64 },
    /// Discard values below `max(atol, rtol * normInf)`, where `normInf` is the
    /// unweighted maximum value. This matches TensorKit `trunctol(..., p=Inf)`.
    #[non_exhaustive]
    ToleranceInf { atol: f64, rtol: f64 },
    /// Discard the smallest values while the weighted 2-norm of everything
    /// discarded stays at or below `rtol * norm`.
    #[non_exhaustive]
    DiscardWeight { rtol: f64 },
    /// Keep a value only if every component keeps it. Prefix rules compose to
    /// a prefix rule, so this is the per-sector minimum of the kept counts.
    All(Vec<Truncation>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TruncationError {
    InvalidPolicy { message: &'static str },
    InvalidSpectrum { message: &'static str },
}

impl fmt::Display for TruncationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy { message } => write!(f, "invalid truncation policy: {message}"),
            Self::InvalidSpectrum { message } => {
                write!(f, "invalid truncation spectrum: {message}")
            }
        }
    }
}

impl std::error::Error for TruncationError {}

impl From<TruncationError> for tenet_tensors::OperationError {
    fn from(_: TruncationError) -> Self {
        Self::InvalidArgument {
            message: "invalid truncation input",
        }
    }
}

impl Truncation {
    /// Keep at most `rank` weighted dimensions.
    pub fn rank(rank: usize) -> Self {
        Self::Rank(rank)
    }

    /// Discard values below the absolute cutoff.
    pub fn absolute_cutoff(atol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            atol,
            "tolerance absolute cutoff must be finite and non-negative",
        )?;
        Ok(Self::Tolerance { atol, rtol: 0.0 })
    }

    /// Discard values below `rtol` times the weighted 2-norm.
    pub fn relative_cutoff(rtol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            rtol,
            "tolerance relative cutoff must be finite and non-negative",
        )?;
        Ok(Self::Tolerance { atol: 0.0, rtol })
    }

    /// Discard values below `rtol` times the largest value.
    pub fn relative_inf_cutoff(rtol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            rtol,
            "infinity-norm relative cutoff must be finite and non-negative",
        )?;
        Ok(Self::ToleranceInf { atol: 0.0, rtol })
    }

    /// Bound the relative truncation error (weighted 2-norm of the discarded
    /// tail) by `rtol`.
    pub fn relative_error(rtol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            rtol,
            "discard-weight tolerance must be finite and non-negative",
        )?;
        Ok(Self::DiscardWeight { rtol })
    }

    /// Intersects two policies (both must keep a value).
    pub fn and(self, other: Truncation) -> Self {
        match (self, other) {
            (Truncation::Full, other) => other,
            (this, Truncation::Full) => this,
            (Truncation::All(mut components), Truncation::All(others)) => {
                components.extend(others);
                Truncation::All(components)
            }
            (Truncation::All(mut components), other) => {
                components.push(other);
                Truncation::All(components)
            }
            (this, Truncation::All(mut components)) => {
                components.insert(0, this);
                Truncation::All(components)
            }
            (this, other) => Truncation::All(vec![this, other]),
        }
    }
}

/// One coupled sector's spectrum offered to the selection: its quantum
/// dimension and its values, non-negative and descending.
#[derive(Clone, Copy, Debug)]
pub struct WeightedSpectrum<'a> {
    pub weight: f64,
    pub values: &'a [f64],
}

/// The outcome of a truncation decision: per-sector kept prefix lengths and
/// the weighted 2-norm of everything discarded.
#[derive(Clone, Debug, PartialEq)]
pub struct TruncationDecision {
    pub kept: Vec<usize>,
    pub error: f64,
}

/// Selects the kept prefix per sector for `truncation` over `spectra`.
///
/// Host-side scalar computation by design: spectra are small compared to the
/// tensors, so the decision never needs to touch device data.
pub fn select_truncation(
    spectra: &[WeightedSpectrum<'_>],
    truncation: &Truncation,
) -> Result<TruncationDecision, TruncationError> {
    validate_truncation(truncation)?;
    validate_spectra(spectra)?;
    let kept = kept_counts(spectra, truncation);
    let error = discarded_norm(spectra, &kept);
    Ok(TruncationDecision { kept, error })
}

fn validate_nonnegative_finite(value: f64, message: &'static str) -> Result<(), TruncationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(TruncationError::InvalidPolicy { message })
    }
}

fn validate_truncation(truncation: &Truncation) -> Result<(), TruncationError> {
    match truncation {
        Truncation::Full | Truncation::Rank(_) => Ok(()),
        Truncation::Tolerance { atol, rtol } => {
            validate_nonnegative_finite(
                *atol,
                "tolerance absolute cutoff must be finite and non-negative",
            )?;
            validate_nonnegative_finite(
                *rtol,
                "tolerance relative cutoff must be finite and non-negative",
            )
        }
        Truncation::ToleranceInf { atol, rtol } => {
            validate_nonnegative_finite(
                *atol,
                "infinity-norm absolute cutoff must be finite and non-negative",
            )?;
            validate_nonnegative_finite(
                *rtol,
                "infinity-norm relative cutoff must be finite and non-negative",
            )
        }
        Truncation::DiscardWeight { rtol } => validate_nonnegative_finite(
            *rtol,
            "discard-weight tolerance must be finite and non-negative",
        ),
        Truncation::All(components) => {
            for component in components {
                validate_truncation(component)?;
            }
            Ok(())
        }
    }
}

fn validate_spectra(spectra: &[WeightedSpectrum<'_>]) -> Result<(), TruncationError> {
    for spectrum in spectra {
        if !spectrum.weight.is_finite() || spectrum.weight <= 0.0 {
            return Err(TruncationError::InvalidSpectrum {
                message: "sector weight must be finite and positive",
            });
        }
        for &value in spectrum.values {
            if !value.is_finite() || value < 0.0 {
                return Err(TruncationError::InvalidSpectrum {
                    message: "spectrum values must be finite and non-negative",
                });
            }
        }
        for pair in spectrum.values.windows(2) {
            if pair[0] < pair[1] {
                return Err(TruncationError::InvalidSpectrum {
                    message: "spectrum values must be descending",
                });
            }
        }
    }
    Ok(())
}

fn kept_counts(spectra: &[WeightedSpectrum<'_>], truncation: &Truncation) -> Vec<usize> {
    match truncation {
        Truncation::Full => spectra
            .iter()
            .map(|spectrum| spectrum.values.len())
            .collect(),
        Truncation::Rank(rank) => {
            let mut order = descending_candidates(spectra);
            let mut kept = vec![0usize; spectra.len()];
            let mut used = 0.0;
            let budget = *rank as f64;
            for (sector, index) in order.drain(..) {
                let weight = spectra[sector].weight;
                if used + weight > budget + 1e-12 {
                    break;
                }
                debug_assert_eq!(index, kept[sector]);
                used += weight;
                kept[sector] += 1;
            }
            kept
        }
        Truncation::Tolerance { atol, rtol } => {
            let threshold = atol.max(rtol * full_norm(spectra));
            spectra
                .iter()
                .map(|spectrum| {
                    spectrum
                        .values
                        .iter()
                        .take_while(|&&value| value >= threshold)
                        .count()
                })
                .collect()
        }
        Truncation::ToleranceInf { atol, rtol } => {
            let threshold = atol.max(rtol * full_norm_inf(spectra));
            spectra
                .iter()
                .map(|spectrum| {
                    spectrum
                        .values
                        .iter()
                        .take_while(|&&value| value >= threshold)
                        .count()
                })
                .collect()
        }
        Truncation::DiscardWeight { rtol } => {
            let norm = full_norm(spectra);
            let budget = (rtol * norm) * (rtol * norm);
            let mut kept: Vec<usize> = spectra
                .iter()
                .map(|spectrum| spectrum.values.len())
                .collect();
            let mut discarded = 0.0;
            while let Some(sector) = smallest_tail_candidate(spectra, &kept) {
                let index = kept[sector] - 1;
                let value = spectra[sector].values[index];
                let next = discarded + spectra[sector].weight * value * value;
                if next > budget + 1e-15 {
                    break;
                }
                discarded = next;
                kept[sector] -= 1;
            }
            kept
        }
        Truncation::All(components) => {
            let mut kept: Vec<usize> = spectra
                .iter()
                .map(|spectrum| spectrum.values.len())
                .collect();
            for component in components {
                for (slot, count) in kept.iter_mut().zip(kept_counts(spectra, component)) {
                    *slot = (*slot).min(count);
                }
            }
            kept
        }
    }
}

fn smallest_tail_candidate(spectra: &[WeightedSpectrum<'_>], kept: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (sector, (spectrum, &count)) in spectra.iter().zip(kept).enumerate() {
        if count == 0 {
            continue;
        }
        let value = spectrum.values[count - 1];
        match best {
            None => best = Some((sector, value)),
            Some((best_sector, best_value))
                if value < best_value || (value == best_value && sector < best_sector) =>
            {
                best = Some((sector, value));
            }
            _ => {}
        }
    }
    best.map(|(sector, _)| sector)
}

/// Candidates as `(sector, index)` sorted by descending value; ties keep the
/// parent storage order, matching TensorKit `sortperm(parent(values); rev=true)`.
fn descending_candidates(spectra: &[WeightedSpectrum<'_>]) -> Vec<(usize, usize)> {
    let total = spectra.iter().map(|spectrum| spectrum.values.len()).sum();
    let mut heap = BinaryHeap::with_capacity(spectra.len());
    for (sector, spectrum) in spectra.iter().enumerate() {
        if let Some(&value) = spectrum.values.first() {
            heap.push(DescendingCandidate {
                value,
                sector,
                index: 0,
            });
        }
    }

    let mut candidates = Vec::with_capacity(total);
    while let Some(candidate) = heap.pop() {
        candidates.push((candidate.sector, candidate.index));
        let next_index = candidate.index + 1;
        if let Some(&value) = spectra[candidate.sector].values.get(next_index) {
            heap.push(DescendingCandidate {
                value,
                sector: candidate.sector,
                index: next_index,
            });
        }
    }
    candidates
}

#[derive(Clone, Copy, Debug)]
struct DescendingCandidate {
    value: f64,
    sector: usize,
    index: usize,
}

impl PartialEq for DescendingCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.sector == other.sector && self.index == other.index
    }
}

impl Eq for DescendingCandidate {}

impl PartialOrd for DescendingCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DescendingCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .partial_cmp(&other.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.sector.cmp(&self.sector))
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn full_norm(spectra: &[WeightedSpectrum<'_>]) -> f64 {
    spectra
        .iter()
        .map(|spectrum| {
            spectrum.weight
                * spectrum
                    .values
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
}

fn full_norm_inf(spectra: &[WeightedSpectrum<'_>]) -> f64 {
    spectra
        .iter()
        .flat_map(|spectrum| spectrum.values.iter().copied())
        .fold(0.0, f64::max)
}

fn discarded_norm(spectra: &[WeightedSpectrum<'_>], kept: &[usize]) -> f64 {
    spectra
        .iter()
        .zip(kept)
        .map(|(spectrum, &count)| {
            spectrum.weight
                * spectrum.values[count..]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spectra<'a>(entries: &'a [(f64, Vec<f64>)]) -> Vec<WeightedSpectrum<'a>> {
        entries
            .iter()
            .map(|(weight, values)| WeightedSpectrum {
                weight: *weight,
                values,
            })
            .collect()
    }

    #[test]
    fn rank_budget_is_quantum_dimension_weighted() {
        let entries = [(1.0, vec![5.0, 1.0]), (3.0, vec![4.0, 0.5])];
        let spectra = spectra(&entries);
        // Budget 4: keep 5.0 (weight 1) and 4.0 (weight 3) exactly.
        let decision = select_truncation(&spectra, &Truncation::rank(4)).unwrap();
        assert_eq!(decision.kept, vec![1, 1]);
        // Budget 5: the next candidate (1.0, weight 1) fits.
        let decision = select_truncation(&spectra, &Truncation::rank(5)).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
        // Budget 6: 0.5 has weight 3 and does not fit.
        let decision = select_truncation(&spectra, &Truncation::rank(6)).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
    }

    #[test]
    fn rank_ties_keep_parent_storage_order() {
        let entries = [(1.0, vec![2.0, 1.0]), (1.0, vec![2.0, 1.0])];
        let spectra = spectra(&entries);
        let decision = select_truncation(&spectra, &Truncation::rank(1)).unwrap();
        assert_eq!(decision.kept, vec![1, 0]);

        let decision = select_truncation(&spectra, &Truncation::rank(3)).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
    }

    #[test]
    fn tolerance_thresholds_against_weighted_norm() {
        let entries = [(1.0, vec![4.0, 3.0, 0.1])];
        let spectra = spectra(&entries);
        let truncation = Truncation::absolute_cutoff(1.0).unwrap();
        let decision = select_truncation(&spectra, &truncation).unwrap();
        assert_eq!(decision.kept, vec![2]);
        assert!((decision.error - 0.1).abs() < 1e-12);

        // norm = 5.001..., rtol 0.5 => threshold ~2.5: keeps 4 and 3.
        let truncation = Truncation::relative_cutoff(0.5).unwrap();
        let decision = select_truncation(&spectra, &truncation).unwrap();
        assert_eq!(decision.kept, vec![2]);
    }

    #[test]
    fn tolerance_inf_thresholds_against_unweighted_max() {
        let entries = [(3.0, vec![4.0, 3.0, 0.1]), (1.0, vec![2.5])];
        let spectra = spectra(&entries);
        let truncation = Truncation::relative_inf_cutoff(0.7).unwrap();
        let decision = select_truncation(&spectra, &truncation).unwrap();
        assert_eq!(decision.kept, vec![2, 0]);
    }

    #[test]
    fn discard_weight_bounds_relative_error() {
        let entries = [(2.0, vec![3.0, 1.0, 0.5, 0.5])];
        let spectra = spectra(&entries);
        let norm = full_norm(&spectra);
        let truncation = Truncation::relative_error(0.3).unwrap();
        let decision = select_truncation(&spectra, &truncation).unwrap();
        assert!(decision.error <= 0.3 * norm + 1e-12);
        assert!(decision.kept[0] < 4, "a 30% budget must discard something");
        // Discarding one more value would exceed the budget.
        let mut tighter = decision.kept.clone();
        tighter[0] -= 1;
        assert!(discarded_norm(&spectra, &tighter) > 0.3 * norm);
    }

    #[test]
    fn and_composition_takes_the_stricter_prefix() {
        let entries = [(1.0, vec![4.0, 3.0, 2.0, 1.0])];
        let spectra = spectra(&entries);
        let combined = Truncation::rank(3).and(Truncation::absolute_cutoff(2.5).unwrap());
        let decision = select_truncation(&spectra, &combined).unwrap();
        assert_eq!(decision.kept, vec![2]);

        let combined = Truncation::rank(1).and(Truncation::absolute_cutoff(0.5).unwrap());
        let decision = select_truncation(&spectra, &combined).unwrap();
        assert_eq!(decision.kept, vec![1]);
    }

    #[test]
    fn full_keeps_everything_with_zero_error() {
        let entries = [(1.0, vec![2.0, 1.0]), (2.0, vec![1.5])];
        let spectra = spectra(&entries);
        let decision = select_truncation(&spectra, &Truncation::Full).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
        assert_eq!(decision.error, 0.0);
    }

    #[test]
    fn non_finite_spectrum_returns_typed_error_for_every_policy() {
        let entries = [(1.0, vec![3.0, f64::NAN, 1.0])];
        let spectra = spectra(&entries);
        let policies = [
            Truncation::rank(1),
            Truncation::absolute_cutoff(1.0).unwrap(),
            Truncation::relative_inf_cutoff(0.5).unwrap(),
            Truncation::relative_error(0.1).unwrap(),
            Truncation::rank(2).and(Truncation::absolute_cutoff(0.5).unwrap()),
        ];

        for policy in policies {
            assert!(matches!(
                select_truncation(&spectra, &policy),
                Err(TruncationError::InvalidSpectrum { .. })
            ));
        }
    }

    #[test]
    fn invalid_policy_returns_typed_error() {
        let policies = [
            Truncation::absolute_cutoff(f64::NAN),
            Truncation::relative_cutoff(f64::INFINITY),
            Truncation::relative_inf_cutoff(-1.0),
            Truncation::relative_error(f64::NAN),
        ];

        for policy in policies {
            assert!(matches!(policy, Err(TruncationError::InvalidPolicy { .. })));
        }

        let entries = [(1.0, vec![3.0, 2.0, 1.0])];
        let spectra = spectra(&entries);
        let unchecked = Truncation::rank(3).and(Truncation::Tolerance {
            atol: 0.0,
            rtol: f64::NAN,
        });
        assert!(matches!(
            select_truncation(&spectra, &unchecked),
            Err(TruncationError::InvalidPolicy { .. })
        ));
    }

    #[test]
    fn non_descending_spectrum_returns_typed_error() {
        let entries = [(1.0, vec![3.0, 1.0, 2.0])];
        let spectra = spectra(&entries);
        assert!(matches!(
            select_truncation(&spectra, &Truncation::rank(2)),
            Err(TruncationError::InvalidSpectrum { .. })
        ));
    }
}
