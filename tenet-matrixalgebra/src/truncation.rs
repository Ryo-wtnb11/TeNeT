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
    Tolerance { atol: f64, rtol: f64 },
    /// Discard the smallest values while the weighted 2-norm of everything
    /// discarded stays at or below `rtol * norm`.
    DiscardWeight { rtol: f64 },
    /// Keep a value only if every component keeps it. Prefix rules compose to
    /// a prefix rule, so this is the per-sector minimum of the kept counts.
    All(Vec<Truncation>),
}

impl Truncation {
    /// Keep at most `rank` weighted dimensions.
    pub fn rank(rank: usize) -> Self {
        Self::Rank(rank)
    }

    /// Discard values below the absolute cutoff.
    pub fn absolute_cutoff(atol: f64) -> Self {
        Self::Tolerance { atol, rtol: 0.0 }
    }

    /// Discard values below `rtol` times the weighted 2-norm.
    pub fn relative_cutoff(rtol: f64) -> Self {
        Self::Tolerance { atol: 0.0, rtol }
    }

    /// Bound the relative truncation error (weighted 2-norm of the discarded
    /// tail) by `rtol`.
    pub fn relative_error(rtol: f64) -> Self {
        Self::DiscardWeight { rtol }
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
) -> TruncationDecision {
    let kept = kept_counts(spectra, truncation);
    let error = discarded_norm(spectra, &kept);
    TruncationDecision { kept, error }
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
                // Descending order guarantees `index` extends the prefix.
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
        Truncation::DiscardWeight { rtol } => {
            let norm = full_norm(spectra);
            let budget = (rtol * norm) * (rtol * norm);
            let mut order = descending_candidates(spectra);
            let mut kept: Vec<usize> = spectra
                .iter()
                .map(|spectrum| spectrum.values.len())
                .collect();
            let mut discarded = 0.0;
            while let Some((sector, index)) = order.pop() {
                let value = spectra[sector].values[index];
                let next = discarded + spectra[sector].weight * value * value;
                if next > budget + 1e-15 {
                    break;
                }
                debug_assert_eq!(index + 1, kept[sector]);
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

/// Candidates as `(sector, index)` sorted by descending value; ties keep the
/// input sector order so the decision is deterministic.
fn descending_candidates(spectra: &[WeightedSpectrum<'_>]) -> Vec<(usize, usize)> {
    let mut candidates: Vec<(usize, usize)> = spectra
        .iter()
        .enumerate()
        .flat_map(|(sector, spectrum)| (0..spectrum.values.len()).map(move |index| (sector, index)))
        .collect();
    candidates.sort_by(|&(ls, li), &(rs, ri)| {
        spectra[rs].values[ri]
            .partial_cmp(&spectra[ls].values[li])
            .expect("finite spectrum values")
            .then(ls.cmp(&rs))
            .then(li.cmp(&ri))
    });
    candidates
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
        let decision = select_truncation(&spectra, &Truncation::rank(4));
        assert_eq!(decision.kept, vec![1, 1]);
        // Budget 5: the next candidate (1.0, weight 1) fits.
        let decision = select_truncation(&spectra, &Truncation::rank(5));
        assert_eq!(decision.kept, vec![2, 1]);
        // Budget 6: 0.5 has weight 3 and does not fit.
        let decision = select_truncation(&spectra, &Truncation::rank(6));
        assert_eq!(decision.kept, vec![2, 1]);
    }

    #[test]
    fn tolerance_thresholds_against_weighted_norm() {
        let entries = [(1.0, vec![4.0, 3.0, 0.1])];
        let spectra = spectra(&entries);
        let decision = select_truncation(&spectra, &Truncation::absolute_cutoff(1.0));
        assert_eq!(decision.kept, vec![2]);
        assert!((decision.error - 0.1).abs() < 1e-12);

        // norm = 5.001..., rtol 0.5 => threshold ~2.5: keeps 4 and 3.
        let decision = select_truncation(&spectra, &Truncation::relative_cutoff(0.5));
        assert_eq!(decision.kept, vec![2]);
    }

    #[test]
    fn discard_weight_bounds_relative_error() {
        let entries = [(2.0, vec![3.0, 1.0, 0.5, 0.5])];
        let spectra = spectra(&entries);
        let norm = full_norm(&spectra);
        let decision = select_truncation(&spectra, &Truncation::relative_error(0.3));
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
        let combined = Truncation::rank(3).and(Truncation::absolute_cutoff(2.5));
        let decision = select_truncation(&spectra, &combined);
        assert_eq!(decision.kept, vec![2]);

        let combined = Truncation::rank(1).and(Truncation::absolute_cutoff(0.5));
        let decision = select_truncation(&spectra, &combined);
        assert_eq!(decision.kept, vec![1]);
    }

    #[test]
    fn full_keeps_everything_with_zero_error() {
        let entries = [(1.0, vec![2.0, 1.0]), (2.0, vec![1.5])];
        let spectra = spectra(&entries);
        let decision = select_truncation(&spectra, &Truncation::Full);
        assert_eq!(decision.kept, vec![2, 1]);
        assert_eq!(decision.error, 0.0);
    }
}
