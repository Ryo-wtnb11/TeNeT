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
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;

use tenet_core::{RuleIdentity, SectorId};

/// A fixed per-sector prefix count, TensorKit's `TruncationSpace`
/// (`src/factorizations/truncation.jl:261-269`).
///
/// TensorKit reads the target rank of coupled sector `c` as
/// `dim(strategy.space, c)` — the *reduced* (per-sector degeneracy) dimension
/// of a target space — and then applies a plain `truncrank` inside that block.
/// That is exactly a per-sector prefix count, which is why this fits the
/// prefix-only decision layer instead of needing the non-prefix filter layer
/// `truncfilter` would.
///
/// Build one from a space via `Space::truncspace` / `GradedSpace::truncspace`
/// rather than by hand: the sector keys are the engine's opaque
/// [`SectorId`]s, so the space that produced them is the only honest source,
/// and the [`RuleIdentity`] recorded alongside is what lets the factorization
/// reject a profile built against a different fusion rule.
#[derive(Clone, Debug, PartialEq)]
pub struct TruncationSpace {
    rule: RuleIdentity,
    ranks: BTreeMap<SectorId, usize>,
}

impl TruncationSpace {
    /// Builds a profile from a rule identity and its `(sector, rank)` pairs.
    ///
    /// Intended for the facade adapters, which take both from one space. A
    /// sector missing from `ranks` is truncated away entirely (rank zero),
    /// matching TensorKit: `dim(V, c)` of an absent sector is zero.
    pub fn new(rule: RuleIdentity, ranks: impl IntoIterator<Item = (SectorId, usize)>) -> Self {
        Self {
            rule,
            ranks: ranks.into_iter().collect(),
        }
    }

    /// The fusion rule this profile's sector ids belong to.
    pub fn rule(&self) -> &RuleIdentity {
        &self.rule
    }

    /// The requested rank of a coupled sector; `0` when the sector is absent.
    pub fn rank(&self, sector: SectorId) -> usize {
        self.ranks.get(&sector).copied().unwrap_or(0)
    }
}

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
    /// Keep exactly the requested prefix of every coupled sector, clamped to
    /// what the spectrum offers. TensorKit `TruncationSpace`.
    Space(TruncationSpace),
    /// Keep a value only if every component keeps it. Prefix rules compose to
    /// a prefix rule, so this is the per-sector minimum of the kept counts.
    All(Vec<Truncation>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TruncationError {
    InvalidPolicy {
        message: &'static str,
    },
    InvalidSpectrum {
        message: &'static str,
    },
    /// A [`Truncation::Space`] profile was built against a different fusion
    /// rule than the tensor being factorized, so its [`SectorId`] keys name
    /// different sectors than the spectra do.
    RuleMismatch,
}

impl fmt::Display for TruncationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy { message } => write!(f, "invalid truncation policy: {message}"),
            Self::InvalidSpectrum { message } => {
                write!(f, "invalid truncation spectrum: {message}")
            }
            Self::RuleMismatch => write!(
                f,
                "truncation space profile was built for a different fusion rule"
            ),
        }
    }
}

impl std::error::Error for TruncationError {}

impl From<TruncationError> for tenet_tensors::OperationError {
    fn from(error: TruncationError) -> Self {
        Self::InvalidArgument {
            // `OperationError` carries a `&'static str`, so the policy/spectrum
            // detail cannot travel; the rule mismatch gets its own message
            // because it is the one a caller can actually act on (they passed
            // a profile built from the wrong space).
            message: match error {
                TruncationError::RuleMismatch => {
                    "truncation space profile was built for a different fusion rule"
                }
                _ => "invalid truncation input",
            },
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

    /// Keep the fixed per-sector prefix `profile` names (TensorKit
    /// `truncspace`).
    pub fn space(profile: TruncationSpace) -> Self {
        Self::Space(profile)
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

/// One coupled sector's spectrum offered to the selection: its identity, its
/// quantum dimension and its values, non-negative and descending.
///
/// `sector` is only read by [`Truncation::Space`], the one policy whose
/// decision is per-sector rather than magnitude-driven; every other policy
/// stays identity-blind.
#[derive(Clone, Copy, Debug)]
pub struct WeightedSpectrum<'a> {
    pub sector: SectorId,
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
///
/// `rule` is the fusion rule the caller's [`WeightedSpectrum::sector`] ids
/// belong to. It is checked against every [`Truncation::Space`] profile in
/// `truncation` *before* any selection runs, so a profile built from another
/// rule's space is rejected rather than silently reading its sector ids as
/// this rule's — which would truncate to rank zero at random.
///
/// Why the rule is a parameter rather than a check each factorization makes
/// for itself: this is the single seam every truncated factorization already
/// routes through, so putting the guard here is the one place a future caller
/// cannot forget it.
///
/// # Errors
///
/// [`TruncationError::RuleMismatch`] for a foreign profile,
/// [`TruncationError::InvalidPolicy`] for a policy with a non-finite or
/// negative tolerance, [`TruncationError::InvalidSpectrum`] for spectra that
/// are not finite, non-negative and descending.
pub fn select_truncation(
    spectra: &[WeightedSpectrum<'_>],
    truncation: &Truncation,
    rule: &RuleIdentity,
) -> Result<TruncationDecision, TruncationError> {
    validate_rule(truncation, rule)?;
    validate_truncation(truncation)?;
    validate_spectra(spectra)?;
    let kept = kept_counts(spectra, truncation);
    let error = discarded_norm(spectra, &kept);
    Ok(TruncationDecision { kept, error })
}

/// Rejects every [`Truncation::Space`] profile in `truncation` that was built
/// against a rule other than `rule`. Recurses through [`Truncation::All`]
/// because [`Truncation::and`] can bury a profile inside a composite.
fn validate_rule(truncation: &Truncation, rule: &RuleIdentity) -> Result<(), TruncationError> {
    match truncation {
        Truncation::Space(profile) => {
            if profile.rule == *rule {
                Ok(())
            } else {
                Err(TruncationError::RuleMismatch)
            }
        }
        Truncation::All(components) => components
            .iter()
            .try_for_each(|component| validate_rule(component, rule)),
        _ => Ok(()),
    }
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
        // A `Space` profile carries only `usize` ranks and a rule identity;
        // the rule is checked by `validate_rule` and there is no numeric
        // domain left to reject here.
        Truncation::Full | Truncation::Rank(_) | Truncation::Space(_) => Ok(()),
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
        // TensorKit `findtruncated(values, ::TruncationSpace)`
        // (truncation.jl:261-269): a plain `truncrank(dim(space, c))` inside
        // each block. Absent sector -> rank zero (TK's `dim(V, c)` is zero
        // there); clamped to what the spectrum actually offers, since asking
        // for more than exists is a request the prefix cannot honour rather
        // than an error. No magnitude enters, so the descending-prefix
        // invariant is preserved by construction.
        Truncation::Space(profile) => spectra
            .iter()
            .map(|spectrum| profile.rank(spectrum.sector).min(spectrum.values.len()))
            .collect(),
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

    /// The rule the test spectra's sector ids belong to. `select_truncation`
    /// now takes one; any stable identity works for the magnitude-driven
    /// policies, which never look at a sector.
    struct TestRule;

    fn rule() -> RuleIdentity {
        RuleIdentity::of_type::<TestRule>()
    }

    fn other_rule() -> RuleIdentity {
        struct OtherRule;
        RuleIdentity::of_type::<OtherRule>()
    }

    /// Sector ids are the entry positions, so a profile keyed by position is
    /// the same thing a space-derived profile would be.
    fn spectra<'a>(entries: &'a [(f64, Vec<f64>)]) -> Vec<WeightedSpectrum<'a>> {
        entries
            .iter()
            .enumerate()
            .map(|(index, (weight, values))| WeightedSpectrum {
                sector: SectorId::new(index),
                weight: *weight,
                values,
            })
            .collect()
    }

    fn select(
        spectra: &[WeightedSpectrum<'_>],
        truncation: &Truncation,
    ) -> Result<TruncationDecision, TruncationError> {
        select_truncation(spectra, truncation, &rule())
    }

    fn profile(pairs: [(usize, usize); 2]) -> TruncationSpace {
        TruncationSpace::new(
            rule(),
            pairs.map(|(sector, rank)| (SectorId::new(sector), rank)),
        )
    }

    #[test]
    fn rank_budget_is_quantum_dimension_weighted() {
        let entries = [(1.0, vec![5.0, 1.0]), (3.0, vec![4.0, 0.5])];
        let spectra = spectra(&entries);
        // Budget 4: keep 5.0 (weight 1) and 4.0 (weight 3) exactly.
        let decision = select(&spectra, &Truncation::rank(4)).unwrap();
        assert_eq!(decision.kept, vec![1, 1]);
        // Budget 5: the next candidate (1.0, weight 1) fits.
        let decision = select(&spectra, &Truncation::rank(5)).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
        // Budget 6: 0.5 has weight 3 and does not fit.
        let decision = select(&spectra, &Truncation::rank(6)).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
    }

    #[test]
    fn rank_ties_keep_parent_storage_order() {
        let entries = [(1.0, vec![2.0, 1.0]), (1.0, vec![2.0, 1.0])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::rank(1)).unwrap();
        assert_eq!(decision.kept, vec![1, 0]);

        let decision = select(&spectra, &Truncation::rank(3)).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
    }

    #[test]
    fn tolerance_thresholds_against_weighted_norm() {
        let entries = [(1.0, vec![4.0, 3.0, 0.1])];
        let spectra = spectra(&entries);
        let truncation = Truncation::absolute_cutoff(1.0).unwrap();
        let decision = select(&spectra, &truncation).unwrap();
        assert_eq!(decision.kept, vec![2]);
        assert!((decision.error - 0.1).abs() < 1e-12);

        // norm = 5.001..., rtol 0.5 => threshold ~2.5: keeps 4 and 3.
        let truncation = Truncation::relative_cutoff(0.5).unwrap();
        let decision = select(&spectra, &truncation).unwrap();
        assert_eq!(decision.kept, vec![2]);
    }

    #[test]
    fn tolerance_inf_thresholds_against_unweighted_max() {
        let entries = [(3.0, vec![4.0, 3.0, 0.1]), (1.0, vec![2.5])];
        let spectra = spectra(&entries);
        let truncation = Truncation::relative_inf_cutoff(0.7).unwrap();
        let decision = select(&spectra, &truncation).unwrap();
        assert_eq!(decision.kept, vec![2, 0]);
    }

    #[test]
    fn discard_weight_bounds_relative_error() {
        let entries = [(2.0, vec![3.0, 1.0, 0.5, 0.5])];
        let spectra = spectra(&entries);
        let norm = full_norm(&spectra);
        let truncation = Truncation::relative_error(0.3).unwrap();
        let decision = select(&spectra, &truncation).unwrap();
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
        let decision = select(&spectra, &combined).unwrap();
        assert_eq!(decision.kept, vec![2]);

        let combined = Truncation::rank(1).and(Truncation::absolute_cutoff(0.5).unwrap());
        let decision = select(&spectra, &combined).unwrap();
        assert_eq!(decision.kept, vec![1]);
    }

    #[test]
    fn full_keeps_everything_with_zero_error() {
        let entries = [(1.0, vec![2.0, 1.0]), (2.0, vec![1.5])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::Full).unwrap();
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
                select(&spectra, &policy),
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
            select(&spectra, &unchecked),
            Err(TruncationError::InvalidPolicy { .. })
        ));
    }

    #[test]
    fn space_profile_keeps_exactly_the_requested_prefix_counts() {
        // What: the counts come from the profile alone. The magnitudes are
        // arranged so that no magnitude-driven policy would produce `[1, 3]` —
        // sector 0 holds the three largest values — so a decision that leaked
        // into `Rank` / `Tolerance` behaviour cannot pass.
        let entries = [(1.0, vec![9.0, 8.0, 7.0]), (3.0, vec![2.0, 1.0, 0.5])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::space(profile([(0, 1), (1, 3)]))).unwrap();
        assert_eq!(decision.kept, vec![1, 3]);
        // The reported error is still the weighted 2-norm of the discarded tail.
        let expected = (8.0f64 * 8.0 + 7.0 * 7.0).sqrt();
        assert!((decision.error - expected).abs() < 1e-12);
    }

    #[test]
    fn space_profile_treats_absent_sectors_as_rank_zero_and_clamps_the_rest() {
        // What: TensorKit reads `dim(space, c)`, which is zero for a sector the
        // target space does not carry — so an absent key drops that sector
        // entirely. A key asking for more than the spectrum has is clamped
        // rather than rejected: the prefix simply cannot be longer.
        let entries = [(1.0, vec![5.0, 4.0]), (2.0, vec![3.0])];
        let spectra = spectra(&entries);
        let sparse = TruncationSpace::new(rule(), [(SectorId::new(0), 9)]);
        let decision = select(&spectra, &Truncation::space(sparse)).unwrap();
        assert_eq!(decision.kept, vec![2, 0]);
    }

    #[test]
    fn space_profile_composes_as_a_prefix_rule() {
        // What: `and` still takes the per-sector minimum, so a profile can be
        // intersected with a magnitude rule without leaving prefix-land.
        let entries = [(1.0, vec![4.0, 3.0, 0.1]), (1.0, vec![2.0, 1.0])];
        let spectra = spectra(&entries);
        let combined = Truncation::space(profile([(0, 3), (1, 1)]))
            .and(Truncation::absolute_cutoff(1.0).unwrap());
        let decision = select(&spectra, &combined).unwrap();
        assert_eq!(decision.kept, vec![2, 1]);
    }

    #[test]
    fn a_profile_from_another_rule_is_a_typed_error_before_any_selection() {
        // What: a foreign rule's sector ids name different sectors, so reading
        // them as this rule's would silently zero the spectrum out instead of
        // failing. Checked through `and` too, since a profile can be buried
        // inside a composite.
        let entries = [(1.0, vec![5.0, 4.0]), (2.0, vec![3.0])];
        let spectra = spectra(&entries);
        let foreign = TruncationSpace::new(other_rule(), [(SectorId::new(0), 1)]);

        assert_eq!(
            select(&spectra, &Truncation::space(foreign.clone())),
            Err(TruncationError::RuleMismatch)
        );
        assert_eq!(
            select(
                &spectra,
                &Truncation::rank(2).and(Truncation::space(foreign)),
            ),
            Err(TruncationError::RuleMismatch)
        );
    }

    #[test]
    fn non_descending_spectrum_returns_typed_error() {
        let entries = [(1.0, vec![3.0, 1.0, 2.0])];
        let spectra = spectra(&entries);
        assert!(matches!(
            select(&spectra, &Truncation::rank(2)),
            Err(TruncationError::InvalidSpectrum { .. })
        ));
    }
}
