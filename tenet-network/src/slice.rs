//! Clean-room greedy contraction *slicing* (Gray & Kourtis 2021, §4.7).
//!
//! Given a fixed [`ContractionPlan`], greedily fix ("slice") index labels — each
//! sliced index of dimension `d` splits the contraction into `d` independent
//! sub-contractions — until the largest intermediate tensor (peak memory) fits
//! under a target size, choosing at each step the index that increases total
//! time complexity the least (ties broken toward the larger memory reduction).
//!
//! This is a clean-room implementation from the published algorithm; it does
//! **not** derive from cotengrust (AGPL). It operates on a [`ContractionPlan`]
//! and [`DenseCostModel`], so it composes with any path optimizer. The slice
//! decision uses dense (scalar) index dimensions; for symmetric tensors pass the
//! effective (total) bond dimension — a safe peak-memory upper bound.

use std::collections::{BTreeSet, HashMap};

use crate::cost::DenseCostModel;
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::TemporaryLabel;
use crate::plan::ContractionPlan;

const SLICE_PLAN_HEADER: &str = "tenet-slice-plan-v1";

/// Whether a sliced index is **internal** (contracted, summed over) or
/// **output** (open/free, stacked/scattered into the result).
///
/// cotengra's `gather_slices` makes the same distinction: internal sliced
/// indices are *summed* across slices, output sliced indices are *stacked*
/// (each per-slice partial lands in a different output coordinate). A sliced
/// executor uses this to recombine partials correctly (e.g. the new-core
/// a sliced executor, which sums internal slices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceKind {
    /// Contracted index: partials are summed.
    Internal,
    /// Open/output index: partials are scattered into output coordinates.
    Output,
}

impl SliceKind {
    /// Stable text representation used by [`SlicePlan::to_text`].
    pub fn as_str(self) -> &'static str {
        match self {
            SliceKind::Internal => "internal",
            SliceKind::Output => "output",
        }
    }

    /// Parse a [`SliceKind`] written by [`as_str`](Self::as_str).
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "internal" => Ok(SliceKind::Internal),
            "output" => Ok(SliceKind::Output),
            other => Err(invalid_serialized_slice(format!(
                "invalid slice kind `{other}`"
            ))),
        }
    }
}

/// The set of indices to slice plus summary cost metrics.
///
/// Each sliced label carries a [`SliceKind`] marking whether it is internal
/// (summed) or output (stacked). [`greedy_slice`] only ever marks labels
/// `Internal`; [`greedy_slice_with_output`] may also mark labels `Output`.
#[derive(Debug, Clone, PartialEq)]
pub struct SlicePlan {
    sliced: Vec<TemporaryLabel>,
    kinds: Vec<SliceKind>,
    nslices: u128,
    sliced_width: usize,
    unsliced_width: usize,
    per_slice_flops: f64,
}

impl SlicePlan {
    /// Index labels chosen to slice (sorted, deterministic).
    pub fn sliced_indices(&self) -> &[TemporaryLabel] {
        &self.sliced
    }

    /// The [`SliceKind`] of each sliced label, parallel to
    /// [`sliced_indices`](Self::sliced_indices).
    pub fn sliced_kinds(&self) -> &[SliceKind] {
        &self.kinds
    }

    /// `(label, kind)` pairs for every sliced index.
    pub fn sliced_with_kinds(&self) -> impl Iterator<Item = (&TemporaryLabel, SliceKind)> {
        self.sliced.iter().zip(self.kinds.iter().copied())
    }

    /// True if any sliced index is an **output** (open) index — i.e. the
    /// recombination requires scatter/stack, not plain sum (an output-aware
    /// sliced executor is required in that case).
    pub fn has_output_slices(&self) -> bool {
        self.kinds.iter().any(|k| *k == SliceKind::Output)
    }

    /// Number of independent sub-contractions = product of sliced index dims.
    pub fn nslices(&self) -> u128 {
        self.nslices
    }

    /// Estimated FLOPs for a single slice's contraction (size proxy).
    pub fn per_slice_flops(&self) -> f64 {
        self.per_slice_flops
    }

    /// Total time-complexity estimate across all slices = `nslices *
    /// per_slice_flops` (cotengra's `tc = 2^|Es| * tc(T)`). Lower is better; use
    /// it to rank candidate (path, slice) schemes.
    pub fn total_flops(&self) -> f64 {
        self.nslices as f64 * self.per_slice_flops
    }

    /// Largest intermediate (in elements) per slice, after slicing.
    pub fn sliced_width(&self) -> usize {
        self.sliced_width
    }

    /// Largest intermediate (in elements) with no slicing.
    pub fn unsliced_width(&self) -> usize {
        self.unsliced_width
    }

    /// True when no slicing was needed (the plan already fits the target).
    pub fn is_empty(&self) -> bool {
        self.sliced.is_empty()
    }

    /// Serialize this slice decision to a compact text format.
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        text.push_str(SLICE_PLAN_HEADER);
        text.push('\n');
        text.push_str("nslices ");
        text.push_str(&self.nslices.to_string());
        text.push('\n');
        text.push_str("sliced_width ");
        text.push_str(&self.sliced_width.to_string());
        text.push('\n');
        text.push_str("unsliced_width ");
        text.push_str(&self.unsliced_width.to_string());
        text.push('\n');
        text.push_str("per_slice_flops ");
        text.push_str(&self.per_slice_flops.to_string());
        text.push('\n');
        for (label, kind) in self.sliced_with_kinds() {
            text.push_str("slice ");
            text.push_str(kind.as_str());
            text.push(' ');
            text.push_str(label.as_str());
            text.push('\n');
        }
        text
    }

    /// Restore a slice decision serialized by [`to_text`](Self::to_text).
    pub fn from_text(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| invalid_serialized_slice("missing header"))?;
        if header != SLICE_PLAN_HEADER {
            return Err(invalid_serialized_slice("unsupported slice plan header"));
        }

        let nslices = parse_field::<u128>(
            lines
                .next()
                .ok_or_else(|| invalid_serialized_slice("missing nslices line"))?,
            "nslices",
        )?;
        let sliced_width = parse_field::<usize>(
            lines
                .next()
                .ok_or_else(|| invalid_serialized_slice("missing sliced_width line"))?,
            "sliced_width",
        )?;
        let unsliced_width = parse_field::<usize>(
            lines
                .next()
                .ok_or_else(|| invalid_serialized_slice("missing unsliced_width line"))?,
            "unsliced_width",
        )?;
        let per_slice_flops = parse_field::<f64>(
            lines
                .next()
                .ok_or_else(|| invalid_serialized_slice("missing per_slice_flops line"))?,
            "per_slice_flops",
        )?;

        let mut sliced = Vec::new();
        let mut kinds = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 3 || parts[0] != "slice" {
                return Err(invalid_serialized_slice("invalid slice line"));
            }
            kinds.push(SliceKind::parse(parts[1])?);
            sliced.push(TemporaryLabel::from(parts[2]));
        }

        Ok(Self {
            sliced,
            kinds,
            nslices,
            sliced_width,
            unsliced_width,
            per_slice_flops,
        })
    }
}

fn invalid_serialized_slice(message: impl Into<String>) -> ContractError {
    ContractError::InvalidContractionPlan(format!(
        "invalid serialized slice plan: {}",
        message.into()
    ))
}

fn parse_field<T>(line: &str, name: &str) -> Result<T>
where
    T: std::str::FromStr,
{
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.len() != 2 || parts[0] != name {
        return Err(invalid_serialized_slice(format!("invalid {name} line")));
    }
    parts[1]
        .parse::<T>()
        .map_err(|_| invalid_serialized_slice(format!("invalid {name} value")))
}

struct StepShape {
    /// Union of both operands' labels (drives the per-step work / flops).
    union: Vec<TemporaryLabel>,
    /// Result (intermediate) labels (drives the per-step peak memory).
    result: Vec<TemporaryLabel>,
}

fn dim_sliced(
    cost: &DenseCostModel,
    label: &TemporaryLabel,
    sliced: &BTreeSet<TemporaryLabel>,
) -> usize {
    if sliced.contains(label) {
        1
    } else {
        cost.dim(label).unwrap_or(1)
    }
}

fn product_dims(
    labels: &[TemporaryLabel],
    cost: &DenseCostModel,
    sliced: &BTreeSet<TemporaryLabel>,
) -> usize {
    labels.iter().fold(1usize, |acc, l| {
        acc.saturating_mul(dim_sliced(cost, l, sliced))
    })
}

fn step_shapes(ir: &NetworkIR, plan: &ContractionPlan) -> Vec<StepShape> {
    let mut labels_by_id: HashMap<usize, Vec<TemporaryLabel>> = HashMap::new();
    for tensor in ir.tensors() {
        labels_by_id.insert(tensor.id().index(), tensor.labels().to_vec());
    }
    for step in plan.steps() {
        labels_by_id.insert(step.result().index(), step.result_labels().to_vec());
    }

    plan.steps()
        .iter()
        .map(|step| {
            let lhs = labels_by_id
                .get(&step.lhs().index())
                .cloned()
                .unwrap_or_default();
            let rhs = labels_by_id
                .get(&step.rhs().index())
                .cloned()
                .unwrap_or_default();
            let mut union = lhs;
            for label in rhs {
                if !union.contains(&label) {
                    union.push(label);
                }
            }
            StepShape {
                union,
                result: step.result_labels().to_vec(),
            }
        })
        .collect()
}

fn width_of(
    shapes: &[StepShape],
    cost: &DenseCostModel,
    sliced: &BTreeSet<TemporaryLabel>,
) -> usize {
    shapes
        .iter()
        .map(|s| product_dims(&s.result, cost, sliced))
        .max()
        .unwrap_or(0)
}

fn per_slice_flops(
    shapes: &[StepShape],
    cost: &DenseCostModel,
    sliced: &BTreeSet<TemporaryLabel>,
) -> f64 {
    shapes
        .iter()
        .map(|s| product_dims(&s.union, cost, sliced) as f64)
        .sum()
}

fn nslices_of(cost: &DenseCostModel, sliced: &BTreeSet<TemporaryLabel>) -> u128 {
    sliced.iter().fold(1u128, |acc, l| {
        acc.saturating_mul(cost.dim(l).unwrap_or(1) as u128)
    })
}

fn nslices_of_labels(cost: &DenseCostModel, sliced: &[TemporaryLabel]) -> u128 {
    sliced.iter().fold(1u128, |acc, l| {
        acc.saturating_mul(cost.dim(l).unwrap_or(1) as u128)
    })
}

/// Greedily choose indices to slice until the largest intermediate fits under
/// `target_width` (in elements). At each step picks the not-yet-sliced index
/// that strictly reduces peak memory and minimizes total time complexity
/// (`per_slice_flops * nslices`), breaking ties toward the larger memory
/// reduction. Stops early if no remaining index can reduce the peak further.
///
/// Only **internal (contracted)** indices are sliced (dim > 1, not an output
/// label), so every per-slice partial has the full output shape and partials are
/// summed. Slicing output/open indices (which would need stack/chunk
/// recombination) is intentionally out of scope here.
pub fn greedy_slice(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    cost: &DenseCostModel,
    target_width: usize,
) -> SlicePlan {
    let shapes = step_shapes(ir, plan);

    // Candidates are INTERNAL (contracted) indices only: dim > 1 and not an
    // output/open label. A sliced internal index makes every per-slice result
    // the same shape as the full output, so partials are summed (the simple
    // case). Slicing output indices needs stack/chunk recombination
    // (cotengra's `gather_slices` distinction) and is a later extension.
    let mut candidates: BTreeSet<TemporaryLabel> = BTreeSet::new();
    for tensor in ir.tensors() {
        for label in tensor.labels() {
            if cost.dim(label).unwrap_or(1) > 1 && !ir.output_labels().contains(label) {
                candidates.insert(label.clone());
            }
        }
    }

    let mut sliced: BTreeSet<TemporaryLabel> = BTreeSet::new();
    let unsliced_width = width_of(&shapes, cost, &sliced);

    loop {
        let width = width_of(&shapes, cost, &sliced);
        if width <= target_width {
            break;
        }

        let mut best: Option<(TemporaryLabel, f64, usize)> = None;
        for candidate in candidates.iter() {
            if sliced.contains(candidate) {
                continue;
            }
            let mut trial = sliced.clone();
            trial.insert(candidate.clone());
            let trial_width = width_of(&shapes, cost, &trial);
            if trial_width >= width {
                continue; // no peak-memory progress
            }
            let total = per_slice_flops(&shapes, cost, &trial) * (nslices_of(cost, &trial) as f64);
            let better = match &best {
                None => true,
                Some((_, best_total, best_width)) => {
                    total < *best_total || (total == *best_total && trial_width < *best_width)
                }
            };
            if better {
                best = Some((candidate.clone(), total, trial_width));
            }
        }

        match best {
            Some((label, _, _)) => {
                sliced.insert(label);
            }
            None => break, // cannot reduce peak further with available indices
        }
    }

    let sliced_width = width_of(&shapes, cost, &sliced);
    let per_slice_flops = per_slice_flops(&shapes, cost, &sliced);
    let sliced: Vec<TemporaryLabel> = sliced.iter().cloned().collect();
    let kinds = vec![SliceKind::Internal; sliced.len()];
    SlicePlan {
        nslices: nslices_of_labels(cost, &sliced),
        kinds,
        sliced,
        sliced_width,
        unsliced_width,
        per_slice_flops,
    }
}

/// Like [`greedy_slice`], but candidates **may** include output (open) indices.
///
/// Output indices reduce peak memory by shrinking the final-result shape (and any
/// intermediate that carries them), but their per-slice partials must be
/// *scattered* into output coordinates rather than summed (cotengra's
/// `gather_slices`: internal → sum, output → stack). The returned [`SlicePlan`]
/// marks each chosen label with its [`SliceKind`] so an output-aware sliced
/// executor can handle both kinds.
///
/// The greedy objective (minimize total time complexity, ties toward larger peak
/// reduction) is unchanged; only the candidate set is widened to include output
/// labels of dim > 1.
pub fn greedy_slice_with_output(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    cost: &DenseCostModel,
    target_width: usize,
) -> SlicePlan {
    let shapes = step_shapes(ir, plan);

    // Candidates: any index of dim > 1, INTERNAL or OUTPUT.
    let mut candidates: BTreeSet<TemporaryLabel> = BTreeSet::new();
    for tensor in ir.tensors() {
        for label in tensor.labels() {
            if cost.dim(label).unwrap_or(1) > 1 {
                candidates.insert(label.clone());
            }
        }
    }

    let mut sliced: BTreeSet<TemporaryLabel> = BTreeSet::new();
    let unsliced_width = width_of(&shapes, cost, &sliced);

    loop {
        let width = width_of(&shapes, cost, &sliced);
        if width <= target_width {
            break;
        }

        let mut best: Option<(TemporaryLabel, f64, usize)> = None;
        for candidate in candidates.iter() {
            if sliced.contains(candidate) {
                continue;
            }
            let mut trial = sliced.clone();
            trial.insert(candidate.clone());
            let trial_width = width_of(&shapes, cost, &trial);
            if trial_width >= width {
                continue; // no peak-memory progress
            }
            let total = per_slice_flops(&shapes, cost, &trial) * (nslices_of(cost, &trial) as f64);
            let better = match &best {
                None => true,
                Some((_, best_total, best_width)) => {
                    total < *best_total || (total == *best_total && trial_width < *best_width)
                }
            };
            if better {
                best = Some((candidate.clone(), total, trial_width));
            }
        }

        match best {
            Some((label, _, _)) => {
                sliced.insert(label);
            }
            None => break,
        }
    }

    let sliced_width = width_of(&shapes, cost, &sliced);
    let per_slice_flops = per_slice_flops(&shapes, cost, &sliced);
    let sliced: Vec<TemporaryLabel> = sliced.iter().cloned().collect();
    let kinds = sliced
        .iter()
        .map(|l| {
            if ir.output_labels().contains(l) {
                SliceKind::Output
            } else {
                SliceKind::Internal
            }
        })
        .collect();
    SlicePlan {
        nslices: nslices_of_labels(cost, &sliced),
        kinds,
        sliced,
        sliced_width,
        unsliced_width,
        per_slice_flops,
    }
}

/// Peak intermediate size (in elements) of a plan under a cost model — the
/// largest tensor produced by any contraction step. Pass a cost model whose
/// already-sliced labels are unit ([`DenseCostModel::with_unit_dims`]) to get the
/// per-slice width.
pub fn contraction_width(ir: &NetworkIR, plan: &ContractionPlan, cost: &DenseCostModel) -> usize {
    let shapes = step_shapes(ir, plan);
    let empty = BTreeSet::new();
    width_of(&shapes, cost, &empty)
}

/// The single internal (contracted, dim > 1, non-output) index whose slicing
/// reduces the peak intermediate the most while increasing total time the least,
/// or `None` if no index reduces the peak. Used for one step of dynamic slicing.
pub fn best_next_internal_index(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    cost: &DenseCostModel,
) -> Option<TemporaryLabel> {
    best_next_slice_index(ir, plan, cost, false)
}

/// The single index whose slicing reduces the peak intermediate the most while
/// increasing total time the least, or `None` if no index reduces the peak.
/// When `allow_output` is true, output/open indices are also candidates and the
/// returned label should be packaged with [`SliceKind::Output`] by
/// [`slice_plan_for`].
pub fn best_next_slice_index(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    cost: &DenseCostModel,
    allow_output: bool,
) -> Option<TemporaryLabel> {
    let shapes = step_shapes(ir, plan);
    let empty = BTreeSet::new();
    let width = width_of(&shapes, cost, &empty);

    let mut candidates: BTreeSet<TemporaryLabel> = BTreeSet::new();
    for tensor in ir.tensors() {
        for label in tensor.labels() {
            let is_output = ir.output_labels().contains(label);
            if cost.dim(label).unwrap_or(1) > 1 && (allow_output || !is_output) {
                candidates.insert(label.clone());
            }
        }
    }

    let mut best: Option<(TemporaryLabel, f64, usize)> = None;
    for candidate in &candidates {
        let mut trial = BTreeSet::new();
        trial.insert(candidate.clone());
        let trial_width = width_of(&shapes, cost, &trial);
        if trial_width >= width {
            continue;
        }
        // Compare candidates by TOTAL time complexity (`per_slice_flops *
        // nslices`), matching the metric `greedy_slice` uses (see the `total`
        // expression there). Candidates have different slice counts (a sliced
        // index's dim = its number of slices), so the per-slice FLOPs alone is
        // not a like-for-like comparison and could pick a worse total.
        let total = per_slice_flops(&shapes, cost, &trial) * (nslices_of(cost, &trial) as f64);
        let better = match &best {
            None => true,
            Some((_, best_total, best_width)) => {
                total < *best_total || (total == *best_total && trial_width < *best_width)
            }
        };
        if better {
            best = Some((candidate.clone(), total, trial_width));
        }
    }
    best.map(|(label, _, _)| label)
}

/// Build a [`SlicePlan`] describing the cost metrics of slicing exactly
/// `sliced_labels` of this plan (used to package a dynamic-slicing decision).
pub fn slice_plan_for(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    cost: &DenseCostModel,
    sliced_labels: &[TemporaryLabel],
) -> SlicePlan {
    let mut sliced_vec = sliced_labels.to_vec();
    sliced_vec.sort();
    sliced_vec.dedup();
    slice_plan_for_ordered(ir, plan, cost, &sliced_vec)
}

/// Build a [`SlicePlan`] while preserving the caller supplied sliced-label
/// order. This is useful for external planners such as cotengra, which carry a
/// concrete slice enumeration order. Duplicate labels are ignored after their
/// first occurrence.
pub fn slice_plan_for_ordered(
    ir: &NetworkIR,
    plan: &ContractionPlan,
    cost: &DenseCostModel,
    sliced_labels: &[TemporaryLabel],
) -> SlicePlan {
    let shapes = step_shapes(ir, plan);
    let mut sliced = BTreeSet::<TemporaryLabel>::new();
    let mut sliced_vec = Vec::<TemporaryLabel>::new();
    for label in sliced_labels {
        if sliced.insert(label.clone()) {
            sliced_vec.push(label.clone());
        }
    }
    let empty = BTreeSet::new();
    let kinds = sliced_vec
        .iter()
        .map(|l| {
            if ir.output_labels().contains(l) {
                SliceKind::Output
            } else {
                SliceKind::Internal
            }
        })
        .collect();
    SlicePlan {
        nslices: nslices_of(cost, &sliced),
        sliced_width: width_of(&shapes, cost, &sliced),
        unsliced_width: width_of(&shapes, cost, &empty),
        per_slice_flops: per_slice_flops(&shapes, cost, &sliced),
        kinds,
        sliced: sliced_vec,
    }
}

const SLICED_PLAN_HEADER: &str = "tenet-sliced-plan-v1";
const BEGIN_PLAN: &str = "BEGIN_CONTRACTION_PLAN";
const END_PLAN: &str = "END_CONTRACTION_PLAN";
const BEGIN_SLICE: &str = "BEGIN_SLICE_PLAN";
const END_SLICE: &str = "END_SLICE_PLAN";

/// A contraction order ([`ContractionPlan`]) bundled with a slicing decision
/// ([`SlicePlan`]) so the two can be cached and executed together.
///
/// This is a pure-structure type (order + slicing decision + serialization): it
/// holds no tensor data and is shared with the `tenet-cotengrust` path/slice
/// search. The new-core sliced einsum executor lives in
/// a sliced executor.
#[derive(Debug, Clone, PartialEq)]
pub struct SlicedPlan {
    plan: ContractionPlan,
    slice: SlicePlan,
}

impl SlicedPlan {
    /// Bundle a contraction plan with a slicing decision.
    pub fn new(plan: ContractionPlan, slice: SlicePlan) -> Self {
        Self { plan, slice }
    }

    /// The contraction order plan.
    pub fn plan(&self) -> &ContractionPlan {
        &self.plan
    }

    /// The slicing decision.
    pub fn slice(&self) -> &SlicePlan {
        &self.slice
    }

    /// Serialize the contraction order and slicing decision together.
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        text.push_str(SLICED_PLAN_HEADER);
        text.push('\n');
        text.push_str(BEGIN_PLAN);
        text.push('\n');
        text.push_str(&self.plan.to_text());
        text.push_str(END_PLAN);
        text.push('\n');
        text.push_str(BEGIN_SLICE);
        text.push('\n');
        text.push_str(&self.slice.to_text());
        text.push_str(END_SLICE);
        text.push('\n');
        text
    }

    /// Restore a [`SlicedPlan`] serialized by [`to_text`](Self::to_text).
    pub fn from_text(text: &str) -> Result<Self> {
        let lines = text.lines().collect::<Vec<_>>();
        if lines.first().copied() != Some(SLICED_PLAN_HEADER) {
            return Err(invalid_serialized_sliced_plan(
                "unsupported sliced plan header",
            ));
        }

        let plan_begin = find_sliced_line(&lines, BEGIN_PLAN)?;
        let plan_end = find_sliced_line(&lines, END_PLAN)?;
        let slice_begin = find_sliced_line(&lines, BEGIN_SLICE)?;
        let slice_end = find_sliced_line(&lines, END_SLICE)?;
        if !(plan_begin < plan_end && plan_end < slice_begin && slice_begin < slice_end) {
            return Err(invalid_serialized_sliced_plan("invalid section order"));
        }

        let plan_text = lines[(plan_begin + 1)..plan_end].join("\n") + "\n";
        let slice_text = lines[(slice_begin + 1)..slice_end].join("\n") + "\n";
        let plan = ContractionPlan::from_text(&plan_text)?;
        let slice = SlicePlan::from_text(&slice_text)?;
        Ok(Self::new(plan, slice))
    }
}

fn invalid_serialized_sliced_plan(message: impl Into<String>) -> ContractError {
    ContractError::InvalidContractionPlan(format!(
        "invalid serialized sliced plan: {}",
        message.into()
    ))
}

fn find_sliced_line(lines: &[&str], needle: &str) -> Result<usize> {
    lines
        .iter()
        .position(|line| *line == needle)
        .ok_or_else(|| invalid_serialized_sliced_plan(format!("missing {needle}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_einsum;
    use crate::{ActivePair, DenseTensorInfo};

    fn chain_abc(
        a: usize,
        b: usize,
        c: usize,
        d: usize,
    ) -> (NetworkIR, ContractionPlan, DenseCostModel) {
        // "ab,bc,cd->ad" with the order fixed to create the intermediate "ac"
        // (= a*c). Internal index c lives in "ac", so slicing c shrinks the
        // peak intermediate; outputs a,d are not sliceable.
        let ir = parse_einsum("ab,bc,cd->ad").unwrap();
        let infos = vec![
            DenseTensorInfo::new(vec![a, b]),
            DenseTensorInfo::new(vec![b, c]),
            DenseTensorInfo::new(vec![c, d]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let path = vec![ActivePair::new(0, 1), ActivePair::new(0, 1)];
        let plan = ContractionPlan::from_dense_active_pair_path(&ir, &path, &cost).unwrap();
        (ir, plan, cost)
    }

    #[test]
    fn greedy_slice_reaches_target_width() {
        // Intermediate "ac" = a*c = 2*8 = 16 is the peak; output "ad" = 4.
        // target 8 forces slicing the internal index c.
        let (ir, plan, cost) = chain_abc(2, 2, 8, 2);
        let sp = greedy_slice(&ir, &plan, &cost, 8);

        assert_eq!(sp.unsliced_width(), 16);
        assert!(!sp.is_empty(), "expected slicing to be required");
        assert!(
            sp.sliced_width() <= 8,
            "sliced width {} should be <= 8",
            sp.sliced_width()
        );
        // Only internal "c" can shrink the "ac" intermediate.
        assert_eq!(sp.sliced_indices(), &[TemporaryLabel::new("c")]);

        let expected: u128 = sp
            .sliced_indices()
            .iter()
            .map(|l| cost.dim(l).unwrap() as u128)
            .product();
        assert_eq!(sp.nslices(), expected);
    }

    #[test]
    fn slice_plan_text_roundtrip_preserves_output_kinds() {
        let (ir, plan, cost) = chain_abc(6, 1, 1, 6);
        let sp = greedy_slice_with_output(&ir, &plan, &cost, 6);
        assert!(sp.has_output_slices());

        let text = sp.to_text();
        let restored = SlicePlan::from_text(&text).unwrap();
        assert_eq!(restored, sp);
    }

    #[test]
    fn greedy_slice_never_slices_output_indices() {
        // a,d are output; only internal b,c are sliceable. Forcing slicing must
        // never pick an output label.
        let (ir, plan, cost) = chain_abc(2, 2, 8, 2);
        let sp = greedy_slice(&ir, &plan, &cost, 8);
        assert!(!sp.is_empty());
        assert!(!sp.sliced_indices().contains(&TemporaryLabel::new("a")));
        assert!(!sp.sliced_indices().contains(&TemporaryLabel::new("d")));
    }

    #[test]
    fn best_next_slice_index_can_choose_output() {
        let (ir, plan, cost) = chain_abc(6, 1, 1, 6);

        assert_eq!(best_next_internal_index(&ir, &plan, &cost), None);
        let next = best_next_slice_index(&ir, &plan, &cost, true)
            .expect("output slicing should reduce the output peak");
        assert!(ir.output_labels().contains(&next));
    }

    /// `best_next_slice_index` must rank candidates by TOTAL time complexity
    /// (`per_slice_flops * nslices`), the same basis as `greedy_slice` (see the
    /// `total` expression there) — not by per-slice FLOPs alone.
    ///
    /// For the network and dims below the peak intermediate is `be` (= 3*6 = 18);
    /// only the internal indices `b` and `e` reduce that peak, so both are
    /// candidates. Their two metrics disagree:
    ///   - per-slice FLOPs only:  e = 48  <  b = 72        → would pick `e`
    ///   - total time complexity: b = 72*3 = 216  <  e = 48*6 = 288  → picks `b`
    /// The old code compared per-slice FLOPs and would have returned `e`; with
    /// the total-cost metric it must now return `b`. (The `dc,cd` factor is an
    /// independent scalar block; it shapes the step order but not the decision.)
    #[test]
    fn best_next_slice_index_uses_total_cost_not_per_slice() {
        let ir = parse_einsum("dc,cd,ab,ae,be->").unwrap();
        let infos = vec![
            DenseTensorInfo::new(vec![3, 8]), // dc: d=3, c=8
            DenseTensorInfo::new(vec![8, 3]), // cd: c=8, d=3
            DenseTensorInfo::new(vec![6, 3]), // ab: a=6, b=3
            DenseTensorInfo::new(vec![6, 6]), // ae: a=6, e=6
            DenseTensorInfo::new(vec![3, 6]), // be: b=3, e=6
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let path = vec![
            ActivePair::new(0, 1),
            ActivePair::new(0, 1),
            ActivePair::new(0, 1),
            ActivePair::new(0, 1),
        ];
        let plan = ContractionPlan::from_dense_active_pair_path(&ir, &path, &cost).unwrap();

        let next = best_next_slice_index(&ir, &plan, &cost, false)
            .expect("an internal index should reduce the peak");
        assert_eq!(
            next,
            TemporaryLabel::new("b"),
            "total-cost ranking must pick `b`; per-slice-only ranking would pick `e`"
        );
    }
}
