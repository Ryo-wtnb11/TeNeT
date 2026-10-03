use super::*;

/// The test probes follow the scope body: it may run on a backend worker
/// thread, and the probes are thread-local so that parallel tests stay apart.
/// Moving them in and out makes a probe observe the call, whichever thread
/// runs it.
#[cfg(test)]
macro_rules! test_probes {
    ($($field:ident: $key:ident: $ty:ty),* $(,)?) => {
        pub(super) struct TestProbes {
            $($field: $ty,)*
        }

        impl TestProbes {
            pub(super) fn take() -> Self {
                Self { $($field: $key.take(),)* }
            }

            pub(super) fn put(self) {
                $($key.set(self.$field);)*
            }
        }
    };
}

#[cfg(test)]
test_probes! {
    compact_svd: COMPACT_SVD_COPY_PROBE: CompactSvdCopyProbe,
    compact_qr: COMPACT_QR_COPY_PROBE: CompactQrCopyProbe,
    eigh: EIGH_COPY_PROBE: EighCopyProbe,
    eigh_vectors: EIGH_OWNED_VECTOR_POINTERS: Vec<usize>,
    checked_eigh_pairs: CHECKED_EIGH_PAIR_POINTERS: Vec<usize>,
    checked_svd_stage: CHECKED_COMPACT_SVD_STAGE_POINTERS: Vec<(usize, usize)>,
    mf_svd_fallback: MF_COMPACT_SVD_FALLBACK_POINTERS: Vec<(usize, usize)>,
    compact_lq: COMPACT_LQ_COPY_PROBE: CompactLqCopyProbe,
    diagonal_bond: DIAGONAL_BOND_BUILD_PROBE: DiagonalBondBuildProbe,
    values_fallbacks: VALUES_MATRICIZATION_FALLBACKS: usize,
    checked_inputs: CHECKED_COMPACT_INPUT_OBSERVATIONS: Vec<CheckedCompactInputObservation>,
    pair_publication: GENERIC_PAIR_PUBLICATION_PROBE: GenericPairPublicationProbe,
    one_sided: ONE_SIDED_PUBLICATION_PROBE: OneSidedPublicationProbe,
    buffer_builds: FACTOR_BUFFER_BUILD_COUNTS: (usize, usize),
    placement_index: PLACEMENT_INDEX_PROBE: PlacementIndexProbe,
    scatter_visits: SCATTER_VISIT_PROBE: ScatterVisitProbe,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckedCompactOperation {
    Qr,
    Svd,
    Lq,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckedCompactInputObservation {
    pub operation: CheckedCompactOperation,
    pub input_pointer: usize,
    pub matrix_pointer: usize,
    pub adjoint_pointer: Option<usize>,
    pub elements: usize,
}

#[cfg(test)]
pub(crate) fn reset_checked_compact_input_observations() {
    CHECKED_COMPACT_INPUT_OBSERVATIONS.with(|observations| observations.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn checked_compact_input_observations() -> Vec<CheckedCompactInputObservation> {
    CHECKED_COMPACT_INPUT_OBSERVATIONS.with(|observations| observations.borrow().clone())
}

#[cfg(test)]
pub(super) fn record_checked_compact_input<D>(
    operation: CheckedCompactOperation,
    input: &[D],
    matrix: &[D],
    adjoint: Option<&[D]>,
) {
    CHECKED_COMPACT_INPUT_OBSERVATIONS.with(|observations| {
        observations
            .borrow_mut()
            .push(CheckedCompactInputObservation {
                operation,
                input_pointer: input.as_ptr() as usize,
                matrix_pointer: matrix.as_ptr() as usize,
                adjoint_pointer: adjoint.map(|data| data.as_ptr() as usize),
                elements: matrix.len(),
            });
    });
}

#[cfg(test)]
pub(crate) fn reset_values_matricization_fallbacks() {
    VALUES_MATRICIZATION_FALLBACKS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn values_matricization_fallbacks() -> usize {
    VALUES_MATRICIZATION_FALLBACKS.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_values_matricization_fallback() {
    VALUES_MATRICIZATION_FALLBACKS.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
thread_local! {
    pub(super) static FORCE_INPUT_PACK: Cell<bool> = const { Cell::new(false) };
    pub(super) static INPUT_PACK_BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Runs `f` with region admission disabled, so every input packs.
#[cfg(test)]
pub(crate) fn with_forced_input_pack<T>(f: impl FnOnce() -> T) -> T {
    let previous = FORCE_INPUT_PACK.with(|force| force.replace(true));
    let result = f();
    FORCE_INPUT_PACK.with(|force| force.set(previous));
    result
}

#[cfg(test)]
pub(crate) fn reset_input_pack_bytes() {
    INPUT_PACK_BYTES.with(|bytes| bytes.set(0));
}

/// Bytes allocated by `sector_matricizations{,_generic}` on this thread.
#[cfg(test)]
pub(crate) fn input_pack_bytes() -> usize {
    INPUT_PACK_BYTES.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_input_pack_bytes<D>(matricizations: &[SectorMatricization<D>]) {
    let bytes = matricizations
        .iter()
        .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
        .sum::<usize>();
    INPUT_PACK_BYTES.with(|total| total.set(total.get() + bytes));
}

#[cfg(test)]
thread_local! {
    pub(super) static FACTOR_BUFFER_BUILD_COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

#[cfg(test)]
pub(crate) fn reset_factor_buffer_build_counts_for_test() {
    FACTOR_BUFFER_BUILD_COUNTS.set((0, 0));
}

#[cfg(test)]
pub(crate) fn factor_buffer_build_counts_for_test() -> (usize, usize) {
    FACTOR_BUFFER_BUILD_COUNTS.get()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PlacementIndexProbe {
    /// Index tables allocated (one per owner call).
    pub index_builds: usize,
    /// `(matricization, side)` pairs inserted into a table.
    pub indexed_sides: usize,
    pub indexed_trees: usize,
    pub lookups: usize,
}

#[cfg(test)]
thread_local! {
    pub(super) static PLACEMENT_INDEX_PROBE: Cell<PlacementIndexProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_placement_index_probe() {
    PLACEMENT_INDEX_PROBE.set(PlacementIndexProbe::default());
}

#[cfg(test)]
pub(crate) fn placement_index_probe() -> PlacementIndexProbe {
    PLACEMENT_INDEX_PROBE.get()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ScatterVisitProbe {
    /// Output blocks scanned by the grouping pass, per side (`B`).
    pub left_grouped: usize,
    pub right_grouped: usize,
    /// Groupings built, per side (one per publication call).
    pub left_groups_built: usize,
    pub right_groups_built: usize,
    /// Output blocks iterated by the paired scatter helpers, per side (`F`).
    pub left_visits: usize,
    pub right_visits: usize,
}

#[cfg(test)]
thread_local! {
    pub(super) static SCATTER_VISIT_PROBE: Cell<ScatterVisitProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_scatter_visit_probe() {
    SCATTER_VISIT_PROBE.set(ScatterVisitProbe::default());
}

#[cfg(test)]
pub(crate) fn scatter_visit_probe() -> ScatterVisitProbe {
    SCATTER_VISIT_PROBE.get()
}

#[cfg(test)]
pub(super) fn record_scatter_visit(side: FactorSide) {
    SCATTER_VISIT_PROBE.with(|probe| {
        let mut value = probe.get();
        match side {
            FactorSide::Left => value.left_visits += 1,
            FactorSide::Right => value.right_visits += 1,
        }
        probe.set(value);
    });
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GenericPairPublicationProbe {
    pub ordered_key_validation_events: usize,
    pub fallback_row_lookups: usize,
    pub fallback_col_lookups: usize,
    pub left_owner_reused: usize,
    pub right_owner_reused: usize,
    pub left_appended_elements: usize,
    pub right_appended_elements: usize,
    pub left_scattered_elements: usize,
    pub right_scattered_elements: usize,
    pub left_scatter_calls: usize,
    pub right_scatter_calls: usize,
    pub canonical_publications: usize,
    pub fallback_publications: usize,
}

#[cfg(test)]
thread_local! {
    pub(super) static GENERIC_PAIR_PUBLICATION_PROBE: Cell<GenericPairPublicationProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_generic_pair_publication_probe() {
    GENERIC_PAIR_PUBLICATION_PROBE.set(GenericPairPublicationProbe::default());
}

#[cfg(test)]
pub(crate) fn generic_pair_publication_probe() -> GenericPairPublicationProbe {
    GENERIC_PAIR_PUBLICATION_PROBE.get()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OneSidedPublicationProbe {
    pub canonical_publications: usize,
    pub fallback_publications: usize,
    pub owner_reused: usize,
    /// Elements written by canonical publication beyond the moved first
    /// owner: appended factors plus in-place identity blocks.
    pub appended_elements: usize,
    /// Heap bytes held by publication plans (identity segment lists).
    pub plan_bytes: usize,
}

#[cfg(test)]
thread_local! {
    pub(super) static ONE_SIDED_PUBLICATION_PROBE: Cell<OneSidedPublicationProbe> = Cell::default();
}

#[cfg(test)]
pub(crate) fn reset_one_sided_publication_probe() {
    ONE_SIDED_PUBLICATION_PROBE.set(OneSidedPublicationProbe::default());
}

#[cfg(test)]
pub(crate) fn one_sided_publication_probe() -> OneSidedPublicationProbe {
    ONE_SIDED_PUBLICATION_PROBE.get()
}

#[cfg(test)]
pub(super) fn record_generic_pair_ordered_key_validation() {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.ordered_key_validation_events += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_generic_pair_ordered_key_validation() {}

#[cfg(test)]
pub(super) fn record_generic_pair_fallback_lookup(side: FactorSide) {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        match side {
            FactorSide::Left => value.fallback_row_lookups += 1,
            FactorSide::Right => value.fallback_col_lookups += 1,
        }
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_generic_pair_fallback_lookup(_side: FactorSide) {}

#[cfg(test)]
pub(super) fn record_one_sided_fallback_publication() {
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.fallback_publications += 1;
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_one_sided_fallback_publication() {}

#[cfg(test)]
pub(super) fn record_generic_pair_appended(left: usize, right: usize) {
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.left_appended_elements += left;
        value.right_appended_elements += right;
        probe.set(value);
    });
}

#[cfg(not(test))]
pub(super) fn record_generic_pair_appended(_left: usize, _right: usize) {}

#[cfg(test)]
pub(super) fn record_diagonal_bond_build<V>(spectrum: &[SectorSpectrum<V>]) {
    DIAGONAL_BOND_BUILD_PROBE.with(|probe| {
        let mut current = probe.get();
        current.calls += 1;
        current.values += spectrum
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactSvdCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
    pub owned_output_publications: usize,
    pub owned_output_owner_reused: usize,
}

#[cfg(test)]
thread_local! {
    pub(super) static COMPACT_SVD_COPY_PROBE: Cell<CompactSvdCopyProbe> = Cell::default();
    pub(super) static COMPACT_QR_COPY_PROBE: Cell<CompactQrCopyProbe> = Cell::default();
    pub(super) static EIGH_COPY_PROBE: Cell<EighCopyProbe> = Cell::default();
    pub(super) static EIGH_OWNED_VECTOR_POINTERS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    pub(super) static CHECKED_EIGH_PAIR_POINTERS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    pub(super) static CHECKED_COMPACT_SVD_STAGE_POINTERS: RefCell<Vec<(usize, usize)>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static MF_COMPACT_SVD_FALLBACK_POINTERS: RefCell<Vec<(usize, usize)>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static COMPACT_LQ_COPY_PROBE: Cell<CompactLqCopyProbe> = Cell::default();
    pub(super) static DIAGONAL_BOND_BUILD_PROBE: Cell<DiagonalBondBuildProbe> = Cell::default();
    pub(super) static VALUES_MATRICIZATION_FALLBACKS: Cell<usize> = const { Cell::new(0) };
    pub(super) static CHECKED_COMPACT_INPUT_OBSERVATIONS: RefCell<Vec<CheckedCompactInputObservation>> =
        const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn reset_checked_compact_svd_stage_pointers() {
    CHECKED_COMPACT_SVD_STAGE_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn checked_compact_svd_stage_pointers() -> Vec<(usize, usize)> {
    CHECKED_COMPACT_SVD_STAGE_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
pub(super) fn record_checked_compact_svd_stage_gauge<D>(u: &[D], vt: &[D]) {
    CHECKED_COMPACT_SVD_STAGE_POINTERS.with(|pointers| {
        pointers
            .borrow_mut()
            .push((u.as_ptr() as usize, vt.as_ptr() as usize));
    });
}

#[cfg(test)]
pub(crate) fn reset_mf_compact_svd_fallback_pointers() {
    MF_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn mf_compact_svd_fallback_pointers() -> Vec<(usize, usize)> {
    MF_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
pub(super) fn record_mf_compact_svd_fallback_gauge<D>(u: &[D], vt: &[D]) {
    MF_COMPACT_SVD_FALLBACK_POINTERS.with(|pointers| {
        pointers
            .borrow_mut()
            .push((u.as_ptr() as usize, vt.as_ptr() as usize));
    });
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DiagonalBondBuildProbe {
    pub calls: usize,
    pub values: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_svd_copy_probe() {
    COMPACT_SVD_COPY_PROBE.with(|probe| probe.set(CompactSvdCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_svd_copy_probe() -> CompactSvdCopyProbe {
    COMPACT_SVD_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_diagonal_bond_build_probe() {
    DIAGONAL_BOND_BUILD_PROBE.with(|probe| probe.set(DiagonalBondBuildProbe::default()));
}

#[cfg(test)]
pub(crate) fn diagonal_bond_build_probe() -> DiagonalBondBuildProbe {
    DIAGONAL_BOND_BUILD_PROBE.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_compact_svd_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_SVD_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_svd_output_scatter<D>(elements: usize) {
    COMPACT_SVD_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += 1;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactQrCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
    pub owned_output_publications: usize,
    pub owned_output_owner_reused: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_qr_copy_probe() {
    COMPACT_QR_COPY_PROBE.with(|probe| probe.set(CompactQrCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_qr_copy_probe() -> CompactQrCopyProbe {
    COMPACT_QR_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompactLqCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
    pub scratch_buffer_count: usize,
    pub scratch_capacity_bytes: usize,
    pub adjoint_scratch_fill_calls: usize,
    pub adjoint_scratch_fill_bytes: usize,
    pub final_adjoint_copy_calls: usize,
    pub final_adjoint_copy_bytes: usize,
    pub output_prefill_bytes: usize,
}

#[cfg(test)]
pub(crate) fn reset_compact_lq_copy_probe() {
    COMPACT_LQ_COPY_PROBE.with(|probe| probe.set(CompactLqCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn compact_lq_copy_probe() -> CompactLqCopyProbe {
    COMPACT_LQ_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_compact_qr_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_qr_output_scatter<D>(elements: usize) {
    record_compact_qr_output_scatter_work::<D>(1, elements);
}

#[cfg(test)]
pub(super) fn record_compact_qr_output_scatter_work<D>(calls: usize, elements: usize) {
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += calls;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_output_scatter<D>(elements: usize) {
    record_compact_lq_output_scatter_work::<D>(1, elements);
}

#[cfg(test)]
pub(super) fn record_compact_lq_output_scatter_work<D>(calls: usize, elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += calls;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_scratch<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.scratch_buffer_count += 1;
        current.scratch_capacity_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_adjoint_fill<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.adjoint_scratch_fill_calls += 1;
        current.adjoint_scratch_fill_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_output_prefill<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_prefill_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_compact_lq_final_adjoint_copy<D>(elements: usize) {
    COMPACT_LQ_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.final_adjoint_copy_calls += 1;
        current.final_adjoint_copy_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct EighCopyProbe {
    pub input_pack_calls: usize,
    pub input_pack_bytes: usize,
    pub output_scatter_calls: usize,
    pub output_scatter_bytes: usize,
}

#[cfg(test)]
pub(crate) fn reset_eigh_copy_probe() {
    EIGH_COPY_PROBE.with(|probe| probe.set(EighCopyProbe::default()));
}

#[cfg(test)]
pub(crate) fn eigh_copy_probe() -> EighCopyProbe {
    EIGH_COPY_PROBE.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_eigh_owned_vector_pointers() {
    EIGH_OWNED_VECTOR_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn eigh_owned_vector_pointers() -> Vec<usize> {
    EIGH_OWNED_VECTOR_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
pub(super) fn record_eigh_owned_vector_before_scatter<D>(vectors: &[D]) {
    EIGH_OWNED_VECTOR_POINTERS.with(|pointers| {
        pointers.borrow_mut().push(vectors.as_ptr() as usize);
    });
}

#[cfg(test)]
pub(crate) fn reset_checked_eigh_pair_pointers() {
    CHECKED_EIGH_PAIR_POINTERS.with(|pointers| pointers.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn checked_eigh_pair_pointers() -> Vec<usize> {
    CHECKED_EIGH_PAIR_POINTERS.with(|pointers| pointers.borrow().clone())
}

#[cfg(test)]
pub(super) fn record_checked_eigh_pair_pointers<D>(pairs: &[FactorPair<D>]) {
    CHECKED_EIGH_PAIR_POINTERS.with(|pointers| {
        *pointers.borrow_mut() = pairs
            .iter()
            .map(|pair| pair.left.as_ptr() as usize)
            .collect();
    });
}

#[cfg(test)]
pub(super) fn record_eigh_input_pack<D>(matricizations: &[SectorMatricization<D>]) {
    EIGH_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.input_pack_calls += matricizations.len();
        current.input_pack_bytes += matricizations
            .iter()
            .map(|matrix| matrix.data.len() * std::mem::size_of::<D>())
            .sum::<usize>();
        probe.set(current);
    });
}

#[cfg(test)]
pub(super) fn record_eigh_output_scatter<D>(elements: usize) {
    EIGH_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.output_scatter_calls += 1;
        current.output_scatter_bytes += elements * std::mem::size_of::<D>();
        probe.set(current);
    });
}
