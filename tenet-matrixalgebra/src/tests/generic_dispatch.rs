//! Cross-cutting checked-Generic factor-plan, tree-stacking and provider
//! dispatch tests that span more than one factorization (#1596 split of
//! tests.rs).

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValuesOperation {
    Svd,
    Eigh,
    Eig,
}

struct FailSecondValues {
    inner: tenet_dense::DefaultDenseExecutor,
    operation: ValuesOperation,
    calls: usize,
}

impl FailSecondValues {
    fn new(operation: ValuesOperation) -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            operation,
            calls: 0,
        }
    }

    fn fail(&mut self, operation: ValuesOperation) -> Result<(), DenseError> {
        assert_eq!(self.operation, operation);
        self.calls += 1;
        if self.calls == 2 {
            Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "values",
                message: "injected second-sector failure".to_string(),
            })
        } else {
            Ok(())
        }
    }
}

impl DenseExecutor for FailSecondValues {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.fail(ValuesOperation::Svd)?;
        self.inner.svd_vals(input)
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.fail(ValuesOperation::Eigh)?;
        self.inner.eigh_vals(input)
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.fail(ValuesOperation::Eig)?;
        self.inner.eig_vals(input)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises values-only operations")
    }
}

fn scale_vt_rows_by_singular_values<const NIN: usize>(
    vt: &mut TensorMap<f64, 1, NIN>,
    singular_values: &[SectorSpectrum],
) {
    let structure = std::sync::Arc::clone(vt.structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = key.codomain_tree().coupled();
        let values = &singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .expect("singular values for every Vt sector")
            .values;
        let shape = block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = block.offset()
                + multi_index
                    .iter()
                    .zip(block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            vt.data_mut()[position] *= values[multi_index[0]];
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
}

fn run_tsvd_reconstruction_case<R>(rule: &R, sectors: &[SectorId], coupled_layout: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Clone,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(rule).len();
    let dense = TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
    let shapes = vec![vec![degeneracy; 4]; key_count];
    let space = if coupled_layout {
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense, homspace, rule, shapes).unwrap()
    } else {
        FusionTensorMapSpace::from_degeneracy_shapes(dense, homspace, rule, shapes).unwrap()
    };
    let len = space.required_len().unwrap();
    let tensor = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 7 + 3) % 23) as f64 * 0.5 - 5.0)
            .collect(),
        space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule.clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_legacy_shapes(svd.u.space());
    assert_factor_layout_matches_legacy_shapes(svd.s.space());
    assert_factor_layout_matches_legacy_shapes(svd.vh.space());

    for entry in &svd.singular_values {
        for pair in entry.values.windows(2) {
            assert!(
                pair[0] >= pair[1] - 1e-12,
                "singular values must be descending in sector {:?}",
                entry.sector
            );
        }
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }

    let mut scaled_vt = svd.vh.tensor().clone();
    scale_vt_rows_by_singular_values(&mut scaled_vt, &svd.singular_values);

    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; len],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut reconstructed,
            &svd.u,
            &scaled_vt,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn tsvd_fusion_reconstructs_z2_tensor_packed_layout() {
    run_tsvd_reconstruction_case(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)], false);
}

#[test]
fn tsvd_fusion_reconstructs_z2_tensor_coupled_layout() {
    run_tsvd_reconstruction_case(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)], true);
}

fn one_sector_generic_factorization_input(
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
    let provider = Arc::new(FactorGenericRule);
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(x, 2)], false),
            SectorLeg::new([(x, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(x, 3)], false),
            SectorLeg::new([(vacuum, 1)], false),
        ]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, homspace).unwrap();
    let data = (0..space.space().required_len().unwrap())
        .map(|index| 1.0 + index as f64 / 8.0)
        .collect();
    (space, data)
}

fn assert_generic_factor_close(
    actual: &BoundDynFactor<FactorGenericRule, f64>,
    expected: &BoundDynFactor<FactorGenericRule, f64>,
) {
    assert_eq!(
        actual.space().space().homspace(),
        expected.space().space().homspace()
    );
    assert_eq!(actual.data().len(), expected.data().len());
    for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn provider_neutral_generic_compact_factorizations_remain_covered() {
    let (space, data) = generic_factorization_input();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    assert!(!svd_vals_dyn_generic(&mut dense, &input).unwrap().is_empty());
    let (u, vh, values) = svd_compact_factors_dyn_generic(&mut dense, &input).unwrap();
    assert!(!values.is_empty());
    assert!(Arc::ptr_eq(u.space().provider_arc(), space.provider_arc()));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), space.provider_arc()));
    qr_compact_dyn_generic(&mut dense, &input).unwrap();
    lq_compact_dyn_generic(&mut dense, &input).unwrap();
}

#[test]
fn provider_neutral_generic_factorizations_keep_the_strided_fallback() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, padded_data) =
        padded_generic_factorization_input(&canonical_space, &canonical_data);
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let canonical_values = svd_vals_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_values = svd_vals_dyn_generic(&mut dense, &padded).unwrap();
    assert_real_spectra_close(&padded_values, &canonical_values);

    let canonical_svd = svd_compact_factors_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_svd = svd_compact_factors_dyn_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_svd.0, &canonical_svd.0);
    assert_generic_factor_close(&padded_svd.1, &canonical_svd.1);
    assert_real_spectra_close(&padded_svd.2, &canonical_svd.2);

    let canonical_qr = qr_compact_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_qr = qr_compact_dyn_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_qr.q, &canonical_qr.q);
    assert_generic_factor_close(&padded_qr.r, &canonical_qr.r);

    let canonical_lq = lq_compact_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_lq = lq_compact_dyn_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_lq.l, &canonical_lq.l);
    assert_generic_factor_close(&padded_lq.q, &canonical_lq.q);
}

#[test]
fn generic_pair_publication_keeps_reordered_tree_scatter_fallback() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (reordered_space, reordered_data) =
        expert_generic_factorization_input(&canonical_space, &canonical_data, true);
    let reordered = BoundDynamicTensorRef::try_new(&reordered_space, &reordered_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_generic_pair_publication_probe();
    crate::factorize::reset_compact_qr_copy_probe();
    let actual_qr = qr_compact_dyn_generic(&mut dense, &reordered).unwrap();
    assert!(!actual_qr.q.data().is_empty());
    assert!(!actual_qr.r.data().is_empty());
    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert!(probe.left_scattered_elements > 0);
    assert!(probe.right_scattered_elements > 0);
    let qr_copy = crate::factorize::compact_qr_copy_probe();
    assert_eq!(
        qr_copy.output_scatter_calls,
        probe.left_scatter_calls + probe.right_scatter_calls
    );
    assert_eq!(
        qr_copy.output_scatter_bytes,
        (actual_qr.q.data().len() + actual_qr.r.data().len()) * std::mem::size_of::<f64>()
    );

    crate::factorize::reset_generic_pair_publication_probe();
    crate::factorize::reset_compact_lq_copy_probe();
    let actual_lq = lq_compact_dyn_generic(&mut dense, &reordered).unwrap();
    assert!(!actual_lq.l.data().is_empty());
    assert!(!actual_lq.q.data().is_empty());
    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert!(probe.left_scattered_elements > 0);
    assert!(probe.right_scattered_elements > 0);
    let lq_copy = crate::factorize::compact_lq_copy_probe();
    assert_eq!(
        lq_copy.output_scatter_calls,
        probe.left_scatter_calls + probe.right_scatter_calls
    );
    assert_eq!(
        lq_copy.output_scatter_bytes,
        (actual_lq.l.data().len() + actual_lq.q.data().len()) * std::mem::size_of::<f64>()
    );
}

#[test]
fn checked_generic_pair_publication_reuses_one_sector_owners() {
    let (source, data) = one_sector_generic_factorization_input();
    let (provider, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_generic_pair_publication_probe();

    let Qr { q: left, r: right } = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();

    let probe = crate::factorize::generic_pair_publication_probe();
    let blocks = left.space().space().structure().block_count()
        + right.space().space().structure().block_count();
    assert_eq!(probe.canonical_publications, 1);
    assert_eq!(probe.fallback_publications, 0);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (1, 1));
    assert_eq!(
        (probe.left_appended_elements, probe.right_appended_elements),
        (0, 0)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
    // The checked route validates each staged key once (one cursor event per
    // block) and commits the structure it validated, so the admitted-key
    // equality of a two-source construction (formerly one visit and one more
    // event per block: `blocks` visits, `2 * blocks` events) no longer runs.
    assert_eq!(probe.output_blocks_visited, 0);
    assert_eq!(probe.ordered_key_validation_events, blocks);
    assert_eq!(
        (probe.fallback_row_lookups, probe.fallback_col_lookups),
        (0, 0)
    );
    assert!(Arc::ptr_eq(left.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(right.space().provider_arc(), &provider));
}

fn assert_pair_reconstructs_checked_literal<R, D>(
    left: &BoundDynFactor<R, D>,
    right: &BoundDynFactor<R, D>,
    complex: bool,
) where
    D: FactorScalar,
{
    let left_regions = left
        .space()
        .space()
        .structure()
        .coupled_sector_regions(left.space().space().nout())
        .unwrap()
        .unwrap();
    let right_regions = right
        .space()
        .space()
        .structure()
        .coupled_sector_regions(right.space().space().nout())
        .unwrap()
        .unwrap();
    for sector in [SectorId::new(0), SectorId::new(1)] {
        let left_region = left_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let right_region = right_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let (rows, cols, expected) = checked_svd_matrix(sector, complex);
        assert_eq!((left_region.rows(), right_region.cols()), (rows, cols));
        assert_eq!(left_region.cols(), right_region.rows());
        let kept = left_region.cols();
        for col in 0..cols {
            for row in 0..rows {
                let actual = (0..kept).fold(D::zero(), |sum, bond| {
                    sum + left.data()[left_region.range().start + row + rows * bond]
                        * right.data()[right_region.range().start + bond + kept * col]
                });
                assert!((actual.widen_complex() - expected[row + rows * col]).norm() < 1.0e-10);
            }
        }
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn staged_generic_pair_callers_publish_canonical_owned_payloads() {
    let (source, data) = one_sector_generic_factorization_input();
    let (padded_space, padded_data) = padded_generic_factorization_input(&source, &data);
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let checked_input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_generic_pair_publication_probe();
    crate::factorize::reset_one_sided_publication_probe();

    qr_compact_dyn_generic(&mut dense, &padded).unwrap();
    lq_compact_dyn_generic(&mut dense, &padded).unwrap();
    qr_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    svd_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    lq_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    qr_full_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    lq_full_dyn_checked_generic(&mut dense, &checked_input).unwrap();

    // Full QR/LQ publish each side through the side-aware builder (#1524).
    let one_sided = crate::factorize::one_sided_publication_probe();
    assert_eq!(
        (
            one_sided.canonical_publications,
            one_sided.fallback_publications
        ),
        (4, 0)
    );
    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(probe.canonical_publications, 5);
    assert_eq!(probe.fallback_publications, 0);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (5, 5));
    assert_eq!(
        (probe.left_appended_elements, probe.right_appended_elements),
        (0, 0)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
}

fn assert_checked_generic_pair_publication_appends_multiple_literal_sectors<D>(complex: bool)
where
    D: FactorScalar,
{
    let (provider, space, data) = checked_svd_truncation_input::<D>(complex);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_generic_pair_publication_probe();
    crate::factorize::reset_one_sided_publication_probe();

    let qr = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&qr.q, &qr.r, complex);
    let qr_probe = crate::factorize::generic_pair_publication_probe();
    let qr_blocks = qr.q.space().space().structure().block_count()
        + qr.r.space().space().structure().block_count();
    // Formerly `qr_blocks` visits and `2 * qr_blocks` events: the checked
    // route no longer cross-checks a separate key list against its structure.
    assert_eq!(qr_probe.output_blocks_visited, 0);
    assert_eq!(qr_probe.ordered_key_validation_events, qr_blocks);
    assert_eq!(
        (qr_probe.fallback_row_lookups, qr_probe.fallback_col_lookups),
        (0, 0)
    );
    svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let lq = lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&lq.l, &lq.q, complex);
    let full_qr = qr_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_qr.q, &full_qr.r, complex);
    let full_lq = lq_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_lq.l, &full_lq.q, complex);

    // Full QR/LQ publish each side through the side-aware builder (#1524).
    let one_sided = crate::factorize::one_sided_publication_probe();
    assert_eq!(
        (
            one_sided.canonical_publications,
            one_sided.fallback_publications
        ),
        (4, 0)
    );
    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(probe.canonical_publications, 3);
    assert_eq!(probe.fallback_publications, 0);
    assert!(probe.left_appended_elements > 0);
    assert!(probe.right_appended_elements > 0);
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
    assert!(Arc::ptr_eq(qr.q.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(qr.r.space().provider_arc(), &provider));
}

#[test]
fn checked_generic_pair_publication_appends_multiple_literal_sectors() {
    assert_checked_generic_pair_publication_appends_multiple_literal_sectors::<f64>(false);
    assert_checked_generic_pair_publication_appends_multiple_literal_sectors::<Complex64>(true);
}

#[derive(Debug)]
struct ValuesInputObservation {
    operation: ValuesOperation,
    pointer: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    values: Vec<Complex64>,
}

struct ValuesInputSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    observations: Vec<ValuesInputObservation>,
}

#[derive(Debug)]
struct CompactInputObservation {
    pointer: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    values: Vec<Complex64>,
}

struct CompactInputSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    operation: crate::factorize::CheckedCompactOperation,
    observations: Vec<CompactInputObservation>,
}

impl CompactInputSpy {
    fn new(operation: crate::factorize::CheckedCompactOperation) -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            operation,
            observations: Vec::new(),
        }
    }

    fn observe(&mut self, input: DenseRead<'_>) {
        let (pointer, strides, offset, values) = match input {
            DenseRead::F64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data()
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            DenseRead::C64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data().to_vec(),
            ),
            _ => panic!("checked Generic compact fixture must be f64 or c64"),
        };
        self.observations.push(CompactInputObservation {
            pointer,
            shape: input.shape().to_vec(),
            strides,
            offset,
            values,
        });
    }
}

impl DenseExecutor for CompactInputSpy {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        assert_eq!(
            self.operation,
            crate::factorize::CheckedCompactOperation::Svd
        );
        self.observe(input);
        self.inner.svd(input)
    }

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let _ = (input, u, s, vt);
        panic!("checked compact SVD must use the owned API")
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        assert!(matches!(
            self.operation,
            crate::factorize::CheckedCompactOperation::Qr
                | crate::factorize::CheckedCompactOperation::Lq
        ));
        self.observe(input);
        self.inner.qr(input)
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let _ = (input, q, r);
        panic!("compact QR/LQ must not use qr_into")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises compact QR/SVD/LQ")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises compact QR/SVD/LQ")
    }
}

impl Default for ValuesInputSpy {
    fn default() -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            observations: Vec::new(),
        }
    }
}

impl ValuesInputSpy {
    fn observe(&mut self, operation: ValuesOperation, input: DenseRead<'_>) {
        let (pointer, strides, offset, values) = match input {
            DenseRead::F64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data()
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            DenseRead::C64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data().to_vec(),
            ),
            _ => panic!("checked Generic values fixture must be f64 or c64"),
        };
        self.observations.push(ValuesInputObservation {
            operation,
            pointer,
            shape: input.shape().to_vec(),
            strides,
            offset,
            values,
        });
    }
}

impl DenseExecutor for ValuesInputSpy {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.observe(ValuesOperation::Svd, input);
        self.inner.svd_vals(input)
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.observe(ValuesOperation::Eigh, input);
        self.inner.eigh_vals(input)
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.observe(ValuesOperation::Eig, input);
        self.inner.eig_vals(input)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises values-only operations")
    }
}

fn assert_borrowed_values_inputs<D: FactorScalar>(
    observations: &[ValuesInputObservation],
    operation: ValuesOperation,
    data: &[D],
    regions: &[tenet_core::CoupledSectorRegion],
) {
    assert_eq!(observations.len(), regions.len());
    for (observation, region) in observations.iter().zip(regions) {
        assert_eq!(observation.operation, operation);
        assert_eq!(observation.shape, [region.rows(), region.cols()]);
        assert_eq!(observation.strides, [1, region.rows()]);
        assert_eq!(observation.offset, 0);
        assert_eq!(
            observation.pointer,
            data.as_ptr() as usize + region.range().start * std::mem::size_of::<D>()
        );
        let expected = data[region.range()]
            .iter()
            .map(|&value| value.widen_complex())
            .collect::<Vec<_>>();
        assert_eq!(observation.values, expected);
    }
}

fn assert_compact_input_observations<D: FactorScalar>(
    operation: crate::factorize::CheckedCompactOperation,
    dense: &[CompactInputObservation],
    data: &[D],
    regions: &[tenet_core::CoupledSectorRegion],
) {
    let lowering = crate::factorize::checked_compact_input_observations();
    assert_eq!(lowering.len(), regions.len());
    assert_eq!(dense.len(), regions.len());
    for ((lowering, dense), region) in lowering.iter().zip(dense).zip(regions) {
        let source_pointer =
            data.as_ptr() as usize + region.range().start * std::mem::size_of::<D>();
        assert_eq!(lowering.operation, operation);
        assert_eq!(lowering.input_pointer, data.as_ptr() as usize);
        assert_eq!(lowering.matrix_pointer, source_pointer);
        assert_eq!(lowering.elements, region.range().len());
        assert_eq!(dense.offset, 0);
        match operation {
            crate::factorize::CheckedCompactOperation::Qr
            | crate::factorize::CheckedCompactOperation::Svd => {
                assert_eq!(lowering.adjoint_pointer, None);
                assert_eq!(dense.pointer, source_pointer);
                assert_eq!(dense.shape, [region.rows(), region.cols()]);
                assert_eq!(dense.strides, [1, region.rows()]);
                assert_eq!(
                    dense.values,
                    data[region.range()]
                        .iter()
                        .map(|&value| value.widen_complex())
                        .collect::<Vec<_>>()
                );
            }
            crate::factorize::CheckedCompactOperation::Lq => {
                assert_eq!(lowering.adjoint_pointer, Some(dense.pointer));
                assert_ne!(dense.pointer, source_pointer);
                assert_eq!(dense.shape, [region.cols(), region.rows()]);
                assert_eq!(dense.strides, [1, region.cols()]);
                let source = &data[region.range()];
                let expected = (0..region.rows())
                    .flat_map(|column| {
                        (0..region.cols()).map(move |row| {
                            source[column + region.rows() * row].widen_complex().conj()
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(dense.values, expected);
            }
        }
    }
}

fn assert_checked_compact_input_borrowing<D: FactorScalar>(
    space: BoundDynamicFusionMapSpace<LateGenericSpy>,
    data: Vec<D>,
) {
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = data.clone();

    crate::factorize::reset_compact_qr_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let mut qr = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Qr);
    let factors = qr_compact_dyn_checked_generic(&mut qr, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.q, None, &factors.r);
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Qr,
        &qr.observations,
        &data,
        &regions,
    );
    assert_eq!(
        crate::factorize::compact_qr_copy_probe().input_pack_calls,
        0
    );

    crate::factorize::reset_compact_svd_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let mut svd = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Svd);
    let factors = svd_compact_dyn_checked_generic(&mut svd, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.u, Some(&factors.s), &factors.vh);
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Svd,
        &svd.observations,
        &data,
        &regions,
    );
    assert_eq!(
        crate::factorize::compact_svd_copy_probe().input_pack_calls,
        0
    );

    crate::factorize::reset_compact_lq_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let mut lq = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Lq);
    let factors = lq_compact_dyn_checked_generic(&mut lq, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.l, None, &factors.q);
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Lq,
        &lq.observations,
        &data,
        &regions,
    );
    let probe = crate::factorize::compact_lq_copy_probe();
    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.adjoint_scratch_fill_calls, regions.len());
    assert_eq!(
        probe.adjoint_scratch_fill_bytes,
        data.len() * std::mem::size_of::<D>()
    );
    assert!(data == before);
}

#[test]
fn checked_generic_compact_factors_borrow_real_and_complex_canonical_inputs() {
    let (_, real_space, real_data) = checked_svd_truncation_input::<f64>(false);
    assert_checked_compact_input_borrowing(real_space, real_data);
    let (_, complex_space, complex_data) = checked_svd_truncation_input::<Complex64>(true);
    assert_checked_compact_input_borrowing(complex_space, complex_data);
    let (_, wide_space, wide_data) = checked_svd_wide_input::<Complex64>();
    assert_checked_compact_input_borrowing(wide_space, wide_data);
}

#[test]
fn checked_generic_compact_geometry_preserves_complete_multi_tree_identity() {
    let (source, _, data) = generic_values_endomorphism_input();
    let (_, space) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap();
    let multiplicity_region = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap();
    assert_eq!(multiplicity_region.row_trees().len(), 2);
    let first = multiplicity_region.row_trees()[0].tree();
    let second = multiplicity_region.row_trees()[1].tree();
    assert_eq!(first.uncoupled(), second.uncoupled());
    assert_eq!(first.is_dual(), second.is_dual());
    assert_eq!(first.innerlines(), second.innerlines());
    assert_ne!(first.vertices(), second.vertices());

    crate::factorize::reset_checked_compact_input_observations();
    let mut qr = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Qr);
    let qr_factors = qr_compact_dyn_checked_generic(&mut qr, &input).unwrap();
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Qr,
        &qr.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(&input, &qr_factors.q, None, &qr_factors.r);

    crate::factorize::reset_checked_compact_input_observations();
    let mut svd = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Svd);
    let svd_factors = svd_compact_dyn_checked_generic(&mut svd, &input).unwrap();
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Svd,
        &svd.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(
        &input,
        &svd_factors.u,
        Some(&svd_factors.s),
        &svd_factors.vh,
    );

    crate::factorize::reset_checked_compact_input_observations();
    let mut lq = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Lq);
    let lq_factors = lq_compact_dyn_checked_generic(&mut lq, &input).unwrap();
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Lq,
        &lq.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(&input, &lq_factors.l, None, &lq_factors.q);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_factors_keep_padded_reordered_input_pack() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (expert_space, expert_data) =
        expert_generic_factorization_input(&canonical_space, &canonical_data, true);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let expert_space = BoundDynamicFusionMapSpace::bind_generic(
        expert_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let expert = BoundDynamicTensorRef::try_new(&expert_space, &expert_data).unwrap();
    let canonical_before = canonical_data.clone();
    let expert_before = expert_data.clone();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_qr_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let actual_qr = qr_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(&expert, &actual_qr.q, None, &actual_qr.r);
    assert!(crate::factorize::compact_qr_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_svd_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let actual_svd = svd_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(
        &expert,
        &actual_svd.u,
        Some(&actual_svd.s),
        &actual_svd.vh,
    );
    assert!(crate::factorize::compact_svd_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_lq_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let actual_lq = lq_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(&expert, &actual_lq.l, None, &actual_lq.q);
    let lq_probe = crate::factorize::compact_lq_copy_probe();
    assert!(lq_probe.input_pack_calls > 0);
    assert!(lq_probe.adjoint_scratch_fill_calls > 0);

    let start = expert_data.as_ptr() as usize;
    let end = start + expert_data.len() * std::mem::size_of::<f64>();
    for observation in crate::factorize::checked_compact_input_observations() {
        assert_eq!(observation.input_pointer, start);
        assert!(observation.matrix_pointer < start || observation.matrix_pointer >= end);
    }
    assert!(canonical_data == canonical_before);
    assert!(expert_data == expert_before);
    assert!(Arc::ptr_eq(actual_qr.q.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(actual_svd.s.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(actual_lq.q.space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_interleaved_fallback_keeps_literal_matrix_order() {
    let (canonical, _, canonical_data) = generic_values_endomorphism_input();
    let (expert_space, expert_data) =
        interleaved_generic_endomorphism_input(&canonical, &canonical_data);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let expert_space = BoundDynamicFusionMapSpace::bind_generic(
        expert_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let expert = BoundDynamicTensorRef::try_new(&expert_space, &expert_data).unwrap();
    let before = expert_data.clone();
    let direct = [
        vec![
            Complex64::new(3.0, -1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0),
        ],
        vec![Complex64::new(2.0, -1.0)],
    ];
    let adjoint = [
        vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ],
        vec![Complex64::new(2.0, 1.0)],
    ];

    crate::factorize::reset_compact_qr_copy_probe();
    let mut qr = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Qr);
    let factors = qr_compact_dyn_checked_generic(&mut qr, &expert).unwrap();
    assert_eq!(
        qr.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        direct
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.q, None, &factors.r);
    assert!(crate::factorize::compact_qr_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_svd_copy_probe();
    let mut svd = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Svd);
    let factors = svd_compact_dyn_checked_generic(&mut svd, &expert).unwrap();
    assert_eq!(
        svd.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        direct
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.u, Some(&factors.s), &factors.vh);
    assert!(crate::factorize::compact_svd_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_lq_copy_probe();
    let mut lq = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Lq);
    let factors = lq_compact_dyn_checked_generic(&mut lq, &expert).unwrap();
    assert_eq!(
        lq.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        adjoint
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.l, None, &factors.q);
    let probe = crate::factorize::compact_lq_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.adjoint_scratch_fill_calls > 0);
    assert_eq!(expert_data, before);
}

fn assert_real_spectra_by_sector_close(actual: &[SectorSpectrum], expected: &[SectorSpectrum]) {
    assert_eq!(actual.len(), expected.len());
    for expected in expected {
        let actual = actual
            .iter()
            .find(|actual| actual.sector == expected.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}

fn assert_complex_spectra_by_sector_close(
    actual: &[SectorSpectrum<Complex64>],
    expected: &[SectorSpectrum<Complex64>],
) {
    assert_eq!(actual.len(), expected.len());
    for expected in expected {
        let actual = actual
            .iter()
            .find(|actual| actual.sector == expected.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).norm() < 1.0e-10);
        }
    }
}

#[test]
fn checked_only_generic_values_borrow_canonical_input_regions() {
    let (_, rectangular_space, rectangular_data) = checked_svd_truncation_input::<f64>(false);
    let (rectangular_provider, rectangular_space) = bind_checked_only(&rectangular_space);
    let rectangular_calls = rectangular_provider.calls.get();
    let rectangular_before = rectangular_data.clone();
    let rectangular_regions = rectangular_space
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let mut spy = ValuesInputSpy::default();
    crate::factorize::reset_values_matricization_fallbacks();
    let spectra = svd_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&rectangular_space, &rectangular_data).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(
        &spy.observations,
        ValuesOperation::Svd,
        &rectangular_data,
        &rectangular_regions,
    );
    assert_real_spectra_by_sector_close(
        &spectra,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![4.0, 1.0],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![3.0, 2.0],
            },
        ],
    );
    assert_eq!(rectangular_provider.calls.get(), rectangular_calls);
    assert_eq!(rectangular_data, rectangular_before);

    let (endomorphism_space, hermitian, general) = generic_values_endomorphism_input();
    let (endomorphism_provider, endomorphism_space) = bind_checked_only(&endomorphism_space);
    let provider_calls = endomorphism_provider.calls.get();
    let regions = endomorphism_space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let multiplicity_region = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap();
    assert_eq!(multiplicity_region.row_trees().len(), 2);
    assert_eq!(multiplicity_region.col_trees().len(), 2);
    let hermitian_before = hermitian.clone();
    let general_before = general.clone();
    spy.observations.clear();
    let eigh = eigh_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&endomorphism_space, &hermitian).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(
        &spy.observations,
        ValuesOperation::Eigh,
        &hermitian,
        &regions,
    );
    assert_real_spectra_by_sector_close(
        &eigh,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![-4.0],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![3.0, 1.0],
            },
        ],
    );
    spy.observations.clear();
    let eig = eig_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&endomorphism_space, &general).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(&spy.observations, ValuesOperation::Eig, &general, &regions);
    assert_complex_spectra_by_sector_close(
        &eig,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![Complex64::new(2.0, -1.0)],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![Complex64::new(3.0, -1.0), Complex64::new(1.0, 1.0)],
            },
        ],
    );
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);
    assert_eq!(endomorphism_provider.calls.get(), provider_calls);
    assert_eq!(hermitian, hermitian_before);
    assert_eq!(general, general_before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_values_keep_padded_reordered_fallback() {
    let (canonical_space, hermitian, general) = generic_values_endomorphism_input();
    let (padded_space, padded_hermitian) =
        padded_reordered_generic_endomorphism_input(&canonical_space, &hermitian);
    let (_, padded_general) =
        padded_reordered_generic_endomorphism_input(&canonical_space, &general);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let canonical_space = BoundDynamicFusionMapSpace::bind_generic(
        canonical_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let padded_space = BoundDynamicFusionMapSpace::bind_generic(
        padded_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    assert!(padded_space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let provider_calls = provider.calls.get();
    let canonical_general = BoundDynamicTensorRef::try_new(&canonical_space, &general).unwrap();
    let canonical_hermitian = BoundDynamicTensorRef::try_new(&canonical_space, &hermitian).unwrap();
    let padded_general_input =
        BoundDynamicTensorRef::try_new(&padded_space, &padded_general).unwrap();
    let padded_hermitian_input =
        BoundDynamicTensorRef::try_new(&padded_space, &padded_hermitian).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let expected_svd = svd_vals_dyn_checked_generic(&mut dense, &canonical_general).unwrap();
    let expected_eigh = eigh_vals_dyn_checked_generic(&mut dense, &canonical_hermitian).unwrap();
    let expected_eig = eig_vals_dyn_checked_generic(&mut dense, &canonical_general).unwrap();

    let general_before = padded_general.clone();
    let hermitian_before = padded_hermitian.clone();
    let mut spy = ValuesInputSpy::default();
    crate::factorize::reset_values_matricization_fallbacks();
    let actual_svd = svd_vals_dyn_checked_generic(&mut spy, &padded_general_input).unwrap();
    let actual_eigh = eigh_vals_dyn_checked_generic(&mut spy, &padded_hermitian_input).unwrap();
    let actual_eig = eig_vals_dyn_checked_generic(&mut spy, &padded_general_input).unwrap();
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 3);
    assert_real_spectra_by_sector_close(&actual_svd, &expected_svd);
    assert_real_spectra_by_sector_close(&actual_eigh, &expected_eigh);
    assert_complex_spectra_by_sector_close(&actual_eig, &expected_eig);
    assert_eq!(provider.calls.get(), provider_calls);
    assert_eq!(padded_general, general_before);
    assert_eq!(padded_hermitian, hermitian_before);

    let general_start = padded_general.as_ptr() as usize;
    let general_end = general_start + std::mem::size_of_val(padded_general.as_slice());
    let hermitian_start = padded_hermitian.as_ptr() as usize;
    let hermitian_end = hermitian_start + std::mem::size_of_val(padded_hermitian.as_slice());
    for observation in &spy.observations {
        let (start, end) = if observation.operation == ValuesOperation::Eigh {
            (hermitian_start, hermitian_end)
        } else {
            (general_start, general_end)
        };
        assert!(observation.pointer < start || observation.pointer >= end);
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_only_generic_values_preserve_empty_scalar_and_shape_boundaries() {
    let x = SectorId::new(1);
    let empty_leg = SectorLeg::new([(x, 0)], false);
    let empty_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([empty_leg.clone()]),
        FusionProductSpace::new([empty_leg]),
    );
    let empty_provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let empty_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&empty_provider),
        empty_homspace,
    )
    .unwrap();
    let empty_calls = empty_provider.calls.get();
    let empty_data: [f64; 0] = [];
    let empty = BoundDynamicTensorRef::try_new(&empty_space, &empty_data).unwrap();
    let mut reject = RejectExecutorCalls;
    assert!(svd_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    assert!(eigh_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    let empty_eigh = eigh_full_dyn_checked_generic(&mut reject, &empty).unwrap();
    assert!(empty_eigh.v().data().is_empty());
    assert!(empty_eigh.eigenvalues().is_empty());
    assert!(eig_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    assert_eq!(empty_provider.calls.get(), empty_calls);

    let scalar_provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let scalar_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&scalar_provider),
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
    )
    .unwrap();
    let scalar_calls = scalar_provider.calls.get();
    let scalar_data = [-3.0];
    let scalar = BoundDynamicTensorRef::try_new(&scalar_space, &scalar_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    assert_eq!(
        svd_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [3.0]
    );
    assert_eq!(
        eigh_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [-3.0]
    );
    assert_eq!(
        eig_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [Complex64::new(-3.0, 0.0)]
    );
    assert_eq!(scalar_provider.calls.get(), scalar_calls);

    let (_, rectangular, data) = checked_svd_truncation_input::<f64>(false);
    let (rectangular_provider, rectangular) = bind_checked_only(&rectangular);
    let rectangular_calls = rectangular_provider.calls.get();
    let rectangular = BoundDynamicTensorRef::try_new(&rectangular, &data).unwrap();
    assert!(matches!(
        eigh_vals_dyn_checked_generic(&mut reject, &rectangular),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope { .. }
        ))
    ));
    assert!(matches!(
        eig_vals_dyn_checked_generic(&mut reject, &rectangular),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope { .. }
        ))
    ));
    assert_eq!(rectangular_provider.calls.get(), rectangular_calls);
}

#[test]
fn checked_generic_factor_plan_late_failure_precedes_commit() {
    let (space, _data) = generic_factorization_input();
    let complete = LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    };
    let prepared =
        crate::factorize::prepare_compact_factor_plan_generic_checked_for_test(&space, &complete)
            .unwrap()
            .expect("canonical checked plan");
    let final_call = complete.calls.get();
    assert!(final_call > 1);
    crate::factorize::finish_compact_factor_plan_generic_for_test(&space, prepared).unwrap();
    assert_eq!(complete.calls.get(), final_call);

    let failing = LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: final_call,
        calls: Cell::new(0),
    };
    crate::factorize::reset_generic_factor_plan_finish_calls();
    let error = match crate::factorize::prepare_compact_factor_plan_generic_checked_for_test(
        &space, &failing,
    ) {
        Err(error) => error,
        Ok(_) => panic!("late provider failure must abort checked preparation"),
    };
    assert!(matches!(
        error,
        crate::factorize::CheckedGenericFactorPlanError::Provider(LateGenericError(call))
            if call == final_call
    ));
    assert_eq!(crate::factorize::generic_factor_plan_finish_calls(), 0);
}

// Checked compact-QR provider sequence for `generic_factorization_input`: the
// compact plan issues no preflight query, the left (Q) space enumerates once
// (calls 1-3), then the right (R) space enumerates once (calls 4-6). Before
// the paired builder enumerated each side once, each side enumerated twice
// (left keys 1-3, left space 4-6, right keys 7-9, right space 10-12) and the
// first right-side call was the seventh.
const COMPACT_PAIR_LEFT_LAST_CALL: usize = 3;
const COMPACT_PAIR_RIGHT_FIRST_CALL: usize = 4;
const COMPACT_PAIR_RIGHT_LAST_CALL: usize = 6;

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_pair_builder_failure_preserves_provider_context() {
    // What: the last call of the left enumeration and the first call of the
    // right enumeration propagate their exact provider error without
    // publishing either factor or touching the input.
    let (source, data) = generic_factorization_input();
    for fail_at in [COMPACT_PAIR_LEFT_LAST_CALL, COMPACT_PAIR_RIGHT_FIRST_CALL] {
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at,
            calls: Cell::new(0),
        });
        let checked =
            BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
                .unwrap();
        let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
        let mut dense = CountingDense::default();
        let result = qr_compact_dyn_checked_generic(&mut dense, &input);
        assert!(matches!(
            result,
            Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call))) if call == fail_at
        ));
        assert_eq!(provider.calls.get(), fail_at);
        assert_eq!(dense.qr_calls, 2);
        assert_eq!(input.data(), data);
    }

    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let Qr { q, r } =
        qr_compact_dyn_checked_generic(&mut CountingDense::default(), &input).unwrap();
    assert_eq!(provider.calls.get(), COMPACT_PAIR_RIGHT_LAST_CALL);
    assert_eq!(checked_enumeration_calls(&q), COMPACT_PAIR_LEFT_LAST_CALL);
    assert_eq!(
        checked_enumeration_calls(&r),
        COMPACT_PAIR_RIGHT_LAST_CALL - COMPACT_PAIR_LEFT_LAST_CALL
    );
}

#[test]
fn hermitian_region_validation_rejects_short_storage_without_panicking() {
    // What: the cross-crate region validator reports malformed storage as a typed structural error.
    let tensor = one_sector_matrix(vec![1.0_f64, 0.0, 0.0, 2.0]);
    let regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();

    let error = validate_hermitian_regions(&tensor.data()[..3], &regions).unwrap_err();

    assert_eq!(
        error,
        OperationError::ElementCountMismatch {
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn compact_factor_plan_does_not_retain_provider() {
    // What: plan construction does not retain the input space's provider.
    let tensor = rectangular_svd_tensor(19, 11);
    let provider = Arc::new(Z2FusionRule);
    let weak = Arc::downgrade(&provider);
    let bound = bound_tensor(Arc::clone(&provider), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    drop(bound);
    drop(provider);

    assert!(weak.upgrade().is_none());
    drop(plan);
}

#[test]
fn compact_factor_plan_is_identical_across_calls_on_one_space() {
    // What: rebuilding the per-call plan on the same bound space yields the
    // same routes and the same region tables (shared `Arc`s), with the first
    // plan still alive.
    let charges =
        [U1Irrep::new(-1), U1Irrep::new(0), U1Irrep::new(1)].map(|charge| charge.sector_id());
    let tensor = tsvd_test_tensor(&U1FusionRule, &charges);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let first = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let second = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    let (first_source, first_u, first_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&first);
    let (second_source, second_u, second_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&second);
    assert!(first_source.len() >= 3);
    assert!(Arc::ptr_eq(&first_source, &second_source));
    assert!(Arc::ptr_eq(&first_u, &second_u));
    assert!(Arc::ptr_eq(&first_vh, &second_vh));
    assert_eq!(
        crate::factorize::compact_factor_plan_routes_for_test(&first),
        crate::factorize::compact_factor_plan_routes_for_test(&second)
    );
}

#[test]
fn compact_factor_routes_agree_between_sorted_and_unsorted_region_tables() {
    // What: canonical factor regions are strictly sorted by coupled sector and
    // route without a map; an expert (unsorted) region table routes through
    // the map path to the same regions in the same source order.
    let charges =
        [U1Irrep::new(-1), U1Irrep::new(0), U1Irrep::new(1)].map(|charge| charge.sector_id());
    let tensor = tsvd_test_tensor(&U1FusionRule, &charges);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let (source, u, vh) = crate::factorize::compact_factor_plan_regions_for_test(&plan);
    assert!(u
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));
    assert!(vh
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));

    let mut shuffled_u = u.to_vec();
    let mut shuffled_vh = vh.to_vec();
    shuffled_u.rotate_left(2);
    shuffled_vh.reverse();
    assert!(!shuffled_u
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));
    let sorted =
        crate::factorize::validate_compact_factor_routes_for_test(&source, &u, &vh).unwrap();
    let unsorted = crate::factorize::validate_compact_factor_routes_for_test(
        &source,
        &shuffled_u,
        &shuffled_vh,
    )
    .unwrap();
    assert_eq!(
        sorted,
        crate::factorize::compact_factor_plan_routes_for_test(&plan)
    );
    assert_eq!(sorted.len(), unsorted.len());
    for (sorted_route, unsorted_route) in sorted.iter().zip(&unsorted) {
        let (sorted_source, sorted_left, sorted_right) = sorted_route.factor_regions_for_test();
        let (unsorted_source, unsorted_left, unsorted_right) =
            unsorted_route.factor_regions_for_test();
        assert_eq!(sorted_source, unsorted_source);
        assert_eq!(
            sorted_left.map(|index| &u[index]),
            unsorted_left.map(|index| &shuffled_u[index])
        );
        assert_eq!(
            sorted_right.map(|index| &vh[index]),
            unsorted_right.map(|index| &shuffled_vh[index])
        );
    }
}

#[test]
fn compact_factor_plan_rejects_duplicate_missing_mismatched_and_extra_routes() {
    // What: every nonzero source sector has one shape-correct left/right route and no extras.
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(17, 13);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let (source, u, vh) = crate::factorize::compact_factor_plan_regions_for_test(&plan);

    let mut duplicate = u.to_vec();
    duplicate.push(u[0].clone());
    assert!(
        crate::factorize::validate_compact_factor_routes_for_test(&source, &duplicate, &vh,)
            .is_err()
    );
    assert!(crate::factorize::validate_compact_factor_routes_for_test(&source, &[], &vh,).is_err());
    assert!(crate::factorize::validate_compact_factor_routes_for_test(&source, &vh, &vh,).is_err());

    let multi = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let multi_bound = bound_tensor(Arc::new(rule), &multi);
    let multi_plan = crate::factorize::compact_factor_plan_for_test(multi_bound.space())
        .unwrap()
        .unwrap();
    let (multi_source, multi_u, multi_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&multi_plan);
    let mut reversed_u = multi_u.to_vec();
    let mut reversed_vh = multi_vh.to_vec();
    reversed_u.reverse();
    reversed_vh.reverse();
    crate::factorize::validate_compact_factor_routes_for_test(
        &multi_source,
        &reversed_u,
        &reversed_vh,
    )
    .unwrap();
    let mut extra = u.to_vec();
    extra.push(
        multi_u
            .iter()
            .find(|region| region.coupled() == SectorId::new(1))
            .unwrap()
            .clone(),
    );
    assert!(
        crate::factorize::validate_compact_factor_routes_for_test(&source, &extra, &vh,).is_err()
    );
}

#[test]
fn tsvd_fusion_reconstructs_su2_tensor() {
    run_tsvd_reconstruction_case(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
        true,
    );
}

#[test]
fn tsvd_fusion_reconstructs_u1_tensor() {
    run_tsvd_reconstruction_case(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
        false,
    );
}

#[test]
fn tsvd_fusion_reconstructs_fermion_parity_tensor() {
    // What: the canonical direct SVD preserves both fermion-parity sectors.
    run_tsvd_reconstruction_case(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
        true,
    );
}

#[test]
fn tsvd_fusion_reconstructs_product_rule_tensor() {
    // What: direct sector spans are keyed by the encoded product SectorId.
    let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let sectors = [
        rule.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        rule.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    run_tsvd_reconstruction_case(&rule, &sectors, true);
}

#[test]
fn typed_factor_axis_sum_overflow_is_exact_without_storage_materialization() {
    // What: an axis whose structural-zero degeneracies exceed usize reports
    // the exact checked error without allocating storage for those dimensions.
    let rule = U1FusionRule;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [
                (U1Irrep::new(1).sector_id(), usize::MAX),
                (U1Irrep::new(2).sector_id(), 1),
            ],
            false,
        )]),
        FusionProductSpace::new([]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        homspace,
        &rule,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap();

    let error = typed_from_dyn::<_, f64, 1, 0>(
        &rule,
        (
            tenet_tensors::DynamicFusionMapSpace::from_typed(&space),
            Vec::new(),
        ),
    )
    .unwrap_err();

    assert_eq!(error, OperationError::Core(CoreError::ElementCountOverflow));
}

fn u1_lowest_label_matrix(rows: usize, cols: usize) -> TensorMap<f64, 1, 1> {
    let rule = U1FusionRule;
    let minimum = U1Irrep::new(i32::MIN + 1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(minimum, rows)], false)]),
        FusionProductSpace::new([SectorLeg::new([(minimum, cols)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([rows], [cols]).unwrap(),
        homspace,
        &rule,
        [vec![rows, cols]],
    )
    .unwrap();
    TensorMap::from_vec_with_fusion_space(
        (0..rows * cols)
            .map(|index| ((index * 7 + 3) % 17) as f64 - 5.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn assert_matrix_product(
    expected: &[f64],
    rows: usize,
    inner: usize,
    cols: usize,
    left: &[f64],
    right: &[f64],
) {
    for col in 0..cols {
        for row in 0..rows {
            let actual = (0..inner)
                .map(|index| left[row + rows * index] * right[index + inner * col])
                .sum::<f64>();
            assert!(
                (actual - expected[row + rows * col]).abs() < 1e-10,
                "matrix product differs at ({row}, {col}): {actual} != {}",
                expected[row + rows * col]
            );
        }
    }
}

#[test]
fn compact_factors_do_not_relabel_the_lowest_u1_sector() {
    // What: compact SVD, QR, and LQ return correctly oriented factors at the lowest U(1) label.
    let rule = U1FusionRule;
    let tensor = u1_lowest_label_matrix(3, 2);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(svd.u.tensor().space().dims(), &[3, 2]);
    assert_eq!(svd.s.tensor().space().dims(), &[2, 2]);
    assert_eq!(svd.vh.tensor().space().dims(), &[2, 2]);
    let singular = &svd.singular_values[0].values;
    let mut scaled_vh = svd.vh.data().to_vec();
    for col in 0..2 {
        for row in 0..2 {
            scaled_vh[row + 2 * col] *= singular[row];
        }
    }
    assert_matrix_product(tensor.data(), 3, 2, 2, svd.u.data(), &scaled_vh);

    let Qr { q, r } = qr_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(q.tensor().space().dims(), &[3, 2]);
    assert_eq!(r.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, q.data(), r.data());

    let Lq { l, q } = lq_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(l.tensor().space().dims(), &[3, 2]);
    assert_eq!(q.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, l.data(), q.data());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn compact_factorizations_do_not_relabel_product_lowest_u1_sectors() {
    // What: product-sector factors inherit the no-relabel contract for compact SVD, QR, LQ, and full EIGH.
    let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let minimum = rule
        .try_encode_sector(SectorId::new(1), U1Irrep::new(i32::MIN + 1).sector_id())
        .unwrap();
    let matrix = |rows: usize, cols: usize, data: Vec<f64>| {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(minimum, rows)], false)]),
            FusionProductSpace::new([SectorLeg::new([(minimum, cols)], false)]),
        );
        let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([rows], [cols]).unwrap(),
            homspace,
            &rule,
            [vec![rows, cols]],
        )
        .unwrap();
        TensorMap::from_vec_with_fusion_space(data, space).unwrap()
    };
    let rectangular = matrix(3, 2, vec![-2.0, 5.0, 1.0, 4.0, -3.0, 2.0]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(svd.u.tensor().space().dims(), &[3, 2]);
    assert_eq!(svd.s.tensor().space().dims(), &[2, 2]);
    assert_eq!(svd.vh.tensor().space().dims(), &[2, 2]);

    let Qr { q, r } = qr_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(q.tensor().space().dims(), &[3, 2]);
    assert_eq!(r.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, q.data(), r.data());

    let Lq { l, q } = lq_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(l.tensor().space().dims(), &[3, 2]);
    assert_eq!(q.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, l.data(), q.data());

    let hermitian = matrix(2, 2, vec![4.0, 1.0, 1.0, 3.0]);
    let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &hermitian)).unwrap();
    assert_eq!(result.v.tensor().space().dims(), &[2, 2]);
    assert_eq!(result.d.tensor().space().dims(), &[2, 2]);
    let values = &result.eigenvalues[0].values;
    assert_eq!(values.len(), 2);
    for (col, &value) in values.iter().enumerate() {
        for row in 0..2 {
            let lhs = (0..2)
                .map(|index| hermitian.data()[row + 2 * index] * result.v.data()[index + 2 * col])
                .sum::<f64>();
            let rhs = result.v.data()[row + 2 * col] * value;
            assert!((lhs - rhs).abs() < 1e-10);
        }
    }
}

#[test]
fn spectral_outputs_retain_the_exact_input_provider_allocation() {
    // What: scalar promotion and spectral recomposition preserve provider authority by Arc identity.
    let rule = Z2FusionRule;
    let provider = Arc::new(rule);
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let hermitian_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&hermitian).unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();
    let hermitian_input =
        BoundDynamicTensorRef::try_new(&hermitian_space, hermitian.data()).unwrap();
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&general).unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();
    let general_input = BoundDynamicTensorRef::try_new(&general_space, general.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let eigh = eigh_full_dyn(&mut dense, &hermitian_input).unwrap();
    assert!(Arc::ptr_eq(&provider, eigh.v().space().provider_arc()));

    let eig = eig_full_dyn(&mut dense, &general_input).unwrap();
    assert!(Arc::ptr_eq(&provider, eig.v().space().provider_arc()));

    let mut context = default_context();
    let exponential = exp_dyn(&mut dense, &mut context, &hermitian_input).unwrap();
    assert!(Arc::ptr_eq(&provider, exponential.space().provider_arc()));
}

#[test]
fn leftorth_fusion_reconstructs_z2_and_su2_tensors() {
    for (rule_case, sectors) in [
        (0usize, vec![SectorId::new(0), SectorId::new(1)]),
        (
            1usize,
            vec![
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        ),
    ] {
        if rule_case == 0 {
            let rule = Z2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let Qr { q, r } = qr_compact(
                &mut dense_executor,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let Qr { q, r } = qr_compact(
                &mut dense_executor,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        }
    }
}

#[test]
fn rightorth_fusion_reconstructs_z2_and_su2_tensors() {
    {
        let rule = Z2FusionRule;
        let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
        let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
        let Lq { l, q } = lq_compact(
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule), &tensor),
        )
        .unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
    {
        let rule = SU2FusionRule;
        let tensor = tsvd_test_tensor(
            &rule,
            &[
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        );
        let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
        let Lq { l, q } = lq_compact(
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule), &tensor),
        )
        .unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
}

#[test]
fn tsvd_singular_tensor_composes_u_s_vt() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let s_tensor = svd.s.clone();

    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut u_s = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; svd.u.data().len()],
        svd.u.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut u_s,
            &svd.u,
            &s_tensor,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();

    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; tensor.data().len()],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut reconstructed,
            &u_s,
            &svd.vh,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn full_factorizations_agree_with_compact_on_matching_square_support() {
    // What: on square support the full factorizations need no completion and
    // agree with the compact ones (path agreement under the workspace rule,
    // 2 = the block dimension); each builds only its returned buffers.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        homspace,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(vec![-1.0, 3.0, 2.0, 4.0], space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let compact = svd_compact(&mut dense, &input.as_ref()).unwrap();
    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let full = svd_full(&mut dense, &input.as_ref()).unwrap();
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (1, 1),
        "full SVD must build exactly its returned U and Vh buffers"
    );
    numerics::assert_slices_close("U", full.u.data(), compact.u.data(), 2);
    numerics::assert_slices_close("S", full.s.data(), compact.s.data(), 2);
    numerics::assert_slices_close("Vh", full.vh.data(), compact.vh.data(), 2);

    let Qr {
        q: q_compact,
        r: r_compact,
    } = qr_compact(&mut dense, &input.as_ref()).unwrap();
    let Qr {
        q: q_full,
        r: r_full,
    } = qr_full(&mut dense, &input.as_ref()).unwrap();
    numerics::assert_slices_close("Q", q_full.data(), q_compact.data(), 2);
    numerics::assert_slices_close("R", r_full.data(), r_compact.data(), 2);

    let Lq {
        l: l_compact,
        q: q_compact,
    } = lq_compact(&mut dense, &input.as_ref()).unwrap();
    let Lq {
        l: l_full,
        q: q_full,
    } = lq_full(&mut dense, &input.as_ref()).unwrap();
    numerics::assert_slices_close("L", l_full.data(), l_compact.data(), 2);
    numerics::assert_slices_close("Q", q_full.data(), q_compact.data(), 2);
}

#[test]
fn full_factorizations_skip_dense_backend_for_disjoint_support() {
    let rule = U1FusionRule;
    let positive = U1Irrep::new(1).sector_id();
    let neutral = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(positive, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 3)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
        homspace,
        &rule,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(Vec::<f64>::new(), space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);

    svd_full(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
    qr_full(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
    lq_full(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
}

#[test]
fn spectrum_only_entry_points_return_descending_magnitudes() {
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    assert!(!svd.is_empty());
    for entry in &svd {
        for pair in entry.values.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12);
        }
    }
    let eigh = eigh_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap();
    assert!(!eigh.is_empty());
    for entry in &eigh {
        for pair in entry.values.windows(2) {
            assert!(pair[0].abs() >= pair[1].abs() - 1e-12);
        }
    }
    let eig = eig_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    assert!(!eig.is_empty());
    for entry in &eig {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-12);
        }
    }
}

fn assert_complex_spectra_close(
    lhs: &[SectorSpectrum<Complex64>],
    rhs: &[SectorSpectrum<Complex64>],
) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        assert_eq!(lhs.sector, rhs.sector);
        assert_eq!(lhs.values.len(), rhs.values.len());
        for (&lhs, &rhs) in lhs.values.iter().zip(&rhs.values) {
            assert!((lhs - rhs).norm() <= 1e-10, "{lhs} vs {rhs}");
        }
    }
}

fn assert_value_region_paths_match<R, D>(
    rule: Arc<R>,
    general: &TensorMap<D, 2, 2>,
    hermitian: &TensorMap<D, 2, 2>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let general_padded = padded_copy(rule.as_ref(), general);
    let hermitian_padded = padded_copy(rule.as_ref(), hermitian);
    let general_bound = bound_tensor(Arc::clone(&rule), general);
    let hermitian_bound = bound_tensor(Arc::clone(&rule), hermitian);
    let general_fallback = bound_tensor(Arc::clone(&rule), &general_padded);
    let hermitian_fallback = bound_tensor(rule, &hermitian_padded);
    assert!(general_bound
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    assert!(general_fallback
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    assert!(hermitian_fallback
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let general_before = general_bound.data().to_vec();
    let hermitian_before = hermitian_bound.data().to_vec();
    let general_fallback_before = general_fallback.data().to_vec();
    let hermitian_fallback_before = hermitian_fallback.data().to_vec();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_values_matricization_fallbacks();
    let direct_svd = svd_vals_dyn(&mut dense, &general_bound.as_ref().dynamic()).unwrap();
    let direct_eigh = eigh_vals_dyn(&mut dense, &hermitian_bound.as_ref().dynamic()).unwrap();
    let direct_eig = eig_vals_dyn(&mut dense, &general_bound.as_ref().dynamic()).unwrap();
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);

    crate::factorize::reset_values_matricization_fallbacks();
    let packed_svd = svd_vals_dyn(&mut dense, &general_fallback.as_ref().dynamic()).unwrap();
    let packed_eigh = eigh_vals_dyn(&mut dense, &hermitian_fallback.as_ref().dynamic()).unwrap();
    let packed_eig = eig_vals_dyn(&mut dense, &general_fallback.as_ref().dynamic()).unwrap();
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 3);

    assert_real_spectra_close(&direct_svd, &packed_svd);
    assert_real_spectra_close(&direct_eigh, &packed_eigh);
    assert_complex_spectra_close(&direct_eig, &packed_eig);
    assert!(general_bound.data() == general_before);
    assert!(hermitian_bound.data() == hermitian_before);
    assert!(general_fallback.data() == general_fallback_before);
    assert!(hermitian_fallback.data() == hermitian_fallback_before);
}

#[test]
fn value_region_paths_match_packed_oracles_across_supported_rules() {
    // What: canonical region borrowing and noncanonical packing return the same
    // ordered spectra for Abelian, non-Abelian, fermionic, and product rules.
    let u1 = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    assert_value_region_paths_match(
        Arc::new(U1FusionRule),
        &tsvd_test_tensor(&U1FusionRule, &u1),
        &hermitian_test_tensor(&U1FusionRule, &u1),
    );

    let su2 = [
        SU2Irrep::from_twice_spin(0).sector_id(),
        SU2Irrep::from_twice_spin(1).sector_id(),
    ];
    assert_value_region_paths_match(
        Arc::new(SU2FusionRule),
        &tsvd_test_tensor(&SU2FusionRule, &su2),
        &hermitian_test_tensor(&SU2FusionRule, &su2),
    );

    let fz2 = [SectorId::new(0), SectorId::new(1)];
    assert_value_region_paths_match(
        Arc::new(FermionParityFusionRule),
        &tsvd_test_tensor(&FermionParityFusionRule, &fz2),
        &hermitian_test_tensor(&FermionParityFusionRule, &fz2),
    );

    let product = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        product.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    assert_value_region_paths_match(
        Arc::new(product.clone()),
        &tsvd_test_tensor(&product, &product_sectors),
        &hermitian_test_tensor(&product, &product_sectors),
    );
}

#[test]
fn value_region_paths_match_packed_oracles_for_complex64() {
    // What: borrowed C64 spans preserve nonreal general matrices and conjugate
    // off-diagonal Hermitian matrices across all three values-only operations.
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let general = tsvd_test_tensor(&rule, &sectors);
    let hermitian = hermitian_test_tensor(&rule, &sectors);
    let general = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        general
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex64::new(value, (index % 7) as f64 * 0.125 - 0.25))
            .collect(),
        general.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let regions = hermitian
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let mut hermitian_data = hermitian
        .data()
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    for region in regions.iter().filter(|region| region.rows() >= 2) {
        let start = region.range().start;
        hermitian_data[start + 1].im = -0.75;
        hermitian_data[start + region.rows()].im = 0.75;
    }
    let hermitian = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        hermitian_data,
        hermitian.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();

    assert_value_region_paths_match(Arc::new(rule), &general, &hermitian);
}

#[test]
fn values_only_public_boundaries_distinguish_empty_sectors_from_a_scalar() {
    // What: zero degeneracies remove the sector entirely, while a rank-zero
    // scalar remains one vacuum-sector 1x1 matrix for every values operation.
    let empty = rectangular_svd_tensor(0, 0);
    assert_eq!(empty.structure().block_count(), 0);
    assert!(empty.data().is_empty());
    let mut reject = RejectExecutorCalls;
    crate::factorize::reset_values_matricization_fallbacks();
    assert!(svd_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert!(eigh_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert!(eig_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);

    let rule = Z2FusionRule;
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let shapes = vec![Vec::new(); homspace.fusion_tree_keys(&rule).len()];
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let scalar = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![-3.0], space).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    let eigh = eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    let eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    assert_eq!(svd[0].sector, rule.vacuum());
    assert_eq!(svd[0].values, vec![3.0]);
    assert_eq!(eigh[0].sector, rule.vacuum());
    assert_eq!(eigh[0].values, vec![-3.0]);
    assert_eq!(eig[0].sector, rule.vacuum());
    assert_eq!(eig[0].values, vec![Complex64::new(-3.0, 0.0)]);
}

#[test]
fn values_only_second_sector_failures_publish_no_partial_spectrum() {
    // What: after one successful sector, each dense values failure returns Err
    // without exposing the accumulated prefix or mutating borrowed input.
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let general = tsvd_test_tensor(&Z2FusionRule, &sectors);
    let hermitian = hermitian_test_tensor(&Z2FusionRule, &sectors);
    for operation in [
        ValuesOperation::Svd,
        ValuesOperation::Eigh,
        ValuesOperation::Eig,
    ] {
        let tensor = if operation == ValuesOperation::Eigh {
            &hermitian
        } else {
            &general
        };
        let input = bound_tensor(Arc::new(Z2FusionRule), tensor);
        let before = input.data().to_vec();
        let mut dense = FailSecondValues::new(operation);
        let result = match operation {
            ValuesOperation::Svd => svd_vals(&mut dense, &input.as_ref()).map(|_| ()),
            ValuesOperation::Eigh => eigh_vals(&mut dense, &input.as_ref()).map(|_| ()),
            ValuesOperation::Eig => eig_vals(&mut dense, &input.as_ref()).map(|_| ()),
        };

        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(dense.calls, 2);
        assert_eq!(input.data(), before);
    }
}

#[test]
fn values_only_stable_ties_match_provider_order_on_direct_and_padded_layouts() {
    // What: equal singular values and equal-magnitude eigenvalues retain the
    // dense provider's order on both the borrowed and packed sector paths.
    let rule = Z2FusionRule;
    let svd_input =
        one_sector_rectangular_matrix(vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let eigh_input =
        one_sector_rectangular_matrix(vec![1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 2.0], 3, 3);
    let eig_input =
        one_sector_rectangular_matrix(vec![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let svd_padded = padded_copy(&rule, &svd_input);
    let eigh_padded = padded_copy(&rule, &eigh_input);
    let eig_padded = padded_copy(&rule, &eig_input);
    let shape = [3, 3];
    let strides = [1, 3];
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let raw_svd = dense
        .svd_vals(DenseRead::F64(
            tenet_dense::DenseView::new(svd_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_f64_slice()
        .unwrap()
        .to_vec();
    let mut raw_eigh = dense
        .eigh_vals(DenseRead::F64(
            tenet_dense::DenseView::new(eigh_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_f64_slice()
        .unwrap()
        .to_vec();
    raw_eigh.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    let mut raw_eig = dense
        .eig_vals(DenseRead::F64(
            tenet_dense::DenseView::new(eig_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_c64_slice()
        .unwrap()
        .to_vec();
    raw_eig.sort_by(|a, b| b.norm().partial_cmp(&a.norm()).unwrap());

    let direct_svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &svd_input)).unwrap();
    let padded_svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &svd_padded)).unwrap();
    let direct_eigh =
        eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eigh_input)).unwrap();
    let padded_eigh =
        eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eigh_padded)).unwrap();
    let direct_eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eig_input)).unwrap();
    let padded_eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eig_padded)).unwrap();

    assert_eq!(direct_svd[0].values, raw_svd);
    assert_eq!(direct_eigh[0].values, raw_eigh);
    assert_eq!(direct_eig[0].values, raw_eig);
    assert_real_spectra_close(&direct_svd, &padded_svd);
    assert_real_spectra_close(&direct_eigh, &padded_eigh);
    assert_complex_spectra_close(&direct_eig, &padded_eig);
}

#[test]
fn values_only_entry_points_match_untruncated_decomposition_spectra() {
    // The `_vals` paths call LAPACK `job='N'` (no vectors) and must reproduce
    // the untruncated decomposition's spectrum. This is a numerical-agreement check,
    // not bit-for-bit: LAPACK backends may route the vectors-vs-no-vectors
    // cases through different routines (e.g. `gesdd` divide-and-conquer for the
    // full SVD vs `gesvd` QR for values-only), which differ in the last ULPs.
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let tol = 1e-10;
    let assert_real_close = |vals: &[SectorSpectrum], full: &[SectorSpectrum]| {
        assert_eq!(vals.len(), full.len());
        for (a, b) in vals.iter().zip(full) {
            assert_eq!(a.sector, b.sector);
            assert_eq!(a.values.len(), b.values.len());
            for (x, y) in a.values.iter().zip(&b.values) {
                assert!((x - y).abs() <= tol, "{x} vs {y}");
            }
        }
    };

    let svd_vals_spectra = svd_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    let svd_compact_spectra = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap()
    .singular_values;
    assert_real_close(&svd_vals_spectra, &svd_compact_spectra);

    let eigh_vals_spectra = eigh_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap();
    let eigh_full_spectra = eigh_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap()
    .eigenvalues;
    assert_real_close(&eigh_vals_spectra, &eigh_full_spectra);

    let eig_vals_spectra = eig_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    let eig_full_spectra = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap()
    .eigenvalues;
    assert_eq!(eig_vals_spectra.len(), eig_full_spectra.len());
    for (a, b) in eig_vals_spectra.iter().zip(&eig_full_spectra) {
        assert_eq!(a.sector, b.sector);
        assert_eq!(a.values.len(), b.values.len());
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!((x - y).norm() <= tol, "{x} vs {y}");
        }
    }
}

fn lowered_z2_binding<const NOUT: usize, const NIN: usize>(
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> BoundDynamicFusionMapSpace<Z2FusionRule> {
    let provider = Arc::new(Z2FusionRule);
    let raw = dyn_space_of(tensor).unwrap();
    let hom = raw.homspace().clone();
    // Why not caller-supplied per-tree shapes: the #586 sweep narrowed the
    // shape-admission bridge to tenet-tensors; the kept public installer
    // derives the identical blocks from the final homspace's leg
    // degeneracies for these dense-leg fixtures.
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(provider, hom)
        .unwrap()
}

#[test]
fn ordinary_factorizations_and_composition_inherit_lowered_layout_strategy() {
    // What: cold compact SVD, compact QR, full EIGH, adjoint, and factor
    // composition all retain the ordinary built-in layout-build strategy.
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = lowered_z2_binding(&tensor);
    let expert = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&tensor).unwrap(),
        Arc::new(Z2FusionRule),
    )
    .unwrap();
    let malformed = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(99), 1)], false)]),
        FusionProductSpace::new([]),
    );
    assert!(expert.prime_derived_homspace(&malformed).is_ok());
    let input = BoundDynamicTensorRef::try_new(&bound, tensor.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact_dyn(&mut dense, &input).unwrap();
    for factor in [svd.u(), svd.s(), svd.vh()] {
        assert!(factor.space().prime_derived_homspace(&malformed).is_err());
    }
    let Qr { q, r } = qr_compact_dyn(&mut dense, &input).unwrap();
    assert!(q.space().prime_derived_homspace(&malformed).is_err());
    assert!(r.space().prime_derived_homspace(&malformed).is_err());
    let eigh = eigh_full_dyn(&mut dense, &input).unwrap();
    assert!(eigh.v().space().prime_derived_homspace(&malformed).is_err());

    let adjoint = crate::factorize::adjoint_bound_factor(svd.u()).unwrap();
    assert!(adjoint.space().prime_derived_homspace(&malformed).is_err());
    let mut context = default_context();
    let composed = crate::compose::compose_bound_dyn(&mut context, svd.u(), svd.s()).unwrap();
    assert!(composed.space().prime_derived_homspace(&malformed).is_err());
}

#[test]
fn derived_matrix_functions_inherit_the_exact_provider_arc() {
    // What: every migrated owned result retains the input authority allocation.
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let provider = Arc::new(Z2FusionRule);
    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&tensor).unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&bound, tensor.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let LeftPolar {
        w: w_left,
        p: p_left,
    } = left_polar_dyn(&mut dense, &mut context, &input).unwrap();
    let RightPolar {
        p: p_right,
        wh: w_right,
    } = right_polar_dyn(&mut dense, &mut context, &input).unwrap();
    let inverse = inv_dyn(&mut dense, &mut context, &input).unwrap();
    let pseudo_inverse = pinv_dyn(&mut dense, &mut context, &input, 1.0e-13).unwrap();

    for factor in [
        &w_left,
        &p_left,
        &p_right,
        &w_right,
        &inverse,
        &pseudo_inverse,
    ] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
    }
}

#[test]
fn adjoint_composition_gives_the_identity_on_the_bond() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let Qr { q, .. } = qr_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let qh = tenet_tensors::adjoint(&rule, &q).unwrap();
    let mut context = default_context();
    let identity = crate::compose::compose(&mut context, &rule, &qh, &q).unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &identity));
}

#[test]
fn positive_diagonal_gauge_complex_phase_and_zero_diagonal() {
    use num_complex::Complex64;
    let c = Complex64::new;
    // q: 3 x 3, r: 3 x 3 upper triangular with complex diagonal phases and a
    // zero diagonal entry (row 1), column-major.
    let q: Vec<Complex64> = (0..9)
        .map(|i| c((i as f64 * 0.7 - 2.0).sin(), (i as f64 * 1.3 + 0.5).cos()))
        .collect();
    let r = vec![
        c(-3.0, 4.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(1.0, -2.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(0.5, 0.25),
        c(2.0, 1.0),
        c(0.0, -7.0),
    ];
    let product = |q: &[Complex64], r: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..3 {
                    out[row + 3 * col] += q[row + 3 * k] * r[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&q, &r);
    let mut q_gauged = q.clone();
    let mut r_gauged = r.clone();
    crate::factorize::positive_diagonal_gauge(&mut q_gauged, 3, &mut r_gauged, 3, 3);
    // Diagonal of R is real non-negative; the zero entry keeps phase 1.
    for j in 0..3 {
        let diagonal = r_gauged[j + 3 * j];
        assert!(
            diagonal.im.abs() < 1e-14,
            "R[{j},{j}] = {diagonal} not real"
        );
        assert!(diagonal.re >= 0.0, "R[{j},{j}] = {diagonal} negative");
    }
    let zero_diagonal_row = 1;
    let leading_dimension = 3;
    let zero_diagonal_index = zero_diagonal_row + leading_dimension * zero_diagonal_row;
    assert_eq!(r_gauged[zero_diagonal_index], c(0.0, 0.0));
    assert_eq!(q_gauged[3], q[3], "zero diagonal must not rescale Q column");
    // Q * R is unchanged.
    let after = product(&q_gauged, &r_gauged);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn eigenvector_gauge_matches_matrixalgebrakit_phase_rule() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut vectors = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];

    crate::factorize::eigenvector_gauge(&mut vectors, 3, 3, 2);

    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = vectors[row + 3 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
}

/// Returns a batch that violates the `factorize_batch` contract: one input's
/// entry missing, or one entry missing a factor.
struct MalformedBatch {
    inner: tenet_dense::DefaultDenseExecutor,
    drop_entry: bool,
}

impl DenseExecutor for MalformedBatch {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn factorize_batch(
        &mut self,
        op: tenet_dense::DenseFactorization,
        inputs: &[DenseRead<'_>],
    ) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
        let mut outputs = self.inner.factorize_batch(op, inputs)?;
        if self.drop_entry {
            outputs.pop();
        } else {
            outputs[0].pop();
        }
        Ok(outputs)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises factorizations")
    }
}

#[test]
fn compact_factorizations_reject_a_malformed_executor_batch() {
    // What: a batch with a missing input entry, or an entry missing a factor,
    // is a typed backend error, never a dropped sector or a panic. A missing
    // factor keeps the per-matrix path's "exactly (...)" error.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let input = bound.as_ref();
    for (drop_entry, qr_op, svd_op) in [
        (true, "factorize_batch", "factorize_batch"),
        (false, "qr_into", "svd_into"),
    ] {
        let mut dense = MalformedBatch {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            drop_entry,
        };
        let is_backend_error = |error: &OperationError, expected: &str| {
            matches!(
                error,
                OperationError::Dense(DenseError::Backend { op, .. }) if *op == expected
            )
        };
        let qr = qr_compact(&mut dense, &input).map(|_| ()).unwrap_err();
        assert!(is_backend_error(&qr, qr_op), "qr_compact: {qr:?}");
        let svd = svd_compact(&mut dense, &input).map(|_| ()).unwrap_err();
        assert!(is_backend_error(&svd, svd_op), "svd_compact: {svd:?}");
    }
}

/// Hermitian and general payloads of one coupled-sector tiling, written per
/// region so the Hermitian one is Hermitian in the tree basis.
fn region_payloads(
    regions: &[tenet_core::CoupledSectorRegion],
    len: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let entry = |salt: usize| {
        let value = |seed: usize| ((seed * 2_654_435_761) % 1009) as f64 / 1009.0 - 0.5;
        Complex64::new(value(salt), value(salt + 7919))
    };
    let mut general = vec![Complex64::zero(); len];
    let mut hermitian = general.clone();
    for region in regions {
        let (rows, start) = (region.rows(), region.range().start);
        for column in 0..region.cols() {
            for row in 0..rows {
                general[start + row + rows * column] = entry(start + row + rows * column);
            }
        }
        if rows == region.cols() {
            for column in 0..rows {
                for row in 0..rows {
                    hermitian[start + row + rows * column] = general[start + row + rows * column]
                        + general[start + column + rows * row].conj();
                }
            }
        }
    }
    (hermitian, general)
}

// #1494: the leg-degeneracy (facade) builder lists multi-leg Generic trees in
// its own order, which the former `FusionTreeKey`-Ord admission refused, so
// every one of these packed the whole payload before QR/SVD/LQ and the
// value-only spectra. A coupled-sector tiling is the same matricization the
// pack would build, so no pack may happen.
#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_facade_layouts_take_the_direct_region_path() {
    let x = SectorId::new(1);
    let v = SectorLeg::new([(SectorId::new(0), 1), (x, 2)], false);
    let w = SectorLeg::new([(x, 1)], false);
    let product =
        |legs: &[&SectorLeg]| FusionProductSpace::new(legs.iter().map(|&leg| leg.clone()));
    let cases = [
        (product(&[&v, &v]), product(&[&w])),
        (product(&[&w]), product(&[&v, &v])),
        (product(&[&v, &w]), product(&[&v, &w])),
        (product(&[&v, &v, &w]), product(&[&v])),
    ];
    let mut unordered_fixtures = 0;
    for (codomain, domain) in cases {
        let endomorphism = codomain == domain;
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        });
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            FusionTreeHomSpace::new(codomain, domain),
        )
        .unwrap();
        let regions = space
            .space()
            .structure()
            .coupled_sector_regions(space.space().nout())
            .unwrap()
            .expect("the facade builder lays out a coupled-sector tiling");
        let ord_ordered = regions.iter().all(|region| {
            [region.row_trees(), region.col_trees()]
                .iter()
                .all(|trees| trees.windows(2).all(|pair| pair[0].tree() < pair[1].tree()))
        });
        unordered_fixtures += usize::from(!ord_ordered);
        let (hermitian, general) = region_payloads(&regions, space.space().required_len().unwrap());
        let input = BoundDynamicTensorRef::try_new(&space, &general).unwrap();
        let mut dense = tenet_dense::DefaultDenseExecutor::new();

        crate::factorize::reset_generic_pair_publication_probe();
        crate::factorize::reset_compact_qr_copy_probe();
        let qr = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &qr.q, None, &qr.r);
        assert_eq!(
            crate::factorize::compact_qr_copy_probe().input_pack_bytes,
            0
        );

        crate::factorize::reset_compact_svd_copy_probe();
        let svd = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &svd.u, Some(&svd.s), &svd.vh);
        assert_eq!(
            crate::factorize::compact_svd_copy_probe().input_pack_bytes,
            0
        );

        crate::factorize::reset_compact_lq_copy_probe();
        let lq = lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &lq.l, None, &lq.q);
        assert_eq!(
            crate::factorize::compact_lq_copy_probe().input_pack_bytes,
            0
        );

        // The fresh bond-space builder enumerates the preserved side's trees
        // in the facade input's order, so the per-call route proof publishes
        // every factor in place: no pack in, no scatter out.
        let publication = crate::factorize::generic_pair_publication_probe();
        assert_eq!(
            (
                publication.canonical_publications,
                publication.fallback_publications
            ),
            (3, 0),
            "{publication:?}"
        );
        crate::factorize::reset_values_matricization_fallbacks();
        svd_vals_dyn_checked_generic(&mut dense, &input).unwrap();
        if endomorphism {
            let hermitian_input = BoundDynamicTensorRef::try_new(&space, &hermitian).unwrap();
            eigh_vals_dyn_checked_generic(&mut dense, &hermitian_input).unwrap();
            eig_vals_dyn_checked_generic(&mut dense, &input).unwrap();
        }
        assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);
    }
    assert!(
        unordered_fixtures > 0,
        "the fixtures must include the facade order that Ord admission packed"
    );
}

// Tree order is not admission, but it is still proven where it matters:
// sector `x` of this tiling lists its rows as (t0, t1) and its columns as
// (t1, t0). The direct QR/SVD/LQ outputs must still reconstruct the input
// (the fresh bond spaces are reached by tree identity, never by position),
// and eigenvalues must refuse the mis-stacked endomorphism instead of
// returning the spectrum of a column-permuted block (which the packed path
// silently did, since it stacks rows and columns by first appearance too).
#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_mis_stacked_tiling_scatters_factors_and_refuses_eigenvalues() {
    let (canonical, hermitian, general) = generic_values_endomorphism_input();
    let structure = canonical.space().structure();
    let matrix = structure
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap()
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap()
        .clone();
    let (t0, t1) = (matrix.row_trees()[0].tree(), matrix.row_trees()[1].tree());
    let key_of = |index: usize| {
        let block = structure.block(index).unwrap();
        (
            block.key().as_fusion_tree_pair().unwrap().clone(),
            block.shape().to_vec(),
        )
    };
    let find = |row: &FusionTreeKey, col: &FusionTreeKey| {
        (0..structure.block_count())
            .find(|&index| {
                let key = key_of(index).0;
                key.codomain_tree() == row && key.domain_tree() == col
            })
            .unwrap()
    };
    let scalar = (0..structure.block_count())
        .find(|&index| key_of(index).0.coupled() == SectorId::new(0))
        .unwrap();
    let order = [
        scalar,
        find(t0, t1),
        find(t0, t0),
        find(t1, t0),
        find(t1, t1),
    ];
    let blocks = order.iter().map(|&index| key_of(index)).collect();
    let mis_stacked =
        BlockStructure::coupled_sector_matrix_with_keys(&FactorGenericRule, 2, 4, blocks).unwrap();
    let region = mis_stacked
        .coupled_sector_regions(2)
        .unwrap()
        .expect("the reordered layout is still a coupled-sector tiling")
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap()
        .clone();
    assert_ne!(region.row_trees(), region.col_trees());

    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let relayout = |source: &[Complex64]| {
        let typed_space = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
            canonical.space().homspace().clone(),
            mis_stacked.clone(),
        )
        .unwrap()
        .try_bind_rule(&FactorGenericRule)
        .unwrap();
        let tensor = TensorMap::<Complex64, 2, 2>::from_block_fn_with_fusion_space(
            typed_space,
            Complex64::zero(),
            |key, indices| {
                let block = structure
                    .block(structure.find_block_index_by_key(key).unwrap())
                    .unwrap();
                source[block.offset()
                    + indices
                        .iter()
                        .zip(block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>()]
            },
        )
        .unwrap();
        (
            DynamicFusionMapSpace::from_typed(tensor.fusion_space().unwrap()),
            tensor.data().to_vec(),
        )
    };
    let (dynamic, general_data) = relayout(&general);
    let (_, hermitian_data) = relayout(&hermitian);
    let space = BoundDynamicFusionMapSpace::bind_generic(dynamic, Arc::clone(&provider)).unwrap();
    let input = BoundDynamicTensorRef::try_new(&space, &general_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_qr_copy_probe();
    crate::factorize::reset_generic_pair_publication_probe();
    let qr = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &qr.q, None, &qr.r);
    assert_eq!(
        crate::factorize::compact_qr_copy_probe().input_pack_bytes,
        0
    );
    assert_eq!(
        crate::factorize::generic_pair_publication_probe().canonical_publications,
        0
    );
    let svd = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &svd.u, Some(&svd.s), &svd.vh);
    let lq = lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &lq.l, None, &lq.q);

    let hermitian_input = BoundDynamicTensorRef::try_new(&space, &hermitian_data).unwrap();
    for (error, operation) in [
        (
            eigh_vals_dyn_checked_generic(&mut dense, &hermitian_input).unwrap_err(),
            "eigh_vals ",
        ),
        (
            eig_vals_dyn_checked_generic(&mut dense, &input).unwrap_err(),
            "eig_vals ",
        ),
    ] {
        assert!(
            matches!(
                error,
                CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope { message }
                ) if message.starts_with(operation)
            ),
            "{error:?}"
        );
    }
}

fn assert_multiplicity_free_eigen_ops_refuse_mis_stacking<R>(rule: R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = RuleIdentity>
        + Clone
        + fmt::Debug
        + 'static,
{
    let provider = Arc::new(rule.clone());
    let general = mis_stacked_endomorphism_copy(&rule, &tsvd_test_tensor(&rule, sectors));
    let hermitian = mis_stacked_endomorphism_copy(&rule, &hermitian_test_tensor(&rule, sectors));
    let general_bound = bound_tensor(Arc::clone(&provider), &general);
    let hermitian_bound = bound_tensor(Arc::clone(&provider), &hermitian);
    let (general, hermitian) = (general_bound.as_ref(), hermitian_bound.as_ref());
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    assert_stacking_refusal(eig_vals(&mut dense, &general), "eig_vals ");
    assert_stacking_refusal(eig_full(&mut dense, &general), "eig_full ");
    assert_stacking_refusal(eigh_vals(&mut dense, &hermitian), "eigh_vals ");
    assert_stacking_refusal(eigh_full(&mut dense, &hermitian), "eigh_full ");
    assert_stacking_refusal(exp(&mut dense, &mut context, &general), "exp ");
    assert_stacking_refusal(exp(&mut dense, &mut context, &hermitian), "exp ");
    assert_stacking_refusal(
        exp_pade13_direct_into_dyn(&mut dense, &general.dynamic()),
        "exp ",
    );

    // The lazy adjoint is a noncanonical layout, so these reach the packed
    // matricization branches; the adjoint of a mis-stacked block stays
    // mis-stacked.
    let general_adjoint = general_bound.space().adjoint_view().unwrap();
    let general_adjoint =
        BoundDynamicTensorRef::try_new(&general_adjoint, general_bound.data()).unwrap();
    let hermitian_adjoint = hermitian_bound.space().adjoint_view().unwrap();
    let hermitian_adjoint =
        BoundDynamicTensorRef::try_new(&hermitian_adjoint, hermitian_bound.data()).unwrap();
    assert_stacking_refusal(eig_vals_dyn(&mut dense, &general_adjoint), "eig_vals ");
    assert_stacking_refusal(eig_full_dyn(&mut dense, &general_adjoint), "eig_full ");
    assert_stacking_refusal(eigh_vals_dyn(&mut dense, &hermitian_adjoint), "eigh_vals ");
    assert_stacking_refusal(eigh_full_dyn(&mut dense, &hermitian_adjoint), "eigh_full ");
    assert_stacking_refusal(exp_dyn(&mut dense, &mut context, &general_adjoint), "exp ");
}

#[test]
fn multiplicity_free_eigen_ops_refuse_mis_stacked_z2_endomorphisms() {
    // What: every multiplicity-free spectral op refuses a tiling whose row and
    // column tree stackings differ, instead of returning the spectrum (or the
    // function) of a column-permuted block.
    assert_multiplicity_free_eigen_ops_refuse_mis_stacking(
        Z2FusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
}

#[test]
fn multiplicity_free_eigen_ops_refuse_mis_stacked_u1_endomorphisms() {
    assert_multiplicity_free_eigen_ops_refuse_mis_stacking(
        U1FusionRule,
        &[-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id()),
    );
}

#[test]
fn multiplicity_free_eigen_ops_refuse_mis_stacked_su2_endomorphisms() {
    assert_multiplicity_free_eigen_ops_refuse_mis_stacking(
        SU2FusionRule,
        &[0, 1].map(|twice| SU2Irrep::from_twice_spin(twice).sector_id()),
    );
}

#[test]
fn multiplicity_free_eigen_ops_accept_consistently_reordered_tree_stacking() {
    // What: the guard is stacking identity, not canonical order — reversing
    // rows and columns together is the same operator in a permuted basis and
    // keeps the facade spectra.
    let rule = U1FusionRule;
    let sectors = [-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id());
    let provider = Arc::new(rule);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let general = tsvd_test_tensor(&rule, &sectors);
    let hermitian = hermitian_test_tensor(&rule, &sectors);
    let reordered_general = reversed_coupled_tree_basis_copy(&rule, &general);
    let reordered_hermitian = reversed_coupled_tree_basis_copy(&rule, &hermitian);

    let eig = |dense: &mut tenet_dense::DefaultDenseExecutor, tensor| {
        eig_vals(dense, &bound_tensor_ref!(Arc::clone(&provider), tensor)).unwrap()
    };
    let eigh = |dense: &mut tenet_dense::DefaultDenseExecutor, tensor| {
        eigh_vals(dense, &bound_tensor_ref!(Arc::clone(&provider), tensor)).unwrap()
    };
    for (left, right) in eig(&mut dense, &general)
        .iter()
        .zip(&eig(&mut dense, &reordered_general))
    {
        assert_eq!(left.sector, right.sector);
        for (a, b) in left.values.iter().zip(&right.values) {
            assert!((a - b).norm() <= 1e-10 * a.norm().max(1.0), "{a} vs {b}");
        }
    }
    for (left, right) in eigh(&mut dense, &hermitian)
        .iter()
        .zip(&eigh(&mut dense, &reordered_hermitian))
    {
        assert_eq!(left.sector, right.sector);
        for (a, b) in left.values.iter().zip(&right.values) {
            assert!((a - b).abs() <= 1e-10 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
    eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::clone(&provider), &reordered_hermitian),
    )
    .unwrap();
    eig_full(
        &mut dense,
        &bound_tensor_ref!(Arc::clone(&provider), &reordered_general),
    )
    .unwrap();
}

/// Both factors come from the fresh builder of one homspace, so equal
/// layouts compare position by position.
fn assert_factor_matches_facade<R, D>(
    what: &str,
    expected: &BoundDynFactor<R, D>,
    actual: &BoundDynFactor<R, D>,
) where
    D: FactorScalar,
{
    let (left, right) = (
        expected.space().space().structure(),
        actual.space().space().structure(),
    );
    assert_eq!(left.block_count(), right.block_count(), "{what}");
    for index in 0..left.block_count() {
        let (a, b) = (left.block(index).unwrap(), right.block(index).unwrap());
        assert_eq!(a.key(), b.key(), "{what}");
        assert_eq!(a.offset(), b.offset(), "{what}");
        assert_eq!(a.shape(), b.shape(), "{what}");
    }
    let scale = expected
        .data()
        .iter()
        .fold(1.0f64, |max, &value| max.max(value.widen_complex().norm()));
    assert_eq!(expected.data().len(), actual.data().len(), "{what}");
    for (position, (&a, &b)) in expected.data().iter().zip(actual.data()).enumerate() {
        let (a, b) = (a.widen_complex(), b.widen_complex());
        assert!(
            (a - b).norm() <= 1e-10 * scale,
            "{what} at {position}: {a} vs {b}"
        );
    }
}

/// Dense eig leaves each eigenvector's scale free, so compare every bond
/// column of a sector up to one complex factor fixed at its largest entry.
fn assert_eig_columns_match_up_to_scale<R>(
    expected: &BoundDynFactor<R, Complex64>,
    actual: &BoundDynFactor<R, Complex64>,
) {
    let structure = expected.space().space().structure();
    let actual_structure = actual.space().space().structure();
    let mut columns = std::collections::BTreeMap::<(SectorId, usize), Vec<usize>>::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let other = actual_structure.block(index).unwrap();
        assert_eq!(block.key(), other.key());
        assert_eq!(block.offset(), other.offset());
        let BlockKey::FusionTree(key) = block.key() else {
            unreachable!("fusion-tree blocks")
        };
        let shape = block.shape();
        let extent = shape.iter().product::<usize>();
        for flat in 0..extent {
            let mut rest = flat;
            let mut position = block.offset();
            let mut bond = 0;
            for (axis, (&dim, &stride)) in shape.iter().zip(block.strides()).enumerate() {
                let index = rest % dim;
                rest /= dim;
                position += index * stride;
                if axis + 1 == shape.len() {
                    bond = index;
                }
            }
            columns
                .entry((key.coupled(), bond))
                .or_default()
                .push(position);
        }
    }
    for (column, positions) in columns {
        let pivot = *positions
            .iter()
            .max_by(|&&a, &&b| {
                expected.data()[a]
                    .norm()
                    .total_cmp(&expected.data()[b].norm())
            })
            .unwrap();
        let ratio = actual.data()[pivot] / expected.data()[pivot];
        for position in positions {
            let (a, b) = (expected.data()[position] * ratio, actual.data()[position]);
            assert!(
                (a - b).norm() <= 1e-10 * ratio.norm().max(1.0),
                "eig_full column {column:?} at {position}: {a} vs {b}"
            );
        }
    }
}

/// [`hermitian_test_tensor`]'s layout with unsymmetrized hashed entries, so
/// every coupled-sector matrix is full rank with a simple spectrum and its
/// gauge-fixed factors are unique.
fn generic_spectrum_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let template = hermitian_test_tensor(rule, sectors);
    let space = template.fusion_space().unwrap().as_ref().clone();
    TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        let mut hash = 0x9e37_79b9_7f4a_7c15u64;
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        for &sector in tree
            .codomain_tree()
            .uncoupled()
            .iter()
            .chain(tree.domain_tree().uncoupled())
        {
            hash = (hash ^ (sector.id() as u64 + 1)).wrapping_mul(0x100_0000_01b3);
        }
        for &index in indices {
            hash = (hash ^ (index as u64 + 7)).wrapping_mul(0x100_0000_01b3);
        }
        ((hash >> 40) % 1009) as f64 / 100.0 - 5.0
    })
    .unwrap()
}

/// What: a consistently reordered compact input (every coupled sector's rows
/// and columns stacked in reverse tree order) is the same tensor, so its
/// factors reconstruct it by tree identity and its SVD factors equal the
/// facade's gauge-fixed ones.
fn assert_reordered_compact_factor_publication<R, D>(provider: Arc<R>, facade: &TensorMap<D, 2, 2>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let reordered = reversed_coupled_tree_basis_copy(provider.as_ref(), facade);
    let facade = bound_tensor(Arc::clone(&provider), facade);
    let reordered = bound_tensor(provider, &reordered);
    assert!(reordered
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let (facade, reordered) = (facade.as_ref(), reordered.as_ref());
    let (facade, input) = (facade.dynamic(), reordered.dynamic());
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let expected = svd_compact_dyn(&mut dense, &facade).unwrap();
    let actual = svd_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, actual.u(), Some(actual.s()), actual.vh());
    assert_factor_matches_facade("svd_compact U", expected.u(), actual.u());
    assert_factor_matches_facade("svd_compact Vh", expected.vh(), actual.vh());

    // Why reconstruction only: QR of a column-permuted matrix is a different
    // factorization, so QR/LQ factors are not facade-comparable here.
    let actual = qr_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.q, None, &actual.r);

    let actual = lq_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.l, None, &actual.q);

    // Full factorizations publish through the one-sided tree-identity
    // scatter; their completion columns are not unique, so only
    // reconstruction is checked.
    let actual = svd_full_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, actual.u(), Some(actual.s()), actual.vh());
    let actual = qr_full_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.q, None, &actual.r);
    let actual = lq_full_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.l, None, &actual.q);
}

/// What: eigenvectors of a consistently reordered Hermitian or general
/// endomorphism map rows by tree identity: `A V = V Λ` against the facade
/// operator, `V D Vᴴ = A` over the reordered input, and the facade's
/// gauge-fixed eigenvectors, polar factors and eig vectors.
fn assert_reordered_eigen_factor_publication<R>(rule: R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = RuleIdentity>
        + Clone,
{
    let provider = Arc::new(rule.clone());
    let hermitian = hermitian_test_tensor(&rule, sectors);
    let general = generic_spectrum_test_tensor(&rule, sectors);
    let reordered_hermitian = reversed_coupled_tree_basis_copy(&rule, &hermitian);
    let reordered_general = reversed_coupled_tree_basis_copy(&rule, &general);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let eigh = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::clone(&provider), &reordered_hermitian),
    )
    .unwrap();
    assert_eigen_equation(&rule, &hermitian, &eigh.v, &eigh.d);

    let input = bound_tensor(Arc::clone(&provider), &reordered_hermitian);
    let input = input.as_ref();
    let input = input.dynamic();
    let eigh = eigh_full_dyn(&mut dense, &input).unwrap();
    let vh = crate::factorize::adjoint_bound_factor(eigh.v()).unwrap();
    let mut vd = eigh.v().clone();
    let vd_space = vd.space().space().clone();
    scale_axis_by_spectrum(&vd_space, vd.data_mut(), None, eigh.eigenvalues()).unwrap();
    assert_compact_factors_reconstruct_input(&input, &vd, None, &vh);

    let facade = bound_tensor(Arc::clone(&provider), &general);
    let facade = facade.as_ref();
    let facade = facade.dynamic();
    let reordered = bound_tensor(Arc::clone(&provider), &reordered_general);
    let reordered = reordered.as_ref();
    let reordered = reordered.dynamic();
    let expected = eig_full_dyn(&mut dense, &facade).unwrap();
    let actual = eig_full_dyn(&mut dense, &reordered).unwrap();
    assert_eig_columns_match_up_to_scale(expected.v(), actual.v());

    let mut context = default_context();
    let expected = left_polar_dyn(&mut dense, &mut context, &facade).unwrap();
    let actual = left_polar_dyn(&mut dense, &mut context, &reordered).unwrap();
    assert_factor_matches_facade("left_polar W", &expected.w, &actual.w);
    assert_factor_matches_facade("left_polar P", &expected.p, &actual.p);

    assert_reordered_compact_factor_publication(Arc::clone(&provider), &general);
    let complex = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        general
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex64::new(value, ((index * 7 + 3) % 13) as f64 * 0.2 - 1.0))
            .collect(),
        general.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    assert_reordered_compact_factor_publication(provider, &complex);
}

#[test]
fn z2_factor_publication_maps_consistently_reordered_trees_by_identity() {
    assert_reordered_eigen_factor_publication(Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
}

#[test]
fn u1_factor_publication_maps_consistently_reordered_trees_by_identity() {
    assert_reordered_eigen_factor_publication(
        U1FusionRule,
        &[-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id()),
    );
}

#[test]
fn su2_factor_publication_maps_consistently_reordered_trees_by_identity() {
    assert_reordered_eigen_factor_publication(
        SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
}
