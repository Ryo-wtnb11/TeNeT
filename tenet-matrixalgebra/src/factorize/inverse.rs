use super::*;

#[derive(Clone, Copy)]
pub(super) struct InverseSectorRoute {
    pub(super) source: usize,
    pub(super) output: usize,
}

#[derive(Clone, Copy)]
pub(super) struct SolveLeftSectorRoute {
    pub(super) divisor: usize,
    pub(super) rhs: usize,
    pub(super) output: usize,
}

pub(super) struct InverseMatrixRoute {
    pub(super) source: usize,
    pub(super) output: usize,
    pub(super) rows: Vec<InverseBasisExtent>,
    pub(super) cols: Vec<InverseBasisExtent>,
}

#[derive(Clone, Copy)]
pub(super) struct InverseBasisExtent {
    pub(super) source_offset: usize,
    pub(super) output_offset: usize,
    pub(super) extent: usize,
}

pub(crate) fn inverse_by_sector_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let authority = MfAuthority(input.space());
    let homspace = input.space().space().homspace();
    if !authority.isomorphic(input.space())? {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inv requires isomorphic codomain and domain",
        });
    }
    let output_space = authority.output_space(inverse_homspace(homspace))?;
    inverse_by_sector_dyn_into(dense, input, output_space)
}

/// Coefficient-free inverse execution into an already admitted swapped output.
///
/// The caller owns categorical admission; this body only routes the existing
/// coupled-sector layout and performs the dense solves.
pub(crate) fn inverse_by_sector_dyn_into<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let mut output_data = vec![D::zero(); output_space.space().required_len()?];

    let source_regions = checked_sector_regions(source_space.structure(), source_space.nout())?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "inverse derived output requires canonical coupled-sector storage",
    })?;
    match source_regions {
        Some(source) => {
            let routes = compile_inverse_region_routes(
                &source,
                &output_regions,
                input.data().len(),
                output_data.len(),
            )?;
            let max_order = routes
                .iter()
                .map(|route| source[route.source].rows())
                .max()
                .unwrap_or(0);
            let identity = identity_workspace::<D>(max_order)?;
            for route in routes {
                let source_region = &source[route.source];
                if source_region.rows() == 0 {
                    continue;
                }
                let source_matrix = &input.data()[source_region.range()];
                let output_matrix = &mut output_data[output_regions[route.output].range()];
                solve_inverse_sector(
                    dense,
                    source_matrix,
                    output_matrix,
                    source_region.rows(),
                    source_region.rows(),
                    &identity,
                    max_order,
                )?;
            }
        }
        None => {
            let source_matrices =
                sector_matricizations(source_space.structure(), input.data(), source_space.nout())?;
            let routes = compile_inverse_matrix_routes(
                &source_matrices,
                &output_regions,
                output_data.len(),
            )?;
            let max_order = routes
                .iter()
                .map(|route| source_matrices[route.source].rows)
                .max()
                .unwrap_or(0);
            let identity = identity_workspace::<D>(max_order)?;
            let mut solution = vec![D::zero(); identity.len()];
            for route in routes {
                let source = &source_matrices[route.source];
                if source.rows == 0 {
                    continue;
                }
                solve_inverse_sector(
                    dense,
                    &source.data,
                    &mut solution,
                    source.rows,
                    max_order,
                    &identity,
                    max_order,
                )?;
                let output = &output_regions[route.output];
                reorder_inverse_solution(
                    &solution,
                    max_order,
                    &mut output_data[output.range()],
                    output.rows(),
                    &route.rows,
                    &route.cols,
                );
            }
        }
    }

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nin(),
        source_space.nout(),
    )
}

/// `rcond * sigma_max` over every sector's singular values, rejecting a
/// non-finite one in the same pass.
///
/// Why an error rather than NaN propagation: `f64::max` discards NaN, and even
/// a NaN-propagating maximum only yields `sigma > NaN == false`, i.e. a silent
/// all-zero pseudo-inverse. The references are no better — Julia
/// `LinearAlgebra.pinv` (`tol2 = max(rtol*maximum(S), atol)`, `S .> tol2`)
/// zeroes every value under a NaN tolerance — so the only answer that does not
/// hide the NaN is a typed failure, matching `select_truncation`'s
/// `InvalidSpectrum`.
pub(crate) fn pinv_cutoff(
    singular_values: impl IntoIterator<Item = f64>,
    rcond: f64,
) -> Result<f64, OperationError> {
    let sigma_max = singular_values
        .into_iter()
        .try_fold(0.0_f64, |largest, sigma| {
            sigma.is_finite().then(|| largest.max(sigma))
        })
        .ok_or(OperationError::InvalidArgument {
            message: "pinv singular values must be finite",
        })?;
    Ok(rcond * sigma_max)
}

/// Coefficient-free pseudo-inverse into an already admitted swapped space.
///
/// All layout checks happen before staging or dense work.  The local staging
/// keeps an SVD or GEMM failure from publishing a partial output.
pub(crate) fn pinv_by_sector_dyn_into<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let Some(source_identity) = source_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(output_identity) = output_space.space().admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    if source_identity != output_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: source_identity.clone(),
                actual: output_identity.clone(),
            },
        ));
    }
    if !Arc::ptr_eq(input.space().provider_arc(), output_space.provider_arc()) {
        return Err(OperationError::StructureMismatch {
            tensor: "pinv output provider authority",
        });
    }
    let expected_homspace = FusionTreeHomSpace::new(
        source_space.homspace().domain().clone(),
        source_space.homspace().codomain().clone(),
    );
    let expected_nout = expected_homspace.codomain().len();
    let expected_nin = expected_homspace.domain().len();
    if output_space.space().nout() != expected_nout
        || output_space.space().nin() != expected_nin
        || output_space.space().rank() != expected_nout + expected_nin
        || output_space.space().homspace() != &expected_homspace
    {
        return Err(OperationError::StructureMismatch {
            tensor: "pinv output space",
        });
    }
    let source_regions = checked_sector_regions(source_space.structure(), source_space.nout())?
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "pinv requires coupled-sector input storage",
        })?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "pinv requires coupled-sector output storage",
    })?;
    let routes = compile_pinv_region_routes(
        &source_regions,
        &output_regions,
        input.data().len(),
        output_space.space().required_len()?,
    )?;

    struct Stage<D> {
        pub(super) route: InverseSectorRoute,
        pub(super) rows: usize,
        pub(super) cols: usize,
        pub(super) rank: usize,
        pub(super) u: Vec<D>,
        pub(super) singular_values: Vec<f64>,
        pub(super) vt: Vec<D>,
    }

    let mut output_data = vec![D::zero(); output_space.space().required_len()?];
    let mut staged = Vec::with_capacity(routes.len());
    let data = input.data();
    // One scope spans both passes: the staged SVDs and the per-sector
    // reconstruction GEMMs.
    in_linalg_scope(dense, |dense| {
        for route in routes {
            let source = &source_regions[route.source];
            let rows = source.rows();
            let cols = source.cols();
            let rank = rows.min(cols);
            let (u, singular_values, vt) = if rank == 0 {
                (Vec::new(), Vec::new(), Vec::new())
            } else {
                compact_svd_owned(dense, &data[source.range()], rows, cols)?
            };
            staged.push(Stage {
                route,
                rows,
                cols,
                rank,
                u,
                singular_values,
                vt,
            });
        }
        let cutoff = pinv_cutoff(
            staged
                .iter()
                .flat_map(|stage| stage.singular_values.iter().copied()),
            rcond,
        )?;
        for stage in staged {
            if stage.rank == 0 {
                continue;
            }
            let mut vt = stage.vt;
            for (row, &sigma) in stage.singular_values.iter().enumerate() {
                let reciprocal = D::from_real(if sigma > cutoff { 1.0 / sigma } else { 0.0 });
                for column in 0..stage.cols {
                    vt[row + stage.rank * column] = vt[row + stage.rank * column] * reciprocal;
                }
            }
            let output = &mut output_data[output_regions[stage.route.output].range()];
            let output_shape = [stage.cols, stage.rows];
            let output_strides = [1, stage.cols];
            let v_shape = [stage.cols, stage.rank];
            let v_strides = [stage.rank, 1];
            let uh_shape = [stage.rank, stage.rows];
            let uh_strides = [stage.rows, 1];
            let output_view = DenseViewMut::new(output, &output_shape, &output_strides, 0)
                .map_err(OperationError::Dense)?;
            let v_view =
                DenseView::new(&vt, &v_shape, &v_strides, 0).map_err(OperationError::Dense)?;
            let uh_view = DenseView::new(&stage.u, &uh_shape, &uh_strides, 0)
                .map_err(OperationError::Dense)?;
            dense
                .dot_general_into(
                    D::dense_write(output_view),
                    D::dense_read(v_view),
                    D::dense_read(uh_view),
                    &DenseDotConfig::matmul().with_conjugation(true, true),
                )
                .map_err(OperationError::Dense)?;
        }
        Ok(())
    })?;
    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nin(),
        source_space.nout(),
    )
}

pub(super) fn compile_pinv_region_routes(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    source_len: usize,
    output_len: usize,
) -> Result<Vec<InverseSectorRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_region) in source.iter().enumerate() {
        let output_index = output_by_sector
            .get(&source_region.coupled())
            .copied()
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "pinv output is missing a source coupled sector",
            })?;
        let output_region = &output[output_index];
        if output_region.rows() != source_region.cols()
            || output_region.cols() != source_region.rows()
            || source_region.col_trees() != output_region.row_trees()
            || source_region.row_trees() != output_region.col_trees()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "pinv output does not transpose the source coupled-sector layout",
            });
        }
        validate_region_range(source_region, source_len)?;
        validate_region_range(output_region, output_len)?;
        used[output_index] = true;
        routes.push(InverseSectorRoute {
            source: source_index,
            output: output_index,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "pinv output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

pub(crate) fn solve_left_by_sector_dyn<E, R, D>(
    dense: &mut E,
    divisor: &BoundDynamicTensorRef<'_, R, D>,
    rhs: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let divisor_space = divisor.space().space();
    let rhs_space = rhs.space().space();
    let expected = divisor.space().provider().rule_identity();
    let actual = rhs.space().provider().rule_identity();
    if expected != actual {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch { expected, actual },
        ));
    }
    if divisor_space.homspace().codomain() != rhs_space.homspace().codomain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve requires equal divisor and right-hand-side codomains",
        });
    }
    let authority = MfAuthority(divisor.space());
    if !authority.isomorphic(divisor.space())? {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve requires an isomorphic divisor codomain and domain",
        });
    }
    let output_space = authority.output_space(solve_homspace(
        divisor_space.homspace(),
        rhs_space.homspace(),
    ))?;
    solve_left_by_sector_dyn_into(dense, divisor, rhs, output_space)
}

pub(crate) fn solve_left_by_sector_dyn_into<E, R, D>(
    dense: &mut E,
    divisor: &BoundDynamicTensorRef<'_, R, D>,
    rhs: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let divisor_space = divisor.space().space();
    let rhs_space = rhs.space().space();
    let Some(divisor_identity) = divisor_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(rhs_identity) = rhs_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(output_identity) = output_space.space().admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    if divisor_identity != rhs_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: divisor_identity.clone(),
                actual: rhs_identity.clone(),
            },
        ));
    }
    if divisor_identity != output_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: divisor_identity.clone(),
                actual: output_identity.clone(),
            },
        ));
    }
    if !Arc::ptr_eq(divisor.space().provider_arc(), output_space.provider_arc()) {
        return Err(OperationError::StructureMismatch {
            tensor: "solve output provider authority",
        });
    }
    if divisor_space.homspace().codomain() != rhs_space.homspace().codomain() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve requires equal divisor and right-hand-side codomains",
        });
    }
    let expected_homspace = FusionTreeHomSpace::new(
        divisor_space.homspace().domain().clone(),
        rhs_space.homspace().domain().clone(),
    );
    let expected_nout = expected_homspace.codomain().len();
    let expected_nin = expected_homspace.domain().len();
    if output_space.space().nout() != expected_nout
        || output_space.space().nin() != expected_nin
        || output_space.space().rank() != expected_nout + expected_nin
        || output_space.space().homspace() != &expected_homspace
    {
        return Err(OperationError::StructureMismatch {
            tensor: "solve output space",
        });
    }

    let divisor_regions = checked_sector_regions(divisor_space.structure(), divisor_space.nout())?
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "solve requires canonical coupled-sector divisor storage",
        })?;
    for region in divisor_regions.iter() {
        validate_region_range(region, divisor.data().len())?;
        if region.rows() != region.cols() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "solve coupled-sector divisor matrices must be square",
            });
        }
    }
    let rhs_regions = checked_sector_regions(rhs_space.structure(), rhs_space.nout())?.ok_or(
        OperationError::UnsupportedTensorContractScope {
            message: "solve requires canonical coupled-sector right-hand-side storage",
        },
    )?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "solve derived output requires canonical coupled-sector storage",
    })?;
    let output_len = output_space.space().required_len()?;
    let routes = compile_solve_left_region_routes(
        &divisor_regions,
        &rhs_regions,
        &output_regions,
        divisor.data().len(),
        rhs.data().len(),
        output_len,
    )?;

    let mut output_data = vec![D::zero(); output_len];

    for route in routes {
        let divisor_region = &divisor_regions[route.divisor];
        if divisor_region.rows() == 0 || output_regions[route.output].cols() == 0 {
            continue;
        }
        solve_left_sector(
            dense,
            &divisor.data()[divisor_region.range()],
            &rhs.data()[rhs_regions[route.rhs].range()],
            &mut output_data[output_regions[route.output].range()],
            divisor_region.rows(),
            rhs_regions[route.rhs].cols(),
        )?;
    }

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        divisor_space.nin(),
        rhs_space.nin(),
    )
}

pub(super) fn compile_solve_left_region_routes(
    divisor: &[CoupledSectorRegion],
    rhs: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    divisor_len: usize,
    rhs_len: usize,
    output_len: usize,
) -> Result<Vec<SolveLeftSectorRoute>, OperationError> {
    let divisor_by_sector = sector_region_index_map(divisor)?;
    let rhs_by_sector = sector_region_index_map(rhs)?;
    let mut routes = Vec::with_capacity(output.len());
    for (output_index, output_region) in output.iter().enumerate() {
        let sector = output_region.coupled();
        let divisor_index = divisor_by_sector.get(&sector).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "solve divisor is missing an output coupled sector",
            },
        )?;
        let rhs_index = rhs_by_sector.get(&sector).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "solve right-hand side is missing an output coupled sector",
            },
        )?;
        let divisor_region = &divisor[divisor_index];
        let rhs_region = &rhs[rhs_index];
        if divisor_region.rows() != divisor_region.cols()
            || rhs_region.rows() != divisor_region.rows()
            || output_region.rows() != divisor_region.cols()
            || output_region.cols() != rhs_region.cols()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "solve coupled-sector matrix dimensions are incompatible",
            });
        }
        if divisor_region.row_trees() != rhs_region.row_trees()
            || divisor_region.col_trees() != output_region.row_trees()
            || rhs_region.col_trees() != output_region.col_trees()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "solve coupled-sector tree bases are incompatible",
            });
        }
        validate_region_range(divisor_region, divisor_len)?;
        validate_region_range(rhs_region, rhs_len)?;
        validate_region_range(output_region, output_len)?;
        routes.push(SolveLeftSectorRoute {
            divisor: divisor_index,
            rhs: rhs_index,
            output: output_index,
        });
    }
    if rhs_by_sector.len() != routes.len() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "solve right-hand side contains a sector absent from the output",
        });
    }
    Ok(routes)
}

pub(super) fn solve_left_sector<E, D>(
    dense: &mut E,
    divisor: &[D],
    rhs: &[D],
    output: &mut [D],
    order: usize,
    columns: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let divisor_shape = [order, order];
    let rhs_shape = [order, columns];
    let divisor_strides = [1, order];
    let rhs_strides = [1, order];
    let divisor = DenseView::new(divisor, &divisor_shape, &divisor_strides, 0)
        .map_err(OperationError::Dense)?;
    let rhs = DenseView::new(rhs, &rhs_shape, &rhs_strides, 0).map_err(OperationError::Dense)?;
    let output =
        DenseViewMut::new(output, &rhs_shape, &rhs_strides, 0).map_err(OperationError::Dense)?;
    dense
        .solve_into(
            D::dense_read(divisor),
            D::dense_read(rhs),
            D::dense_write(output),
        )
        .map_err(OperationError::Dense)
}

/// Walks the coupled-sector matricization of an endomorphism and replaces each
/// square block by `apply`'s image of it, writing into a freshly derived
/// canonical layout.
///
/// This is the block-data half of a matrix function that is *not* a spectral
/// function — the caller supplies only the dense `n x n -> n x n` kernel and
/// never sees the layout. `init` is handed the largest block order once, before
/// the loop, so a kernel can size its scratch to `O(max_c n_c²)` and allocate
/// nothing per sector.
///
/// That bound is the kernel's own. It is also the whole of the scratch on the
/// canonical direct-region layout, where the blocks are read in place; the
/// packed fallback below matricizes *every* sector up front and so costs
/// `O(Σ_c n_c²)` on top of it, whatever the kernel does.
///
/// `apply(state, source, n, out, out_leading)` reads the column-major `n x n`
/// block at `source` (leading dimension `n`) and writes its image into `out`
/// with leading dimension `out_leading`.
///
/// Publication is atomic: the result tensor is built only after every sector
/// has succeeded, so a failure in the last block leaves no half-written tensor
/// behind and never mutates the input.
///
/// Why the `inverse_*` route compilers are reused rather than copied: they map
/// source blocks onto the derived output layout under a codomain/domain swap,
/// and on an endomorphism that swap is the identity — the two tree lists are
/// the same list. Duplicating 80 lines to spell the identity differently would
/// only give the two copies a chance to drift.
///
/// # Panics
///
/// Debug-asserts `codomain == domain`. Refusing a non-endomorphism is the
/// caller's job, so that the message names the function the user called
/// (`exp`) rather than whichever helper noticed first.
pub(crate) fn map_square_sectors_dyn<R, D, S, I, F>(
    input: &BoundDynamicTensorRef<'_, R, D>,
    init: I,
    apply: F,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
    I: FnOnce(usize) -> Result<S, OperationError>,
    F: FnMut(&mut S, &[D], usize, &mut [D], usize) -> Result<(), OperationError>,
{
    let source_space = input.space().space();
    debug_assert_eq!(
        source_space.homspace().codomain(),
        source_space.homspace().domain(),
        "callers refuse a non-endomorphism in their own words first"
    );
    let output_space = input
        .space()
        .derive_from_final_homspace(source_space.homspace().clone())?;
    map_square_sectors_dyn_into(input, output_space, init, apply)
}

pub(crate) fn map_square_sectors_dyn_into<R, D, S, I, F>(
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    init: I,
    mut apply: F,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    D: FactorScalar,
    I: FnOnce(usize) -> Result<S, OperationError>,
    F: FnMut(&mut S, &[D], usize, &mut [D], usize) -> Result<(), OperationError>,
{
    let source_space = input.space().space();
    if source_space.homspace() != output_space.space().homspace() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "matrix function output must preserve the input homspace",
        });
    }

    let source_regions = checked_sector_regions(source_space.structure(), source_space.nout())?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "matrix function output requires canonical coupled-sector storage",
    })?;
    let output_len = output_space.space().required_len()?;
    let output_data = match source_regions {
        Some(source) => {
            validate_endomorphism_tree_stacking(source.as_ref(), EXP_STACKING)?;
            let routes = compile_inverse_region_routes(
                &source,
                &output_regions,
                input.data().len(),
                output_len,
            )?;
            let max_order = routes
                .iter()
                .map(|route| source[route.source].rows())
                .max()
                .unwrap_or(0);
            let mut output_data = vec![D::zero(); output_len];
            let mut state = init(max_order)?;
            for route in routes {
                let region = &source[route.source];
                let order = region.rows();
                if order == 0 {
                    continue;
                }
                let output = &output_regions[route.output];
                apply(
                    &mut state,
                    &input.data()[region.range()],
                    order,
                    &mut output_data[output.range()],
                    output.rows(),
                )?;
            }
            output_data
        }
        None => {
            let source_matrices =
                sector_matricizations(source_space.structure(), input.data(), source_space.nout())?;
            // Why not rely on the tree-identity routes as `inv` does: they are
            // right for `A^-1` of any stacking, but `f(P_R A P_C^T)` is not
            // `P_R f(A) P_C^T` unless the row and column stackings coincide.
            validate_endomorphism_tree_stacking(&source_matrices, EXP_STACKING)?;
            let routes =
                compile_inverse_matrix_routes(&source_matrices, &output_regions, output_len)?;
            let max_order = routes
                .iter()
                .map(|route| source_matrices[route.source].rows)
                .max()
                .unwrap_or(0);
            let mut output_data = vec![D::zero(); output_len];
            let mut state = init(max_order)?;
            let mut image = vec![
                D::zero();
                max_order
                    .checked_mul(max_order)
                    .ok_or(OperationError::ElementCountOverflow)?
            ];
            for route in routes {
                let source = &source_matrices[route.source];
                if source.rows == 0 {
                    continue;
                }
                apply(&mut state, &source.data, source.rows, &mut image, max_order)?;
                let output = &output_regions[route.output];
                reorder_inverse_solution(
                    &image,
                    max_order,
                    &mut output_data[output.range()],
                    output.rows(),
                    &route.rows,
                    &route.cols,
                );
            }
            output_data
        }
    };

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nout(),
        source_space.nin(),
    )
}

pub(super) fn compile_inverse_region_routes(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    source_len: usize,
    output_len: usize,
) -> Result<Vec<InverseSectorRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_region) in source.iter().enumerate() {
        let output_index = output_by_sector
            .get(&source_region.coupled())
            .copied()
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "inverse output is missing a source coupled sector",
            })?;
        validate_inverse_region(source_region, &output[output_index])?;
        validate_region_range(source_region, source_len)?;
        validate_region_range(&output[output_index], output_len)?;
        used[output_index] = true;
        routes.push(InverseSectorRoute {
            source: source_index,
            output: output_index,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

#[cfg(test)]
pub(crate) fn validate_inverse_region_routes_for_test(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
) -> Result<(), OperationError> {
    let source_len = source
        .iter()
        .map(|region| region.range().end)
        .max()
        .unwrap_or(0);
    let output_len = output
        .iter()
        .map(|region| region.range().end)
        .max()
        .unwrap_or(0);
    compile_inverse_region_routes(source, output, source_len, output_len).map(|_| ())
}

pub(super) fn validate_inverse_region(
    source: &CoupledSectorRegion,
    output: &CoupledSectorRegion,
) -> Result<(), OperationError> {
    if source.rows() != source.cols()
        || output.rows() != source.cols()
        || output.cols() != source.rows()
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse coupled-sector matrix is not square",
        });
    }
    if source.col_trees() != output.row_trees() || source.row_trees() != output.col_trees() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output tree basis does not transpose the source basis",
        });
    }
    Ok(())
}

pub(super) fn validate_region_range(
    region: &CoupledSectorRegion,
    data_len: usize,
) -> Result<(), OperationError> {
    let range = region.range();
    if range.end > data_len {
        return Err(OperationError::ElementCountMismatch {
            expected: range.end,
            actual: data_len,
        });
    }
    Ok(())
}

pub(super) fn compile_inverse_matrix_routes<D>(
    source: &[SectorMatricization<D>],
    output: &[CoupledSectorRegion],
    output_len: usize,
) -> Result<Vec<InverseMatrixRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_matrix) in source.iter().enumerate() {
        let output_index = output_by_sector.get(&source_matrix.sector).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "inverse output is missing a source coupled sector",
            },
        )?;
        let output_region = &output[output_index];
        if source_matrix.rows != source_matrix.cols
            || output_region.rows() != source_matrix.cols
            || output_region.cols() != source_matrix.rows
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "inverse coupled-sector matrix is not square",
            });
        }
        validate_region_range(output_region, output_len)?;
        let rows =
            compile_inverse_basis_extents(&source_matrix.col_trees, output_region.row_trees())?;
        let cols =
            compile_inverse_basis_extents(&source_matrix.row_trees, output_region.col_trees())?;
        used[output_index] = true;
        routes.push(InverseMatrixRoute {
            source: source_index,
            output: output_index,
            rows,
            cols,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

pub(super) fn compile_inverse_basis_extents(
    source: &[(FusionTreeKey, usize, Vec<usize>)],
    output: &[CoupledTreeExtent],
) -> Result<Vec<InverseBasisExtent>, OperationError> {
    let output_by_tree = output
        .iter()
        .enumerate()
        .map(|(index, extent)| (extent.tree(), index))
        .collect::<FxHashMap<_, _>>();
    if output_by_tree.len() != output.len() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a duplicate tree basis",
        });
    }
    let mut used = vec![false; output.len()];
    let mut extents = Vec::with_capacity(source.len());
    for (tree, source_offset, source_shape) in source {
        let output_index = output_by_tree.get(tree).copied().ok_or(
            OperationError::UnsupportedTensorContractScope {
                message: "inverse output is missing a source tree basis",
            },
        )?;
        let output_extent = &output[output_index];
        if source_shape != output_extent.shape() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "inverse output tree basis has an unexpected shape",
            });
        }
        let extent = output_extent
            .extent()
            .map_err(OperationError::from_core_preserving_context)?;
        used[output_index] = true;
        extents.push(InverseBasisExtent {
            source_offset: *source_offset,
            output_offset: output_extent.offset(),
            extent,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "inverse output contains a tree basis absent from the source",
        });
    }
    Ok(extents)
}

pub(super) fn identity_workspace<D: FactorScalar>(order: usize) -> Result<Vec<D>, OperationError> {
    let elements = order
        .checked_mul(order)
        .ok_or(OperationError::ElementCountOverflow)?;
    let mut identity = vec![D::zero(); elements];
    for index in 0..order {
        identity[index + order * index] = D::from_real(1.0);
    }
    Ok(identity)
}

pub(super) fn solve_inverse_sector<E, D>(
    dense: &mut E,
    source: &[D],
    output: &mut [D],
    order: usize,
    output_leading: usize,
    identity: &[D],
    identity_order: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [order, order];
    let matrix_strides = [1, order];
    let output_strides = [1, output_leading];
    let identity_strides = [1, identity_order];
    let source =
        DenseView::new(source, &shape, &matrix_strides, 0).map_err(OperationError::Dense)?;
    let identity =
        DenseView::new(identity, &shape, &identity_strides, 0).map_err(OperationError::Dense)?;
    let output =
        DenseViewMut::new(output, &shape, &output_strides, 0).map_err(OperationError::Dense)?;
    dense
        .solve_into(
            D::dense_read(source),
            D::dense_read(identity),
            D::dense_write(output),
        )
        .map_err(OperationError::Dense)
}

pub(super) fn reorder_inverse_solution<D: Copy>(
    source: &[D],
    source_rows: usize,
    output: &mut [D],
    output_rows: usize,
    row_extents: &[InverseBasisExtent],
    col_extents: &[InverseBasisExtent],
) {
    for rows in row_extents {
        for cols in col_extents {
            for col in 0..cols.extent {
                let source_start = rows.source_offset + source_rows * (cols.source_offset + col);
                let output_start = rows.output_offset + output_rows * (cols.output_offset + col);
                output[output_start..output_start + rows.extent]
                    .copy_from_slice(&source[source_start..source_start + rows.extent]);
            }
        }
    }
}
