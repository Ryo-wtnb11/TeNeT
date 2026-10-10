//! The compact-diagonal arms of contraction, composition, trace, transform and
//! twist (TensorKit's `DiagonalTensorMap` methods). Each answers `None` when
//! its operands or destination do not fit, and the dense route runs.
//!
//! Only the multiplicity-free real-coefficient hooks reach them until #1866
//! calls them from the shared bodies.

use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
    /// The compact arms of [`Self::compose`], or `None` when the operands or
    /// the destination cannot support one and the dense route must run.
    ///
    /// Each arm proves its destination rather than deriving it. Composition
    /// glues `self.codomain <- other.domain`, so:
    ///
    /// - `D * D` — both operands are bond spaces (`codomain == domain`), so
    ///   when the two spaces are equal the destination *is* that space, and
    ///   [`is_diagonal_bond_space`] certifies it can hold a compact result.
    /// - `t * D` — the destination is `t.codomain <- D.domain`; `D` is a bond
    ///   space, so `D.domain == D.codomain`, and requiring that to equal
    ///   `t.domain` makes the destination `t`'s own space. `t`'s payload is
    ///   then `t`'s data with each block's trailing axis scaled.
    /// - `D * t` — the mirror image, scaling `t`'s leading axis.
    ///
    /// Without those equalities the destination is a different space (a dual
    /// leg on the contracted side is the reachable case) and reusing an
    /// operand's would silently produce a tensor on the wrong space, so the
    /// arm declines and the expert layer decides — including by rejecting a
    /// composition that is not one at all.
    pub(super) fn compose_compact(&self, other: &Self) -> Result<Option<Self>, Error> {
        if !self.same_rule(other) {
            return Ok(None);
        }
        let (left, right) = (self.logical_space().space(), other.logical_space().space());
        match (self.spectrum(), other.spectrum()) {
            (Some(lhs), Some(rhs)) => {
                // Both clauses are unreachable today and stay for the reason
                // [`is_diagonal_bond_space`] gives. `left != right` is the
                // weaker one: two compact payloads on unequal bond spaces
                // necessarily carry spectra that differ in their sectors or
                // their lengths, so the elementwise product below would refuse
                // them anyway — just with `spectra_disagree`'s message instead
                // of the expert layer's. Removing it would change which error a
                // caller sees, not whether one is reported.
                if left != right || !is_diagonal_bond_space(left) {
                    return Ok(None);
                }
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let product = lhs
                    .iter()
                    .zip(rhs)
                    .map(|(left, right)| {
                        if left.sector != right.sector || left.values.len() != right.values.len() {
                            return Err(spectra_disagree());
                        }
                        Ok(tenet_matrixalgebra::SectorSpectrum {
                            sector: left.sector,
                            values: left
                                .values
                                .iter()
                                .zip(&right.values)
                                .map(|(&a, &b)| a * b)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Some(self.with_spectrum(product)))
            }
            // `t * D`: scale each block's trailing axis (TensorKit `rmul!`).
            (None, Some(spectrum)) => {
                if !is_diagonal_bond_space(right)
                    || left.homspace().domain().legs() != right.homspace().codomain().legs()
                {
                    return Ok(None);
                }
                self.scaled_axis(None, spectrum).map(Some)
            }
            // `D * t`: scale each block's leading axis (TensorKit `lmul!`).
            (Some(spectrum), None) => {
                if !is_diagonal_bond_space(left)
                    || right.homspace().codomain().legs() != left.homspace().domain().legs()
                {
                    return Ok(None);
                }
                other.scaled_axis(Some(0), spectrum).map(Some)
            }
            (None, None) => Ok(None),
        }
    }

    /// The compact arm of [`Self::contract`] (issue #584), or `None` when the
    /// operands, the axis pattern or the output order do not fit one and the
    /// dense route must run.
    ///
    /// A one-axis contraction against a compact operand is a bond scaling, so
    /// the spectrum multiplies the *other* operand's contracted leg — O(d·n)
    /// against the dense route's O(d²·n) GEMM on a materialized `Σ_c d_c²`
    /// buffer — and one [`Self::permute`] lays the result out. The permute is
    /// what carries every recoupling and bend, so this adds no mathematics of
    /// its own; it is a scale followed by one permutation.
    ///
    /// # Which patterns, and why only those
    ///
    /// The engine admits a contracted pair only when the two legs agree on
    /// their raw duality flag on the compose-shaped pairing (one operand's
    /// domain leg against the other's codomain leg), and a compact operand's
    /// leg *is* its bond on both sides. So each arm requires exactly that
    /// pairing and compares the two legs itself: raw equality is the engine's
    /// admissibility condition here, so a mismatch is a contraction the dense
    /// route must reject rather than one this arm may answer, and the arm
    /// declines so the expert layer reports it in its own words. `D · D` is
    /// handed to [`Self::compose_compact`], which is the same product and
    /// already proves its destination.
    ///
    /// # The twist, and why it is not folded
    ///
    /// [`Self::contract`] applies the fermionic supertrace twist to a **dual**
    /// contracted leg of `other`, where [`Self::compose`] does not. Here the case
    /// cannot arise, so the arm declines instead of carrying arithmetic no test
    /// could reach: a compact payload's bond leg is built non-dual
    /// (`seam::spectrum_bond`), the arms pair it with a *codomain*
    /// leg of `other` whose external duality is exactly its raw flag, and
    /// admissibility forces that flag to equal the bond's. The guard stays
    /// because the first constructor of a compact payload on a dual bond leg —
    /// or of an arm pairing a domain leg of `other` — should decline rather
    /// than silently return a wrong sign.
    pub(super) fn try_contract_diagonal(
        &self,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Result<Option<Self>, TypedFacadeError<R>>
    where
        // Why not mode-free yet: the `D · t` arm rebinds its result to the
        // left provider through the multiplicity-free layout (#1866).
        R: tenet_core::FusionRule,
    {
        if lhs_axes.len() != 1 || rhs_axes.len() != 1 || !self.same_rule(other) {
            return Ok(None);
        }
        let (lhs_axis, rhs_axis) = (lhs_axes[0], rhs_axes[0]);
        if lhs_axis >= self.rank() || rhs_axis >= other.rank() {
            return Ok(None);
        }
        // Why the provider rather than a stored flag: `braiding_style` is the
        // rule's own answer.
        let fermionic = <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider())
            == tenet_core::BraidingStyleKind::Fermionic;
        if fermionic
            && other
                .logical_space()
                .space()
                .homspace()
                .external_axis_is_dual(rhs_axis)
                != Some(false)
        {
            return Ok(None);
        }
        let (left, right) = (self.logical_space().space(), other.logical_space().space());
        let (left_home, right_home) = (left.homspace(), right.homspace());
        match (self.spectrum(), other.spectrum()) {
            // `D · D`: the same product as `D * D`, which already knows how to
            // stay compact and which destinations may hold the result.
            (Some(_), Some(_)) => {
                if lhs_axis != 1
                    || rhs_axis != 0
                    || codomain_rank != 1
                    || output_axes.iter().copied().ne(0..2)
                {
                    // Why not a reordered output: `pAB` can move the surviving
                    // bond across the codomain/domain split, and rebinding the
                    // product spectrum there is not equivalent to a permute
                    // (#453).
                    return Ok(None);
                }
                Ok(self.compose_compact(other)?)
            }
            // `t · D` (TensorKit `rmul!`): scale `t`'s contracted domain leg,
            // then move it to where the contraction's output order wants it.
            (None, Some(spectrum)) => {
                if rhs_axis != 0 || lhs_axis < self.codomain_rank() {
                    return Ok(None);
                }
                if left_home.domain().legs()[lhs_axis - self.codomain_rank()]
                    != right_home.codomain().legs()[0]
                {
                    return Ok(None);
                }
                let mut source: Vec<usize> = (0..self.rank()).filter(|&a| a != lhs_axis).collect();
                source.push(lhs_axis);
                self.scaled_axis(Some(lhs_axis), spectrum)?
                    .permuted_to_output(&source, output_axes, codomain_rank)
            }
            // `D · t` (TensorKit `lmul!`): the mirror image, scaling the
            // contracted codomain leg of `t` at whatever position it sits.
            (Some(spectrum), None) => {
                if lhs_axis != 1 || rhs_axis >= other.codomain_rank() {
                    return Ok(None);
                }
                if left_home.domain().legs()[0] != right_home.codomain().legs()[rhs_axis] {
                    return Ok(None);
                }
                let mut source = vec![rhs_axis];
                source.extend((0..other.rank()).filter(|&a| a != rhs_axis));
                let Some(completed) = other
                    .scaled_axis(Some(rhs_axis), spectrum)?
                    .permuted_to_output(&source, output_axes, codomain_rank)?
                else {
                    return Ok(None);
                };
                // Scaling and permutation deliberately run on `other`; only
                // after both succeed do we rebind their validated owned-dense
                // result to the public contract's exact left authority.
                let TensorMap { repr, .. } = completed;
                let TypedTensorRepr::Owned(body) = repr else {
                    return Ok(None);
                };
                if !matches!(body.data.as_ref(), TypedData::Dense(_)) {
                    return Ok(None);
                }
                let space = self
                    .logical_space()
                    .rebind_validated(&body.space.validated_layout())
                    .map_err(Error::from)?;
                Ok(Some(Self {
                    runtime: self.runtime.clone(),
                    repr: owned_repr(TypedTensorBody::with_shared_payload(
                        space,
                        Arc::clone(&body.data),
                    )),
                }))
            }
            (None, None) => Ok(None),
        }
    }

    /// This tensor's axes, listed in `source[output_axes[..]]` order and split
    /// at `codomain_rank`, or `None` when `output_axes` is not a permutation of
    /// `0..source.len()`.
    ///
    /// `source` is the contraction's default output order expressed as axes of
    /// the scaled operand. An `output_axes` that is not a
    /// permutation declines rather than errors: the dense route validates it
    /// and reports it, and one error message beats two.
    fn permuted_to_output(
        &self,
        source: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Result<Option<Self>, TypedFacadeError<R>> {
        let mut sorted = output_axes.to_vec();
        sorted.sort_unstable();
        if sorted.iter().copied().ne(0..source.len()) {
            return Ok(None);
        }
        let ordered: Vec<usize> = output_axes.iter().map(|&axis| source[axis]).collect();
        if codomain_rank == self.codomain_rank()
            && ordered[..codomain_rank]
                .iter()
                .copied()
                .eq(0..codomain_rank)
            && ordered[codomain_rank..]
                .iter()
                .copied()
                .eq(codomain_rank..source.len())
        {
            return Ok(Some(self.clone()));
        }
        self.tree_transform(TreeTransformOperation::permute(
            ordered[..codomain_rank].iter().copied(),
            ordered[codomain_rank..].iter().copied(),
        ))
        .map(Some)
    }

    /// This tensor with one bond axis of every block scaled by `spectrum`,
    /// on its own space. `axis = None` scales the trailing axis, `Some(0)` the
    /// leading one, exactly as the seam names them.
    fn scaled_axis(
        &self,
        axis: Option<usize>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> Result<Self, Error> {
        let _host_pool = self.runtime.enter_host_pool();
        let mut data = if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let (operand, source) = self.fusion_operand_and_data();
            tenet_tensors::oriented_fusion_add_owned(
                self.logical_space().space().structure(),
                operand,
                &source,
                operand,
                &source,
                D::from_real(1.0),
                D::from_real(0.0),
            )?
        } else {
            self.owned_body()
                .expect("owned scaled-axis input")
                .materialized_dense_data()
                .as_ref()
                .to_vec()
        };
        tenet_matrixalgebra::seam::scale_axis_by_spectrum_mapped(
            self.logical_space().space(),
            &mut data,
            axis,
            spectrum,
            |value| value,
        )?;
        Ok(self.with_data(data))
    }

    /// Whether two operands' providers are the same rule. The compact paths
    /// below skip the expert layer, which is where a mismatch would otherwise
    /// be caught, so they have to ask themselves.
    fn same_rule(&self, other: &Self) -> bool {
        TypedSectorAdmission::typed_rule_identity(self.provider())
            == TypedSectorAdmission::typed_rule_identity(other.provider())
    }
}

/// `twist` of a compact diagonal, or `None` for any other payload. A bond
/// space's two legs both carry the block's coupled sector, so the per-block
/// factor collapses to `θ(sector)^|legs|`; the space is unchanged, so the
/// payload stays compact.
pub(super) fn twist_spectrum<R, D>(
    tensor: &TensorMap<R, D>,
    legs: &[usize],
    inverse: bool,
    theta: impl Fn(SectorId) -> f64,
) -> Option<TensorMap<R, D>>
where
    D: TensorScalar,
{
    let spectrum = tensor.spectrum()?;
    let sector_factor = |sector: SectorId| -> f64 {
        let factor = legs.iter().map(|_| theta(sector)).product();
        twist_factor_with_inverse(factor, inverse)
    };
    if spectrum
        .iter()
        .all(|entry| sector_factor(entry.sector) == 1.0)
    {
        return Some(tensor.clone());
    }
    let scaled = spectrum
        .iter()
        .map(|entry| {
            let factor = D::from_real(sector_factor(entry.sector));
            tenet_matrixalgebra::SectorSpectrum {
                sector: entry.sector,
                values: entry.values.iter().map(|&value| value * factor).collect(),
            }
        })
        .collect();
    Some(tensor.with_spectrum_on(tensor.logical_space().clone(), scaled))
}

/// A rank-(1,1) leg swap of a compact diagonal kept compact, or an admitted
/// rank-(1,1) braid read straight from the spectrum into a dense result;
/// `None` for every other payload or operation.
pub(super) fn transform_rank_one_diagonal<R, D>(
    tensor: &TensorMap<R, D>,
    operation: &TreeTransformOperation,
) -> Result<Option<TensorMap<R, D>>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let Some(spectrum) = tensor.spectrum() else {
        return Ok(None);
    };
    let (codomain_rank, domain_rank) = (tensor.codomain_rank(), tensor.domain_rank());
    if crate::tensor_core::is_rank_one_diagonal_swap(codomain_rank, domain_rank, operation) {
        let destination = tensor
            .logical_space()
            .transformed_multiplicity_free(operation)?;
        let transformed = crate::tensor_core::transform_rank_one_diagonal_spectrum(
            tensor.logical_space().provider(),
            tensor.logical_space().space(),
            destination.space(),
            operation,
            spectrum,
        )?;
        return Ok(Some(tensor.with_spectrum_on(destination, transformed)));
    }
    if crate::tensor_core::is_rank_one_diagonal_braid(codomain_rank, domain_rank, operation) {
        if let Ok(destination) = tensor
            .logical_space()
            .transformed_multiplicity_free(operation)
        {
            let compiled = {
                let mut lease = tensor.runtime.lease_context()?;
                lease
                    .context()
                    .multiplicity_free_lane::<D>()?
                    .tree_context_mut()
                    .compile_tree_pair_structure(
                        tensor.logical_space().provider(),
                        operation,
                        destination.space().structure(),
                        tensor.logical_space().space().structure(),
                    )
                    .ok()
            };
            if let Some(compiled) = compiled {
                if let Some(data) = crate::tensor_core::try_braid_rank_one_diagonal_data(
                    tensor.logical_space().space(),
                    destination.space(),
                    &compiled,
                    spectrum,
                ) {
                    return Ok(Some(TensorMap {
                        runtime: tensor.runtime.clone(),
                        repr: owned_repr(TypedTensorBody::dense(destination, data)),
                    }));
                }
            }
        }
    }
    Ok(None)
}

/// The full trace of a rank-(1,1) compact diagonal over its only pair, or
/// `None` for any other payload or geometry (#604).
///
/// This is a reduction of the stored spectrum, so there is nothing to
/// materialize. It is the *categorical* trace, not `tr()`'s — the engine's
/// `trace_channel_factor` carries the quantum dimension of the traced
/// channel and, exactly where the traced leg is *not* dual, its fermionic
/// twist, which is what makes this the supertrace for a fermionic rule and
/// the coefficient `dim(c) · θ(c)` rather than `tr()`'s unconditional
/// `dim(c)`. The guard is this narrow because with one pair and rank two the
/// destination is the empty tree, so the traced channel is a single uncoupled
/// sector and the coefficient collapses to a per-sector scalar; any wider
/// geometry leaves an open destination tree whose recoupling is not a
/// per-sector scaling. Today the geometric conditions are implied by the
/// Group 4 contract (`TypedData::Diagonal` lives on bond spaces only), so
/// they are defensive, not a reachable branch, and the destination is staged
/// once. A lazy dense adjoint has no compact spectrum and therefore goes
/// through the parent-oriented trace seam; a compact adjoint remains an owned
/// compact tensor. The coefficient is pinned against the engine route by the
/// `compact_full_trace_*` oracle sweeps in `tests/typed_facade.rs`.
pub(super) fn full_trace_spectrum<R, D>(
    tensor: &TensorMap<R, D>,
    space: &BoundDynamicFusionMapSpace<R>,
    axes: tenet_tensors::TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Result<Option<TensorMap<R, D>>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let Some(spectrum) = tensor.spectrum() else {
        return Ok(None);
    };
    if tensor.rank() != 2 || tensor.codomain_rank() != 1 || axes.trace_lhs_axes().len() != 1 {
        return Ok(None);
    }
    let destination = tenet_tensors::tensortrace_stage_multiplicity_free(space, axes, dst_nout)?;
    let traced_leg_is_dual: bool =
        tensor.logical_space().space().homspace().codomain().legs()[0].is_dual();
    let provider: &R = tensor.logical_space().provider();
    // Accumulated in `Complex64` and narrowed once through the #568
    // `UserScalar` surface, with the same per-sector reduction order as
    // compact `tr`. The typed spectrum already stores `SectorSpectrum<D>`,
    // and the coefficient is the provider's real scalar, so the result is a
    // plain `D`.
    let mut total: num_complex::Complex64 = num_complex::Complex64::new(0.0, 0.0);
    for entry in spectrum {
        let dim = multiplicity_free_dim(provider, entry.sector)?;
        let coefficient: f64 = if traced_leg_is_dual {
            dim
        } else {
            dim * provider.twist_scalar(entry.sector)
        };
        let mut partial = D::Wide::from_real(0.0);
        for &value in &entry.values {
            partial = partial + value.widen();
        }
        total += partial.widen_complex() * coefficient;
    }
    // A fully traced rank-(1,1) destination is one scalar.
    if destination.space().required_len()? != 1 {
        return Err(internal_layout_error(
            "a fully traced rank-one destination is not a single scalar",
        ));
    }
    Ok(Some(
        tensor.published(destination, vec![D::from_complex64(total)]),
    ))
}
