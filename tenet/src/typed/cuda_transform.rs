use super::*;

#[cfg(feature = "cuda")]
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    /// Device lowering shared by every structural transform, mirroring Host
    /// `tree_transform_multiplicity_free_real` branch for branch.
    ///
    /// Everything categorical is the Host's, unchanged: the same
    /// [`TreeTransformOperation`], the same rule validation, the same
    /// categorical group plan, the same `RuleIdentity`-keyed transform cache
    /// and the same completed `TreeTransformStructure`. Only the executor
    /// below that structure differs, so a device result is the Host result up
    /// to dtype tolerance.
    ///
    /// Lock order — and the reason the two phases are written apart: planning
    /// runs under the pooled Host context lease, which is dropped before the
    /// device lease is taken, and nothing reached under the device lease
    /// leases again (`std::sync::Mutex` is not re-entrant). Exactly one device
    /// lease per replay.
    fn tree_transform_cuda(
        &self,
        operation_name: &'static str,
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        // Host order: a lazy adjoint lowers the operation onto its parent and
        // re-wraps the owned result, rather than materializing the adjoint.
        // `.adjoint()` is itself lease-free, so this recursion never nests a
        // device lease.
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent_space = view.parent.space.space();
            let lowered = lower_adjoint_tree_transform_operation(
                parent_space.nout(),
                parent_space.nin(),
                &operation,
            )?;
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .tree_transform_cuda(operation_name, lowered)?
                .adjoint();
        }
        // Host's compact-diagonal arm is unreachable here — `to_cuda`
        // densifies a diagonal — but a device tensor that somehow carried one
        // is an explicit capability error, never a silent dense fallback.
        let src = self.direct_cuda_storage(operation_name)?;
        let body = self
            .owned_body()
            .ok_or_else(|| internal_layout_error("owned device tree transform input"))?;
        let dst_space = body.space.transformed_multiplicity_free(&operation)?;
        let required_len = dst_space.space().required_len()?;

        let structure = {
            let mut lease = self.runtime.lease_context()?;
            lease
                .context()
                .multiplicity_free_lane::<D>()?
                .tree_context_mut()
                .compile_tree_pair_structure(
                    body.space.provider(),
                    &operation,
                    dst_space.space().structure(),
                    body.space.space().structure(),
                )?
        };

        let mut lease = self.runtime.lease_cuda()?;
        let (cuda, executor) = lease.split();
        if src.placement() != Placement::Cuda(cuda.device()) {
            return Err(Error::PlacementMismatch);
        }
        // ponytail: #740 — the device seam still initializes an output by
        // uploading zeros; replace only with a measured native allocation.
        //
        // Why not report an inexpressible destination layout before spending
        // this upload: a returning transform cannot produce one. `dst_space`
        // comes from `transformed_multiplicity_free`, i.e. the canonical
        // final-homspace layout, never from the source's strides, and block
        // strides are `usize`, so both arms of the executor's layout rejection
        // are unreachable here; the buffer is this method's own and is dropped
        // on any error, so no caller-visible state is touched either way.
        // G2b-3 has no such argument — its destination is the caller's — so it
        // must validate before it writes, and must redo this reasoning if it
        // admits a non-canonical layout through `admit_exact_tree_pair_layout`.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        executor.replay(
            cuda,
            &structure,
            dst_space.space().structure(),
            body.space.space().structure(),
            &mut dst,
            src,
            D::from_real(1.0),
            CudaTreeTransformDestination::Overwrite,
        )?;
        drop(lease);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(dst_space, dst)),
        })
    }

    /// Shared body of the three planar device operations, mirroring Host
    /// `planar`: derive the planar axis order, let the expert layer validate
    /// it, and run it as a transpose — never as a permute, because domain
    /// trees run opposite to the planar boundary.
    fn planar_cuda(
        &self,
        operation_name: &'static str,
        kind: PlanarRequestKind<'_>,
    ) -> Result<Self, Error> {
        let operation = with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            kind,
            |codomain_axes, domain_axes| {
                Ok(TreeTransformOperation::transpose(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
        )?;
        self.tree_transform_cuda(operation_name, operation)
    }

    /// TensorKit `permute` on a device tensor: the Host
    /// [`TensorMap::permute`] semantics, executed on the device.
    ///
    /// # Cost
    ///
    /// A warm call uploads no structure data and downloads nothing. Its only
    /// transfer is the output initialisation (#740): exactly one H2D of
    /// `required_len * size_of::<D>()` bytes, from one host `Vec` of that
    /// size, and its only device allocation is the returned output. The first
    /// call for a given structure additionally uploads that structure's
    /// coefficient payload once and may grow the pack/scatter workspace once.
    ///
    /// On device the replay runs in overwrite mode, so it also zero-fills
    /// every *inactive* destination layout on the device — redundant work over
    /// an output that was just uploaded as zeros, costing one submission and
    /// `Σ(inactive layout elements)` device writes per call. It is kept
    /// because it is what makes the executor's destination mode independent of
    /// what the destination held, which `*_into` with `beta = 0` relies on; removing
    /// it here would need a second replay mode for no transfer saving.
    ///
    /// That warm contract holds only while this Runtime's Host transform store
    /// admits the structure: with a tree-transform cache byte budget of zero,
    /// or for a structure whose entry exceeds that budget, the
    /// store hands back a fresh allocation per call and the device therefore
    /// re-uploads its coefficients per call.
    ///
    /// # Numerics
    ///
    /// Inherited from the device executor: the block set written is the Host's
    /// exactly; coefficient-`1` `f64` moves are bitwise, and everything else
    /// agrees to dtype tolerance, with one rounding position moved for Single
    /// blocks (`alpha * (c * x)` where the Host folds the scales and rounds as
    /// `(alpha * c) * x`) and with the recoupling GEMM's summation order that
    /// of cuTENSOR rather than the Host kernel.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOnDevice`] for diagonal storage;
    /// [`Error::PlacementMismatch`] for a payload on another device;
    /// [`Error::Operation`] with
    /// [`crate::typed::OperationError::InvalidPermutation`] for malformed axis
    /// lists, as on the Host — all before any device write.
    ///
    /// Checked-Generic, Generic and complex-coefficient providers are excluded
    /// by this impl's bound, so they are a compile-time boundary:
    ///
    /// ```compile_fail
    /// use tenet::sector::{CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, TypedSectorAdmission};
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn no_checked_generic_cuda_permute<R>(tensor: &TensorMap<R, f64, CudaStorage>)
    /// where
    ///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
    ///         + CheckedGenericFusion
    ///         + CheckedGenericRigidSymbols<Scalar = f64>,
    /// {
    ///     let _ = tensor.permute(&[1], &[0]);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use tenet::sector::FibonacciFusionRule;
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn no_generic_cuda_permute(tensor: &TensorMap<FibonacciFusionRule, f64, CudaStorage>) {
    ///     let _ = tensor.permute(&[1], &[0]);
    /// }
    /// ```
    ///
    /// # Source compatibility
    ///
    /// Enabling the `cuda` feature adds these four names to a second `impl`,
    /// so the *path* forms `TensorMap::permute`, `TensorMap::braid`,
    /// `TensorMap::transpose` and
    /// `TensorMap::repartition` become ambiguous (`E0034`) where they were not
    /// before — the same non-additivity the device `adjoint`, `norm` and
    /// `scale` already have. Method-call syntax (`tensor.permute(..)`), a
    /// closure (`|t| t.permute(..)`) or a fully qualified
    /// `TensorMap::<R, D>::permute` all keep working.
    ///
    /// The multiplicity-free twin compiles, for either device payload:
    ///
    /// ```
    /// use tenet::sector::U1FusionRule;
    /// use tenet::typed::{CudaPayload, CudaStorage, TensorMap};
    ///
    /// fn device_permute<D: CudaPayload>(tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>) {
    ///     let _ = tensor.permute(&[1], &[0]);
    ///     let _ = tensor.braid(&[1], &[0], &[0, 1]);
    ///     let _ = tensor.transpose(&[1], &[0]);
    ///     let _ = tensor.repartition(0);
    /// }
    /// ```
    pub fn permute(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> Result<Self, Error> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        self.tree_transform_cuda(
            "permute",
            TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
        )
    }

    /// The device form of the Host leg-role composition: `op` runs on
    /// `permute(self, rows, cols)`, and the current split borrows `self`
    /// without a transform or a clone.
    pub(super) fn with_cuda_leg_roles<T>(
        &self,
        rows: &[usize],
        cols: &[usize],
        op: impl FnOnce(&Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if self.axes_are_identity(rows, cols) {
            return op(self);
        }
        op(&self.permute(rows, cols)?)
    }

    /// TensorKit `braid` on a device tensor: the Host [`TensorMap::braid`]
    /// semantics, executed on the device.
    ///
    /// Cost, numerics and provider boundary are [`Self::permute`]'s.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `levels` does not list one level per
    /// source axis — checked **before** the identity short circuit, as on
    /// Host, so a mis-lengthed `levels` is reported even for an identity
    /// permutation. Otherwise as [`Self::permute`].
    pub fn braid(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, Error> {
        let rank = self.rank();
        if levels.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "braid levels must list one level per source axis \
                 (expected {rank}, got {})",
                levels.len()
            )));
        }
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        let nout = self.codomain_rank();
        self.tree_transform_cuda(
            "braid",
            TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            ),
        )
    }

    /// TensorKit `repartition(t, N1, N2)` on a device tensor: the Host
    /// [`TensorMap::repartition`] semantics, executed on the device.
    ///
    /// Requesting the split the tensor already has returns a clone and does no
    /// device work at all.
    ///
    /// Cost, numerics and provider boundary are [`Self::permute`]'s.
    pub fn repartition(&self, num_codomain: usize) -> Result<Self, Error> {
        if num_codomain == self.codomain_rank() {
            return Ok(self.clone());
        }
        self.planar_cuda(
            "repartition",
            PlanarRequestKind::Repartition { num_codomain },
        )
    }

    /// TensorKit `transpose(t, (p₁, p₂))` on a device tensor: the Host
    /// [`TensorMap::transpose`] semantics, executed on the device.
    ///
    /// Cost, numerics and provider boundary are [`Self::permute`]'s.
    pub fn transpose(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> Result<Self, Error> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        self.planar_cuda(
            "transpose",
            PlanarRequestKind::Explicit {
                codomain_axes,
                domain_axes,
            },
        )
    }

    /// `destination = alpha * self.permute(codomain_axes, domain_axes) +
    /// beta * destination` on the device, without replacing its provider,
    /// space, body, or device allocation: the Host [`TensorMap::permute_into`].
    ///
    /// The admission sequence is the Host one plus the device's own placement
    /// check: same Runtime, same rule identity, an owned dense device source
    /// (a lazy adjoint is *rejected*, not lowered onto its parent, exactly as
    /// on Host — the destination is the caller's, so there is no re-wrapping
    /// to do), an owned dense device destination that does not alias the
    /// source payload, the admitted tree-pair operation or else the
    /// transform's own fusion space and block layout, the exact destination
    /// length, and unique destination ownership ([`Error::DestinationShared`]).
    /// Every one of them, and the structure compilation, happens before the
    /// device lease is taken; the placement check is the first thing under
    /// it. After a successful replay the exact source/destination layout pair
    /// is admitted on the Runtime, so a second call with the same pair skips
    /// the layout derivation — the same store, and the same admitted path, a
    /// Host call would take.
    ///
    /// There is no identity short circuit, because Host has none here: an
    /// identity axis list still writes `alpha * self + beta * destination`.
    ///
    /// `beta` rides the `dot_general` epilogue of the move or scatter that
    /// writes each destination layout, once per element; the layouts no move
    /// writes become `beta * destination` through a zero-source move (zeros
    /// for `beta = 0`, which never reads the destination; nothing for
    /// `beta = 1`).
    ///
    /// # Cost
    ///
    /// A warm call transfers nothing in either direction and allocates no
    /// device buffer: 0 H2D, 0 D2H, 0 device allocations. Unlike the returning
    /// [`Self::permute`] it pays no output initialisation. The first call for
    /// a given structure uploads that structure's coefficient payload once,
    /// may grow the pack/scatter workspace once, and reserves the context zero
    /// template; that warm contract holds only while this Runtime's Host
    /// transform store admits the structure, as for [`Self::permute`].
    /// `alpha == 0` writes `beta * destination` over every written layout
    /// through the same context zero template, so the first zero-scale call on
    /// a context whose template is still short pays one upload — once per
    /// context, not per call.
    ///
    /// # Numerics
    ///
    /// [`Self::permute`]'s, with the caller scale applied as the executor's
    /// rule: a Single block rounds as `alpha * (c * x)` where the Host folds
    /// the scales and rounds as `(alpha * c) * x`; Multi blocks agree in order
    /// with the Host (`alpha * (U x)`). `alpha == 0` — `-0.0` included, by
    /// IEEE comparison — does not read the source (VectorInterface's
    /// `scale(x, 0) = zero(x)`), as on Host; only the sign of an exact zero
    /// may differ.
    ///
    /// # Errors
    ///
    /// The Host variants and messages, with the storage noun naming the
    /// placement (as for [`Self::contract_into`]), plus
    /// [`Error::PlacementMismatch`] for a payload on another device.
    /// Plan-construction failures count as validation.
    ///
    /// # Source compatibility
    ///
    /// As for [`Self::permute`], the `cuda` feature adds these names to a
    /// second `impl`, so the *path* forms such as `TensorMap::permute_into`
    /// become ambiguous (`E0034`). Method-call syntax and
    /// `TensorMap::<R, D>::permute_into` keep working.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn permute_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |_, _| {
                Ok(TreeTransformOperation::permute(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
            |_, _, probe| {
                probe(tree_operation_view(
                    TreeTransformOperationKind::Permute,
                    codomain_axes,
                    domain_axes,
                ))
            },
        )
    }

    /// `destination = alpha * self.braid(codomain_axes, domain_axes, levels) +
    /// beta * destination` on the device: the Host [`TensorMap::braid_into`].
    /// Validation, cost, numerics and failure behavior are
    /// [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn braid_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        let operation = braid_operation(
            self.rank(),
            self.codomain_rank(),
            codomain_axes,
            domain_axes,
            levels,
        )?;
        let probe_operation = operation.clone();
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |_, _| Ok(operation),
            |_, _, probe| {
                probe(tenet_tensors::TreeTransformOperationView::of(
                    &probe_operation,
                ))
            },
        )
    }

    /// `destination = alpha * self.transpose(codomain_axes, domain_axes) +
    /// beta * destination` on the device. Validation, cost, numerics and
    /// failure behavior are [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn transpose_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |source, _| {
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Explicit {
                        codomain_axes,
                        domain_axes,
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |_, _, probe| {
                probe(tree_operation_view(
                    TreeTransformOperationKind::Transpose,
                    codomain_axes,
                    domain_axes,
                ))
            },
        )
    }

    /// `destination = alpha * self.repartition(destination.codomain_rank()) +
    /// beta * destination` on the device. Validation, cost, numerics and
    /// failure behavior are [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn repartition_into(&self, destination: &mut Self, alpha: D, beta: D) -> Result<(), Error> {
        let source_codomain_rank = self.codomain_rank();
        let source_rank = self.rank();
        let destination_codomain_rank = destination.codomain_rank();
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |source, destination| {
                if destination.rank() != source.rank() {
                    return Err(Error::InvalidArgument(format!(
                        "repartition destination rank {} does not match source rank {}",
                        destination.rank(),
                        source.rank()
                    )));
                }
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Repartition {
                        num_codomain: destination.codomain_rank(),
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |source, destination, probe| {
                if destination.rank() == source.rank() {
                    repartition_probe(
                        source_codomain_rank,
                        source_rank,
                        destination_codomain_rank,
                        probe,
                    );
                }
            },
        )
    }

    /// One admission and replay boundary for every typed device `*_into`
    /// tree transform,
    /// mirroring Host `tree_transform_into` step for step.
    ///
    /// The destination is the caller's device buffer, so — unlike the
    /// returning [`Self::tree_transform_cuda`], which owns a freshly uploaded
    /// output and may therefore spend it before the executor's layout verdict
    /// — nothing here may write before every rejection has been decided. That
    /// holds without a new preflight: the executor decides Stage A, the
    /// prepared-structure layout expressibility and Stage C before its first
    /// submission, so a rejected replay leaves the destination byte-identical.
    ///
    /// Lock order is [`Self::tree_transform_cuda`]'s: the pooled Host context
    /// lease that compiles the structure is dropped before the device lease,
    /// and nothing under the device lease leases again.
    fn tree_transform_into_cuda(
        &self,
        destination: &mut Self,
        alpha: D,
        beta: D,
        operation: impl FnOnce(&Self, &Self) -> Result<TreeTransformOperation, Error>,
        exact_layout_probe: impl FnOnce(
            &Self,
            &Self,
            &mut dyn FnMut(tenet_tensors::TreeTransformOperationView<'_>),
        ),
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if identity != TypedSectorAdmission::typed_rule_identity(destination.provider()) {
            return Err(Error::RuleMismatch);
        }

        // Same error kinds, wording and order as Host; only the storage noun
        // names the placement, as in `contract_into`.
        // A lazy adjoint source is rejected rather than lowered: Host rejects
        // it here too, because lowering would produce a result shaped like the
        // adjoint's parent, not like the caller's destination.
        let (source_body, source_storage) =
            match &self.repr {
                TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                    TypedData::Dense(storage) => (body, storage),
                    TypedData::Diagonal(_) => return Err(Error::InvalidArgument(
                        "typed destination tree transform requires an ordinary dense CUDA source"
                            .to_string(),
                    )),
                },
                TypedTensorRepr::Adjoint(_) => {
                    return Err(Error::InvalidArgument(
                        "typed destination tree transform requires an ordinary dense CUDA source"
                            .to_string(),
                    ))
                }
            };
        let (destination_body, destination_storage) = match &destination.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => (body, storage),
                TypedData::Diagonal(_) => {
                    return Err(Error::InvalidArgument(
                        "destination must use ordinary dense CUDA storage".to_string(),
                    ))
                }
            },
            TypedTensorRepr::Adjoint(_) => {
                return Err(Error::InvalidArgument(
                    "destination must use ordinary dense CUDA storage".to_string(),
                ))
            }
        };
        if Arc::ptr_eq(&source_body.data, &destination_body.data) {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }

        // As on Host: a destination proved once to be this operation's
        // result space of this source hits the completed transformer
        // directly, without rebuilding the operation or the result space.
        let mut exact_layout_hit = None;
        exact_layout_probe(self, destination, &mut |view| {
            exact_layout_hit = crate::runtime::Runtime::exact_layout_tree_pair_hit(
                &identity,
                view,
                &source_body.space,
                &destination_body.space,
            );
        });
        let operation = match exact_layout_hit {
            Some(_) => None,
            None => {
                let operation = operation(self, destination)?;
                let expected = source_body
                    .space
                    .transformed_multiplicity_free(&operation)?;
                if destination_body.space.space() != expected.space() {
                    return Err(Error::from(tenet_tensors::OperationError::SpaceMismatch {
                        message: "destination fusion space or block layout does not match the operation result",
                    }));
                }
                Some(operation)
            }
        };
        let required = destination_body.space.space().required_len()?;
        let actual = TensorStorage::len(destination_storage);
        if actual != required {
            return Err(Error::InvalidArgument(format!(
                "destination storage length {actual} does not match required length {required}"
            )));
        }
        if Arc::strong_count(destination_body) != 1
            || Arc::strong_count(&destination_body.data) != 1
        {
            return Err(Error::DestinationShared);
        }
        let destination_placement = destination_storage.placement();

        let source_structure = Arc::clone(source_body.space.space().structure());
        let destination_structure = Arc::clone(destination_body.space.space().structure());
        // Host compiles the structure inside its replay entry point, under the
        // same pooled context lease; here it must happen before the device
        // lease is taken, as in `tree_transform_cuda`. The destination's
        // provider is the authority, exactly as Host's
        // `tree_transform_dyn_into_ref` call passes it.
        let structure = match (exact_layout_hit, &operation) {
            (Some(structure), _) => structure,
            (None, Some(operation)) => {
                let mut lease = self.runtime.lease_context()?;
                lease
                    .context()
                    .multiplicity_free_lane::<D>()?
                    .tree_context_mut()
                    .compile_tree_pair_structure(
                        destination_body.space.provider(),
                        operation,
                        &destination_structure,
                        &source_structure,
                    )?
            }
            (None, None) => unreachable!("a miss builds the operation"),
        };

        {
            let mut lease = self.runtime.lease_cuda()?;
            let (cuda, executor) = lease.split();
            let expected_placement = Placement::Cuda(cuda.device());
            if source_storage.placement() != expected_placement
                || destination_placement != expected_placement
            {
                return Err(Error::PlacementMismatch);
            }
            let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
                return Err(internal_layout_error(
                    "ordinary CUDA destination checked above",
                ));
            };
            let destination_body = Arc::get_mut(destination_body).ok_or_else(|| {
                internal_layout_error("unique CUDA destination body checked above")
            })?;
            let destination_data = Arc::get_mut(&mut destination_body.data).ok_or_else(|| {
                internal_layout_error("unique CUDA destination payload checked above")
            })?;
            let TypedData::Dense(destination_data) = destination_data else {
                return Err(internal_layout_error(
                    "dense CUDA destination checked above",
                ));
            };
            executor.replay(
                cuda,
                &structure,
                &destination_structure,
                &source_structure,
                destination_data,
                source_storage,
                alpha,
                CudaTreeTransformDestination::Axpby(beta),
            )?;
        }
        if let Some(operation) = &operation {
            // Retention is an optimization: a lookup-only key keeps no proof.
            let _ = crate::runtime::Runtime::admit_exact_tree_pair_layout(
                &identity,
                operation,
                &source_body.space,
                destination.logical_space(),
            );
        }
        Ok(())
    }

    /// TensorKit `twist(t, inds)` on a device tensor: each fusion-tree block
    /// is multiplied by the product over `legs` (flat leg indices, codomain
    /// first) of that leg's ribbon-twist eigenvalue.
    ///
    /// Same branches, in the same order, as the Host `twist`: an out-of-range
    /// leg is rejected before the empty-list short circuit, a `NoBraiding`
    /// provider is preflighted for non-unit legs, a twist that is the identity
    /// on every block returns a body-sharing clone without touching the
    /// device, and a lazy adjoint redirects through its parent with the
    /// inverse operation. A twist does not change the space, so the result
    /// keeps the receiver's layout.
    ///
    /// Reference: TensorKit `twist!`
    /// (`src/tensors/indexmanipulations.jl:62-77` @`cfaa073`), which tests
    /// `has_shared_twist` and then `scale!(t[f₁,f₂], θ)` per fusion-tree
    /// block. QSpace has no ribbon-twist path at all — a concrete absence, not
    /// an unexamined one — so the block-scaling decomposition follows
    /// TensorKit and only the dataflow below is Rust/device specific.
    ///
    /// # Cost
    ///
    /// θ is built on the Host from the block structure alone, before the
    /// device lease, and reaches the kernel as the contraction descriptor's
    /// own scale: for every provider this impl admits (`Scalar = f64`) a
    /// ribbon twist is a real sign, never zero, so the descriptor cannot let
    /// CUDA skip a source read: NaN and real infinities propagate as on Host.
    /// An infinite *complex* entry instead becomes NaN in both components,
    /// because the device always multiplies where Host copies a factor-1 block
    /// (the #1301 deviation, disclosed by `cuda_region_axpby`); `f64` and every
    /// finite payload are exact. No θ table is uploaded. A warm call therefore transfers only the #740 output
    /// initialisation — one H2D of the output bytes — downloads nothing,
    /// allocates exactly one device buffer and submits one strided move per
    /// non-empty block. Residual: a whole-buffer bitwise device copy followed
    /// by in-place per-block scaling would submit fewer kernels, but Tenferro
    /// (0.5.0 and 0.6.0) has no in-place strided scale (`cuda_region_axpby` cannot alias
    /// its source and destination), so it is not expressible today.
    ///
    /// Flat elements that belong to no block — only reachable under a padded
    /// expert layout — are zero here, where Host copies the source bytes
    /// through; device structural results have carried that convention since
    /// the transform leaf.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOnDevice`] for a compact (diagonal) device payload:
    /// the Host's compact-spectrum arm needs a representation `to_cuda` never
    /// produces, and a silent dense fallback would hide the cost.
    ///
    /// Under the `cuda` feature `TensorMap<R, D, CudaStorage<D>>` and the Host
    /// `TensorMap<R, D>` both have a `twist`; they are distinct inherent
    /// methods on distinct types, so a path-form call must name the storage.
    pub fn twist(&self, legs: &[usize], direction: Direction) -> Result<Self, Error> {
        self.twist_with_inverse_cuda(legs, direction.is_inverse())
    }

    fn twist_with_inverse_cuda(&self, legs: &[usize], inverse: bool) -> Result<Self, Error> {
        let rank = self.rank();
        let name = if inverse { "inverse twist" } else { "twist" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let provider = self.logical_space().provider();
        reject_unbraided_nonunit_legs(
            provider,
            self.logical_space().space().homspace(),
            legs,
            name,
            true,
        )?;
        // Host order: the lazy adjoint lowers onto its parent with the
        // opposite direction rather than materializing itself. `.adjoint()` is
        // lease-free, so this recursion never nests a device lease.
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            return parent.twist_with_inverse_cuda(&axes, !inverse)?.adjoint();
        }
        let nout = self.codomain_rank();
        let space = self.logical_space().clone();
        // Ahead of the storage kind, as on Host, where an all-ones twist
        // leaves even a compact spectrum untouched.
        if twist_is_identity_over_blocks(provider, space.space().structure(), nout, legs)? {
            return Ok(self.clone());
        }
        let src = self.direct_cuda_storage(name)?;
        let structure = space.space().structure();
        let required_len = space.space().required_len()?;
        // The whole θ table, on the Host, before the lease: one region per
        // block with that block's factor as the descriptor scale. A block
        // whose factor is 1 still has to be moved — the output starts as
        // zeros, where the Host started from a copy of the source.
        let blocks = (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index)?;
                let factor = match block.key() {
                    BlockKey::FusionTree(key) => {
                        twist_block_factor(provider, key, nout, legs, inverse)
                    }
                    _ => 1.0,
                };
                let region = tenet_dense::CudaRegion::new(
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
                .and_then(|region| region.validate_as_destination(name).map(|()| region))
                .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
                Ok((region, D::from_real(factor)))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        Self::validate_cuda_owned_metadata(
            Placement::Cuda(cuda.device()),
            src.placement(),
            required_len,
            src.len(),
        )?;
        // ponytail: #740 — the device seam still initializes an output by
        // uploading zeros; replace only with a measured native allocation.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        for (region, factor) in &blocks {
            // A ribbon twist is never zero, so the factor rides the descriptor
            // scale: no coefficient buffer, and CUDA still has to read the
            // source rather than skipping it as it may for a zero alpha. That
            // keeps NaN and real infinities propagating as on Host; an infinite
            // complex entry is the disclosed #1301 difference, because Host
            // bit-copies a factor-1 block where this always multiplies.
            tenet_dense::cuda_region_axpby::<D>(
                cuda,
                &src.0,
                region,
                false,
                *factor,
                tenet_dense::CudaRegionCoefficient::One,
                tenet_dense::CudaRegionBeta::Overwrite,
                &mut dst.0,
                region,
            )
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
        }
        drop(lease);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, dst)),
        })
    }
}
