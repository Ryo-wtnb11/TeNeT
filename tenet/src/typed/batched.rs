//! Structure signatures: the key under which tensors can share one prepared
//! plan and one stacked payload layout (#1287).
//!
//! Neither TensorKit nor QSpace has a batched path, so the signature is
//! defined by what TeNeT's eager operations read from an operand: the rule
//! authority, the logical hom space, the fusion-tree block structure, the
//! payload placement and the Runtime. The payload dtype is the type parameter
//! `D` of every consumer and is therefore checked by the compiler, not here.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use tenet_core::{
    BlockStructureContent, CheckedFusionAlgebra, CoupledSectorRegion, HomSpaceId,
    MultiplicityFreeRigidSymbols, Placement, RuleIdentity, SectorCodec, SectorId, TensorStorage,
};
use tenet_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use tenet_operations::FusionBlockContractPlan;
use tenet_tensors::{zeroed_payload, BoundDynamicTensorRef, FusionOperand, OperationError};

#[cfg(feature = "cuda")]
use super::{dense_err, CudaFactorizationPayload, CudaPayload, CudaStorage};
use super::{
    owned_repr, require_selected_leg_of, space_with_replaced_leg, BoundDynamicFusionMapSpace,
    FactorizationScalar, LegSelection, Runtime, SectorSpectrum, TensorMap, TensorScalar, TypedData,
    TypedFacadeError, TypedSectorAdmission, TypedTensorBody, TypedTensorRepr,
    TypedTensorRootDispatch,
};
use crate::error::Error;
use crate::tensor_core::internal_layout_error;
use crate::RuntimeIdentity;

/// Content identity of a tensor's structure, placement and Runtime.
///
/// Two signatures are equal exactly when the fusion-rule identity, the hom
/// space (every codomain and domain leg's sectors, degeneracies and dual
/// flags), the block structure content (fusion-tree blocks, their order and
/// dense offsets), the placement and the Runtime are all equal. Equality
/// never depends on process-local intern ids, so it survives interner
/// eviction, `reset_core_intern_tables` and oversized structures that bypass
/// the interner.
///
/// The signature describes the logical structure only. It does not record
/// whether the payload is an owned dense buffer, a lazy adjoint or a compact
/// diagonal spectrum; a consumer that relies on the dense layout must admit
/// the representation separately.
///
/// Comparison is O(1) while the compared structures share their interned
/// allocations, and O(blocks) otherwise. `Hash` covers only O(1) prehashes;
/// signatures that differ only in block structure share a bucket and are
/// separated by `Eq`. A signature holds a non-owning Runtime handle and does
/// not keep the Runtime alive.
#[derive(Clone)]
pub struct StructureSignature {
    rule: RuleIdentity,
    homspace: HomSpaceId,
    // Why the content and not `content_id`: the id is a process-local
    // insertion counter reissued after eviction or reset, and oversized
    // contents never enter the interner, so equal content can carry
    // different ids.
    structure: Arc<BlockStructureContent>,
    placement: Placement,
    runtime: RuntimeIdentity,
}

impl PartialEq for StructureSignature {
    fn eq(&self, other: &Self) -> bool {
        self.first_mismatch(other).is_none()
    }
}

impl Eq for StructureSignature {}

impl Hash for StructureSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.homspace.hash(state);
        self.placement.hash(state);
        self.runtime.hash(state);
    }
}

/// The first determinant in which two [`StructureSignature`]s differ.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SignatureField {
    /// Host or device ordinal.
    Placement,
    /// The owning Runtime.
    Runtime,
    /// The fusion-rule identity.
    Rule,
    /// A codomain or domain leg's sectors, degeneracies or dual flag.
    HomSpace,
    /// The fusion-tree blocks, their order or dense offsets.
    BlockStructure,
}

/// A payload representation a [`StackedTensorMap`] cannot hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BatchMemberRepresentation {
    /// A lazy adjoint view over its parent's payload.
    LazyAdjoint,
    /// A compact diagonal spectrum, whose dense layout is not stored.
    CompactDiagonal,
}

impl StructureSignature {
    /// The first field in which `other` differs, in the order `==` checks
    /// them, or `None` when the signatures are equal.
    pub(crate) fn first_mismatch(&self, other: &Self) -> Option<SignatureField> {
        if self.placement != other.placement {
            Some(SignatureField::Placement)
        } else if self.runtime != other.runtime {
            Some(SignatureField::Runtime)
        } else if self.rule != other.rule {
            Some(SignatureField::Rule)
        } else if self.homspace != other.homspace {
            Some(SignatureField::HomSpace)
        } else if !(Arc::ptr_eq(&self.structure, &other.structure)
            || self.structure == other.structure)
        {
            Some(SignatureField::BlockStructure)
        } else {
            None
        }
    }

    /// The process-local block-structure intern id, for diagnostics and
    /// tests only. It is not part of equality.
    #[doc(hidden)]
    pub fn content_id(&self) -> usize {
        self.structure.id()
    }
}

impl std::fmt::Debug for StructureSignature {
    // Why not derive: the hom-space key and block list are O(legs + blocks)
    // and would make a diagnostic proportional to the tensor's structure.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StructureSignature")
            .field("rule", &self.rule)
            .field("content_id", &self.structure.id())
            .field("placement", &self.placement)
            .field("runtime_alive", &self.runtime.is_alive())
            .finish_non_exhaustive()
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    S: TensorStorage<D>,
{
    /// The [`StructureSignature`] of this tensor's logical structure.
    ///
    /// Builds no new structure: it clones the shared hom-space and block
    /// structure handles and a weak Runtime handle.
    pub fn structure_signature(&self) -> StructureSignature {
        StructureSignature::of_space(self.logical_space(), self.placement(), &self.runtime)
    }
}

impl StructureSignature {
    /// The signature of a tensor over `space` with `placement` on `runtime`.
    fn of_space<R: TypedSectorAdmission>(
        space: &BoundDynamicFusionMapSpace<R>,
        placement: Placement,
        runtime: &Runtime,
    ) -> Self {
        // Why the admission's identity first: it is the identity the layout
        // was validated against and clones without allocating, whereas
        // `typed_rule_identity` rebuilds canonical bytes for content-keyed
        // checked Generic providers. Every bound space carries one.
        let rule = match space.space().admission().rule_identity() {
            Some(identity) => identity.clone(),
            None => TypedSectorAdmission::typed_rule_identity(space.provider()),
        };
        StructureSignature {
            rule,
            homspace: space.space().homspace().id(),
            structure: space.space().structure().content_key(),
            placement,
            runtime: runtime.identity(),
        }
    }
}

/// `B` tensors of one [`StructureSignature`] in one buffer: member `i`
/// occupies `[i * L, (i + 1) * L)`, where `L` is the signature's required
/// payload length. There is no padding and no member leg; the member axis
/// exists only as this stride.
///
/// A stack is the operand of a prepared batched handle, not a tensor. It
/// deliberately does not implement `TensorStorage`, whose `len` would be
/// `B * L` and would let an ordinary per-tensor kernel run on member 0 only:
///
/// ```compile_fail
/// use tenet::core::{TensorStorage, U1FusionRule};
/// use tenet::typed::StackedTensorMap;
///
/// fn as_storage<S: TensorStorage<f64>>(_: &S) {}
/// fn reject(stack: &StackedTensorMap<U1FusionRule, f64>) {
///     as_storage(stack);
/// }
/// ```
///
/// The same imports compile when the stack is used as a stack:
///
/// ```
/// use tenet::core::{TensorStorage, U1FusionRule};
/// use tenet::typed::StackedTensorMap;
///
/// fn as_storage<S: TensorStorage<f64>>(_: &S) {}
/// fn accept(stack: &StackedTensorMap<U1FusionRule, f64>) {
///     let _ = stack.signature();
///     as_storage(&vec![0.0_f64]);
/// }
/// ```
pub struct StackedTensorMap<R, D, S = Vec<D>> {
    runtime: Runtime,
    space: BoundDynamicFusionMapSpace<R>,
    signature: StructureSignature,
    storage: S,
    members: usize,
    member_len: usize,
    _payload: PhantomData<D>,
}

impl<R, D, S> StackedTensorMap<R, D, S> {
    /// The signature every member shares, with the stack's placement.
    pub fn signature(&self) -> &StructureSignature {
        &self.signature
    }

    /// Number of legs of every member.
    fn rank(&self) -> usize {
        let homspace = self.space.space().homspace();
        homspace.codomain().len() + homspace.domain().len()
    }

    /// The member count `B`. Never zero: [`Self::pack`] rejects an empty
    /// batch.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.members
    }

    /// Checks a [`Self::select`] index list and returns the selected payload
    /// length `B' * L`.
    fn selection_len(&self, members: &[usize]) -> Result<usize, Error> {
        if members.is_empty() {
            return Err(Error::InvalidArgument(
                "a stack needs at least one member".into(),
            ));
        }
        if let Some(&member) = members.iter().find(|&&member| member >= self.members) {
            return Err(Error::BatchMemberOutOfRange {
                member,
                len: self.members,
            });
        }
        self.member_len
            .checked_mul(members.len())
            .ok_or_else(|| Error::InvalidArgument("stacked payload length overflows usize".into()))
    }

    /// The destination space and the start table of [`Self::restrict_leg`],
    /// checked exactly as the eager [`TensorMap::restrict_leg`] checks them.
    #[allow(clippy::type_complexity)]
    fn restrict_plan(
        &self,
        axis: usize,
        selection: &LegSelection<R>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<(SectorId, usize)>), TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        require_selected_leg_of(&self.space, axis, selection.parent(), "restrict_leg")?;
        let destination = space_with_replaced_leg(&self.space, axis, selection.subspace().leg())?;
        Ok((destination, selection.start_table()))
    }

    /// A stack of this one's member count and Runtime over `space`, holding
    /// `storage`; the placement is read from `storage`.
    fn with_space<T: TensorStorage<D>>(
        &self,
        space: BoundDynamicFusionMapSpace<R>,
        storage: T,
    ) -> Result<StackedTensorMap<R, D, T>, Error>
    where
        R: TypedSectorAdmission,
    {
        Ok(StackedTensorMap {
            runtime: self.runtime.clone(),
            signature: StructureSignature::of_space(&space, storage.placement(), &self.runtime),
            member_len: space.space().required_len()?,
            space,
            storage,
            members: self.members,
            _payload: PhantomData,
        })
    }

    /// This stack's space, signature and Runtime over `storage` holding
    /// `members` members; the placement is read from `storage`.
    fn with_storage<T: TensorStorage<D>>(
        &self,
        storage: T,
        members: usize,
    ) -> StackedTensorMap<R, D, T> {
        StackedTensorMap {
            runtime: self.runtime.clone(),
            space: self.space.clone(),
            signature: StructureSignature {
                placement: storage.placement(),
                ..self.signature.clone()
            },
            storage,
            members,
            member_len: self.member_len,
            _payload: PhantomData,
        }
    }
}

impl<R, D> StackedTensorMap<R, D>
where
    R: TypedSectorAdmission,
    D: Copy,
{
    /// Copies `members` into one stacked buffer: one allocation of `B * L`
    /// elements and `B` copies.
    ///
    /// Every member is checked before anything is copied. A lazy adjoint or
    /// a compact diagonal member returns [`Error::UnsupportedBatchMember`];
    /// a member whose signature differs from member 0's returns
    /// [`Error::BatchSignatureMismatch`] naming that member and the first
    /// differing field. An empty batch is an [`Error::InvalidArgument`].
    ///
    /// Lazy adjoints are not packed: a handle consumes adjoint operands
    /// through its orientation flag (#1287 leaf L8) instead.
    pub fn pack<T: AsRef<TensorMap<R, D>>>(members: &[T]) -> Result<Self, Error> {
        let first = members
            .first()
            .ok_or_else(|| Error::InvalidArgument("a stack needs at least one member".into()))?
            .as_ref();
        let signature = first.structure_signature();
        let payloads = members
            .iter()
            .enumerate()
            .map(|(member, tensor)| {
                let tensor = tensor.as_ref();
                let payload = dense_payload(tensor).map_err(|representation| {
                    Error::UnsupportedBatchMember {
                        member,
                        representation,
                    }
                })?;
                match signature.first_mismatch(&tensor.structure_signature()) {
                    Some(field) => Err(Error::BatchSignatureMismatch {
                        member: Some(member),
                        field,
                    }),
                    None => Ok(payload),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let space = first.logical_space().clone();
        let member_len = space.space().required_len()?;
        let total = member_len.checked_mul(members.len()).ok_or_else(|| {
            Error::InvalidArgument("stacked payload length overflows usize".into())
        })?;
        let mut storage = Vec::with_capacity(total);
        for payload in payloads {
            storage.extend_from_slice(payload);
        }
        Ok(Self {
            runtime: first.runtime.clone(),
            space,
            signature,
            storage,
            members: members.len(),
            member_len,
            _payload: PhantomData,
        })
    }

    /// An owned, bit-identical copy of member `i`.
    ///
    /// An index at or past [`Self::len`] is
    /// [`Error::BatchMemberOutOfRange`].
    pub fn member(&self, i: usize) -> Result<TensorMap<R, D>, Error> {
        if i >= self.members {
            return Err(Error::BatchMemberOutOfRange {
                member: i,
                len: self.members,
            });
        }
        let start = i * self.member_len;
        let data = self.storage[start..start + self.member_len].to_vec();
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.space.clone(), data)),
        })
    }

    /// A new stack whose member `j` is member `members[j]` of this one, with
    /// the same signature, placement and Runtime: one allocation of
    /// `members.len() * L` elements and one slice copy per selected member.
    ///
    /// Indices may appear in any order and may repeat (a repeated index
    /// replicates that member), so the result can be longer than this stack.
    /// An index at or past [`Self::len`] is
    /// [`Error::BatchMemberOutOfRange`]; an empty list is an
    /// [`Error::InvalidArgument`], because a stack is never empty. Both are
    /// checked before anything is allocated.
    ///
    /// This is how a caller drops the members a batched handle rejected, or
    /// splits a stack into groups whose truncated signatures agree.
    pub fn select(&self, members: &[usize]) -> Result<Self, Error> {
        let mut storage = Vec::with_capacity(self.selection_len(members)?);
        for &member in members {
            storage.extend_from_slice(&self.storage[member * self.member_len..][..self.member_len]);
        }
        Ok(self.with_storage(storage, members.len()))
    }
}

impl<R, D> StackedTensorMap<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Restricts leg `axis` of every member to the subspace `selection` names:
    /// member `i` of the result is `self.member(i)?.restrict_leg(axis,
    /// selection)?`, bit for bit, with the same space and signature.
    ///
    /// One selection serves every member, which is what grouping members by
    /// their truncated signature (then [`Self::select`]) produces.
    ///
    /// # Cost
    ///
    /// The eager block plan is built once; each destination block is then one
    /// strided copy for all `B` members, the member axis being one more
    /// dimension of the copy. `B · L'` elements move into one output
    /// allocation, left unfilled when the blocks tile it (as eager), plus
    /// `O(rank + blocks)` structural work independent of `B`.
    ///
    /// # Errors
    ///
    /// Exactly the eager [`TensorMap::restrict_leg`] errors: an out-of-range
    /// `axis` or a leg other than [`LegSelection::parent`] is an
    /// [`Error::InvalidArgument`], a selection of another rule is
    /// [`Error::RuleMismatch`]. Nothing is allocated before every check.
    pub fn restrict_leg(
        &self,
        axis: usize,
        selection: &LegSelection<R>,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R::Mode: TypedTensorRootDispatch<R>,
    {
        let (destination, table) = self.restrict_plan(axis, selection)?;
        let _host_pool = self.runtime.enter_host_pool();
        let mut starts: Vec<tenet_tensors::SectorStartTable<'_>> = vec![None; self.rank()];
        starts[axis] = Some(table.as_slice());
        let data = tenet_tensors::stacked_fusion_restrict_owned(
            destination.space().structure(),
            FusionOperand::direct(self.space.space()),
            &self.storage,
            self.members,
            &starts,
        )
        .map_err(Error::from)?;
        Ok(self.with_space(destination, data)?)
    }
}

/// Composition `lhs · rhs` (TensorKit `mul!`) of every member pair of two
/// stacks, prepared once for one pair of structure signatures (#1287).
///
/// Per member it is the eager [`TensorMap::compose`] of the same placement:
/// the same destination space and the same fully-direct coupled-block plan,
/// compiled once here instead of once per member and call. The member axis
/// exists only as a dense stride: every coupled-sector GEMM of the plan runs
/// once for all members.
///
/// - **CUDA**: one batched GEMM per coupled sector per call, independent of
///   the member count `B`.
/// - **Host**: one grouped GEMM submission per call over all `B × blocks`
///   matrices, through the Runtime's dense backend.
///
/// The handle is an expert opt-in for repeated work over a stable `B`; the
/// eager API stays primary. The caller owns the set of handles (one per
/// signature pair); a stack of another signature is a typed error, never a
/// silent re-plan.
///
/// # Supported scope
///
/// Identity orientation only: an adjoint operand must be materialized before
/// it is packed. A composition whose plan is not the canonical fully-direct
/// one (for example an expert-layer tiling whose tree stackings differ) is
/// rejected by [`Self::new`] with the same
/// `UnsupportedTensorContractScope` the eager device composition reports.
///
/// # Retained state and shared effects
///
/// The handle owns its output stack and the per-`B` job layout, reported by
/// [`Self::retained_bytes`] and freed on drop. A change of `B` rebuilds them.
/// On CUDA it also reserves its cuTENSOR plan entries in the device context's
/// ledger at [`Self::new`] and returns them on drop (reported by
/// `Runtime::cuda_plan_cache_stats`), and `execute_into` grows the context's
/// shared zero template to the largest inactive block times `B`.
pub struct PreparedCompose<R, D, S = Vec<D>> {
    runtime: Runtime,
    lhs: StructureSignature,
    rhs: StructureSignature,
    output_signature: StructureSignature,
    space: BoundDynamicFusionMapSpace<R>,
    plan: Arc<FusionBlockContractPlan<f64>>,
    member_len: usize,
    output: Option<StackedTensorMap<R, D, S>>,
    /// Host only: the plan expanded over the current `B`.
    replay: Option<StackedDirectReplay>,
    #[cfg(feature = "cuda")]
    device: DeviceComposeState,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct DeviceComposeState {
    /// The `B` the zero regions and the zero template were sized for.
    members: usize,
    /// Each inactive destination layout with a trailing member axis.
    zero_regions: Vec<tenet_dense::CudaRegion>,
    /// Plan entries this handle holds in the device context's ledger.
    reserved_plan_entries: usize,
}

impl<R, D, S> PreparedCompose<R, D, S>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    S: TensorStorage<D>,
{
    /// Prepares the composition of stacks with `lhs`'s and `rhs`'s
    /// signatures. Their payloads are not read, and `B` is not fixed.
    ///
    /// On a device placement it also reserves the handle's cuTENSOR plan
    /// entries in the device context's ledger: one per distinct GEMM shape and
    /// one per distinct inactive fill layout. The count does not depend on
    /// `B`, so it is reserved once here and returned on drop.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`] and [`Error::PlacementMismatch`] for
    /// operands of different Runtimes or placements; the eager composition's
    /// structural errors for spaces that do not compose; and
    /// `UnsupportedTensorContractScope` for a plan that is not fully direct.
    pub fn new(
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
    ) -> Result<Self, Error> {
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut handle = Self::prepare(lhs, rhs)?;
        #[cfg(feature = "cuda")]
        if let Placement::Cuda(_) = handle.output_signature.placement {
            handle.reserve_plan_entries()?;
        }
        Ok(handle)
    }

    fn prepare(
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
    ) -> Result<Self, Error> {
        if !lhs.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let placement = lhs.signature.placement;
        if rhs.signature.placement != placement {
            return Err(Error::PlacementMismatch);
        }
        // The plan compile is lease-free Host work, so it runs in this
        // runtime's pool like every other eager Host operation (#1531).
        let _host_pool = lhs.runtime.enter_host_pool();
        let lhs_space = lhs.space.space();
        let rhs_space = rhs.space.space();
        let lhs_axes: Vec<usize> = (lhs_space.nout()..lhs_space.nout() + lhs_space.nin()).collect();
        let rhs_axes: Vec<usize> = (0..rhs_space.nout()).collect();
        let space = BoundDynamicFusionMapSpace::contracted_multiplicity_free(
            &lhs.space, &rhs.space, &lhs_axes, &rhs_axes,
        )?;
        let plan = tenet_tensors::compile_direct_composition_plan(
            &space,
            FusionOperand::direct(lhs_space),
            FusionOperand::direct(rhs_space),
            &lhs_axes,
            &rhs_axes,
        )?;
        plan.require_identity_direct_replay()?;
        Ok(Self {
            runtime: lhs.runtime.clone(),
            lhs: lhs.signature.clone(),
            rhs: rhs.signature.clone(),
            output_signature: StructureSignature::of_space(&space, placement, &lhs.runtime),
            member_len: space.space().required_len()?,
            space,
            plan,
            output: None,
            replay: None,
            #[cfg(feature = "cuda")]
            device: DeviceComposeState::default(),
        })
    }

    #[cfg(feature = "cuda")]
    fn reserve_plan_entries(&mut self) -> Result<(), Error> {
        let gemms = self.plan.distinct_direct_gemm_shapes();
        let fills = self
            .plan
            .inactive_destination_regions()
            .iter()
            .map(|layout| (&layout.block.shape, &layout.block.strides))
            .collect::<std::collections::HashSet<_>>()
            .len();
        self.device.reserved_plan_entries = self
            .runtime
            .lease_cuda()?
            .reserve_plan_entries(gemms + fills)
            .map_err(tenet_operations::OperationError::Dense)?;
        Ok(())
    }
}

impl<R, D, S> PreparedCompose<R, D, S> {
    /// The signature of every output member.
    pub fn output_signature(&self) -> &StructureSignature {
        &self.output_signature
    }

    /// Moves the handle-owned output out; the next [`Self::execute`]
    /// allocates a new one.
    pub fn take_output(&mut self) -> Option<StackedTensorMap<R, D, S>> {
        self.output.take()
    }

    /// Bytes the handle itself retains: the output payload (host or device)
    /// and the host job layout. The shared plan and the context-level effects
    /// (plan-entry reservation, zero template) are not included.
    pub fn retained_bytes(&self) -> usize {
        let output = self.output.as_ref().map_or(0, |output| {
            output.members * output.member_len * std::mem::size_of::<D>()
        });
        let replay = self
            .replay
            .as_ref()
            .map_or(0, StackedDirectReplay::retained_bytes);
        #[cfg(feature = "cuda")]
        let regions = self.device.zero_regions.capacity()
            * std::mem::size_of::<tenet_dense::CudaRegion>()
            + self
                .device
                .zero_regions
                .iter()
                .map(|region| 2 * region.dims().len() * std::mem::size_of::<usize>())
                .sum::<usize>();
        #[cfg(not(feature = "cuda"))]
        let regions = 0;
        output + replay + regions
    }

    /// Checks both operand stacks against the prepared signatures, O(1) per
    /// stack, and returns their common member count.
    fn check_operands(
        &self,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
    ) -> Result<usize, Error> {
        for (stack, expected) in [(lhs, &self.lhs), (rhs, &self.rhs)] {
            if let Some(field) = expected.first_mismatch(&stack.signature) {
                return Err(Error::BatchSignatureMismatch {
                    member: None,
                    field,
                });
            }
        }
        if lhs.members != rhs.members {
            return Err(Error::InvalidArgument(format!(
                "operand stacks hold {} and {} members",
                lhs.members, rhs.members
            )));
        }
        Ok(lhs.members)
    }

    /// Checks a caller destination against the output signature and `B`.
    fn check_destination(
        &self,
        dst: &StackedTensorMap<R, D, S>,
        members: usize,
    ) -> Result<(), Error> {
        if let Some(field) = self.output_signature.first_mismatch(&dst.signature) {
            return Err(Error::BatchSignatureMismatch {
                member: None,
                field,
            });
        }
        if dst.members != members {
            return Err(Error::InvalidArgument(format!(
                "destination stack holds {} members, operands {members}",
                dst.members
            )));
        }
        Ok(())
    }

    fn output_stack(&self, storage: S, members: usize) -> StackedTensorMap<R, D, S> {
        StackedTensorMap {
            runtime: self.runtime.clone(),
            space: self.space.clone(),
            signature: self.output_signature.clone(),
            storage,
            members,
            member_len: self.member_len,
            _payload: PhantomData,
        }
    }

    fn payload_len(&self, members: usize) -> Result<usize, Error> {
        self.member_len
            .checked_mul(members)
            .ok_or_else(|| Error::InvalidArgument("stacked payload length overflows usize".into()))
    }
}

impl<R, D> PreparedCompose<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Composes every member pair into the handle-owned output and borrows
    /// it.
    ///
    /// The output is allocated zeroed when absent or when `B` changes; the
    /// destination blocks the plan never writes stay zero, and every other
    /// block is overwritten. A warm call allocates nothing of TeNeT's: the
    /// output, the job list and the fill strides are reused. The dense
    /// backend's own grouped-GEMM validation (Tenferro 0.7.1) still allocates
    /// once per call, as it does for eager `compose`.
    ///
    /// # Errors
    ///
    /// [`Error::BatchSignatureMismatch`] with `member: None` for a stack of
    /// another signature, and [`Error::InvalidArgument`] for operand stacks
    /// of different member counts, all before any work. On any error the
    /// handle-owned output is unspecified until the next successful call.
    pub fn execute(
        &mut self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
    ) -> Result<&StackedTensorMap<R, D>, Error> {
        let members = self.check_operands(lhs, rhs)?;
        self.prepare_host_replay(members)?;
        let mut output = match self
            .output
            .take()
            .filter(|output| output.members == members)
        {
            Some(output) => output,
            None => self.output_stack(zeroed_payload(self.payload_len(members)?), members),
        };
        let result = self.run_host(lhs, rhs, &mut output.storage, false);
        let output = self.output.insert(output);
        result.map(|()| &*output)
    }

    /// Composes every member pair into the caller's `dst`, overwriting it.
    ///
    /// `dst` must carry [`Self::output_signature`] and the operands' member
    /// count. Its prior contents never reach the result: the blocks the plan
    /// does not write are zero-filled on every call (one strided fill per
    /// such block over all members).
    ///
    /// # Errors
    ///
    /// As [`Self::execute`], plus the same typed errors for `dst`. After an
    /// error the contents of `dst` are unspecified.
    pub fn execute_into(
        &mut self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut StackedTensorMap<R, D>,
    ) -> Result<(), Error> {
        let members = self.check_operands(lhs, rhs)?;
        self.check_destination(dst, members)?;
        self.prepare_host_replay(members)?;
        self.run_host(lhs, rhs, &mut dst.storage, true)
    }

    fn prepare_host_replay(&mut self, members: usize) -> Result<(), Error> {
        if self
            .replay
            .as_ref()
            .is_none_or(|replay| replay.members() != members)
        {
            self.replay = None;
            self.replay = Some(StackedDirectReplay::new(Arc::clone(&self.plan), members)?);
        }
        Ok(())
    }

    fn run_host(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut Vec<D>,
        zero_inactive: bool,
    ) -> Result<(), Error> {
        let replay = self
            .replay
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("host replay is not prepared".into()))?;
        let members = replay.members();
        let lhs =
            StackedStorageView::new::<D>(&lhs.storage, lhs.member_len, members, lhs.member_len)?;
        let rhs =
            StackedStorageView::new::<D>(&rhs.storage, rhs.member_len, members, rhs.member_len)?;
        let mut dst =
            StackedStorageViewMut::new::<D>(dst, self.member_len, members, self.member_len)?;
        let mut lease = self.runtime.lease_context()?;
        lease
            .context()
            .multiplicity_free_lane::<D>()?
            .execute_stacked_direct_host(replay, &mut dst, &lhs, &rhs, zero_inactive)?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl<R, D> PreparedCompose<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    /// The device form of the Host [`PreparedCompose::execute`]: one batched
    /// GEMM per coupled sector, and no transfer on a warm call. A new or
    /// changed `B` allocates the output with one zero upload, as the eager
    /// composition initializes its output.
    pub fn execute(
        &mut self,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
    ) -> Result<&StackedTensorMap<R, D, CudaStorage<D>>, Error> {
        let members = self.check_operands(lhs, rhs)?;
        let len = self.payload_len(members)?;
        let mut lease = self.runtime.lease_cuda()?;
        let mut output = match self
            .output
            .take()
            .filter(|output| output.members == members)
        {
            Some(output) => output,
            None => {
                let storage = CudaStorage::upload_members(
                    &lease,
                    zeroed_payload(len),
                    self.member_len,
                    members,
                )?;
                self.output_stack(storage, members)
            }
        };
        let result = self.run_cuda(&mut lease, lhs, rhs, &mut output.storage, false);
        drop(lease);
        let output = self.output.insert(output);
        result.map(|()| &*output)
    }

    /// The device form of the Host [`PreparedCompose::execute_into`]: one
    /// rank-2 zero fill (`[len, B]`) per inactive block per call, from the
    /// context's zero template, which a new `B` grows once.
    pub fn execute_into(
        &mut self,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        dst: &mut StackedTensorMap<R, D, CudaStorage<D>>,
    ) -> Result<(), Error> {
        let members = self.check_operands(lhs, rhs)?;
        self.check_destination(dst, members)?;
        // A second handle to the Runtime (one reference count) so the lease
        // does not borrow `self` while the zero regions are rebuilt.
        let runtime = self.runtime.clone();
        let mut lease = runtime.lease_cuda()?;
        self.prepare_zero_regions(&mut lease, members)?;
        self.run_cuda(&mut lease, lhs, rhs, &mut dst.storage, true)
    }

    fn prepare_zero_regions(
        &mut self,
        ctx: &mut tenet_dense::CudaDenseContext,
        members: usize,
    ) -> Result<(), Error> {
        if self.device.members == members {
            return Ok(());
        }
        let mut regions = Vec::with_capacity(self.plan.inactive_destination_regions().len());
        let mut largest = 0usize;
        for layout in self.plan.inactive_destination_regions() {
            let block = &layout.block;
            let unsigned = |value: isize| {
                usize::try_from(value).map_err(|_| {
                    Error::InvalidArgument(
                        "inactive destination layout has a negative stride".into(),
                    )
                })
            };
            let mut dims = block.shape.clone();
            dims.push(members);
            let mut strides = block
                .strides
                .iter()
                .map(|&stride| unsigned(stride))
                .collect::<Result<Vec<_>, _>>()?;
            strides.push(self.member_len);
            let region = tenet_dense::CudaRegion::new(dims, strides, unsigned(block.offset)?)
                .map_err(tenet_operations::OperationError::Dense)?;
            region
                .validate_as_destination("prepared compose zero fill")
                .map_err(tenet_operations::OperationError::Dense)?;
            largest = largest.max(block.shape.iter().product::<usize>());
            regions.push(region);
        }
        ctx.reserve_zero_template::<D>(
            largest
                .checked_mul(members)
                .ok_or_else(|| Error::InvalidArgument("zero template length overflows".into()))?,
        )
        .map_err(tenet_operations::OperationError::Dense)?;
        self.device.zero_regions = regions;
        self.device.members = members;
        Ok(())
    }

    fn run_cuda(
        &self,
        ctx: &mut tenet_dense::CudaDenseContext,
        lhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        rhs: &StackedTensorMap<R, D, CudaStorage<D>>,
        dst: &mut CudaStorage<D>,
        zero_inactive: bool,
    ) -> Result<(), Error> {
        let members = lhs.members;
        let lhs =
            StackedStorageView::new::<D>(&lhs.storage, lhs.member_len, members, lhs.member_len)?;
        let rhs =
            StackedStorageView::new::<D>(&rhs.storage, rhs.member_len, members, rhs.member_len)?;
        let mut dst =
            StackedStorageViewMut::new::<D>(dst, self.member_len, members, self.member_len)?;
        if zero_inactive {
            for region in &self.device.zero_regions {
                tenet_dense::cuda_region_zero::<D>(ctx, &mut dst.storage_mut().0, region)
                    .map_err(tenet_operations::OperationError::Dense)?;
            }
        }
        self.plan.execute_direct_on_storage_prezeroed(
            &mut tenet_operations::cuda::CudaStackedStorageGemm::new(ctx),
            &mut dst,
            &lhs,
            &rhs,
        )?;
        Ok(())
    }
}

impl<R, D, S> Drop for PreparedCompose<R, D, S> {
    /// Returns the ledger reservation through the poison-recovering device
    /// lease; without device state there is nothing to return. Never panics.
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if self.device.reserved_plan_entries > 0 {
            if let Some(mut lease) = self.runtime.lease_cuda_for_maintenance() {
                lease.release_plan_entries(self.device.reserved_plan_entries);
            }
        }
    }
}

/// Why a [`BatchError::MemberRejected`] member failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MemberFault {
    /// A coupled-sector block is not Hermitian under the eager rule
    /// `||(A - A†)/2||_F <= 64 eps ||A||_F`, or holds a non-finite entry.
    NotHermitian,
    /// The solver returned a non-finite eigenvalue.
    NonFiniteEigenvalue,
}

/// The error of a prepared batched factorization. A batch fails as a whole:
/// no member's output is returned when any member fails.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BatchError {
    /// Every member that failed, in member order, with its reason. Admission
    /// rejections are found before any solver work.
    MemberRejected {
        /// `(member, fault)` for each failing member.
        members: Vec<(usize, MemberFault)>,
    },
    /// The device solver failed on coupled-sector block `block` (the index of
    /// the block among the source's coupled sectors, in layout order).
    ///
    /// The failing member is not known: Tenferro 0.7.1 reads the solver
    /// status of a whole batch and reports its first failure without the
    /// member index.
    Solver {
        /// The coupled-sector block whose batched solve failed.
        block: usize,
        /// The backend's error.
        source: Error,
    },
    /// A structural, capability or backend error, as the eager operation
    /// reports it. Nothing is attributed to a member.
    Operation(Error),
}

impl From<Error> for BatchError {
    fn from(error: Error) -> Self {
        Self::Operation(error)
    }
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemberRejected { members } => write!(f, "batch members rejected: {members:?}"),
            Self::Solver { block, source } => {
                write!(f, "solver failed on coupled-sector block {block}: {source}")
            }
            Self::Operation(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for BatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MemberRejected { .. } => None,
            Self::Solver { source, .. } | Self::Operation(source) => Some(source),
        }
    }
}

/// The factors of one [`PreparedEighFull::execute`], borrowed from the
/// handle.
///
/// Member `i` of `d` and `v` is the eager `eigh_full` of the same placement
/// on member `i` of the source; `spectra[i]` is that member's eigenvalues per
/// coupled sector, in sector-label order, each descending by `|λ|` as in `d`.
/// `d` is stored dense: a stack holds dense payloads only, so the eager Host
/// compact diagonal is materialized.
pub struct EighStackOutput<'a, R: SectorCodec, D, S = Vec<D>> {
    /// The eigenvalues on the diagonal of `bond <- bond`.
    pub d: &'a StackedTensorMap<R, D, S>,
    /// The eigenvectors `codomain <- bond`, one column per eigenvalue.
    pub v: &'a StackedTensorMap<R, D, S>,
    /// Per member, per coupled sector: the eigenvalues, already on the host.
    pub spectra: &'a [Vec<SectorSpectrum<<R as SectorCodec>::Sector>>],
}

/// The Hermitian eigendecomposition `t = v d v†` (MatrixAlgebraKit
/// `eigh_full`) of every member of a stack, prepared once for one structure
/// signature (#1287). Real payloads only; a complex payload is rejected by
/// [`Self::new`] until Tenferro's batched complex solver works
/// (tenferro-rs#1923).
///
/// Per member it is the eager [`TensorMap::eigh_full`] of the same placement,
/// within that operation's gauge contract:
///
/// - **CUDA**: the eager device plan (coupled-sector routes, the eigenvector
///   and diagonal factor spaces) is compiled once here from the structure
///   alone, since `eigh_full` keeps every eigenpair. Each call then admits
///   the whole batch (at most three downloads), solves each coupled sector
///   of all members with one batched solver call, downloads every spectrum
///   once, sorts each member by descending `|λ|` on the host as eager does,
///   and assembles `v` by one batched column gather per coupled sector
///   followed by one strided copy per aligned sector (per codomain tree
///   otherwise). Submissions, host syncs and
///   downloads per call do not depend on the member count `B`. The raw
///   cuSOLVER gauge is kept, as eager keeps it; with `B > 1` the batched
///   solver may differ from the eager one in the last ULPs, so results are
///   deterministic for a fixed `B` but not bit-stable across `B`. At `B = 1`
///   the solver and host code are the eager ones.
/// - **Host**: every member runs the eager Host per-block path, gauge-fixed
///   exactly as Host eager (the largest-magnitude entry of each eigenvector
///   real and positive), after the whole batch is admitted.
///
/// # Supported scope
///
/// Endomorphisms with packed coupled-sector regions (every structure a
/// public constructor builds). A batch fails as a whole
/// ([`BatchError`]): on any error no output is returned, and none is
/// observable afterwards ([`Self::take_output`] returns `None`) until the
/// next successful call, which rewrites every member.
///
/// # Retained state and shared effects
///
/// The handle owns its output stacks and host spectra, reported by
/// [`Self::retained_bytes`] and freed on drop. It reserves nothing in the
/// device context's plan-entry ledger: no step submits a cuTENSOR
/// contraction (the admission and gather are CubeCL kernels, and the
/// materializations and copies are permutations, which Tenferro caches
/// separately).
pub struct PreparedEighFull<R: SectorCodec, D, S = Vec<D>> {
    runtime: Runtime,
    source: StructureSignature,
    space: BoundDynamicFusionMapSpace<R>,
    member_len: usize,
    regions: Arc<[CoupledSectorRegion]>,
    /// The source's coupled sectors in label order, as eager reports them.
    labels: Vec<(SectorId, <R as SectorCodec>::Sector)>,
    /// The factors of the last successful call: the only observable output.
    output: Option<StackPair<R, D, S>>,
    /// Buffers kept for reuse after a failed call. Never observable: a
    /// failed call may have written part of them.
    spare: Option<StackPair<R, D, S>>,
    spectra: Vec<Vec<SectorSpectrum<<R as SectorCodec>::Sector>>>,
    #[cfg(feature = "cuda")]
    device: Option<DeviceEighPlan<R>>,
}

/// `(d, v)` of a prepared eigendecomposition.
type StackPair<R, D, S> = (StackedTensorMap<R, D, S>, StackedTensorMap<R, D, S>);

/// The data-independent part of the device eigendecomposition.
#[cfg(feature = "cuda")]
struct DeviceEighPlan<R> {
    plan: super::TypedCudaEighPlan<R>,
    /// `(offset, n)` of every source coupled sector, for admission.
    admission: Vec<(usize, usize)>,
    /// First eigenvalue of each route in a member's concatenated spectra.
    spectrum_offsets: Vec<usize>,
    /// Eigenvalues per member (`Σ n` over the routes).
    spectrum_len: usize,
    /// Each route's diagonal in `d`: `(offset, step)` within a member.
    diagonals: Vec<(usize, usize)>,
    /// Each route's `(source row, target row, rows)` copies within its
    /// target region: one for a layout-aligned route, one per codomain tree
    /// otherwise.
    copies: Vec<Vec<(usize, usize, usize)>>,
    d_len: usize,
    v_len: usize,
    d_signature: StructureSignature,
    v_signature: StructureSignature,
}

impl<R, D, S> PreparedEighFull<R, D, S>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    /// Prepares the eigendecomposition of stacks with `source`'s signature.
    /// Its payload is not read, and `B` is not fixed.
    ///
    /// Why a stack and not a bare signature: the plan needs the bound space
    /// and its provider, which a signature does not carry.
    ///
    /// # Errors
    ///
    /// The eager structural errors (not an endomorphism, a non-square
    /// coupled-sector block), `UnsupportedTensorContractScope` for a complex
    /// payload or a layout without packed coupled-sector regions.
    pub fn new(source: &StackedTensorMap<R, D, S>) -> Result<Self, Error> {
        if !D::CONJUGATION_IS_IDENTITY {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "a prepared eigh supports real payloads until tenferro-rs#1923",
            }
            .into());
        }
        let _host_pool = source.runtime.enter_host_pool();
        let space = source.space.space();
        if space.homspace().codomain() != space.homspace().domain() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "eigh requires an endomorphism (codomain == domain)",
            }
            .into());
        }
        let regions = space
            .structure()
            .coupled_sector_regions(space.nout())?
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "a prepared eigh needs packed coupled-sector regions",
            })?;
        if regions.iter().any(|region| region.rows() != region.cols()) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "eigh requires square coupled-sector matrices",
            }
            .into());
        }
        tenet_matrixalgebra::validate_endomorphism_region_stacking(
            &regions,
            tenet_matrixalgebra::EIGH_FULL_STACKING,
        )?;
        let provider = source.space.provider();
        let mut labels = regions
            .iter()
            .map(|region| Ok((region.coupled(), provider.decode_sector(region.coupled())?)))
            .collect::<Result<Vec<_>, Error>>()?;
        labels.sort_by(|left, right| left.1.cmp(&right.1));
        #[cfg(feature = "cuda")]
        let device = match source.signature.placement {
            Placement::Cuda(_) => Some(DeviceEighPlan::new(source, &regions)?),
            Placement::Host => None,
        };
        Ok(Self {
            runtime: source.runtime.clone(),
            source: source.signature.clone(),
            space: source.space.clone(),
            member_len: source.member_len,
            regions,
            labels,
            output: None,
            spare: None,
            spectra: Vec::new(),
            #[cfg(feature = "cuda")]
            device,
        })
    }
}

impl<R: SectorCodec, D, S> PreparedEighFull<R, D, S> {
    /// Moves the handle-owned `(d, v)` of the last successful call out; the
    /// next execute allocates new ones. `None` after a failed call: a batch
    /// fails as a whole, so no partially written factor is ever handed out.
    pub fn take_output(&mut self) -> Option<StackPair<R, D, S>> {
        self.output.take()
    }

    /// The output buffers for a call over `members`: the last output or the
    /// spare, if either has that member count. Taking them also makes the
    /// previous output unobservable until this call succeeds.
    fn take_buffers(&mut self, members: usize) -> Option<StackPair<R, D, S>> {
        let buffers = self.output.take().or_else(|| self.spare.take());
        self.spare = None;
        buffers.filter(|(d, _)| d.members == members)
    }

    /// Publishes `buffers` after a successful call, or keeps them as the
    /// unobservable spare after a failed one.
    fn settle<T>(
        &mut self,
        buffers: Option<StackPair<R, D, S>>,
        result: Result<T, BatchError>,
    ) -> Result<T, BatchError> {
        match result {
            Ok(value) => {
                self.output = buffers;
                Ok(value)
            }
            Err(error) => {
                self.spare = buffers;
                Err(error)
            }
        }
    }

    /// Bytes the handle itself retains: the output payloads (host or device)
    /// and the host spectra. The shared plan is not included.
    pub fn retained_bytes(&self) -> usize {
        let payloads = self
            .output
            .iter()
            .chain(&self.spare)
            .map(|(d, v)| {
                (d.members * d.member_len + v.members * v.member_len) * std::mem::size_of::<D>()
            })
            .sum::<usize>();
        let spectra = self
            .spectra
            .iter()
            .flatten()
            .map(|entry| entry.values.capacity() * std::mem::size_of::<f64>())
            .sum::<usize>();
        payloads + spectra
    }

    fn check_source(&self, source: &StackedTensorMap<R, D, S>) -> Result<(), Error> {
        match self.source.first_mismatch(&source.signature) {
            Some(field) => Err(Error::BatchSignatureMismatch {
                member: None,
                field,
            }),
            None => Ok(()),
        }
    }

    fn stack(
        &self,
        space: BoundDynamicFusionMapSpace<R>,
        signature: StructureSignature,
        storage: S,
        members: usize,
        member_len: usize,
    ) -> StackedTensorMap<R, D, S> {
        StackedTensorMap {
            runtime: self.runtime.clone(),
            space,
            signature,
            storage,
            members,
            member_len,
            _payload: PhantomData,
        }
    }

    fn output_ref(&self) -> Result<EighStackOutput<'_, R, D, S>, Error> {
        let (d, v) = self
            .output
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("eigh output is missing".into()))?;
        Ok(EighStackOutput {
            d,
            v,
            spectra: &self.spectra,
        })
    }
}

impl<R, D> PreparedEighFull<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar,
{
    /// Decomposes every member into the handle-owned `(d, v)` and borrows
    /// them with the host spectra.
    ///
    /// Every member is admitted (the eager Hermitian rule, per coupled
    /// sector) before any member is solved; then each member runs the eager
    /// Host eigendecomposition. The outputs are allocated zeroed when absent
    /// or when `B` changes and reused otherwise: `v` is overwritten, and `d`
    /// only on its diagonal, since the handle never writes `d` elsewhere.
    ///
    /// # Errors
    ///
    /// [`BatchError::Operation`] with [`Error::BatchSignatureMismatch`] for a
    /// stack of another signature; [`BatchError::MemberRejected`] listing
    /// every non-Hermitian member (before any solve) or, after the solves,
    /// every member with a non-finite eigenvalue; otherwise the eager
    /// operation's errors as [`BatchError::Operation`].
    pub fn execute(
        &mut self,
        source: &StackedTensorMap<R, D>,
    ) -> Result<EighStackOutput<'_, R, D>, BatchError> {
        let mut buffers = self.take_buffers(source.members);
        let result = self.run_host(source, &mut buffers);
        let spectra = self.settle(buffers, result)?;
        publish_spectra(
            &mut self.spectra,
            &self.labels,
            source.members,
            |member, sector| {
                spectra[member]
                    .binary_search_by_key(&sector, |entry| entry.sector)
                    .map(|index| spectra[member][index].values.as_slice())
                    .map_err(|_| internal_layout_error("a member is missing a coupled sector"))
            },
        )?;
        Ok(self.output_ref()?)
    }

    /// The Host call into `buffers`, allocated zeroed when absent; returns
    /// every member's spectra. `d`'s off-diagonal entries are never written,
    /// so a reused `d` needs only its diagonals.
    #[allow(clippy::type_complexity)]
    fn run_host(
        &self,
        source: &StackedTensorMap<R, D>,
        buffers: &mut Option<StackPair<R, D, Vec<D>>>,
    ) -> Result<Vec<Vec<tenet_matrixalgebra::SectorSpectrum>>, BatchError> {
        self.check_source(source)?;
        let members = source.members;
        let len = self.member_len;
        let runtime = self.runtime.clone();
        let mut dense = runtime.lease_dense();
        let rejected = source
            .storage
            .chunks_exact(len.max(1))
            .take(members)
            .enumerate()
            .filter_map(|(member, data)| {
                match tenet_matrixalgebra::validate_hermitian_regions(data, &self.regions) {
                    Ok(()) => None,
                    // Shapes were admitted by `new`, so this is the content
                    // rule, the only other error the check reports.
                    Err(OperationError::InvalidArgument { .. }) => {
                        Some(Ok((member, MemberFault::NotHermitian)))
                    }
                    Err(other) => Some(Err(Error::from(other))),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !rejected.is_empty() {
            return Err(BatchError::MemberRejected { members: rejected });
        }

        let mut spectra = Vec::with_capacity(members);
        let mut faults = Vec::new();
        let mut diagonals = Vec::new();
        for member in 0..members {
            let data = &source.storage[member * len..(member + 1) * len];
            let input = BoundDynamicTensorRef::try_new(&self.space, data).map_err(Error::from)?;
            let (v, mut eigenvalues) =
                match tenet_matrixalgebra::eigh_full_dyn(dense.dense(), &input) {
                    Ok(out) => out.into_parts(),
                    // The eager rejection of a non-finite eigenvalue; every
                    // other member is still solved so all of them are named.
                    Err(OperationError::InvalidArgument {
                        message: "eigenvalues must be finite",
                    }) => {
                        faults.push((member, MemberFault::NonFiniteEigenvalue));
                        spectra.push(Vec::new());
                        continue;
                    }
                    Err(other) => return Err(Error::from(other).into()),
                };
            // As eager `diagonal_factor`: the bond is built in sector order.
            eigenvalues.sort_unstable_by_key(|entry| entry.sector);
            let (v_space, v_data) = v.into_parts();
            if buffers.is_none() {
                let d_space =
                    tenet_matrixalgebra::diagonal_bond_bound_space_like(&self.space, &eigenvalues)
                        .map_err(Error::from)?;
                let d_len = d_space.space().required_len().map_err(Error::from)?;
                let v_len = v_data.len();
                let signature =
                    |space| StructureSignature::of_space(space, Placement::Host, &runtime);
                *buffers = Some((
                    self.stack(
                        d_space.clone(),
                        signature(&d_space),
                        vec![D::zero(); d_len * members],
                        members,
                        d_len,
                    ),
                    self.stack(
                        v_space.clone(),
                        signature(&v_space),
                        vec![D::zero(); v_len * members],
                        members,
                        v_len,
                    ),
                ));
            }
            let Some((d_stack, v_stack)) = buffers.as_mut() else {
                return Err(internal_layout_error("eigh output stacks are missing").into());
            };
            if v_space.space() != v_stack.space.space() {
                return Err(
                    internal_layout_error("members produced different eigenvector spaces").into(),
                );
            }
            if diagonals.is_empty() {
                diagonals = sector_diagonals(d_stack.space.space().structure())?;
            }
            // Only the diagonal, as eager `diagonal_bond_data` fills it; the
            // rest of `d` is zero from its allocation and never written.
            let d_member = d_stack
                .storage
                .get_mut(member * d_stack.member_len..(member + 1) * d_stack.member_len)
                .ok_or_else(|| internal_layout_error("a member overruns its stack"))?;
            for &(sector, offset, step, count) in &diagonals {
                let Ok(index) = eigenvalues.binary_search_by_key(&sector, |entry| entry.sector)
                else {
                    continue;
                };
                for (position, &value) in eigenvalues[index].values[..count].iter().enumerate() {
                    d_member[offset + position * step] = D::from_real(value);
                }
            }
            copy_member(&mut v_stack.storage, member, &v_data)?;
            spectra.push(eigenvalues);
        }
        drop(dense);
        if !faults.is_empty() {
            return Err(BatchError::MemberRejected { members: faults });
        }
        Ok(spectra)
    }
}

/// Every diagonal fusion-tree block of a `bond <- bond` structure as
/// `(coupled sector, offset, step, count)`, read as eager
/// `diagonal_bond_data` reads it.
fn sector_diagonals(
    structure: &tenet_core::BlockStructure,
) -> Result<Vec<(SectorId, usize, usize, usize)>, Error> {
    let mut diagonals = Vec::with_capacity(structure.block_count());
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let tenet_core::BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let strides = block.strides();
        let shape = block.shape();
        diagonals.push((
            tree.codomain_tree().coupled(),
            block.offset(),
            strides[0] + strides[1],
            shape[0].min(shape[1]),
        ));
    }
    Ok(diagonals)
}

/// Resizes `spectra` to `members` and rewrites each member's coupled
/// sectors, in `labels` order, from `values(member, sector)`, reusing every
/// buffer of the same shape.
fn publish_spectra<'v, S: Clone>(
    spectra: &mut Vec<Vec<SectorSpectrum<S>>>,
    labels: &[(SectorId, S)],
    members: usize,
    mut values: impl FnMut(usize, SectorId) -> Result<&'v [f64], Error>,
) -> Result<(), Error> {
    spectra.resize_with(members, Vec::new);
    for (member, entries) in spectra.iter_mut().enumerate() {
        entries.truncate(labels.len());
        for (position, (sector, label)) in labels.iter().enumerate() {
            let source = values(member, *sector)?;
            match entries.get_mut(position) {
                Some(entry) => {
                    entry.sector.clone_from(label);
                    entry.values.clear();
                    entry.values.extend_from_slice(source);
                }
                None => entries.push(SectorSpectrum {
                    sector: label.clone(),
                    values: source.to_vec(),
                }),
            }
        }
    }
    Ok(())
}

/// Overwrites member `member` of a stacked host buffer with `data`.
fn copy_member<D: Copy>(storage: &mut [D], member: usize, data: &[D]) -> Result<(), Error> {
    storage
        .get_mut(member * data.len()..(member + 1) * data.len())
        .ok_or_else(|| internal_layout_error("a member overruns its stack"))?
        .copy_from_slice(data);
    Ok(())
}

#[cfg(feature = "cuda")]
impl<R> DeviceEighPlan<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    /// Compiles the eager device plan from the structure: `eigh_full` keeps
    /// every eigenpair, so each coupled sector's rank is its full `n`.
    fn new<D, S>(
        source: &StackedTensorMap<R, D, S>,
        regions: &Arc<[CoupledSectorRegion]>,
    ) -> Result<Self, Error> {
        let source_plan = super::compile_cuda_qr_plan(&source.space, Arc::clone(regions))?;
        let ranks: Vec<(SectorId, usize)> = source_plan
            .source_regions
            .iter()
            .filter(|region| region.rows() != 0)
            .map(|region| (region.coupled(), region.rows()))
            .collect();
        let plan = super::compile_cuda_eigh_plan(
            &source.space,
            source_plan.source_regions,
            ranks.iter().copied(),
        )?;
        let admission = plan
            .source_regions
            .iter()
            .map(|region| (region.range().start, region.rows()))
            .collect();
        let mut spectrum_offsets = Vec::with_capacity(plan.routes.len());
        let mut copies = Vec::with_capacity(plan.routes.len());
        let mut spectrum_len = 0usize;
        for route in &plan.routes {
            spectrum_offsets.push(spectrum_len);
            spectrum_len += route.full_rank;
            copies.push(route_copies(&plan, route)?);
        }
        let diagonals = route_diagonals(&plan)?;
        let placement = source.signature.placement;
        let d_signature =
            StructureSignature::of_space(&plan.middle_space, placement, &source.runtime);
        let v_signature =
            StructureSignature::of_space(&plan.left_space, placement, &source.runtime);
        let d_len = plan.middle_space.space().required_len()?;
        let v_len = plan.left_space.space().required_len()?;
        Ok(Self {
            plan,
            admission,
            spectrum_offsets,
            spectrum_len,
            diagonals,
            copies,
            d_len,
            v_len,
            d_signature,
            v_signature,
        })
    }
}

/// A route's `(source row, target row, rows)` copies within its target region,
/// exactly the placements of eager's assembly: the whole region when the
/// plan proved it layout-aligned, else one row slice per codomain tree,
/// matched by tree identity.
#[cfg(feature = "cuda")]
fn route_copies<R>(
    plan: &super::TypedCudaEighPlan<R>,
    route: &super::TypedCudaEighRoute,
) -> Result<Vec<(usize, usize, usize)>, Error> {
    let target = &plan.left_regions[route.left];
    #[cfg(test)]
    let aligned = route.aligned && !tests::FORCE_TREEWISE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let aligned = route.aligned;
    if aligned {
        return Ok(vec![(0, 0, target.rows())]);
    }
    let source = &plan.source_regions[route.source];
    let mut copies = Vec::with_capacity(target.row_trees().len());
    for target_tree in target.row_trees() {
        let rows = target_tree.extent()?;
        if rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees()
            .iter()
            .find(|tree| tree.tree() == target_tree.tree())
            .map(|tree| tree.offset())
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        copies.push((src_row, target_tree.offset(), rows));
    }
    Ok(copies)
}

/// Each route's diagonal in the dense `d` of one member, `(offset, step)`.
#[cfg(feature = "cuda")]
fn route_diagonals<R>(plan: &super::TypedCudaEighPlan<R>) -> Result<Vec<(usize, usize)>, Error> {
    let diagonals = sector_diagonals(plan.middle_space.space().structure())?;
    plan.routes
        .iter()
        .map(|route| {
            let sector = plan.source_regions[route.source].coupled();
            match diagonals.iter().find(|entry| entry.0 == sector) {
                Some(&(_, offset, step, count)) if count == route.kept => Ok((offset, step)),
                _ => Err(internal_layout_error(
                    "CUDA EIGH route has no matching diagonal block",
                )),
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
impl<R, D> PreparedEighFull<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    /// The device form of the Host [`PreparedEighFull::execute`].
    ///
    /// Per call, independent of `B`: at most three admission downloads and
    /// one spectra download; one batched solver call per coupled sector
    /// (each reads its solver status, a host barrier, inside the backend);
    /// one column gather per coupled sector, then one strided copy per
    /// aligned coupled sector (per codomain tree otherwise); one strided copy
    /// per coupled sector onto `d`'s diagonals. Host to device, per call:
    /// the `B · Σ n` sorted eigenvalues (the only payload of `d` that
    /// changes), one `[B · n, 2]` index table per coupled sector and at most
    /// two `[B]` normalizer vectors per coupled sector.
    ///
    /// `d` and `v` are reused across calls at the same `B`. A new `B`, or a
    /// call after [`Self::take_output`], allocates both with one zero upload
    /// each (`B · Σ n²` and `B · L_v` elements, the only device allocation
    /// path until #740); `d`'s off-diagonal entries are never written.
    ///
    /// # Errors
    ///
    /// As the Host form, with non-Hermitian members found by the batched
    /// admission before any solver launch, and [`BatchError::Solver`] when
    /// the solver reports a numerical failure on a block. Argument, bounds
    /// and backend failures are [`BatchError::Operation`].
    pub fn execute(
        &mut self,
        source: &StackedTensorMap<R, D, CudaStorage<D>>,
    ) -> Result<EighStackOutput<'_, R, D, CudaStorage<D>>, BatchError> {
        let mut buffers = self.take_buffers(source.members);
        let result = self.run_cuda(source, &mut buffers);
        let sorted = self.settle(buffers, result)?;
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| internal_layout_error("a device eigh handle has no device plan"))?;
        let total = device.spectrum_len;
        publish_spectra(
            &mut self.spectra,
            &self.labels,
            source.members,
            |member, sector| {
                device
                    .plan
                    .routes
                    .iter()
                    .zip(&device.spectrum_offsets)
                    .find(|(route, _)| device.plan.source_regions[route.source].coupled() == sector)
                    .map(|(route, &offset)| &sorted[member * total + offset..][..route.kept])
                    .ok_or_else(|| internal_layout_error("a coupled sector has no route"))
            },
        )?;
        Ok(self.output_ref()?)
    }

    /// The device call into `buffers`, allocated zeroed when absent; returns
    /// every member's sorted eigenvalues, member-major (`Σ n` per member).
    fn run_cuda(
        &self,
        source: &StackedTensorMap<R, D, CudaStorage<D>>,
        buffers: &mut Option<StackPair<R, D, CudaStorage<D>>>,
    ) -> Result<Vec<f64>, BatchError> {
        self.check_source(source)?;
        let members = source.members;
        let runtime = self.runtime.clone();
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| internal_layout_error("a device eigh handle has no device plan"))?;
        let mut lease = runtime.lease_cuda()?;
        let cuda = &mut *lease;
        let routes = &device.plan.routes;

        let accepted = tenet_dense::cuda_hermitian_regions_batched::<D>(
            cuda,
            &source.storage.0,
            &device.admission,
            members,
            self.member_len,
        )
        .map_err(dense_err)?;
        let rejected: Vec<_> = accepted
            .chunks(device.admission.len().max(1))
            .enumerate()
            .filter(|(_, member)| member.contains(&false))
            .map(|(member, _)| (member, MemberFault::NotHermitian))
            .collect();
        if !rejected.is_empty() {
            return Err(BatchError::MemberRejected { members: rejected });
        }

        let mut device_spectra = Vec::with_capacity(routes.len());
        let mut vectors = Vec::with_capacity(routes.len());
        for route in routes {
            let region = &device.plan.source_regions[route.source];
            let (values, vector) = tenet_dense::cuda_eigh_region_batched::<D>(
                cuda,
                &source.storage.0,
                region.range().start,
                route.full_rank,
                members,
                self.member_len,
            )
            .map_err(|error| match error {
                // Only the solver's own status names a block; argument,
                // bounds and backend failures are the operation's.
                error @ tenet_dense::DenseError::NumericalFailure { .. } => BatchError::Solver {
                    block: route.source,
                    source: dense_err(error),
                },
                other => BatchError::Operation(dense_err(other)),
            })?;
            device_spectra.push(values);
            vectors.push(vector);
        }
        let raw = tenet_dense::cuda_download_batched_spectra::<D>(cuda, &device_spectra, members)
            .map_err(dense_err)?;
        drop(device_spectra);

        // Per member and route, eager's order: descending |λ|, index
        // tie-break, on the solver's own (ascending) order.
        let total = device.spectrum_len;
        let mut sorted = vec![0.0_f64; total * members];
        let mut orders = vec![0usize; total * members];
        let mut faults = Vec::new();
        for member in 0..members {
            let mut finite = true;
            for (route, &offset) in routes.iter().zip(&device.spectrum_offsets) {
                let n = route.full_rank;
                let base = member * total + offset;
                let values = &raw[base..base + n];
                finite &= values.iter().all(|value| value.is_finite());
                let order = &mut orders[base..base + n];
                for (index, slot) in order.iter_mut().enumerate() {
                    *slot = index;
                }
                order.sort_by(|&left, &right| {
                    values[right]
                        .abs()
                        .total_cmp(&values[left].abs())
                        .then(left.cmp(&right))
                });
                for (slot, &index) in sorted[base..base + n].iter_mut().zip(order.iter()) {
                    *slot = values[index];
                }
            }
            if !finite {
                faults.push((member, MemberFault::NonFiniteEigenvalue));
            }
        }
        if !faults.is_empty() {
            return Err(BatchError::MemberRejected { members: faults });
        }

        // `d` and `v` are allocated zeroed once per `B` (the only device
        // allocation path, #740). The handle never writes `d` off its
        // diagonal, so each call moves only the `B · Σ n` sorted values.
        if buffers.is_none() {
            let zeros = |len: usize| {
                CudaStorage::<D>::upload_members(cuda, vec![D::zero(); len * members], len, members)
                    .map_err(Error::from)
            };
            *buffers = Some((
                self.stack(
                    device.plan.middle_space.clone(),
                    device.d_signature.clone(),
                    zeros(device.d_len)?,
                    members,
                    device.d_len,
                ),
                self.stack(
                    device.plan.left_space.clone(),
                    device.v_signature.clone(),
                    zeros(device.v_len)?,
                    members,
                    device.v_len,
                ),
            ));
        }
        let Some((d, v)) = buffers.as_mut() else {
            return Err(internal_layout_error("eigh output stacks are missing").into());
        };
        let values = CudaStorage::<D>::upload_owned(
            cuda,
            sorted.iter().map(|&value| D::from_real(value)).collect(),
        )
        .map_err(Error::from)?;
        for ((&(offset, step), &spectrum), route) in device
            .diagonals
            .iter()
            .zip(&device.spectrum_offsets)
            .zip(routes)
        {
            let region = |dims: Vec<usize>, strides: Vec<usize>, offset: usize| {
                tenet_dense::CudaRegion::new(dims, strides, offset).map_err(dense_err)
            };
            tenet_dense::cuda_copy_strided_into::<D>(
                cuda,
                &values.0,
                &region(vec![route.kept, members], vec![1, total], spectrum)?,
                &mut d.storage.0,
                &region(vec![route.kept, members], vec![step, device.d_len], offset)?,
            )
            .map_err(dense_err)?;
        }
        for (((route, raw_vectors), &offset), copies) in routes
            .iter()
            .zip(&vectors)
            .zip(&device.spectrum_offsets)
            .zip(&device.copies)
        {
            let target = &device.plan.left_regions[route.left];
            let columns: Vec<usize> = (0..members)
                .flat_map(|member| {
                    orders[member * total + offset..][..route.kept]
                        .iter()
                        .copied()
                })
                .collect();
            tenet_dense::cuda_gather_columns_batched_into::<D>(
                cuda,
                &mut v.storage.0,
                target.range().start,
                target.rows(),
                device.v_len,
                raw_vectors,
                route.full_rank,
                members,
                &columns,
                copies,
            )
            .map_err(dense_err)?;
        }
        Ok(sorted)
    }
}

/// The owned dense payload of `tensor`, or the representation a stack
/// cannot hold.
fn dense_payload<R, D>(tensor: &TensorMap<R, D>) -> Result<&[D], BatchMemberRepresentation> {
    match &tensor.repr {
        TypedTensorRepr::Adjoint(_) => Err(BatchMemberRepresentation::LazyAdjoint),
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(data) => Ok(data),
            TypedData::Diagonal(_) => Err(BatchMemberRepresentation::CompactDiagonal),
        },
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> StackedTensorMap<R, D> {
    /// Uploads the whole stack with one host-to-device transfer to this
    /// stack's Runtime CUDA context.
    pub fn to_cuda(&self) -> Result<StackedTensorMap<R, D, CudaStorage<D>>, Error> {
        let lease = self.runtime.lease_cuda()?;
        let storage = CudaStorage::upload_members(
            &lease,
            self.storage.clone(),
            self.member_len,
            self.members,
        )?;
        Ok(self.with_storage(storage, self.members))
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> StackedTensorMap<R, D, CudaStorage<D>> {
    /// Downloads the whole stack with one device-to-host transfer.
    pub fn to_host(&self) -> Result<StackedTensorMap<R, D>, Error> {
        let lease = self.runtime.lease_cuda()?;
        let storage = self.storage.download(&lease)?;
        Ok(self.with_storage(storage, self.members))
    }

    /// The device form of the Host [`StackedTensorMap::restrict_leg`], with
    /// the same result space and errors: one upload of an `L'`-entry element
    /// table and one Tenferro `gather` launch per call, whatever `B` and the
    /// block count. Nothing is downloaded and nothing waits on the device.
    ///
    /// The table maps each destination element to its source element within
    /// a member. It is the eager restriction kernel applied once, on the Host,
    /// to the payload `0, 1, …, L - 1`, so the block plan has one
    /// implementation; the gather then reads that element of every member
    /// (its window is the member axis) and allocates the `[L', B]` result.
    pub fn restrict_leg(
        &self,
        axis: usize,
        selection: &LegSelection<R>,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        let (destination, table) = self.restrict_plan(axis, selection)?;
        // The element table is the eager kernel itself run over the payload
        // `0, 1, …, L - 1` (in `i64`, the gather's index type), so the block
        // plan keeps one implementation. Why not enumerate the destination
        // blocks here instead: that would restate the sector lookup, start
        // offsets and rectangle checks of `restrict_block`. The price is one
        // `L`-element iota per call, host work `O(L + L')`, independent of `B`.
        let elements = {
            let _host_pool = self.runtime.enter_host_pool();
            let mut starts: Vec<tenet_tensors::SectorStartTable<'_>> = vec![None; self.rank()];
            starts[axis] = Some(table.as_slice());
            let iota = (0..self.member_len)
                .map(i64::try_from)
                .collect::<Result<Vec<i64>, _>>()
                .map_err(|_| Error::InvalidArgument("restrict_leg: a member exceeds i64".into()))?;
            tenet_tensors::oriented_fusion_restrict_owned(
                destination.space().structure(),
                FusionOperand::direct(self.space.space()),
                &iota,
                &starts,
            )
            .map_err(Error::from)?
        };
        let mut lease = self.runtime.lease_cuda()?;
        let storage = self
            .storage
            .gather_member_elements(&mut lease, self.member_len, self.members, elements)
            .map_err(Error::from)?;
        drop(lease);
        Ok(self.with_space(destination, storage)?)
    }

    /// The device form of the Host [`StackedTensorMap::select`], with the
    /// same index policy: one upload of the `members.len()` member offsets
    /// and one Tenferro `gather` launch per call, whatever `B` and the block
    /// structure. The gather allocates the result buffer. Nothing is
    /// downloaded and nothing waits on the device.
    pub fn select(&self, members: &[usize]) -> Result<Self, Error> {
        self.selection_len(members)?;
        let mut lease = self.runtime.lease_cuda()?;
        let storage =
            self.storage
                .gather_members(&mut lease, self.member_len, self.members, members)?;
        Ok(self.with_storage(storage, members.len()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use std::sync::Arc;

    use tenet_core::{Placement, RuleIdentity, TensorStorage, U1FusionRule, U1Irrep};

    use super::super::Runtime;
    use super::super::{owned_repr, GradedSpace, TensorMap, TypedTensorBody};

    thread_local! {
        /// Forces every device eigh route onto the per-tree copies, so the
        /// branch the fixtures' aligned routes never take is exercised.
        pub(super) static FORCE_TREEWISE: std::cell::Cell<bool> = const {
            std::cell::Cell::new(false)
        };
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn tree_wise_eigenvector_copies_equal_the_aligned_ones_bitwise() {
        // What: the per-tree placement (one copy per codomain tree) writes
        // exactly the bits of the aligned placement (one copy per sector),
        // and both are eager's device `v` at B = 1.
        use super::PreparedEighFull;
        use tenet_core::{SU2FusionRule, SU2Irrep};

        let runtime = Runtime::builder().cuda(0).build().unwrap();
        let j = SU2Irrep::from_twice_spin;
        let leg = GradedSpace::try_new(SU2FusionRule, [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap();
        let members: Vec<_> = (0..3)
            .map(|seed| {
                let x =
                    TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], seed)
                        .unwrap();
                x.axpby(1.0, &x.adjoint().unwrap(), 1.0).unwrap()
            })
            .collect();
        let stack = super::StackedTensorMap::pack(&members)
            .unwrap()
            .to_cuda()
            .unwrap();
        let run = |treewise: bool| {
            FORCE_TREEWISE.with(|flag| flag.set(treewise));
            let mut handle = PreparedEighFull::new(&stack).unwrap();
            FORCE_TREEWISE.with(|flag| flag.set(false));
            let copies: usize = handle
                .device
                .as_ref()
                .unwrap()
                .copies
                .iter()
                .map(Vec::len)
                .sum();
            let routes = handle.device.as_ref().unwrap().copies.len();
            let output = handle.execute(&stack).unwrap();
            (
                copies,
                routes,
                output.v.to_host().unwrap(),
                output.d.to_host().unwrap(),
            )
        };
        let (aligned_copies, routes, aligned_v, aligned_d) = run(false);
        let (tree_copies, _, tree_v, tree_d) = run(true);
        assert_eq!(aligned_copies, routes, "every fixture route is aligned");
        assert!(tree_copies > routes, "several codomain trees per sector");
        assert!(tree_v.storage == aligned_v.storage);
        assert!(tree_d.storage == aligned_d.storage);
        let single = super::StackedTensorMap::pack(&members[..1])
            .unwrap()
            .to_cuda()
            .unwrap();
        FORCE_TREEWISE.with(|flag| flag.set(true));
        let mut handle = PreparedEighFull::new(&single).unwrap();
        FORCE_TREEWISE.with(|flag| flag.set(false));
        let v = handle
            .execute(&single)
            .unwrap()
            .v
            .to_host()
            .unwrap()
            .member(0)
            .unwrap();
        let crate::typed::Eigh { v: eager_v, .. } =
            members[0].to_cuda().unwrap().eigh_full().unwrap();
        assert!(v.data() == eager_v.to_host().unwrap().data());
    }

    struct PlacedStorage {
        len: usize,
        placement: Placement,
    }

    impl TensorStorage<f64> for PlacedStorage {
        fn len(&self) -> usize {
            self.len
        }

        fn placement(&self) -> Placement {
            self.placement
        }
    }

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn placement_alone_separates_signatures() {
        // What: the same space and Runtime on different placements (Host,
        // two device ordinals) gives unequal signatures; equal placement
        // gives equal signatures and equal hashes.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(
            U1FusionRule,
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 3),
            ],
        )
        .unwrap();
        let host = TensorMap::<_, f64>::zeros(&runtime, [&leg, &leg], [&leg]).unwrap();
        let placed = |placement| TensorMap::<_, f64, PlacedStorage> {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(
                host.logical_space().clone(),
                PlacedStorage {
                    len: host.data().len(),
                    placement,
                },
            )),
        };
        let host_signature = host.structure_signature();
        let cuda0 = placed(Placement::Cuda(0)).structure_signature();
        let cuda1 = placed(Placement::Cuda(1)).structure_signature();
        let host_again = placed(Placement::Host).structure_signature();

        assert!(host_signature != cuda0);
        assert!(cuda0 != cuda1);
        assert!(host_signature == host_again);
        assert_eq!(hash_of(&host_signature), hash_of(&host_again));
        assert!(cuda0 == placed(Placement::Cuda(0)).structure_signature());
    }

    #[test]
    fn first_mismatch_names_each_field_alone() {
        // What: replacing one field of an equal signature with another
        // signature's value is reported as exactly that field, and `==`
        // agrees with `first_mismatch`.
        use super::SignatureField;

        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let q = U1Irrep::new;
        let leg = GradedSpace::try_new(U1FusionRule, [(q(0), 2), (q(1), 1)]).unwrap();
        let wide = GradedSpace::try_new(U1FusionRule, [(q(0), 3), (q(1), 1)]).unwrap();
        let signature = |runtime: &Runtime, leg: &GradedSpace<U1FusionRule>| {
            TensorMap::<_, f64>::zeros(runtime, [leg, leg], [leg])
                .unwrap()
                .structure_signature()
        };
        let base = signature(&runtime, &leg);
        let other = signature(&other_runtime, &wide);
        assert_eq!(base.first_mismatch(&base.clone()), None);

        let mut cases = Vec::new();
        let mut changed = base.clone();
        changed.placement = Placement::Cuda(0);
        cases.push((changed, SignatureField::Placement));
        let mut changed = base.clone();
        changed.runtime = other.runtime.clone();
        cases.push((changed, SignatureField::Runtime));
        let mut changed = base.clone();
        changed.rule = RuleIdentity::from_canonical_bytes::<u8>(1, Arc::from([]));
        cases.push((changed, SignatureField::Rule));
        let mut changed = base.clone();
        changed.homspace = other.homspace.clone();
        cases.push((changed, SignatureField::HomSpace));
        let mut changed = base.clone();
        changed.structure = other.structure.clone();
        cases.push((changed, SignatureField::BlockStructure));
        for (changed, field) in cases {
            assert_eq!(base.first_mismatch(&changed), Some(field));
            assert!(base != changed, "{field:?}");
        }
    }

    #[cfg(feature = "racah-generated")]
    #[test]
    fn checked_generic_rule_instance_alone_separates_signatures() {
        // What: SU(3) and SU(4) trivial legs give the same hom space and
        // block structure content, so the rule identity alone separates them.
        use super::super::SUNFusionRule;
        use std::sync::Arc;

        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let signature = |rank: usize, trivial: Vec<i64>| {
            let rule = Arc::new(SUNFusionRule::new(rank).unwrap());
            let leg = GradedSpace::try_new_with_arc(rule, [(trivial, 2)]).unwrap();
            TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg])
                .unwrap()
                .structure_signature()
        };
        let su3 = signature(3, vec![0, 0]);
        let su4 = signature(4, vec![0, 0, 0]);
        assert!(su3.homspace == su4.homspace);
        assert!(*su3.structure == *su4.structure);
        assert!(su3.placement == su4.placement && su3.runtime == su4.runtime);
        assert!(su3.rule != su4.rule);
        assert!(su3 != su4);
        assert!(su3 == signature(3, vec![0, 0]));
    }
}
