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
    BlockStructureContent, CheckedFusionAlgebra, HomSpaceId, MultiplicityFreeRigidSymbols,
    Placement, RuleIdentity, SectorCodec, TensorStorage,
};
use tenet_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use tenet_operations::FusionBlockContractPlan;
use tenet_tensors::{zeroed_payload, FusionOperand};

use super::{
    owned_repr, BoundDynamicFusionMapSpace, Runtime, TensorMap, TensorScalar, TypedData,
    TypedSectorAdmission, TypedTensorBody, TypedTensorRepr,
};
#[cfg(feature = "cuda")]
use super::{CudaPayload, CudaStorage};
use crate::error::Error;
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

    /// The member count `B`. Never zero: [`Self::pack`] rejects an empty
    /// batch.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.members
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
    pub fn member(&self, i: usize) -> Result<TensorMap<R, D>, Error> {
        if i >= self.members {
            return Err(Error::InvalidArgument(format!(
                "member {i} is out of range for a stack of {}",
                self.members
            )));
        }
        let start = i * self.member_len;
        let data = self.storage[start..start + self.member_len].to_vec();
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.space.clone(), data)),
        })
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
        let gemms = self
            .plan
            .direct_batch()
            .iter()
            .map(|job| (job.rows, job.contracted, job.cols))
            .collect::<std::collections::HashSet<_>>()
            .len();
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
    /// The output is allocated zeroed when absent or when `B` changes, and
    /// warm calls allocate nothing: the destination blocks the plan never
    /// writes stay zero, and every other block is overwritten.
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
                let storage = CudaStorage::upload_owned(&lease, zeroed_payload(len))?;
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
        let storage = CudaStorage::upload(&lease, &self.storage)?;
        Ok(self.with_storage(storage))
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> StackedTensorMap<R, D, CudaStorage<D>> {
    /// Downloads the whole stack with one device-to-host transfer.
    pub fn to_host(&self) -> Result<StackedTensorMap<R, D>, Error> {
        let lease = self.runtime.lease_cuda()?;
        let storage = self.storage.download(&lease)?;
        Ok(self.with_storage(storage))
    }
}

#[cfg(feature = "cuda")]
impl<R, D, S> StackedTensorMap<R, D, S> {
    fn with_storage<T: TensorStorage<D>>(&self, storage: T) -> StackedTensorMap<R, D, T> {
        StackedTensorMap {
            runtime: self.runtime.clone(),
            space: self.space.clone(),
            signature: StructureSignature {
                placement: storage.placement(),
                ..self.signature.clone()
            },
            storage,
            members: self.members,
            member_len: self.member_len,
            _payload: PhantomData,
        }
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
