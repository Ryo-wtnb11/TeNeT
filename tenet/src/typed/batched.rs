//! Structure signatures: the key under which tensors can share one prepared
//! plan and one stacked payload layout (#1287).
//!
//! Neither TensorKit nor QSpace has a batched path, so the signature is
//! defined by what TeNeT's eager operations read from an operand: the rule
//! authority, the logical hom space, the fusion-tree block structure, the
//! payload placement and the Runtime. The payload dtype is the type parameter
//! `D` of every consumer and is therefore checked by the compiler, not here.
//!
//! # Failure rule of every batched plan
//!
//! [`ComposePlan`], [`ContractPlan`] and [`EighFullPlan`] (Host and CUDA)
//! share one rule. A workspace created by another plan is rejected before it
//! is touched. Otherwise a failed `execute` or `execute_into` (a validation
//! error, a signature mismatch or a backend error) leaves no observable
//! workspace output: `take_output` returns `None` until the next successful
//! `execute`, which rewrites every member. The failed call's buffers stay in
//! the workspace and are reused, so the next call allocates no more than a
//! warm one. After a failed `execute_into` the caller's destination contents
//! are unspecified; a validation error never writes them.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use tenet_core::{
    BlockStructureContent, CheckedFusionAlgebra, CoupledSectorRegion, HomSpaceId,
    MultiplicityFreeRigidSymbols, Placement, RuleIdentity, SectorCodec, SectorId, TensorStorage,
};
use tenet_matrixalgebra::HermitianTol;
#[cfg(test)]
use tenet_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use tenet_tensors::{zeroed_payload, FusionOperand, OperationError};

#[cfg(feature = "cuda")]
use super::{dense_err, CudaFactorizationPayload, CudaPayload, CudaStorage};
use super::{
    owned_repr, require_restriction_set, restricted_space, restriction_runs,
    BoundDynamicFusionMapSpace, FactorizationScalar, LegSelection, Runtime, SectorSpectrum,
    TensorMap, TensorScalar, TypedData, TypedFacadeError, TypedSectorAdmission, TypedTensorBody,
    TypedTensorRepr, TypedTensorRootDispatch,
};
use crate::error::Error;
use crate::runtime::RuntimeIdentity;
use crate::tensor_core::internal_layout_error;

#[path = "contract_batch.rs"]
mod contract_batch;
pub use contract_batch::{ComposePlan, ContractPlan, ContractWorkspace};

/// Content identity of a tensor's structure, placement and Runtime.
///
/// Two signatures are equal exactly when the fusion-rule identity, the hom
/// space (every codomain and domain leg's sectors, degeneracies and dual
/// flags), the block structure content (fusion-tree blocks, their order and
/// dense offsets), the placement and the Runtime are all equal. Equality
/// never depends on process-local intern ids, so it survives interner
/// eviction, `tenet::cache::clear` and oversized structures that bypass
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
/// A stack is an operand of a batched operation, not a tensor. It
/// deliberately does not implement `TensorStorage`, whose `len` would be
/// `B * L` and would let an ordinary per-tensor kernel run on member 0 only:
///
/// ```compile_fail
/// use tenet::expert::TensorStorage;
/// use tenet::sector::U1FusionRule;
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
/// use tenet::expert::TensorStorage;
/// use tenet::sector::U1FusionRule;
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
    /// Lazy adjoints are not packed. Materialize one before passing it to the
    /// identity-oriented compose plan; oriented batching is outside this API.
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
    /// Restricts the listed legs of every member: member `i` of the result
    /// is `self.member(i)?.restrict_leg(legs)?`, bit for bit, with the same
    /// space and signature.
    ///
    /// One set of selections serves every member, which is what grouping
    /// members by their truncated signature (then [`Self::select`])
    /// produces.
    ///
    /// # Cost
    ///
    /// The eager block plan is built once; each destination block is then one
    /// strided copy for all `B` members and all restricted legs, the member
    /// axis being one more dimension of the copy. `B · L'` elements move into
    /// one output allocation, left unfilled when the blocks tile it (as
    /// eager), plus `O(rank + blocks)` structural work independent of `B`.
    ///
    /// # Errors
    ///
    /// Exactly the eager [`TensorMap::restrict_leg`] errors for a dense
    /// payload: an empty set, an out-of-range or repeated axis, or a leg
    /// other than [`LegSelection::parent`] is an [`Error::InvalidArgument`],
    /// a selection of another rule is [`Error::RuleMismatch`]. Nothing is
    /// allocated before every check.
    pub fn restrict_leg(
        &self,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R::Mode: TypedTensorRootDispatch<R>,
    {
        require_restriction_set(&self.space, legs)?;
        let destination = restricted_space(&self.space, legs)?;
        let _host_pool = self.runtime.enter_host_pool();
        let data = tenet_tensors::stacked_fusion_restrict_owned(
            destination.space().structure(),
            FusionOperand::direct(self.space.space()),
            &self.storage,
            self.members,
            &restriction_runs(self.rank(), legs),
        )
        .map_err(Error::from)?;
        Ok(self.with_space(destination, data)?)
    }
}

/// The workspace-owned output of a batched plan under the module's failure
/// rule: at most one of `output` and `spare` is set.
struct OutputSlot<T> {
    /// The output of the last successful call: the only observable one.
    output: Option<T>,
    /// Buffers kept for reuse after a failed call. Never observable: a failed
    /// call may have written part of them.
    spare: Option<T>,
}

impl<T> Default for OutputSlot<T> {
    fn default() -> Self {
        Self {
            output: None,
            spare: None,
        }
    }
}

impl<T> OutputSlot<T> {
    fn take_output(&mut self) -> Option<T> {
        self.output.take()
    }

    /// Takes the buffers for a call; the previous output stays unobservable
    /// until the call publishes.
    fn take(&mut self) -> Option<T> {
        let spare = self.spare.take();
        self.output.take().or(spare)
    }

    /// Keeps `buffers` as the unobservable spare and returns `error`.
    fn fail<E>(&mut self, buffers: Option<T>, error: E) -> E {
        self.spare = buffers;
        error
    }

    /// Hides the output after a failed call that did not take the buffers.
    fn hide<E>(&mut self, error: E) -> E {
        let buffers = self.take();
        self.fail(buffers, error)
    }

    /// Makes `buffers` the observable output of a successful call.
    fn publish(&mut self, buffers: T) -> &T {
        self.output.insert(buffers)
    }

    /// Publishes `buffers` after a successful call, or keeps them as the spare.
    fn settle<E>(&mut self, buffers: T, result: Result<(), E>) -> Result<&T, E> {
        match result {
            Ok(()) => Ok(self.publish(buffers)),
            Err(error) => Err(self.fail(Some(buffers), error)),
        }
    }

    /// The output and the spare, for retained-byte accounting.
    fn buffers(&self) -> impl Iterator<Item = &T> {
        self.output.iter().chain(&self.spare)
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

/// The factors of one [`EighFullPlan::execute`], borrowed from its workspace.
///
/// Member `i` of `d` and `v` is the eager `eigh_full` of the same placement
/// on member `i` of the source; `spectra[i]` is that member's eigenvalues per
/// coupled sector, in sector-label order, each ascending as in `d`.
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
/// within that operation's gauge contract. Both placements compile the eager
/// plan (coupled-sector routes, the eigenvector and diagonal factor spaces,
/// each sector's eigenvector placement and `d` diagonal) once here from the
/// structure alone, since `eigh_full` keeps every eigenpair:
///
/// - **CUDA**: each call admits the whole batch (at most three downloads),
///   solves each coupled sector of all members with one batched solver
///   call, downloads every spectrum
///   once, orders each member ascending on the host as eager does (an O(n)
///   check: the solver already returns ascending values),
///   and assembles `v` by one batched column gather per coupled sector
///   followed by one strided copy per aligned sector (per codomain tree
///   otherwise). Submissions, host syncs and
///   downloads per call do not depend on the member count `B`. The raw
///   cuSOLVER gauge is kept, as eager keeps it; with `B > 1` the batched
///   solver may differ from the eager one in the last ULPs, so results are
///   deterministic for a fixed `B` but not bit-stable across `B`. At `B = 1`
///   the solver and host code are the eager ones.
/// - **Host**: after the whole batch is admitted, each member runs eager's
///   per-sector solver stage, gauge-fixed exactly as Host eager (the
///   largest-magnitude entry of each eigenvector real and positive), and is
///   written straight into its slice of the output stacks; results are bit
///   for bit eager's at every `B`.
///
/// # Supported scope
///
/// Endomorphisms with packed coupled-sector regions (every structure a
/// public constructor builds). A batch fails as a whole
/// ([`BatchError`]): on any error no output is returned, and none is
/// observable afterwards ([`EighFullWorkspace::take_output`] returns `None`)
/// until the next successful call, which rewrites every member.
///
/// # Retained state and shared effects
///
/// Each workspace owns its output stacks and host spectra, reported by
/// [`EighFullWorkspace::retained_bytes`] and freed on drop. It reserves
/// nothing in the
/// device context's plan-entry ledger: no step submits a cuTENSOR
/// contraction (the admission and gather are CubeCL kernels, and the
/// materializations and copies are permutations, which Tenferro caches
/// separately).
///
/// ```
/// use std::sync::Arc;
/// use tenet::sector::{U1FusionRule, U1Irrep};
/// use tenet::typed::{
///     EighFullPlan, GradedSpace, HermitianTol, Runtime, StackedTensorMap, TensorMap,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = Runtime::builder().build()?;
/// let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
/// let stack = StackedTensorMap::pack(&[TensorMap::<_, f64>::zeros(
///     &runtime, [&leg], [&leg],
/// )?])?;
/// let plan = EighFullPlan::new(&stack, &[0], &[1], HermitianTol::DEFAULT)?;
/// let mut workspace = plan.workspace()?;
/// let _output = plan.execute(&stack, &mut workspace)?;
/// # Ok(())
/// # }
/// ```
pub struct EighFullPlan<R: SectorCodec, D, S = Vec<D>> {
    identity: Arc<()>,
    runtime: Runtime,
    source: StructureSignature,
    member_len: usize,
    /// The source's coupled sectors in label order, as eager reports them.
    labels: Vec<(SectorId, <R as SectorCodec>::Sector)>,
    hermitian_tol: HermitianTol,
    routes: EighRoutePlan<R>,
    _payload: PhantomData<(D, S)>,
}

/// Caller-owned mutable state for repeated execution of an [`EighFullPlan`].
pub struct EighFullWorkspace<R: SectorCodec, D, S = Vec<D>> {
    binding: Arc<()>,
    output: OutputSlot<StackPair<R, D, S>>,
    spectra: Vec<Vec<SectorSpectrum<<R as SectorCodec>::Sector>>>,
    /// Every member's ascending eigenvalues, member-major (`Σ n` each).
    sorted: Vec<f64>,
    /// The solver-to-ascending column order of `sorted`'s entries.
    #[cfg(feature = "cuda")]
    orders: Vec<usize>,
}

/// `(d, v)` of a prepared eigendecomposition.
type StackPair<R, D, S> = (StackedTensorMap<R, D, S>, StackedTensorMap<R, D, S>);

/// The data-independent part of the eigendecomposition, compiled once from
/// the structure for both placements (`eigh_full` keeps every eigenpair, so
/// nothing depends on the payload).
struct EighRoutePlan<R> {
    /// The source and eigenvector regions of the compact factor plan, all
    /// EIGH reads of it besides the routes.
    source_regions: Arc<[CoupledSectorRegion]>,
    /// The plan's executed (nonzero-rank) routes; every other per-route
    /// table below follows their order.
    routes: Vec<tenet_matrixalgebra::seam::CompactFactorRoute>,
    left_space: tenet_tensors::BoundDynamicFusionMapSpace<R>,
    middle_space: tenet_tensors::BoundDynamicFusionMapSpace<R>,
    /// `(offset, n)` of every source coupled sector, for admission.
    #[cfg(feature = "cuda")]
    admission: Vec<(usize, usize)>,
    /// First eigenvalue of each route in a member's concatenated spectra.
    spectrum_offsets: Vec<usize>,
    /// For each of the plan's `labels`, its route; `None` for an empty
    /// sector, whose spectrum is empty as in eager.
    label_routes: Vec<Option<usize>>,
    /// Each route's eigenvector region within a member: its range and rows.
    targets: Vec<(std::ops::Range<usize>, usize)>,
    /// The largest coupled-sector order, sizing the Host reorder scratch.
    max_n: usize,
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

impl<R, D, S> EighFullPlan<R, D, S>
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
    /// `rows`, `cols` and `hermitian_tol` are the leg roles and the
    /// Hermiticity admission of the eager [`TensorMap::eigh_full`]; the plan
    /// fixes the tolerance for every execution. A stack has no batched
    /// permute, so only the
    /// members' current split (`rows = 0..nout`, `cols = nout..rank`) is
    /// supported here.
    ///
    /// # Errors
    ///
    /// The eager structural errors (not an endomorphism, a non-square
    /// coupled-sector block), `UnsupportedTensorContractScope` for a complex
    /// payload, a layout without packed coupled-sector regions, or leg roles
    /// other than the current split.
    pub fn new(
        source: &StackedTensorMap<R, D, S>,
        rows: &[usize],
        cols: &[usize],
        hermitian_tol: HermitianTol,
    ) -> Result<Self, Error> {
        let nout = source.space.space().nout();
        if !rows.iter().copied().eq(0..nout)
            || !cols.iter().copied().eq(nout..source.space.space().rank())
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "a prepared eigh supports only the members' current split; \
                          permute the members before pack",
            }
            .into());
        }
        if !D::CONJUGATION_IS_IDENTITY {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "a prepared eigh supports real payloads until tenferro-rs#1923",
            }
            .into());
        }
        let _host_pool = source.runtime.enter_host_pool();
        let space = source.space.space();
        if space.homspace().codomain() != space.homspace().domain() {
            return Err(OperationError::SpaceMismatch {
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
            return Err(OperationError::SpaceMismatch {
                message: "eigh requires square coupled-sector matrices",
            }
            .into());
        }
        tenet_matrixalgebra::seam::validate_endomorphism_region_stacking(
            &regions,
            tenet_matrixalgebra::seam::EIGH_FULL_STACKING,
        )?;
        let provider = source.space.provider();
        let mut labels = regions
            .iter()
            .map(|region| Ok((region.coupled(), provider.decode_sector(region.coupled())?)))
            .collect::<Result<Vec<_>, Error>>()?;
        labels.sort_by(|left, right| left.1.cmp(&right.1));
        let routes = EighRoutePlan::new(source, &labels)?;
        Ok(Self {
            identity: Arc::new(()),
            runtime: source.runtime.clone(),
            source: source.signature.clone(),
            member_len: source.member_len,
            labels,
            hermitian_tol,
            routes,
            _payload: PhantomData,
        })
    }

    /// Creates independent mutable execution state for this plan.
    pub fn workspace(&self) -> Result<EighFullWorkspace<R, D, S>, Error> {
        Ok(EighFullWorkspace {
            binding: Arc::clone(&self.identity),
            output: OutputSlot::default(),
            spectra: Vec::new(),
            sorted: Vec::new(),
            #[cfg(feature = "cuda")]
            orders: Vec::new(),
        })
    }
}

impl<R: SectorCodec, D, S> EighFullWorkspace<R, D, S> {
    /// Moves the workspace-owned `(d, v)` of the last successful call out; the
    /// next execute allocates new ones. `None` after a failed call: a batch
    /// fails as a whole, so no partially written factor is ever handed out.
    pub fn take_output(&mut self) -> Option<StackPair<R, D, S>> {
        self.output.take_output()
    }

    /// The output buffers for a call over `members`, taken after the source
    /// is admitted so a rejected call keeps buffers of any member count.
    fn take_buffers(&mut self, members: usize) -> Option<StackPair<R, D, S>> {
        self.output.take().filter(|(d, _)| d.members == members)
    }

    /// Bytes retained exclusively by this workspace: output payloads, host
    /// spectra, and Host/CUDA staging buffers. The shared plan and backend
    /// context resources are not included.
    pub fn retained_bytes(&self) -> usize {
        let payloads = self
            .output
            .buffers()
            .map(|(d, v)| {
                (d.members * d.member_len + v.members * v.member_len) * std::mem::size_of::<D>()
            })
            .sum::<usize>();
        let spectra = self.spectra.capacity()
            * std::mem::size_of::<Vec<SectorSpectrum<<R as SectorCodec>::Sector>>>()
            + self
                .spectra
                .iter()
                .map(|member| {
                    member.capacity()
                        * std::mem::size_of::<SectorSpectrum<<R as SectorCodec>::Sector>>()
                        + member
                            .iter()
                            .map(|entry| entry.values.capacity() * std::mem::size_of::<f64>())
                            .sum::<usize>()
                })
                .sum::<usize>();
        let sorted = self.sorted.capacity() * std::mem::size_of::<f64>();
        #[cfg(feature = "cuda")]
        let device_scratch = self.orders.capacity() * std::mem::size_of::<usize>();
        #[cfg(not(feature = "cuda"))]
        let device_scratch = 0;
        payloads + spectra + sorted + device_scratch
    }
}

impl<R: SectorCodec, D, S> EighFullPlan<R, D, S> {
    fn check_workspace(&self, workspace: &EighFullWorkspace<R, D, S>) -> Result<(), Error> {
        if Arc::ptr_eq(&self.identity, &workspace.binding) {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "eigh workspace belongs to another plan".into(),
            ))
        }
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

    fn output_ref<'a>(
        &self,
        workspace: &'a EighFullWorkspace<R, D, S>,
    ) -> Result<EighStackOutput<'a, R, D, S>, Error> {
        let (d, v) = workspace
            .output
            .output
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("eigh output is missing".into()))?;
        Ok(EighStackOutput {
            d,
            v,
            spectra: &workspace.spectra,
        })
    }
}

impl<R, D> EighFullPlan<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar,
{
    /// Decomposes every member into the workspace-owned `(d, v)` and borrows
    /// them with the host spectra.
    ///
    /// Every member is admitted (the eager Hermitian rule, per coupled
    /// sector, one read of its `Σ n²` region entries) before any member is
    /// solved; then each member runs eager's finite-input check and, per
    /// coupled sector, eager's solver stage, compiled once by [`Self::new`].
    /// The stage's eigenvectors are written straight into the member's
    /// slice of `v` (one copy of `Σ n²` per member, from the solver's own
    /// output buffer) and its ascending values onto `d`'s diagonal and the
    /// spectra. Per member TeNeT allocates nothing: the only per-member
    /// allocations are the dense solver's outputs, one set per coupled
    /// sector. The outputs are allocated zeroed when absent or when `B`
    /// changes and reused otherwise: `v` is overwritten, and `d` only on its
    /// diagonal, since the workspace never writes `d` elsewhere.
    ///
    /// # Errors
    ///
    /// [`BatchError::Operation`] with [`Error::BatchSignatureMismatch`] for a
    /// stack of another signature; [`BatchError::MemberRejected`] listing
    /// every non-Hermitian member (before any solve) or, after the solves,
    /// every member with a non-finite eigenvalue; otherwise the eager
    /// operation's errors as [`BatchError::Operation`]. A workspace from
    /// another plan is rejected before it is touched. Once the workspace is
    /// bound to this plan, any execution error leaves no observable output.
    pub fn execute<'a>(
        &self,
        source: &StackedTensorMap<R, D>,
        workspace: &'a mut EighFullWorkspace<R, D>,
    ) -> Result<EighStackOutput<'a, R, D>, BatchError> {
        self.check_workspace(workspace)?;
        if let Err(error) = self.check_source(source) {
            return Err(workspace.output.hide(error.into()));
        }
        let mut buffers = workspace.take_buffers(source.members);
        if let Err(error) = self.run_host(source, &mut buffers, &mut workspace.sorted) {
            return Err(workspace.output.fail(buffers, error));
        }
        if let Err(error) = self.publish(workspace, source.members) {
            return Err(workspace.output.fail(buffers, error.into()));
        }
        if let Some(buffers) = buffers {
            workspace.output.publish(buffers);
        }
        Ok(self.output_ref(workspace)?)
    }

    /// The Host call into `buffers`, allocated zeroed when absent; writes
    /// every member's ascending eigenvalues into `sorted`. `d`'s
    /// off-diagonal entries are never written, so a reused `d` needs only
    /// its diagonals.
    fn run_host(
        &self,
        source: &StackedTensorMap<R, D>,
        buffers: &mut Option<StackPair<R, D, Vec<D>>>,
        sorted: &mut Vec<f64>,
    ) -> Result<(), BatchError> {
        let members = source.members;
        let len = self.member_len;
        let plan = &self.routes;
        let rejected = source
            .storage
            .chunks_exact(len.max(1))
            .take(members)
            .enumerate()
            .filter_map(|(member, data)| {
                match tenet_matrixalgebra::seam::validate_hermitian_regions(
                    data,
                    &plan.source_regions,
                    self.hermitian_tol,
                ) {
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

        if buffers.is_none() {
            *buffers = Some((
                self.stack(
                    plan.middle_space.clone(),
                    plan.d_signature.clone(),
                    vec![D::zero(); plan.d_len * members],
                    members,
                    plan.d_len,
                ),
                self.stack(
                    plan.left_space.clone(),
                    plan.v_signature.clone(),
                    vec![D::zero(); plan.v_len * members],
                    members,
                    plan.v_len,
                ),
            ));
        }
        let Some((d, v)) = buffers.as_mut() else {
            return Err(internal_layout_error("eigh output stacks are missing").into());
        };
        let total = plan.spectrum_len;
        sorted.clear();
        sorted.resize(total * members, 0.0);
        let mut scratch = tenet_matrixalgebra::seam::EighScratch::<D>::for_order(plan.max_n);
        // The scope body is `Send`; the bound spaces are not, so it borrows
        // only the plan's tables and the payloads.
        let storage = source.storage.as_slice();
        let (d_len, v_len) = (plan.d_len, plan.v_len);
        let (d_storage, v_storage) = (d.storage.as_mut_slice(), v.storage.as_mut_slice());
        let EighRoutePlan {
            source_regions,
            routes,
            spectrum_offsets,
            targets,
            copies,
            diagonals,
            ..
        } = plan;
        let runtime = self.runtime.clone();
        let mut dense = runtime.lease_dense();
        // One linear-algebra scope for the batch: eager opens one per member.
        let faults = tenet_matrixalgebra::seam::in_linalg_scope(dense.dense(), |dense| {
            let mut faults = Vec::new();
            for member in 0..members {
                let data = &storage[member * len..(member + 1) * len];
                tenet_matrixalgebra::seam::require_finite_eigh_input(data)?;
                let values = &mut sorted[member * total..(member + 1) * total];
                let d_member = &mut d_storage[member * d_len..(member + 1) * d_len];
                let v_member = &mut v_storage[member * v_len..(member + 1) * v_len];
                for (index, route) in routes.iter().enumerate() {
                    let n = route.rank();
                    let region = &source_regions[route.source_region()];
                    let values = &mut values[spectrum_offsets[index]..][..n];
                    let vectors = match tenet_matrixalgebra::seam::eigh_sector_into(
                        dense,
                        &data[region.range()],
                        n,
                        &mut scratch,
                        values,
                    ) {
                        Ok(vectors) => vectors,
                        // The eager rejection of a non-finite eigenvalue;
                        // every other member is still solved so all of them
                        // are named. An executor's own numerical failure is
                        // not a member fault.
                        Err(OperationError::Dense(tenet_dense::DenseError::NumericalFailure {
                            op: tenet_matrixalgebra::seam::EIGH_EIGENVALUE_CHECK,
                            ..
                        })) => {
                            faults.push((member, MemberFault::NonFiniteEigenvalue));
                            break;
                        }
                        Err(other) => return Err(other),
                    };
                    // One contiguous copy for an aligned route, one row slice
                    // per column and codomain tree otherwise.
                    let (ref range, target_rows) = targets[index];
                    let target = &mut v_member[range.clone()];
                    for &(source_row, target_row, rows) in &copies[index] {
                        if rows == n && target_rows == n {
                            target.copy_from_slice(&vectors);
                            continue;
                        }
                        for column in 0..n {
                            target[column * target_rows + target_row..][..rows]
                                .copy_from_slice(&vectors[column * n + source_row..][..rows]);
                        }
                    }
                    let (offset, step) = diagonals[index];
                    for (position, &value) in values.iter().enumerate() {
                        d_member[offset + position * step] = D::from_real(value);
                    }
                }
            }
            Ok(faults)
        })
        .map_err(Error::from)?;
        drop(dense);
        if !faults.is_empty() {
            return Err(BatchError::MemberRejected { members: faults });
        }
        Ok(())
    }
}

impl<R: SectorCodec, D, S> EighFullPlan<R, D, S> {
    /// Rewrites the workspace spectra from its member-major `sorted` values,
    /// in label order, reusing every buffer of the same shape.
    fn publish(
        &self,
        workspace: &mut EighFullWorkspace<R, D, S>,
        members: usize,
    ) -> Result<(), Error> {
        let plan = &self.routes;
        let total = plan.spectrum_len;
        let sorted = &workspace.sorted;
        publish_spectra(
            &mut workspace.spectra,
            &self.labels,
            members,
            |member, position| match plan.label_routes.get(position).copied().flatten() {
                None => Ok(&[][..]),
                Some(route) => plan
                    .spectrum_offsets
                    .get(route)
                    .zip(plan.routes.get(route))
                    .and_then(|(&offset, route)| {
                        sorted.get(member * total + offset..member * total + offset + route.rank())
                    })
                    .ok_or_else(|| internal_layout_error("a coupled sector has no route")),
            },
        )
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
/// sectors, in `labels` order, from `values(member, label position)`,
/// reusing every buffer of the same shape.
fn publish_spectra<'v, S: Clone>(
    spectra: &mut Vec<Vec<SectorSpectrum<S>>>,
    labels: &[(SectorId, S)],
    members: usize,
    mut values: impl FnMut(usize, usize) -> Result<&'v [f64], Error>,
) -> Result<(), Error> {
    spectra.resize_with(members, Vec::new);
    for (member, entries) in spectra.iter_mut().enumerate() {
        entries.truncate(labels.len());
        for (position, (_, label)) in labels.iter().enumerate() {
            let source = values(member, position)?;
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

impl<R> EighRoutePlan<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    /// Compiles the eager plan from the structure: `eigh_full` keeps every
    /// eigenpair, so each coupled sector's rank is its full `n`.
    fn new<D, S>(
        source: &StackedTensorMap<R, D, S>,
        labels: &[(SectorId, <R as SectorCodec>::Sector)],
    ) -> Result<Self, Error> {
        let plan = tenet_matrixalgebra::seam::compact_factor_routes(&source.space)?;
        let routes: Vec<_> = plan
            .routes()
            .iter()
            .filter(|route| route.left_region().is_some() && route.right_region().is_some())
            .copied()
            .collect();
        let left_space = plan.left_space(&source.space)?;
        let middle_space = plan.bond_space(&source.space)?;
        let label_routes = labels
            .iter()
            .map(|(sector, _)| routes.iter().position(|route| route.sector() == *sector))
            .collect();
        let max_n = routes.iter().map(|route| route.rank()).max().unwrap_or(0);
        let targets = routes
            .iter()
            .map(|route| {
                let region = &plan.left_regions()[left_region(route)?];
                Ok((region.range(), region.rows()))
            })
            .collect::<Result<_, Error>>()?;
        #[cfg(feature = "cuda")]
        let admission = plan
            .source_regions()
            .iter()
            .map(|region| (region.range().start, region.rows()))
            .collect();
        let mut spectrum_offsets = Vec::with_capacity(routes.len());
        let mut copies = Vec::with_capacity(routes.len());
        let mut spectrum_len = 0usize;
        for route in &routes {
            spectrum_offsets.push(spectrum_len);
            spectrum_len += route.rank();
            copies.push(route_copies(&plan, route)?);
        }
        let diagonals = route_diagonals(&routes, &middle_space)?;
        let placement = source.signature.placement;
        let d_signature = StructureSignature::of_space(&middle_space, placement, &source.runtime);
        let v_signature = StructureSignature::of_space(&left_space, placement, &source.runtime);
        let d_len = middle_space.space().required_len()?;
        let v_len = left_space.space().required_len()?;
        Ok(Self {
            source_regions: Arc::clone(plan.source_regions()),
            routes,
            left_space,
            middle_space,
            #[cfg(feature = "cuda")]
            admission,
            spectrum_offsets,
            label_routes,
            targets,
            max_n,
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
fn route_copies(
    plan: &tenet_matrixalgebra::seam::CompactFactorPlan,
    route: &tenet_matrixalgebra::seam::CompactFactorRoute,
) -> Result<Vec<(usize, usize, usize)>, Error> {
    let target = &plan.left_regions()[left_region(route)?];
    #[cfg(test)]
    let aligned =
        plan.left_preserves_trees(route) && !tests::FORCE_TREEWISE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let aligned = plan.left_preserves_trees(route);
    if aligned {
        return Ok(vec![(0, 0, target.rows())]);
    }
    let source = &plan.source_regions()[route.source_region()];
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

/// An executed route's eigenvector region.
fn left_region(route: &tenet_matrixalgebra::seam::CompactFactorRoute) -> Result<usize, Error> {
    route
        .left_region()
        .ok_or_else(|| internal_layout_error("an executed EIGH route has no eigenvector region"))
}

/// Each route's diagonal in the dense `d` of one member, `(offset, step)`.
fn route_diagonals<R>(
    routes: &[tenet_matrixalgebra::seam::CompactFactorRoute],
    middle_space: &tenet_tensors::BoundDynamicFusionMapSpace<R>,
) -> Result<Vec<(usize, usize)>, Error> {
    let diagonals = sector_diagonals(middle_space.space().structure())?;
    routes
        .iter()
        .map(|route| {
            let sector = route.sector();
            match diagonals.iter().find(|entry| entry.0 == sector) {
                Some(&(_, offset, step, count)) if count == route.rank() => Ok((offset, step)),
                _ => Err(internal_layout_error(
                    "CUDA EIGH route has no matching diagonal block",
                )),
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
impl<R, D> EighFullPlan<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    /// The device form of the Host [`EighFullPlan::execute`].
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
    /// call after [`EighFullWorkspace::take_output`], allocates both with one
    /// zero upload each (`B · Σ n²` and `B · L_v` elements, the only device
    /// allocation path until #740); `d`'s off-diagonal entries are never
    /// written.
    ///
    /// # Errors
    ///
    /// As the Host form, with non-Hermitian members found by the batched
    /// admission before any solver launch, and [`BatchError::Solver`] when
    /// the solver reports a numerical failure on a block. Argument, bounds
    /// and backend failures are [`BatchError::Operation`].
    pub fn execute<'a>(
        &self,
        source: &StackedTensorMap<R, D, CudaStorage<D>>,
        workspace: &'a mut EighFullWorkspace<R, D, CudaStorage<D>>,
    ) -> Result<EighStackOutput<'a, R, D, CudaStorage<D>>, BatchError> {
        self.check_workspace(workspace)?;
        if let Err(error) = self.check_source(source) {
            return Err(workspace.output.hide(error.into()));
        }
        let mut buffers = workspace.take_buffers(source.members);
        if let Err(error) = self.run_cuda(
            source,
            &mut buffers,
            &mut workspace.sorted,
            &mut workspace.orders,
        ) {
            return Err(workspace.output.fail(buffers, error));
        }
        if let Err(error) = self.publish(workspace, source.members) {
            return Err(workspace.output.fail(buffers, error.into()));
        }
        if let Some(buffers) = buffers {
            workspace.output.publish(buffers);
        }
        Ok(self.output_ref(workspace)?)
    }

    /// The device call into `buffers`, allocated zeroed when absent; returns
    /// every member's sorted eigenvalues, member-major (`Σ n` per member).
    fn run_cuda(
        &self,
        source: &StackedTensorMap<R, D, CudaStorage<D>>,
        buffers: &mut Option<StackPair<R, D, CudaStorage<D>>>,
        sorted: &mut Vec<f64>,
        orders: &mut Vec<usize>,
    ) -> Result<(), BatchError> {
        let members = source.members;
        let runtime = self.runtime.clone();
        let device = &self.routes;
        let mut lease = runtime.lease_cuda()?;
        let cuda = &mut *lease;
        let routes = &device.routes;

        let accepted = tenet_dense::cuda_hermitian_regions_batched::<D>(
            cuda,
            &source.storage.0,
            &device.admission,
            members,
            self.member_len,
            self.hermitian_tol
                .resolve(<D as tenet_matrixalgebra::FactorScalar>::epsilon()),
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
            let region = &device.source_regions[route.source_region()];
            let (values, vector) = tenet_dense::cuda_eigh_region_batched::<D>(
                cuda,
                &source.storage.0,
                region.range().start,
                route.rank(),
                members,
                self.member_len,
            )
            .map_err(|error| match error {
                // Only the solver's own status names a block; argument,
                // bounds and backend failures are the operation's.
                error @ tenet_dense::DenseError::NumericalFailure { .. } => BatchError::Solver {
                    block: route.source_region(),
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

        // Per member and route, eager's order (one authority).
        let total = device.spectrum_len;
        sorted.clear();
        sorted.resize(total * members, 0.0);
        orders.clear();
        orders.resize(total * members, 0);
        let mut faults = Vec::new();
        for member in 0..members {
            let mut finite = true;
            for (route, &offset) in routes.iter().zip(&device.spectrum_offsets) {
                let n = route.rank();
                let base = member * total + offset;
                let values = &raw[base..base + n];
                finite &= values.iter().all(|value| value.is_finite());
                let order = &mut orders[base..base + n];
                tenet_matrixalgebra::seam::ascending_eigh_order(values, order);
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
        // allocation path, #740). The workspace never writes `d` off its
        // diagonal, so each call moves only the `B · Σ n` sorted values.
        if buffers.is_none() {
            let zeros = |len: usize| {
                CudaStorage::<D>::upload_members(cuda, vec![D::zero(); len * members], len, members)
                    .map_err(Error::from)
            };
            *buffers = Some((
                self.stack(
                    device.middle_space.clone(),
                    device.d_signature.clone(),
                    zeros(device.d_len)?,
                    members,
                    device.d_len,
                ),
                self.stack(
                    device.left_space.clone(),
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
                &region(vec![route.rank(), members], vec![1, total], spectrum)?,
                &mut d.storage.0,
                &region(
                    vec![route.rank(), members],
                    vec![step, device.d_len],
                    offset,
                )?,
            )
            .map_err(dense_err)?;
        }
        for ((((route, raw_vectors), &offset), copies), (target, target_rows)) in routes
            .iter()
            .zip(&vectors)
            .zip(&device.spectrum_offsets)
            .zip(&device.copies)
            .zip(&device.targets)
        {
            let columns: Vec<usize> = (0..members)
                .flat_map(|member| {
                    orders[member * total + offset..][..route.rank()]
                        .iter()
                        .copied()
                })
                .collect();
            tenet_dense::cuda_gather_columns_batched_into::<D>(
                cuda,
                &mut v.storage.0,
                target.start,
                *target_rows,
                device.v_len,
                raw_vectors,
                route.rank(),
                members,
                &columns,
                copies,
            )
            .map_err(dense_err)?;
        }
        Ok(())
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
    /// table and one Tenferro `gather` launch per call, whatever `B`, the
    /// block count and the number of restricted legs. Nothing is downloaded
    /// and nothing waits on the device.
    ///
    /// The table maps each destination element to its source element within
    /// a member. It is the eager restriction kernel applied once, on the Host,
    /// to the payload `0, 1, …, L - 1`, so the block plan has one
    /// implementation; the gather then reads that element of every member
    /// (its window is the member axis) and allocates the `[L', B]` result.
    pub fn restrict_leg(
        &self,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        require_restriction_set(&self.space, legs)?;
        let destination = restricted_space(&self.space, legs)?;
        // The element table is the eager kernel itself run over the payload
        // `0, 1, …, L - 1` (in `i64`, the gather's index type), so the block
        // plan keeps one implementation. Why not enumerate the destination
        // blocks here instead: that would restate the sector lookup, start
        // offsets and rectangle checks of `restrict_block`. The price is one
        // `L`-element iota per call, host work `O(L + L')`, independent of `B`
        // and of the number of restricted legs.
        let elements = {
            let _host_pool = self.runtime.enter_host_pool();
            let iota = (0..self.member_len)
                .map(i64::try_from)
                .collect::<Result<Vec<i64>, _>>()
                .map_err(|_| Error::InvalidArgument("restrict_leg: a member exceeds i64".into()))?;
            tenet_tensors::oriented_fusion_restrict_owned(
                destination.space().structure(),
                FusionOperand::direct(self.space.space()),
                &iota,
                &restriction_runs(self.rank(), legs),
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
        /// Forces every eigh route onto the per-tree copies, so the branch
        /// the fixtures' aligned routes never take is exercised.
        pub(super) static FORCE_TREEWISE: std::cell::Cell<bool> = const {
            std::cell::Cell::new(false)
        };
    }

    #[test]
    fn spectra_publication_failure_does_not_publish_factors() {
        let runtime = Runtime::builder().build().unwrap();
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
        let x = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 1).unwrap();
        let input = x.axpby(1.0, &x.adjoint().unwrap(), 1.0).unwrap();
        let stack = super::StackedTensorMap::pack(&[input]).unwrap();
        let mut plan =
            super::EighFullPlan::new(&stack, &[0], &[1], super::HermitianTol::DEFAULT).unwrap();
        let mut workspace = plan.workspace().unwrap();
        plan.routes.label_routes[0] = Some(usize::MAX);

        assert!(plan.execute(&stack, &mut workspace).is_err());
        assert!(workspace.take_output().is_none());
    }

    #[test]
    fn compose_failure_after_the_output_is_taken_keeps_it_unobservable() {
        // What: an error inside the replay, after the output buffer is
        // taken (forced by an operand whose layout and payload are
        // consistently one entry short, which only a corrupted stack
        // carries), leaves no observable output; the buffer is kept and the
        // next call recovers.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v], 3).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 4).unwrap();
        let lhs = super::StackedTensorMap::pack(&[&a, &a]).unwrap();
        let rhs = super::StackedTensorMap::pack(&[&b, &b]).unwrap();
        let mut corrupt = super::StackedTensorMap::pack(&[&a, &a]).unwrap();
        corrupt.member_len -= 1;
        corrupt
            .storage
            .truncate(corrupt.member_len * corrupt.members);
        let plan = super::ComposePlan::new(&lhs, &rhs).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let expected = plan
            .execute(&lhs, &rhs, &mut workspace)
            .unwrap()
            .storage
            .clone();
        let bytes = workspace.retained_bytes();

        // Only the route executor compares the payload with the plan's
        // operand structure; every earlier check passes.
        let Err(crate::error::Error::Operation(error)) =
            plan.execute(&corrupt, &rhs, &mut workspace).map(|_| ())
        else {
            panic!("the corrupted stack must fail as an operation error");
        };
        assert!(
            matches!(
                *error,
                tenet_tensors::OperationError::ElementCountMismatch { .. }
            ),
            "{error:?}"
        );
        assert!(
            workspace.output.spare.is_some(),
            "the error is after the take"
        );
        assert!(workspace.take_output().is_none());
        assert_eq!(workspace.retained_bytes(), bytes);
        assert_eq!(
            plan.execute(&lhs, &rhs, &mut workspace).unwrap().storage,
            expected
        );
        assert!(workspace.take_output().is_some());
    }

    #[test]
    fn host_tree_wise_eigenvector_copies_equal_the_aligned_ones_and_eager_bitwise() {
        // What: on Host the per-tree placement (one row slice per column
        // and codomain tree) writes exactly the bits of the aligned one (one
        // copy per sector), and both are eager's `d` and `v` per member.
        use super::EighFullPlan;
        use tenet_core::{SU2FusionRule, SU2Irrep};

        let runtime = Runtime::builder().build().unwrap();
        let j = SU2Irrep::from_twice_spin;
        let leg = GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 2), (j(2), 1)])
            .unwrap();
        let members: Vec<_> = (0..3)
            .map(|seed| {
                let x =
                    TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], seed)
                        .unwrap();
                x.axpby(1.0, &x.adjoint().unwrap(), 1.0).unwrap()
            })
            .collect();
        let stack = super::StackedTensorMap::pack(&members).unwrap();
        let run = |treewise: bool| {
            FORCE_TREEWISE.with(|flag| flag.set(treewise));
            let plan =
                EighFullPlan::new(&stack, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT).unwrap();
            FORCE_TREEWISE.with(|flag| flag.set(false));
            let mut workspace = plan.workspace().unwrap();
            let copies: usize = plan.routes.copies.iter().map(Vec::len).sum();
            let routes = plan.routes.copies.len();
            let output = plan.execute(&stack, &mut workspace).unwrap();
            (
                copies,
                routes,
                output.d.storage.clone(),
                output.v.storage.clone(),
            )
        };
        let (aligned_copies, routes, aligned_d, aligned_v) = run(false);
        let (tree_copies, _, tree_d, tree_v) = run(true);
        assert_eq!(aligned_copies, routes, "every fixture route is aligned");
        assert!(tree_copies > routes, "several codomain trees per sector");
        assert!(tree_v == aligned_v && tree_d == aligned_d);
        let d_len = aligned_d.len() / members.len();
        let v_len = aligned_v.len() / members.len();
        for (member, input) in members.iter().enumerate() {
            let crate::typed::Eigh { d, v } = input
                .eigh_full(&[0, 1], &[2, 3], super::HermitianTol::DEFAULT)
                .unwrap();
            assert!(tree_v[member * v_len..][..v_len] == *v.dense_data().unwrap());
            assert!(
                tree_d[member * d_len..][..d_len]
                    == *d.materialize().unwrap().dense_data().unwrap()
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn tree_wise_eigenvector_copies_equal_the_aligned_ones_bitwise() {
        // What: the per-tree placement (one copy per codomain tree) writes
        // exactly the bits of the aligned placement (one copy per sector),
        // and both are eager's device `v` at B = 1.
        use super::EighFullPlan;
        use tenet_core::{SU2FusionRule, SU2Irrep};

        let runtime = Runtime::builder().cuda(0).build().unwrap();
        let j = SU2Irrep::from_twice_spin;
        let leg = GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 2), (j(2), 1)])
            .unwrap();
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
            let plan =
                EighFullPlan::new(&stack, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT).unwrap();
            let mut workspace = plan.workspace().unwrap();
            FORCE_TREEWISE.with(|flag| flag.set(false));
            let copies: usize = plan.routes.copies.iter().map(Vec::len).sum();
            let routes = plan.routes.copies.len();
            let output = plan.execute(&stack, &mut workspace).unwrap();
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
        let plan =
            EighFullPlan::new(&single, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT).unwrap();
        let mut workspace = plan.workspace().unwrap();
        FORCE_TREEWISE.with(|flag| flag.set(false));
        let v = plan
            .execute(&single, &mut workspace)
            .unwrap()
            .v
            .to_host()
            .unwrap()
            .member(0)
            .unwrap();
        let crate::typed::Eigh { v: eager_v, .. } = members[0]
            .to_cuda()
            .unwrap()
            .eigh_full(&[0, 1], &[2, 3], super::HermitianTol::DEFAULT)
            .unwrap();
        assert!(v.dense_data().unwrap() == eager_v.to_host().unwrap().dense_data().unwrap());
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
            Arc::new(U1FusionRule),
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
                    len: host.dense_data().unwrap().len(),
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
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 2), (q(1), 1)]).unwrap();
        let wide = GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 3), (q(1), 1)]).unwrap();
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
        use std::sync::Arc;
        use tenet_core::SUNFusionRule;

        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let signature = |rank: usize, trivial: Vec<i64>| {
            let rule = Arc::new(SUNFusionRule::new(rank).unwrap());
            let leg = GradedSpace::try_new(rule, [(trivial, 2)]).unwrap();
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
