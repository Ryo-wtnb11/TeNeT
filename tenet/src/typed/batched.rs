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

use tenet_core::{BlockStructureContent, HomSpaceId, Placement, RuleIdentity, TensorStorage};

use super::{
    owned_repr, BoundDynamicFusionMapSpace, Runtime, TensorMap, TypedData, TypedSectorAdmission,
    TypedTensorBody, TypedTensorRepr,
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
pub enum BatchMemberRepresentation {
    /// A lazy adjoint view over its parent's payload.
    LazyAdjoint,
    /// A compact diagonal spectrum, whose dense layout is not stored.
    CompactDiagonal,
}

impl StructureSignature {
    /// The first field in which `other` differs, in the order `==` checks
    /// them, or `None` when the signatures are equal.
    pub fn first_mismatch(&self, other: &Self) -> Option<SignatureField> {
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
        let space = self.logical_space();
        // Why the admission's identity first: it is the identity the layout
        // was validated against and clones without allocating, whereas
        // `typed_rule_identity` rebuilds canonical bytes for content-keyed
        // checked Generic providers. Every bound space carries one.
        let rule = match space.space().admission().rule_identity() {
            Some(identity) => identity.clone(),
            None => TypedSectorAdmission::typed_rule_identity(self.provider()),
        };
        StructureSignature {
            rule,
            homspace: space.space().homspace().id(),
            structure: space.space().structure().content_key(),
            placement: self.placement(),
            runtime: self.runtime.identity(),
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
