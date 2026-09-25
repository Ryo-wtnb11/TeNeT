//! Structure signatures: the key under which tensors can share one prepared
//! plan and one stacked payload layout (#1287).
//!
//! Neither TensorKit nor QSpace has a batched path, so the signature is
//! defined by what TeNeT's eager operations read from an operand: the rule
//! authority, the logical hom space, the fusion-tree block structure, the
//! payload placement and the Runtime. The payload dtype is the type parameter
//! `D` of every consumer and is therefore checked by the compiler, not here.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenet_core::{BlockStructureContent, HomSpaceId, Placement, RuleIdentity, TensorStorage};

use super::{TensorMap, TypedSectorAdmission};
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
        self.placement == other.placement
            && self.runtime == other.runtime
            && self.rule == other.rule
            && self.homspace == other.homspace
            && (Arc::ptr_eq(&self.structure, &other.structure) || self.structure == other.structure)
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

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use tenet_core::{Placement, TensorStorage, U1FusionRule, U1Irrep};

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
