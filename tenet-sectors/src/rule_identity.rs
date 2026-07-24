use std::hash::Hash;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct RuleIdentity(RuleIdentityNode);

#[derive(Clone, Debug)]
enum RuleIdentityNode {
    Type(std::any::TypeId),
    Unique(std::any::TypeId, u64),
    Content {
        rule_type: std::any::TypeId,
        prehash: u64,
        bytes: Arc<[u8]>,
    },
    Product(Arc<ProductRuleIdentity>),
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct ProductRuleIdentity {
    codec: std::any::TypeId,
    left: RuleIdentity,
    right: RuleIdentity,
}

impl PartialEq for RuleIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for RuleIdentity {}

impl std::hash::Hash for RuleIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for RuleIdentityNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Type(left), Self::Type(right)) => left == right,
            (Self::Unique(left_type, left), Self::Unique(right_type, right)) => {
                left_type == right_type && left == right
            }
            (
                Self::Content {
                    bytes: left,
                    prehash: left_hash,
                    rule_type: left_type,
                },
                Self::Content {
                    bytes: right,
                    prehash: right_hash,
                    rule_type: right_type,
                },
            ) => {
                left_type == right_type
                    && left_hash == right_hash
                    && (Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref())
            }
            (Self::Product(left), Self::Product(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for RuleIdentityNode {}

impl std::hash::Hash for RuleIdentityNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Type(rule_type) => rule_type.hash(state),
            Self::Unique(rule_type, instance) => {
                rule_type.hash(state);
                instance.hash(state);
            }
            Self::Content {
                rule_type, prehash, ..
            } => {
                rule_type.hash(state);
                prehash.hash(state);
            }
            Self::Product(identity) => identity.hash(state),
        }
    }
}

impl RuleIdentity {
    pub fn of_type<R: 'static + ?Sized>() -> Self {
        Self(RuleIdentityNode::Type(std::any::TypeId::of::<R>()))
    }

    pub fn new_unique<R: 'static>() -> Self {
        static NEXT_INSTANCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let instance = NEXT_INSTANCE
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |current| current.checked_add(1),
            )
            .expect("fusion-rule identity space exhausted");
        Self(RuleIdentityNode::Unique(
            std::any::TypeId::of::<R>(),
            instance,
        ))
    }

    pub fn from_canonical_bytes<R: 'static>(prehash: u64, bytes: Arc<[u8]>) -> Self {
        Self(RuleIdentityNode::Content {
            rule_type: std::any::TypeId::of::<R>(),
            prehash,
            bytes,
        })
    }

    /// Composes two semantic category identities under one product codec type.
    #[doc(hidden)]
    pub fn compose_with_codec<Codec: 'static>(left: Self, right: Self) -> Self {
        Self(RuleIdentityNode::Product(Arc::new(ProductRuleIdentity {
            codec: std::any::TypeId::of::<Codec>(),
            left,
            right,
        })))
    }

    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        const ARC_HEADER_BYTES: usize = 2 * std::mem::size_of::<usize>();

        // Why not charge only `size_of::<RuleIdentity>()`: content and product
        // identities retain heap allocations whose size can dominate the key.
        // Shared descendants are deliberately charged recursively per entry;
        // the admission budget is conservative rather than allocator-exact.
        match &self.0 {
            RuleIdentityNode::Type(_) | RuleIdentityNode::Unique(_, _) => 0,
            RuleIdentityNode::Content { bytes, .. } => ARC_HEADER_BYTES
                .saturating_add(bytes.len().saturating_mul(std::mem::size_of::<u8>())),
            RuleIdentityNode::Product(identity) => ARC_HEADER_BYTES
                .saturating_add(std::mem::size_of::<ProductRuleIdentity>())
                .saturating_add(identity.left.charged_retained_bytes())
                .saturating_add(identity.right.charged_retained_bytes()),
        }
    }
}
