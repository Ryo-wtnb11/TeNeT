use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock, RwLock};

use rustc_hash::FxHashMap;
use tenet_core::{
    BlockKey, BlockStructure, CoreError, DimVec, FusionRule, FusionTensorMapSpace,
    FusionTreeBlockKey, FusionTreeHomSpace, HomSpaceId, LoweredMultiplicityFreeAlgebra,
    MultiplicityFreeFusionRule, MultiplicityFreeRigidSymbols, RuleIdentity,
};

use crate::cache::registered_operation_cache;
use crate::{OperationError, TreeTransformOperation};
use tenet_operations::{OutputAxisOrder, TensorContractSpec};

pub(crate) type LayoutPrimer<R> = fn(&R, &FusionTreeHomSpace) -> Result<(), OperationError>;

struct LayoutBuildCapability<R> {
    prime: LayoutPrimer<R>,
}

impl<R> Copy for LayoutBuildCapability<R> {}

impl<R> Clone for LayoutBuildCapability<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> LayoutBuildCapability<R> {
    const fn encoded() -> Self {
        Self {
            prime: encoded_layout_primer::<R>,
        }
    }

    fn same_as(self, other: Self) -> bool {
        self.prime as usize == other.prime as usize
    }

    fn prime(self, rule: &R, homspace: &FusionTreeHomSpace) -> Result<(), OperationError> {
        (self.prime)(rule, homspace)
    }
}

impl<R> LayoutBuildCapability<R>
where
    R: LoweredMultiplicityFreeAlgebra,
{
    const fn lowered() -> Self {
        Self {
            prime: lowered_layout_primer::<R>,
        }
    }
}

pub(crate) fn encoded_layout_primer<R>(
    _rule: &R,
    _homspace: &FusionTreeHomSpace,
) -> Result<(), OperationError> {
    Ok(())
}

pub(crate) fn lowered_layout_primer<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
) -> Result<(), OperationError>
where
    R: LoweredMultiplicityFreeAlgebra,
{
    homspace
        .try_fusion_tree_keys_lowered(rule)
        .map(|_| ())
        .map_err(|error| OperationError::InvalidArgument {
            message: error.static_message(),
        })
}

/// Identity of a pairwise contraction's output space: the two operand hom
/// spaces (by value — `Arc` gives cheap clones while `Hash`/`Eq` delegate to
/// the pointee, so a rebuilt-but-identical space still matches) plus the
/// contracted axis lists. The hom spaces carry the full sector/leg structure
/// (authoritative even for structural-zero keys the subblock layout omits, so
/// a subblock content id alone is NOT enough), and the output is a pure
/// function of these. The same contraction across sweeps/evals thus reuses one
/// built space instead of rebuilding the coupled-sector layout each call.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ContractedSpaceKey {
    /// Fusion rule discriminant: the same sector ids fuse differently under
    /// different rules, so the process-global cache must key on the rule.
    /// Mirrors `FusionTreeHomSpaceCacheKey` in tenet-core.
    rule: tenet_core::RuleIdentity,
    lhs_homspace: Arc<FusionTreeHomSpace>,
    rhs_homspace: Arc<FusionTreeHomSpace>,
    lhs_axes: DimVec,
    rhs_axes: DimVec,
    output_axes: DimVec,
}

fn contracted_space_cache() -> &'static RwLock<FxHashMap<ContractedSpaceKey, DynamicFusionMapSpace>>
{
    static CACHE: OnceLock<RwLock<FxHashMap<ContractedSpaceKey, DynamicFusionMapSpace>>> =
        OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

/// Identity of a tree-transform (permute / braid / transpose) output space:
/// the source hom space (by value — the legs carry the authoritative
/// sector→degeneracy map the output shapes derive from) plus the transform
/// operation, under a given fusion rule. Mirrors [`ContractedSpaceKey`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TransformedSpaceKey {
    rule: tenet_core::RuleIdentity,
    source_homspace: Arc<FusionTreeHomSpace>,
    operation: TreeTransformOperation,
}

fn transformed_space_cache(
) -> &'static RwLock<FxHashMap<TransformedSpaceKey, DynamicFusionMapSpace>> {
    static CACHE: OnceLock<RwLock<FxHashMap<TransformedSpaceKey, DynamicFusionMapSpace>>> =
        OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

#[cfg(test)]
thread_local! {
    static FINAL_RESULT_LAYOUT_BUILDS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[inline]
fn observe_final_result_layout_build() {
    #[cfg(test)]
    FINAL_RESULT_LAYOUT_BUILDS.with(|builds| builds.set(builds.get() + 1));
}

#[cfg(test)]
fn reset_final_result_layout_builds() {
    FINAL_RESULT_LAYOUT_BUILDS.with(|builds| builds.set(0));
}

#[cfg(test)]
fn final_result_layout_builds() -> usize {
    FINAL_RESULT_LAYOUT_BUILDS.with(std::cell::Cell::get)
}

/// Identity of a rule-specific coupled-storage layout: the fusion rule's
/// coefficient provenance paired with the interned [`HomSpaceId`]. Kept as a
/// named primitive because the two must never be conflated — the coupled
/// layout is a function of BOTH the hom space and the rule, and PR-1 proved
/// that keying a layout on a hom-space-independent structure id (a subblock
/// `content_id` alone) aliases distinct sources (a finite-torus singlet then
/// failed with a dimension mismatch).
///
/// Why-not (`type_name::<R>()`): the trait permits two instances of one Rust
/// type to carry different fusion tables. [`RuleIdentity`] is the semantic
/// provenance boundary already used by the sibling transformed/contracted
/// caches, and prevents those instances from sharing a layout.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct LayoutId {
    rule: RuleIdentity,
    homspace: HomSpaceId,
}

/// Cache identity of a coupled scratch [`BlockStructure`]: its rule-layout plus
/// the concrete degeneracy shapes (which carry the bond dimension χ, so a
/// differently-truncated build keys separately).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ScratchStructureKey {
    layout: LayoutId,
    nout: usize,
    rank: usize,
    shapes: Arc<[DimVec]>,
}

/// Bounded LRU store of built scratch structures. Payload is a STRONG `Arc`,
/// not a `Weak`: `from_degeneracy_shapes` returns a fresh `DynamicFusionMapSpace`
/// whose owner (a transient per-eval scratch space) is dropped between warm
/// evals, so a `Weak` would be dead on the next eval and rebuild every time —
/// the same by-value-ownership trap PR-1 hit for the adjoint cache. The typed
/// `coupled_block_structure_cache` can use `Weak` because its structures are
/// owned by long-lived network tensors; the dynamic scratch path cannot.
struct ScratchStructureCache {
    entries: lru::LruCache<ScratchStructureKey, Arc<BlockStructure>>,
}

impl Default for ScratchStructureCache {
    fn default() -> Self {
        Self {
            entries: lru::LruCache::new(
                std::num::NonZeroUsize::new(SCRATCH_STRUCTURE_CACHE_CAP).unwrap(),
            ),
        }
    }
}

/// Residency bound for the scratch-structure cache; matches the adjoint cache
/// cap. A finite-torus eval interns O(100) distinct scratch structures, so this
/// never evicts in that workload — it only keeps an adversarial
/// many-distinct-layout run from growing the strong cache without bound.
const SCRATCH_STRUCTURE_CACHE_CAP: usize = 8192;

/// Process-global scratch-structure cache, held in the operation registry so
/// `reset_global_operation_caches` clears it. Own accessor (not `typed_global_map`)
/// because the map and its LRU order must share one lock, mirroring the adjoint
/// cache accessor.
fn scratch_structure_cache() -> Arc<RwLock<ScratchStructureCache>> {
    registered_operation_cache::<RwLock<ScratchStructureCache>>().1
}

/// Builds scratch structures in the coupled-sector matrix layout. Scratch
/// spaces enumerate the full tree set of their hom spaces, so the coupled
/// grid is always complete; there is no other layout.
// `R: FusionRule` (not mult-free): the coupled-sector matrix layout only needs
// fusion channels/dual, so this helper serves both the mult-free and the
// Generic (SU(3)) space builders. Relaxing the bound leaves the mult-free
// callers unchanged.
fn scratch_subblock_structure<R>(
    rule: &R,
    nout: usize,
    rank: usize,
    blocks: Vec<(BlockKey, Vec<usize>)>,
) -> Result<BlockStructure, OperationError>
where
    R: FusionRule,
{
    let mut tree_blocks = Vec::with_capacity(blocks.len());
    for (index, (key, shape)) in blocks.iter().enumerate() {
        match key {
            BlockKey::FusionTree(tree) => tree_blocks.push((tree.clone(), shape.clone())),
            BlockKey::Dense => {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "scratch",
                    index,
                })
            }
        }
    }
    BlockStructure::coupled_sector_matrix_with_keys(rule, nout, rank, tree_blocks)
        .map_err(OperationError::from_core_preserving_context)
}

use super::fusion::FusionContractPlan;
use super::structure::TensorContractAxisPlan;

/// Dynamic-rank fusion space: the expert-layer space handle whose
/// codomain/domain split is a runtime property.
///
/// Typed [`FusionTensorMapSpace`] facades lower to this type internally; the
/// dynamic expert entry points (`*_dyn_into`) take it directly together with
/// raw `f64` slices in the coupled-sector matrix layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicFusionMapSpace {
    nout: usize,
    nin: usize,
    homspace: Arc<FusionTreeHomSpace>,
    subblock_structure: Arc<BlockStructure>,
    rule_identity: Option<tenet_core::RuleIdentity>,
}

/// Internal contraction operand separating categorical and storage authority.
///
/// `logical_space` defines sectors, trees, and user axes; `storage_space`
/// defines the physical block buffer. Why not expose this as a general public
/// API: only TeNeT's validated lazy-adjoint representation can prove that the
/// two spaces describe the same tensor.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct FusionOperand<'a> {
    logical_space: &'a DynamicFusionMapSpace,
    storage_space: &'a DynamicFusionMapSpace,
    storage_conjugate: bool,
}

impl<'a> FusionOperand<'a> {
    pub fn direct(space: &'a DynamicFusionMapSpace) -> Self {
        Self {
            logical_space: space,
            storage_space: space,
            storage_conjugate: false,
        }
    }

    pub fn prelowered_adjoint(
        logical_space: &'a DynamicFusionMapSpace,
        storage_space: &'a DynamicFusionMapSpace,
    ) -> Result<Self, OperationError> {
        if logical_space.rank() != storage_space.rank()
            || logical_space.nout() != storage_space.nin()
            || logical_space.nin() != storage_space.nout()
            || logical_space.homspace().codomain() != storage_space.homspace().domain()
            || logical_space.homspace().domain() != storage_space.homspace().codomain()
            || logical_space.rule_identity != storage_space.rule_identity
        {
            return Err(OperationError::StructureMismatch {
                tensor: "prelowered adjoint operand",
            });
        }
        Ok(Self {
            logical_space,
            storage_space,
            storage_conjugate: true,
        })
    }

    #[inline]
    pub fn logical_space(self) -> &'a DynamicFusionMapSpace {
        self.logical_space
    }

    #[inline]
    pub fn storage_space(self) -> &'a DynamicFusionMapSpace {
        self.storage_space
    }

    #[inline]
    pub fn storage_conjugate(self) -> bool {
        self.storage_conjugate
    }
}

fn validate_bound_space_invariants(space: &DynamicFusionMapSpace) -> Result<(), OperationError> {
    let expected_nout = space.homspace().codomain().len();
    let expected_nin = space.homspace().domain().len();
    if space.nout() != expected_nout || space.nin() != expected_nin {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionSpaceSplitMismatch {
                expected_nout,
                expected_nin,
                actual_nout: space.nout(),
                actual_nin: space.nin(),
            },
        ));
    }
    let expected_rank = expected_nout + expected_nin;
    if space.structure().rank() != expected_rank {
        return Err(OperationError::from_core_preserving_context(
            CoreError::StructureRankMismatch {
                expected: expected_rank,
                actual: space.structure().rank(),
            },
        ));
    }
    Ok(())
}

/// A complete dynamic fusion space bound to the provider that defines it.
///
/// Construct this with [`Self::bind_multiplicity_free`] for a
/// [`MultiplicityFreeFusionRule`] and [`Self::bind_generic`] for a generic
/// fusion rule. Both constructors compare the space against the exact tree set
/// produced by the selected enumeration mode, so choosing the wrong mode can
/// only succeed when the two modes are semantically identical for that space.
/// A missing rule identity is rejected rather than inferred.
pub struct BoundDynamicFusionMapSpace<R> {
    space: DynamicFusionMapSpace,
    provider: Arc<R>,
    layout_build: LayoutBuildCapability<R>,
}

#[derive(Clone, Debug)]
/// Provider-neutral dynamic layout that has passed a bound space's full validation.
///
/// Why not expose the raw space, its metadata fields, or the first provider:
/// cached consumers must preserve one validation proof without reconstructing
/// identity or retaining an arbitrary provider allocation.
pub struct ValidatedDynamicFusionLayout(DynamicFusionMapSpace);

impl ValidatedDynamicFusionLayout {
    /// Flat storage length required by this validated layout.
    ///
    /// Why not expose the raw space: executors only need allocation length;
    /// structural access would let consumers rebuild a second authority.
    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.0.required_len()
    }
}

impl PartialEq for ValidatedDynamicFusionLayout {
    fn eq(&self, other: &Self) -> bool {
        self.0.rule_identity == other.0.rule_identity
            && self.0.homspace().id() == other.0.homspace().id()
            && self.0.structure().content_id() == other.0.structure().content_id()
            && self.0.nout() == other.0.nout()
            && self.0.nin() == other.0.nin()
    }
}

impl Eq for ValidatedDynamicFusionLayout {}

impl Hash for ValidatedDynamicFusionLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.rule_identity.hash(state);
        self.0.homspace().id().hash(state);
        self.0.structure().content_id().hash(state);
        self.0.nout().hash(state);
        self.0.nin().hash(state);
    }
}

impl<R> Clone for BoundDynamicFusionMapSpace<R> {
    fn clone(&self) -> Self {
        Self {
            space: self.space.clone(),
            provider: Arc::clone(&self.provider),
            layout_build: self.layout_build,
        }
    }
}

impl<R> fmt::Debug for BoundDynamicFusionMapSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoundDynamicFusionMapSpace")
            .field("space", &self.space)
            .field("provider_type", &std::any::type_name::<R>())
            .finish_non_exhaustive()
    }
}

impl<R> BoundDynamicFusionMapSpace<R>
where
    R: FusionRule,
{
    fn from_derived_with_capability(
        provider: Arc<R>,
        space: DynamicFusionMapSpace,
        layout_build: LayoutBuildCapability<R>,
    ) -> Result<Self, OperationError> {
        // Why not enumerate the tree grid again: callers in this crate create
        // `space` through checked structural operations from an already-bound
        // source. Re-enumeration would duplicate that work without adding a
        // new trust boundary; the rule identity remains cheap to verify.
        validate_bound_space_invariants(&space)?;
        space.validate_rule(provider.as_ref())?;
        Ok(Self {
            space,
            provider,
            layout_build,
        })
    }

    pub(crate) fn from_derived(
        provider: Arc<R>,
        space: DynamicFusionMapSpace,
    ) -> Result<Self, OperationError> {
        Self::from_derived_with_capability(provider, space, LayoutBuildCapability::encoded())
    }

    pub(crate) fn from_derived_like(
        source: &Self,
        space: DynamicFusionMapSpace,
    ) -> Result<Self, OperationError> {
        Self::from_derived_with_capability(Arc::clone(&source.provider), space, source.layout_build)
    }

    /// Builds and binds a multiplicity-free space in one checked pass.
    pub fn from_degeneracy_shapes<Shapes>(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let space =
            DynamicFusionMapSpace::from_degeneracy_shapes(provider.as_ref(), homspace, shapes)?;
        Self::from_derived(provider, space)
    }

    /// Builds the ordinary TeNeT multiplicity-free root with lowered metadata.
    #[doc(hidden)]
    pub fn from_degeneracy_shapes_lowered<Shapes>(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + LoweredMultiplicityFreeAlgebra,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let layout_build = LayoutBuildCapability::lowered();
        layout_build.prime(provider.as_ref(), &homspace)?;
        let space =
            DynamicFusionMapSpace::from_degeneracy_shapes(provider.as_ref(), homspace, shapes)?;
        Self::from_derived_with_capability(provider, space, layout_build)
    }

    /// Builds and binds a multiplicity-aware space in one checked pass.
    pub fn from_degeneracy_shapes_generic<Shapes>(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let space = DynamicFusionMapSpace::from_degeneracy_shapes_generic(
            provider.as_ref(),
            homspace,
            shapes,
        )?;
        Self::from_derived(provider, space)
    }

    fn bind_with_keys(
        space: DynamicFusionMapSpace,
        provider: Arc<R>,
        keys: Vec<FusionTreeBlockKey>,
    ) -> Result<Self, OperationError> {
        validate_bound_space_invariants(&space)?;
        space.validate_rule(provider.as_ref())?;
        space.validate_complete_tree_grid(&keys)?;
        Ok(Self {
            space,
            provider,
            layout_build: LayoutBuildCapability::encoded(),
        })
    }

    /// Binds a space using multiplicity-free tree enumeration.
    pub fn bind_multiplicity_free(
        space: DynamicFusionMapSpace,
        provider: Arc<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let keys = space
            .homspace()
            .fusion_tree_keys(provider.as_ref())
            .to_vec();
        Self::bind_with_keys(space, provider, keys)
    }

    /// Builds a checked contraction result while retaining the exact provider
    /// allocation shared by both operands.
    pub fn contracted_multiplicity_free(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        if lhs.provider.rule_identity() != rhs.provider.rule_identity() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch {
                    expected: lhs.provider.rule_identity(),
                    actual: rhs.provider.rule_identity(),
                },
            ));
        }
        // The lhs is the authority for a result of two independently-built but
        // semantically identical tensors. Why not require Arc::ptr_eq: public
        // tensors may own distinct provider allocations with one RuleIdentity.
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let space = DynamicFusionMapSpace::contracted_with_spec_and_primer(
            lhs.provider.as_ref(),
            &lhs.space,
            &rhs.space,
            axes,
            lhs.layout_build.prime,
        )?;
        Self::from_derived_like(lhs, space)
    }

    #[doc(hidden)]
    pub fn contracted_multiplicity_free_lowered(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + LoweredMultiplicityFreeAlgebra,
    {
        Self::contracted_multiplicity_free(lhs, rhs, lhs_axes, rhs_axes)
    }

    /// Builds a checked contraction result directly in the requested output
    /// order while retaining the exact lhs provider allocation.
    ///
    /// Why not accept [`TensorContractSpec`]: conjugation flags belong to the
    /// numerical execution plan after categorical adjoints have been lowered.
    /// Destination metadata is derived from the already-visible bound spaces.
    pub fn contracted_multiplicity_free_ordered(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Self::validate_shared_provider(lhs, rhs)?;
        let axes = TensorContractSpec::new(lhs_axes, rhs_axes, output_order);
        let space = DynamicFusionMapSpace::contracted_with_spec_and_primer(
            lhs.provider.as_ref(),
            &lhs.space,
            &rhs.space,
            axes,
            lhs.layout_build.prime,
        )?;
        Self::from_derived_like(lhs, space)
    }

    #[doc(hidden)]
    pub fn contracted_multiplicity_free_ordered_lowered(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + LoweredMultiplicityFreeAlgebra,
    {
        Self::contracted_multiplicity_free_ordered(lhs, rhs, lhs_axes, rhs_axes, output_order)
    }

    /// Validates contraction compatibility without building a coupled result
    /// layout. Used to retain historical contraction-before-pAB error order.
    pub fn validate_contracted_homspace_multiplicity_free(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Self::validate_shared_provider(lhs, rhs)?;
        DynamicFusionMapSpace::validate_contracted_homspace(
            lhs.provider.as_ref(),
            &lhs.space,
            &rhs.space,
            lhs_axes,
            rhs_axes,
        )
    }

    fn validate_shared_provider(lhs: &Self, rhs: &Self) -> Result<(), OperationError> {
        if lhs.provider.rule_identity() != rhs.provider.rule_identity() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch {
                    expected: lhs.provider.rule_identity(),
                    actual: rhs.provider.rule_identity(),
                },
            ));
        }
        Ok(())
    }

    /// Builds a bound space from the final HomSpace's stored leg
    /// degeneracies without materializing per-tree shape scratch.
    pub fn from_final_homspace_multiplicity_free(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let space = DynamicFusionMapSpace::from_final_homspace(provider.as_ref(), homspace)?;
        Self::from_derived(provider, space)
    }

    #[doc(hidden)]
    pub fn from_final_homspace_multiplicity_free_lowered(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + LoweredMultiplicityFreeAlgebra,
    {
        let layout_build = LayoutBuildCapability::lowered();
        let space = DynamicFusionMapSpace::from_final_homspace_with_primer(
            provider.as_ref(),
            homspace,
            layout_build.prime,
        )?;
        Self::from_derived_with_capability(provider, space, layout_build)
    }

    /// Builds a multiplicity-aware contraction result and normalizes its
    /// authority to the lhs provider allocation.
    pub fn contracted_generic(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError> {
        if lhs.provider.rule_identity() != rhs.provider.rule_identity() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch {
                    expected: lhs.provider.rule_identity(),
                    actual: rhs.provider.rule_identity(),
                },
            ));
        }
        let space = DynamicFusionMapSpace::contracted_generic(
            lhs.provider.as_ref(),
            &lhs.space,
            &rhs.space,
            lhs_axes,
            rhs_axes,
        )?;
        Self::from_derived_like(lhs, space)
    }

    /// Generic sibling of [`Self::from_final_homspace_multiplicity_free`].
    pub fn from_final_homspace_generic(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError> {
        let space =
            DynamicFusionMapSpace::from_final_homspace_generic(provider.as_ref(), homspace)?;
        Self::from_derived(provider, space)
    }

    /// Tree-transform result retaining the source provider proof.
    pub fn transformed_multiplicity_free(
        &self,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let space = self.space.transformed_with_primer(
            self.provider.as_ref(),
            operation,
            self.layout_build.prime,
        )?;
        Self::from_derived_like(self, space)
    }

    #[doc(hidden)]
    pub fn transformed_multiplicity_free_lowered(
        &self,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + LoweredMultiplicityFreeAlgebra,
    {
        self.transformed_multiplicity_free(operation)
    }

    /// Generic tree-transform result retaining the source provider proof.
    pub fn transformed_generic(
        &self,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError> {
        let space = self
            .space
            .transformed_generic(self.provider.as_ref(), operation)?;
        Self::from_derived_like(self, space)
    }

    /// Creates the zero-copy adjoint replay view while retaining the exact
    /// provider allocation of the source binding.
    pub fn adjoint_view(&self) -> Result<Self, OperationError> {
        let space = self.space.adjoint_view()?;
        Self::from_derived_like(self, space)
    }

    /// Binds a space using multiplicity-aware generic tree enumeration.
    pub fn bind_generic(
        space: DynamicFusionMapSpace,
        provider: Arc<R>,
    ) -> Result<Self, OperationError> {
        let keys = space
            .homspace()
            .fusion_tree_keys_generic(provider.as_ref())
            .map_err(OperationError::from_core_preserving_context)?;
        Self::bind_with_keys(space, provider, keys)
    }

    #[inline]
    /// Read-only access to the validated dynamic layout for expert planning
    /// and diagnostics. The provider remains attached to this binding.
    pub fn space(&self) -> &DynamicFusionMapSpace {
        &self.space
    }

    #[inline]
    pub fn provider(&self) -> &R {
        self.provider.as_ref()
    }

    #[inline]
    pub fn provider_arc(&self) -> &Arc<R> {
        &self.provider
    }

    pub(crate) fn layout_primer(&self) -> LayoutPrimer<R> {
        self.layout_build.prime
    }

    /// Primes a derived HomSpace with this binding's opaque build strategy.
    #[doc(hidden)]
    pub fn prime_derived_homspace(
        &self,
        homspace: &FusionTreeHomSpace,
    ) -> Result<(), OperationError> {
        self.layout_build.prime(self.provider.as_ref(), homspace)
    }

    /// Reports whether two bindings derive layouts through the same opaque strategy.
    #[doc(hidden)]
    pub fn has_same_layout_build_strategy(&self, other: &Self) -> bool {
        self.layout_build.same_as(other.layout_build)
    }

    /// Builds a derived layout while preserving this binding's build strategy.
    #[doc(hidden)]
    pub fn derive_from_degeneracy_shapes<Shapes>(
        &self,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        self.layout_build.prime(self.provider.as_ref(), &homspace)?;
        let space = DynamicFusionMapSpace::from_degeneracy_shapes(
            self.provider.as_ref(),
            homspace,
            shapes,
        )?;
        Self::from_derived_like(self, space)
    }

    /// Builds a final derived layout from the HomSpace's leg degeneracies.
    #[doc(hidden)]
    pub fn derive_from_final_homspace(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let space = DynamicFusionMapSpace::from_final_homspace_with_primer(
            self.provider.as_ref(),
            homspace,
            self.layout_build.prime,
        )?;
        Self::from_derived_like(self, space)
    }

    /// Erases only the provider allocation after preserving the checked layout proof.
    ///
    /// Why not return [`DynamicFusionMapSpace`]: a raw value does not carry the
    /// complete-tree-grid proof established by the bound constructor.
    pub fn validated_layout(&self) -> ValidatedDynamicFusionLayout {
        ValidatedDynamicFusionLayout(self.space.clone())
    }

    /// Rebinds a validated cached layout to this space's exact provider allocation.
    ///
    /// Why not retain the provider that first populated a process-global cache:
    /// semantically equal callers may carry distinct provider allocations.
    pub fn rebind_validated(
        &self,
        layout: &ValidatedDynamicFusionLayout,
    ) -> Result<Self, OperationError> {
        let expected = self.provider.rule_identity();
        let actual = layout.0.rule_identity.clone().ok_or_else(|| {
            OperationError::from_core_preserving_context(CoreError::MissingFusionRuleIdentity)
        })?;
        if expected != actual {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch { expected, actual },
            ));
        }
        Ok(Self {
            space: layout.0.clone(),
            provider: Arc::clone(&self.provider),
            layout_build: self.layout_build,
        })
    }
}

impl DynamicFusionMapSpace {
    fn from_final_homspace<R>(
        rule: &R,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Self::from_final_homspace_with_primer(rule, homspace, encoded_layout_primer::<R>)
    }

    fn from_final_homspace_with_primer<R>(
        rule: &R,
        homspace: FusionTreeHomSpace,
        primer: LayoutPrimer<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        observe_final_result_layout_build();
        primer(rule, &homspace)?;
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let subblock_structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            rule_identity: Some(rule.rule_identity()),
        })
    }

    fn from_final_homspace_generic<R>(
        rule: &R,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: FusionRule,
    {
        observe_final_result_layout_build();
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let subblock_structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies_generic(rule)
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            rule_identity: Some(rule.rule_identity()),
        })
    }

    fn validate_complete_tree_grid(
        &self,
        keys: &[FusionTreeBlockKey],
    ) -> Result<(), OperationError> {
        let structure = self.structure();
        if structure.block_count() != keys.len() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::BlockCountMismatch {
                    expected: keys.len(),
                    actual: structure.block_count(),
                },
            ));
        }
        let mut shapes = Vec::with_capacity(keys.len());
        for key in keys {
            let block_index = structure
                .find_block_index_by_key(&BlockKey::FusionTree(key.clone()))
                .ok_or_else(|| {
                    OperationError::from_core_preserving_context(CoreError::MissingBlockKey {
                        key: Box::new(BlockKey::FusionTree(key.clone())),
                    })
                })?;
            let block = structure
                .block(block_index)
                .map_err(OperationError::from_core_preserving_context)?;
            shapes.push(block.shape().to_vec());
        }
        self.homspace
            .validate_degeneracy_shapes(keys, &shapes)
            .map_err(OperationError::from_core_preserving_context)
    }

    /// Rank-erases a typed fusion space (shares the hom space and subblock
    /// structure handles; no data copies).
    pub fn from_typed<const NOUT: usize, const NIN: usize>(
        space: &FusionTensorMapSpace<NOUT, NIN>,
    ) -> Self {
        Self {
            nout: NOUT,
            nin: NIN,
            homspace: Arc::clone(space.homspace_arc()),
            subblock_structure: Arc::clone(space.subblock_structure()),
            rule_identity: space.rule_identity(),
        }
    }

    /// Builds a dynamic space directly from an untyped description: a hom
    /// space plus one degeneracy shape per fusion-tree key (in
    /// [`FusionTreeHomSpace::fusion_tree_keys`] order). The storage layout is
    /// the TensorKit-equivalent coupled-sector matrix layout, identical to
    /// [`FusionTensorMapSpace::from_degeneracy_shapes`].
    pub fn from_degeneracy_shapes<R, Shapes>(
        rule: &R,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let rank = nout + nin;
        let shapes = shapes
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Vec<usize>>>();
        let cache_key = ScratchStructureKey {
            layout: LayoutId {
                rule: rule.rule_identity(),
                homspace: homspace.id(),
            },
            nout,
            rank,
            shapes: shapes.iter().map(|s| s.iter().copied().collect()).collect(),
        };
        let cache = scratch_structure_cache();
        if let Ok(mut guard) = cache.write() {
            if let Some(subblock_structure) = guard.entries.get(&cache_key).cloned() {
                return Ok(Self {
                    nout,
                    nin,
                    homspace: Arc::new(homspace),
                    subblock_structure,
                    rule_identity: Some(rule.rule_identity()),
                });
            }
        }
        let keys = homspace.fusion_tree_keys(rule);
        if keys.len() != shapes.len() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::BlockCountMismatch {
                    expected: keys.len(),
                    actual: shapes.len(),
                },
            ));
        }
        homspace
            .validate_degeneracy_shapes(&keys, &shapes)
            .map_err(OperationError::from_core_preserving_context)?;
        let blocks = keys
            .iter()
            .cloned()
            .map(BlockKey::from)
            .zip(shapes)
            .collect::<Vec<_>>();
        let built = Arc::new(scratch_subblock_structure(rule, nout, rank, blocks)?);
        let subblock_structure = if let Ok(mut guard) = cache.write() {
            if let Some(existing) = guard.entries.get(&cache_key).cloned() {
                existing
            } else {
                guard.entries.put(cache_key, Arc::clone(&built));
                built
            }
        } else {
            built
        };
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            rule_identity: Some(rule.rule_identity()),
        })
    }

    pub(crate) fn transformed_from_typed<R, const NOUT: usize, const NIN: usize>(
        rule: &R,
        source: &FusionTensorMapSpace<NOUT, NIN>,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Self::from_typed(source).transformed(rule, operation)
    }

    /// Space of the tree-transformed (permute / braid / transpose) tensor:
    /// the hom space is permuted and the full tree set of the result is
    /// enumerated (trees the transform coefficients never reach stay as
    /// structural zeros, keeping every coupled sector grid complete).
    pub(crate) fn transformed<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        self.transformed_with_primer(rule, operation, encoded_layout_primer::<R>)
    }

    pub(crate) fn transformed_with_primer<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
        primer: LayoutPrimer<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        self.validate_rule(rule)?;
        let source = self;
        let cache_key = TransformedSpaceKey {
            rule: rule.rule_identity(),
            source_homspace: Arc::clone(&source.homspace),
            operation: operation.clone(),
        };
        if let Some(cached) = transformed_space_cache()
            .read()
            .ok()
            .and_then(|map| map.get(&cache_key).cloned())
        {
            return Ok(cached);
        }
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let nout = codomain_axes.len();
        let nin = domain_axes.len();
        let homspace = source
            .homspace()
            .permute(rule, codomain_axes, domain_axes)
            .map_err(OperationError::from_core_preserving_context)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        // Why not rebuild external source legs and per-tree shape vectors:
        // #256 already carried the authoritative degeneracies into the final
        // HomSpace. The miss builder consumes that value directly.
        let space = Self::from_final_homspace_with_primer(rule, homspace, primer)?;
        if let Ok(mut map) = transformed_space_cache().write() {
            map.insert(cache_key, space.clone());
        }
        Ok(space)
    }

    /// Generic-fusion (SU(3)) sibling of [`Self::from_degeneracy_shapes`] for
    /// caller-supplied per-tree shapes. Derived transform/contraction results
    /// instead use the final HomSpace's stored leg degeneracies directly.
    pub fn from_degeneracy_shapes_generic<R, Shapes>(
        rule: &R,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: FusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let keys = homspace
            .fusion_tree_keys_generic(rule)
            .map_err(OperationError::from_core_preserving_context)?;
        let shapes = shapes.into_iter().map(Into::into).collect::<Vec<_>>();
        if keys.len() != shapes.len() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::BlockCountMismatch {
                    expected: keys.len(),
                    actual: shapes.len(),
                },
            ));
        }
        homspace
            .validate_degeneracy_shapes(&keys, &shapes)
            .map_err(OperationError::from_core_preserving_context)?;
        let blocks = keys
            .iter()
            .cloned()
            .map(BlockKey::from)
            .zip(shapes)
            .collect::<Vec<_>>();
        let subblock_structure =
            Arc::new(scratch_subblock_structure(rule, nout, nout + nin, blocks)?);
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            rule_identity: Some(rule.rule_identity()),
        })
    }

    /// Generic-fusion (SU(3)) sibling of [`Self::transformed`]: the permuted /
    /// braided / transposed result space, enumerated with multiplicity-aware
    /// keys. Not cached (the Generic path is not on a hot loop yet — same
    /// non-memoized rationale as the Stage B3b cache siblings).
    pub(crate) fn transformed_generic<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: FusionRule,
    {
        self.validate_rule(rule)?;
        let source = self;
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let nout = codomain_axes.len();
        let nin = domain_axes.len();
        let homspace = source
            .homspace()
            .permute(rule, codomain_axes, domain_axes)
            .map_err(OperationError::from_core_preserving_context)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        Self::from_final_homspace_generic(rule, homspace)
    }

    /// Space of the contraction result in the default output order (`lhs`
    /// open axes ascending on the codomain side, `rhs` open axes ascending on
    /// the domain side). Mirrors the destination TensorKit's
    /// `tensorcontract!` with default `pAB` writes into.
    #[cfg(test)]
    pub(crate) fn contracted<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        Self::contracted_with_spec(rule, lhs, rhs, axes)
    }

    #[cfg(test)]
    fn contracted_with_spec<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Self::contracted_with_spec_and_primer(rule, lhs, rhs, axes, encoded_layout_primer::<R>)
    }

    fn contracted_with_spec_and_primer<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        primer: LayoutPrimer<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        lhs.validate_rule(rule)?;
        rhs.validate_rule(rule)?;
        if axes.lhs_contracting_axes().len() != axes.rhs_contracting_axes().len() {
            return Err(OperationError::ContractAxisCountMismatch {
                lhs: axes.lhs_contracting_axes().len(),
                rhs: axes.rhs_contracting_axes().len(),
            });
        }
        let nout = lhs
            .rank()
            .checked_sub(axes.lhs_contracting_axes().len())
            .ok_or(OperationError::RankMismatch {
                expected: axes.lhs_contracting_axes().len(),
                actual: lhs.rank(),
            })?;
        let nin = rhs
            .rank()
            .checked_sub(axes.rhs_contracting_axes().len())
            .ok_or(OperationError::RankMismatch {
                expected: axes.rhs_contracting_axes().len(),
                actual: rhs.rank(),
            })?;
        let output_axes = match axes.output_permutation() {
            tenet_operations::OutputAxisOrder::Identity => (0..nout + nin).collect(),
            tenet_operations::OutputAxisOrder::Axes(output_axes) => {
                output_axes.iter().copied().collect()
            }
        };
        let key = ContractedSpaceKey {
            rule: rule.rule_identity(),
            lhs_homspace: Arc::clone(&lhs.homspace),
            rhs_homspace: Arc::clone(&rhs.homspace),
            lhs_axes: axes.lhs_contracting_axes().iter().copied().collect(),
            rhs_axes: axes.rhs_contracting_axes().iter().copied().collect(),
            output_axes,
        };
        if let Some(cached) = contracted_space_cache()
            .read()
            .ok()
            .and_then(|map| map.get(&key).cloned())
        {
            return Ok(cached);
        }
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        debug_assert_eq!(key.output_axes.as_slice(), axis_plan.output_axes);
        let space =
            Self::contracted_space_from_plan(rule, lhs, rhs, axes, &axis_plan, nout, nin, primer)?;
        if let Ok(mut map) = contracted_space_cache().write() {
            map.insert(key, space.clone());
        }
        Ok(space)
    }

    fn validate_contracted_homspace<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        lhs.validate_rule(rule)?;
        rhs.validate_rule(rule)?;
        let nout = lhs
            .rank()
            .checked_sub(lhs_axes.len())
            .ok_or(OperationError::RankMismatch {
                expected: lhs_axes.len(),
                actual: lhs.rank(),
            })?;
        let nin = rhs
            .rank()
            .checked_sub(rhs_axes.len())
            .ok_or(OperationError::RankMismatch {
                expected: rhs_axes.len(),
                actual: rhs.rank(),
            })?;
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        Self::contracted_homspace_from_plan(rule, lhs, rhs, axes, &axis_plan, nout)?;
        Ok(())
    }

    pub(crate) fn core_dst<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        plan: &FusionContractPlan,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Self::core_dst_with_primer(rule, lhs, rhs, plan, encoded_layout_primer::<R>)
    }

    pub(crate) fn core_dst_with_primer<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        plan: &FusionContractPlan,
        primer: LayoutPrimer<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let nout = plan.core_dst_open_lhs_rank();
        let nin = plan.core_dst_open_rhs_rank();
        Self::contracted_space(
            rule,
            lhs,
            rhs,
            plan.core_axes().as_spec(),
            nout,
            nin,
            primer,
        )
    }

    fn contracted_space<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        nout: usize,
        nin: usize,
        primer: LayoutPrimer<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        Self::contracted_space_from_plan(rule, lhs, rhs, axes, &axis_plan, nout, nin, primer)
    }

    fn contracted_space_from_plan<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        axis_plan: &TensorContractAxisPlan,
        nout: usize,
        nin: usize,
        primer: LayoutPrimer<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let homspace = Self::contracted_homspace_from_plan(rule, lhs, rhs, axes, axis_plan, nout)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        Self::from_final_homspace_with_primer(rule, homspace, primer)
    }

    fn contracted_homspace_from_plan<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        axis_plan: &TensorContractAxisPlan,
        nout: usize,
    ) -> Result<FusionTreeHomSpace, OperationError>
    where
        R: FusionRule,
    {
        FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.homspace(),
            rhs.homspace(),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &axis_plan.output_axes,
            nout,
        )
        .map_err(OperationError::from_core_preserving_context)
    }

    /// Generic-fusion (Stage B3c-1) sibling of [`Self::contracted`]: the
    /// contraction result space for an outer-multiplicity rule, enumerated with
    /// multiplicity-aware fusion-tree keys (`fusion_tree_keys_generic`). Not
    /// cached (the Generic path is not on a hot loop yet — same non-memoized
    /// rationale as the B3b transform siblings). The final HomSpace is consumed
    /// directly by the multiplicity-aware single-pass layout builder.
    pub(crate) fn contracted_generic<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: FusionRule,
    {
        lhs.validate_rule(rule)?;
        rhs.validate_rule(rule)?;
        if lhs_axes.len() != rhs_axes.len() {
            return Err(OperationError::ContractAxisCountMismatch {
                lhs: lhs_axes.len(),
                rhs: rhs_axes.len(),
            });
        }
        let nout = lhs
            .rank()
            .checked_sub(lhs_axes.len())
            .ok_or(OperationError::RankMismatch {
                expected: lhs_axes.len(),
                actual: lhs.rank(),
            })?;
        let nin = rhs
            .rank()
            .checked_sub(rhs_axes.len())
            .ok_or(OperationError::RankMismatch {
                expected: rhs_axes.len(),
                actual: rhs.rank(),
            })?;
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        let homspace = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.homspace(),
            rhs.homspace(),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &axis_plan.output_axes,
            nout,
        )
        .map_err(OperationError::from_core_preserving_context)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        Self::from_final_homspace_generic(rule, homspace)
    }

    /// Adjoint view: codomain and domain swap (spaces and per-block shapes),
    /// no data movement implied. The block layout is a strided view into the
    /// source layout, so this space is for replay bookkeeping, not for
    /// allocating fresh coupled storage.
    pub(crate) fn adjoint_view(&self) -> Result<Self, OperationError> {
        let homspace = FusionTreeHomSpace::new(
            self.homspace.domain().clone(),
            self.homspace.codomain().clone(),
        );
        let structure = crate::lowering::adjoint_block_structure_view(
            self.nout,
            self.nin,
            &self.subblock_structure,
        )?;
        Ok(Self {
            nout: self.nin,
            nin: self.nout,
            homspace: Arc::new(homspace),
            subblock_structure: Arc::new(structure),
            rule_identity: self.rule_identity.clone(),
        })
    }

    /// Number of codomain legs.
    #[inline]
    pub fn nout(&self) -> usize {
        self.nout
    }

    /// Number of domain legs.
    #[inline]
    pub fn nin(&self) -> usize {
        self.nin
    }

    /// Total number of legs.
    #[inline]
    pub fn rank(&self) -> usize {
        self.nout + self.nin
    }

    pub(crate) fn validate_rule<R: FusionRule>(&self, rule: &R) -> Result<(), OperationError> {
        match self.rule_identity.as_ref() {
            Some(expected) if expected != &rule.rule_identity() => Err(
                OperationError::from_core_preserving_context(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual: rule.rule_identity(),
                }),
            ),
            Some(_) => Ok(()),
            None => Err(OperationError::from_core_preserving_context(
                CoreError::MissingFusionRuleIdentity,
            )),
        }
    }

    #[inline]
    pub fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    /// Shared hom-space handle for pointer-identity fast paths in replay
    /// caches.
    pub fn homspace_arc(&self) -> &Arc<FusionTreeHomSpace> {
        &self.homspace
    }

    /// Subblock (fusion-tree) block structure of the coupled storage layout.
    #[inline]
    pub fn structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    /// Flat storage length this space requires.
    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }
}

fn tree_transform_operation_axes(operation: &TreeTransformOperation) -> (&[usize], &[usize]) {
    match operation {
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            ..
        } => (
            codomain_permutation.as_slice(),
            domain_permutation.as_slice(),
        ),
    }
}

#[cfg(test)]
mod bound_invariant_tests {
    use super::*;
    use tenet_core::{BlockSpec, Z2FusionRule};

    fn matrix_space() -> DynamicFusionMapSpace {
        let rule = Z2FusionRule;
        let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace, [vec![1, 1]]).unwrap()
    }

    #[test]
    fn bound_space_rejects_incoherent_axis_split() {
        // What: a bound space cannot disagree with its hom-space codomain/domain split.
        let raw = DynamicFusionMapSpace {
            nout: 0,
            nin: 2,
            ..matrix_space()
        };

        let error = BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::new(Z2FusionRule))
            .unwrap_err();

        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionSpaceSplitMismatch { .. })
        ));
    }

    #[test]
    fn bound_space_rejects_incoherent_structure_rank() {
        // What: a bound space cannot attach storage with a rank different from its hom space.
        let raw = matrix_space();
        let block = raw.structure().block(0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(
            1,
            vec![BlockSpec::with_key(block.key().clone(), vec![1], vec![1], 0).unwrap()],
        )
        .unwrap();
        let raw = DynamicFusionMapSpace {
            subblock_structure: Arc::new(structure),
            ..raw
        };

        let error = BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::new(Z2FusionRule))
            .unwrap_err();

        assert!(matches!(
            error,
            OperationError::Core(CoreError::StructureRankMismatch { .. })
        ));
    }

    #[test]
    fn direct_bound_builders_keep_coherent_split_and_rank() {
        // What: both multiplicity-free and generic direct builders satisfy bound invariants.
        let provider = Arc::new(Z2FusionRule);
        let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
        let multiplicity_free = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            homspace.clone(),
            [vec![1, 1]],
        )
        .unwrap();
        let generic = BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(
            provider,
            homspace,
            [vec![1, 1]],
        )
        .unwrap();

        assert_eq!(multiplicity_free.space().nout(), 1);
        assert_eq!(generic.space().nin(), 1);
        assert_eq!(multiplicity_free.space().structure().rank(), 2);
        assert_eq!(generic.space().structure().rank(), 2);
    }
}

#[cfg(test)]
mod lowered_metadata_tests {
    use super::*;
    use crate::test_support::CACHE_TEST_LOCK;
    use std::cell::Cell;
    use tenet_core::{
        FermionParityFusionRule, FusionProductSpace, Fz2SectorLayout, PackedProductCodec,
        ProductFusionRule, ProductSectorCodec, ProductSectorLayout, SU2FusionRule, SU2Irrep,
        SectorId, SectorLeg, Su2SectorLayout, U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
    };

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

    thread_local! {
        static PRIMER_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    fn rule() -> TripleRule {
        TripleRule::new(
            Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
            SU2FusionRule,
        )
    }

    fn sector(parity: usize, charge: i32, twice_spin: usize) -> SectorId {
        TripleCodec::encode(
            Fz2U1Codec::encode(
                Z2Irrep::new(parity as u8).sector_id(),
                U1Irrep::new(charge).sector_id(),
            ),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    }

    fn homspace() -> FusionTreeHomSpace {
        let vacuum = sector(0, 0, 0);
        let charged = sector(1, 1, 1);
        let leg = |dual| SectorLeg::new([(vacuum, 1), (charged, 1)], dual);
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false), leg(true)]),
            FusionProductSpace::new([leg(true), leg(false)]),
        )
    }

    fn counting_primer(
        rule: &TripleRule,
        homspace: &FusionTreeHomSpace,
    ) -> Result<(), OperationError> {
        PRIMER_CALLS.with(|calls| calls.set(calls.get() + 1));
        lowered_layout_primer(rule, homspace)
    }

    fn reset_primer_calls() {
        PRIMER_CALLS.with(|calls| calls.set(0));
    }

    fn primer_calls() -> usize {
        PRIMER_CALLS.with(Cell::get)
    }

    fn source(rule: &TripleRule) -> DynamicFusionMapSpace {
        let homspace = homspace();
        lowered_layout_primer(rule, &homspace).unwrap();
        let count = homspace.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, homspace, vec![vec![1; 4]; count])
            .unwrap()
    }

    #[test]
    fn lowered_metadata_routes_prime_only_after_operation_cache_misses() {
        // What: final, transform, and ordered contraction metadata enter the
        // lowered primer once on a cold miss and skip it on their warm cache hit.
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let rule = rule();
        let source = source(&rule);

        reset_primer_calls();
        let final_space = DynamicFusionMapSpace::from_final_homspace_with_primer(
            &rule,
            homspace(),
            counting_primer,
        )
        .unwrap();
        assert_eq!(primer_calls(), 1);
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let encoded_final = DynamicFusionMapSpace::from_final_homspace(&rule, homspace()).unwrap();
        assert_eq!(final_space, encoded_final);

        crate::reset_global_operation_caches();
        reset_primer_calls();
        let operation = TreeTransformOperation::permute([1, 0], [3, 2]);
        let transformed = source
            .transformed_with_primer(&rule, &operation, counting_primer)
            .unwrap();
        let warm = source
            .transformed_with_primer(&rule, &operation, counting_primer)
            .unwrap();
        assert_eq!(primer_calls(), 1);
        assert_eq!(transformed, warm);
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let encoded_transformed = source.transformed(&rule, &operation).unwrap();
        assert_eq!(transformed, encoded_transformed);

        crate::reset_global_operation_caches();
        reset_primer_calls();
        let axes = TensorContractSpec::new(
            &[],
            &[],
            OutputAxisOrder::from_axes(&[1, 0, 2, 3, 4, 5, 6, 7]),
        );
        let contracted = DynamicFusionMapSpace::contracted_with_spec_and_primer(
            &rule,
            &source,
            &source,
            axes,
            counting_primer,
        )
        .unwrap();
        let warm = DynamicFusionMapSpace::contracted_with_spec_and_primer(
            &rule,
            &source,
            &source,
            axes,
            counting_primer,
        )
        .unwrap();
        assert_eq!(primer_calls(), 1);
        assert_eq!(contracted, warm);
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let encoded_contracted =
            DynamicFusionMapSpace::contracted_with_spec(&rule, &source, &source, axes).unwrap();
        assert_eq!(contracted, encoded_contracted);
    }

    #[test]
    fn lowered_metadata_error_maps_to_operation_invalid_argument() {
        // What: malformed built-in product IDs cross into the operation layer
        // as its structured invalid-argument variant and never panic.
        let malformed = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(usize::MAX), 1)], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        );
        let error = lowered_layout_primer(&rule(), &malformed).unwrap_err();
        assert!(matches!(error, OperationError::InvalidArgument { .. }));
    }
}

#[cfg(test)]
mod scratch_cache_tests {
    use super::*;
    use crate::test_support::CACHE_TEST_LOCK;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tenet_core::{
        BlockSpec, BraidingStyleKind, FusionProductSpace, FusionStyleKind,
        MultiplicityFreeFusionSymbols, RuleIdentity, SectorId, SectorLeg, SectorVec, U1FusionRule,
        U1Irrep, Z2FusionRule, Z2Irrep,
    };

    #[derive(Clone)]
    struct CountingRule {
        identity: RuleIdentity,
        calls: Arc<AtomicUsize>,
    }

    impl CountingRule {
        fn new() -> Self {
            Self {
                identity: RuleIdentity::new_unique::<Self>(),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl FusionRule for CountingRule {
        fn rule_identity(&self) -> RuleIdentity {
            self.identity.clone()
        }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }
        fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
            self.calls.fetch_add(1, Ordering::Relaxed);
            [SectorId::new(0)].into_iter().collect()
        }
    }

    impl tenet_core::MultiplicityFreeFusionRule for CountingRule {}
    impl MultiplicityFreeFusionSymbols for CountingRule {
        type Scalar = f64;
        fn scalar_one(&self) -> f64 {
            1.0
        }
        fn scalar_conj(&self, value: f64) -> f64 {
            value
        }
        fn f_symbol_scalar(
            &self,
            _: SectorId,
            _: SectorId,
            _: SectorId,
            _: SectorId,
            _: SectorId,
            _: SectorId,
        ) -> f64 {
            1.0
        }
        fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
            1.0
        }
    }
    impl MultiplicityFreeRigidSymbols for CountingRule {
        fn dim_scalar(&self, _: SectorId) -> f64 {
            1.0
        }
        fn inv_dim_scalar(&self, _: SectorId) -> f64 {
            1.0
        }
        fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
            1.0
        }
        fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
            1.0
        }
        fn twist_scalar(&self, _: SectorId) -> f64 {
            1.0
        }
        fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
            1.0
        }
    }

    #[test]
    fn raw_transform_rejects_a_different_rule_identity() {
        // What: crate-internal derivation still rejects a provider other than
        // the one recorded by the source, even though public callers can only
        // reach this operation through BoundDynamicFusionMapSpace.
        let first = CountingRule::new();
        let second = CountingRule::new();
        let space = DynamicFusionMapSpace::from_degeneracy_shapes(
            &first,
            FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
            [vec![1]],
        )
        .unwrap();

        let error = space
            .transformed(&second, &TreeTransformOperation::permute([0], []))
            .unwrap_err();
        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
    }

    // Single-charge U(1) source in a chosen bond dimension (the last leg of each
    // block shape); one coupled sector, block shape [deg, deg].
    fn u1_space(charge: i32, deg: usize) -> DynamicFusionMapSpace {
        let rule = U1FusionRule;
        let sid = U1Irrep::new(charge).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(sid, deg)], false)]);
        let hom = FusionTreeHomSpace::new(leg(), leg());
        let count = hom.fusion_tree_keys(&rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, hom, vec![vec![deg, deg]; count])
            .unwrap()
    }

    fn reset_final_result_layout_test_state() {
        transformed_space_cache()
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        contracted_space_cache()
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        reset_final_result_layout_builds();
    }

    #[test]
    fn transformed_cache_miss_builds_once_and_warm_hit_builds_zero() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_final_result_layout_test_state();
        let rule = U1FusionRule;
        let source = u1_space(301, 3);
        let operation = TreeTransformOperation::permute([1], [0]);
        let cache_key = TransformedSpaceKey {
            rule: rule.rule_identity(),
            source_homspace: Arc::clone(&source.homspace),
            operation: operation.clone(),
        };

        let first = source.transformed(&rule, &operation).unwrap();
        assert_eq!(final_result_layout_builds(), 1);
        assert!(transformed_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&cache_key));

        reset_final_result_layout_builds();
        let second = source.transformed(&rule, &operation).unwrap();
        assert_eq!(final_result_layout_builds(), 0);
        assert!(transformed_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&cache_key));
        assert!(Arc::ptr_eq(first.structure(), second.structure()));
    }

    #[test]
    fn contracted_cache_miss_builds_once_and_warm_hit_builds_zero() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_final_result_layout_test_state();
        let rule = U1FusionRule;
        let source = u1_space(901, 4);
        let cache_key = ContractedSpaceKey {
            rule: rule.rule_identity(),
            lhs_homspace: Arc::clone(&source.homspace),
            rhs_homspace: Arc::clone(&source.homspace),
            lhs_axes: DimVec::from_slice(&[1]),
            rhs_axes: DimVec::from_slice(&[0]),
            output_axes: DimVec::from_slice(&[0, 1]),
        };

        let first = DynamicFusionMapSpace::contracted(&rule, &source, &source, &[1], &[0]).unwrap();
        assert_eq!(final_result_layout_builds(), 1);
        assert!(contracted_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&cache_key));

        reset_final_result_layout_builds();
        let second =
            DynamicFusionMapSpace::contracted(&rule, &source, &source, &[1], &[0]).unwrap();
        assert_eq!(final_result_layout_builds(), 0);
        assert!(contracted_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&cache_key));
        assert!(Arc::ptr_eq(first.structure(), second.structure()));
    }

    #[test]
    fn ordered_contraction_builds_only_final_layout_and_warm_hit_builds_zero() {
        use tenet_operations::OutputAxisOrder;

        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_final_result_layout_test_state();
        let rule = U1FusionRule;
        let source = u1_space(902, 4);
        let axes = TensorContractSpec::new(&[1], &[0], OutputAxisOrder::from_axes(&[1, 0]));
        let cache_key = ContractedSpaceKey {
            rule: rule.rule_identity(),
            lhs_homspace: Arc::clone(&source.homspace),
            rhs_homspace: Arc::clone(&source.homspace),
            lhs_axes: DimVec::from_slice(&[1]),
            rhs_axes: DimVec::from_slice(&[0]),
            output_axes: DimVec::from_slice(&[1, 0]),
        };
        let default_homspace = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            source.homspace(),
            source.homspace(),
            &[1],
            &[0],
            &[0, 1],
            1,
        )
        .unwrap();
        let legacy_transform_key = TransformedSpaceKey {
            rule: rule.rule_identity(),
            source_homspace: Arc::new(default_homspace),
            operation: TreeTransformOperation::permute([1], [0]),
        };

        let first =
            DynamicFusionMapSpace::contracted_with_spec(&rule, &source, &source, axes).unwrap();
        assert_eq!(final_result_layout_builds(), 1);
        assert!(contracted_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&cache_key));
        assert!(!transformed_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&legacy_transform_key));

        reset_final_result_layout_builds();
        let second =
            DynamicFusionMapSpace::contracted_with_spec(&rule, &source, &source, axes).unwrap();
        assert_eq!(final_result_layout_builds(), 0);
        assert!(contracted_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&cache_key));
        assert!(Arc::ptr_eq(first.structure(), second.structure()));
    }

    #[test]
    fn validation_only_contraction_never_builds_or_caches_a_default_layout() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_final_result_layout_test_state();
        let rule = U1FusionRule;
        let compatible = u1_space(903, 4);
        let compatible_key = ContractedSpaceKey {
            rule: rule.rule_identity(),
            lhs_homspace: Arc::clone(&compatible.homspace),
            rhs_homspace: Arc::clone(&compatible.homspace),
            lhs_axes: DimVec::from_slice(&[1]),
            rhs_axes: DimVec::from_slice(&[0]),
            output_axes: DimVec::from_slice(&[0, 1]),
        };

        DynamicFusionMapSpace::validate_contracted_homspace(
            &rule,
            &compatible,
            &compatible,
            &[1],
            &[0],
        )
        .unwrap();
        assert_eq!(final_result_layout_builds(), 0);
        assert!(!contracted_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&compatible_key));

        let incompatible = u1_space(903, 5);
        let incompatible_key = ContractedSpaceKey {
            rule: rule.rule_identity(),
            lhs_homspace: Arc::clone(&compatible.homspace),
            rhs_homspace: Arc::clone(&incompatible.homspace),
            lhs_axes: DimVec::from_slice(&[1]),
            rhs_axes: DimVec::from_slice(&[0]),
            output_axes: DimVec::from_slice(&[0, 1]),
        };
        assert!(DynamicFusionMapSpace::validate_contracted_homspace(
            &rule,
            &compatible,
            &incompatible,
            &[1],
            &[0],
        )
        .is_err());
        assert_eq!(final_result_layout_builds(), 0);
        assert!(!contracted_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&incompatible_key));
    }

    #[test]
    fn generic_transform_uses_single_pass_without_entering_mult_free_cache() {
        use tenet_core::Su3FusionRule;

        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_final_result_layout_test_state();
        let rule = Su3FusionRule::new();
        let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
        let key_count = homspace.fusion_tree_keys_generic(&rule).unwrap().len();
        let source = DynamicFusionMapSpace::from_degeneracy_shapes_generic(
            &rule,
            homspace,
            vec![vec![2, 3]; key_count],
        )
        .unwrap();
        let operation = TreeTransformOperation::permute([1], [0]);
        let mult_free_cache_key = TransformedSpaceKey {
            rule: rule.rule_identity(),
            source_homspace: Arc::clone(&source.homspace),
            operation: operation.clone(),
        };

        let transformed = source.transformed_generic(&rule, &operation).unwrap();
        assert_eq!(final_result_layout_builds(), 1);
        assert!(!transformed_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&mult_free_cache_key));
        assert_eq!(transformed.structure().block_count(), key_count);
    }

    #[test]
    fn generic_contraction_builds_once_without_entering_mult_free_cache() {
        use tenet_core::Su3FusionRule;

        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_final_result_layout_test_state();
        let rule = Su3FusionRule::new();
        let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 2)]);
        let key_count = homspace.fusion_tree_keys_generic(&rule).unwrap().len();
        let source = DynamicFusionMapSpace::from_degeneracy_shapes_generic(
            &rule,
            homspace,
            vec![vec![2, 2]; key_count],
        )
        .unwrap();
        let mult_free_cache_key = ContractedSpaceKey {
            rule: rule.rule_identity(),
            lhs_homspace: Arc::clone(&source.homspace),
            rhs_homspace: Arc::clone(&source.homspace),
            lhs_axes: DimVec::from_slice(&[1]),
            rhs_axes: DimVec::from_slice(&[0]),
            output_axes: DimVec::from_slice(&[0, 1]),
        };

        let contracted =
            DynamicFusionMapSpace::contracted_generic(&rule, &source, &source, &[1], &[0]).unwrap();
        assert_eq!(final_result_layout_builds(), 1);
        assert!(!contracted_space_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&mult_free_cache_key));
        assert_eq!(contracted.structure().block_count(), key_count);
    }

    #[test]
    fn layout_authority_distinguishes_rules_and_reuses_semantic_spaces() {
        // What: opaque layout identity aliases equal bound layouts but never distinct rules.
        let first_provider = Arc::new(Z2FusionRule);
        let second_provider = Arc::new(Z2FusionRule);
        let raw = z2_matrix_space();
        let first = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            raw.clone(),
            Arc::clone(&first_provider),
        )
        .unwrap();
        let second =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::clone(&second_provider))
                .unwrap();
        assert_eq!(first.validated_layout(), second.validated_layout());

        let wrong = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            u1_space(0, 1),
            Arc::new(U1FusionRule),
        )
        .unwrap();
        assert_ne!(first.validated_layout(), wrong.validated_layout());
        let error = first
            .rebind_validated(&wrong.validated_layout())
            .unwrap_err();
        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
        assert!(Arc::ptr_eq(first.provider_arc(), &first_provider));
    }

    #[test]
    fn layout_authority_rebinds_derived_space_to_exact_provider_arc() {
        // What: a cached raw derived layout inherits the current caller's provider allocation.
        let first_provider = Arc::new(Z2FusionRule);
        let second_provider = Arc::new(Z2FusionRule);
        let raw = z2_matrix_space();
        let first = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            raw.clone(),
            Arc::clone(&first_provider),
        )
        .unwrap();
        let second =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::clone(&second_provider))
                .unwrap();
        let validated = first.validated_layout();

        let rebound = second.rebind_validated(&validated).unwrap();

        assert!(Arc::ptr_eq(rebound.provider_arc(), &second_provider));
        assert!(!Arc::ptr_eq(rebound.provider_arc(), &first_provider));
    }

    #[test]
    fn layout_authority_separates_zero_legs_and_storage_geometry() {
        // What: structural-zero legs and storage geometry remain distinct authorities.
        let provider = Arc::new(Z2FusionRule);
        let even = Z2Irrep::EVEN.sector_id();
        let odd = Z2Irrep::ODD.sector_id();
        let leg = |include_zero| {
            let sectors = if include_zero {
                vec![(even, 1), (odd, 0)]
            } else {
                vec![(even, 1)]
            };
            FusionProductSpace::new([SectorLeg::new(sectors, false)])
        };
        let without_zero_hom = FusionTreeHomSpace::new(leg(false), leg(false));
        let with_zero_hom = FusionTreeHomSpace::new(leg(true), leg(true));
        let without_zero = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            without_zero_hom,
            [vec![1, 1]],
        )
        .unwrap();
        let zero_shapes = with_zero_hom
            .fusion_tree_keys(provider.as_ref())
            .iter()
            .map(|key| {
                vec![
                    usize::from(key.codomain_tree().coupled() == Some(even)),
                    usize::from(key.domain_tree().coupled() == Some(even)),
                ]
            })
            .collect::<Vec<_>>();
        let with_zero = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            with_zero_hom,
            zero_shapes,
        )
        .unwrap();
        assert_ne!(
            without_zero.validated_layout(),
            with_zero.validated_layout()
        );

        let raw = z2_matrix_space();
        let shifted_blocks = (0..raw.structure().block_count())
            .map(|index| {
                let block = raw.structure().block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset() + 1,
                )
                .unwrap()
            })
            .collect();
        let shifted_raw = DynamicFusionMapSpace {
            subblock_structure: Arc::new(BlockStructure::from_blocks(shifted_blocks).unwrap()),
            ..raw.clone()
        };
        let canonical =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(raw.clone(), Arc::clone(&provider))
                .unwrap();
        let shifted =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(shifted_raw, Arc::clone(&provider))
                .unwrap();
        assert_ne!(canonical.validated_layout(), shifted.validated_layout());
    }

    #[test]
    fn validated_layout_does_not_retain_provider_allocation() {
        // What: erasing a bound layout drops the provider allocation while preserving layout data.
        let provider = Arc::new(Z2FusionRule);
        let weak = Arc::downgrade(&provider);
        let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            z2_matrix_space(),
            Arc::clone(&provider),
        )
        .unwrap();
        let validated = bound.validated_layout();

        drop(bound);
        drop(provider);

        assert!(weak.upgrade().is_none());
        assert_eq!(validated.required_len().unwrap(), 2);
    }

    fn z2_matrix_space() -> DynamicFusionMapSpace {
        let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
        DynamicFusionMapSpace::from_degeneracy_shapes(
            &Z2FusionRule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap()
    }

    #[test]
    fn bound_space_rejects_wrong_and_missing_rule_identity() {
        let space = z2_matrix_space();
        let wrong = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            space.clone(),
            Arc::new(U1FusionRule),
        )
        .unwrap_err();
        assert!(matches!(
            wrong,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));

        let unbound = DynamicFusionMapSpace {
            rule_identity: None,
            ..space
        };
        let missing =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(unbound, Arc::new(Z2FusionRule))
                .unwrap_err();
        assert!(matches!(
            missing,
            OperationError::Core(CoreError::MissingFusionRuleIdentity)
        ));
    }

    #[test]
    fn bound_space_requires_the_complete_tree_grid() {
        let complete = z2_matrix_space();
        let first = complete.structure().block(0).unwrap();
        let incomplete_structure = BlockStructure::from_blocks_with_rank(
            complete.rank(),
            vec![BlockSpec::with_key(
                first.key().clone(),
                first.shape().to_vec(),
                first.strides().to_vec(),
                first.offset(),
            )
            .unwrap()],
        )
        .unwrap();
        let incomplete = DynamicFusionMapSpace {
            subblock_structure: Arc::new(incomplete_structure),
            ..complete
        };

        let error =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(incomplete, Arc::new(Z2FusionRule))
                .unwrap_err();
        assert!(matches!(
            error,
            OperationError::Core(CoreError::BlockCountMismatch { .. })
        ));
    }

    #[test]
    fn binding_mode_mismatch_is_rejected_by_exact_tree_validation() {
        let space = z2_matrix_space();
        let multiplicity_free = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            space.clone(),
            Arc::new(Z2FusionRule),
        )
        .unwrap();
        let generic_error =
            BoundDynamicFusionMapSpace::bind_generic(space, Arc::new(Z2FusionRule)).unwrap_err();

        assert!(matches!(
            generic_error,
            OperationError::MissingBlockKey { .. }
        ));
        assert_eq!(multiplicity_free.clone().space(), multiplicity_free.space());
    }

    #[test]
    fn direct_bound_construction_enumerates_no_more_than_raw_construction() {
        let hom = || FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
        let raw_rule = CountingRule::new();
        DynamicFusionMapSpace::from_degeneracy_shapes(&raw_rule, hom(), [vec![1, 1]]).unwrap();
        let raw_calls = raw_rule.calls.load(Ordering::Relaxed);

        let bound_rule = Arc::new(CountingRule::new());
        BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&bound_rule),
            hom(),
            [vec![1, 1]],
        )
        .unwrap();

        assert_eq!(bound_rule.calls.load(Ordering::Relaxed), raw_calls);
    }

    #[test]
    fn derived_bound_contract_and_transform_do_not_reenumerate_for_binding() {
        // What: a derived proof adds no tree-grid pass beyond raw output build.
        let hom = || FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);

        let raw_rule = CountingRule::new();
        let raw =
            DynamicFusionMapSpace::from_degeneracy_shapes(&raw_rule, hom(), [vec![1, 1]]).unwrap();
        raw_rule.calls.store(0, Ordering::Relaxed);
        let _ = DynamicFusionMapSpace::contracted(&raw_rule, &raw, &raw, &[1], &[0]).unwrap();
        let raw_contract_calls = raw_rule.calls.load(Ordering::Relaxed);

        let bound_rule = Arc::new(CountingRule::new());
        let bound = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&bound_rule),
            hom(),
            [vec![1, 1]],
        )
        .unwrap();
        bound_rule.calls.store(0, Ordering::Relaxed);
        let contracted =
            BoundDynamicFusionMapSpace::contracted_multiplicity_free(&bound, &bound, &[1], &[0])
                .unwrap();
        assert_eq!(bound_rule.calls.load(Ordering::Relaxed), raw_contract_calls);
        assert!(Arc::ptr_eq(contracted.provider_arc(), &bound_rule));

        let raw_transform_rule = CountingRule::new();
        let raw_transform =
            DynamicFusionMapSpace::from_degeneracy_shapes(&raw_transform_rule, hom(), [vec![1, 1]])
                .unwrap();
        raw_transform_rule.calls.store(0, Ordering::Relaxed);
        let operation = TreeTransformOperation::permute([0], [1]);
        let _ = raw_transform
            .transformed(&raw_transform_rule, &operation)
            .unwrap();
        let raw_transform_calls = raw_transform_rule.calls.load(Ordering::Relaxed);

        let bound_transform_rule = Arc::new(CountingRule::new());
        let bound_transform = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&bound_transform_rule),
            hom(),
            [vec![1, 1]],
        )
        .unwrap();
        bound_transform_rule.calls.store(0, Ordering::Relaxed);
        let transformed = bound_transform
            .transformed_multiplicity_free(&operation)
            .unwrap();
        assert_eq!(
            bound_transform_rule.calls.load(Ordering::Relaxed),
            raw_transform_calls
        );
        assert!(Arc::ptr_eq(
            transformed.provider_arc(),
            &bound_transform_rule
        ));
    }

    #[test]
    fn bound_contract_normalizes_equal_identity_to_lhs_provider() {
        // What: independently allocated providers with one semantic identity
        // compose, while a distinct identity is rejected before execution.
        let rule = CountingRule::new();
        let lhs_provider = Arc::new(rule.clone());
        let rhs_provider = Arc::new(rule);
        assert!(!Arc::ptr_eq(&lhs_provider, &rhs_provider));
        let hom = || FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
        let lhs = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&lhs_provider),
            hom(),
            [vec![1, 1]],
        )
        .unwrap();
        let rhs = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&rhs_provider),
            hom(),
            [vec![1, 1]],
        )
        .unwrap();
        let output =
            BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lhs, &rhs, &[1], &[0])
                .unwrap();
        assert!(Arc::ptr_eq(output.provider_arc(), &lhs_provider));

        let other_provider = Arc::new(CountingRule::new());
        let other =
            BoundDynamicFusionMapSpace::from_degeneracy_shapes(other_provider, hom(), [vec![1, 1]])
                .unwrap();
        let error =
            BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lhs, &other, &[1], &[0])
                .unwrap_err();
        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
    }

    #[test]
    fn equal_layout_and_shapes_share_one_structure() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let a = u1_space(1, 3);
        let b = u1_space(1, 3);
        assert!(Arc::ptr_eq(a.structure(), b.structure()));
    }

    #[test]
    fn different_bond_dimension_is_not_shared() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // The shapes carry chi, so a differently-truncated build keys separately.
        let a = u1_space(1, 3);
        let b = u1_space(1, 4);
        assert!(!Arc::ptr_eq(a.structure(), b.structure()));
    }

    #[test]
    fn different_homspace_is_not_shared() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let a = u1_space(1, 3);
        let b = u1_space(2, 3);
        assert!(!Arc::ptr_eq(a.structure(), b.structure()));
    }

    // Rule provenance is first-class in the layout id: two rules that index the
    // same hom space still key to distinct layouts. `Su3FusionRule` (runtime
    // provenance) never reaches this multiplicity-free cache, but its type is a
    // distinct discriminant from any mult-free rule all the same.
    #[test]
    fn distinct_rule_provenance_gives_distinct_layout_ids() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        use tenet_core::{SU2FusionRule, Su3FusionRule};
        let homspace = u1_space(1, 3).homspace().id();
        let make = |rule| LayoutId {
            rule,
            homspace: homspace.clone(),
        };
        let u1 = make(RuleIdentity::of_type::<U1FusionRule>());
        let su2 = make(RuleIdentity::of_type::<SU2FusionRule>());
        let su3 = make(RuleIdentity::of_type::<Su3FusionRule>());
        assert_ne!(u1, su2);
        assert_ne!(u1, su3);
        assert_ne!(su2, su3);
    }

    #[test]
    fn same_rule_type_with_distinct_provenance_gives_distinct_layout_ids() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let homspace = u1_space(1, 3).homspace().id();
        let first = LayoutId {
            rule: RuleIdentity::new_unique::<U1FusionRule>(),
            homspace: homspace.clone(),
        };
        let second = LayoutId {
            rule: RuleIdentity::new_unique::<U1FusionRule>(),
            homspace,
        };
        assert_ne!(first, second);
    }

    #[test]
    fn scratch_structure_reuses_after_homspace_intern_eviction() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::reset_global_operation_caches();
        let before = u1_space(73, 3);
        for charge in 10_000..19_000 {
            let sid = U1Irrep::new(charge).sector_id();
            let leg = || FusionProductSpace::new([SectorLeg::new([(sid, 1)], false)]);
            let _ = FusionTreeHomSpace::new(leg(), leg());
        }
        let after = u1_space(73, 3);
        assert_eq!(before.homspace().id(), after.homspace().id());
        assert!(Arc::ptr_eq(before.structure(), after.structure()));
    }

    #[test]
    fn large_shapes_reuse_without_allocating_tensor_storage() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rule = U1FusionRule;
        let sid = U1Irrep::new(91).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(sid, 4096)], false)]);
        let hom = FusionTreeHomSpace::new(leg(), leg());
        let first =
            DynamicFusionMapSpace::from_degeneracy_shapes(&rule, hom.clone(), [vec![4096, 4096]])
                .unwrap();
        let second =
            DynamicFusionMapSpace::from_degeneracy_shapes(&rule, hom, [vec![4096, 4096]]).unwrap();
        assert_eq!(first.required_len().unwrap(), 4096 * 4096);
        assert!(Arc::ptr_eq(first.structure(), second.structure()));
    }

    #[test]
    fn reset_and_concurrent_rebuild_keep_structure_semantics() {
        let _guard = CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::reset_global_operation_caches();
        let spaces = std::thread::scope(|scope| {
            let resetter = scope.spawn(|| {
                for _ in 0..32 {
                    crate::reset_global_operation_caches();
                }
            });
            let builders = (0..4)
                .map(|_| scope.spawn(|| (0..32).map(|_| u1_space(111, 5)).collect::<Vec<_>>()))
                .collect::<Vec<_>>();
            resetter.join().unwrap();
            builders
                .into_iter()
                .flat_map(|builder| builder.join().unwrap())
                .collect::<Vec<_>>()
        });
        let expected = spaces[0].structure().as_ref();
        assert!(spaces
            .iter()
            .all(|space| space.structure().as_ref() == expected));
        let rebuilt = u1_space(111, 5);
        let cached = u1_space(111, 5);
        assert!(Arc::ptr_eq(rebuilt.structure(), cached.structure()));
    }

    #[test]
    fn bound_layout_strategy_propagates_without_heap_ownership() {
        // What: clone, validated-layout rebind, derived construction, transform,
        // and contraction retain one opaque lowered strategy, while an expert
        // encoded root remains distinct and a mixed binary result follows lhs.
        assert_eq!(
            std::mem::size_of::<LayoutBuildCapability<tenet_core::SU2FusionRule>>(),
            std::mem::size_of::<usize>()
        );
        let provider = Arc::new(tenet_core::SU2FusionRule);
        let half = tenet_core::SU2Irrep::from_twice_spin(1).sector_id();
        let leg = || FusionProductSpace::new([tenet_core::SectorLeg::new([(half, 1)], false)]);
        let make_hom = || FusionTreeHomSpace::new(leg(), leg());
        let hom = make_hom();
        lowered_layout_primer(provider.as_ref(), &hom).unwrap();
        let shapes = vec![vec![1; 2]; hom.fusion_tree_keys(provider.as_ref()).len()];
        let lowered = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
            Arc::clone(&provider),
            hom.clone(),
            shapes.clone(),
        )
        .unwrap();
        let encoded =
            BoundDynamicFusionMapSpace::from_degeneracy_shapes(Arc::clone(&provider), hom, shapes)
                .unwrap();

        let cloned = lowered.clone();
        let rebound = lowered
            .rebind_validated(&lowered.validated_layout())
            .unwrap();
        let derived = lowered.derive_from_final_homspace(make_hom()).unwrap();
        let transformed = lowered
            .transformed_multiplicity_free(&TreeTransformOperation::permute([0], [1]))
            .unwrap();
        let contracted =
            BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lowered, &cloned, &[], &[])
                .unwrap();

        for output in [&cloned, &rebound, &derived, &transformed, &contracted] {
            assert!(lowered.has_same_layout_build_strategy(output));
        }
        assert!(!lowered.has_same_layout_build_strategy(&encoded));
        let mixed =
            BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lowered, &encoded, &[], &[])
                .unwrap();
        assert!(lowered.has_same_layout_build_strategy(&mixed));
    }
}
