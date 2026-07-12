use std::any::{Any, TypeId};
use std::sync::{Arc, OnceLock, RwLock};

use rustc_hash::FxHashMap;
use tenet_core::{
    BlockKey, BlockStructure, CoreError, DimVec, FusionRule, FusionTensorMapSpace,
    FusionTreeHomSpace, HomSpaceId, MultiplicityFreeRigidSymbols, RuleIdentity,
};

use crate::cache::operation_global_registry;
use crate::{OperationError, TreeTransformOperation};
use tenet_operations::TensorContractSpec;

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
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
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
    let registry = operation_global_registry();
    let type_id = TypeId::of::<ScratchStructureCache>();
    if let Some(cache) = registry
        .read()
        .expect("global cache registry poisoned")
        .get(&type_id)
    {
        return Arc::downcast::<RwLock<ScratchStructureCache>>(Arc::clone(cache))
            .expect("scratch structure cache type id collision");
    }
    let mut caches = registry.write().expect("global cache registry poisoned");
    if let Some(cache) = caches.get(&type_id) {
        return Arc::downcast::<RwLock<ScratchStructureCache>>(Arc::clone(cache))
            .expect("scratch structure cache type id collision");
    }
    let cache: Arc<RwLock<ScratchStructureCache>> =
        Arc::new(RwLock::new(ScratchStructureCache::default()));
    caches.insert(type_id, Arc::clone(&cache) as Arc<dyn Any + Send + Sync>);
    cache
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

impl DynamicFusionMapSpace {
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
    pub fn transformed<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
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
        let src_axes = codomain_axes
            .iter()
            .chain(domain_axes.iter())
            .copied()
            .collect::<Vec<_>>();
        // Legs are authoritative for degeneracies: the external leg of each
        // source axis carries the full sector -> degeneracy map, keyed by
        // the placement-invariant external sector labels.
        let src_legs = src_axes
            .iter()
            .map(|&src_axis| source.homspace().external_axis_leg(rule, src_axis))
            .collect::<Vec<_>>();
        let keys = homspace.fusion_tree_keys(rule);
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys.iter() {
            let sectors = key.external_sectors(rule);
            let mut shape = Vec::with_capacity(src_axes.len());
            for (out_axis, leg) in src_legs.iter().enumerate() {
                let dim =
                    leg.degeneracy(sectors[out_axis])
                        .ok_or(OperationError::StructureMismatch {
                            tensor: "transformed scratch",
                        })?;
                shape.push(dim);
            }
            blocks.push((BlockKey::from(key.clone()), shape));
        }
        let subblock_structure =
            Arc::new(scratch_subblock_structure(rule, nout, nout + nin, blocks)?);
        let space = Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            rule_identity: Some(rule.rule_identity()),
        };
        if let Ok(mut map) = transformed_space_cache().write() {
            map.insert(cache_key, space.clone());
        }
        Ok(space)
    }

    /// Generic-fusion (SU(3)) sibling of [`Self::from_degeneracy_shapes`]:
    /// builds the multiplicity-aware block structure from a `FusionRule` whose
    /// `nsymbol` can exceed 1. Only `fusion_tree_keys_generic` differs from the
    /// mult-free path (every other step is rule-agnostic or `FusionRule`-bound).
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
    pub fn transformed_generic<R>(
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
        let src_axes = codomain_axes
            .iter()
            .chain(domain_axes.iter())
            .copied()
            .collect::<Vec<_>>();
        let src_legs = src_axes
            .iter()
            .map(|&src_axis| source.homspace().external_axis_leg(rule, src_axis))
            .collect::<Vec<_>>();
        let keys = homspace
            .fusion_tree_keys_generic(rule)
            .map_err(OperationError::from_core_preserving_context)?;
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys.iter() {
            let sectors = key.external_sectors(rule);
            let mut shape = Vec::with_capacity(src_axes.len());
            for (out_axis, leg) in src_legs.iter().enumerate() {
                let dim =
                    leg.degeneracy(sectors[out_axis])
                        .ok_or(OperationError::StructureMismatch {
                            tensor: "transformed scratch",
                        })?;
                shape.push(dim);
            }
            blocks.push((BlockKey::from(key.clone()), shape));
        }
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

    /// Space of the contraction result in the default output order (`lhs`
    /// open axes ascending on the codomain side, `rhs` open axes ascending on
    /// the domain side). Mirrors the destination TensorKit's
    /// `tensorcontract!` with default `pAB` writes into.
    pub fn contracted<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
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
        let key = ContractedSpaceKey {
            rule: rule.rule_identity(),
            lhs_homspace: Arc::clone(&lhs.homspace),
            rhs_homspace: Arc::clone(&rhs.homspace),
            lhs_axes: lhs_axes.to_vec(),
            rhs_axes: rhs_axes.to_vec(),
        };
        if let Some(cached) = contracted_space_cache()
            .read()
            .ok()
            .and_then(|map| map.get(&key).cloned())
        {
            return Ok(cached);
        }
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let space = Self::contracted_space(rule, lhs, rhs, axes, nout, nin)?;
        if let Ok(mut map) = contracted_space_cache().write() {
            map.insert(key, space.clone());
        }
        Ok(space)
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
        let nout = plan.core_dst_open_lhs_rank();
        let nin = plan.core_dst_open_rhs_rank();
        Self::contracted_space(rule, lhs, rhs, plan.core_axes().as_spec(), nout, nin)
    }

    fn contracted_space<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        nout: usize,
        nin: usize,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        let output_axes = (0..nout + nin).collect::<Vec<_>>();
        let homspace = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.homspace(),
            rhs.homspace(),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &output_axes,
            nout,
        )
        .map_err(OperationError::from_core_preserving_context)?;

        // The legs are authoritative for every subblock shape: a fusion-tree
        // key's shape is fully determined by each open axis' leg degeneracy
        // at the key's external sector (TensorKit GradedSpace parity: the
        // legs carry the complete sector -> degeneracy map, so shapes are
        // recoverable even for structural-zero keys the contraction pairing
        // never produces, e.g. sparse product states or factors of a
        // truncated SVD that dropped a whole coupled sector).
        let lhs_open = axis_plan.lhs_open_axes.clone();
        let rhs_open = axis_plan.rhs_open_axes.clone();
        let open_legs = lhs_open
            .iter()
            .map(|&axis| lhs.homspace().external_axis_leg(rule, axis))
            .chain(
                rhs_open
                    .iter()
                    .map(|&axis| rhs.homspace().external_axis_leg(rule, axis)),
            )
            .collect::<Vec<_>>();
        let keys = homspace.fusion_tree_keys(rule);
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys.iter() {
            let sectors = key.external_sectors(rule);
            let shape = sectors
                .iter()
                .zip(&open_legs)
                .map(|(&sector, leg)| {
                    leg.degeneracy(sector)
                        .ok_or(OperationError::StructureMismatch {
                            tensor: "contracted result",
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            blocks.push((BlockKey::from(key.clone()), shape));
        }
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

    /// Generic-fusion (Stage B3c-1) sibling of [`Self::contracted`]: the
    /// contraction result space for an outer-multiplicity rule, enumerated with
    /// multiplicity-aware fusion-tree keys (`fusion_tree_keys_generic`). Not
    /// cached (the Generic path is not on a hot loop yet — same non-memoized
    /// rationale as the B3b transform siblings). Every other step
    /// (`tensorcontract_homspace`, leg degeneracies, `scratch_subblock_structure`)
    /// is already rule-agnostic or `FusionRule`-bound.
    pub fn contracted_generic<R>(
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
        let output_axes = (0..nout + nin).collect::<Vec<_>>();
        let homspace = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.homspace(),
            rhs.homspace(),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &output_axes,
            nout,
        )
        .map_err(OperationError::from_core_preserving_context)?;

        let open_legs = axis_plan
            .lhs_open_axes
            .iter()
            .map(|&axis| lhs.homspace().external_axis_leg(rule, axis))
            .chain(
                axis_plan
                    .rhs_open_axes
                    .iter()
                    .map(|&axis| rhs.homspace().external_axis_leg(rule, axis)),
            )
            .collect::<Vec<_>>();
        let keys = homspace
            .fusion_tree_keys_generic(rule)
            .map_err(OperationError::from_core_preserving_context)?;
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys.iter() {
            let sectors = key.external_sectors(rule);
            let shape = sectors
                .iter()
                .zip(&open_legs)
                .map(|(&sector, leg)| {
                    leg.degeneracy(sector)
                        .ok_or(OperationError::StructureMismatch {
                            tensor: "contracted result",
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            blocks.push((BlockKey::from(key.clone()), shape));
        }
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

    /// Adjoint view: codomain and domain swap (spaces and per-block shapes),
    /// no data movement implied. The block layout is a strided view into the
    /// source layout, so this space is for replay bookkeeping, not for
    /// allocating fresh coupled storage.
    pub fn adjoint_view(&self) -> Result<Self, OperationError> {
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
mod scratch_cache_tests {
    use super::*;
    use std::sync::Mutex;
    use tenet_core::{FusionProductSpace, SectorLeg, U1FusionRule, U1Irrep};

    static CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn equal_layout_and_shapes_share_one_structure() {
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
        let a = u1_space(1, 3);
        let b = u1_space(1, 3);
        assert!(Arc::ptr_eq(a.structure(), b.structure()));
    }

    #[test]
    fn different_bond_dimension_is_not_shared() {
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
        // The shapes carry chi, so a differently-truncated build keys separately.
        let a = u1_space(1, 3);
        let b = u1_space(1, 4);
        assert!(!Arc::ptr_eq(a.structure(), b.structure()));
    }

    #[test]
    fn different_homspace_is_not_shared() {
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
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
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
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
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
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
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
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
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
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
        let _guard = CACHE_TEST_LOCK.lock().unwrap();
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
}
