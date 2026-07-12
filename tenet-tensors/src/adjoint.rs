//! Adjoint (dagger) of a fusion tensor.
//!
//! TensorKit semantics: the adjoint of `t : codomain <- domain` is
//! `t^H : domain <- codomain`, whose coupled-sector blocks are the conjugate
//! transposes of `t`'s blocks (`block(t^H, c) = block(t, c)^H`). Codomain and
//! domain swap as spaces; leg duality flags are unchanged.

use std::any::{Any, TypeId};
use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use rustc_hash::FxHashMap;
use tenet_core::{
    BlockKey, CoreError, FusionRule, FusionTensorMapSpace, FusionTreeBlockKey, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, TensorMap, TensorMapSpace,
};

use crate::cache::{enforce_lru_limit, operation_global_registry, touch_lru_key};
use crate::contract::DynamicFusionMapSpace;
use crate::tree_transform::TreeTransformRuleCacheKey;
use crate::{ConjugateValue, OperationError};

/// Identity of an adjoint (dagger) output space. The dagger is a pure function
/// of the source: its hom space (legs carry the authoritative
/// sector→degeneracy/duality map every adjoint block shape derives from) and its
/// coupled subblock layout, under a given fusion rule. Mirrors
/// `TransformedSpaceKey`/`ContractedSpaceKey` in `contract::dynamic_space`; the
/// operation is fixed (dagger) so there is no operation field.
///
/// Why-not (`rule_type: &str` provenance): a type name distinguishes rule types
/// but not two tables of the same type (a regenerated `TabulatedFusionRule` /
/// SU(3) provider fuses the same sector ids differently), so the key carries the
/// rule's `TreeTransformRuleCacheKey` — the provenance-bearing replay cache key.
///
/// Why-not (source `content_id` alone): the subblock structure is the
/// coupled-sector *matrix* layout, coarser than the hom space, so distinct
/// sources can share a `content_id` yet need different daggers — an id-only key
/// aliases them (measured: the finite-torus singlet then fails with a dimension
/// mismatch). The hom space plus `content_id`/`nout`/`nin` is sound; a cheap
/// interned `HomSpaceId` replaces the by-value hom space clone in PR-2.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct AdjointSpaceKey<RuleKey> {
    rule_key: RuleKey,
    source_homspace: Arc<FusionTreeHomSpace>,
    source_content_id: usize,
    nout: usize,
    nin: usize,
}

/// Bounded LRU store of built adjoint spaces (strong `Arc`, since
/// `adjoint_space_dyn` returns by value and would leave a `Weak` with no owner).
struct AdjointSpaceCache<RuleKey> {
    entries: FxHashMap<AdjointSpaceKey<RuleKey>, Arc<DynamicFusionMapSpace>>,
    order: VecDeque<AdjointSpaceKey<RuleKey>>,
}

impl<RuleKey> Default for AdjointSpaceCache<RuleKey> {
    fn default() -> Self {
        Self {
            entries: FxHashMap::default(),
            order: VecDeque::new(),
        }
    }
}

/// Residency bound for the process-global adjoint-space cache. A finite-torus
/// energy eval interns ~O(100) distinct daggers, so this never evicts in that
/// workload; it exists only to keep an adversarial (many-distinct-space) run
/// from growing the strong cache without limit, mirroring the operation cache
/// policy's LRU cap rather than the sibling spaces' unbounded global maps.
const ADJOINT_SPACE_CACHE_CAP: usize = 8192;

/// Process-global adjoint-space cache, one bounded store per fusion `RuleKey`
/// type — the per-type registry the tree-transform/contraction replay caches
/// use, so `reset_global_operation_caches` clears it too. (Own accessor rather
/// than `typed_global_map` because the map and its LRU order must share one lock
/// to stay consistent.)
fn adjoint_space_cache<RuleKey>() -> Arc<RwLock<AdjointSpaceCache<RuleKey>>>
where
    RuleKey: 'static + Send + Sync,
{
    let registry = operation_global_registry();
    let type_id = TypeId::of::<AdjointSpaceCache<RuleKey>>();
    if let Some(cache) = registry
        .read()
        .expect("global cache registry poisoned")
        .get(&type_id)
    {
        return Arc::downcast::<RwLock<AdjointSpaceCache<RuleKey>>>(Arc::clone(cache))
            .expect("adjoint space cache type id collision");
    }
    let mut caches = registry.write().expect("global cache registry poisoned");
    if let Some(cache) = caches.get(&type_id) {
        return Arc::downcast::<RwLock<AdjointSpaceCache<RuleKey>>>(Arc::clone(cache))
            .expect("adjoint space cache type id collision");
    }
    let cache: Arc<RwLock<AdjointSpaceCache<RuleKey>>> =
        Arc::new(RwLock::new(AdjointSpaceCache::default()));
    caches.insert(type_id, Arc::clone(&cache) as Arc<dyn Any + Send + Sync>);
    cache
}

/// Dynamic-rank adjoint space (dagger of the homspace): codomain and domain
/// swapped, per-block shapes transposed. Pure metadata — touches no data — so a
/// lazy adjoint can present the correct fresh coupled space in O(blocks)
/// without copying any elements. `adjoint_dyn` is exactly this plus the
/// conjugate-transpose of the block data, and its output data lives in this
/// space's layout.
///
/// Process-cached: the warm energy loop rebuilds the same daggers every eval,
/// and even this metadata build pays per-key shape lookups plus
/// `from_degeneracy_shapes`/`scratch_subblock_structure`/content interning, so
/// an equal source resolves the already-built space instead (#118).
pub fn adjoint_space_dyn<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let key = AdjointSpaceKey {
        rule_key: rule.tree_transform_rule_cache_key(),
        source_homspace: Arc::clone(space.homspace_arc()),
        source_content_id: space.structure().content_id(),
        nout: space.nout(),
        nin: space.nin(),
    };
    let cache = adjoint_space_cache::<R::Key>();
    if let Ok(mut guard) = cache.write() {
        if let Some(hit) = guard.entries.get(&key).cloned() {
            touch_lru_key(&mut guard.order, &key);
            // Clone the inner hom-space/subblock Arcs for the by-value return.
            return Ok((*hit).clone());
        }
    }
    let built = Arc::new(build_adjoint_space_dyn(rule, space)?);
    if let Ok(mut guard) = cache.write() {
        guard.entries.insert(key.clone(), Arc::clone(&built));
        touch_lru_key(&mut guard.order, &key);
        let AdjointSpaceCache { entries, order } = &mut *guard;
        enforce_lru_limit(entries, order, ADJOINT_SPACE_CACHE_CAP);
    }
    Ok((*built).clone())
}

/// Uncached build of the adjoint (dagger) space; [`adjoint_space_dyn`] memoizes
/// it.
fn build_adjoint_space_dyn<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let nout = space.nout();
    let homspace = space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());

    let structure = Arc::clone(space.structure());
    let keys = adjoint_hom.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ));
            let index = structure.find_block_index_by_key(&source_key).ok_or(
                OperationError::MissingBlockKey {
                    key: source_key.clone(),
                },
            )?;
            let source_shape = structure
                .block(index)
                .map_err(OperationError::from_core_preserving_context)?
                .shape();
            let mut shape = source_shape[nout..].to_vec();
            shape.extend_from_slice(&source_shape[..nout]);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    Ok(DynamicFusionMapSpace::from_degeneracy_shapes(
        rule,
        adjoint_hom,
        shapes,
    )?)
}

/// Dynamic-rank adjoint (dagger): returns the adjoint space (codomain and
/// domain swapped) together with freshly allocated coupled-layout data whose
/// blocks are the conjugate transposes of the source blocks.
pub fn adjoint_dyn<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let nout = space.nout();
    let nin = space.nin();
    let structure = Arc::clone(space.structure());
    // Uncached build: the eager adjoint (SVD/eigh consumers) is a separate,
    // out-of-scope path from the cached lazy `adjoint_space_dyn` (#118 PR-1),
    // and routing it here keeps the replay-cache-key bound off matrix algebra.
    let adjoint_space = build_adjoint_space_dyn(rule, space)?;
    let len = adjoint_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut result = vec![D::zero(); len];

    let result_structure = Arc::clone(adjoint_space.structure());
    for index in 0..result_structure.block_count() {
        let block = result_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let source_index = structure
            .find_block_index_by_key(&source_key)
            .ok_or(OperationError::MissingBlockKey { key: source_key })?;
        let source_block = structure
            .block(source_index)
            .map_err(OperationError::from_core_preserving_context)?;

        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let source_strides = source_block.strides().to_vec();
        let source_offset = source_block.offset();
        // Adjoint index map: result (j[..nin], i[..nout]) reads
        // conj(source(i, j)).
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let source_position = source_offset
                + indices[nin..]
                    .iter()
                    .zip(&source_strides[..nout])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>()
                + indices[..nin]
                    .iter()
                    .zip(&source_strides[nout..])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            result[position] = data[source_position].maybe_conj(true);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok((adjoint_space, result))
}

/// Generic-fusion (SU(N)) sibling of [`adjoint_space_dyn`]. The adjoint is a
/// pure per-block relabel — codomain and domain trees swap wholesale and each
/// coupled block is transposed — with NO leg bending and NO recoupling, so it
/// is self-duality-independent and identical for a Generic (outer-multiplicity)
/// rule: the only difference from the mult-free path is that the block keys are
/// enumerated multiplicity-aware (vertex-labelled fusion trees). Bound relaxes
/// to [`FusionRule`] accordingly (no F/R symbols are touched).
pub fn adjoint_space_dyn_generic<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: FusionRule,
{
    let nout = space.nout();
    let homspace = space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());

    let structure = Arc::clone(space.structure());
    let keys = adjoint_hom
        .fusion_tree_keys_generic(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    let shapes = keys
        .iter()
        .map(|key| {
            let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ));
            let index = structure.find_block_index_by_key(&source_key).ok_or(
                OperationError::MissingBlockKey {
                    key: source_key.clone(),
                },
            )?;
            let source_shape = structure
                .block(index)
                .map_err(OperationError::from_core_preserving_context)?
                .shape();
            let mut shape = source_shape[nout..].to_vec();
            shape.extend_from_slice(&source_shape[..nout]);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    Ok(DynamicFusionMapSpace::from_degeneracy_shapes_generic(
        rule,
        adjoint_hom,
        shapes,
    )?)
}

/// Generic-fusion (SU(N)) sibling of [`adjoint_dyn`]: same block relabel +
/// conjugate-transpose, over the multiplicity-aware adjoint space (see
/// [`adjoint_space_dyn_generic`]). The data movement is byte-identical to the
/// mult-free path — no recoupling coefficients enter — so this is the eager
/// materialization TensorKit takes when an SU(N) adjoint is consumed by
/// something other than a contraction.
pub fn adjoint_dyn_generic<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: FusionRule,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let nout = space.nout();
    let nin = space.nin();
    let structure = Arc::clone(space.structure());
    let adjoint_space = adjoint_space_dyn_generic(rule, space)?;
    let len = adjoint_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut result = vec![D::zero(); len];

    let result_structure = Arc::clone(adjoint_space.structure());
    for index in 0..result_structure.block_count() {
        let block = result_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let source_index = structure
            .find_block_index_by_key(&source_key)
            .ok_or(OperationError::MissingBlockKey { key: source_key })?;
        let source_block = structure
            .block(source_index)
            .map_err(OperationError::from_core_preserving_context)?;

        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let source_strides = source_block.strides().to_vec();
        let source_offset = source_block.offset();
        // Adjoint index map: result (j[..nin], i[..nout]) reads
        // conj(source(i, j)).
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let source_position = source_offset
                + indices[nin..]
                    .iter()
                    .zip(&source_strides[..nout])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>()
                + indices[..nin]
                    .iter()
                    .zip(&source_strides[nout..])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            result[position] = data[source_position].maybe_conj(true);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok((adjoint_space, result))
}

/// Eager blockwise adjoint; the output uses the coupled-sector matrix layout.
pub fn adjoint<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<TensorMap<D, NIN, NOUT>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let homspace = fusion_space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());

    let structure = Arc::clone(tensor.structure());
    let source_shape_of = |key: &FusionTreeBlockKey| -> Result<Vec<usize>, OperationError> {
        // The adjoint block for (dom_tree, cod_tree) reads the source block
        // keyed by the swapped pair.
        let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let index = structure.find_block_index_by_key(&source_key).ok_or(
            OperationError::MissingBlockKey {
                key: source_key.clone(),
            },
        )?;
        Ok(structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?
            .shape()
            .to_vec())
    };

    let keys = adjoint_hom.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let source_shape = source_shape_of(key)?;
            let mut shape = source_shape[NOUT..].to_vec();
            shape.extend_from_slice(&source_shape[..NOUT]);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    let dims = tensor.space().dims();
    let mut domain_dims = [0usize; NIN];
    domain_dims.copy_from_slice(&dims[NOUT..]);
    let mut codomain_dims = [0usize; NOUT];
    codomain_dims.copy_from_slice(&dims[..NOUT]);
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<NIN, NOUT>::from_dims(domain_dims, codomain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        adjoint_hom,
        rule,
        shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut result =
        TensorMap::<D, NIN, NOUT>::from_vec_with_fusion_space(vec![D::zero(); len], space)
            .map_err(OperationError::from_core_preserving_context)?;

    let result_structure = Arc::clone(result.structure());
    let source_data = tensor.data();
    for index in 0..result_structure.block_count() {
        let block = result_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let source_index = structure
            .find_block_index_by_key(&source_key)
            .ok_or(OperationError::MissingBlockKey { key: source_key })?;
        let source_block = structure
            .block(source_index)
            .map_err(OperationError::from_core_preserving_context)?;

        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let source_strides = source_block.strides().to_vec();
        let source_offset = source_block.offset();
        // Adjoint index map: result (j[..NIN], i[..NOUT]) reads
        // conj(source(i, j)).
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        let data = result.data_mut();
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let source_position = source_offset
                + indices[NIN..]
                    .iter()
                    .zip(&source_strides[..NOUT])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>()
                + indices[..NIN]
                    .iter()
                    .zip(&source_strides[NOUT..])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            data[position] = source_data[source_position].maybe_conj(true);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::tree_transform::TreeTransformBuiltinRuleCacheKey;
    use tenet_core::{FusionProductSpace, SectorLeg, U1FusionRule, U1Irrep};

    // Single-charge U(1) source: one coupled sector, block shape [deg, deg].
    fn u1_source(charge: i32, deg: usize) -> DynamicFusionMapSpace {
        let rule = U1FusionRule;
        let sid = U1Irrep::new(charge).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(sid, deg)], false)]);
        let hom = FusionTreeHomSpace::new(leg(), leg());
        let count = hom.fusion_tree_keys(&rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, hom, vec![vec![deg, deg]; count])
            .unwrap()
    }

    #[test]
    fn equal_source_returns_the_cached_layout() {
        let rule = U1FusionRule;
        let src = u1_source(1, 2);
        let first = adjoint_space_dyn(&rule, &src).unwrap();
        let second = adjoint_space_dyn(&rule, &src).unwrap();
        // A hit clones the cached space's inner Arcs, so both share one layout.
        assert!(Arc::ptr_eq(first.structure(), second.structure()));
    }

    #[test]
    fn different_source_is_not_aliased() {
        let rule = U1FusionRule;
        let a = adjoint_space_dyn(&rule, &u1_source(1, 2)).unwrap();
        let b = adjoint_space_dyn(&rule, &u1_source(3, 2)).unwrap();
        assert!(!Arc::ptr_eq(a.structure(), b.structure()));
    }

    // Provenance is first-class in the key: two rules that would otherwise index
    // the same source get distinct cache entries.
    #[test]
    fn distinct_rule_provenance_gives_distinct_keys() {
        let src = u1_source(1, 2);
        let make = |rule_key| AdjointSpaceKey {
            rule_key,
            source_homspace: Arc::clone(src.homspace_arc()),
            source_content_id: src.structure().content_id(),
            nout: src.nout(),
            nin: src.nin(),
        };
        assert_ne!(
            make(TreeTransformBuiltinRuleCacheKey::U1),
            make(TreeTransformBuiltinRuleCacheKey::SU2)
        );
    }
}
