//! Adjoint (dagger) of a fusion tensor.
//!
//! TensorKit semantics: the adjoint of `t : codomain <- domain` is
//! `t^H : domain <- codomain`, whose coupled-sector blocks are the conjugate
//! transposes of `t`'s blocks (`block(t^H, c) = block(t, c)^H`). Codomain and
//! domain swap as spaces; leg duality flags are unchanged.

use std::hash::Hash;
use std::sync::{Arc, RwLock};

use tenet_core::{
    BlockKey, CoreError, FusionRule, FusionTensorMapSpace, FusionTreeBlockKey, FusionTreeHomSpace,
    HomSpaceId, LoweredMultiplicityFreeAlgebra, MultiplicityFreeRigidSymbols, TensorMap,
    TensorMapSpace,
};

use crate::cache::registered_operation_cache;
#[cfg(test)]
use crate::contract::{encoded_layout_primer, lowered_layout_primer, PreparedLayoutKeys};
use crate::contract::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace, LayoutKeyBuilder};
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
/// mismatch). The hom space plus `content_id`/`nout`/`nin` is sound; the source
/// hom space enters as its interned [`HomSpaceId`] (PR-2), so the key hashes a
/// small id instead of walking the space's legs by value every warm eval.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct AdjointSpaceKey<RuleKey> {
    rule_key: RuleKey,
    source_homspace_id: HomSpaceId,
    source_content_id: usize,
    nout: usize,
    nin: usize,
}

/// Bounded LRU store of built adjoint spaces (strong `Arc`, since
/// `adjoint_space_dyn` returns by value and would leave a `Weak` with no owner).
struct AdjointSpaceCache<RuleKey> {
    entries: lru::LruCache<AdjointSpaceKey<RuleKey>, Arc<DynamicFusionMapSpace>>,
}

impl<RuleKey: Eq + Hash> Default for AdjointSpaceCache<RuleKey> {
    fn default() -> Self {
        Self {
            entries: lru::LruCache::new(
                std::num::NonZeroUsize::new(ADJOINT_SPACE_CACHE_CAP).unwrap(),
            ),
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
    RuleKey: 'static + Eq + Hash + Send + Sync,
{
    registered_operation_cache::<RwLock<AdjointSpaceCache<RuleKey>>>().1
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
#[cfg(test)]
pub(crate) fn adjoint_space_dyn<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    adjoint_space_dyn_with_primer(rule, space, encoded_layout_primer::<R>)
}

fn adjoint_space_dyn_with_primer<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    space.validate_rule(rule)?;
    let key = AdjointSpaceKey {
        rule_key: rule.tree_transform_rule_cache_key(),
        source_homspace_id: space.homspace().id(),
        source_content_id: space.structure().content_id(),
        nout: space.nout(),
        nin: space.nin(),
    };
    let cache = adjoint_space_cache::<R::Key>();
    if let Ok(mut guard) = cache.write() {
        if let Some(hit) = guard.entries.get(&key).cloned() {
            // Clone the inner hom-space/subblock Arcs for the by-value return.
            return Ok((*hit).clone());
        }
    }
    let built = Arc::new(build_adjoint_space_dyn_with_primer(rule, space, primer)?);
    if let Ok(mut guard) = cache.write() {
        guard.entries.put(key, Arc::clone(&built));
    }
    Ok((*built).clone())
}

/// Uncached build of the adjoint (dagger) space; [`adjoint_space_dyn`] memoizes
/// it.
fn build_adjoint_space_dyn_with_primer<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let nout = space.nout();
    let homspace = space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());

    let structure = Arc::clone(space.structure());
    let prepared = primer(rule, &adjoint_hom)?;
    let keys = prepared.keys(rule, &adjoint_hom);
    let shapes = keys
        .iter()
        .map(|key| {
            let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ));
            // Why-not eager `ok_or`: this closure runs once per block on the
            // warm adjoint-fold loop (#231) -- an eager `ok_or` builds the
            // 96 B `OperationError` payload on every success, not just on the
            // (cold) missing-key path; `ok_or_else` defers construction to
            // the actual error case.
            let index = structure
                .find_block_index_by_key(&source_key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(source_key.clone()),
                })?;
            let source_shape = structure
                .block(index)
                .map_err(OperationError::from_core_preserving_context)?
                .shape();
            let mut shape = source_shape[nout..].to_vec();
            shape.extend_from_slice(&source_shape[..nout]);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
        rule,
        adjoint_hom,
        shapes,
        move |_, _| Ok(prepared),
    )
}

/// Dynamic-rank adjoint (dagger): returns the adjoint space (codomain and
/// domain swapped) together with freshly allocated coupled-layout data whose
/// blocks are the conjugate transposes of the source blocks.
#[cfg(test)]
pub(crate) fn adjoint_dyn<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    adjoint_dyn_with_primer(rule, space, data, encoded_layout_primer::<R>)
}

fn adjoint_dyn_with_primer<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
    primer: LayoutKeyBuilder<R>,
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    space.validate_rule(rule)?;
    let nout = space.nout();
    let nin = space.nin();
    let structure = Arc::clone(space.structure());
    // Uncached build: the eager adjoint (SVD/eigh consumers) is a separate,
    // out-of-scope path from the cached lazy `adjoint_space_dyn` (#118 PR-1),
    // and routing it here keeps the replay-cache-key bound off matrix algebra.
    let adjoint_space = build_adjoint_space_dyn_with_primer(rule, space, primer)?;
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
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(source_key),
            })?;
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

/// Dynamic-rank adjoint that retains the exact provider allocation of its
/// checked source space.
pub fn adjoint_bound_dyn<R, D>(
    space: &BoundDynamicFusionMapSpace<R>,
    data: &[D],
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let (output, data) =
        adjoint_dyn_with_primer(space.provider(), space.space(), data, space.layout_primer())?;
    let output = BoundDynamicFusionMapSpace::from_derived_like(space, output)?;
    Ok((output, data))
}

#[doc(hidden)]
pub fn adjoint_bound_dyn_lowered<R, D>(
    space: &BoundDynamicFusionMapSpace<R>,
    data: &[D],
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + LoweredMultiplicityFreeAlgebra,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let (output, data) =
        adjoint_dyn_with_primer(space.provider(), space.space(), data, space.layout_primer())?;
    let output = BoundDynamicFusionMapSpace::from_derived_like(space, output)?;
    Ok((output, data))
}

/// Generic dynamic-rank adjoint that retains the exact provider allocation of
/// its checked source space.
pub fn adjoint_bound_dyn_generic<R, D>(
    space: &BoundDynamicFusionMapSpace<R>,
    data: &[D],
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
where
    R: FusionRule,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let (output, data) = adjoint_dyn_generic(space.provider(), space.space(), data)?;
    let output =
        BoundDynamicFusionMapSpace::from_derived(Arc::clone(space.provider_arc()), output)?;
    Ok((output, data))
}

/// Lazy-adjoint metadata retaining the exact provider allocation of the
/// checked source space.
pub fn adjoint_bound_space_dyn<R>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let output =
        adjoint_space_dyn_with_primer(space.provider(), space.space(), space.layout_primer())?;
    BoundDynamicFusionMapSpace::from_derived_like(space, output)
}

#[doc(hidden)]
pub fn adjoint_bound_space_dyn_lowered<R>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey
        + LoweredMultiplicityFreeAlgebra,
{
    let output =
        adjoint_space_dyn_with_primer(space.provider(), space.space(), space.layout_primer())?;
    BoundDynamicFusionMapSpace::from_derived_like(space, output)
}

/// Generic lazy-adjoint metadata retaining the exact provider allocation of
/// the checked source space.
pub fn adjoint_bound_space_dyn_generic<R>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: FusionRule,
{
    let output = adjoint_space_dyn_generic(space.provider(), space.space())?;
    BoundDynamicFusionMapSpace::from_derived(Arc::clone(space.provider_arc()), output)
}

/// Generic-fusion (SU(N)) sibling of [`adjoint_space_dyn`]. The adjoint is a
/// pure per-block relabel — codomain and domain trees swap wholesale and each
/// coupled block is transposed — with NO leg bending and NO recoupling, so it
/// is self-duality-independent and identical for a Generic (outer-multiplicity)
/// rule: the only difference from the mult-free path is that the block keys are
/// enumerated multiplicity-aware (vertex-labelled fusion trees). Bound relaxes
/// to [`FusionRule`] accordingly (no F/R symbols are touched).
pub(crate) fn adjoint_space_dyn_generic<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: FusionRule,
{
    space.validate_rule(rule)?;
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
            let index = structure
                .find_block_index_by_key(&source_key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(source_key.clone()),
                })?;
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
pub(crate) fn adjoint_dyn_generic<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: FusionRule,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    space.validate_rule(rule)?;
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
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(source_key),
            })?;
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
    fusion_space
        .validate_rule(rule)
        .map_err(OperationError::Core)?;
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
        let index = structure
            .find_block_index_by_key(&source_key)
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(source_key.clone()),
            })?;
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
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(source_key),
            })?;
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
    use std::cell::Cell;
    use tenet_core::{
        BlockStructure, FermionParityFusionRule, FusionProductSpace, Fz2SectorLayout,
        PackedProductCodec, ProductFusionRule, ProductSectorCodec, ProductSectorLayout,
        SU2FusionRule, SU2Irrep, SectorId, SectorLeg, Su2SectorLayout, U1FusionRule, U1Irrep,
        U1SectorLayout, Z2Irrep,
    };

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

    thread_local! {
        static LOWERED_PRIMER_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    fn triple_rule() -> TripleRule {
        TripleRule::new(
            Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
            SU2FusionRule,
        )
    }

    fn triple_sector(parity: u8, charge: i32, twice_spin: usize) -> SectorId {
        TripleCodec::encode(
            Fz2U1Codec::encode(
                Z2Irrep::new(parity).sector_id(),
                U1Irrep::new(charge).sector_id(),
            ),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    }

    fn triple_source(rule: &TripleRule) -> DynamicFusionMapSpace {
        let vacuum = triple_sector(0, 0, 0);
        let charged = triple_sector(1, 1, 1);
        let leg = |dual| SectorLeg::new([(vacuum, 1), (charged, 1)], dual);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false), leg(true)]),
            FusionProductSpace::new([leg(true), leg(false)]),
        );
        lowered_layout_primer(rule, &hom).unwrap();
        let count = hom.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, vec![vec![1; 4]; count]).unwrap()
    }

    fn counting_primer(
        rule: &TripleRule,
        homspace: &FusionTreeHomSpace,
    ) -> Result<PreparedLayoutKeys, OperationError> {
        LOWERED_PRIMER_CALLS.with(|calls| calls.set(calls.get() + 1));
        lowered_layout_primer(rule, homspace)
    }

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
        // What: the adjoint-space cache lives in the same process-global
        // registry `reset_global_operation_caches` clears (see the accessor
        // doc above), so a concurrent reset landing between the two builds
        // below could evict the first entry and hand the second a fresh
        // `Arc`, breaking the ptr_eq this test exists to check.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            source_homspace_id: src.homspace().id(),
            source_content_id: src.structure().content_id(),
            nout: src.nout(),
            nin: src.nin(),
        };
        assert_ne!(
            make(TreeTransformBuiltinRuleCacheKey::U1),
            make(TreeTransformBuiltinRuleCacheKey::SU2Exact {
                authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
            })
        );
    }

    #[test]
    fn lowered_adjoint_metadata_primes_once_and_eager_data_matches_encoded() {
        // What: lazy adjoint metadata primes only on its operation-cache miss,
        // while eager materialization preserves the encoded oracle's layout and data.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let rule = triple_rule();
        let source = triple_source(&rule);

        LOWERED_PRIMER_CALLS.with(|calls| calls.set(0));
        let lowered = adjoint_space_dyn_with_primer(&rule, &source, counting_primer).unwrap();
        let warm = adjoint_space_dyn_with_primer(&rule, &source, counting_primer).unwrap();
        assert_eq!(LOWERED_PRIMER_CALLS.with(Cell::get), 1);
        assert_eq!(lowered, warm);
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let encoded_lazy = adjoint_space_dyn(&rule, &source).unwrap();
        assert_eq!(lowered, encoded_lazy);

        let data = (0..source.required_len().unwrap())
            .map(|index| index as f64 + 0.25)
            .collect::<Vec<_>>();
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let (lowered_space, lowered_data) =
            adjoint_dyn_with_primer(&rule, &source, &data, lowered_layout_primer::<TripleRule>)
                .unwrap();
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let (encoded_space, encoded_data) = adjoint_dyn(&rule, &source, &data).unwrap();
        assert_eq!(lowered_space, encoded_space);
        assert_eq!(lowered_data, encoded_data);
    }

    #[test]
    fn lowered_adjoint_missing_source_key_abandons_prepared_layout() {
        // What: adjoint source-key mapping fails before its checked target
        // layout receives an ID or cache admission.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::reset_global_operation_caches();
        tenet_core::reset_core_intern_tables();
        let rule = U1FusionRule;
        let one = U1Irrep::new(1).sector_id();
        let two = U1Irrep::new(2).sector_id();
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(one, 1)], false),
                SectorLeg::new([(one, 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(two, 1)], false)]),
        );
        let typed = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
            homspace,
            BlockStructure::from_blocks_with_rank(3, Vec::new()).unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let source = DynamicFusionMapSpace::from_typed(&typed);
        let before = tenet_core::fusion_tree_layout_cache_info();

        let error =
            adjoint_space_dyn_with_primer(&rule, &source, lowered_layout_primer::<U1FusionRule>)
                .unwrap_err();

        assert!(matches!(error, OperationError::MissingBlockKey { .. }));
        assert_eq!(tenet_core::fusion_tree_layout_cache_info(), before);
    }
}
